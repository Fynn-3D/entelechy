//! The passes beyond a map: a run repeated in place, a reduction's
//! fold, a scan's prefix, and a scatter's target and combine.

use super::*;

impl GpuBackend {
    /// The whole of a repeated run: staged once, then its times in one
    /// pass of one command buffer, two buffers taking turns.
    pub(crate) fn repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: usize,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let pipeline = built(prepared)?;
        check_invocation(module, invocation, scope)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let domain = execution_domain(module, invocation, scope)?;
        // Two buffers take turns, so the output is held twice.
        let schema = &module.manifest.outputs[0].ty.schema;
        scope.can_hold(&[(schema, domain), (schema, domain)])?;
        let staged = self.stage(pipeline, module, invocation, scope, domain, None)?;
        let Staged {
            mut encoder,
            context_buffer,
            params_buffer,
            mut input_buffers,
            memory_outputs: _,
            output_buffer,
            further_outputs: _,
            output_words: _,
            element_count,
            failure: _,
            target: _,
            highs: _,
            bound_inputs: _,
        } = staged;
        let scopes = Scopes::push(&self.device);
        // The first time reads the input it was given and writes the
        // first of the two; every time after reads what the one before
        // wrote and writes the other.
        let spare = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat"),
            size: (module.manifest.outputs[0]
                .ty
                .schema
                .storage_words(element_count)
                * 4)
            .max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (groups_x, groups_y, groups_z, _) = wgsl::dispatch(domain);
        // Two buffers take turns: what a time wrote the next reads, and what it
        // read is free for the time after. The Field the run was given is only
        // read, by the pass already recorded. So there are two bindings, one
        // for each turn.
        let turns: Vec<(wgpu::BindGroup, wgpu::Buffer)> = [
            (output_buffer.clone(), spare.clone()),
            (spare.clone(), output_buffer.clone()),
        ]
        .into_iter()
        .take((repeat as usize).saturating_sub(1).min(2))
        .map(|(reads, written)| {
            input_buffers[chain] = reads;
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PARAMS_BINDING,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::OUTPUT_BINDING,
                    resource: written.as_entire_binding(),
                },
            ];
            for (index, buffer) in input_buffers.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: wgsl::INPUT_BINDING + index as u32,
                    resource: buffer.as_entire_binding(),
                });
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy repeat"),
                layout: &pipeline.main.1,
                entries: &entries,
            });
            (group, written)
        })
        .collect();
        let times = repeat.saturating_sub(1) as usize;
        let describe = wgpu::ComputePassDescriptor {
            label: Some("entelechy repeat"),
            timestamp_writes: None,
        };
        if wgsl::output_shares_words(module, 0) {
            // A time that stores by atomic starts from a cleared buffer, so
            // each time is its own pass after its clear.
            for time in 0..times {
                let (group, written) = &turns[time % 2];
                encoder.clear_buffer(written, 0, None);
                let mut pass = encoder.begin_compute_pass(&describe);
                pass.set_pipeline(&pipeline.main.0);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        } else {
            // Dispatches in one pass run in order, each seeing what the one
            // before wrote.
            let mut pass = encoder.begin_compute_pass(&describe);
            pass.set_pipeline(&pipeline.main.0);
            for time in 0..times {
                pass.set_bind_group(0, &turns[time % 2].0, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        }
        let (last, other) = if times % 2 == 1 {
            (spare, output_buffer)
        } else {
            (output_buffer, spare)
        };
        self.submit(encoder.finish());
        // The other of the two goes to the pile for a later output.
        self.spare.keep(other, None);
        let _ = self.device.poll(wgpu::PollType::Poll);
        scopes
            .pop(self.wait, "")
            .map_err(|error| ExecutionError::Device(error.said()))?;
        // What the last time wrote is what the run gives.
        adopt_outputs(
            scope,
            module,
            domain,
            last,
            Vec::new(),
            Vec::new(),
            Some(self.spare.clone()),
        )
    }

    /// A prefix fold on the device: three passes over the map's output, in
    /// place. One invocation walks each block and leaves its total, one scans
    /// the blocks' totals, and one per element adds its block's offset. The
    /// rule is order-free, so this grouping gives the reference's words; the
    /// passes order themselves, with no shared memory or barrier.
    ///
    /// A block holds about the square root of the count, so both walks are
    /// the same length.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_prefix(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prefix: &Prefix,
        output: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
        failure: Option<&wgpu::Buffer>,
    ) -> wgpu::Buffer {
        let count = element_count.max(1);
        let per_block = entelechy_rules::prefix::block(count);
        let blocks = count.div_ceil(per_block);
        let plan: [u32; 4] = [count as u32, per_block as u32, blocks as u32, 0];
        let plan_bytes: Vec<u8> = plan.iter().flat_map(|v| v.to_le_bytes()).collect();
        let plan_buffer = self.run_buffer("prefix plan", &plan_bytes, wgpu::BufferUsages::UNIFORM);
        let sums = self.spare.take(
            &self.device,
            ((blocks * output_words * 4) as u64).max(16).div_ceil(16) * 16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        // A whole sum's high words beside the output and the totals.
        let highs = prefix.whole.then(|| {
            let take = |words: usize| {
                self.spare.take(
                    &self.device,
                    ((words * 4) as u64).max(16).div_ceil(16) * 16,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
            };
            (take(count * output_words), take(blocks * output_words))
        });
        let passes = [
            (&prefix.blocks, blocks as u32),
            (&prefix.tops, 1),
            (&prefix.offsets, count as u32),
        ];
        for (index, ((pipeline, layout), threads)) in passes.iter().enumerate() {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_PLAN_BINDING,
                    resource: plan_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_OUT_BINDING,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_SUMS_BINDING,
                    resource: sums.as_entire_binding(),
                },
            ];
            if let (Some((out_highs, sum_highs)), Some(failure)) = (&highs, failure) {
                entries.extend([
                    wgpu::BindGroupEntry {
                        binding: wgsl::PREFIX_OUT_HIGHS_BINDING,
                        resource: out_highs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: wgsl::PREFIX_SUMS_HIGHS_BINDING,
                        resource: sum_highs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: wgsl::FAILURE_BINDING,
                        resource: failure.as_entire_binding(),
                    },
                ]);
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy prefix"),
                layout,
                entries: &entries,
            });
            let last = index + 1 == passes.len();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy prefix"),
                timestamp_writes: timing
                    .and_then(|timing| last.then(|| timing.compute(None, Some(1)))),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let (x, y) = wgsl::folded(threads.div_ceil(wgsl::PREFIX_WORKGROUP));
            pass.dispatch_workgroups(x, y, 1);
        }
        self.spare.keep(sums, None);
        if let Some((out_highs, sum_highs)) = highs {
            self.spare.keep(out_highs, None);
            self.spare.keep(sum_highs, None);
        }
        output
    }

    /// A reduction's fold: level by level into a second buffer,
    /// alternating, in the same pairwise tree as the CPU reference.
    /// Returns the buffer holding the one element.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fold(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        module: &Validated,
        reduce: &Pass,
        context_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        bound_inputs: &[(u32, wgpu::Buffer)],
        mapped: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
        failure: Option<&wgpu::Buffer>,
    ) -> wgpu::Buffer {
        let (reduce_pipeline, reduce_layout) = reduce;
        let params_buffer_for_reduce = params_buffer;
        let final_buffer = mapped;
        let stride = output_words as u64 * 4;
        let mut count = element_count as u64;
        // The level between levels, from the pile: a scene reduces the same
        // count every frame. Every element a level reads was written by the
        // level before, so nothing stale is seen.
        let other = self.spare.take(
            &self.device,
            (entelechy_rules::reduce::after_fold(count as usize) as u64 * stride).max(16),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut current = final_buffer;
        let mut next = other;
        while count > 1 {
            let level: [u32; 4] = [count as u32, 0, 0, 0];
            let level_bytes: Vec<u8> = level.iter().flat_map(|v| v.to_le_bytes()).collect();
            let level_buffer =
                self.run_buffer("reduce count", &level_bytes, wgpu::BufferUsages::UNIFORM);
            let (count_binding, in_binding, out_binding) = wgsl::reduce_bindings(module);
            let mut fold_entries = vec![
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PARAMS_BINDING,
                    resource: params_buffer_for_reduce.as_entire_binding(),
                },
            ];
            for (binding, buffer) in bound_inputs {
                fold_entries.push(wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                });
            }
            fold_entries.extend([
                wgpu::BindGroupEntry {
                    binding: count_binding,
                    resource: level_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: in_binding,
                    resource: current.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: out_binding,
                    resource: next.as_entire_binding(),
                },
            ]);
            if let Some(failure) = failure {
                fold_entries.push(wgpu::BindGroupEntry {
                    binding: wgsl::FAILURE_BINDING,
                    resource: failure.as_entire_binding(),
                });
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy reduce"),
                layout: reduce_layout,
                entries: &fold_entries,
            });
            // Every level closes the window; the last one to write is
            // the end of the node's work.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy reduce"),
                timestamp_writes: timing.map(|timing| timing.compute(None, Some(1))),
            });
            pass.set_pipeline(reduce_pipeline);
            pass.set_bind_group(0, &group, &[]);
            let ahead = entelechy_rules::reduce::after_fold(count as usize) as u32;
            let (x, y) = wgsl::folded(ahead.div_ceil(wgsl::REDUCE_WORKGROUP));
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            count = u64::from(ahead);
            std::mem::swap(&mut current, &mut next);
        }
        // One of the two is the answer and becomes the Field; the other goes
        // back to the pile. What read it was submitted before whatever writes
        // it next, and one queue runs in order.
        self.spare.keep(next, None);
        current
    }

    /// The target made ready for pairs: the base copied into it, or,
    /// where the plan compares its words with atomics, the base's words
    /// turned into ordered keys as they are written into it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_target(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scatter: &Scatter,
        target: &wgpu::Buffer,
        highs: Option<&wgpu::Buffer>,
        base_buffer: &wgpu::Buffer,
        context_buffer: &wgpu::Buffer,
        count: usize,
        words: usize,
    ) {
        // A whole sum's high words start as the base's signs.
        if let (
            Scatter::Parallel {
                totals: Some(totals),
                ..
            },
            Some(highs),
        ) = (scatter, highs)
        {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy scatter highs"),
                layout: &totals.from_base.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: wgsl::CONTEXT_BINDING,
                        resource: context_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: wgsl::SCATTER_PAIRS_BINDING,
                        resource: base_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: wgsl::SCATTER_HIGHS_BINDING,
                        resource: highs.as_entire_binding(),
                    },
                ],
            });
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy scatter highs"),
                timestamp_writes: None,
            });
            compute.set_pipeline(&totals.from_base.0);
            compute.set_bind_group(0, &group, &[]);
            let (x, y) = wgsl::folded(((count * words) as u32).div_ceil(wgsl::SCATTER_WORKGROUP));
            compute.dispatch_workgroups(x, y, 1);
        }
        let Scatter::Parallel {
            keys: Some(keys), ..
        } = scatter
        else {
            encoder.copy_buffer_to_buffer(base_buffer, 0, target, 0, (count * words * 4) as u64);
            return;
        };
        let keys = &keys.from_base;
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy scatter"),
            layout: &keys.1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::SCATTER_PAIRS_BINDING,
                    resource: base_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::SCATTER_OUT_BINDING,
                    resource: target.as_entire_binding(),
                },
            ],
        });
        let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy scatter keys"),
            timestamp_writes: None,
        });
        compute.set_pipeline(&keys.0);
        compute.set_bind_group(0, &group, &[]);
        let (x, y) = wgsl::folded((count as u32).div_ceil(wgsl::SCATTER_WORKGROUP));
        compute.dispatch_workgroups(x, y, 1);
    }

    /// Between the landings of a scatter's Longs' halves: each low word of
    /// the target's `words` set aside where its high word is no longer the
    /// base's ([`wgsl::emit_scatter_lows`]).
    pub(crate) fn lows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lows: &Pass,
        context_buffer: &wgpu::Buffer,
        base_buffer: &wgpu::Buffer,
        target: &wgpu::Buffer,
        words: usize,
    ) {
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy scatter lows"),
            layout: &lows.1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::SCATTER_PAIRS_BINDING,
                    resource: base_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::SCATTER_OUT_BINDING,
                    resource: target.as_entire_binding(),
                },
            ],
        });
        let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy scatter lows"),
            timestamp_writes: None,
        });
        compute.set_pipeline(&lows.0);
        compute.set_bind_group(0, &group, &[]);
        let (x, y) = wgsl::folded(((words / 2) as u32).div_ceil(wgsl::SCATTER_WORKGROUP));
        compute.dispatch_workgroups(x, y, 1);
    }

    /// A scatter's combine: the target made ready (unless the emitting entry
    /// did), then the pairs applied by the plan. A float sum sorts pairs by
    /// target and source (bitonic, one small dispatch per step) and folds each
    /// target's run in source order; order-free rules apply one invocation per
    /// pair with atomics, `min` and `max` over floats between two key passes,
    /// `last` followed by a gather. Returns the target.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scatter: &Scatter,
        (target, highs): (Option<wgpu::Buffer>, Option<wgpu::Buffer>),
        failure: Option<&wgpu::Buffer>,
        base_buffer: &wgpu::Buffer,
        context_buffer: &wgpu::Buffer,
        pairs: wgpu::Buffer,
        target_count: usize,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
    ) -> wgpu::Buffer {
        let final_buffer = pairs;
        // Ready already when the entry emitted into it; the entry then landed
        // every pair itself, and nothing is left to apply.
        let landed = target.is_some();
        let whole = matches!(
            scatter,
            Scatter::Parallel {
                totals: Some(_),
                ..
            }
        );
        let (target, highs) = match target {
            Some(target) => (target, highs),
            None => {
                let target = self.target_buffer(target_count, output_words);
                let highs = whole.then(|| self.target_buffer(target_count, output_words));
                self.prepare_target(
                    encoder,
                    scatter,
                    &target,
                    highs.as_ref(),
                    base_buffer,
                    context_buffer,
                    target_count,
                    output_words,
                );
                (target, highs)
            }
        };
        let over_target = (target_count as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
        let over_sources = (element_count as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
        let storage = |label: &str, words: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words as u64 * 4).max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let common = [
            wgpu::BindGroupEntry {
                binding: wgsl::CONTEXT_BINDING,
                resource: context_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: wgsl::SCATTER_PAIRS_BINDING,
                resource: final_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: wgsl::SCATTER_OUT_BINDING,
                resource: target.as_entire_binding(),
            },
        ];
        let bind = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy scatter"),
                layout,
                entries,
            })
        };
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        pass: &Pass,
                        group: &wgpu::BindGroup,
                        offsets: &[u32],
                        groups: u32| {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy scatter"),
                timestamp_writes: timing.map(|timing| timing.compute(None, Some(1))),
            });
            compute.set_pipeline(&pass.0);
            compute.set_bind_group(0, group, offsets);
            let (x, y) = wgsl::folded(groups);
            compute.dispatch_workgroups(x, y, 1);
        };
        match scatter {
            Scatter::Sorted {
                keys,
                step,
                fold,
                whole,
            } => {
                let padded = element_count.next_power_of_two().max(1);
                let keys_buffer = storage("scatter sort keys", padded);
                let order_buffer = storage("scatter sort order", padded);
                // Read by this run alone.
                self.spare.let_go(keys_buffer.clone());
                self.spare.let_go(order_buffer.clone());
                // One parameter record per dispatch, at the alignment
                // dynamic offsets need: the padded count, then k and j.
                let mut records: Vec<[u32; 4]> = vec![[padded as u32, 0, 0, 0]];
                let mut k = 2;
                while k <= padded {
                    let mut j = k / 2;
                    while j >= 1 {
                        records.push([padded as u32, k as u32, j as u32, 0]);
                        j /= 2;
                    }
                    k *= 2;
                }
                let alignment = self
                    .device
                    .limits()
                    .min_uniform_buffer_offset_alignment
                    .max(16) as usize;
                let mut bytes = vec![0u8; records.len() * alignment];
                for (index, record) in records.iter().enumerate() {
                    for (word, value) in record.iter().enumerate() {
                        let at = index * alignment + word * 4;
                        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                let params =
                    self.run_buffer("scatter sort params", &bytes, wgpu::BufferUsages::UNIFORM);
                let params_entry = wgpu::BindGroupEntry {
                    binding: wgsl::SCATTER_SORT_PARAMS_BINDING,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: std::num::NonZeroU64::new(16),
                    }),
                };
                let sort_entries = |read_only: bool| {
                    let _ = read_only;
                    [
                        wgpu::BindGroupEntry {
                            binding: wgsl::SCATTER_SORT_KEYS_BINDING,
                            resource: keys_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: wgsl::SCATTER_SORT_ORDER_BINDING,
                            resource: order_buffer.as_entire_binding(),
                        },
                        params_entry.clone(),
                    ]
                };
                let keys_group = bind(
                    &keys.1,
                    &[common[0].clone(), common[1].clone()]
                        .into_iter()
                        .chain(sort_entries(false))
                        .collect::<Vec<_>>(),
                );
                let step_group = bind(&step.1, &sort_entries(false));
                let fold_group = bind(
                    &fold.1,
                    &common
                        .iter()
                        .cloned()
                        .chain(sort_entries(true))
                        .chain(
                            failure
                                .filter(|_| *whole)
                                .map(|failure| wgpu::BindGroupEntry {
                                    binding: wgsl::FAILURE_BINDING,
                                    resource: failure.as_entire_binding(),
                                }),
                        )
                        .collect::<Vec<_>>(),
                );
                let over_padded = (padded as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
                dispatch(encoder, keys, &keys_group, &[0], over_padded);
                // Every step of the sort in one pass: dispatches in a
                // pass run in order, each seeing the last one's writes.
                {
                    let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("entelechy scatter sort"),
                        timestamp_writes: timing.map(|timing| timing.compute(None, Some(1))),
                    });
                    compute.set_pipeline(&step.0);
                    for index in 1..records.len() {
                        compute.set_bind_group(0, &step_group, &[(index * alignment) as u32]);
                        let (x, y) = wgsl::folded(over_padded);
                        compute.dispatch_workgroups(x, y, 1);
                    }
                }
                dispatch(encoder, fold, &fold_group, &[0], over_target);
            }
            Scatter::Parallel {
                apply,
                gather,
                keys,
                totals,
                halves,
            } => {
                // Winners for `last`, zeroed as every new buffer is; a
                // token buffer where the pass has none to record.
                let winners = storage(
                    "scatter winners",
                    if gather.is_some() { target_count } else { 1 },
                );
                self.spare.let_go(winners.clone());
                let with_winners: Vec<wgpu::BindGroupEntry> = common
                    .iter()
                    .cloned()
                    .chain([wgpu::BindGroupEntry {
                        binding: wgsl::SCATTER_WINNER_BINDING,
                        resource: winners.as_entire_binding(),
                    }])
                    .chain(highs.as_ref().map(|highs| wgpu::BindGroupEntry {
                        binding: wgsl::SCATTER_HIGHS_BINDING,
                        resource: highs.as_entire_binding(),
                    }))
                    .collect();
                if !landed {
                    let group = bind(&apply.1, &with_winners);
                    dispatch(encoder, apply, &group, &[], over_sources);
                    // The high words in, the low ones of the pairs that won.
                    if let Some(halves) = halves {
                        self.lows(
                            encoder,
                            &halves.lows,
                            context_buffer,
                            base_buffer,
                            &target,
                            target_count * output_words,
                        );
                        let group = bind(&halves.apply.1, &with_winners);
                        dispatch(encoder, &halves.apply, &group, &[], over_sources);
                    }
                }
                if let Some(gather) = gather {
                    let group = bind(&gather.1, &with_winners);
                    dispatch(encoder, gather, &group, &[], over_target);
                }
                if let Some(keys) = keys {
                    let group = bind(&keys.in_place.1, &common);
                    dispatch(encoder, &keys.in_place, &group, &[], over_target);
                }
                // Every value in: each element's whole total checked.
                if let (Some(totals), Some(highs), Some(failure)) = (totals, &highs, failure) {
                    let group = bind(
                        &totals.check.1,
                        &[
                            common[0].clone(),
                            common[2].clone(),
                            wgpu::BindGroupEntry {
                                binding: wgsl::SCATTER_HIGHS_BINDING,
                                resource: highs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: wgsl::FAILURE_BINDING,
                                resource: failure.as_entire_binding(),
                            },
                        ],
                    );
                    dispatch(encoder, &totals.check, &group, &[], over_target);
                }
            }
        }
        // The pairs and the high words go back to the pile: what read them
        // was recorded before whatever writes them next.
        self.spare.keep(final_buffer, None);
        if let Some(highs) = highs {
            self.spare.keep(highs, None);
        }
        target
    }
}
