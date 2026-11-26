//! The passes beyond a map: a run repeated in place, a reduction's
//! fold, a scan's prefix, and a scatter's target and combine.

use super::*;

impl WgpuBackend {
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
        check_invocation(module, invocation, scope)?;
        let pipeline = self.built_for(prepared, module, invocation, scope)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let encoding = self.waiter_control.begin_encoding();
        let domain = execution_domain(module, invocation, scope)?;
        // Two buffers take turns, so the output is held twice.
        let schema = &module.manifest.outputs[0].ty.schema;
        scope.can_hold(&[(schema, domain), (schema, domain)])?;
        let allow_batch = self.batch.can_batch(self.measured(), &self.frames);
        let staged = self.stage(
            pipeline,
            module,
            invocation,
            scope,
            domain,
            None,
            usize::MAX,
            allow_batch,
        )?;
        let Staged {
            mut encoder,
            batched,
            bind_group: _,
            dynamic_offsets,
            dynamic_offset_len,
            further_slices: _,
            context_buffer: _,
            context_slice,
            params_slice,
            mut input_buffers,
            memory_outputs: _,
            output_buffer,
            further_outputs: _,
            output_words: _,
            element_count,
            failure: _,
            target: _,
            highs: _,
            indirect_bad_check: _,
            bound_inputs: _,
            bound_textures,
            temp_buffers,
        } = staged;
        let scopes = (!batched).then(|| Scopes::push(&self.device));
        // The first time reads the input it was given and writes the
        // first of the two; every time after reads what the one before
        // wrote and writes the other.
        let spare = self.spare.take(
            &self.device,
            (module.manifest.outputs[0]
                .ty
                .schema
                .storage_words(element_count)
                * 4)
            .max(16) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let (groups_x, groups_y, groups_z, _) = wgsl::dispatch_for(module, domain);
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
            input_buffers[chain] = Some(reads);
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_slice.dynamic_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PARAMS_BINDING,
                    resource: params_slice.dynamic_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::OUTPUT_BINDING,
                    resource: written.as_entire_binding(),
                },
            ];
            for (index, buffer) in input_buffers.iter().enumerate() {
                let binding = wgsl::INPUT_BINDING + index as u32;
                if let Some((_, view)) = bound_textures.iter().find(|(b, _)| *b == binding) {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                } else if let Some(buffer) = buffer {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    });
                }
            }
            self.push_sampler_entries(pipeline.samplers, &mut entries);
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
        let repeat_offsets = &dynamic_offsets[..dynamic_offset_len];
        if wgsl::output_shares_words_for(module, 0, self.shader_f16) {
            // A time that stores by atomic starts from a cleared buffer, so
            // each time is its own pass after its clear.
            for time in 0..times {
                let (group, written) = &turns[time % 2];
                encoder.clear_buffer(written, 0, None);
                let mut pass = encoder.begin_compute_pass(&describe);
                pass.set_pipeline(&pipeline.main.0);
                pass.set_bind_group(0, group, repeat_offsets);
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        } else {
            // Dispatches in one pass run in order, each seeing what the one
            // before wrote.
            let mut pass = encoder.begin_compute_pass(&describe);
            pass.set_pipeline(&pipeline.main.0);
            for time in 0..times {
                pass.set_bind_group(0, &turns[time % 2].0, repeat_offsets);
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        }
        let (last, other) = if times % 2 == 1 {
            (spare, output_buffer)
        } else {
            (output_buffer, spare)
        };
        if batched {
            self.frames.note_batched_run();
            let page_budget = self
                .wait
                .parks_with_host()
                .then(|| self.page_pace_ops_budget());
            let elems = (domain.count() as u64).saturating_mul(u64::from(repeat));
            let ops = elems.saturating_mul(effective_page_unit_ops(module, &invocation.params));
            let mut keep = temp_buffers;
            keep.push(other);
            drop(encoding);
            self.batch
                .finish_node(encoder, ops, keep, Vec::new(), page_budget)?;
        } else {
            self.submit(encoder.finish());
            // The other of the two goes to the pile for a later output.
            self.spare.keep(other, None);
            for buffer in temp_buffers {
                self.spare.keep(buffer, None);
            }
            drop(encoding);
            if self._waiter.is_none() {
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
            if let Some(scopes) = scopes {
                scopes
                    .pop(self.wait, "")
                    .map_err(|error| ExecutionError::Device(error.said()))?;
            }
        }
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

    /// A hierarchical parallel prefix fold on the device: scans contiguous
    /// blocks of [`wgsl::PREFIX_BLOCK`] elements in place, recursively scans
    /// the block totals with the same shader until one block remains, and
    /// propagates block offsets back down to level 0.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_prefix(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prefix: &Prefix,
        output: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
        failure: Option<&StagedFailure>,
        keep_buffers: &mut Vec<wgpu::Buffer>,
    ) -> wgpu::Buffer {
        let block_size = wgsl::PREFIX_BLOCK as usize;
        let count = element_count.max(1);
        let alloc_storage = |elems: usize| {
            self.spare.take(
                &self.device,
                ((elems * output_words * 4) as u64).max(16).div_ceil(16) * 16,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )
        };
        let mut used_plans = Vec::new();
        // Level 0 operates on `output`; each subsequent level operates on the
        // previous level's block totals buffer until `blocks == 1`.
        struct Level {
            count: usize,
            blocks: usize,
            data: wgpu::Buffer,
            sums: wgpu::Buffer,
            data_highs: Option<wgpu::Buffer>,
            sums_highs: Option<wgpu::Buffer>,
        }
        let mut levels: Vec<Level> = Vec::new();
        let mut cur_count = count;
        let mut cur_data = output.clone();
        let mut cur_highs = prefix.whole.then(|| alloc_storage(count));
        loop {
            let blocks = cur_count.div_ceil(block_size);
            let sums = alloc_storage(blocks);
            let sums_highs = prefix.whole.then(|| alloc_storage(blocks));
            levels.push(Level {
                count: cur_count,
                blocks,
                data: cur_data,
                sums: sums.clone(),
                data_highs: cur_highs,
                sums_highs: sums_highs.clone(),
            });
            if blocks <= 1 {
                break;
            }
            cur_count = blocks;
            cur_data = sums;
            cur_highs = sums_highs;
        }
        let num_levels = levels.len();
        let mut dispatch_prefix = |pipeline: &wgpu::ComputePipeline,
                                   layout: &wgpu::BindGroupLayout,
                                   level: &Level,
                                   flags: u32,
                                   groups: u32,
                                   is_last: bool| {
            let plan: [u32; 4] = [level.count as u32, level.blocks as u32, 0, flags];
            let mut plan_bytes = [0u8; 16];
            for (slot, word) in plan_bytes.chunks_exact_mut(4).zip(plan) {
                slot.copy_from_slice(&word.to_le_bytes());
            }
            let plan_buffer = self.take_uniform(&plan_bytes);
            let base_entries = [
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_PLAN_BINDING,
                    resource: plan_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_OUT_BINDING,
                    resource: level.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PREFIX_SUMS_BINDING,
                    resource: level.sums.as_entire_binding(),
                },
            ];
            let full_entries;
            let entries: &[wgpu::BindGroupEntry] =
                if let (Some(out_highs), Some(sum_highs), Some(failure)) =
                    (&level.data_highs, &level.sums_highs, failure)
                {
                    full_entries = [
                        base_entries[0].clone(),
                        base_entries[1].clone(),
                        base_entries[2].clone(),
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
                            resource: failure.binding(),
                        },
                    ];
                    &full_entries
                } else {
                    &base_entries
                };
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy prefix"),
                layout,
                entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy prefix"),
                timestamp_writes: timing
                    .and_then(|timing| is_last.then(|| timing.compute(None, Some(1)))),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let (x, y) = wgsl::folded(groups);
            pass.dispatch_workgroups(x, y, 1);
            used_plans.push(plan_buffer);
        };
        // Up-sweep: scan each level in parallel blocks of 256.
        for (depth, level) in levels.iter().enumerate() {
            let mut flags = 0u32;
            if depth == 0 {
                flags |= wgsl::PREFIX_FLAG_WIDEN;
            }
            let is_single_pass = num_levels == 1;
            if is_single_pass {
                flags |= wgsl::PREFIX_FLAG_CHECK_FITS;
            }
            dispatch_prefix(
                &prefix.blocks.0,
                &prefix.blocks.1,
                level,
                flags,
                level.blocks as u32,
                is_single_pass,
            );
        }
        // Down-sweep: propagate scanned block totals back to level 0.
        if num_levels > 1 {
            for depth in (0..num_levels - 1).rev() {
                let level = &levels[depth];
                let is_last = depth == 0;
                let flags = if is_last {
                    wgsl::PREFIX_FLAG_CHECK_FITS
                } else {
                    0
                };
                let groups = (level.count as u32).div_ceil(wgsl::PREFIX_WORKGROUP);
                dispatch_prefix(
                    &prefix.offsets.0,
                    &prefix.offsets.1,
                    level,
                    flags,
                    groups,
                    is_last,
                );
            }
        }
        keep_buffers.extend(used_plans);
        // `level.data` / `level.data_highs` at depth > 0 alias the previous
        // level's `sums` / `sums_highs`, so return only each freshly allocated
        // scratch buffer once.
        for (depth, level) in levels.into_iter().enumerate() {
            if depth == 0
                && let Some(h) = level.data_highs
            {
                keep_buffers.push(h);
            }
            keep_buffers.push(level.sums);
            if let Some(h) = level.sums_highs {
                keep_buffers.push(h);
            }
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
        reduce_fold: usize,
        context_slice: &crate::batch::UniformSlice,
        params_slice: &crate::batch::UniformSlice,
        bound_inputs: &[(u32, wgpu::Buffer)],
        mapped: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
        failure: Option<&StagedFailure>,
        keep_buffers: &mut Vec<wgpu::Buffer>,
    ) -> wgpu::Buffer {
        let (reduce_pipeline, reduce_layout) = reduce;
        let final_buffer = mapped;
        let stride = output_words as u64 * 4;
        let mut count = element_count as u64;
        let parallel_block = reduce_fold == wgsl::REDUCE_BLOCK as usize;
        // The level between levels, from the pile: a scene reduces the same
        // count every frame. Every element a level reads was written by the
        // level before, so nothing stale is seen.
        let first_ahead = (count as usize).div_ceil(reduce_fold);
        let other = self.spare.take(
            &self.device,
            (first_ahead as u64 * stride).max(16),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut current = final_buffer;
        let mut next = other;
        let (count_binding, in_binding, out_binding) = wgsl::reduce_bindings(module);
        let mut used_uniforms = Vec::new();
        while count > 1 {
            let mut level_bytes = [0u8; 16];
            level_bytes[..4].copy_from_slice(&(count as u32).to_le_bytes());
            let level_buffer = self.take_uniform(&level_bytes);
            let mut fold_entries = Vec::with_capacity(bound_inputs.len() + 6);
            fold_entries.extend([
                wgpu::BindGroupEntry {
                    binding: wgsl::CONTEXT_BINDING,
                    resource: context_slice.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: wgsl::PARAMS_BINDING,
                    resource: params_slice.binding(),
                },
            ]);
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
                    resource: failure.binding(),
                });
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy reduce"),
                layout: reduce_layout,
                entries: &fold_entries,
            });
            drop(fold_entries);
            used_uniforms.push(level_buffer);
            // Every level closes the window; the last one to write is
            // the end of the node's work.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy reduce"),
                timestamp_writes: timing.map(|timing| timing.compute(None, Some(1))),
            });
            pass.set_pipeline(reduce_pipeline);
            pass.set_bind_group(0, &group, &[]);
            let ahead = (count as usize).div_ceil(reduce_fold) as u32;
            let groups = if parallel_block {
                ahead
            } else {
                ahead.div_ceil(wgsl::REDUCE_WORKGROUP)
            };
            let (x, y) = wgsl::folded(groups);
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            count = u64::from(ahead);
            std::mem::swap(&mut current, &mut next);
        }
        keep_buffers.extend(used_uniforms);
        keep_buffers.push(next);
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
        context_slice: &crate::batch::UniformSlice,
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
                        resource: context_slice.binding(),
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
                    resource: context_slice.binding(),
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
        context_slice: &crate::batch::UniformSlice,
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
                    resource: context_slice.binding(),
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
        failure: Option<&StagedFailure>,
        base_buffer: &wgpu::Buffer,
        context_slice: &crate::batch::UniformSlice,
        pairs: wgpu::Buffer,
        target_count: usize,
        element_count: usize,
        output_words: usize,
        timing: Option<&Timing>,
        keep_buffers: &mut Vec<wgpu::Buffer>,
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
                    context_slice,
                    target_count,
                    output_words,
                );
                (target, highs)
            }
        };
        let over_target = (target_count as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
        let over_sources = (element_count as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
        let storage = |words: usize, usage: wgpu::BufferUsages| {
            self.spare.take(
                &self.device,
                (words as u64 * 4).max(16).div_ceil(16) * 16,
                usage | wgpu::BufferUsages::STORAGE,
            )
        };
        let common = [
            wgpu::BindGroupEntry {
                binding: wgsl::CONTEXT_BINDING,
                resource: context_slice.binding(),
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
                let keys_buffer = storage(padded, wgpu::BufferUsages::empty());
                let order_buffer = storage(padded, wgpu::BufferUsages::empty());
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
                let params = self.take_uniform(&bytes);
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
                                    resource: failure.binding(),
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
                keep_buffers.extend([keys_buffer, order_buffer, params]);
            }
            Scatter::Parallel {
                apply,
                gather,
                keys,
                totals,
                halves,
            } => {
                // Winners for `last`, cleared before the pass records into it;
                // a token buffer where the pass has none to record.
                let winners = storage(
                    if gather.is_some() { target_count } else { 1 },
                    wgpu::BufferUsages::COPY_DST,
                );
                if gather.is_some() {
                    encoder.clear_buffer(&winners, 0, None);
                }
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
                            context_slice,
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
                                resource: failure.binding(),
                            },
                        ],
                    );
                    dispatch(encoder, &totals.check, &group, &[], over_target);
                }
                keep_buffers.push(winners);
            }
        }
        keep_buffers.push(final_buffer);
        if let Some(highs) = highs {
            keep_buffers.push(highs);
        }
        target
    }
}
