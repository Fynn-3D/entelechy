//! Turning a validated module into what the device runs: its WGSL,
//! pipelines and binding layouts, or why it cannot run here.

use super::*;

pub(crate) struct Pipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    /// The fold of a reduce module, run level by level after the map.
    pub(crate) reduce: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// The combine of a scatter module, run after the map.
    pub(crate) scatter: Option<Scatter>,
    /// The fold of a prefix module, run over the map's output in place.
    pub(crate) prefix: Option<Prefix>,
    /// Whether the module's arithmetic can fail, so its map and fold
    /// passes record their first failure; see [`wgsl::can_fail`].
    pub(crate) fails: bool,
    /// Whether a scatter's entry emits pairs, combining them into the
    /// target as it runs: the target is then ready before the entry.
    pub(crate) emits: bool,
    /// The inputs its passes read from one buffer, copied there before they
    /// run, where binding each would exceed the device's limit; see
    /// [`wgsl::packed_inputs`].
    pub(crate) packed: Vec<usize>,
}

/// A prefix fold's three passes: a block a walk, the blocks' totals
/// scanned, then every element given its block's offset.
pub(crate) struct Prefix {
    pub(crate) blocks: Pass,
    pub(crate) tops: Pass,
    pub(crate) offsets: Pass,
    /// Whether it sums whole ([`wgsl::sums_whole`]): high words beside
    /// the output and the totals, and a failure record.
    pub(crate) whole: bool,
}

/// One compute pipeline with its bind group layout.
pub(crate) type Pass = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

/// The key transform's two passes: in place, and from the base.
pub(crate) struct Keys {
    pub(crate) in_place: Pass,
    pub(crate) from_base: Pass,
}

/// A whole sum's two passes around the parallel apply: the high words
/// set from the base, and the totals checked once every value is in.
pub(crate) struct Totals {
    pub(crate) from_base: Pass,
    pub(crate) check: Pass,
}

/// A scatter's combine as pipelines, by its plan.
pub(crate) enum Scatter {
    /// A float sum: sort keys, the sort's step, and the fold, which with
    /// `whole` also folds integers whole and records a total that does
    /// not fit.
    Sorted {
        keys: Pass,
        step: Pass,
        fold: Pass,
        whole: bool,
    },
    /// An order-free rule: the parallel apply; the gather after `last`; for
    /// `min` and `max` over floats and longs, the key transform in place
    /// after, and before from the base into the target, sparing a base copy.
    Parallel {
        apply: Pass,
        gather: Option<Pass>,
        keys: Option<Keys>,
        totals: Option<Totals>,
    },
}

impl Prepared for Pipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A uniform binding read at a dynamic offset, one record per dispatch.
pub(crate) fn dynamic_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuBackend {
    /// Why this device does not run `module`, if it does not: more
    /// buffers than it binds at once, a `Double` it has no arithmetic for,
    /// or a `Long` where the device has none or the mode folds words it
    /// would have to fold in pairs.
    pub(crate) fn refusal(&self, module: &Validated) -> Option<String> {
        if let Some(why) = self.wider_than_device(module) {
            return Some(why);
        }
        let braces = wgsl::brace_depth(module);
        if braces > wgsl::BRACE_LIMIT {
            return Some(format!(
                "its branches and loops nest {braces} deep, and a shader nests at most {}",
                wgsl::BRACE_LIMIT
            ));
        }
        if module.holds(orla_abi::Type::Double) {
            return Some("it holds a Double, which the device does not compute".to_owned());
        }

        if !module.holds(orla_abi::Type::Long) {
            return None;
        }
        if !self.int64 {
            return Some("it holds a Long, and the device has no 64-bit integers".to_owned());
        }
        match module.manifest.mode {
            orla_ir::Mode::Map | orla_ir::Mode::Reduce => None,
            orla_ir::Mode::Scatter { .. } if wgsl::long_extremes(module) => (!self.int64_atomics)
                .then(|| {
                    "it scatters Longs, and the device has no 64-bit atomic min and max".to_owned()
                }),
            _ => Some(
                "it holds a Long, which the device computes in a map, a reduction, or a scatter of Longs under min or max only"
                    .to_owned(),
            ),
        }
    }

    /// The inputs `module` packs into one buffer on this device.
    pub(crate) fn packing(&self, module: &Validated) -> Vec<usize> {
        let limits = self.device.limits();
        wgsl::packed_inputs(
            module,
            limits.max_buffers_and_acceleration_structures_per_shader_stage as usize,
            limits.max_storage_buffers_per_shader_stage as usize,
        )
    }

    /// Why `module`'s widest pass binds more buffers than this device does at
    /// once, if it does. Counted: its uniforms (context and parameters) and
    /// storage (every input, every output it writes, the failure record where
    /// it can fail, an emitting scatter's target), against the device's
    /// storage, uniform and total buffer limits. Such a module runs on the
    /// ordered backend.
    pub(crate) fn wider_than_device(&self, module: &Validated) -> Option<String> {
        let (uniforms, unpacked) = wgsl::bindings(module);
        let inputs = module.manifest.inputs.len();
        let written = wgsl::written_outputs(module);
        let failure = usize::from(wgsl::can_fail(module));
        let emitting = usize::from(!wgsl::emitted_pairs(module).is_empty())
            * (1 + usize::from(wgsl::highs_beside(module)));
        // Inputs packed into one buffer bind as one.
        let packed = self.packing(module).len();
        let storage = unpacked - packed + usize::from(packed > 0);
        let limits = self.device.limits();
        let most_storage = limits.max_storage_buffers_per_shader_stage as usize;
        let most_uniform = limits.max_uniform_buffers_per_shader_stage as usize;
        let most = limits.max_buffers_and_acceleration_structures_per_shader_stage as usize;
        let spelled = format!(
            "{inputs} input(s), {written} output(s){}{} and {uniforms} uniforms",
            if failure > 0 {
                ", a failure record"
            } else {
                ""
            },
            if emitting > 0 {
                ", an emitting target"
            } else {
                ""
            },
        );
        if storage > most_storage {
            return Some(format!(
                "it binds {storage} storage buffers at once ({spelled}) and the device binds at most {most_storage}"
            ));
        }
        if uniforms > most_uniform {
            return Some(format!(
                "it binds {uniforms} uniform buffers and the device binds at most {most_uniform}"
            ));
        }
        if storage + uniforms > most {
            return Some(format!(
                "it binds {} buffers at once ({spelled}) and the device binds at most {most}",
                storage + uniforms
            ));
        }
        None
    }

    /// Builds one compute pipeline from WGSL with the given bindings, or
    /// says why not ([`Unbuilt`]).
    pub(crate) fn pass(
        &self,
        label: &str,
        source: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Pass, Unbuilt> {
        let scopes = Scopes::push(&self.device);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        scopes.pop(self.wait, source)?;
        Ok((pipeline, layout))
    }

    /// Compiles `module` into a pipeline, or says why it cannot run here.
    pub(crate) fn prepare_kernel(&self, module: &Validated) -> Result<Box<dyn Prepared>, Unbuilt> {
        if let Some(reason) = self.refusal(module) {
            return Err((format!(
                "`{}`: {reason}; a session with an ordered backend runs it there",
                module.manifest.id
            ))
            .into());
        }
        // A port keeps every scalar one way, and only a map or a scan
        // writes an output narrow: the other modes fold into their
        // output in place, at full width.
        let manifest = &module.manifest;
        for port in manifest.inputs.iter().chain(&manifest.outputs) {
            if port.ty.schema.uniform_encoding().is_none() && !port.ty.schema.is_wide() {
                return Err((format!(
                    "`{}` keeps its scalars in more than one encoding ({}); a port keeps them one way",
                    port.name, port.ty.schema
                )).into());
            }
        }
        if !matches!(manifest.mode, orla_ir::Mode::Map | orla_ir::Mode::Scan)
            && manifest
                .outputs
                .iter()
                .any(|port| !port.ty.schema.is_wide())
        {
            return Err((format!(
                "`{}` is a {} and writes a narrow output; it folds at full width",
                manifest.id,
                manifest.mode.name()
            ))
            .into());
        }
        // A Kernel's ports are buffers the device binds at once, checked here
        // so a Kernel too wide for this device gets this error instead of the
        // driver's validation; a session asks `declines` first and runs it
        // elsewhere.
        if let Some(why) = self.refusal(module) {
            return Err((format!(
                "`{}`: {why}; the interpreter and the JIT have no such limit",
                module.manifest.id
            ))
            .into());
        }
        let written = wgsl::written_outputs(module);
        let packed = self.packing(module);
        let source = wgsl::emit_packed(module, &packed);
        let scopes = Scopes::push(&self.device);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&module.manifest.id.to_string()),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let mut entries = vec![
            uniform_entry(wgsl::CONTEXT_BINDING),
            uniform_entry(wgsl::PARAMS_BINDING),
            storage_entry(wgsl::OUTPUT_BINDING, false),
        ];
        for (index, port) in module.manifest.inputs.iter().enumerate() {
            if packed.contains(&index) {
                continue;
            }
            let memory = module.manifest.memories.contains(&port.name);
            entries.push(storage_entry(wgsl::INPUT_BINDING + index as u32, !memory));
        }
        if !packed.is_empty() {
            entries.push(storage_entry(wgsl::PACKED_BINDING, true));
        }
        for k in 1..written {
            entries.push(storage_entry(wgsl::output_binding(module, k), false));
        }
        let fails = wgsl::can_fail(module);
        if fails {
            entries.push(storage_entry(wgsl::FAILURE_BINDING, false));
        }
        let emits = !wgsl::emitted_pairs(module).is_empty();
        if emits {
            entries.push(storage_entry(wgsl::EMIT_BINDING, false));
            if wgsl::highs_beside(module) {
                entries.push(storage_entry(wgsl::EMIT_HIGHS_BINDING, false));
            }
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("orla kernel"),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("orla kernel"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&module.manifest.id.to_string()),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        scopes.pop(self.wait, &source)?;
        let reduce = if module.manifest.mode == orla_ir::Mode::Reduce {
            let scopes = Scopes::push(&self.device);
            let reduce_source = wgsl::emit_reduce_packed(module, &packed);
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("reduce"),
                    source: wgpu::ShaderSource::Wgsl(reduce_source.as_str().into()),
                });
            // The map pass's bindings, then the fold's own.
            let (count_binding, in_binding, out_binding) = wgsl::reduce_bindings(module);
            let mut entries = vec![
                uniform_entry(wgsl::CONTEXT_BINDING),
                uniform_entry(wgsl::PARAMS_BINDING),
            ];
            for index in 0..module.manifest.inputs.len() {
                if !packed.contains(&index) {
                    entries.push(storage_entry(wgsl::INPUT_BINDING + index as u32, true));
                }
            }
            if !packed.is_empty() {
                entries.push(storage_entry(wgsl::PACKED_BINDING, true));
            }
            entries.extend([
                uniform_entry(count_binding),
                storage_entry(in_binding, true),
                storage_entry(out_binding, false),
            ]);
            if fails {
                entries.push(storage_entry(wgsl::FAILURE_BINDING, false));
            }
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("orla reduce"),
                    entries: &entries,
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("orla reduce"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("reduce"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            scopes.pop(self.wait, &reduce_source)?;
            Some((pipeline, layout))
        } else {
            None
        };
        let prefix = if matches!(module.manifest.mode, orla_ir::Mode::Prefix { .. }) {
            let mut entries = vec![
                uniform_entry(wgsl::PREFIX_PLAN_BINDING),
                storage_entry(wgsl::PREFIX_OUT_BINDING, false),
                storage_entry(wgsl::PREFIX_SUMS_BINDING, false),
            ];
            // A whole sum's high words, and where its first total that
            // does not fit is recorded.
            if wgsl::sums_whole(module) {
                entries.extend([
                    storage_entry(wgsl::PREFIX_OUT_HIGHS_BINDING, false),
                    storage_entry(wgsl::PREFIX_SUMS_HIGHS_BINDING, false),
                    storage_entry(wgsl::FAILURE_BINDING, false),
                ]);
            }
            Some(Prefix {
                blocks: self.pass(
                    "prefix blocks",
                    &wgsl::emit_prefix(module, wgsl::PrefixPass::Blocks),
                    &entries,
                )?,
                tops: self.pass(
                    "prefix tops",
                    &wgsl::emit_prefix(module, wgsl::PrefixPass::Tops),
                    &entries,
                )?,
                offsets: self.pass(
                    "prefix offsets",
                    &wgsl::emit_prefix(module, wgsl::PrefixPass::Offsets),
                    &entries,
                )?,
                whole: wgsl::sums_whole(module),
            })
        } else {
            None
        };
        let scatter = if matches!(module.manifest.mode, orla_ir::Mode::Scatter { .. }) {
            let scatter = match wgsl::scatter_plan(module) {
                wgsl::ScatterPlan::Sorted => Scatter::Sorted {
                    keys: self.pass(
                        "scatter sort keys",
                        &wgsl::emit_scatter_sort_keys(module),
                        &[
                            uniform_entry(wgsl::CONTEXT_BINDING),
                            storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                            storage_entry(wgsl::SCATTER_SORT_KEYS_BINDING, false),
                            storage_entry(wgsl::SCATTER_SORT_ORDER_BINDING, false),
                            dynamic_uniform_entry(wgsl::SCATTER_SORT_PARAMS_BINDING),
                        ],
                    )?,
                    step: self.pass(
                        "scatter sort step",
                        &wgsl::emit_scatter_sort_step(),
                        &[
                            storage_entry(wgsl::SCATTER_SORT_KEYS_BINDING, false),
                            storage_entry(wgsl::SCATTER_SORT_ORDER_BINDING, false),
                            dynamic_uniform_entry(wgsl::SCATTER_SORT_PARAMS_BINDING),
                        ],
                    )?,
                    fold: self.pass(
                        "scatter fold",
                        &wgsl::emit_scatter_fold(module),
                        &[
                            uniform_entry(wgsl::CONTEXT_BINDING),
                            storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                            storage_entry(wgsl::SCATTER_OUT_BINDING, false),
                            storage_entry(wgsl::SCATTER_SORT_KEYS_BINDING, true),
                            storage_entry(wgsl::SCATTER_SORT_ORDER_BINDING, true),
                            dynamic_uniform_entry(wgsl::SCATTER_SORT_PARAMS_BINDING),
                        ]
                        .into_iter()
                        // Where it folds integers whole, the first total
                        // that does not fit is recorded.
                        .chain(
                            wgsl::sums_whole(module)
                                .then(|| storage_entry(wgsl::FAILURE_BINDING, false)),
                        )
                        .collect::<Vec<_>>(),
                    )?,
                    whole: wgsl::sums_whole(module),
                },
                wgsl::ScatterPlan::Parallel { last, keys } => Scatter::Parallel {
                    apply: self.pass(
                        "scatter apply",
                        &wgsl::emit_scatter_parallel(module),
                        &[
                            uniform_entry(wgsl::CONTEXT_BINDING),
                            storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                            storage_entry(wgsl::SCATTER_OUT_BINDING, false),
                            storage_entry(wgsl::SCATTER_WINNER_BINDING, false),
                        ]
                        .into_iter()
                        .chain(
                            wgsl::sums_whole(module)
                                .then(|| storage_entry(wgsl::SCATTER_HIGHS_BINDING, false)),
                        )
                        .collect::<Vec<_>>(),
                    )?,
                    totals: wgsl::sums_whole(module)
                        .then(|| -> Result<Totals, Unbuilt> {
                            Ok(Totals {
                                from_base: self.pass(
                                    "scatter highs from the base",
                                    &wgsl::emit_scatter_highs(module),
                                    &[
                                        uniform_entry(wgsl::CONTEXT_BINDING),
                                        storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                                        storage_entry(wgsl::SCATTER_HIGHS_BINDING, false),
                                    ],
                                )?,
                                check: self.pass(
                                    "scatter totals",
                                    &wgsl::emit_scatter_totals(module),
                                    &[
                                        uniform_entry(wgsl::CONTEXT_BINDING),
                                        storage_entry(wgsl::SCATTER_OUT_BINDING, true),
                                        storage_entry(wgsl::SCATTER_HIGHS_BINDING, true),
                                        storage_entry(wgsl::FAILURE_BINDING, false),
                                    ],
                                )?,
                            })
                        })
                        .transpose()?,
                    gather: last
                        .then(|| {
                            self.pass(
                                "scatter gather",
                                &wgsl::emit_scatter_gather(module),
                                &[
                                    uniform_entry(wgsl::CONTEXT_BINDING),
                                    storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                                    storage_entry(wgsl::SCATTER_OUT_BINDING, false),
                                    storage_entry(wgsl::SCATTER_WINNER_BINDING, true),
                                ],
                            )
                        })
                        .transpose()?,
                    keys: keys
                        .then(|| -> Result<Keys, Unbuilt> {
                            let entries = [
                                uniform_entry(wgsl::CONTEXT_BINDING),
                                storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                                storage_entry(wgsl::SCATTER_OUT_BINDING, false),
                            ];
                            Ok(Keys {
                                in_place: self.pass(
                                    "scatter keys",
                                    &wgsl::emit_scatter_keys(module, false),
                                    &entries,
                                )?,
                                from_base: self.pass(
                                    "scatter keys from the base",
                                    &wgsl::emit_scatter_keys(module, true),
                                    &entries,
                                )?,
                            })
                        })
                        .transpose()?,
                },
            };
            Some(scatter)
        } else {
            None
        };
        Ok(Box::new(Pipeline {
            pipeline,
            layout,
            reduce,
            scatter,
            prefix,
            fails,
            emits,
            packed,
        }))
    }
}
