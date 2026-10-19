//! Turning a validated module into what the device runs: its WGSL,
//! pipelines and binding layouts, or why it cannot run here.

use super::*;

/// What the device runs for a Kernel, its passes `P` built ([`Pass`]),
/// or validated and waiting for the device's compiler ([`Shader`]).
pub(crate) struct Pipeline<P = Pass> {
    /// The map, or a mode's entry.
    pub(crate) main: P,
    /// The fold of a reduce module, run level by level after the map.
    pub(crate) reduce: Option<P>,
    /// The combine of a scatter module, run after the map.
    pub(crate) scatter: Option<Scatter<P>>,
    /// The fold of a prefix module, run over the map's output in place.
    pub(crate) prefix: Option<Prefix<P>>,
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
pub(crate) struct Prefix<P = Pass> {
    pub(crate) blocks: P,
    pub(crate) tops: P,
    pub(crate) offsets: P,
    /// Whether it sums whole ([`wgsl::sums_whole`]): high words beside
    /// the output and the totals, and a failure record.
    pub(crate) whole: bool,
}

/// One compute pipeline with its bind group layout.
pub(crate) type Pass = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

/// The key transform's two passes: in place, and from the base.
pub(crate) struct Keys<P = Pass> {
    pub(crate) in_place: P,
    pub(crate) from_base: P,
}

/// A whole sum's two passes around the parallel apply: the high words
/// set from the base, and the totals checked once every value is in.
pub(crate) struct Totals<P = Pass> {
    pub(crate) from_base: P,
    pub(crate) check: P,
}

/// A scatter's combine as pipelines, by its plan.
pub(crate) enum Scatter<P = Pass> {
    /// A float sum: sort keys, the sort's step, and the fold, which with
    /// `whole` also folds integers whole and records a total that does
    /// not fit.
    Sorted {
        keys: P,
        step: P,
        fold: P,
        whole: bool,
    },
    /// An order-free rule: the parallel apply; the gather after `last`; for
    /// `min` and `max` over floats and longs, the key transform in place
    /// after, and before from the base into the target, sparing a base copy.
    Parallel {
        apply: P,
        gather: Option<P>,
        keys: Option<Keys<P>>,
        totals: Option<Totals<P>>,
    },
}

/// A pass validated as the device takes it, its pipeline not yet built:
/// its source and its bindings. The shader module the device made of the
/// source to validate it is not kept: a module holds the device's whole
/// reading of its shader, many times the source, and a scene prepares
/// many Kernels it runs late, rarely or only fused into others. One is
/// made again for the pipeline, and let go once it is built.
pub(crate) struct Shader {
    label: String,
    source: String,
    /// The bindings, which a page's browser compiles the pipeline with.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    layout: wgpu::BindGroupLayout,
}

impl Shader {
    /// The pipeline, from the device's own compiler: the long part of
    /// preparing, and the part a device's compiler can fail in.
    fn built(self, device: &wgpu::Device, wait: Wait) -> Result<Pass, Unbuilt> {
        // A page's browser compiles it first, beside the thread that
        // answers the device; elsewhere the device does by itself.
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        crate::page::compile_beside(device, wait, &self.label, &self.source, &self.entries);
        let scopes = Scopes::push(device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&self.label),
            source: wgpu::ShaderSource::Wgsl(self.source.as_str().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&self.label),
            bind_group_layouts: &[Some(&self.layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&self.label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        scopes.pop(wait, &self.source)?;
        Ok((pipeline, self.layout))
    }
}

/// Each pass of a [`Pipeline`] and its parts, made into another, in order.
type Make<'a, P, Q> = &'a mut dyn FnMut(P) -> Result<Q, Unbuilt>;

impl<P> Pipeline<P> {
    fn made<Q>(self, make: Make<'_, P, Q>) -> Result<Pipeline<Q>, Unbuilt> {
        Ok(Pipeline {
            main: make(self.main)?,
            reduce: self.reduce.map(&mut *make).transpose()?,
            scatter: self.scatter.map(|s| s.made(make)).transpose()?,
            prefix: self.prefix.map(|p| p.made(make)).transpose()?,
            fails: self.fails,
            emits: self.emits,
            packed: self.packed,
        })
    }
}

impl<P> Prefix<P> {
    fn made<Q>(self, make: Make<'_, P, Q>) -> Result<Prefix<Q>, Unbuilt> {
        Ok(Prefix {
            blocks: make(self.blocks)?,
            tops: make(self.tops)?,
            offsets: make(self.offsets)?,
            whole: self.whole,
        })
    }
}

impl<P> Scatter<P> {
    fn made<Q>(self, make: Make<'_, P, Q>) -> Result<Scatter<Q>, Unbuilt> {
        Ok(match self {
            Self::Sorted {
                keys,
                step,
                fold,
                whole,
            } => Scatter::Sorted {
                keys: make(keys)?,
                step: make(step)?,
                fold: make(fold)?,
                whole,
            },
            Self::Parallel {
                apply,
                gather,
                keys,
                totals,
            } => Scatter::Parallel {
                apply: make(apply)?,
                gather: gather.map(&mut *make).transpose()?,
                keys: keys
                    .map(|k| -> Result<Keys<Q>, Unbuilt> {
                        Ok(Keys {
                            in_place: make(k.in_place)?,
                            from_base: make(k.from_base)?,
                        })
                    })
                    .transpose()?,
                totals: totals
                    .map(|t| -> Result<Totals<Q>, Unbuilt> {
                        Ok(Totals {
                            from_base: make(t.from_base)?,
                            check: make(t.check)?,
                        })
                    })
                    .transpose()?,
            },
        })
    }
}

/// A Kernel as the device prepared it: validated, its pipelines built the
/// first time a run needs them, or when [`Backend::finish`] asks; once
/// built, kept. A device's compiler that fails on it then is remembered
/// ([`GpuBackend::declines`]), so a runtime with another backend runs it
/// there.
pub(crate) struct Lazy {
    id: String,
    module: Validated,
    /// Its inputs packed into one buffer, known before it is built.
    pub(crate) packed: Vec<usize>,
    device: wgpu::Device,
    wait: Wait,
    unbuilt: Arc<Mutex<Vec<(Validated, String)>>>,
    shaders: Mutex<Option<Pipeline<Shader>>>,
    built: OnceLock<Result<Pipeline, String>>,
}

impl Lazy {
    /// The pipelines, built now if they are not yet; a second caller waits
    /// for the first.
    pub(crate) fn built(&self) -> Result<&Pipeline, ExecutionError> {
        self.built
            .get_or_init(|| {
                let shaders = self
                    .shaders
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .ok_or_else(|| format!("`{}` was left unbuilt by a failed build", self.id))?;
                let device = &self.device;
                let wait = self.wait;
                match shaders.made(&mut |shader: Shader| shader.built(device, wait)) {
                    Ok(pipeline) => Ok(pipeline),
                    Err(Unbuilt::Ours(error)) => Err(error),
                    Err(Unbuilt::Device(error)) => {
                        // Its most particular line, the last, said shortly.
                        let last = error
                            .lines()
                            .rev()
                            .find(|line| !line.trim().is_empty())
                            .unwrap_or("")
                            .trim();
                        let short: String = last.chars().take(160).collect();
                        let mut unbuilt = self
                            .unbuilt
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if !unbuilt.iter().any(|(seen, _)| seen == &self.module) {
                            unbuilt.push((self.module.clone(), short));
                        }
                        Err(format!(
                            "the device's own compiler could not build `{}`: {error}",
                            self.id
                        ))
                    }
                }
            })
            .as_ref()
            .map_err(|error| ExecutionError::Device(error.clone()))
    }
}

/// The pipelines of a Kernel this device prepared, built now if they are
/// not yet.
pub(crate) fn built(prepared: &dyn Prepared) -> Result<&Pipeline, ExecutionError> {
    prepared
        .as_any()
        .downcast_ref::<Lazy>()
        .ok_or_else(|| ExecutionError::Device("kernel was prepared by a different backend".into()))?
        .built()
}

impl Prepared for Lazy {
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
        if module.holds(entelechy_abi::Type::Double) {
            return Some("it holds a Double, which the device does not compute".to_owned());
        }

        if !module.holds(entelechy_abi::Type::Long) {
            return None;
        }
        if !self.int64 {
            // Without the device's own, a Long is a pair of words where the
            // module allows it.
            return (!wgsl::pairs_take(module)).then(|| {
                "it holds a Long the device, which has no 64-bit integers, computes only as scalars in a map or a reduction"
                    .to_owned()
            });
        }
        match module.manifest.mode {
            entelechy_ir::Mode::Map | entelechy_ir::Mode::Reduce => None,
            entelechy_ir::Mode::Scatter { .. } if wgsl::long_extremes(module) => (!self.int64_atomics)
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

    /// One pass from WGSL with the given bindings, validated as the device
    /// takes it and not yet built, or why not ([`Unbuilt`]).
    pub(crate) fn pass(
        &self,
        label: &str,
        source: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Shader, Unbuilt> {
        // Only what the entry point reaches goes to the device.
        let source = &wgsl::pruned(source);
        let scopes = Scopes::push(&self.device);
        // Validated as the device takes it, and let go ([`Shader`]). A page
        // cannot let a module go but to its collector: there the source is
        // validated as the pipeline is built.
        if !self.wait.parks_with_host() {
            drop(
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    }),
            );
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
        scopes.pop(self.wait, source)?;
        Ok(Shader {
            label: label.to_owned(),
            source: source.to_owned(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            entries: entries.to_vec(),
            layout,
        })
    }

    /// `module` as the device takes it, every pass validated and none
    /// built ([`Lazy`]), or why it cannot run here.
    pub(crate) fn prepare_kernel(&self, module: &Validated) -> Result<Lazy, Unbuilt> {
        let shaders = self.shaders(module)?;
        Ok(Lazy {
            id: module.manifest.id.to_string(),
            module: module.clone(),
            packed: shaders.packed.clone(),
            device: self.device.clone(),
            wait: self.wait,
            unbuilt: Arc::clone(&self.unbuilt),
            shaders: Mutex::new(Some(shaders)),
            built: OnceLock::new(),
        })
    }

    /// `module`'s passes, validated.
    fn shaders(&self, module: &Validated) -> Result<Pipeline<Shader>, Unbuilt> {
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
        if !matches!(
            manifest.mode,
            entelechy_ir::Mode::Map | entelechy_ir::Mode::Scan
        ) && manifest
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
        let longs = self.longs();
        let source = wgsl::emit_packed_longs(module, &packed, longs);
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
        let main = self.pass(&module.manifest.id.to_string(), &source, &entries)?;
        let reduce = if module.manifest.mode == entelechy_ir::Mode::Reduce {
            let reduce_source = wgsl::emit_reduce_packed_longs(module, &packed, longs);
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
            Some(self.pass("reduce", &reduce_source, &entries)?)
        } else {
            None
        };
        let prefix = if matches!(module.manifest.mode, entelechy_ir::Mode::Prefix { .. }) {
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
        let scatter = if matches!(module.manifest.mode, entelechy_ir::Mode::Scatter { .. }) {
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
                        .then(|| -> Result<Totals<Shader>, Unbuilt> {
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
                        .then(|| -> Result<Keys<Shader>, Unbuilt> {
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
        Ok(Pipeline {
            main,
            reduce,
            scatter,
            prefix,
            fails,
            emits,
            packed,
        })
    }
}
