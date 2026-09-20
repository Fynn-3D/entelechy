#![warn(missing_docs)]
//! GPU backend.
//!
//! Emits WGSL from validated IR at preparation time and runs one compute
//! dispatch per execution through wgpu. The Host keeps owning images, but
//! their pixels stay resident on the device: an input that is already
//! resident is bound directly, an input in Host memory is uploaded once and
//! made resident, and outputs are adopted by the Host as resident images.
//! Pixels come back only at observation boundaries, through
//! [`Backend::download`]. Semantics match the CPU backend except for
//! floating-point rounding and the absence of per-pixel integer arithmetic
//! errors, which WGSL defines instead of reporting.

pub mod present;
mod raster;
pub mod wgsl;

use std::any::Any;
use std::fmt;
use std::sync::{Mutex, OnceLock};

use orla_abi::{Handle, Invocation, Scope, Value};
use orla_cpu::{ExecutionError, check_invocation};
use orla_ir::Validated;
use orla_runtime::{Backend, NativeOp, Prepared};

/// A resident image: its pixels as a storage buffer of `vec4<f32>`.
pub(crate) struct DeviceImage {
    pub(crate) buffer: wgpu::Buffer,
}

/// The GPU could not be opened.
#[derive(Debug)]
pub struct GpuError(pub String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu unavailable: {}", self.0)
    }
}

impl std::error::Error for GpuError {}

/// A wgpu device with a compute-only contract.
pub struct GpuBackend {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    /// Serializes executions; wgpu is thread-safe but readback ordering is
    /// simpler with one execution at a time.
    lock: Mutex<()>,
    /// The rasterizer's pipelines, built on first use.
    raster: OnceLock<Result<raster::RasterPipelines, String>>,
}

impl fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBackend({})", self.adapter_name)
    }
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The fold of a reduce module, run level by level after the map.
    reduce: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// The combine of a scatter module, run after the map.
    scatter: Option<Scatter>,
}

/// One compute pipeline with its bind group layout.
type Pass = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

/// A scatter's combine as pipelines, by its plan.
enum Scatter {
    /// A float sum: sort keys, the sort's step, and the fold.
    Sorted { keys: Pass, step: Pass, fold: Pass },
    /// An order-free rule: the parallel apply; the gather after `last`;
    /// the key transform around `min` and `max` over floats, the same
    /// pass before and after.
    Parallel {
        apply: Pass,
        gather: Option<Pass>,
        keys: Option<Pass>,
    },
}

impl Prepared for Pipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GpuBackend {
    /// Opens the default adapter.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| GpuError(error.to_string()))?;
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orla"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        }))
        .map_err(|error| GpuError(error.to_string()))?;
        let adapter_name = adapter.get_info().name;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_name,
            lock: Mutex::new(()),
            raster: OnceLock::new(),
        })
    }

    /// The adapter's name, for diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The wgpu device, for interfaces and tests that speak wgpu directly.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn parts(&self) -> (&wgpu::Instance, &wgpu::Adapter, &wgpu::Device, &wgpu::Queue) {
        (&self.instance, &self.adapter, &self.device, &self.queue)
    }

    /// The device buffer holding an input: the resident one, or an upload
    /// of its words that then stays resident.
    /// Builds one compute pipeline from WGSL with the given bindings, or
    /// returns the validation error with the source.
    fn pass(
        &self,
        label: &str,
        source: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Pass, String> {
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(format!("{error}\n--- emitted WGSL ---\n{source}"));
        }
        Ok((pipeline, layout))
    }

    pub(crate) fn bind_input(
        &self,
        scope: &mut Scope<'_>,
        handle: Handle,
        index: usize,
    ) -> Result<wgpu::Buffer, ExecutionError> {
        let resident = scope
            .resident(handle)?
            .and_then(|device| device.downcast_ref::<DeviceImage>())
            .map(|device| device.buffer.clone());
        if let Some(buffer) = resident {
            return Ok(buffer);
        }
        let bytes = scope
            .words(handle)?
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>();
        let buffer = self.buffer(
            &format!("input {index}"),
            &bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        scope.make_resident(
            handle,
            Box::new(DeviceImage {
                buffer: buffer.clone(),
            }),
        )?;
        Ok(buffer)
    }

    pub(crate) fn buffer(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let size = (contents.len().max(16) as u64).div_ceil(16) * 16;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !contents.is_empty() {
            self.queue.write_buffer(&buffer, 0, contents);
        }
        buffer
    }
}

fn param_bytes(params: &[Value]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(params.len().max(1) * 16);
    for value in params {
        let words: [u32; 4] = match value {
            Value::Float(v) => [v.to_bits(), 0, 0, 0],
            Value::Int(v) => [*v as u32, 0, 0, 0],
            Value::Bool(v) => [(*v as u8 as f32).to_bits(), 0, 0, 0],
            Value::Vec2([x, y]) => [x.to_bits(), y.to_bits(), 0, 0],
            Value::Vec3([x, y, z]) => [x.to_bits(), y.to_bits(), z.to_bits(), 0],
            Value::Vec4([x, y, z, w]) => [x.to_bits(), y.to_bits(), z.to_bits(), w.to_bits()],
        };
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    if bytes.is_empty() {
        bytes.resize(16, 0);
    }
    bytes
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let source = wgsl::emit(module);
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
            let memory = module.manifest.memories.contains(&port.name);
            entries.push(storage_entry(wgsl::INPUT_BINDING + index as u32, !memory));
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
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(format!("{error}\n--- emitted WGSL ---\n{source}"));
        }
        let reduce = if module.manifest.mode == orla_ir::Mode::Reduce {
            let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let reduce_source = wgsl::emit_reduce(module);
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
                entries.push(storage_entry(wgsl::INPUT_BINDING + index as u32, true));
            }
            entries.extend([
                uniform_entry(count_binding),
                storage_entry(in_binding, true),
                storage_entry(out_binding, false),
            ]);
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
            if let Some(error) = pollster::block_on(scope_guard.pop()) {
                return Err(format!("{error}\n--- emitted WGSL ---\n{reduce_source}"));
            }
            Some((pipeline, layout))
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
                        ],
                    )?,
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
                        ],
                    )?,
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
                        .then(|| {
                            self.pass(
                                "scatter keys",
                                &wgsl::emit_scatter_keys(module, true),
                                &[
                                    uniform_entry(wgsl::CONTEXT_BINDING),
                                    storage_entry(wgsl::SCATTER_PAIRS_BINDING, true),
                                    storage_entry(wgsl::SCATTER_OUT_BINDING, false),
                                ],
                            )
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
        }))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        check_invocation(module, invocation)?;
        let Some(pipeline) = prepared.as_any().downcast_ref::<Pipeline>() else {
            return Err(ExecutionError::Device(
                "kernel was prepared by a different backend".into(),
            ));
        };
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A reduction maps over its first input and folds afterwards; a
        // scan walks its first input's rows in one invocation; a scatter
        // maps over its first input into pairs and combines them into a
        // copy of its target afterwards.
        let reducing = module.manifest.mode == orla_ir::Mode::Reduce;
        let scanning = module.manifest.mode == orla_ir::Mode::Scan;
        let scattering = match module.manifest.mode {
            orla_ir::Mode::Scatter { base, .. } => Some(base as usize),
            _ => None,
        };
        let domain = if reducing {
            scope.size(invocation.inputs[0])?
        } else if scanning {
            orla_cpu::check_scan(invocation, scope)?
        } else if let Some(base) = scattering {
            orla_cpu::check_scatter(invocation, scope, base)?;
            scope.size(invocation.inputs[0])?
        } else {
            invocation.domain
        };
        let element_count = domain.count();

        // Context head (see `wgsl::CONTEXT_HEAD`), then one [w, h, d, 0] per input.
        let input = &invocation.context.input;
        let mut context = vec![
            domain.width() as f32,
            domain.height() as f32,
            invocation.context.time,
            input.button_bits() as f32,
            input.pointer[0],
            input.pointer[1],
            domain.depth() as f32,
            invocation.context.rate,
        ];
        context.extend(input.key_words().iter().map(|word| f32::from_bits(*word)));
        let window = invocation.context.window;
        context.extend_from_slice(&[input.wheel[0], input.wheel[1], window[0], window[1]]);
        let (groups_x, groups_y, groups_z, fold) = wgsl::dispatch(domain);
        // The third word is a one the shader compiler cannot see through
        // (`wgsl::OPAQUE_ONE`); the fourth is spare.
        context.extend_from_slice(&[invocation.context.scale, fold as f32, 1.0, 0.0]);
        // Inputs: bind resident buffers directly; upload the rest once and
        // leave them resident.
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(invocation.inputs.len());
        // A memory's steps write it, so the bound buffer is a copy of the
        // input's, and that copy is the memory's output afterwards.
        let mut memory_outputs: Vec<(usize, wgpu::Buffer, orla_abi::Domain)> = Vec::new();
        let mut copies = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orla memories"),
            });
        for (index, handle) in invocation.inputs.iter().enumerate() {
            let extent = scope.size(*handle)?;
            context.extend_from_slice(&[
                extent.width() as f32,
                extent.height() as f32,
                extent.depth() as f32,
                0.0,
            ]);
            let bound = self.bind_input(scope, *handle, index)?;
            let port = &module.manifest.inputs[index];
            if module.manifest.memories.contains(&port.name) {
                let size = (extent.count() * port.ty.schema.words() * 4) as u64;
                let copy = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("memory"),
                    size: size.max(16).div_ceil(16) * 16,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                copies.copy_buffer_to_buffer(&bound, 0, &copy, 0, size);
                memory_outputs.push((index, copy.clone(), extent));
                input_buffers.push(copy);
            } else {
                input_buffers.push(bound);
            }
        }
        if !memory_outputs.is_empty() {
            self.queue.submit(Some(copies.finish()));
        }
        let context_bytes: Vec<u8> = context.iter().flat_map(|v| v.to_le_bytes()).collect();
        let context_buffer = self.buffer("context", &context_bytes, wgpu::BufferUsages::UNIFORM);
        let params_buffer = self.buffer(
            "params",
            &param_bytes(&invocation.params),
            wgpu::BufferUsages::UNIFORM,
        );
        let output_port = &module.manifest.outputs[0];
        let output_words = output_port.ty.schema.words();
        // A scatter's map pass stores a pair per source element.
        let map_words = if scattering.is_some() {
            wgsl::type_words(module, wgsl::result_type(&module.entry))
        } else {
            output_words
        };
        let output_size = (element_count * map_words * 4) as u64;
        let params_buffer_for_reduce = params_buffer.clone();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
                resource: output_buffer.as_entire_binding(),
            },
        ];
        for (index, buffer) in input_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::INPUT_BINDING + index as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orla kernel"),
            layout: &pipeline.layout,
            entries: &entries,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orla"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orla kernel"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            if scanning {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        }
        // The fold: level by level into a second buffer, alternating, in
        // the same pairwise tree as the CPU reference.
        let mut final_buffer = output_buffer;
        if let Some((reduce_pipeline, reduce_layout)) = pipeline.reduce.as_ref() {
            let stride = output_words as u64 * 4;
            let mut count = element_count as u64;
            let other = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce level"),
                size: (count.div_ceil(2) * stride).max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut current = final_buffer;
            let mut next = other;
            while count > 1 {
                let level: [u32; 4] = [count as u32, 0, 0, 0];
                let level_bytes: Vec<u8> = level.iter().flat_map(|v| v.to_le_bytes()).collect();
                let level_buffer =
                    self.buffer("reduce count", &level_bytes, wgpu::BufferUsages::UNIFORM);
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
                for (index, buffer) in input_buffers.iter().enumerate() {
                    fold_entries.push(wgpu::BindGroupEntry {
                        binding: wgsl::INPUT_BINDING + index as u32,
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
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("orla reduce"),
                    layout: reduce_layout,
                    entries: &fold_entries,
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("orla reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(reduce_pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(
                    (count.div_ceil(2) as u32).div_ceil(wgsl::REDUCE_WORKGROUP),
                    1,
                    1,
                );
                drop(pass);
                count = count.div_ceil(2);
                std::mem::swap(&mut current, &mut next);
            }
            final_buffer = current;
        }
        // The combine: the target copied on the device, then the pairs
        // applied by the plan. A float sum sorts the pairs by target and
        // source (a bitonic sort, one small dispatch per step) and folds
        // each target's run in source order; the order-free rules apply
        // one invocation per pair with atomics, `min` and `max` over
        // floats between two key passes, `last` followed by a gather.
        if let (Some(base), Some(scatter)) = (scattering, pipeline.scatter.as_ref()) {
            let target_count = invocation.domain.count();
            let size = (target_count * output_words * 4) as u64;
            let target = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter target"),
                size: size.max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&input_buffers[base], 0, &target, 0, size);
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
                    label: Some("orla scatter"),
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
                    label: Some("orla scatter"),
                    timestamp_writes: None,
                });
                compute.set_pipeline(&pass.0);
                compute.set_bind_group(0, group, offsets);
                compute.dispatch_workgroups(groups, 1, 1);
            };
            match scatter {
                Scatter::Sorted { keys, step, fold } => {
                    let padded = element_count.next_power_of_two().max(1);
                    let keys_buffer = storage("scatter sort keys", padded);
                    let order_buffer = storage("scatter sort order", padded);
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
                        self.buffer("scatter sort params", &bytes, wgpu::BufferUsages::UNIFORM);
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
                            .collect::<Vec<_>>(),
                    );
                    let over_padded = (padded as u32).div_ceil(wgsl::SCATTER_WORKGROUP);
                    dispatch(&mut encoder, keys, &keys_group, &[0], over_padded);
                    // Every step of the sort in one pass: dispatches in a
                    // pass run in order, each seeing the last one's writes.
                    {
                        let mut compute =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("orla scatter sort"),
                                timestamp_writes: None,
                            });
                        compute.set_pipeline(&step.0);
                        for index in 1..records.len() {
                            compute.set_bind_group(0, &step_group, &[(index * alignment) as u32]);
                            compute.dispatch_workgroups(over_padded, 1, 1);
                        }
                    }
                    dispatch(&mut encoder, fold, &fold_group, &[0], over_target);
                }
                Scatter::Parallel {
                    apply,
                    gather,
                    keys,
                } => {
                    // Winners for `last`, zeroed as every new buffer is; a
                    // token buffer where the pass has none to record.
                    let winners = storage(
                        "scatter winners",
                        if gather.is_some() { target_count } else { 1 },
                    );
                    let with_winners: Vec<wgpu::BindGroupEntry> = common
                        .iter()
                        .cloned()
                        .chain([wgpu::BindGroupEntry {
                            binding: wgsl::SCATTER_WINNER_BINDING,
                            resource: winners.as_entire_binding(),
                        }])
                        .collect();
                    if let Some(keys) = keys {
                        let group = bind(&keys.1, &common);
                        dispatch(&mut encoder, keys, &group, &[], over_target);
                    }
                    let group = bind(&apply.1, &with_winners);
                    dispatch(&mut encoder, apply, &group, &[], over_sources);
                    if let Some(gather) = gather {
                        let group = bind(&gather.1, &with_winners);
                        dispatch(&mut encoder, gather, &group, &[], over_target);
                    }
                    if let Some(keys) = keys {
                        let group = bind(&keys.1, &common);
                        dispatch(&mut encoder, keys, &group, &[], over_target);
                    }
                }
            }
            final_buffer = target;
        }
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.queue.submit(Some(encoder.finish()));
        // Submissions on one queue run in order, so a later node reading
        // this output needs no wait here; only observation does.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(ExecutionError::Device(error.to_string()));
        }
        let device = Box::new(DeviceImage {
            buffer: final_buffer,
        });
        let output_domain = if reducing {
            orla_abi::Domain::Rows(1)
        } else if scattering.is_some() {
            invocation.domain
        } else {
            domain
        };
        let output = scope.adopt_resident(output_port.ty.schema.clone(), output_domain, device)?;
        let mut outputs = vec![output];
        for (index, buffer, extent) in memory_outputs {
            let port = &module.manifest.inputs[index];
            let adopted = scope.adopt_resident(
                port.ty.schema.clone(),
                extent,
                Box::new(DeviceImage { buffer }),
            );
            match adopted {
                Ok(handle) => outputs.push(handle),
                Err(error) => {
                    for handle in outputs {
                        let _ = scope.release(handle);
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(outputs)
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        match op {
            NativeOp::Rasterize => raster::rasterize(self, &self.raster, invocation, scope),
        }
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        let image = device
            .downcast_ref::<DeviceImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let size = byte_len as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("orla download"),
            });
        encoder.copy_buffer_to_buffer(&image.buffer, 0, &staging, 0, size);
        let submission = self.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| error.to_string())?;
        receiver
            .recv()
            .map_err(|_| "readback was dropped".to_string())?
            .map_err(|error| error.to_string())?;
        let bytes: Vec<u8> = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            view[..byte_len].to_vec()
        };
        staging.unmap();
        Ok(bytes)
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
fn dynamic_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
