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

use orla_abi::{ImageHandle, Invocation, Scope, Value};
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
    pub(crate) fn bind_input(
        &self,
        scope: &mut Scope<'_>,
        handle: ImageHandle,
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
        for index in 0..module.manifest.inputs.len() {
            entries.push(storage_entry(wgsl::INPUT_BINDING + index as u32, true));
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
            let entries = [
                uniform_entry(wgsl::REDUCE_COUNT_BINDING),
                storage_entry(wgsl::REDUCE_IN_BINDING, true),
                storage_entry(wgsl::REDUCE_OUT_BINDING, false),
                uniform_entry(3),
            ];
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
        Ok(Box::new(Pipeline {
            pipeline,
            layout,
            reduce,
        }))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
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
        // scan walks its first input's rows in one invocation.
        let reducing = module.manifest.mode == orla_ir::Mode::Reduce;
        let scanning = module.manifest.mode == orla_ir::Mode::Scan;
        let domain = if reducing {
            scope.size(invocation.inputs[0])?
        } else if scanning {
            orla_cpu::check_scan(invocation, scope)?
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
            input.pointer_down as u8 as f32,
            input.pointer[0],
            input.pointer[1],
            domain.depth() as f32,
            0.0,
        ];
        context.extend(input.key_words().iter().map(|word| f32::from_bits(*word)));
        // Inputs: bind resident buffers directly; upload the rest once and
        // leave them resident.
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(invocation.inputs.len());
        for (index, handle) in invocation.inputs.iter().enumerate() {
            let extent = scope.size(*handle)?;
            context.extend_from_slice(&[
                extent.width() as f32,
                extent.height() as f32,
                extent.depth() as f32,
                0.0,
            ]);
            input_buffers.push(self.bind_input(scope, *handle, index)?);
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
        let output_size = (element_count * output_words * 4) as u64;
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
                pass.dispatch_workgroups(
                    domain.width().div_ceil(wgsl::WORKGROUP),
                    domain.height().div_ceil(wgsl::WORKGROUP),
                    domain.depth(),
                );
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
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("orla reduce"),
                    layout: reduce_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: wgsl::REDUCE_COUNT_BINDING,
                            resource: level_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: wgsl::REDUCE_IN_BINDING,
                            resource: current.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: wgsl::REDUCE_OUT_BINDING,
                            resource: next.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer_for_reduce.as_entire_binding(),
                        },
                    ],
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
        } else {
            domain
        };
        let output = scope.adopt_resident(output_port.ty.schema.clone(), output_domain, device)?;
        Ok(vec![output])
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
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
