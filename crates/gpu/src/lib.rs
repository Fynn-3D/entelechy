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

pub mod wgsl;

use std::any::Any;
use std::fmt;
use std::sync::Mutex;

use orla_abi::{ImageHandle, Invocation, Pixel, Scope, Value};
use orla_cpu::{ExecutionError, check_invocation};
use orla_ir::Validated;
use orla_runtime::{Backend, Prepared};

/// A resident image: its pixels as a storage buffer of `vec4<f32>`.
struct DeviceImage {
    buffer: wgpu::Buffer,
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    /// Serializes executions; wgpu is thread-safe but readback ordering is
    /// simpler with one execution at a time.
    lock: Mutex<()>,
}

impl fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBackend({})", self.adapter_name)
    }
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
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
            device,
            queue,
            adapter_name,
            lock: Mutex::new(()),
        })
    }

    /// The adapter's name, for diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
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

fn pixel_bytes(pixels: &[Pixel]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 16);
    for pixel in pixels {
        for component in pixel {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    bytes
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        "gpu"
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
        Ok(Box::new(Pipeline { pipeline, layout }))
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
        let domain = invocation.domain;
        let pixel_count = domain.width as usize * domain.height as usize;

        // Context head (see `wgsl::CONTEXT_HEAD`), then one [w, h, 0, 0] per input.
        let input = &invocation.context.input;
        let mut context = vec![
            domain.width as f32,
            domain.height as f32,
            invocation.context.time,
            input.pointer_down as u8 as f32,
            input.pointer[0],
            input.pointer[1],
            0.0,
            0.0,
        ];
        context.extend(input.key_words().iter().map(|word| f32::from_bits(*word)));
        // Inputs: bind resident buffers directly; upload the rest once and
        // leave them resident.
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(invocation.inputs.len());
        for (index, handle) in invocation.inputs.iter().enumerate() {
            let (width, height) = scope.image_size(*handle)?;
            context.extend_from_slice(&[width as f32, height as f32, 0.0, 0.0]);
            let resident = scope
                .image_resident(*handle)?
                .and_then(|device| device.downcast_ref::<DeviceImage>())
                .map(|device| device.buffer.clone());
            let buffer = match resident {
                Some(buffer) => buffer,
                None => {
                    let pixels = scope.image_pixels(*handle)?;
                    let buffer = self.buffer(
                        &format!("input {index}"),
                        &pixel_bytes(pixels),
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    );
                    scope.image_make_resident(
                        *handle,
                        Box::new(DeviceImage {
                            buffer: buffer.clone(),
                        }),
                    )?;
                    buffer
                }
            };
            input_buffers.push(buffer);
        }
        let context_bytes: Vec<u8> = context.iter().flat_map(|v| v.to_le_bytes()).collect();
        let context_buffer = self.buffer("context", &context_bytes, wgpu::BufferUsages::UNIFORM);
        let params_buffer = self.buffer(
            "params",
            &param_bytes(&invocation.params),
            wgpu::BufferUsages::UNIFORM,
        );
        let output_size = (pixel_count * 16) as u64;
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
            pass.dispatch_workgroups(
                domain.width.div_ceil(wgsl::WORKGROUP),
                domain.height.div_ceil(wgsl::WORKGROUP),
                1,
            );
        }
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.queue.submit(Some(encoder.finish()));
        // Submissions on one queue run in order, so a later node reading
        // this output needs no wait here; only observation does.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(ExecutionError::Device(error.to_string()));
        }
        let output = scope.image_adopt_resident(
            domain.width,
            domain.height,
            Box::new(DeviceImage {
                buffer: output_buffer,
            }),
        )?;
        Ok(vec![output])
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        width: u32,
        height: u32,
    ) -> Result<Vec<Pixel>, String> {
        let image = device
            .downcast_ref::<DeviceImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pixel_count = width as usize * height as usize;
        let size = (pixel_count * 16) as u64;
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
        let pixels: Vec<Pixel> = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            view.chunks_exact(16)
                .take(pixel_count)
                .map(|chunk| {
                    std::array::from_fn(|i| {
                        f32::from_le_bytes([
                            chunk[i * 4],
                            chunk[i * 4 + 1],
                            chunk[i * 4 + 2],
                            chunk[i * 4 + 3],
                        ])
                    })
                })
                .collect()
        };
        staging.unmap();
        Ok(pixels)
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
