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
use std::sync::{Arc, Mutex, OnceLock};

use orla_abi::{Handle, Invocation, Scope, Value};
use orla_ir::{Mode, Validated};
use orla_rules::{ExecutionError, check_invocation};
use orla_runtime::{Backend, NativeOp, Prepared};

/// A resident image: its pixels as a storage buffer of `vec4<f32>`.
pub(crate) struct DeviceImage {
    pub(crate) buffer: wgpu::Buffer,
    /// Where the buffer goes when the Field that held it is released,
    /// so the next output of its size takes it rather than asking the
    /// driver for another. `None` for a buffer that is not the
    /// backend's to keep.
    pub(crate) spare: Option<Spare>,
}

/// Buffers released Fields left behind, by size. Asking the driver for
/// tens of megabytes is most of what a full-screen pass costs, and a
/// scene asks for the same sizes every frame.
#[derive(Clone, Default)]
pub(crate) struct Spare {
    held: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

/// How many buffers the pile holds, and how much of it. A frame wants
/// a handful of sizes; beyond that they are the driver's again.
const SPARE_BUFFERS: usize = 12;
const SPARE_BYTES: u64 = 512 * 1024 * 1024;

/// The targets a rasterizer draws into, kept by shape between calls.
/// They belong to the call and to nothing else, so they go back at the
/// end of it; asking the driver for tens of megabytes of texture is
/// most of what a pass of the rasterizer costs.
#[derive(Clone, Default)]
pub(crate) struct SpareTextures {
    held: Arc<Mutex<Vec<wgpu::Texture>>>,
}

/// How many targets the pile holds, and how much of it: a rasterizer
/// wants three of a shape, and a scene rarely rasterizes into more than
/// two shapes. Beyond that they are the driver's again.
const SPARE_TEXTURES: usize = 8;
const SPARE_TEXTURE_BYTES: u64 = 512 * 1024 * 1024;

impl SpareTextures {
    /// A target of exactly this shape, from the pile or from the device.
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = held.iter().position(|texture| {
            texture.width() == width
                && texture.height() == height
                && texture.format() == format
                && texture.usage() == usage
        }) {
            return held.swap_remove(index);
        }
        drop(held);
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// Keeps a target the call is done with, while the pile is small
    /// enough to be worth keeping.
    pub(crate) fn keep(&self, texture: wgpu::Texture) {
        let bytes = |texture: &wgpu::Texture| -> u64 {
            let size = texture.format().target_pixel_byte_cost().unwrap_or(16) as u64;
            u64::from(texture.width()) * u64::from(texture.height()) * size
        };
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let held_bytes: u64 = held.iter().map(bytes).sum();
        if held.len() < SPARE_TEXTURES && held_bytes + bytes(&texture) <= SPARE_TEXTURE_BYTES {
            held.push(texture);
        }
    }
}

impl Spare {
    /// A buffer of exactly this size, from the pile or from the device.
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = held
            .iter()
            .position(|buffer| buffer.size() == size && buffer.usage() == usage)
        {
            return held.swap_remove(index);
        }
        drop(held);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Keeps a buffer a Field no longer holds, while the pile is small
    /// enough to be worth keeping.
    pub(crate) fn keep(&self, buffer: wgpu::Buffer) {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let bytes: u64 = held.iter().map(wgpu::Buffer::size).sum();
        if held.len() < SPARE_BUFFERS && bytes + buffer.size() <= SPARE_BYTES {
            held.push(buffer);
        }
    }
}

impl Drop for DeviceImage {
    fn drop(&mut self) {
        // The Field that held this is gone, so nothing in the scene
        // refers to the buffer; a command that reads it was submitted
        // before whatever writes it next, and one queue runs its
        // submissions in order.
        if let Some(spare) = self.spare.take() {
            spare.keep(self.buffer.clone());
        }
    }
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
    /// Buffers released Fields left behind; see [`Spare`].
    spare: Spare,
    /// The rasterizer's own targets, kept by shape; see [`SpareTextures`].
    targets: SpareTextures,
    /// The context and the parameters of the execution being staged.
    /// One execution runs at a time and a queue's writes are ordered
    /// with its submissions, so these are written again per node rather
    /// than made again: creating two small buffers a node was a fifth
    /// of what a node of a busy scene costs.
    uniforms: Mutex<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// What those buffers were last given. A scene of many small nodes
    /// runs them under the same context with the same parameters, and
    /// writing a uniform costs about six microseconds however little
    /// has changed; this is how the second node of a kind writes
    /// nothing.
    written: Mutex<(Vec<u8>, Vec<u8>)>,
    /// Whether an execution waits for the device before it returns, so
    /// that what times it is the work and not the submission. Off: a
    /// frame submits its nodes and only an observation waits.
    measured: std::sync::atomic::AtomicBool,
    /// Whether the adapter offers timestamp queries at a pass's edges,
    /// which measure the work itself rather than the wait around it.
    /// Offering them is not writing them: a driver that takes the
    /// queries and resolves zeros leaves the wait as the measurement.
    timestamps: bool,
    /// The nanoseconds the last measured execution took on the device,
    /// or `u64::MAX` for none.
    last_nanos: std::sync::atomic::AtomicU64,
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
        // Timestamps at a pass's edges, when the adapter has them: what
        // a profile reads. Writing one inside an encoder is a feature
        // Metal does not offer, so every pass carries its own.
        let timing = wgpu::Features::TIMESTAMP_QUERY;
        let timestamps = adapter.features().contains(timing);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orla"),
            required_features: if timestamps {
                timing
            } else {
                wgpu::Features::empty()
            },
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
            uniforms: Mutex::new(None),
            written: Mutex::new((Vec::new(), Vec::new())),
            raster: OnceLock::new(),
            spare: Spare::default(),
            targets: SpareTextures::default(),
            measured: std::sync::atomic::AtomicBool::new(false),
            timestamps,
            last_nanos: std::sync::atomic::AtomicU64::new(u64::MAX),
        })
    }

    /// Makes every execution wait for the device before it returns, so a
    /// profile measures the work rather than the submission. It costs a
    /// stall per node, so it is for measuring, not for running.
    pub fn set_measured(&self, on: bool) {
        self.measured
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether executions wait for the device; see [`GpuBackend::set_measured`].
    pub fn measured(&self) -> bool {
        self.measured.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What the device took over the last measured execution, by its own
    /// clock; `None` when nothing was measured or the adapter has no
    /// timestamps. Taken once.
    pub fn last_device_time(&self) -> Option<std::time::Duration> {
        let nanos = self
            .last_nanos
            .swap(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        (nanos != u64::MAX).then(|| std::time::Duration::from_nanos(nanos))
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
                spare: None,
            }),
        )?;
        Ok(buffer)
    }

    /// The pile of buffers released Fields left behind, for the parts
    /// of the backend that make outputs of their own.
    pub(crate) fn spare(&self) -> Spare {
        self.spare.clone()
    }

    /// The pile of rasterizer targets.
    pub(crate) fn targets(&self) -> SpareTextures {
        self.targets.clone()
    }

    /// The context and parameter buffers for the execution being
    /// staged: kept from the last one where they are big enough, and
    /// written rather than made. A bigger one takes their place.
    fn uniform_buffers(&self, context: &[u8], params: &[u8]) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut held = self
            .uniforms
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let big_enough = held
            .as_ref()
            .is_some_and(|(context_buffer, params_buffer)| {
                context_buffer.size() >= context.len() as u64
                    && params_buffer.size() >= params.len().max(16) as u64
            });
        if !big_enough {
            *held = Some((
                self.empty(
                    "context",
                    context.len().max(16),
                    wgpu::BufferUsages::UNIFORM,
                ),
                self.empty("params", params.len().max(16), wgpu::BufferUsages::UNIFORM),
            ));
        }
        let (context_buffer, params_buffer) = held.as_ref().expect("the buffers are held");
        let mut written = self
            .written
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A new buffer holds nothing of what was written before it.
        if !big_enough {
            written.0.clear();
            written.1.clear();
        }
        if written.0 != context {
            self.queue.write_buffer(context_buffer, 0, context);
            written.0.clear();
            written.0.extend_from_slice(context);
        }
        if !params.is_empty() && written.1 != params {
            self.queue.write_buffer(params_buffer, 0, params);
            written.1.clear();
            written.1.extend_from_slice(params);
        }
        (context_buffer.clone(), params_buffer.clone())
    }

    /// A buffer of this many bytes, rounded up, with nothing in it.
    fn empty(&self, label: &str, len: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(16) as u64).div_ceil(16) * 16,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
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

    fn last_device_time(&self) -> Option<std::time::Duration> {
        self.last_device_time()
    }

    /// Waits for the queue to run dry. A dispatch is submitted and not
    /// waited for, so a frame a caller times is otherwise the asking; a
    /// measurement wants this, a Player gets it from presenting, and an
    /// ordinary evaluation needs neither.
    fn settle_device(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    fn has_device(&self) -> bool {
        true
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
        let mode = module.manifest.mode;
        let domain = execution_domain(module, invocation, scope)?;
        // What comes out: one element for a reduction, the target's extent
        // for a scatter, the domain otherwise.
        let output_domain = match mode {
            Mode::Reduce => orla_abi::Domain::Rows(1),
            Mode::Scatter { base, .. } => scope.size(invocation.inputs[base as usize])?,
            _ => domain,
        };
        // Under a profile, the device's own clock around every pass of
        // this node: the set is made first so the opening timestamp is
        // the encoder's first command.
        let timing_set = (self.measured() && self.timestamps).then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("orla timing"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let staged = self.stage(
            pipeline,
            module,
            invocation,
            scope,
            domain,
            timing_set.as_ref(),
        )?;
        let Staged {
            mut encoder,
            context_buffer,
            params_buffer,
            input_buffers,
            memory_outputs,
            output_buffer,
            output_words,
            element_count,
        } = staged;
        // What the map pass left is the output, unless the mode has more
        // to do with it: a reduction folds it, a scatter combines it
        // into its target.
        let final_buffer = match (mode, pipeline.reduce.as_ref(), pipeline.scatter.as_ref()) {
            (Mode::Reduce, Some(reduce), _) => self.fold(
                &mut encoder,
                module,
                reduce,
                &context_buffer,
                &params_buffer,
                &input_buffers,
                output_buffer,
                element_count,
                output_words,
                timing_set.as_ref(),
            ),
            (Mode::Scatter { base, .. }, _, Some(scatter)) => self.combine(
                &mut encoder,
                scatter,
                &input_buffers[base as usize],
                &context_buffer,
                output_buffer,
                output_domain.count(),
                element_count,
                output_words,
                timing_set.as_ref(),
            ),
            _ => output_buffer,
        };
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let timing = timing_set.as_ref().map(|set| {
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing read"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(set, 0..2, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, 16);
            read
        });
        let submission = self.queue.submit(Some(encoder.finish()));
        // Submissions on one queue run in order, so a later node reading
        // this output needs no wait here; only observation does. Under
        // `set_measured`, this call waits, so what times it is the work.
        if self.measured() {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            });
            if let Some(read) = timing {
                let (sender, receiver) = std::sync::mpsc::channel();
                read.slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = sender.send(result);
                    });
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                if receiver.recv().is_ok_and(|result| result.is_ok())
                    && let Ok(bytes) = read.slice(..).get_mapped_range()
                {
                    let ticks = |at: usize| {
                        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]))
                    };
                    let (begin, end) = (ticks(0), ticks(8));
                    let period = self.queue.get_timestamp_period() as f64;
                    drop(bytes);
                    read.unmap();
                    // Some drivers take the queries and write nothing
                    // (Metal through wgpu, at the time of writing): then
                    // there is no device time and the wait is the answer.
                    if end > begin {
                        let nanos = ((end - begin) as f64 * period) as u64;
                        self.last_nanos
                            .store(nanos, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        } else {
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(ExecutionError::Device(error.to_string()));
        }
        adopt_outputs(
            scope,
            module,
            output_domain,
            final_buffer,
            memory_outputs,
            Some(self.spare.clone()),
        )
    }

    /// A map run several times over its own output, all of it in one
    /// command buffer: the uniforms are written once, two buffers take
    /// turns holding what a time gives to the next, and the times are
    /// passes one after another, which the queue runs in order. What
    /// this saves is a command buffer and a submission a time, which
    /// for a small node is most of what it costs.
    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        let (input, output) = chain;
        // Only the shape the planner makes: a map of one output whose
        // own output comes back on one input. A measured run times each
        // node, so it goes the ordinary way.
        if repeat < 2
            || output != 0
            || module.manifest.mode != Mode::Map
            || module.manifest.outputs.len() != 1
            || !module.manifest.memories.is_empty()
            || self.measured()
        {
            return None;
        }
        // Every time after the first reads what the one before wrote, so
        // what it reads has the extent of the output. The context says
        // the extents once; where that would not hold, the times go one
        // by one as they did before.
        let domain = execution_domain(module, invocation, scope).ok()?;
        if scope.size(*invocation.inputs.get(input)?).ok()? != domain {
            return None;
        }
        Some(self.repeated(prepared, module, invocation, scope, repeat, input))
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        match op {
            NativeOp::Rasterize => raster::rasterize(self, &self.raster, invocation, scope, false),
            NativeOp::RasterizeCutout => {
                raster::rasterize(self, &self.raster, invocation, scope, true)
            }
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

/// What the map pass leaves ready for the mode's remaining passes.
struct Staged {
    encoder: wgpu::CommandEncoder,
    context_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    input_buffers: Vec<wgpu::Buffer>,
    /// A memory's copy, written by the steps, to adopt as that output.
    memory_outputs: Vec<(usize, wgpu::Buffer, orla_abi::Domain)>,
    output_buffer: wgpu::Buffer,
    output_words: usize,
    element_count: usize,
}

/// The domain the entry maps over: a reduction's and a scatter's first
/// input, a scan's rows (checked), else the invocation's.
fn execution_domain(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<orla_abi::Domain, ExecutionError> {
    Ok(match module.manifest.mode {
        Mode::Reduce => scope.size(invocation.inputs[0])?,
        Mode::Scan => orla_rules::check_scan(invocation, scope)?,
        Mode::Scatter { base, .. } => {
            orla_rules::check_scatter(invocation, scope, base as usize)?;
            invocation.domain
        }
        Mode::Map => invocation.domain,
    })
}

/// The output and the memories as the scope's resident Fields; on a
/// failure, whatever was adopted is released.
fn adopt_outputs(
    scope: &mut Scope<'_>,
    module: &Validated,
    output_domain: orla_abi::Domain,
    final_buffer: wgpu::Buffer,
    memory_outputs: Vec<(usize, wgpu::Buffer, orla_abi::Domain)>,
    spare: Option<Spare>,
) -> Result<Vec<Handle>, ExecutionError> {
    let output_port = &module.manifest.outputs[0];
    let device = Box::new(DeviceImage {
        buffer: final_buffer,
        spare: spare.clone(),
    });
    let output = scope.adopt_resident(output_port.ty.schema.clone(), output_domain, device)?;
    let mut outputs = vec![output];
    for (index, buffer, extent) in memory_outputs {
        let port = &module.manifest.inputs[index];
        let adopted = scope.adopt_resident(
            port.ty.schema.clone(),
            extent,
            Box::new(DeviceImage {
                buffer,
                spare: spare.clone(),
            }),
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

impl GpuBackend {
    /// Binds the inputs (memories as copies the steps may write), writes
    /// the context and the parameters, creates the output, and encodes
    /// the map pass over `domain`.
    #[allow(clippy::too_many_arguments)]
    /// The whole of a repeated run: staged once, then a pass a time
    /// into one command buffer, two buffers taking turns.
    fn repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: usize,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let Some(pipeline) = prepared.as_any().downcast_ref::<Pipeline>() else {
            return Err(ExecutionError::Device(
                "kernel was prepared by a different backend".into(),
            ));
        };
        check_invocation(module, invocation)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let domain = execution_domain(module, invocation, scope)?;
        let staged = self.stage(pipeline, module, invocation, scope, domain, None)?;
        let Staged {
            mut encoder,
            context_buffer,
            params_buffer,
            mut input_buffers,
            memory_outputs: _,
            output_buffer,
            output_words,
            element_count,
        } = staged;
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // The first time reads the input it was given and writes the
        // first of the two; every time after reads what the one before
        // wrote and writes the other.
        let spare = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat"),
            size: (element_count * output_words * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (groups_x, groups_y, groups_z, _) = wgsl::dispatch(domain);
        // Two buffers take turns: what a time wrote is what the next
        // reads, and what it read is free for the time after. The Field
        // the run was given is read once, by the pass already recorded,
        // and never written.
        let mut reads = output_buffer.clone();
        let mut writes = spare;
        for _ in 1..repeat {
            input_buffers[chain] = reads.clone();
            let written = writes.clone();
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
                label: Some("orla repeat"),
                layout: &pipeline.layout,
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orla repeat"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            drop(pass);
            std::mem::swap(&mut reads, &mut writes);
        }
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(ExecutionError::Device(error.to_string()));
        }
        // What the last time wrote is what the run gives.
        adopt_outputs(
            scope,
            module,
            domain,
            reads,
            Vec::new(),
            Some(self.spare.clone()),
        )
    }

    fn stage(
        &self,
        pipeline: &Pipeline,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        domain: orla_abi::Domain,
        timing: Option<&wgpu::QuerySet>,
    ) -> Result<Staged, ExecutionError> {
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
        // The pad: two sticks in one vec4, the triggers in the next.
        let sticks = input.sticks;
        context.extend_from_slice(&[sticks[0][0], sticks[0][1], sticks[1][0], sticks[1][1]]);
        context.extend_from_slice(&[sticks[2][0], sticks[2][1], 0.0, 0.0]);
        // Inputs: bind resident buffers directly; upload the rest once and
        // leave them resident.
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(invocation.inputs.len());
        // A memory's steps write it, so the bound buffer is a copy of the
        // input's, and that copy is the memory's output afterwards.
        let mut memory_outputs: Vec<(usize, wgpu::Buffer, orla_abi::Domain)> = Vec::new();
        let mut copies: Vec<(wgpu::Buffer, wgpu::Buffer, u64)> = Vec::new();
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
                // Recorded below, into the same encoder as the pass and
                // before it, once the uniforms are settled.
                copies.push((bound, copy.clone(), size));
                memory_outputs.push((index, copy.clone(), extent));
                input_buffers.push(copy);
            } else {
                input_buffers.push(bound);
            }
        }
        let context_bytes: Vec<u8> = context.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_bytes = param_bytes(&invocation.params);
        let (context_buffer, params_buffer) = self.uniform_buffers(&context_bytes, &params_bytes);
        let output_words = module.manifest.outputs[0].ty.schema.words();
        // A scatter's map pass stores a pair per source element.
        let map_words = if matches!(module.manifest.mode, Mode::Scatter { .. }) {
            wgsl::type_words(module, wgsl::result_type(&module.entry))
        } else {
            output_words
        };
        let output_buffer = self.spare.take(
            &self.device,
            (element_count * map_words * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
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
        for (from, to, size) in &copies {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, *size);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orla kernel"),
                timestamp_writes: timing.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // A scan walks its rows in one invocation.
            if module.manifest.mode == Mode::Scan {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        }
        Ok(Staged {
            encoder,
            context_buffer,
            params_buffer,
            input_buffers,
            memory_outputs,
            output_buffer,
            output_words,
            element_count,
        })
    }

    /// A reduction's fold: level by level into a second buffer,
    /// alternating, in the same pairwise tree as the CPU reference.
    /// Returns the buffer holding the one element.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn fold(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        module: &Validated,
        reduce: &Pass,
        context_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        input_buffers: &[wgpu::Buffer],
        mapped: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&wgpu::QuerySet>,
    ) -> wgpu::Buffer {
        let (reduce_pipeline, reduce_layout) = reduce;
        let params_buffer_for_reduce = params_buffer;
        let final_buffer = mapped;
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
            // Every level closes the window; the last one to write is
            // the end of the node's work.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orla reduce"),
                timestamp_writes: timing.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(1),
                }),
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
        current
    }

    /// A scatter's combine: the target copied on the device, then the
    /// pairs applied by the plan. A float sum sorts the pairs by target
    /// and source (a bitonic sort, one small dispatch per step) and folds
    /// each target's run in source order; the order-free rules apply one
    /// invocation per pair with atomics, `min` and `max` over floats
    /// between two key passes, `last` followed by a gather. Returns the
    /// target.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scatter: &Scatter,
        base_buffer: &wgpu::Buffer,
        context_buffer: &wgpu::Buffer,
        pairs: wgpu::Buffer,
        target_count: usize,
        element_count: usize,
        output_words: usize,
        timing: Option<&wgpu::QuerySet>,
    ) -> wgpu::Buffer {
        let final_buffer = pairs;
        let size = (target_count * output_words * 4) as u64;
        let target = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter target"),
            size: size.max(16).div_ceil(16) * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(base_buffer, 0, &target, 0, size);
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
                timestamp_writes: timing.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(1),
                }),
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
                dispatch(encoder, keys, &keys_group, &[0], over_padded);
                // Every step of the sort in one pass: dispatches in a
                // pass run in order, each seeing the last one's writes.
                {
                    let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("orla scatter sort"),
                        timestamp_writes: timing.map(|query_set| {
                            wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: None,
                                end_of_pass_write_index: Some(1),
                            }
                        }),
                    });
                    compute.set_pipeline(&step.0);
                    for index in 1..records.len() {
                        compute.set_bind_group(0, &step_group, &[(index * alignment) as u32]);
                        compute.dispatch_workgroups(over_padded, 1, 1);
                    }
                }
                dispatch(encoder, fold, &fold_group, &[0], over_target);
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
                    dispatch(encoder, keys, &group, &[], over_target);
                }
                let group = bind(&apply.1, &with_winners);
                dispatch(encoder, apply, &group, &[], over_sources);
                if let Some(gather) = gather {
                    let group = bind(&gather.1, &with_winners);
                    dispatch(encoder, gather, &group, &[], over_target);
                }
                if let Some(keys) = keys {
                    let group = bind(&keys.1, &common);
                    dispatch(encoder, keys, &group, &[], over_target);
                }
            }
        }
        target
    }
}
