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
use orla_rules::{ArithmeticError, Backend, Check, NativeOp, Prepared};
use orla_rules::{ExecutionError, check_invocation};

/// What asking the device's own clock has been worth; see
/// [`GpuBackend::device_clock_answered`].
mod clock {
    /// Nobody has asked yet.
    pub const UNTRIED: u8 = 0;
    /// The clock answered, so a profile prints what the device says.
    pub const ANSWERED: u8 = 1;
    /// The driver took the queries and resolved nothing, so a profile
    /// prints the call and its wait, and nothing asks again.
    pub const SILENT: u8 = 2;
}

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
/// a handful of sizes; beyond that they are the driver's again. Ten
/// million triangles drawn at 4K pass half a gigabyte of them a frame,
/// and a pile smaller than that makes some anew every frame.
const SPARE_BUFFERS: usize = 12;
const SPARE_BYTES: u64 = 1024 * 1024 * 1024;

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
    /// A buffer at least this size, from the pile or from the device.
    ///
    /// At least, not exactly: nothing reads past the element count the
    /// uniform gives — no shader asks a buffer how long it is — so the
    /// tail of a bigger one is never touched. It is what makes the pile
    /// work for a scene whose Fields change size, which is every scene
    /// that sizes a node by what a Kernel decided: a count that wanders
    /// by a chunk between frames missed an exact match every time and
    /// went to the driver. The smallest that fits is taken, and never
    /// one more than twice the size, so a Field of a thousand elements
    /// cannot be handed the buffer of a million.
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let fits = |buffer: &wgpu::Buffer| {
            buffer.usage() == usage
                && buffer.size() >= size
                && buffer.size() <= size.saturating_mul(2).max(256)
        };
        if let Some(index) = held
            .iter()
            .enumerate()
            .filter(|(_, buffer)| fits(buffer))
            .min_by_key(|(_, buffer)| buffer.size())
            .map(|(index, _)| index)
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
    /// Whether the device computes 64-bit integers.
    int64: bool,
    /// Whether it takes the least and the greatest of them atomically,
    /// which a scatter of `Long`s under `min` or `max` lands with.
    int64_atomics: bool,
    /// Failure words and the buffers they are read back through, ready
    /// to be used again: a checked node takes a pair each run.
    failures: Arc<Mutex<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
    /// Bytes uploaded from Host memory since the backend was made.
    uploaded: std::sync::atomic::AtomicU64,
    /// What asking the device's own clock has been worth: `UNTRIED`
    /// until a measured run asks, then `ANSWERED` or `SILENT`. A driver
    /// that offers timestamps and resolves zeros is asked once and not
    /// again — the asking is a buffer mapped and waited for per node,
    /// which is most of what a profile's overhead was.
    clock: std::sync::atomic::AtomicU8,
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
    /// The fold of a prefix module, run over the map's output in place.
    prefix: Option<Prefix>,
    /// Whether the module's arithmetic can fail, so its map and fold
    /// passes record their first failure; see [`wgsl::can_fail`].
    fails: bool,
    /// Whether a scatter's entry emits pairs, combining them into the
    /// target as it runs: the target is then ready before the entry.
    emits: bool,
    /// The inputs its passes read from one buffer, copied there before
    /// they run, because binding each would be more than the device binds
    /// at once; see [`wgsl::packed_inputs`].
    packed: Vec<usize>,
}

/// A prefix fold's three passes: a block a walk, the blocks' totals
/// scanned, then every element given its block's offset.
struct Prefix {
    blocks: Pass,
    tops: Pass,
    offsets: Pass,
}

/// One compute pipeline with its bind group layout.
type Pass = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

/// The key transform's two passes: in place, and from the base.
struct Keys {
    in_place: Pass,
    from_base: Pass,
}

/// A scatter's combine as pipelines, by its plan.
enum Scatter {
    /// A float sum: sort keys, the sort's step, and the fold.
    Sorted { keys: Pass, step: Pass, fold: Pass },
    /// An order-free rule: the parallel apply; the gather after `last`;
    /// the key transform around `min` and `max` over floats and longs:
    /// in place after, and before from the base into the target, which
    /// spares the target a copy of the base.
    Parallel {
        apply: Pass,
        gather: Option<Pass>,
        keys: Option<Keys>,
    },
}

impl Prepared for Pipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GpuBackend {
    /// Why this device does not run `module`, if it does not: more
    /// buffers than it binds at once, a `Double` it has no arithmetic for,
    /// or a `Long` where the device has none or the mode folds words it
    /// would have to fold in pairs.
    fn refusal(&self, module: &Validated) -> Option<String> {
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
    fn packing(&self, module: &Validated) -> Vec<usize> {
        let limits = self.device.limits();
        wgsl::packed_inputs(
            module,
            limits.max_buffers_and_acceleration_structures_per_shader_stage as usize,
            limits.max_storage_buffers_per_shader_stage as usize,
        )
    }

    /// Why `module`'s widest pass binds more buffers than this device
    /// does at once, if it does: its uniforms (the context and the
    /// parameters) and its storage (every input, every output it writes,
    /// the failure record where its arithmetic can fail, and an emitting
    /// scatter's target), against the device's limits on storage buffers,
    /// on uniform buffers, and on all of them together. Such a module runs
    /// on the ordered backend; nothing about it is wrong.
    fn wider_than_device(&self, module: &Validated) -> Option<String> {
        let (uniforms, unpacked) = wgsl::bindings(module);
        let inputs = module.manifest.inputs.len();
        let written = wgsl::written_outputs(module);
        let failure = usize::from(wgsl::can_fail(module));
        let emitting = usize::from(!wgsl::emitted_pairs(module).is_empty());
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

    /// Opens the default adapter.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| GpuError(error.to_string()))?;
        let adapter_limits = adapter.limits();
        // What the device will bind at once is what a Kernel's ports
        // come to: one buffer for the output and one for each input, so
        // the default of eight refused a Kernel of eight inputs that the
        // interpreter and the JIT run. The adapter offers what it offers
        // — twenty-nine on this one — and there is no reason to ask for
        // less.
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
            // And every buffer together: the default is less than the
            // storage buffers alone that the adapter offers.
            max_uniform_buffers_per_shader_stage: adapter_limits
                .max_uniform_buffers_per_shader_stage,
            max_buffers_and_acceleration_structures_per_shader_stage: adapter_limits
                .max_buffers_and_acceleration_structures_per_shader_stage,
            ..wgpu::Limits::default()
        };
        // Timestamps at a pass's edges, when the adapter has them: what
        // a profile reads. Writing one inside an encoder is a feature
        // Metal does not offer, so every pass carries its own.
        let timing = wgpu::Features::TIMESTAMP_QUERY;
        let timestamps = adapter.features().contains(timing);
        // 64-bit integers, when the adapter computes them: a `Long` then
        // runs here; without them, and for a `Double` always, the module
        // goes to the ordered backend.
        let int64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
        let int64_atomics = int64
            && adapter
                .features()
                .contains(wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX);
        let mut features = wgpu::Features::empty();
        if timestamps {
            features |= timing;
        }
        if int64 {
            features |= wgpu::Features::SHADER_INT64;
        }
        if int64_atomics {
            features |= wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orla"),
            required_features: features,
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
            int64,
            int64_atomics,
            uploaded: std::sync::atomic::AtomicU64::new(0),
            failures: Arc::new(Mutex::new(Vec::new())),
            last_nanos: std::sync::atomic::AtomicU64::new(u64::MAX),
            clock: std::sync::atomic::AtomicU8::new(clock::UNTRIED),
        })
    }

    /// Whether executions wait for the device; see
    /// [`Backend::set_measured`].
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

    /// The query set a measured call writes its passes' edges into, or
    /// `None` when nothing is being measured or the adapter has no
    /// timestamps. Made before the encoder so the opening timestamp is
    /// its first command.
    fn timing_set(&self, count: u32) -> Option<wgpu::QuerySet> {
        let worth_asking = self.clock.load(std::sync::atomic::Ordering::Relaxed) != clock::SILENT;
        (self.measured() && self.timestamps && worth_asking).then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("orla timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            })
        })
    }

    /// Copies a call's timestamps out where the Host can read them,
    /// added to the same encoder so they cost no submission of their own.
    fn resolve_timing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        set: Option<&wgpu::QuerySet>,
        count: u32,
    ) -> Option<wgpu::Buffer> {
        set.map(|set| {
            let bytes = u64::from(count) * 8;
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(set, 0..count, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, bytes);
            read
        })
    }

    /// Waits for a measured call and records what the device's own clock
    /// said about it, from the first timestamp to the last. Does nothing
    /// when the run is not being measured: a submission a later node
    /// reads needs no wait, since one queue runs in order.
    fn settle_measured(&self, submission: wgpu::SubmissionIndex, timing: Option<wgpu::Buffer>) {
        if !self.measured() {
            let _ = self.device.poll(wgpu::PollType::Poll);
            return;
        }
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        let Some(read) = timing else { return };
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
            let ticks =
                |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
            let (begin, end) = (ticks(0), ticks(bytes.len() - 8));
            let period = self.queue.get_timestamp_period() as f64;
            drop(bytes);
            read.unmap();
            // Some drivers take the queries and write nothing (Metal
            // through wgpu, at the time of writing): then there is no
            // device time and the wait is the answer.
            let (answer, kind) = if end > begin {
                (((end - begin) as f64 * period) as u64, clock::ANSWERED)
            } else {
                (u64::MAX, clock::SILENT)
            };
            if answer != u64::MAX {
                self.last_nanos
                    .store(answer, std::sync::atomic::Ordering::Relaxed);
            }
            self.clock.store(kind, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// What a measured execution's wait costs when there is nothing to
    /// wait for: one empty submission polled to completion, the least of
    /// sixteen. Every node a profile times carries exactly this before
    /// it does any work of its own, so a profile that does not take it
    /// off reports a tenth of a millisecond for a node that writes four
    /// kibibytes.
    pub fn wait_floor(&self) -> std::time::Duration {
        let mut least = std::time::Duration::MAX;
        for _ in 0..16 {
            let encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("orla floor"),
                });
            let started = std::time::Instant::now();
            let submission = self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            });
            least = least.min(started.elapsed());
        }
        least
    }

    /// Whether the device's own clock has answered a measured
    /// execution. The adapter's feature bit is not the answer: Metal
    /// through wgpu takes the queries and resolves them to zeros, so
    /// this is set the first time a resolve comes back with two
    /// different numbers in it. Until then a measured run's numbers are
    /// the call and its wait, not the work, and whatever prints them
    /// has to say so.
    pub fn device_clock_answered(&self) -> bool {
        self.clock.load(std::sync::atomic::Ordering::Relaxed) == clock::ANSWERED
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
            return Err(format!(
                "{}\n--- emitted WGSL ---\n{source}",
                device_error(&error)
            ));
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
            .storage(handle)?
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>();
        self.uploaded
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
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

    /// A failure record and its way back to the Host, from the pile or
    /// new: the first failure's word, the blocks of work the run has
    /// taken, and the most it may take ([`wgsl::FAILURE_BYTES`]).
    fn failure_pair(&self) -> (wgpu::Buffer, wgpu::Buffer) {
        if let Some(pair) = self
            .failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
        {
            return pair;
        }
        let word = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("failure"),
            size: wgsl::FAILURE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let back = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("failure back"),
            size: wgsl::FAILURE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (word, back)
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
    // Each parameter's words from a `vec4<u32>` slot of its own, as many
    // slots as they fill (`wgsl::param_slots`).
    let mut bytes = Vec::with_capacity(params.len().max(1) * 16);
    for value in params {
        let mut words = value.words();
        words.resize(words.len().div_ceil(4).max(1) * 4, 0);
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

    fn measurement_floor(&self) -> std::time::Duration {
        // A number the device's own clock gave carries nothing of the
        // wait around it; a number the wait gave carries all of it.
        if self.device_clock_answered() {
            std::time::Duration::ZERO
        } else {
            self.wait_floor()
        }
    }

    fn set_measured(&self, on: bool) {
        self.measured
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    fn measures_the_device(&self) -> bool {
        self.device_clock_answered()
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

    fn declines(&self, module: &Validated) -> Option<String> {
        self.refusal(module)
    }

    fn uploaded(&self) -> u64 {
        self.uploaded.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        if let Some(reason) = self.refusal(module) {
            return Err(format!(
                "`{}`: {reason}; a session with an ordered backend runs it there",
                module.manifest.id
            ));
        }
        // A port keeps every scalar one way, and only a map or a scan
        // writes an output narrow: the other modes fold into their
        // output in place, at full width.
        let manifest = &module.manifest;
        for port in manifest.inputs.iter().chain(&manifest.outputs) {
            if port.ty.schema.uniform_encoding().is_none() && !port.ty.schema.is_wide() {
                return Err(format!(
                    "`{}` keeps its scalars in more than one encoding ({}); a port keeps them one way",
                    port.name, port.ty.schema
                ));
            }
        }
        if !matches!(manifest.mode, orla_ir::Mode::Map | orla_ir::Mode::Scan)
            && manifest
                .outputs
                .iter()
                .any(|port| !port.ty.schema.is_wide())
        {
            return Err(format!(
                "`{}` is a {} and writes a narrow output; it folds at full width",
                manifest.id,
                manifest.mode.name()
            ));
        }
        // A Kernel's ports are buffers the device binds at once. Saying so
        // here is the difference between a Kernel that is too wide for
        // this device and a page of the driver's validation; a session
        // asks `declines` first and runs such a Kernel elsewhere.
        if let Some(why) = self.refusal(module) {
            return Err(format!(
                "`{}`: {why}; the interpreter and the JIT have no such limit",
                module.manifest.id
            ));
        }
        let written = wgsl::written_outputs(module);
        let packed = self.packing(module);
        let source = wgsl::emit_packed(module, &packed);
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
            return Err(format!(
                "{}\n--- emitted WGSL ---\n{source}",
                device_error(&error)
            ));
        }
        let reduce = if module.manifest.mode == orla_ir::Mode::Reduce {
            let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
            if let Some(error) = pollster::block_on(scope_guard.pop()) {
                return Err(format!(
                    "{}\n--- emitted WGSL ---\n{reduce_source}",
                    device_error(&error)
                ));
            }
            Some((pipeline, layout))
        } else {
            None
        };
        let prefix = if matches!(module.manifest.mode, orla_ir::Mode::Prefix { .. }) {
            let entries = [
                uniform_entry(wgsl::PREFIX_PLAN_BINDING),
                storage_entry(wgsl::PREFIX_OUT_BINDING, false),
                storage_entry(wgsl::PREFIX_SUMS_BINDING, false),
            ];
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
                        .then(|| -> Result<Keys, String> {
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

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        // A caller that asks for the outputs alone is answered as the
        // reference would: the run's failure, waited for, is its answer.
        let (outputs, check) = self.execute_checked(prepared, module, invocation, scope)?;
        if let Some(check) = check
            && let Err(error) = check.wait()
        {
            for handle in outputs {
                let _ = scope.release(handle);
            }
            return Err(error);
        }
        Ok(outputs)
    }

    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<orla_rules::backend::Checked, ExecutionError> {
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
        let timing_set = self.timing_set(2);
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
            bound_inputs,
            memory_outputs,
            output_buffer,
            further_outputs,
            output_words,
            element_count,
            failure,
            target,
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
                &bound_inputs,
                output_buffer,
                element_count,
                output_words,
                timing_set.as_ref(),
                failure.as_ref().map(|(word, _)| word),
            ),
            (Mode::Prefix { .. }, _, _) => {
                let prefix = pipeline
                    .prefix
                    .as_ref()
                    .expect("a prefix prepares its passes");
                self.run_prefix(
                    &mut encoder,
                    prefix,
                    output_buffer,
                    element_count,
                    output_words,
                    timing_set.as_ref(),
                )
            }
            (Mode::Scatter { base, .. }, _, Some(scatter)) => self.combine(
                &mut encoder,
                scatter,
                target,
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
        // The failure word, copied where it can be read once the device
        // is waited for: at the caller's commit, not here.
        if let Some((word, back)) = &failure {
            encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
        }
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let timing = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
        let submission = self.queue.submit(Some(encoder.finish()));
        // Submissions on one queue run in order, so a later node reading
        // this output needs no wait here; only observation does. Under
        // `set_measured`, this call waits, so what times it is the work.
        self.settle_measured(submission, timing);
        if let Some(error) = pollster::block_on(scope_guard.pop()) {
            return Err(ExecutionError::Device(device_error(&error)));
        }
        let outputs = adopt_outputs(
            scope,
            module,
            output_domain,
            final_buffer,
            further_outputs,
            memory_outputs,
            Some(self.spare.clone()),
        )?;
        let check = failure.map(|pair| {
            Box::new(FailureCheck {
                device: self.device.clone(),
                pair: Some(pair),
                pile: Arc::clone(&self.failures),
                width: domain.width(),
                budget: module.manifest.budget,
                run: orla_rules::Run::new(scope.work()),
            }) as Box<dyn Check>
        });
        Ok((outputs, check))
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
        // A pack is copied before each run, and the times would read the
        // chain through it: they go one by one instead.
        if prepared
            .as_any()
            .downcast_ref::<Pipeline>()
            .is_some_and(|pipeline| !pipeline.packed.is_empty())
        {
            return None;
        }
        // Only the shape the planner makes: a map of one output whose
        // own output comes back on one input. A measured run times each
        // node, so it goes the ordinary way.
        if repeat < 2
            || output != 0
            || module.manifest.mode != Mode::Map
            || module.manifest.outputs.len() != 1
            || !module.manifest.memories.is_empty()
            || self.measured()
            // A run that can fail is checked a time at a time, as the
            // ordinary way does.
            || wgsl::can_fail(module)
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
    /// What the passes bind for the inputs, by binding: the inputs' own
    /// buffers, and the pack where some are packed.
    bound_inputs: Vec<(u32, wgpu::Buffer)>,
    /// A memory's copy, written by the steps, to adopt as that output.
    memory_outputs: Vec<(usize, wgpu::Buffer, orla_abi::Domain)>,
    output_buffer: wgpu::Buffer,
    /// A map's outputs after the first, in the manifest's order.
    further_outputs: Vec<wgpu::Buffer>,
    output_words: usize,
    element_count: usize,
    /// The word the passes record their first failure in, when the
    /// module can fail, and the buffer it is read back through.
    failure: Option<(wgpu::Buffer, wgpu::Buffer)>,
    /// A scatter's target, ready before the entry when the entry emits.
    target: Option<wgpu::Buffer>,
}

/// The domain the entry maps over: a reduction's and a scatter's first
/// input, a scan's rows (checked), else the invocation's.
fn execution_domain(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<orla_abi::Domain, ExecutionError> {
    Ok(match module.manifest.mode {
        Mode::Reduce | Mode::Prefix { .. } => scope.size(invocation.inputs[0])?,
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
    further_outputs: Vec<wgpu::Buffer>,
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
    // A map's further outputs over the same domain, then the memories,
    // which is the manifest's order.
    let further = module
        .manifest
        .outputs
        .iter()
        .skip(1)
        .zip(further_outputs)
        .map(|(port, buffer)| (port.ty.schema.clone(), output_domain, buffer));
    let memories = memory_outputs.into_iter().map(|(index, buffer, extent)| {
        (
            module.manifest.inputs[index].ty.schema.clone(),
            extent,
            buffer,
        )
    });
    for (schema, extent, buffer) in further.chain(memories) {
        let adopted = scope.adopt_resident(
            schema,
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
            further_outputs: _,
            output_words: _,
            element_count,
            failure: _,
            target: _,
            bound_inputs: _,
        } = staged;
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
            // A time that stores by atomic or starts from a cleared buffer.
            if wgsl::output_shares_words(module, 0) {
                encoder.clear_buffer(&written, 0, None);
            }
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
            return Err(ExecutionError::Device(device_error(&error)));
        }
        // What the last time wrote is what the run gives.
        adopt_outputs(
            scope,
            module,
            domain,
            reads,
            Vec::new(),
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
                let size = (port.ty.schema.storage_words(extent.count()) * 4) as u64;
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
        // What the pass binds for its inputs: each its own, but for those
        // it packs, which are copied into one buffer before it runs, each
        // at the offset in words the parameters carry after their own.
        let mut bound_inputs: Vec<(u32, wgpu::Buffer)> = Vec::new();
        let mut packs: Vec<(wgpu::Buffer, u64, u64)> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut packed_words = 0u64;
        for (index, buffer) in input_buffers.iter().enumerate() {
            if pipeline.packed.contains(&index) {
                let extent = scope.size(invocation.inputs[index])?;
                let words = module.manifest.inputs[index]
                    .ty
                    .schema
                    .storage_words(extent.count()) as u64;
                let offset = u32::try_from(packed_words).map_err(|_| {
                    ExecutionError::Device(
                        "the inputs packed into one buffer pass 2^32 words".to_owned(),
                    )
                })?;
                offsets.push(offset);
                packs.push((buffer.clone(), packed_words * 4, words * 4));
                packed_words += words;
            } else {
                bound_inputs.push((wgsl::INPUT_BINDING + index as u32, buffer.clone()));
            }
        }
        let pack = (!packs.is_empty()).then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed inputs"),
                size: (packed_words * 4).max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        if let Some(pack) = &pack {
            bound_inputs.push((wgsl::PACKED_BINDING, pack.clone()));
        }
        let context_bytes: Vec<u8> = context.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut params_bytes = param_bytes(&invocation.params);
        if !offsets.is_empty() {
            // The offsets start at the first slot after the parameters'.
            params_bytes.truncate(wgsl::param_slots(module).1 * 16);
            for offset in &offsets {
                params_bytes.extend_from_slice(&offset.to_le_bytes());
            }
            params_bytes.resize(params_bytes.len().div_ceil(16) * 16, 0);
        }
        let (context_buffer, params_buffer) = self.uniform_buffers(&context_bytes, &params_bytes);
        let output_schema = &module.manifest.outputs[0].ty.schema;
        let output_words = output_schema.value_words();
        // A scatter's map pass stores a pair per source element; the
        // rest store elements as the output keeps them.
        let map_bytes = if pipeline.emits {
            // A scatter that emits lands every pair from its entry, the
            // one it returns too: its pairs buffer is never written.
            16
        } else if matches!(module.manifest.mode, Mode::Scatter { .. }) {
            element_count * wgsl::type_words(module, wgsl::result_type(&module.entry)) * 4
        } else {
            output_schema.storage_words(element_count) * 4
        };
        let output_buffer = self.spare.take(
            &self.device,
            map_bytes as u64,
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
        for (binding, buffer) in &bound_inputs {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            });
        }
        // The failure word and its way back, from the pile; cleared below,
        // before the pass, which is no failure.
        let failure = pipeline.fails.then(|| self.failure_pair());
        if let Some((failure, _)) = &failure {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::FAILURE_BINDING,
                resource: failure.as_entire_binding(),
            });
        }
        // A map's further outputs, each a buffer of its own words.
        let further_outputs: Vec<wgpu::Buffer> = module
            .manifest
            .outputs
            .iter()
            .take(wgsl::written_outputs(module))
            .skip(1)
            .map(|port| {
                self.spare.take(
                    &self.device,
                    (port.ty.schema.storage_words(element_count) * 4) as u64,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
            })
            .collect();
        for (k, buffer) in further_outputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::output_binding(module, k + 1),
                resource: buffer.as_entire_binding(),
            });
        }
        // A scatter that emits combines into its target as the entry
        // runs, so the target is made, and bound, here.
        let target = match module.manifest.mode {
            Mode::Scatter { base, .. } if pipeline.emits => {
                let count = scope.size(invocation.inputs[base as usize])?.count();
                Some((base, count, self.target_buffer(count, output_words)))
            }
            _ => None,
        };
        if let Some((_, _, buffer)) = &target {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::EMIT_BINDING,
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
        if let Some(pack) = &pack {
            for (from, at, size) in &packs {
                if *size > 0 {
                    encoder.copy_buffer_to_buffer(from, 0, pack, *at, *size);
                }
            }
        }
        if let (Some((base, count, buffer)), Some(scatter)) = (&target, &pipeline.scatter) {
            self.prepare_target(
                &mut encoder,
                scatter,
                buffer,
                &input_buffers[*base as usize],
                &context_buffer,
                *count,
                output_words,
            );
        }
        // No failure yet, no work taken, and the most the run may take;
        // written before this submission's passes run.
        if let Some((failure, _)) = &failure {
            let limit = orla_rules::Run::new(scope.work()).limit();
            let record: Vec<u8> = [0u32, 0, limit, 0]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            self.queue.write_buffer(failure, 0, &record);
        }
        // A narrow output whose elements share words is stored by atomic
        // or, which needs every word it lands in to start at zero.
        for (k, buffer) in std::iter::once(&output_buffer)
            .chain(&further_outputs)
            .enumerate()
        {
            if k < wgsl::written_outputs(module) && wgsl::output_shares_words(module, k) {
                encoder.clear_buffer(buffer, 0, None);
            }
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
            bound_inputs,
            memory_outputs,
            output_buffer,
            further_outputs,
            output_words,
            element_count,
            failure,
            target: target.map(|(_, _, buffer)| buffer),
        })
    }

    /// A prefix fold on the device: three passes over the map's output,
    /// in place. One invocation walks each block of `elements a block`
    /// and leaves its total; one invocation scans the blocks' totals;
    /// one invocation an element takes its block's offset. The rule is
    /// order-free, so this grouping answers with the reference's words,
    /// and nothing here needs shared memory or a barrier — a block is a
    /// walk, and the passes are ordered by being passes.
    ///
    /// A block holds about the square root of the count, which is what
    /// makes the two walks the same length: the blocks' own walk is as
    /// long as one block.
    fn run_prefix(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prefix: &Prefix,
        output: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&wgpu::QuerySet>,
    ) -> wgpu::Buffer {
        let count = element_count.max(1);
        let per_block = orla_rules::prefix::block(count);
        let blocks = count.div_ceil(per_block);
        let plan: [u32; 4] = [count as u32, per_block as u32, blocks as u32, 0];
        let plan_bytes: Vec<u8> = plan.iter().flat_map(|v| v.to_le_bytes()).collect();
        let plan_buffer = self.buffer("prefix plan", &plan_bytes, wgpu::BufferUsages::UNIFORM);
        let sums = self.spare.take(
            &self.device,
            ((blocks * output_words * 4) as u64).max(16).div_ceil(16) * 16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let passes = [
            (&prefix.blocks, blocks as u32),
            (&prefix.tops, 1),
            (&prefix.offsets, count as u32),
        ];
        for (index, ((pipeline, layout), threads)) in passes.iter().enumerate() {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("orla prefix"),
                layout,
                entries: &[
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
                ],
            });
            let last = index + 1 == passes.len();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("orla prefix"),
                timestamp_writes: timing.and_then(|query_set| {
                    last.then_some(wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(1),
                    })
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let (x, y) = wgsl::folded(threads.div_ceil(wgsl::PREFIX_WORKGROUP));
            pass.dispatch_workgroups(x, y, 1);
        }
        self.spare.keep(sums);
        output
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
        bound_inputs: &[(u32, wgpu::Buffer)],
        mapped: wgpu::Buffer,
        element_count: usize,
        output_words: usize,
        timing: Option<&wgpu::QuerySet>,
        failure: Option<&wgpu::Buffer>,
    ) -> wgpu::Buffer {
        let (reduce_pipeline, reduce_layout) = reduce;
        let params_buffer_for_reduce = params_buffer;
        let final_buffer = mapped;
        let stride = output_words as u64 * 4;
        let mut count = element_count as u64;
        // The level between the levels, from the pile: a scene reduces
        // the same count every frame, and asking the driver for it
        // again is what the pile is for. Every element a level reads
        // was written by the level before it, so nothing stale is ever
        // seen.
        let other = self.spare.take(
            &self.device,
            (orla_rules::reduce::after_fold(count as usize) as u64 * stride).max(16),
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
            let ahead = orla_rules::reduce::after_fold(count as usize) as u32;
            let (x, y) = wgsl::folded(ahead.div_ceil(wgsl::REDUCE_WORKGROUP));
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            count = u64::from(ahead);
            std::mem::swap(&mut current, &mut next);
        }
        // One of the two is the answer and goes on to be the Field; the
        // other is this call's own and goes back. What read it was
        // submitted before whatever writes it next, and one queue runs
        // its submissions in order.
        self.spare.keep(next);
        current
    }

    /// A scatter's target, `count` elements of `words` words. It is
    /// copied over whole before anything reads it, so it may come from
    /// the pile.
    fn target_buffer(&self, count: usize, words: usize) -> wgpu::Buffer {
        let size = (count * words * 4) as u64;
        self.spare.take(
            &self.device,
            size.max(16).div_ceil(16) * 16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// The target made ready for pairs: the base copied into it, or,
    /// where the plan compares its words with atomics, the base's words
    /// turned into ordered keys as they are written into it.
    #[allow(clippy::too_many_arguments)]
    fn prepare_target(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scatter: &Scatter,
        target: &wgpu::Buffer,
        base_buffer: &wgpu::Buffer,
        context_buffer: &wgpu::Buffer,
        count: usize,
        words: usize,
    ) {
        let Scatter::Parallel {
            keys: Some(keys), ..
        } = scatter
        else {
            encoder.copy_buffer_to_buffer(base_buffer, 0, target, 0, (count * words * 4) as u64);
            return;
        };
        let keys = &keys.from_base;
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("orla scatter"),
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
            label: Some("orla scatter keys"),
            timestamp_writes: None,
        });
        compute.set_pipeline(&keys.0);
        compute.set_bind_group(0, &group, &[]);
        let (x, y) = wgsl::folded((count as u32).div_ceil(wgsl::SCATTER_WORKGROUP));
        compute.dispatch_workgroups(x, y, 1);
    }

    /// A scatter's combine: the target made ready (unless the entry,
    /// emitting, had it ready already), then the pairs applied by the
    /// plan. A float sum sorts the pairs by target
    /// and source (a bitonic sort, one small dispatch per step) and folds
    /// each target's run in source order; the order-free rules apply one
    /// invocation per pair with atomics, `min` and `max` over floats
    /// between two key passes, `last` followed by a gather. Returns the
    /// target.
    #[allow(clippy::too_many_arguments)]
    fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scatter: &Scatter,
        target: Option<wgpu::Buffer>,
        base_buffer: &wgpu::Buffer,
        context_buffer: &wgpu::Buffer,
        pairs: wgpu::Buffer,
        target_count: usize,
        element_count: usize,
        output_words: usize,
        timing: Option<&wgpu::QuerySet>,
    ) -> wgpu::Buffer {
        let final_buffer = pairs;
        // Ready already when the entry emitted into it, and then the entry
        // landed every pair itself: nothing is left to apply.
        let landed = target.is_some();
        let target = target.unwrap_or_else(|| {
            let target = self.target_buffer(target_count, output_words);
            self.prepare_target(
                encoder,
                scatter,
                &target,
                base_buffer,
                context_buffer,
                target_count,
                output_words,
            );
            target
        });
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
            let (x, y) = wgsl::folded(groups);
            compute.dispatch_workgroups(x, y, 1);
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
                if !landed {
                    let group = bind(&apply.1, &with_winners);
                    dispatch(encoder, apply, &group, &[], over_sources);
                }
                if let Some(gather) = gather {
                    let group = bind(&gather.1, &with_winners);
                    dispatch(encoder, gather, &group, &[], over_target);
                }
                if let Some(keys) = keys {
                    let group = bind(&keys.in_place.1, &common);
                    dispatch(encoder, &keys.in_place, &group, &[], over_target);
                }
            }
        }
        target
    }
}

/// A device run's failure word, to be read once the device is waited
/// for: the first failing element, in row-major order, and why.
struct FailureCheck {
    device: wgpu::Device,
    /// The failure word and the buffer it was copied to, which go back
    /// to the pile once read.
    pair: Option<(wgpu::Buffer, wgpu::Buffer)>,
    pile: Arc<Mutex<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
    /// The width of the domain the run mapped over, which turns the
    /// element's number into the reference's column and row.
    width: u32,
    /// The budget the run had, for naming a failure past it.
    budget: u32,
    /// The run's total, for naming a run past it.
    run: orla_rules::Run,
}

impl Check for FailureCheck {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let (word_buffer, back) = self.pair.take().expect("a check is waited on once");
        let slice = back.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| ExecutionError::Device(error.to_string()))?;
        let (word, blocks, left) = {
            let bytes = slice
                .get_mapped_range()
                .map_err(|error| ExecutionError::Device(error.to_string()))?;
            let at =
                |k: usize| u32::from_le_bytes([bytes[k], bytes[k + 1], bytes[k + 2], bytes[k + 3]]);
            (at(0), at(4), at(12))
        };
        back.unmap();
        self.pile
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((word_buffer, back));
        // An element's own failure names that element; past the run's
        // total and with no element failing, the run is what failed.
        if word == 0 {
            return orla_rules::Run::judge(self.run.work(), u64::from(blocks), u64::from(left));
        }
        let word = u32::MAX - word;
        let element = word / 4;
        let width = self.width.max(1);
        let (x, y) = (element % width, element / width);
        Err(match word % 4 {
            wgsl::FAILED_BUDGET => ExecutionError::Exhausted {
                x,
                y,
                budget: self.budget,
            },
            wgsl::FAILED_DIVISION => ExecutionError::Arithmetic {
                x,
                y,
                error: ArithmeticError::DivisionByZero,
            },
            _ => ExecutionError::Arithmetic {
                x,
                y,
                error: ArithmeticError::Overflow,
            },
        })
    }
}

/// What the device said, in full. A validation error's `to_string` is
/// "Validation Error" and nothing else; what failed, and against which
/// limit, is in its description.
pub(crate) fn device_error(error: &wgpu::Error) -> String {
    match error {
        wgpu::Error::Validation { description, .. } | wgpu::Error::Internal { description, .. } => {
            description.trim().to_owned()
        }
        other => other.to_string(),
    }
}
