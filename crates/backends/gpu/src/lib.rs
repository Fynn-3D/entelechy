#![deny(unsafe_code)]
#![warn(missing_docs)]
//! GPU backend.
//!
//! Emits WGSL (or `#version 300 es` GLSL for a WebGL2 page) from validated IR
//! at preparation and runs each execution on the device. The Host owns images
//! and their pixels stay resident on the device: a resident input binds
//! directly, a Host input uploads once and stays resident, and outputs are
//! adopted as resident images. Pixels return only through [`Backend::download`].
//! Semantics match the CPU backend up to floating-point rounding; integer
//! failures are reported as the CPU reports them.

mod batch;
mod failure;
pub mod glsl;
mod later;
mod modes;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod page;
mod pool;
mod prepare;
pub mod present;
mod raster;
mod raster_hiz;
mod same;
mod stage;
mod texture;
mod timing;
mod trace;
mod trace_lbvh;
mod wait;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod webgl;
pub mod wgsl;

pub use wait::Wait;
use wait::{Spent, Waiter};

/// Whether Entelechy runs in a page: its device is the browser's WebGPU,
/// answered only when the page gets a turn, and its frames are timed by
/// the device's stamps. Code that only a page has is gated by the same
/// `cfg`; how a wait is made is the [`Wait`] the host hands in.
pub(crate) const PAGE: bool = cfg!(all(target_arch = "wasm32", target_os = "unknown"));

use failure::*;
use later::*;
use pool::*;
use prepare::*;
use stage::*;
use timing::*;

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use entelechy_abi::{Handle, Invocation, MadeFrom, Scope, Value, Version};
use entelechy_ir::{Mode, Validated};
use entelechy_rules::{ArithmeticError, Backend, Check, NativeOp, Prepared};
use entelechy_rules::{ExecutionError, check_invocation};

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
pub(crate) struct WgpuBackend {
    /// How to wait for the device: block, or give the page turns.
    wait: Wait,
    /// The time callers have spent blocked on the device's work.
    spent: Spent,
    /// Looks at the device for as long as this lives ([`Waiter`]).
    _waiter: Option<Waiter>,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    /// Serializes executions, so readback is ordered.
    lock: Mutex<()>,
    /// The modules this device's own compiler could not build, and what it
    /// said: declined from then on ([`Backend::declines`]), for another
    /// backend to run.
    unbuilt: Arc<Mutex<Vec<(Validated, String)>>>,
    /// The rasterizer's pipelines, built on first use.
    raster: OnceLock<Result<raster::RasterPipelines, String>>,
    /// Hardware ray-query pipelines (`enable wgpu_ray_query;`), built on first use.
    trace_hw: OnceLock<Result<trace::HwTraceSlot, String>>,
    /// Portable GPU Compute BVH ray-tracing pipelines, built on first use.
    trace_bvh: OnceLock<Result<trace::BvhTracePipelines, String>>,
    /// Bounded cache of hardware BLAS/TLAS acceleration structures and GPU BVHs.
    trace_cache: trace::TraceCache,
    /// The pass that compares two Fields, built on first use.
    same: OnceLock<Result<same::SamePipeline, String>>,
    /// The pass that pads unaligned rows for `copy_buffer_to_texture`, built on first use.
    row_pad: OnceLock<Result<texture::RowPadPipeline, String>>,
    /// The pass that unpacks a hardware BCn texture into a storage buffer when needed.
    decompress_bc: OnceLock<Result<texture::DecompressBcPipelines, String>>,
    /// The passes that generate 2D/2D-array/3D mipmap chains on the GPU.
    mip_gen: OnceLock<Result<texture::MipGenPipelines, String>>,
    /// Persistent hardware filtering samplers (clamp/repeat x trilinear/16x aniso).
    sampler_clamp: wgpu::Sampler,
    sampler_repeat: wgpu::Sampler,
    sampler_aniso_clamp: wgpu::Sampler,
    sampler_aniso_repeat: wgpu::Sampler,
    /// Buffers released Fields left behind; see [`Spare`].
    spare: Spare,
    /// The rasterizer's own targets, kept by shape; see [`SpareTextures`].
    targets: SpareTextures,
    /// Recycled 2D/3D hardware texture mirrors; see [`texture::SpareMirrors`].
    mirrors: texture::SpareMirrors,
    /// The context and parameter buffers of the execution being staged,
    /// written again per node: one execution runs at a time, and a queue's
    /// writes are ordered with its submissions.
    uniforms: Mutex<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// What those buffers were last given, so a node staged with the same
    /// context and parameters writes nothing.
    written: Mutex<(Vec<u8>, Vec<u8>)>,
    /// Reusable byte and word buffers for staging context, parameters, and
    /// Host uploads under [`GpuBackend::lock`].
    staging: Mutex<StageScratch>,
    /// Whether an execution waits for the device before it returns, so a
    /// timing measures the work. Off, only an observation waits.
    measured: std::sync::atomic::AtomicBool,
    /// Whether the adapter offers timestamp queries at a pass's edges; see
    /// [`GpuBackend::device_clock_answered`] for whether it writes them.
    timestamps: bool,
    /// Whether the device computes 64-bit integers.
    int64: bool,
    /// Whether it takes 64-bit min and max atomically, as a scatter of `Long`s
    /// under `min` or `max` needs.
    int64_atomics: bool,
    /// Whether a fragment knows its primitive's index: the rasterizer then
    /// draws its triangles indexed, each vertex shaded once for all of
    /// them; without it, a vertex is pulled a triangle at a time.
    primitive_index: bool,
    /// Whether the adapter offers hardware BC1/BC4/BC5/BC7 block-compressed
    /// textures (`TEXTURE_COMPRESSION_BC`).
    texture_compression_bc: bool,
    /// Whether the adapter offers hardware linear/anisotropic filtering on
    /// 32-bit float textures (`FLOAT32_FILTERABLE`).
    float32_filterable: bool,
    /// Hardware subgroup (wave / SIMD-group) capabilities (`SUBGROUP`).
    subgroups: wgsl::SubgroupCaps,
    /// Whether hardware ray tracing (`EXPERIMENTAL_RAY_QUERY`) is enabled on this device.
    ray_query: bool,
    /// Failure words and their readback buffers, for reuse: a checked node
    /// takes a pair each run.
    failures: Arc<Mutex<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
    /// Bytes uploaded from Host memory since the backend was made.
    uploaded: std::sync::atomic::AtomicU64,
    /// What asking the device's own clock has been worth: `UNTRIED` until a
    /// measured run asks, then `ANSWERED` or `SILENT`, which asks no more.
    clock: std::sync::atomic::AtomicU8,
    /// The nanoseconds the last measured execution took on the device,
    /// or `u64::MAX` for none.
    last_nanos: std::sync::atomic::AtomicU64,
    /// Frames clocked as they run; see [`Backend::set_clocked`].
    frames: timing::Frames,
    /// Validated WGSL sources, bind group layouts, and live compute
    /// pipelines shared across passes of identical source and bindings.
    passes: Arc<prepare::PassCache>,
    /// What the device said by itself: an error no scope caught, its loss.
    /// A readback that fails says it too, as the cause is most often there.
    heard: Arc<Mutex<Vec<String>>>,
    /// Set once the device reports loss so pending warm/finish passes fail fast
    /// instead of queuing dead compilation promises onto a lost device.
    device_lost: Arc<std::sync::atomic::AtomicBool>,
    /// Deferred error-scope checks from native operations (e.g. `rasterize`) on
    /// a page, drained at the frame's end by [`Submitted::wait`].
    pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>>,
    /// Unsettled submissions and estimated operations queued on a page's device
    /// since its last wait, drained adaptively so slow laptop integrated GPUs
    /// stay safely under Windows WDDM's 2-second TDR watchdog without stalling
    /// fast discrete GPUs.
    page_queue: PageQueue,
    /// Smoothed nanoseconds per million effective IR operations observed across
    /// [`WgpuBackend::drain_page`] and frame-end waits, starting conservatively
    /// for laptop integrated GPUs (`crbug.com/369219127`) and converging down
    /// to true hardware throughput on fast discrete GPUs.
    ns_per_mop: Arc<std::sync::atomic::AtomicU64>,
    /// Optional per-slice element ceiling override for testing sliced map
    /// dispatches on native devices (`0` uses the default policy).
    test_slice_limit: usize,
    /// Same-frame command encoder & uniform/upload arena shared with
    /// [`present::WgpuDisplay`] and deferred [`FailureCheck`]s.
    batch: Arc<batch::BatchHub>,
}

/// Runs and target slice duration on a page's device before draining, keeping
/// Windows WDDM's 2-second TDR watchdog safe on slow laptop integrated GPUs
/// (`crbug.com/369219127`) while converging to zero mid-frame stalls on fast
/// discrete GPUs.
const PAGE_PACE_RUNS: u32 = 48;
const PAGE_TARGET_SLICE_NS: u64 = 150_000_000;
const PAGE_INITIAL_NS_PER_MOP: u64 = 2_000_000;
const PAGE_MIN_NS_PER_MOP: u64 = 250;
const PAGE_MAX_NS_PER_MOP: u64 = 20_000_000;
const PAGE_SLICE_MIN_VOLUME_ELEMENTS: usize = 1_024;
const PAGE_SLICE_MIN_GRID_ELEMENTS: usize = 8_192;

pub(crate) fn record_page_rate(
    ns_per_mop: &std::sync::atomic::AtomicU64,
    queued_ops: u64,
    elapsed_ns: u64,
) {
    if queued_ops < 20_000_000 {
        return;
    }
    let sample = (elapsed_ns.saturating_mul(1_000_000) / queued_ops)
        .clamp(PAGE_MIN_NS_PER_MOP, PAGE_MAX_NS_PER_MOP);
    let prev = ns_per_mop.load(std::sync::atomic::Ordering::Relaxed);
    let next = if sample > prev {
        sample
    } else {
        (prev / 4).max(sample)
    };
    ns_per_mop.store(next, std::sync::atomic::Ordering::Relaxed);
}

fn effective_page_unit_ops(module: &Validated, params: &[Value]) -> u64 {
    let typical = module.typical_work_given(params).max(1);
    if module.worst_work() >= 1_000_000_000 {
        typical.saturating_mul(6)
    } else {
        typical
    }
}

#[derive(Clone, Default)]
pub(crate) struct PageQueue(
    Arc<(
        std::sync::atomic::AtomicU32,
        std::sync::atomic::AtomicU64,
        Mutex<Option<entelechy_rules::clock::Moment>>,
    )>,
);

impl PageQueue {
    pub(crate) fn note(&self, ops: u64) {
        let prev_runs = self.0.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.0
            .1
            .fetch_add(ops, std::sync::atomic::Ordering::Relaxed);
        if prev_runs == 0 {
            let mut first = self
                .0
                .2
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if first.is_none() {
                *first = entelechy_rules::clock::now();
            }
        }
    }

    pub(crate) fn take_timing(&self) -> (u64, u64) {
        let ops = self.0.1.swap(0, std::sync::atomic::Ordering::Relaxed);
        self.0.0.store(0, std::sync::atomic::Ordering::Relaxed);
        let started = self
            .0
            .2
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let elapsed_ns = entelechy_rules::clock::since(started)
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        (ops, elapsed_ns)
    }

    pub(crate) fn clear(&self) {
        self.0.0.store(0, std::sync::atomic::Ordering::Relaxed);
        self.0.1.store(0, std::sync::atomic::Ordering::Relaxed);
        *self
            .0
            .2
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    pub(crate) fn full(&self, incoming_ops: u64, pace_ops_limit: u64) -> bool {
        let runs = self.0.0.load(std::sync::atomic::Ordering::Relaxed);
        let ops = self.0.1.load(std::sync::atomic::Ordering::Relaxed);
        runs >= PAGE_PACE_RUNS || (runs > 0 && ops.saturating_add(incoming_ops) > pace_ops_limit)
    }
}

/// The little-endian words a mapped readback holds, copied once.
pub(crate) fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `error` with what the device said by itself before it, when it said
/// anything: a page's mapping fails with a bare "error occurred".
pub(crate) fn told(heard: &Mutex<Vec<String>>, error: String) -> String {
    let heard = heard.lock().map(|h| h.join("; ")).unwrap_or_default();
    if heard.is_empty() {
        error
    } else {
        format!("{error} (the device said: {heard})")
    }
}

impl fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBackend({})", self.adapter_name)
    }
}

impl WgpuBackend {
    /// The bytes a workgroup of this device holds at once: what a group's
    /// exchanges may take.
    pub fn workgroup_memory(&self) -> u32 {
        self.device.limits().max_compute_workgroup_storage_size
    }

    /// Hardware subgroup (wave / SIMD-group) capabilities enabled on this device.
    pub fn subgroups(&self) -> wgsl::SubgroupCaps {
        self.subgroups
    }

    /// Disables subgroup codegen on this backend, exercising non-subgroup
    /// workgroup fallbacks on hardware that supports `SUBGROUP`.
    pub fn without_subgroups(mut self) -> Self {
        self.subgroups = wgsl::SubgroupCaps::default();
        self
    }

    /// Downloads `byte_len` bytes from a resident [`DeviceImage`] while the
    /// caller already holds `self.lock`.
    pub(crate) fn download_unlocked(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, ExecutionError> {
        let image = device.downcast_ref::<DeviceImage>().ok_or_else(|| {
            ExecutionError::Device("image is resident on a different backend".into())
        })?;
        let src_buffer = self.buffer_of(image)?;
        self.batch.flush()?;
        if byte_len == 0 {
            return Ok(Vec::new());
        }
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
                label: Some("entelechy download"),
            });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &staging, 0, size);
        self.submit(encoder.finish());
        self.spent
            .time(|| {
                self.wait
                    .map_read(&self.device, &self.queue, staging.slice(..))
            })
            .map_err(|error| ExecutionError::Device(told(&self.heard, error)))?;
        self.page_queue.clear();
        let words = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| ExecutionError::Device(error.to_string()))?;
            words_of(&view[..byte_len])
        };
        staging.unmap();
        crate::wait::discard(staging);
        Ok(words)
    }

    /// Whether hardware ray tracing (`EXPERIMENTAL_RAY_QUERY` BLAS/TLAS) is
    /// active on this device.
    pub fn ray_query(&self) -> bool {
        self.ray_query
    }

    /// Disables hardware ray queries on this backend, exercising the portable
    /// GPU Compute BVH fallback on hardware that supports `EXPERIMENTAL_RAY_QUERY`.
    pub fn without_ray_query(mut self) -> Self {
        self.ray_query = false;
        self
    }

    /// Gives the device a run, work or copy: every run but the empty ones
    /// a wait is marked by goes through here, so a clocked frame counts
    /// each.
    /// What was let go before it is destroyed after it ([`Spare::bury`]).
    pub(crate) fn submit(&self, commands: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        let submission = self.frames.submit(&self.queue, commands);
        self.spare.bury();
        self.targets.bury();
        self.mirrors.bury();
        submission
    }

    /// Opens the fastest adapter, waiting for the device as `wait` says.
    pub fn with_wait(wait: Wait) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = wait
            // The fast device where there are two: a machine with a discrete
            // card beside the processor's own is otherwise often given the
            // processor's, in a page above all.
            .block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..wgpu::RequestAdapterOptions::default()
            }))
            .map_err(|error| GpuError(error.to_string()))?;
        let adapter_limits = adapter.limits();
        // Every buffer binding the adapter offers: a Kernel binds one buffer
        // for the output and one per input, so this limit bounds the widest
        // Kernel the device runs.
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
            // And every buffer together: the default total is below the
            // adapter's storage-buffer limit.
            max_uniform_buffers_per_shader_stage: adapter_limits
                .max_uniform_buffers_per_shader_stage,
            max_buffers_and_acceleration_structures_per_shader_stage: adapter_limits
                .max_buffers_and_acceleration_structures_per_shader_stage,
            // A group's exchanges, in its workgroup's memory: all the
            // adapter offers.
            max_compute_workgroup_storage_size: adapter_limits.max_compute_workgroup_storage_size,
            // Hardware 2D/3D texture mirror and 2D array bounds: all the adapter offers.
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
            max_texture_dimension_3d: adapter_limits.max_texture_dimension_3d,
            max_texture_array_layers: adapter_limits.max_texture_array_layers,
            max_sampled_textures_per_shader_stage: adapter_limits
                .max_sampled_textures_per_shader_stage,
            ..wgpu::Limits::default()
        };
        // Timestamps at a pass's edges, when offered: what a profile reads.
        // Every pass carries its own; Metal writes none inside an encoder.
        let timing = wgpu::Features::TIMESTAMP_QUERY;
        let timestamps = adapter.features().contains(timing);
        // 64-bit integers, when offered: a `Long` then runs here. Otherwise,
        // and for every `Double`, the module goes to the ordered backend.
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
        let primitive_index = adapter.features().contains(wgpu::Features::PRIMITIVE_INDEX);
        if primitive_index {
            features |= wgpu::Features::PRIMITIVE_INDEX;
        }
        let texture_compression_bc = adapter
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        if texture_compression_bc {
            features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        let float32_filterable = adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        if float32_filterable {
            features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        if has_subgroups {
            features |= wgpu::Features::SUBGROUP;
        }
        let adapter_info = adapter.get_info();
        let subgroups = if has_subgroups {
            // wgpu-hal hardcodes a conservative 4..64 range for all Metal
            // adapters; Apple Silicon GPUs (Apple7+) always run fixed 32-lane
            // SIMD-groups, unlocking single-wave `group <= 32` shuffles and
            // zero-atomic 32-bit `subgroupBallot` compaction.
            let apple_metal = adapter_info.backend == wgpu::Backend::Metal
                && adapter_info.name.starts_with("Apple ");
            let (min_size, max_size) = if apple_metal {
                (32, 32)
            } else {
                (
                    adapter_info.subgroup_min_size,
                    adapter_info.subgroup_max_size,
                )
            };
            wgsl::SubgroupCaps {
                enabled: true,
                min_size,
                max_size,
                enable_directive: PAGE,
            }
        } else {
            wgsl::SubgroupCaps::default()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let ray_query = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        #[cfg(target_arch = "wasm32")]
        let ray_query = false;
        #[cfg(not(target_arch = "wasm32"))]
        let experimental_features = if ray_query {
            features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
            required_limits = required_limits.using_acceleration_structure_values(adapter_limits);
            // SAFETY: Single isolated opt-in for native hardware BLAS/TLAS ray
            // queries (`wgpu::Features::EXPERIMENTAL_RAY_QUERY`); wasm32 builds
            // never call this and run the safe GPU Compute BVH fallback.
            #[allow(unsafe_code)]
            unsafe {
                wgpu::ExperimentalFeatures::enabled()
            }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        #[cfg(target_arch = "wasm32")]
        let experimental_features = wgpu::ExperimentalFeatures::disabled();
        let (device, queue) = wait
            .block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("entelechy"),
                required_features: features,
                required_limits,
                experimental_features,
                ..Default::default()
            }))
            .map_err(|error| GpuError(error.to_string()))?;
        let adapter_name = adapter_info.name;
        let heard: Arc<Mutex<Vec<String>>> = Arc::default();
        let keep = |heard: &Arc<Mutex<Vec<String>>>, line: String| {
            if let Ok(mut heard) = heard.lock() {
                // The first few: the cause comes first, its echoes after.
                if heard.len() < 4 {
                    heard.push(line);
                }
            }
        };
        // In a page, an error no scope caught is kept to be told; elsewhere it
        // stops the program.
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            let heard = Arc::clone(&heard);
            device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                keep(&heard, error.to_string())
            }));
        }
        let device_lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let heard = Arc::clone(&heard);
            let lost = Arc::clone(&device_lost);
            device.set_device_lost_callback(move |reason, message| {
                lost.store(true, std::sync::atomic::Ordering::Relaxed);
                keep(
                    &heard,
                    format!("the device was lost ({reason:?}): {message}"),
                )
            });
        }
        let spent = Spent::default();
        let spare = Spare::default();
        let targets = SpareTextures::default();
        let mirrors = texture::SpareMirrors::default();
        let pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>> = Arc::default();
        let page_queue = PageQueue::default();
        let batch = Arc::new(batch::BatchHub::new(
            device.clone(),
            queue.clone(),
            wait,
            spent.clone(),
            spare.clone(),
            targets.clone(),
            mirrors.clone(),
            page_queue.clone(),
            Arc::clone(&pending_checks),
        ));
        let make_sampler = |label: &str, xy_mode: wgpu::AddressMode, anisotropy_clamp: u16| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: xy_mode,
                address_mode_v: xy_mode,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
                compare: None,
                anisotropy_clamp,
                border_color: None,
            })
        };
        let sampler_clamp = make_sampler("entelechy clamp", wgpu::AddressMode::ClampToEdge, 1);
        let sampler_repeat = make_sampler("entelechy repeat", wgpu::AddressMode::Repeat, 1);
        let sampler_aniso_clamp =
            make_sampler("entelechy aniso clamp", wgpu::AddressMode::ClampToEdge, 16);
        let sampler_aniso_repeat =
            make_sampler("entelechy aniso repeat", wgpu::AddressMode::Repeat, 16);
        Ok(Self {
            wait,
            spent,
            _waiter: Waiter::start(&device),
            instance,
            adapter,
            device,
            queue,
            adapter_name,
            lock: Mutex::new(()),
            uniforms: Mutex::new(None),
            written: Mutex::new((Vec::new(), Vec::new())),
            staging: Mutex::new(StageScratch::default()),
            raster: OnceLock::new(),
            trace_hw: OnceLock::new(),
            trace_bvh: OnceLock::new(),
            trace_cache: trace::TraceCache::default(),
            same: OnceLock::new(),
            row_pad: OnceLock::new(),
            decompress_bc: OnceLock::new(),
            mip_gen: OnceLock::new(),
            sampler_clamp,
            sampler_repeat,
            sampler_aniso_clamp,
            sampler_aniso_repeat,
            spare,
            targets,
            mirrors,
            measured: std::sync::atomic::AtomicBool::new(false),
            unbuilt: Arc::new(Mutex::new(Vec::new())),
            timestamps,
            int64,
            int64_atomics,
            primitive_index,
            texture_compression_bc,
            float32_filterable,
            subgroups,
            ray_query,
            uploaded: std::sync::atomic::AtomicU64::new(0),
            failures: Arc::new(Mutex::new(Vec::new())),
            last_nanos: std::sync::atomic::AtomicU64::new(u64::MAX),
            clock: std::sync::atomic::AtomicU8::new(clock::UNTRIED),
            frames: timing::Frames::default(),
            passes: Arc::default(),
            heard,
            device_lost,
            pending_checks,
            page_queue,
            ns_per_mop: Arc::new(std::sync::atomic::AtomicU64::new(PAGE_INITIAL_NS_PER_MOP)),
            test_slice_limit: 0,
            batch,
        })
    }

    /// Queues a deferred device check (such as a page rasterizer's error scope)
    /// to be resolved at the frame's end without mid-frame round trips.
    pub(crate) fn defer_check(&self, check: Box<dyn Check>) {
        self.pending_checks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(check);
    }

    /// Forces map passes to slice at `limit` elements per submission, for
    /// exercising the page slicing path on a native device.
    pub fn with_slice_limit(mut self, limit: usize) -> Self {
        self.test_slice_limit = limit.max(64);
        self
    }

    pub(crate) fn page_slice_ops_budget(&self) -> u64 {
        let rate = self
            .ns_per_mop
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(PAGE_MIN_NS_PER_MOP);
        (PAGE_TARGET_SLICE_NS.saturating_mul(1_000_000) / rate).clamp(25_000_000, 600_000_000_000)
    }

    pub(crate) fn page_pace_ops_budget(&self) -> u64 {
        let slice_ops = self.page_slice_ops_budget();
        (slice_ops.saturating_mul(3) / 2).max(40_000_000)
    }

    pub(crate) fn drain_page(&self) -> Result<(), ExecutionError> {
        self.batch.flush()?;
        if self.wait.parks_with_host() {
            self.spent
                .time(|| self.wait.drain(&self.device, &self.queue))
                .map_err(|error| ExecutionError::Device(told(&self.heard, error)))?;
            let (queued_ops, elapsed_ns) = self.page_queue.take_timing();
            record_page_rate(&self.ns_per_mop, queued_ops, elapsed_ns);
        } else {
            if self._waiter.is_none() {
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            }
            self.page_queue.clear();
        }
        Ok(())
    }

    fn slice_limit_for(
        &self,
        module: &Validated,
        params: &[Value],
        domain: entelechy_abi::Domain,
    ) -> usize {
        if self.test_slice_limit > 0 {
            return self.test_slice_limit;
        }
        if !self.wait.parks_with_host() || self.measured() {
            return usize::MAX;
        }
        let unit_ops = effective_page_unit_ops(module, params);
        let total_ops = (domain.count() as u64).saturating_mul(unit_ops);
        let slice_ops_budget = self.page_slice_ops_budget();
        if total_ops <= slice_ops_budget {
            return usize::MAX;
        }
        let floor = if domain.depth() >= 16 {
            PAGE_SLICE_MIN_VOLUME_ELEMENTS
        } else {
            PAGE_SLICE_MIN_GRID_ELEMENTS
        };
        usize::try_from(slice_ops_budget / unit_ops)
            .unwrap_or(usize::MAX)
            .max(floor)
    }

    /// This backend computing every `Long` as a pair of 32-bit words, as on
    /// a device without 64-bit integers (a page's WebGPU): what a test runs
    /// that path by on a device that has them.
    pub fn with_long_pairs(mut self) -> Self {
        self.int64 = false;
        self.int64_atomics = false;
        self
    }

    /// This backend drawing triangles a vertex pulled at a time, as on a
    /// device whose fragments do not know their primitive: what a test runs
    /// that path by on a device that has it.
    pub fn with_pulled_vertices(mut self) -> Self {
        self.primitive_index = false;
        self.raster = OnceLock::new();
        self
    }

    /// Disables hardware BCn texture uploads so tests can exercise the software
    /// decompression fallback on a device that offers `TEXTURE_COMPRESSION_BC`.
    pub fn without_texture_compression(mut self) -> Self {
        self.texture_compression_bc = false;
        self
    }

    /// Whether the open device has hardware BC1/BC4/BC5/BC7 block compression.
    pub fn supports_texture_compression_bc(&self) -> bool {
        self.texture_compression_bc
    }

    /// Disables hardware `FLOAT32_FILTERABLE` sampling so tests can exercise
    /// the software mip-filtering fallback on `Word` textures.
    pub fn without_float32_filterable(mut self) -> Self {
        self.float32_filterable = false;
        self
    }

    /// Whether the open device offers hardware linear/anisotropic filtering on
    /// 32-bit float textures (`FLOAT32_FILTERABLE`).
    pub fn supports_float32_filterable(&self) -> bool {
        self.float32_filterable
    }

    /// Appends the hardware `wgpu::Sampler` bind group entries declared by `samplers`.
    pub(crate) fn push_sampler_entries<'a>(
        &'a self,
        samplers: wgsl::SamplerSet,
        entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) {
        if samplers.clamp {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::SAMPLER_CLAMP_BINDING,
                resource: wgpu::BindingResource::Sampler(&self.sampler_clamp),
            });
        }
        if samplers.repeat {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::SAMPLER_REPEAT_BINDING,
                resource: wgpu::BindingResource::Sampler(&self.sampler_repeat),
            });
        }
        if samplers.aniso_clamp {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::SAMPLER_ANISO_CLAMP_BINDING,
                resource: wgpu::BindingResource::Sampler(&self.sampler_aniso_clamp),
            });
        }
        if samplers.aniso_repeat {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::SAMPLER_ANISO_REPEAT_BINDING,
                resource: wgpu::BindingResource::Sampler(&self.sampler_aniso_repeat),
            });
        }
    }

    /// How `module`'s shaders compute a `Long`: as the device's own, unless
    /// it has none, or scatters Longs under `min` or `max` with no 64-bit
    /// atomics to land them, which pairs of words land by halves.
    pub(crate) fn longs(&self, module: &Validated) -> wgsl::Longs {
        if self.int64 && (self.int64_atomics || !wgsl::long_extremes(module)) {
            wgsl::Longs::Native
        } else {
            wgsl::Longs::Pairs
        }
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
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "gpu"
    }

    /// A float sum is folded in source order after a sort, and atomics
    /// apply only rules whose order does not matter: the same words each
    /// run.
    fn repeats_exactly(&self) -> bool {
        true
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

    fn set_clocked(&self, on: bool) {
        self.frames.set_clocked(on, self.timestamps);
    }

    fn clock_mark(&self) -> usize {
        self.frames.mark()
    }

    fn end_frame(&self) -> Option<u64> {
        let _ = self.batch.flush();
        self.frames.end_frame(&self.device, &self.queue)
    }

    fn clocked_frame(&self) -> Option<entelechy_rules::ClockedFrame> {
        self.frames.done()
    }

    fn evaluation_begun(&self) {
        self.spare.evaluation_begun();
        self.mirrors.evaluation_begun();
        self.trace_cache.evaluation_begun();
    }

    fn memory(&self) -> Vec<(&'static str, u64)> {
        let failures = self
            .failures
            .lock()
            .map_or(0, |pairs| pairs.len() as u64 * wgsl::FAILURE_BYTES * 2);
        vec![
            ("spare buffers", self.spare.bytes()),
            ("spare targets", self.targets.bytes()),
            ("spare mirrors", self.mirrors.bytes()),
            ("failure words", failures),
        ]
    }

    /// Waits for the queue to run dry. Dispatches are submitted without a
    /// wait, so a caller timing a frame calls this; a Player gets it from
    /// presenting.
    fn settle_device(&self) {
        let _ = self.batch.flush();
        let _ = self
            .spent
            .time(|| self.wait.drain(&self.device, &self.queue));
        if self.wait.parks_with_host() {
            let (queued_ops, elapsed_ns) = self.page_queue.take_timing();
            record_page_rate(&self.ns_per_mop, queued_ops, elapsed_ns);
        } else {
            self.page_queue.clear();
        }
        let pending = std::mem::take(
            &mut *self
                .pending_checks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for check in pending {
            let _ = check.wait();
        }
    }

    fn submitted(&self) -> Option<Box<dyn Check>> {
        let flushed = match self.batch.flush() {
            Ok(sub) => sub,
            Err(error) => return Some(Box::new(ImmediateError(error))),
        };
        // Reuse the just-flushed batch submission; else an empty submission
        // after the rest: done when they are.
        let after = flushed.unwrap_or_else(|| self.queue.submit(std::iter::empty()));
        let pending = std::mem::take(
            &mut *self
                .pending_checks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        Some(Box::new(Submitted {
            wait: self.wait,
            spent: self.spent.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
            after,
            asked: self.wait.ask_done(&self.queue),
            page_queue: self.page_queue.clone(),
            ns_per_mop: Arc::clone(&self.ns_per_mop),
            pending,
        }))
    }

    fn has_device(&self) -> bool {
        true
    }

    fn declines(&self, module: &Validated) -> Option<String> {
        self.refusal(module)
            .or_else(|| {
                (module.manifest.mode == entelechy_ir::Mode::Scan)
                    .then(|| "a scan runs its steps in order, on one thread".to_owned())
            })
            .or_else(|| {
                // Chrome's WebGPU on Apple GPUs loses its device (Metal:
                // insufficient memory) now and then running a prefix's
                // passes over a count that changes frame to frame; taken
                // back once it does not (HANDOFF.md).
                (self.wait.parks_with_host()
                    && matches!(module.manifest.mode, entelechy_ir::Mode::Prefix { .. }))
                .then(|| "a page's device runs no prefix".to_owned())
            })
            .or_else(|| {
                let unbuilt = self
                    .unbuilt
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                unbuilt
                    .iter()
                    .find(|(seen, _)| seen == module)
                    .map(|(_, why)| format!("the device's own compiler could not build it ({why})"))
            })
    }

    fn uploaded(&self) -> u64 {
        self.uploaded.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn upload(&self, handle: Handle, scope: &mut Scope<'_>) -> Result<bool, ExecutionError> {
        if self.upload_compressed(scope, handle)?.is_some() {
            return Ok(true);
        }
        self.bind_input(scope, handle).map(|_| true)
    }

    /// The parts' ranges of their buffers copied one after another into
    /// one taken from the pile, in a submission of its own: what was
    /// written to a part before lands first, and a part let go afterwards
    /// is not reused before it.
    ///
    /// When joining whole layers of hardware block-compressed textures into a
    /// `Volume` (2D texture array), copies compressed blocks GPU-to-GPU via
    /// `copy_texture_to_texture` with zero decompression.
    fn join(
        &self,
        parts: &[(Handle, std::ops::Range<usize>)],
        schema: &entelechy_abi::Schema,
        domain: entelechy_abi::Domain,
        scope: &mut Scope<'_>,
    ) -> Result<Option<Handle>, ExecutionError> {
        if domain.shape() == entelechy_abi::Shape::Volume
            && self.texture_compression_bc
            && !parts.is_empty()
            && let Some(joined) = self.try_join_compressed_layers(parts, schema, domain, scope)?
        {
            return Ok(Some(joined));
        }
        if !schema.element_bytes().is_multiple_of(4) {
            return Ok(None);
        }
        let element = schema.element_bytes() as u64;
        let mut pieces = Vec::with_capacity(parts.len());
        for (part, range) in parts {
            let from = range.start as u64 * element;
            let bytes = range.len() as u64 * element;
            pieces.push((self.bind_input(scope, *part)?, from, bytes));
        }
        let total: u64 = pieces.iter().map(|(_, _, bytes)| bytes).sum();
        if total != schema.storage_words(domain.count()) as u64 * 4 {
            return Err(ExecutionError::Device(format!(
                "the parts joined hold {total} bytes, where {} over {domain} takes {}",
                schema,
                schema.storage_words(domain.count()) * 4
            )));
        }
        scope.can_hold(&[(schema, domain)])?;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buffer = self
            .spare
            .take(&self.device, total.max(4).div_ceil(16) * 16, usage);
        let batched = self.batch.can_batch(self.measured(), &self.frames);
        if !batched {
            self.batch.flush()?;
        }
        let mut encoder = if batched {
            self.batch.take_or_open()
        } else {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("joined"),
                })
        };
        let mut at = 0;
        for (piece, from, bytes) in &pieces {
            if *bytes > 0 {
                encoder.copy_buffer_to_buffer(piece, *from, &buffer, at, *bytes);
            }
            at += bytes;
        }
        if batched {
            self.frames.note_batched_run();
            self.batch
                .finish_node(encoder, 0, Vec::new(), Vec::new(), None)?;
        } else {
            self.submit(encoder.finish());
        }
        let device = Box::new(DeviceImage::new(buffer, Some(self.spare.clone()), None));
        Ok(Some(scope.adopt_resident(
            schema.clone(),
            domain,
            device,
        )?))
    }

    /// Validates `module` as the device takes it; its pipelines are built
    /// when a run first needs them ([`Lazy`]).
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        self.prepare_kernel(module)
            .map(|lazy| Box::new(lazy) as Box<dyn Prepared>)
            .map_err(Unbuilt::said)
    }

    /// Starts `prepared`'s asynchronous pipeline compilation on a page.
    fn warm(&self, prepared: &dyn Prepared) {
        if let Some(lazy) = prepared.as_any().downcast_ref::<Lazy>() {
            lazy.warm();
        }
    }

    /// Builds `prepared`'s pipelines, if they are not yet built.
    fn finish(&self, prepared: &dyn Prepared) {
        if let Some(lazy) = prepared.as_any().downcast_ref::<Lazy>() {
            let _ = lazy.built();
        }
    }

    /// Whether `prepared`'s pipelines are built, or failed to be.
    fn finished(&self, prepared: &dyn Prepared) -> bool {
        prepared
            .as_any()
            .downcast_ref::<Lazy>()
            .is_none_or(Lazy::is_built)
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        // A caller asking for the outputs alone gets the run's failure, waited
        // for, as the reference answers.
        let (outputs, check) = self.execute_checked(prepared, module, invocation, scope)?;
        let result = match check {
            Some(check) => check.wait(),
            None => self.batch.flush().map(|_| ()),
        };
        if let Err(error) = result {
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
    ) -> Result<entelechy_rules::backend::Checked, ExecutionError> {
        check_invocation(module, invocation, scope)?;
        let pipeline = self.built_for(prepared, module, invocation, scope)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mode = module.manifest.mode;
        let domain = execution_domain(module, invocation, scope)?;
        // What comes out: one element for a reduction, the target's extent
        // for a scatter, the domain otherwise.
        let output_domain = match mode {
            Mode::Reduce => entelechy_abi::Domain::Rows(1),
            Mode::Scatter { base, .. } => scope.size(invocation.inputs[base as usize])?,
            _ => domain,
        };
        outputs_fit(module, output_domain, scope)?;
        let unit_ops = effective_page_unit_ops(module, &invocation.params);
        let total_ops = (domain.count() as u64).saturating_mul(unit_ops);
        let slice_limit = self.slice_limit_for(module, &invocation.params, domain);
        if self.wait.parks_with_host()
            && !self.measured()
            && self.page_queue.full(total_ops, self.page_pace_ops_budget())
        {
            self.drain_page()?;
        }
        // Under a profile, the device's clock around every pass; the set is
        // made first so the opening timestamp is the encoder's first command.
        let timing_set = self.timing_set(2);
        let allow_batch = self.batch.can_batch(self.measured(), &self.frames)
            && matches!(mode, Mode::Map | Mode::Group { .. })
            && !(self.wait.parks_with_host() && pipeline.fails);
        let staged = self.stage(
            pipeline,
            module,
            invocation,
            scope,
            domain,
            timing_set.as_ref(),
            slice_limit,
            allow_batch,
        )?;
        let Staged {
            mut encoder,
            batched,
            bind_group,
            further_slices,
            context_buffer,
            params_buffer,
            context_slice: _,
            params_slice: _,
            input_buffers,
            bound_inputs,
            bound_textures: _,
            temp_buffers,
            memory_outputs,
            output_buffer,
            further_outputs,
            output_words,
            element_count,
            failure,
            target,
            highs,
        } = staged;
        // The map pass's output is the result, unless a reduction folds it or a
        // scatter combines it into its target.
        let final_buffer = match (mode, pipeline.reduce.as_ref(), pipeline.scatter.as_ref()) {
            (Mode::Reduce, Some(reduce), _) => self.fold(
                &mut encoder,
                module,
                reduce,
                pipeline.reduce_fold,
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
                    failure.as_ref().map(|(word, _)| word),
                )
            }
            (Mode::Scatter { base, .. }, _, Some(scatter)) => self.combine(
                &mut encoder,
                scatter,
                (target, highs),
                failure.as_ref().map(|(word, _)| word),
                input_buffers[base as usize]
                    .as_ref()
                    .expect("scatter base holds a storage buffer"),
                &context_buffer,
                output_buffer,
                output_domain.count(),
                element_count,
                output_words,
                timing_set.as_ref(),
            ),
            _ => output_buffer,
        };
        let (submitted, scoped) = if batched {
            if let Some((word, back)) = &failure {
                encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
            }
            let _ = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
            self.frames.note_batched_run();
            let mut keep_buffers: Vec<wgpu::Buffer> = bound_inputs
                .iter()
                .filter(|(binding, _)| wgsl::is_packed_binding(*binding))
                .map(|(_, buffer)| buffer.clone())
                .collect();
            keep_buffers.extend(temp_buffers);
            let page_budget = self
                .wait
                .parks_with_host()
                .then(|| self.page_pace_ops_budget());
            self.batch
                .finish_node(encoder, total_ops, keep_buffers, Vec::new(), page_budget)?;
            (None, None)
        } else {
            let scopes = Scopes::push(&self.device);
            let submission = if further_slices.is_empty() {
                // The failure word, copied for reading once the device is waited for,
                // at the caller's commit.
                if let Some((word, back)) = &failure {
                    encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
                }
                let timing = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
                let sub = self.submit(encoder.finish());
                if self.wait.parks_with_host() && !self.measured() {
                    self.page_queue.note(total_ops);
                }
                self.settle_measured(sub.clone(), timing);
                sub
            } else {
                let total_slices = (further_slices.len() + 1) as u64;
                let per_slice_ops = total_ops.div_ceil(total_slices);
                let _ = self.submit(encoder.finish());
                if self.wait.parks_with_host() && !self.measured() {
                    self.page_queue.note(per_slice_ops);
                    if self
                        .page_queue
                        .full(per_slice_ops, self.page_pace_ops_budget())
                    {
                        self.drain_page()?;
                    }
                }
                let scale_bytes = invocation.context.scale.to_le_bytes();
                let one_bytes = 1.0f32.to_le_bytes();
                let last_idx = further_slices.len() - 1;
                let mut last_sub = None;
                for (idx, &(gx, gy, gz, slice_fold, slice_at)) in further_slices.iter().enumerate()
                {
                    let mut slot5 = [0u8; 16];
                    slot5[0..4].copy_from_slice(&scale_bytes);
                    slot5[4..8].copy_from_slice(&slice_fold.to_le_bytes());
                    slot5[8..12].copy_from_slice(&one_bytes);
                    slot5[12..16].copy_from_slice(&slice_at.to_le_bytes());
                    self.queue.write_buffer(&context_buffer, 80, &slot5);
                    let mut slice_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("entelechy slice"),
                            });
                    {
                        let mut pass =
                            slice_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("entelechy kernel slice"),
                                timestamp_writes: None,
                            });
                        pass.set_pipeline(&pipeline.main.0);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(gx, gy, gz);
                    }
                    if idx == last_idx
                        && let Some((word, back)) = &failure
                    {
                        slice_encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
                    }
                    let sub = self.submit(slice_encoder.finish());
                    if self.wait.parks_with_host() && !self.measured() {
                        self.page_queue.note(per_slice_ops);
                        let pace_limit = self.page_pace_ops_budget();
                        let need_drain = if idx == last_idx {
                            total_ops >= pace_limit
                        } else {
                            self.page_queue.full(per_slice_ops, pace_limit)
                        };
                        if need_drain {
                            self.drain_page()?;
                        }
                    }
                    last_sub = Some(sub);
                }
                self.written
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0
                    .clear();
                last_sub.expect("further_slices is non-empty")
            };
            // The packed inputs' copies are read by this run alone and go back to
            // the pile now that its submissions are queued.
            for (binding, buffer) in &bound_inputs {
                if wgsl::is_packed_binding(*binding) {
                    self.spare.keep(buffer.clone(), None);
                }
            }
            for buffer in temp_buffers {
                self.spare.keep(buffer, None);
            }
            // A page answers a scope by a round trip: there what it caught is
            // answered with the run's check, not before the next run.
            let scoped = if self.wait.parks_with_host() && !self.measured() {
                Some(Box::new(scopes.pop_later(self.wait, self.spent.clone())) as Box<dyn Check>)
            } else {
                scopes
                    .pop(self.wait, "")
                    .map_err(|error| ExecutionError::Device(error.said()))?;
                None
            };
            (Some(submission), scoped)
        };
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
            let asked = if submitted.is_some() {
                self.wait.ask_map(&self.queue, pair.1.slice(..))
            } else {
                None
            };
            Box::new(FailureCheck {
                asked,
                wait: self.wait,
                spent: self.spent.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                pair: Some(pair),
                submitted,
                batch: Arc::clone(&self.batch),
                pile: Arc::clone(&self.failures),
                heard: Arc::clone(&self.heard),
                page_queue: self.page_queue.clone(),
                width: domain.width(),
                totals_width: output_domain.width(),
                budget: module.manifest.budget,
                run: entelechy_rules::Run::new(scope.work()),
            }) as Box<dyn Check>
        });
        let check = match (scoped, check) {
            (Some(scoped), Some(check)) => {
                Some(Box::new(Checks(vec![scoped, check])) as Box<dyn Check>)
            }
            (scoped, check) => scoped.or(check),
        };
        Ok((outputs, check))
    }

    /// A map run several times over its own output in one command buffer: the
    /// uniforms written once, two buffers taking turns between times, the
    /// times in one pass. It saves a command buffer and a submission per time.
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
        // A pack is copied before each run and a textured chain input would
        // need a texture mirror refresh between iterations, so they go one by one.
        if prepared
            .as_any()
            .downcast_ref::<Lazy>()
            .is_some_and(|lazy| !lazy.packed.is_empty() || lazy.textured.contains(&input))
        {
            return None;
        }
        // Only the planner's shape: a map of one output that comes back on one
        // input. A measured run times each node, so it goes one by one.
        if repeat < 2
            || output != 0
            || module.manifest.mode != Mode::Map
            || module.manifest.outputs.len() != 1
            || !module.manifest.memories.is_empty()
            || self.measured()
            // A run that can fail is checked a time at a time.
            || wgsl::can_fail(module)
        {
            return None;
        }
        // Every time after the first reads the previous output, at the output's
        // extent; the context states extents once, so where they differ the
        // times go one by one.
        let domain = execution_domain(module, invocation, scope).ok()?;
        if scope.size(*invocation.inputs.get(input)?).ok()? != domain {
            return None;
        }
        if self.wait.parks_with_host() {
            let elems = (domain.count() as u64).saturating_mul(u64::from(repeat));
            let ops = elems.saturating_mul(effective_page_unit_ops(module, &invocation.params));
            if ops > self.page_pace_ops_budget() {
                return None;
            }
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
            NativeOp::Rasterize
            | NativeOp::RasterizeCutout
            | NativeOp::RasterizeCanvas
            | NativeOp::Cover => raster::rasterize(self, &self.raster, invocation, scope, op),
            NativeOp::Trace | NativeOp::TraceInstances => {
                trace::trace(self, &self.trace_hw, &self.trace_bvh, invocation, scope, op)
            }
        }
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.download_unlocked(device, byte_len)
            .map_err(|error| error.to_string())
    }

    fn download_later(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Box<dyn entelechy_rules::Downloading>, String> {
        let image = device
            .downcast_ref::<DeviceImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let src_buffer = self.buffer_of(image).map_err(|error| error.to_string())?;
        self.batch.flush().map_err(|error| error.to_string())?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy download"),
            });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &staging, 0, byte_len as u64);
        let after = self.submit(encoder.finish());
        let asked = Some(self.wait.ask_map_now(&self.queue, staging.slice(..)));
        Ok(Box::new(Coming {
            asked,
            wait: self.wait,
            spent: self.spent.clone(),
            device: self.device.clone(),
            staging: Some(staging),
            byte_len,
            after,
            heard: Arc::clone(&self.heard),
            page_queue: self.page_queue.clone(),
        }))
    }

    fn same_resident(
        &self,
        a: &(dyn Any + Send + Sync),
        b: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        let (Some(a), Some(b)) = (
            a.downcast_ref::<DeviceImage>(),
            b.downcast_ref::<DeviceImage>(),
        ) else {
            return Err("image is resident on a different backend".into());
        };
        let buf_a = self.buffer_of(a).map_err(|error| error.to_string())?;
        let buf_b = self.buffer_of(b).map_err(|error| error.to_string())?;
        self.same_buffers(&buf_a, &buf_b, byte_len)
    }

    fn waited(&self) -> std::time::Duration {
        self.spent.total()
    }
}

pub(crate) enum Driver {
    Wgpu(Box<WgpuBackend>),
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    WebGl(Box<webgl::WebGlBackend>),
}

/// A GPU execution backend: `wgpu` compute where offered, or WebGL2 fragment
/// passes in a page or mini-game without WebGPU + JSPI.
pub struct GpuBackend(pub(crate) Driver);

impl fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBackend({})", self.adapter_name())
    }
}

impl GpuBackend {
    fn inner(&self) -> &dyn Backend {
        match &self.0 {
            Driver::Wgpu(b) => b.as_ref(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => b.as_ref(),
        }
    }

    /// Opens the default adapter, blocking to wait for the device. Where
    /// Entelechy is WebAssembly nothing blocks: a page opens it with
    /// [`GpuBackend::with_wait`].
    pub fn new() -> Result<Self, GpuError> {
        if cfg!(target_arch = "wasm32") {
            return Err(GpuError(
                "a page's device is waited for by giving the page turns; open it with a Wait"
                    .into(),
            ));
        }
        Self::with_wait(Wait::blocking())
    }

    /// Opens the fastest available GPU driver, waiting for the device as
    /// `wait` says. On `wasm32-unknown-unknown`, opens WebGPU when JSPI is
    /// present (unless `entelechy_webgl_only` is set) and falls back to
    /// synchronous WebGL2 when `globalThis.entelechy_webgl` is provided.
    pub fn with_wait(wait: Wait) -> Result<Self, GpuError> {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            if webgl::prefer_webgl() {
                return webgl::WebGlBackend::open().map(|b| Self(Driver::WebGl(Box::new(b))));
            }
            if webgl::can_park_webgpu() {
                match WgpuBackend::with_wait(wait) {
                    Ok(b) => return Ok(Self(Driver::Wgpu(Box::new(b)))),
                    Err(wgpu_err) => {
                        return webgl::WebGlBackend::open()
                            .map(|b| Self(Driver::WebGl(Box::new(b))))
                            .map_err(|_| wgpu_err);
                    }
                }
            }
            webgl::WebGlBackend::open().map(|b| Self(Driver::WebGl(Box::new(b))))
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            WgpuBackend::with_wait(wait).map(|b| Self(Driver::Wgpu(Box::new(b))))
        }
    }

    /// The bytes a workgroup of this device holds at once: what a group's
    /// exchanges may take. Zero on WebGL2, which runs no workgroup Kernels.
    pub fn workgroup_memory(&self) -> u32 {
        match &self.0 {
            Driver::Wgpu(b) => b.workgroup_memory(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => 0,
        }
    }

    /// Hardware subgroup (wave / SIMD-group) capabilities enabled on this
    /// backend. Disabled on WebGL2.
    pub fn subgroups(&self) -> wgsl::SubgroupCaps {
        match &self.0 {
            Driver::Wgpu(b) => b.subgroups(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => wgsl::SubgroupCaps::default(),
        }
    }

    /// This backend with subgroup codegen disabled, for testing non-subgroup
    /// workgroup fallbacks on hardware that supports `SUBGROUP`.
    pub fn without_subgroups(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).without_subgroups()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// Whether hardware ray tracing (`EXPERIMENTAL_RAY_QUERY` BLAS/TLAS) is
    /// enabled on this backend. Always false on WebGL2 and WebGPU.
    pub fn ray_query(&self) -> bool {
        match &self.0 {
            Driver::Wgpu(b) => b.ray_query(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => false,
        }
    }

    /// This backend with hardware ray queries disabled, exercising the
    /// portable GPU Compute BVH fallback on hardware that supports ray queries.
    pub fn without_ray_query(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).without_ray_query()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// This backend computing every `Long` as a pair of 32-bit words, as on
    /// a device without 64-bit integers.
    pub fn with_long_pairs(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).with_long_pairs()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// This backend drawing triangles a vertex pulled at a time, as on a
    /// device whose fragments do not know their primitive.
    pub fn with_pulled_vertices(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).with_pulled_vertices()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// Forces map passes on `wgpu` to slice at `limit` elements per
    /// submission, for exercising the page slicing path on a native device.
    pub fn with_slice_limit(self, limit: usize) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).with_slice_limit(limit)))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// Disables hardware BCn texture compression so tests can exercise the
    /// software fallback on a device that offers `TEXTURE_COMPRESSION_BC`.
    pub fn without_texture_compression(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).without_texture_compression()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// Whether the open GPU supports hardware BC1/BC4/BC5/BC7 textures.
    pub fn supports_texture_compression_bc(&self) -> bool {
        match &self.0 {
            Driver::Wgpu(b) => b.supports_texture_compression_bc(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => false,
        }
    }

    /// Disables hardware `FLOAT32_FILTERABLE` sampling so tests can exercise
    /// the software mip-filtering fallback on `Word` textures.
    pub fn without_float32_filterable(self) -> Self {
        match self.0 {
            Driver::Wgpu(b) => Self(Driver::Wgpu(Box::new((*b).without_float32_filterable()))),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => Self(Driver::WebGl(b)),
        }
    }

    /// Whether the open GPU supports hardware filtering on 32-bit float textures.
    pub fn supports_float32_filterable(&self) -> bool {
        match &self.0 {
            Driver::Wgpu(b) => b.supports_float32_filterable(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => false,
        }
    }

    /// The adapter's name, for diagnostics.
    pub fn adapter_name(&self) -> &str {
        match &self.0 {
            Driver::Wgpu(b) => b.adapter_name(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => b.adapter_name(),
        }
    }

    /// The wgpu device, for interfaces and tests that speak wgpu directly.
    pub fn device(&self) -> &wgpu::Device {
        match &self.0 {
            Driver::Wgpu(b) => b.device(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => panic!("WebGL2 backend has no wgpu::Device"),
        }
    }

    /// Whether executions wait for the device; see [`Backend::set_measured`].
    pub fn measured(&self) -> bool {
        match &self.0 {
            Driver::Wgpu(b) => b.measured(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(b) => b.measured(),
        }
    }

    /// The device time of the last measured execution; `None` when nothing
    /// was measured. Taken once.
    pub fn last_device_time(&self) -> Option<std::time::Duration> {
        self.inner().last_device_time()
    }

    /// The cost of a measured execution's wait with nothing to wait for.
    pub fn wait_floor(&self) -> std::time::Duration {
        match &self.0 {
            Driver::Wgpu(b) => b.wait_floor(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => std::time::Duration::ZERO,
        }
    }

    /// Whether the device's own clock has answered a measured execution.
    pub fn device_clock_answered(&self) -> bool {
        match &self.0 {
            Driver::Wgpu(b) => b.device_clock_answered(),
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Driver::WebGl(_) => false,
        }
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        self.inner().name()
    }

    fn declines(&self, module: &Validated) -> Option<String> {
        self.inner().declines(module)
    }

    fn repeats_exactly(&self) -> bool {
        self.inner().repeats_exactly()
    }

    fn uploaded(&self) -> u64 {
        self.inner().uploaded()
    }

    fn upload(&self, handle: Handle, scope: &mut Scope<'_>) -> Result<bool, ExecutionError> {
        self.inner().upload(handle, scope)
    }

    fn join(
        &self,
        parts: &[(Handle, std::ops::Range<usize>)],
        schema: &entelechy_abi::Schema,
        domain: entelechy_abi::Domain,
        scope: &mut Scope<'_>,
    ) -> Result<Option<Handle>, ExecutionError> {
        self.inner().join(parts, schema, domain, scope)
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        self.inner().prepare(module)
    }

    fn warm(&self, prepared: &dyn Prepared) {
        self.inner().warm(prepared);
    }

    fn finish(&self, prepared: &dyn Prepared) {
        self.inner().finish(prepared);
    }

    fn finished(&self, prepared: &dyn Prepared) -> bool {
        self.inner().finished(prepared)
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner().execute(prepared, module, invocation, scope)
    }

    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<entelechy_rules::backend::Checked, ExecutionError> {
        self.inner()
            .execute_checked(prepared, module, invocation, scope)
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner().execute_native(op, invocation, scope)
    }

    fn last_device_time(&self) -> Option<std::time::Duration> {
        self.inner().last_device_time()
    }

    fn measurement_floor(&self) -> std::time::Duration {
        self.inner().measurement_floor()
    }

    fn set_measured(&self, on: bool) {
        self.inner().set_measured(on);
    }

    fn measures_the_device(&self) -> bool {
        self.inner().measures_the_device()
    }

    fn set_clocked(&self, on: bool) {
        self.inner().set_clocked(on);
    }

    fn clock_mark(&self) -> usize {
        self.inner().clock_mark()
    }

    fn end_frame(&self) -> Option<u64> {
        self.inner().end_frame()
    }

    fn clocked_frame(&self) -> Option<entelechy_rules::ClockedFrame> {
        self.inner().clocked_frame()
    }

    fn evaluation_begun(&self) {
        self.inner().evaluation_begun();
    }

    fn memory(&self) -> Vec<(&'static str, u64)> {
        self.inner().memory()
    }

    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        self.inner()
            .execute_repeated(prepared, module, invocation, scope, repeat, chain)
    }

    fn settle_device(&self) {
        self.inner().settle_device();
    }

    fn submitted(&self) -> Option<Box<dyn Check>> {
        self.inner().submitted()
    }

    fn has_device(&self) -> bool {
        self.inner().has_device()
    }

    fn has_compute_device(&self) -> bool {
        self.inner().has_compute_device()
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, String> {
        self.inner().download(device, byte_len)
    }

    fn download_later(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Box<dyn entelechy_rules::Downloading>, String> {
        self.inner().download_later(device, byte_len)
    }

    fn same_resident(
        &self,
        a: &(dyn Any + Send + Sync),
        b: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        self.inner().same_resident(a, b, byte_len)
    }

    fn waited(&self) -> std::time::Duration {
        self.inner().waited()
    }
}

#[cfg(test)]
mod page_pacing_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn adaptive_page_rate_converges_for_fast_gpus_and_clamps_up_on_slow_gpus() {
        let rate = AtomicU64::new(PAGE_INITIAL_NS_PER_MOP);
        // Fast discrete GPU completing 50 Gop in 5 ms converges down rapidly.
        for _ in 0..8 {
            record_page_rate(&rate, 50_000_000_000, 5_000_000);
        }
        assert_eq!(rate.load(Ordering::Relaxed), PAGE_MIN_NS_PER_MOP);

        // Slow integrated GPU taking 250 ms for 100 Mop snaps up immediately.
        record_page_rate(&rate, 100_000_000, 250_000_000);
        assert_eq!(rate.load(Ordering::Relaxed), 2_500_000);
    }
}
