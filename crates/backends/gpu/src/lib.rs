#![warn(missing_docs)]
//! GPU backend.
//!
//! Emits WGSL from validated IR at preparation and runs each execution
//! through wgpu. The Host owns images and their pixels stay resident on
//! the device: a resident input binds directly, a Host input uploads once
//! and stays resident, and outputs are adopted as resident images. Pixels
//! return only through [`Backend::download`]. Semantics match the CPU
//! backend up to floating-point rounding; integer failures are reported
//! as the CPU reports them.

mod failure;
mod later;
mod modes;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod page;
mod pool;
mod prepare;
pub mod present;
mod raster;
mod same;
mod stage;
mod timing;
mod wait;
pub mod wgsl;

use wait::Spent;
pub use wait::Wait;

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
pub struct GpuBackend {
    /// How to wait for the device: block, or give the page turns.
    wait: Wait,
    /// The time callers have spent blocked on the device's work.
    spent: Spent,
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
    /// The pass that compares two Fields, built on first use.
    same: OnceLock<Result<same::SamePipeline, String>>,
    /// Buffers released Fields left behind; see [`Spare`].
    spare: Spare,
    /// The rasterizer's own targets, kept by shape; see [`SpareTextures`].
    targets: SpareTextures,
    /// The context and parameter buffers of the execution being staged,
    /// written again per node: one execution runs at a time, and a queue's
    /// writes are ordered with its submissions.
    uniforms: Mutex<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// What those buffers were last given, so a node staged with the same
    /// context and parameters writes nothing.
    written: Mutex<(Vec<u8>, Vec<u8>)>,
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
    /// What the device said by itself: an error no scope caught, its loss.
    /// A readback that fails says it too, as the cause is most often there.
    heard: Arc<Mutex<Vec<String>>>,
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

impl fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBackend({})", self.adapter_name)
    }
}

impl GpuBackend {
    /// The bytes a workgroup of this device holds at once: what a group's
    /// exchanges may take.
    pub fn workgroup_memory(&self) -> u32 {
        self.device.limits().max_compute_workgroup_storage_size
    }

    /// Gives the device a run, work or copy: every run but the empty ones
    /// a wait is marked by goes through here, so a clocked frame counts
    /// each.
    /// What was let go before it is destroyed after it ([`Spare::bury`]).
    pub(crate) fn submit(&self, commands: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        let submission = self.frames.submit(&self.queue, commands);
        self.spare.bury();
        self.targets.bury();
        submission
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
        let required_limits = wgpu::Limits {
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
        let (device, queue) = wait
            .block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("entelechy"),
                required_features: features,
                required_limits,
                ..Default::default()
            }))
            .map_err(|error| GpuError(error.to_string()))?;
        let adapter_name = adapter.get_info().name;
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
        {
            let heard = Arc::clone(&heard);
            device.set_device_lost_callback(move |reason, message| {
                keep(
                    &heard,
                    format!("the device was lost ({reason:?}): {message}"),
                )
            });
        }
        Ok(Self {
            wait,
            spent: Spent::default(),
            instance,
            adapter,
            device,
            queue,
            adapter_name,
            lock: Mutex::new(()),
            uniforms: Mutex::new(None),
            written: Mutex::new((Vec::new(), Vec::new())),
            raster: OnceLock::new(),
            same: OnceLock::new(),
            spare: Spare::default(),
            targets: SpareTextures::default(),
            measured: std::sync::atomic::AtomicBool::new(false),
            unbuilt: Arc::new(Mutex::new(Vec::new())),
            timestamps,
            int64,
            int64_atomics,
            primitive_index,
            uploaded: std::sync::atomic::AtomicU64::new(0),
            failures: Arc::new(Mutex::new(Vec::new())),
            last_nanos: std::sync::atomic::AtomicU64::new(u64::MAX),
            clock: std::sync::atomic::AtomicU8::new(clock::UNTRIED),
            frames: timing::Frames::default(),
            heard,
        })
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

impl Backend for GpuBackend {
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
        self.frames.set_clocked(on, &self.device, self.timestamps);
    }

    fn clock_mark(&self) -> usize {
        self.frames.mark()
    }

    fn end_frame(&self) -> Option<u64> {
        self.frames.end_frame(&self.device, &self.queue)
    }

    fn clocked_frame(&self) -> Option<entelechy_rules::ClockedFrame> {
        self.frames.done()
    }

    fn evaluation_begun(&self) {
        self.spare.evaluation_begun();
    }

    fn memory(&self) -> Vec<(&'static str, u64)> {
        let failures = self
            .failures
            .lock()
            .map_or(0, |pairs| pairs.len() as u64 * wgsl::FAILURE_BYTES * 2);
        vec![
            ("spare buffers", self.spare.bytes()),
            ("spare targets", self.targets.bytes()),
            ("failure words", failures),
        ]
    }

    /// Waits for the queue to run dry. Dispatches are submitted without a
    /// wait, so a caller timing a frame calls this; a Player gets it from
    /// presenting.
    fn settle_device(&self) {
        let _ = self
            .spent
            .time(|| self.wait.drain(&self.device, &self.queue));
    }

    fn submitted(&self) -> Option<Box<dyn Check>> {
        // An empty submission after the rest: done when they are.
        let after = self.queue.submit(std::iter::empty());
        Some(Box::new(Submitted {
            wait: self.wait,
            spent: self.spent.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
            after,
            asked: self.wait.ask_done(&self.queue),
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
        self.bind_input(scope, handle).map(|_| true)
    }

    /// The parts' ranges of their buffers copied one after another into
    /// one taken from the pile, in a submission of its own: what was
    /// written to a part before lands first, and a part let go afterwards
    /// is not reused before it.
    fn join(
        &self,
        parts: &[(Handle, std::ops::Range<usize>)],
        schema: &entelechy_abi::Schema,
        domain: entelechy_abi::Domain,
        scope: &mut Scope<'_>,
    ) -> Result<Option<Handle>, ExecutionError> {
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("joined"),
            });
        let mut at = 0;
        for (piece, from, bytes) in &pieces {
            if *bytes > 0 {
                encoder.copy_buffer_to_buffer(piece, *from, &buffer, at, *bytes);
            }
            at += bytes;
        }
        self.submit(encoder.finish());
        let device = Box::new(DeviceImage {
            buffer,
            spare: Some(self.spare.clone()),
            holds: None,
        });
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
    ) -> Result<entelechy_rules::backend::Checked, ExecutionError> {
        check_invocation(module, invocation, scope)?;
        let pipeline = built(prepared)?;
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
        // Under a profile, the device's clock around every pass; the set is
        // made first so the opening timestamp is the encoder's first command.
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
            highs,
        } = staged;
        // The map pass's output is the result, unless a reduction folds it or a
        // scatter combines it into its target.
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
                    failure.as_ref().map(|(word, _)| word),
                )
            }
            (Mode::Scatter { base, .. }, _, Some(scatter)) => self.combine(
                &mut encoder,
                scatter,
                (target, highs),
                failure.as_ref().map(|(word, _)| word),
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
        // The failure word, copied for reading once the device is waited for,
        // at the caller's commit.
        if let Some((word, back)) = &failure {
            encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
        }
        // The packed inputs' copy is read by this run alone.
        for (binding, buffer) in &bound_inputs {
            if *binding == wgsl::PACKED_BINDING {
                self.spare.let_go(buffer.clone());
            }
        }
        let scopes = Scopes::push(&self.device);
        let timing = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
        let submission = self.submit(encoder.finish());
        let submitted = submission.clone();
        // One queue runs submissions in order, so only observation waits; under
        // `set_measured` this call waits, so its time is the work.
        self.settle_measured(submission, timing);
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
            let asked = self.wait.ask_map(&self.queue, pair.1.slice(..));
            Box::new(FailureCheck {
                asked,
                wait: self.wait,
                spent: self.spent.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                pair: Some(pair),
                submitted,
                pile: Arc::clone(&self.failures),
                heard: Arc::clone(&self.heard),
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
        // A pack is copied before each run and the times would read the chain
        // through it, so they go one by one.
        if prepared
            .as_any()
            .downcast_ref::<Lazy>()
            .is_some_and(|lazy| !lazy.packed.is_empty())
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
        Some(self.repeated(prepared, module, invocation, scope, repeat, input))
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        match op {
            NativeOp::Rasterize | NativeOp::RasterizeCutout | NativeOp::RasterizeCanvas => {
                raster::rasterize(self, &self.raster, invocation, scope, op)
            }
        }
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, String> {
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
                label: Some("entelechy download"),
            });
        encoder.copy_buffer_to_buffer(&image.buffer, 0, &staging, 0, size);
        self.submit(encoder.finish());
        self.spent
            .time(|| {
                self.wait
                    .map_read(&self.device, &self.queue, staging.slice(..))
            })
            .map_err(|error| told(&self.heard, error))?;
        let words = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            words_of(&view[..byte_len])
        };
        staging.unmap();
        staging.destroy();
        Ok(words)
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
        encoder.copy_buffer_to_buffer(&image.buffer, 0, &staging, 0, byte_len as u64);
        let after = self.submit(encoder.finish());
        let asked = self.wait.ask_map(&self.queue, staging.slice(..));
        Ok(Box::new(Coming {
            asked,
            wait: self.wait,
            spent: self.spent.clone(),
            device: self.device.clone(),
            queue: self.queue.clone(),
            staging,
            byte_len,
            after,
            heard: Arc::clone(&self.heard),
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
        self.same_buffers(&a.buffer, &b.buffer, byte_len)
    }

    fn waited(&self) -> std::time::Duration {
        self.spent.total()
    }
}
