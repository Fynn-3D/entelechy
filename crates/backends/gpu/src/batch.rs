//! Same-frame GPU command batching with 3-slot ringed uniform, small-upload,
//! and failure-readback arenas.
//!
//! Instead of creating a fresh `CommandEncoder`, pushing/popping three error
//! scopes, submitting to the queue, and mapping a separate failure buffer for
//! every node in a frame, consecutive GPU passes record into one shared
//! encoder. All per-node uniform records (`context`, `params`,
//! rasterizer/prefix/reduce setup) pack into one 256-byte-aligned uniform slab
//! written once right before submission, small Host uploads (`<= 64 KiB`, such
//! as per-frame poses, morph weights, and camera lenses from JIT nodes) stage
//! into one upload slab copied in exact pass order via
//! `copy_buffer_to_buffer`, and all per-node failure records (plus unclamped
//! indirect-domain checks) pack into one storage-aligned failure slab cleared
//! once before submit, copied once to a single staging buffer, and mapped
//! immediately at submit time tied to that batch's exact [`wgpu::SubmissionIndex`].
//!
//! Slabs rotate across a 3-slot ring (`RING = 3`) so in-flight overlapped
//! frames and multi-batch frames never write over a slab still read by the GPU.
//!
//! Any observation boundary (`submitted`, `settle_device`, `end_frame`,
//! `download`, `download_later`, `same_resident`, `present`,
//! `FailureCheck::wait`, large Host uploads, measured single-node profiling, or
//! WebGPU TDR pacing) flushes the open batch cleanly so ordering and error
//! scopes remain exact.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use entelechy_rules::{Check, ExecutionError};

use crate::PageQueue;
use crate::failure::Scopes;
use crate::pool::{Spare, SpareTextures};
use crate::timing::Frames;
use crate::wait::{Asked, Spent, Wait, WaiterControl};

/// Ring depth for per-batch uniform, upload, and failure slabs so up to two
/// overlapped frames (including multi-batch frames) never overwrite a slab
/// still being read on the GPU.
const RING: usize = 3;
/// Initial capacity of each per-frame uniform arena (`256 KiB` holds ~500 nodes).
const INITIAL_UNIFORM_SLAB: usize = 256 * 1024;
/// Initial capacity of each per-frame small-upload staging arena (`256 KiB`).
const INITIAL_UPLOAD_SLAB: usize = 256 * 1024;
/// Initial capacity of each per-frame failure arena (`64 KiB` holds 256 slots at 256 B align).
const INITIAL_FAILURE_SLAB: usize = 64 * 1024;
/// Largest single Host upload staged through the in-batch copy slab (`64 KiB`).
/// Larger uploads (e.g. Frame-1 static meshes/textures) flush and stream directly.
const MAX_SMALL_UPLOAD: usize = 64 * 1024;
/// Maximum nodes accumulated in one encoder before flushing to keep driver
/// command buffers bounded.
const MAX_BATCH_PASSES: u32 = 128;

/// One uniform slice inside a buffer, bound either at a fixed offset
/// (`binding`) or with a dynamic offset (`dynamic_binding` + `offset as u32`).
#[derive(Clone)]
pub(crate) struct UniformSlice {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl UniformSlice {
    pub(crate) fn entire(buffer: wgpu::Buffer) -> Self {
        let size = buffer.size();
        Self {
            buffer,
            offset: 0,
            size,
        }
    }

    /// Fixed-offset uniform binding (`has_dynamic_offset: false`).
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        })
    }

    /// Base-zero uniform binding (`has_dynamic_offset: true`), paired with
    /// `self.offset as u32` in `pass.set_bind_group`.
    pub(crate) fn dynamic_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

/// One storage-aligned failure record slot inside a batch's shared failure slab.
#[derive(Clone)]
pub(crate) struct FailureBinding {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) offset: u64,
}

impl FailureBinding {
    /// Fixed-offset storage binding for helper passes (`prefix`, `reduce`,
    /// `scatter` combine/totals).
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(crate::wgsl::FAILURE_BYTES),
        })
    }

    /// Base-zero storage binding for `Pipeline.main` (`has_dynamic_offset: true`),
    /// paired with `self.offset as u32` in `pass.set_bind_group`.
    pub(crate) fn dynamic_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(crate::wgsl::FAILURE_BYTES),
        })
    }
}

enum FailureReadbackState {
    /// Slots are staged in the open batch and not yet submitted.
    Pending,
    /// Batch submitted and `map_async` already requested for its exact submission.
    InFlight {
        submission: wgpu::SubmissionIndex,
        staging: wgpu::Buffer,
        byte_len: u64,
        asked: Asked<Result<(), wgpu::BufferAsyncError>>,
    },
    /// Staging buffer mapped, copied into CPU memory, unmapped, and recycled.
    Resolved(Result<Vec<u8>, String>),
}

/// Shared readback for every failure record (and unclamped indirect-domain
/// check) recorded inside one submitted batch.
///
/// Initiates ONE `map_async` right after `queue.submit()` tied to that exact
/// submission index so 2-frame overlap waits strictly for Frame N while Frame
/// N+1 executes in parallel. The first `Check::wait` resolves the map once and
/// copies all slots into CPU memory; every remaining check in the batch reads
/// its 20 bytes directly from RAM with zero GPU calls.
pub(crate) struct BatchFailureReadback {
    state: Mutex<FailureReadbackState>,
    resolved: AtomicBool,
    batch: Weak<BatchHub>,
    device: wgpu::Device,
    wait: Wait,
    spent: Spent,
    spare: Spare,
    heard: Arc<Mutex<Vec<String>>>,
    page_queue: PageQueue,
}

impl Drop for BatchFailureReadback {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock()
            && let FailureReadbackState::InFlight { staging, .. } =
                std::mem::replace(&mut *state, FailureReadbackState::Pending)
        {
            crate::wait::discard(staging);
        }
    }
}

impl BatchFailureReadback {
    fn new(hub: &BatchHub) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(FailureReadbackState::Pending),
            resolved: AtomicBool::new(false),
            batch: hub.self_weak.clone(),
            device: hub.device.clone(),
            wait: hub.wait,
            spent: hub.spent.clone(),
            spare: hub.spare.clone(),
            heard: Arc::clone(&hub.heard),
            page_queue: hub.page_queue.clone(),
        })
    }

    fn mark_in_flight(
        &self,
        submission: wgpu::SubmissionIndex,
        staging: wgpu::Buffer,
        byte_len: u64,
        asked: Asked<Result<(), wgpu::BufferAsyncError>>,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *state = FailureReadbackState::InFlight {
            submission,
            staging,
            byte_len,
            asked,
        };
    }

    /// Whether the batch's readback has already completed on the GPU, proving
    /// its submission index is finished without another `device.poll`.
    pub(crate) fn is_resolved(&self) -> bool {
        self.resolved.load(Ordering::Acquire)
    }

    /// Reads `(word, blocks, left, total)` from the failure record at `offset`.
    pub(crate) fn read_slot(&self, offset: u64) -> Result<(u32, u32, u32, u32), ExecutionError> {
        self.with_bytes(|bytes| {
            let base = offset as usize;
            let at = |k: usize| {
                u32::from_le_bytes([
                    bytes[base + k],
                    bytes[base + k + 1],
                    bytes[base + k + 2],
                    bytes[base + k + 3],
                ])
            };
            (at(0), at(4), at(12), at(16))
        })
    }

    /// Reads the leading `u32` word at `offset` (used for batched
    /// [`crate::indirect::IndirectDomainCheck`]).
    pub(crate) fn read_u32(&self, offset: u64) -> Result<u32, ExecutionError> {
        self.with_bytes(|bytes| {
            let base = offset as usize;
            u32::from_le_bytes([
                bytes[base],
                bytes[base + 1],
                bytes[base + 2],
                bytes[base + 3],
            ])
        })
    }

    fn with_bytes<T>(&self, read: impl FnOnce(&[u8]) -> T) -> Result<T, ExecutionError> {
        loop {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            match &*state {
                FailureReadbackState::Resolved(Ok(bytes)) => return Ok(read(bytes)),
                FailureReadbackState::Resolved(Err(error)) => {
                    return Err(ExecutionError::Device(error.clone()));
                }
                FailureReadbackState::Pending => {
                    drop(state);
                    let Some(hub) = self.batch.upgrade() else {
                        return Err(ExecutionError::Device(
                            "batch failure readback outlived its batcher".to_owned(),
                        ));
                    };
                    let flushed = hub.flush()?;
                    if flushed.is_none() {
                        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
                        if matches!(*guard, FailureReadbackState::Pending) {
                            return Err(ExecutionError::Device(
                                "batch failure readback was never submitted".to_owned(),
                            ));
                        }
                    }
                }
                FailureReadbackState::InFlight { .. } => {
                    let FailureReadbackState::InFlight {
                        submission,
                        staging,
                        byte_len,
                        asked,
                    } = std::mem::replace(&mut *state, FailureReadbackState::Pending)
                    else {
                        unreachable!();
                    };
                    let mapped = self.spent.time(|| {
                        self.wait
                            .answer_after(&self.device, &submission, asked)?
                            .map_err(|error| error.to_string())
                    });
                    match mapped {
                        Ok(()) => {
                            self.page_queue.clear();
                            let copied = staging
                                .slice(..byte_len)
                                .get_mapped_range()
                                .map(|range| range.to_vec())
                                .map_err(|error| error.to_string());
                            staging.unmap();
                            self.spare.keep(staging, None);
                            match copied {
                                Ok(bytes) => {
                                    let out = read(&bytes);
                                    *state = FailureReadbackState::Resolved(Ok(bytes));
                                    self.resolved.store(true, Ordering::Release);
                                    return Ok(out);
                                }
                                Err(error) => {
                                    *state = FailureReadbackState::Resolved(Err(error.clone()));
                                    return Err(ExecutionError::Device(error));
                                }
                            }
                        }
                        Err(error) => {
                            crate::wait::discard(staging);
                            let full = crate::told(&self.heard, error);
                            *state = FailureReadbackState::Resolved(Err(full.clone()));
                            return Err(ExecutionError::Device(full));
                        }
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct BatchState {
    encoder: Option<wgpu::CommandEncoder>,
    ring_idx: usize,
    uniform_slabs: [Option<wgpu::Buffer>; RING],
    uniform_bytes: Vec<u8>,
    upload_slabs: [Option<wgpu::Buffer>; RING],
    upload_bytes: Vec<u8>,
    failure_slabs: [Option<wgpu::Buffer>; RING],
    failure_bytes: Vec<u8>,
    failure_readback: Option<Arc<BatchFailureReadback>>,
    keep_buffers: Vec<wgpu::Buffer>,
    keep_textures: Vec<wgpu::Texture>,
    last_submission: Option<wgpu::SubmissionIndex>,
    passes: u32,
    ops: u64,
}

/// Shared same-frame command batcher held by `WgpuBackend`, `WgpuDisplay`, and
/// deferred failure checks so every observation or present flushes pending GPU
/// passes first.
pub(crate) struct BatchHub {
    self_weak: Weak<Self>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    wait: Wait,
    spent: Spent,
    spare: Spare,
    targets: SpareTextures,
    mirrors: crate::texture::SpareMirrors,
    page_queue: PageQueue,
    pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>>,
    heard: Arc<Mutex<Vec<String>>>,
    waiter_control: WaiterControl,
    uniform_align: usize,
    storage_align: usize,
    state: Mutex<BatchState>,
}

impl BatchHub {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        wait: Wait,
        spent: Spent,
        spare: Spare,
        targets: SpareTextures,
        mirrors: crate::texture::SpareMirrors,
        page_queue: PageQueue,
        pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>>,
        heard: Arc<Mutex<Vec<String>>>,
        waiter_control: WaiterControl,
    ) -> Arc<Self> {
        let limits = device.limits();
        let uniform_align = (limits.min_uniform_buffer_offset_alignment as usize).max(256);
        let storage_align = (limits.min_storage_buffer_offset_alignment as usize).max(256);
        Arc::new_cyclic(|self_weak| Self {
            self_weak: self_weak.clone(),
            device,
            queue,
            wait,
            spent,
            spare,
            targets,
            mirrors,
            page_queue,
            pending_checks,
            heard,
            waiter_control,
            uniform_align,
            storage_align,
            state: Mutex::new(BatchState::default()),
        })
    }

    /// Whether a normal pass may join the open frame batch.
    pub(crate) fn can_batch(&self, measured: bool, frames: &Frames) -> bool {
        !measured && (!frames.clocked() || frames.stamps())
    }

    /// Whether a batch encoder is currently open with unsubmitted passes.
    pub(crate) fn is_open(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .encoder
            .is_some()
    }

    /// Allocates all uniform slices for one node atomically inside the same
    /// uniform slab so no slice can be split across a flush.
    pub(crate) fn alloc_uniforms(
        &self,
        slices: &[&[u8]],
    ) -> Result<Vec<UniformSlice>, ExecutionError> {
        self.alloc_uniforms_and_failure(slices, &[])
            .map(|(uniforms, _, _)| uniforms)
    }

    /// Allocates all uniform slices and any failure / indirect-domain check
    /// slots for one node atomically in the active batch ring slot, flushing
    /// first if either arena would overflow so the node's bindings never split
    /// across two submissions.
    #[allow(clippy::type_complexity)]
    pub(crate) fn alloc_uniforms_and_failure(
        &self,
        slices: &[&[u8]],
        failure_limits: &[u32],
    ) -> Result<
        (
            Vec<UniformSlice>,
            Vec<FailureBinding>,
            Option<Arc<BatchFailureReadback>>,
        ),
        ExecutionError,
    > {
        let u_align = self.uniform_align;
        let f_align = self.storage_align;
        let fail_bytes = crate::wgsl::FAILURE_BYTES as usize;
        let padded_sizes: Vec<usize> = slices
            .iter()
            .map(|b| b.len().max(16).div_ceil(16) * 16)
            .collect();
        let needs_flush = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.encoder.is_none() {
                false
            } else {
                let idx = state.ring_idx;
                let mut u_off = state.uniform_bytes.len();
                for &pad in &padded_sizes {
                    u_off = u_off.div_ceil(u_align) * u_align + pad;
                }
                let u_over = state.uniform_slabs[idx]
                    .as_ref()
                    .is_some_and(|slab| u_off as u64 > slab.size());
                let f_over = if failure_limits.is_empty() {
                    false
                } else {
                    let mut f_off = state.failure_bytes.len();
                    for _ in failure_limits {
                        f_off = f_off.div_ceil(f_align) * f_align + fail_bytes;
                    }
                    state.failure_slabs[idx]
                        .as_ref()
                        .is_some_and(|slab| f_off as u64 > slab.size())
                };
                u_over || f_over
            }
        };
        if needs_flush {
            self.flush()?;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.encoder.is_none() {
            state.uniform_bytes.clear();
            state.upload_bytes.clear();
            state.failure_bytes.clear();
            state.failure_readback = None;
        }
        let idx = state.ring_idx;
        let mut u_offsets = Vec::with_capacity(slices.len());
        let mut u_cursor = state.uniform_bytes.len();
        for &pad in &padded_sizes {
            u_cursor = u_cursor.div_ceil(u_align) * u_align;
            u_offsets.push(u_cursor);
            u_cursor += pad;
        }
        self.ensure_uniform_slab(&mut state, idx, u_cursor as u64);
        state.uniform_bytes.resize(u_cursor, 0);
        for (&off, &bytes) in u_offsets.iter().zip(slices) {
            state.uniform_bytes[off..off + bytes.len()].copy_from_slice(bytes);
        }
        let u_buffer = state.uniform_slabs[idx]
            .as_ref()
            .expect("uniform slab initialized")
            .clone();
        let uniforms: Vec<UniformSlice> = u_offsets
            .into_iter()
            .zip(padded_sizes)
            .map(|(off, pad)| UniformSlice {
                buffer: u_buffer.clone(),
                offset: off as u64,
                size: pad as u64,
            })
            .collect();
        if failure_limits.is_empty() {
            return Ok((uniforms, Vec::new(), None));
        }
        let mut f_offsets = Vec::with_capacity(failure_limits.len());
        let mut f_cursor = state.failure_bytes.len();
        for _ in failure_limits {
            f_cursor = f_cursor.div_ceil(f_align) * f_align;
            f_offsets.push(f_cursor);
            f_cursor += fail_bytes;
        }
        self.ensure_failure_slab(&mut state, idx, f_cursor as u64);
        state.failure_bytes.resize(f_cursor, 0);
        for (&off, &limit) in f_offsets.iter().zip(failure_limits) {
            state.failure_bytes[off + 8..off + 12].copy_from_slice(&limit.to_le_bytes());
        }
        let f_buffer = state.failure_slabs[idx]
            .as_ref()
            .expect("failure slab initialized")
            .clone();
        let readback = Arc::clone(
            state
                .failure_readback
                .get_or_insert_with(|| BatchFailureReadback::new(self)),
        );
        let failures = f_offsets
            .into_iter()
            .map(|off| FailureBinding {
                buffer: f_buffer.clone(),
                offset: off as u64,
            })
            .collect();
        Ok((uniforms, failures, Some(readback)))
    }

    fn ensure_uniform_slab(&self, state: &mut BatchState, idx: usize, needed: u64) {
        let rebuild = state.uniform_slabs[idx]
            .as_ref()
            .is_none_or(|slab| slab.size() < needed);
        if rebuild {
            if let Some(old) = state.uniform_slabs[idx].take() {
                self.spare.let_go(old);
            }
            let cap = (needed as usize)
                .next_power_of_two()
                .max(INITIAL_UNIFORM_SLAB);
            state.uniform_slabs[idx] = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame uniforms"),
                size: cap as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    fn ensure_failure_slab(&self, state: &mut BatchState, idx: usize, needed: u64) {
        let rebuild = state.failure_slabs[idx]
            .as_ref()
            .is_none_or(|slab| slab.size() < needed);
        if rebuild {
            if let Some(old) = state.failure_slabs[idx].take() {
                self.spare.let_go(old);
            }
            let cap = (needed as usize)
                .next_power_of_two()
                .max(INITIAL_FAILURE_SLAB);
            state.failure_slabs[idx] = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame failures"),
                size: cap as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    /// Stages a small Host upload into the open batch via `copy_buffer_to_buffer`
    /// so it executes in exact pass order without flushing the frame.
    /// Returns `Ok(true)` if handled in-batch, or `Ok(false)` (after flushing
    /// any open batch) if the caller should upload directly with `write_words`.
    pub(crate) fn stage_upload(
        &self,
        dst: &wgpu::Buffer,
        dst_word_offset: usize,
        words: &[u32],
    ) -> Result<bool, ExecutionError> {
        let byte_len = words.len() * 4;
        if byte_len == 0 {
            return Ok(true);
        }
        if !self.is_open() {
            return Ok(false);
        }
        if byte_len > MAX_SMALL_UPLOAD {
            self.flush()?;
            return Ok(false);
        }
        let fits = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let idx = state.ring_idx;
            let off = state.upload_bytes.len().div_ceil(4) * 4;
            let cap = state.upload_slabs[idx]
                .as_ref()
                .map_or(INITIAL_UPLOAD_SLAB as u64, wgpu::Buffer::size);
            (off + byte_len) as u64 <= cap
        };
        if !fits {
            self.flush()?;
            return Ok(false);
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.encoder.is_none() {
            return Ok(false);
        }
        let idx = state.ring_idx;
        if state.upload_slabs[idx].is_none() {
            state.upload_slabs[idx] = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame uploads"),
                size: INITIAL_UPLOAD_SLAB as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let off = state.upload_bytes.len().div_ceil(4) * 4;
        state.upload_bytes.resize(off + byte_len, 0);
        for (dst_chunk, &word) in state.upload_bytes[off..off + byte_len]
            .chunks_exact_mut(4)
            .zip(words)
        {
            dst_chunk.copy_from_slice(&word.to_le_bytes());
        }
        let slab = state.upload_slabs[idx]
            .as_ref()
            .expect("upload slab initialized")
            .clone();
        let encoder = state.encoder.as_mut().expect("batch encoder checked open");
        encoder.copy_buffer_to_buffer(
            &slab,
            off as u64,
            dst,
            (dst_word_offset * 4) as u64,
            byte_len as u64,
        );
        Ok(true)
    }

    /// Records a buffer-to-buffer copy into the open batch if one is active.
    pub(crate) fn copy_if_open(&self, from: &wgpu::Buffer, to: &wgpu::Buffer, size: u64) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(encoder) = state.encoder.as_mut() {
            if size > 0 {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, size);
            }
            true
        } else {
            false
        }
    }

    /// Takes the open frame encoder, or opens a new batch.
    pub(crate) fn take_or_open(&self) -> wgpu::CommandEncoder {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(encoder) = state.encoder.take() {
            return encoder;
        }
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy frame"),
            })
    }

    /// Returns the encoder after recording one node's passes, deferring
    /// temporary buffer/texture returns until the batch is submitted.
    /// Flushes automatically if the batch reached its pass ceiling or page budget.
    pub(crate) fn finish_node(
        &self,
        encoder: wgpu::CommandEncoder,
        ops: u64,
        keep_buffers: Vec<wgpu::Buffer>,
        keep_textures: Vec<wgpu::Texture>,
        page_budget: Option<u64>,
    ) -> Result<(), ExecutionError> {
        let should_flush = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.encoder = Some(encoder);
            state.passes = state.passes.saturating_add(1);
            state.ops = state.ops.saturating_add(ops);
            state.keep_buffers.extend(keep_buffers);
            state.keep_textures.extend(keep_textures);
            let over_passes = state.passes >= MAX_BATCH_PASSES;
            let over_page = page_budget.is_some_and(|limit| state.ops >= limit);
            over_passes || over_page
        };
        if should_flush {
            self.flush()?;
        }
        Ok(())
    }

    /// Submits any open frame batch in ONE `queue.submit()`.
    pub(crate) fn flush(&self) -> Result<Option<wgpu::SubmissionIndex>, ExecutionError> {
        self.flush_with_failures().map(|(sub, _)| sub)
    }

    /// Submits any open frame batch in ONE `queue.submit()`, uploading its
    /// staged uniforms, small inputs, and failure slab initializers first,
    /// copying the failure slab to one staging buffer, kicking off async
    /// readback immediately for this exact submission, rotating the 3-slot
    /// ring, burying retired resources, and popping the shared error scope.
    ///
    /// Returns `(Some(submission), failure_readback)` when an open batch was
    /// submitted, or `(None, None)` when no batch was open.
    #[allow(clippy::type_complexity)]
    pub(crate) fn flush_with_failures(
        &self,
    ) -> Result<
        (
            Option<wgpu::SubmissionIndex>,
            Option<Arc<BatchFailureReadback>>,
        ),
        ExecutionError,
    > {
        let (
            mut encoder,
            uniform_slab,
            uniform_data,
            upload_slab,
            upload_data,
            failure_slab,
            failure_data,
            failure_readback,
            keep_buffers,
            keep_textures,
            ops,
        ) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let Some(encoder) = state.encoder.take() else {
                state.failure_bytes.clear();
                state.failure_readback = None;
                return Ok((None, None));
            };
            let idx = state.ring_idx;
            let uniform_slab = state.uniform_slabs[idx].clone();
            let uniform_data = std::mem::take(&mut state.uniform_bytes);
            let upload_slab = state.upload_slabs[idx].clone();
            let upload_data = std::mem::take(&mut state.upload_bytes);
            let failure_slab = state.failure_slabs[idx].clone();
            let failure_data = std::mem::take(&mut state.failure_bytes);
            let failure_readback = state.failure_readback.take();
            let keep_buffers = std::mem::take(&mut state.keep_buffers);
            let keep_textures = std::mem::take(&mut state.keep_textures);
            let ops = state.ops;
            state.passes = 0;
            state.ops = 0;
            state.ring_idx = (idx + 1) % RING;
            (
                encoder,
                uniform_slab,
                uniform_data,
                upload_slab,
                upload_data,
                failure_slab,
                failure_data,
                failure_readback,
                keep_buffers,
                keep_textures,
                ops,
            )
        };
        let encoding = self.waiter_control.begin_encoding();
        let scopes = Scopes::push(&self.device);
        if !uniform_data.is_empty()
            && let Some(slab) = &uniform_slab
        {
            self.queue.write_buffer(slab, 0, &uniform_data);
        }
        if !upload_data.is_empty()
            && let Some(slab) = &upload_slab
        {
            self.queue.write_buffer(slab, 0, &upload_data);
        }
        let failure_copy = if !failure_data.is_empty()
            && let (Some(slab), Some(readback)) = (&failure_slab, &failure_readback)
        {
            self.queue.write_buffer(slab, 0, &failure_data);
            let failure_len = (failure_data.len() as u64).div_ceil(4) * 4;
            let staging = self.spare.take(
                &self.device,
                failure_len.max(crate::wgsl::FAILURE_BYTES),
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );
            encoder.copy_buffer_to_buffer(slab, 0, &staging, 0, failure_len);
            Some((Arc::clone(readback), staging, failure_len))
        } else {
            None
        };
        let submission = self.queue.submit(Some(encoder.finish()));
        if let Some((readback, staging, failure_len)) = failure_copy {
            let asked = self
                .wait
                .ask_map_now(&self.queue, staging.slice(..failure_len));
            readback.mark_in_flight(submission.clone(), staging, failure_len, asked);
        }
        self.waiter_control.note_activity();
        self.spare.bury();
        self.targets.bury();
        self.mirrors.bury();
        for buffer in keep_buffers {
            self.spare.keep(buffer, None);
        }
        for texture in keep_textures {
            self.targets.keep(texture);
        }
        if self.wait.parks_with_host() {
            self.page_queue.note(ops);
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.last_submission = Some(submission.clone());
            // Keep Vec allocations warm across frames for zero-alloc steady state.
            if state.uniform_bytes.capacity() < uniform_data.capacity() {
                let mut reused = uniform_data;
                reused.clear();
                state.uniform_bytes = reused;
            }
            if state.upload_bytes.capacity() < upload_data.capacity() {
                let mut reused = upload_data;
                reused.clear();
                state.upload_bytes = reused;
            }
            if state.failure_bytes.capacity() < failure_data.capacity() {
                let mut reused = failure_data;
                reused.clear();
                state.failure_bytes = reused;
            }
        }
        drop(encoding);
        if self.wait.parks_with_host() {
            let check = Box::new(scopes.pop_later(self.wait, self.spent.clone()));
            self.pending_checks
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(check);
        } else {
            scopes
                .pop(self.wait, "")
                .map_err(|error| ExecutionError::Device(error.said()))?;
        }
        Ok((Some(submission), failure_readback))
    }

    /// Flushes any open batch and returns the latest batch submission index.
    pub(crate) fn flush_for_wait(&self) -> Result<Option<wgpu::SubmissionIndex>, ExecutionError> {
        if let Some(sub) = self.flush()? {
            return Ok(Some(sub));
        }
        Ok(self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .last_submission
            .clone())
    }
}
