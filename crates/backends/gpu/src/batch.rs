//! Same-frame GPU command batching with per-frame uniform and small-upload arenas.
//!
//! Instead of creating a fresh `CommandEncoder`, pushing/popping three error
//! scopes, and submitting to the queue for every node in a frame, consecutive
//! GPU passes record into one shared encoder. All per-node uniform records
//! (`context`, `params`, rasterizer/prefix/reduce setup) pack into one
//! 256-byte-aligned uniform slab written once right before submission, and small
//! Host uploads (`<= 64 KiB`, such as per-frame poses, morph weights, and camera
//! lenses from JIT nodes) stage into one upload slab copied in exact pass order
//! via `copy_buffer_to_buffer`.
//!
//! Any observation boundary (`submitted`, `settle_device`, `end_frame`,
//! `download`, `download_later`, `same_resident`, `present`, `FailureCheck::wait`,
//! large Host uploads, measured single-node profiling, or WebGPU TDR pacing)
//! flushes the open batch cleanly so ordering and error scopes remain exact.

use std::sync::{Arc, Mutex};

use entelechy_rules::{Check, ExecutionError};

use crate::PageQueue;
use crate::failure::Scopes;
use crate::pool::{Spare, SpareTextures};
use crate::timing::Frames;
use crate::wait::{Spent, Wait};

/// Initial capacity of the per-frame uniform arena (`256 KiB` holds ~500 nodes).
const INITIAL_UNIFORM_SLAB: usize = 256 * 1024;
/// Initial capacity of the per-frame small-upload staging arena (`256 KiB`).
const INITIAL_UPLOAD_SLAB: usize = 256 * 1024;
/// Largest single Host upload staged through the in-batch copy slab (`64 KiB`).
/// Larger uploads (e.g. Frame-1 static meshes/textures) flush and stream directly.
const MAX_SMALL_UPLOAD: usize = 64 * 1024;
/// Maximum nodes accumulated in one encoder before flushing to keep driver
/// command buffers bounded.
const MAX_BATCH_PASSES: u32 = 128;

/// One uniform slice inside a buffer, bound via `wgpu::BufferBinding`.
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

    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

#[derive(Default)]
struct BatchState {
    encoder: Option<wgpu::CommandEncoder>,
    uniform_slab: Option<wgpu::Buffer>,
    uniform_bytes: Vec<u8>,
    upload_slab: Option<wgpu::Buffer>,
    upload_bytes: Vec<u8>,
    keep_buffers: Vec<wgpu::Buffer>,
    keep_textures: Vec<wgpu::Texture>,
    last_submission: Option<wgpu::SubmissionIndex>,
    passes: u32,
    ops: u64,
}

/// Shared same-frame command batcher held by `WgpuBackend`, `WgpuDisplay`, and
/// deferred `FailureCheck`s so every observation or present flushes pending GPU
/// passes first.
pub(crate) struct BatchHub {
    device: wgpu::Device,
    queue: wgpu::Queue,
    wait: Wait,
    spent: Spent,
    spare: Spare,
    targets: SpareTextures,
    page_queue: PageQueue,
    pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>>,
    uniform_align: usize,
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
        page_queue: PageQueue,
        pending_checks: Arc<Mutex<Vec<Box<dyn Check>>>>,
    ) -> Self {
        let uniform_align = (device.limits().min_uniform_buffer_offset_alignment as usize).max(256);
        Self {
            device,
            queue,
            wait,
            spent,
            spare,
            targets,
            page_queue,
            pending_checks,
            uniform_align,
            state: Mutex::new(BatchState::default()),
        }
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

    /// Allocates one aligned uniform slice inside the frame's uniform slab.
    /// Flushes any open batch first if the slab would overflow.
    pub(crate) fn alloc_uniform(&self, bytes: &[u8]) -> Result<UniformSlice, ExecutionError> {
        let padded = bytes.len().max(16).div_ceil(16) * 16;
        let align = self.uniform_align;
        let needs_flush = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.encoder.is_none() {
                false
            } else {
                let off = state.uniform_bytes.len().div_ceil(align) * align;
                state
                    .uniform_slab
                    .as_ref()
                    .is_some_and(|slab| (off + padded) as u64 > slab.size())
            }
        };
        if needs_flush {
            self.flush()?;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.encoder.is_none() {
            state.uniform_bytes.clear();
        }
        let off = state.uniform_bytes.len().div_ceil(align) * align;
        let needed = (off + padded) as u64;
        self.ensure_uniform_slab(&mut state, needed);
        state.uniform_bytes.resize(off + padded, 0);
        state.uniform_bytes[off..off + bytes.len()].copy_from_slice(bytes);
        let buffer = state
            .uniform_slab
            .as_ref()
            .expect("uniform slab initialized")
            .clone();
        Ok(UniformSlice {
            buffer,
            offset: off as u64,
            size: padded as u64,
        })
    }

    /// Allocates two aligned uniform slices (`context` and `params`) atomically
    /// inside the same uniform slab so neither can be split across a flush.
    pub(crate) fn alloc_uniform_pair(
        &self,
        first: &[u8],
        second: &[u8],
    ) -> Result<(UniformSlice, UniformSlice), ExecutionError> {
        let pad_a = first.len().max(16).div_ceil(16) * 16;
        let pad_b = second.len().max(16).div_ceil(16) * 16;
        let align = self.uniform_align;
        let needs_flush = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.encoder.is_none() {
                false
            } else {
                let off_a = state.uniform_bytes.len().div_ceil(align) * align;
                let off_b = (off_a + pad_a).div_ceil(align) * align;
                state
                    .uniform_slab
                    .as_ref()
                    .is_some_and(|slab| (off_b + pad_b) as u64 > slab.size())
            }
        };
        if needs_flush {
            self.flush()?;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.encoder.is_none() {
            state.uniform_bytes.clear();
        }
        let off_a = state.uniform_bytes.len().div_ceil(align) * align;
        let off_b = (off_a + pad_a).div_ceil(align) * align;
        let needed = (off_b + pad_b) as u64;
        self.ensure_uniform_slab(&mut state, needed);
        state.uniform_bytes.resize(off_b + pad_b, 0);
        state.uniform_bytes[off_a..off_a + first.len()].copy_from_slice(first);
        state.uniform_bytes[off_b..off_b + second.len()].copy_from_slice(second);
        let buffer = state
            .uniform_slab
            .as_ref()
            .expect("uniform slab initialized")
            .clone();
        Ok((
            UniformSlice {
                buffer: buffer.clone(),
                offset: off_a as u64,
                size: pad_a as u64,
            },
            UniformSlice {
                buffer,
                offset: off_b as u64,
                size: pad_b as u64,
            },
        ))
    }

    fn ensure_uniform_slab(&self, state: &mut BatchState, needed: u64) {
        let rebuild = state
            .uniform_slab
            .as_ref()
            .is_none_or(|slab| slab.size() < needed);
        if rebuild {
            if let Some(old) = state.uniform_slab.take() {
                self.spare.let_go(old);
            }
            let cap = (needed as usize)
                .next_power_of_two()
                .max(INITIAL_UNIFORM_SLAB);
            state.uniform_slab = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame uniforms"),
                size: cap as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            let off = state.upload_bytes.len().div_ceil(4) * 4;
            let cap = state
                .upload_slab
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
        if state.upload_slab.is_none() {
            state.upload_slab = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
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
        let slab = state
            .upload_slab
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

    /// Submits any open frame batch in ONE `queue.submit()`, uploading its
    /// staged uniforms and small inputs first, burying retired resources, and
    /// popping its shared error scope.
    /// Returns `Ok(Some(submission))` when an open batch was submitted, or
    /// `Ok(None)` when no batch was open.
    pub(crate) fn flush(&self) -> Result<Option<wgpu::SubmissionIndex>, ExecutionError> {
        let (
            encoder,
            uniform_slab,
            uniform_data,
            upload_slab,
            upload_data,
            keep_buffers,
            keep_textures,
            ops,
        ) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let Some(encoder) = state.encoder.take() else {
                return Ok(None);
            };
            let uniform_slab = state.uniform_slab.clone();
            let uniform_data = std::mem::take(&mut state.uniform_bytes);
            let upload_slab = state.upload_slab.clone();
            let upload_data = std::mem::take(&mut state.upload_bytes);
            let keep_buffers = std::mem::take(&mut state.keep_buffers);
            let keep_textures = std::mem::take(&mut state.keep_textures);
            let ops = state.ops;
            state.passes = 0;
            state.ops = 0;
            (
                encoder,
                uniform_slab,
                uniform_data,
                upload_slab,
                upload_data,
                keep_buffers,
                keep_textures,
                ops,
            )
        };
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
        let submission = self.queue.submit(Some(encoder.finish()));
        self.spare.bury();
        self.targets.bury();
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
        }
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
        Ok(Some(submission))
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
