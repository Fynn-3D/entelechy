//! Device memory kept between runs: bounded piles of buffers and textures,
//! and the image a run left resident.

use super::*;
use std::ops::Range;

/// A resident image: its pixels as a storage buffer of `vec4<f32>`.
pub(crate) struct DeviceImage {
    pub(crate) buffer: wgpu::Buffer,
    /// Where the buffer goes when its Field is released, for the next output
    /// of its size. `None` when the buffer is not the backend's to keep.
    pub(crate) spare: Option<Spare>,
    /// The words the buffer holds when it was uploaded from the Host,
    /// which it is kept marked with when its Field is released.
    pub(crate) holds: Option<Held>,
}

/// A buffer on the pile, what it holds where that is known, and the
/// evaluation it was kept in ([`Spare::evaluation_begun`]).
pub(crate) type Kept = (wgpu::Buffer, Option<Held>, u64);

/// What an uploaded buffer holds: a Field's words, and, where that Field
/// was made from other words, those and the storage words it may differ
/// from them on.
#[derive(Clone)]
pub(crate) struct Held {
    pub(crate) words: Version,
    pub(crate) made: Vec<MadeFrom>,
}

/// Buffers released Fields left behind, by size, for the next outputs of
/// those sizes, oldest first; one uploaded from the Host is marked with
/// the words it still holds.
#[derive(Clone, Default)]
pub(crate) struct Spare {
    pub(crate) held: Arc<Mutex<Vec<Kept>>>,
    /// The evaluations begun, by which what the pile keeps is aged.
    evaluations: Arc<std::sync::atomic::AtomicU64>,
    /// Buffers let go, let go of once the next run is given to the device
    /// ([`Spare::bury`]): a command of a run being made may still bind
    /// one. A page's are destroyed, not dropped: its device frees a buffer
    /// dropped only once the page's collector finds its object, which it
    /// may not look for in a long while ([`crate::wait::discard`]).
    gone: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

/// How many buffers the pile holds, and how many bytes; beyond that they
/// return to the driver. The byte bound holds a frame of ten million
/// triangles drawn at 4K.
pub(crate) const SPARE_BUFFERS: usize = 256;

pub(crate) const SPARE_BYTES: u64 = 1024 * 1024 * 1024;

/// The targets a rasterizer draws into, kept by shape between calls. Each
/// belongs to one call and comes back at its end.
#[derive(Clone, Default)]
pub(crate) struct SpareTextures {
    pub(crate) held: Arc<Mutex<Vec<wgpu::Texture>>>,
    /// Targets let go, let go of as [`Spare`]'s buffers are.
    gone: Arc<Mutex<Vec<wgpu::Texture>>>,
}

/// How many targets the pile holds, and how many bytes: three per shape,
/// for about two shapes.
pub(crate) const SPARE_TEXTURES: usize = 8;

pub(crate) const SPARE_TEXTURE_BYTES: u64 = 512 * 1024 * 1024;

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

    /// Keeps a target the call is done with, while the pile has room.
    pub(crate) fn keep(&self, texture: wgpu::Texture) {
        let bytes = |texture: &wgpu::Texture| -> u64 {
            let size = texture.format().target_pixel_byte_cost().unwrap_or(16) as u64;
            u64::from(texture.width()) * u64::from(texture.height()) * size
        };
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let held_bytes: u64 = held.iter().map(bytes).sum();
        if held.len() < SPARE_TEXTURES && held_bytes + bytes(&texture) <= SPARE_TEXTURE_BYTES {
            held.push(texture);
        } else {
            lock_gone(&self.gone).push(texture);
        }
    }

    /// The bytes of the targets kept.
    pub(crate) fn bytes(&self) -> u64 {
        let held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        held.iter()
            .map(|texture| {
                let size = texture.format().target_pixel_byte_cost().unwrap_or(16) as u64;
                u64::from(texture.width()) * u64::from(texture.height()) * size
            })
            .sum()
    }

    /// Lets go of the targets let go: called once a run is given to the
    /// device, which frees them when the work before has run
    /// ([`crate::wait::discard_texture`]).
    pub(crate) fn bury(&self) {
        for texture in lock_gone(&self.gone).drain(..) {
            crate::wait::discard_texture(texture);
        }
    }
}

fn lock_gone<T>(gone: &Mutex<Vec<T>>) -> std::sync::MutexGuard<'_, Vec<T>> {
    gone.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Spare {
    /// A buffer at least this size, from the pile or from the device: the
    /// smallest that fits, at most twice the size. Shaders read only up to the
    /// element count the uniform gives, so a larger buffer's tail is never
    /// touched, and Fields whose size varies between frames reuse buffers.
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
        // An unmarked buffer first, so marked ones wait for the Fields made
        // from theirs; of equals the last kept, so the few a frame needs at
        // once go round and the rest age out ([`Spare::evaluation_begun`]).
        if let Some(index) = held
            .iter()
            .enumerate()
            .filter(|(_, (buffer, _, _))| fits(buffer))
            .min_by_key(|(index, (buffer, holds, _))| {
                (holds.is_some(), buffer.size(), std::cmp::Reverse(*index))
            })
            .map(|(index, _)| index)
        {
            return held.remove(index).0;
        }
        drop(held);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Keeps a buffer its Field released, marked with the Field whose
    /// words it holds where it was uploaded; the oldest beyond the pile's
    /// bounds go.
    /// What goes is let go ([`Spare::let_go`]).
    pub(crate) fn keep(&self, buffer: wgpu::Buffer, holds: Option<Held>) {
        if buffer.size() > SPARE_BYTES {
            self.let_go(buffer);
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let now = self.evaluations.load(std::sync::atomic::Ordering::Relaxed);
        held.push((buffer, holds, now));
        let mut bytes: u64 = held.iter().map(|(buffer, _, _)| buffer.size()).sum();
        while held.len() > SPARE_BUFFERS || bytes > SPARE_BYTES {
            let (gone, _, _) = held.remove(0);
            bytes -= gone.size();
            lock_gone(&self.gone).push(gone);
        }
    }

    /// The bytes of the buffers kept.
    pub(crate) fn bytes(&self) -> u64 {
        let held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        held.iter().map(|(buffer, _, _)| buffer.size()).sum()
    }

    /// An evaluation begins: what was kept before the last one began and
    /// went untaken through it is let go ([`Spare::let_go`]), as a scene's
    /// frames ask for the same buffers frame after frame and those no
    /// frame took will not be asked for again soon.
    pub(crate) fn evaluation_begun(&self) {
        let now = self
            .evaluations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let mut gone = lock_gone(&self.gone);
        held.retain(|(buffer, _, kept)| {
            let fresh = kept + 1 >= now;
            if !fresh {
                gone.push(buffer.clone());
            }
            fresh
        });
    }

    /// A buffer nothing reads after the run being made: let go once that
    /// run is given to the device ([`Spare::bury`]).
    pub(crate) fn let_go(&self, buffer: wgpu::Buffer) {
        lock_gone(&self.gone).push(buffer);
    }

    /// Lets go of the buffers let go: called once a run is given to the
    /// device, which frees them when the work before has run
    /// ([`crate::wait::discard`]).
    pub(crate) fn bury(&self) {
        for buffer in lock_gone(&self.gone).drain(..) {
            crate::wait::discard(buffer);
        }
    }

    /// A buffer of `size` bytes or more holding words a Field made as
    /// `made` says differs from on the returned storage words alone: one
    /// holding words it was made from, written on the words it differs
    /// from them on; else one uploaded from a Field made from words it was
    /// made from too, written on the words either differs on. Of those,
    /// the one written on fewest words.
    pub(crate) fn take_made(
        &self,
        made: &[MadeFrom],
        size: u64,
    ) -> Option<(wgpu::Buffer, Vec<Range<usize>>)> {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let spans_for = |holds: &Held| -> Option<Vec<Range<usize>>> {
            if let Some(mine) = made.iter().find(|mine| mine.words == holds.words) {
                return Some(mine.changed.clone());
            }
            made.iter().find_map(|mine| {
                let theirs = holds
                    .made
                    .iter()
                    .find(|theirs| theirs.words == mine.words)?;
                let mut spans: Vec<Range<usize>> = mine
                    .changed
                    .iter()
                    .chain(&theirs.changed)
                    .cloned()
                    .collect();
                spans.sort_by_key(|span| (span.start, span.end));
                spans.dedup();
                Some(spans)
            })
        };
        let (index, spans) = held
            .iter()
            .enumerate()
            .filter(|(_, (buffer, _, _))| buffer.size() >= size)
            .filter_map(|(index, (_, holds, _))| Some((index, spans_for(holds.as_ref()?)?)))
            .min_by_key(|(_, spans)| spans.iter().map(ExactSizeIterator::len).sum::<usize>())?;
        Some((held.remove(index).0, spans))
    }
}

impl Drop for DeviceImage {
    fn drop(&mut self) {
        // The Field is gone, so nothing in the scene refers to the buffer; a
        // command that reads it was submitted before whatever writes it next,
        // and one queue runs in order.
        if let Some(spare) = self.spare.take() {
            spare.keep(self.buffer.clone(), self.holds.take());
        }
    }
}

impl GpuBackend {
    /// The pile of released Fields' buffers, for the parts of the backend that
    /// make their own outputs.
    pub(crate) fn spare(&self) -> Spare {
        self.spare.clone()
    }

    /// The pile of rasterizer targets.
    pub(crate) fn targets(&self) -> SpareTextures {
        self.targets.clone()
    }

    /// A buffer of this many bytes, rounded up, with nothing in it.
    pub(crate) fn empty(&self, label: &str, len: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
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

    /// [`GpuBackend::buffer`] read by the run being made alone: destroyed
    /// once that run is given to the device ([`Spare::let_go`]). Made while
    /// the run holds the backend's lock, so no other run's submission
    /// destroys it first.
    pub(crate) fn run_buffer(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = self.buffer(label, contents, usage);
        self.spare.let_go(buffer.clone());
        buffer
    }

    /// A scatter's target, `count` elements of `words` words, from the pile:
    /// it is copied over whole before anything reads it.
    pub(crate) fn target_buffer(&self, count: usize, words: usize) -> wgpu::Buffer {
        let size = (count * words * 4) as u64;
        self.spare.take(
            &self.device,
            size.max(16).div_ceil(16) * 16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
    }
}
