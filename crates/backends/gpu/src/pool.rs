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

/// A buffer on the pile, and what it holds where that is known.
pub(crate) type Kept = (wgpu::Buffer, Option<Held>);

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
}

/// How many buffers the pile holds, and how many bytes; beyond that they
/// return to the driver. The byte bound holds a frame of ten million
/// triangles drawn at 4K.
pub(crate) const SPARE_BUFFERS: usize = 12;

pub(crate) const SPARE_BYTES: u64 = 1024 * 1024 * 1024;

/// The targets a rasterizer draws into, kept by shape between calls. Each
/// belongs to one call and comes back at its end.
#[derive(Clone, Default)]
pub(crate) struct SpareTextures {
    pub(crate) held: Arc<Mutex<Vec<wgpu::Texture>>>,
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
        }
    }
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
        // from theirs.
        if let Some(index) = held
            .iter()
            .enumerate()
            .filter(|(_, (buffer, _))| fits(buffer))
            .min_by_key(|(_, (buffer, holds))| (holds.is_some(), buffer.size()))
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
    pub(crate) fn keep(&self, buffer: wgpu::Buffer, holds: Option<Held>) {
        if buffer.size() > SPARE_BYTES {
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        held.push((buffer, holds));
        let mut bytes: u64 = held.iter().map(|(buffer, _)| buffer.size()).sum();
        while held.len() > SPARE_BUFFERS || bytes > SPARE_BYTES {
            bytes -= held.remove(0).0.size();
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
            .filter(|(_, (buffer, _))| buffer.size() >= size)
            .filter_map(|(index, (_, holds))| Some((index, spans_for(holds.as_ref()?)?)))
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
