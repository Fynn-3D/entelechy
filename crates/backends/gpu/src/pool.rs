//! What the backend keeps on the device between runs so that a run
//! need not ask the device for memory: buffers and textures laid by,
//! each pile bounded, and the image a run left resident.

use super::*;

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
    pub(crate) held: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

/// How many buffers the pile holds, and how much of it. A frame wants
/// a handful of sizes; beyond that they are the driver's again. Ten
/// million triangles drawn at 4K pass half a gigabyte of them a frame,
/// and a pile smaller than that makes some anew every frame.
pub(crate) const SPARE_BUFFERS: usize = 12;

pub(crate) const SPARE_BYTES: u64 = 1024 * 1024 * 1024;

/// The targets a rasterizer draws into, kept by shape between calls.
/// They belong to the call and to nothing else, so they go back at the
/// end of it; asking the driver for tens of megabytes of texture is
/// most of what a pass of the rasterizer costs.
#[derive(Clone, Default)]
pub(crate) struct SpareTextures {
    pub(crate) held: Arc<Mutex<Vec<wgpu::Texture>>>,
}

/// How many targets the pile holds, and how much of it: a rasterizer
/// wants three of a shape, and a scene rarely rasterizes into more than
/// two shapes. Beyond that they are the driver's again.
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

impl GpuBackend {
    /// The pile of buffers released Fields left behind, for the parts
    /// of the backend that make outputs of their own.
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

    /// A scatter's target, `count` elements of `words` words. It is
    /// copied over whole before anything reads it, so it may come from
    /// the pile.
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
