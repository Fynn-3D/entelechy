//! Device memory kept between runs: bounded piles of buffers and textures,
//! and the image a run left resident.

use super::*;

/// A resident image: its pixels as a storage buffer of `vec4<f32>`.
pub(crate) struct DeviceImage {
    pub(crate) buffer: wgpu::Buffer,
    /// Where the buffer goes when its Field is released, for the next output
    /// of its size. `None` when the buffer is not the backend's to keep.
    pub(crate) spare: Option<Spare>,
}

/// Buffers released Fields left behind, by size, for the next outputs of
/// those sizes.
#[derive(Clone, Default)]
pub(crate) struct Spare {
    pub(crate) held: Arc<Mutex<Vec<wgpu::Buffer>>>,
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

    /// Keeps a buffer its Field released, while the pile has room.
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
        // The Field is gone, so nothing in the scene refers to the buffer; a
        // command that reads it was submitted before whatever writes it next,
        // and one queue runs in order.
        if let Some(spare) = self.spare.take() {
            spare.keep(self.buffer.clone());
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
