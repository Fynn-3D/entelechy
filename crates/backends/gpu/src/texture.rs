//! Lazy hardware 2D/3D `wgpu::Texture` mirroring for read-only `Grid` and
//! `Volume` Fields sampled via `Op::AtLinear` (or 3D `Op::At`).
//!
//! Logical Field identity remains the resident storage buffer (`DeviceImage`),
//! matching `TYPED-FIELD-ARCHITECTURE.md` §2. On first textured read, a
//! `wgpu::Texture` mirror is populated via `copy_buffer_to_texture` (with a
//! lightweight GPU row-padding compute pass whenever `bytes_per_row` is not a
//! multiple of 256) and cached on the immutable `DeviceImage` for zero-copy
//! reuse across all subsequent frames.

use std::sync::{Arc, Mutex};

use entelechy_abi::{Domain, Encoding};
use entelechy_ir::{Port, Shape, Type};
use entelechy_rules::ExecutionError;

use crate::batch::UniformSlice;
use crate::failure::Scopes;
use crate::pool::DeviceImage;
use crate::prepare::{storage_entry, uniform_entry};
use crate::{WgpuBackend, wgsl};

/// Maximum number of recycled mirror textures held in [`SpareMirrors`].
const SPARE_MIRRORS: usize = 64;
/// Maximum byte budget for recycled mirror textures (`512 MiB`).
const SPARE_MIRROR_BYTES: u64 = 512 * 1024 * 1024;
/// WebGPU `copy_buffer_to_texture` row alignment requirement.
const ROW_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Native `wgpu::TextureFormat`, view dimension, and bytes per texel for a
/// texture-eligible port (`Float`/`Vec2`/`Vec4` with `Word`/`F16`/`Unorm8`/`Snorm8`).
pub(crate) fn format_for(
    port: &Port,
) -> Option<(wgpu::TextureFormat, wgpu::TextureViewDimension, u32)> {
    let view_dim = match port.shape() {
        Shape::Grid => wgpu::TextureViewDimension::D2,
        Shape::Volume => wgpu::TextureViewDimension::D3,
        Shape::Rows => return None,
    };
    let encoding = port.ty.schema.uniform_encoding().unwrap_or(Encoding::Word);
    let (format, texel_bytes) = match (port.element(), encoding) {
        (Type::Float, Encoding::Word) => (wgpu::TextureFormat::R32Float, 4),
        (Type::Vec2, Encoding::Word) => (wgpu::TextureFormat::Rg32Float, 8),
        (Type::Vec4, Encoding::Word) => (wgpu::TextureFormat::Rgba32Float, 16),
        (Type::Float, Encoding::F16) => (wgpu::TextureFormat::R16Float, 2),
        (Type::Vec2, Encoding::F16) => (wgpu::TextureFormat::Rg16Float, 4),
        (Type::Vec4, Encoding::F16) => (wgpu::TextureFormat::Rgba16Float, 8),
        (Type::Float, Encoding::Unorm8) => (wgpu::TextureFormat::R8Unorm, 1),
        (Type::Vec2, Encoding::Unorm8) => (wgpu::TextureFormat::Rg8Unorm, 2),
        (Type::Vec4, Encoding::Unorm8) => (wgpu::TextureFormat::Rgba8Unorm, 4),
        (Type::Float, Encoding::Snorm8) => (wgpu::TextureFormat::R8Snorm, 1),
        (Type::Vec2, Encoding::Snorm8) => (wgpu::TextureFormat::Rg8Snorm, 2),
        (Type::Vec4, Encoding::Snorm8) => (wgpu::TextureFormat::Rgba8Snorm, 4),
        _ => return None,
    };
    Some((format, view_dim, texel_bytes))
}

/// Hardware 2D/3D texture mirror cached on an immutable [`DeviceImage`].
pub(crate) struct TextureMirror {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) dims: (u32, u32, u32),
    pub(crate) dimension: wgpu::TextureDimension,
    pub(crate) pool: SpareMirrors,
}

impl Drop for TextureMirror {
    fn drop(&mut self) {
        self.pool.keep(self.texture.clone());
    }
}

/// Recycled 2D/3D texture mirrors aged across frame evaluations and destroyed
/// only after the submission referencing them (`bury`).
#[derive(Clone, Default)]
pub(crate) struct SpareMirrors {
    held: Arc<Mutex<Vec<(wgpu::Texture, u64)>>>,
    evaluations: Arc<std::sync::atomic::AtomicU64>,
    gone: Arc<Mutex<Vec<wgpu::Texture>>>,
}

fn mirror_bytes(texture: &wgpu::Texture) -> u64 {
    let bpp = match texture.format() {
        wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::R8Snorm => 1,
        wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Rg8Snorm
        | wgpu::TextureFormat::R16Float => 2,
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8Snorm
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::R32Float => 4,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg32Float => 8,
        wgpu::TextureFormat::Rgba32Float => 16,
        _ => 16,
    };
    u64::from(texture.width())
        * u64::from(texture.height())
        * u64::from(texture.depth_or_array_layers())
        * bpp
}

impl SpareMirrors {
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = held.iter().rposition(|(tex, _)| {
            tex.width() == width
                && tex.height() == height
                && tex.depth_or_array_layers() == depth
                && tex.dimension() == dimension
                && tex.format() == format
        }) {
            return held.swap_remove(index).0;
        }
        drop(held);
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("entelechy texture mirror"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub(crate) fn keep(&self, texture: wgpu::Texture) {
        let size = mirror_bytes(&texture);
        if size > SPARE_MIRROR_BYTES {
            self.gone
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(texture);
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let now = self.evaluations.load(std::sync::atomic::Ordering::Relaxed);
        held.push((texture, now));
        let mut total: u64 = held.iter().map(|(t, _)| mirror_bytes(t)).sum();
        let mut gone = self.gone.lock().unwrap_or_else(|p| p.into_inner());
        while held.len() > SPARE_MIRRORS || total > SPARE_MIRROR_BYTES {
            let (evicted, _) = held.remove(0);
            total -= mirror_bytes(&evicted);
            gone.push(evicted);
        }
    }

    pub(crate) fn evaluation_begun(&self) {
        let now = self
            .evaluations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let mut gone = self.gone.lock().unwrap_or_else(|p| p.into_inner());
        held.retain(|(tex, kept)| {
            let fresh = kept + 1 >= now;
            if !fresh {
                gone.push(tex.clone());
            }
            fresh
        });
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.held
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|(t, _)| mirror_bytes(t))
            .sum()
    }

    pub(crate) fn bury(&self) {
        for texture in self
            .gone
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .drain(..)
        {
            crate::wait::discard_texture(texture);
        }
    }
}

/// Compute pipeline that packs tightly-stored buffer rows into a 256-byte-row-
/// aligned staging buffer so `copy_buffer_to_texture` works for any width and
/// any texture format (including narrow 1-byte and 2-byte formats with odd row
/// widths).
pub(crate) struct RowPadPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

const ROW_PAD_WORKGROUP: u32 = 64;

const ROW_PAD_WGSL: &str = "
struct Params {
    row_bytes: u32,
    padded_row_words: u32,
    total_rows: u32,
    fold_x: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_words = (p.row_bytes + 3u) >> 2u;
    let total_words = row_words * p.total_rows;
    let idx = gid.x + gid.y * p.fold_x * 64u;
    if (idx >= total_words) {
        return;
    }
    let row = idx / row_words;
    let w = idx - row * row_words;
    var value = 0u;
    if ((p.row_bytes & 3u) == 0u) {
        value = src[row * row_words + w];
    } else {
        let row_base = row * p.row_bytes;
        let col_base = w * 4u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            if (col_base + b < p.row_bytes) {
                let sb = row_base + col_base + b;
                let byte_val = (src[sb >> 2u] >> ((sb & 3u) * 8u)) & 0xFFu;
                value = value | (byte_val << (b * 8u));
            }
        }
    }
    dst[row * p.padded_row_words + w] = value;
}
";

/// A buffer-to-texture mirror copy queued to run right before a compute pass.
pub(crate) struct PendingMirrorCopy {
    pub(crate) src: wgpu::Buffer,
    pub(crate) dst: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) row_bytes: u32,
    pub(crate) padded: Option<(wgpu::Buffer, UniformSlice, u32, u32)>,
}

/// Parameters needed to populate one uncached texture mirror before encoding.
pub(crate) struct MirrorPlan {
    pub(crate) input_index: usize,
    pub(crate) handle: entelechy_abi::Handle,
    pub(crate) mirror: TextureMirror,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) row_bytes: u32,
    pub(crate) pad_params: Option<[u8; 16]>,
    pub(crate) padded_row_bytes: u32,
    pub(crate) fold_x: u32,
}

impl WgpuBackend {
    fn row_pad_pipeline(&self) -> Result<&RowPadPipeline, ExecutionError> {
        let built = self.row_pad.get_or_init(|| {
            let scopes = Scopes::push(&self.device);
            let entries = [
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ];
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("entelechy row pad"),
                    entries: &entries,
                });
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("entelechy row pad"),
                    source: wgpu::ShaderSource::Wgsl(ROW_PAD_WGSL.into()),
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("entelechy row pad"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("entelechy row pad"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            scopes.pop(self.wait, ROW_PAD_WGSL).map_err(|e| e.said())?;
            Ok(RowPadPipeline { pipeline, layout })
        });
        built
            .as_ref()
            .map_err(|e| ExecutionError::Device(e.clone()))
    }

    /// Checks whether a resident [`DeviceImage`] already holds an up-to-date
    /// hardware texture mirror; otherwise allocates a recycled/fresh texture
    /// and returns a [`MirrorPlan`] to copy its pixels into the texture.
    pub(crate) fn plan_texture_mirror(
        &self,
        image: &DeviceImage,
        handle: entelechy_abi::Handle,
        port: &Port,
        extent: Domain,
        input_index: usize,
    ) -> Result<(wgpu::TextureView, Option<MirrorPlan>), ExecutionError> {
        let (format, view_dim, texel_bytes) = format_for(port).ok_or_else(|| {
            ExecutionError::Device(format!(
                "port `{}` has no hardware texture format",
                port.name
            ))
        })?;
        let (width, height, depth) = (extent.width(), extent.height(), extent.depth());
        let limits = self.device.limits();
        let dimension = match view_dim {
            wgpu::TextureViewDimension::D2 => {
                if width > limits.max_texture_dimension_2d
                    || height > limits.max_texture_dimension_2d
                {
                    return Err(ExecutionError::Device(format!(
                        "input `{}` extent {width}x{height} exceeds max 2D texture dimension {}",
                        port.name, limits.max_texture_dimension_2d
                    )));
                }
                wgpu::TextureDimension::D2
            }
            wgpu::TextureViewDimension::D3 => {
                if width > limits.max_texture_dimension_3d
                    || height > limits.max_texture_dimension_3d
                    || depth > limits.max_texture_dimension_3d
                {
                    return Err(ExecutionError::Device(format!(
                        "input `{}` extent {width}x{height}x{depth} exceeds max 3D texture dimension {}",
                        port.name, limits.max_texture_dimension_3d
                    )));
                }
                wgpu::TextureDimension::D3
            }
            _ => unreachable!(),
        };
        if let Some(existing) = image
            .mirror
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            && existing.format == format
            && existing.dims == (width, height, depth)
            && existing.dimension == dimension
        {
            return Ok((existing.view.clone(), None));
        }
        let texture = self
            .mirrors
            .take(&self.device, width, height, depth, dimension, format);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dim),
            ..Default::default()
        });
        let row_bytes = width.checked_mul(texel_bytes).ok_or_else(|| {
            ExecutionError::Device("texture mirror row byte size overflow".to_owned())
        })?;
        let direct = row_bytes.is_multiple_of(ROW_ALIGN) || (height == 1 && depth == 1);
        let (pad_params, padded_row_bytes, fold_x) = if direct {
            (None, row_bytes, 1)
        } else {
            let padded_row_bytes = row_bytes.div_ceil(ROW_ALIGN) * ROW_ALIGN;
            let padded_row_words = padded_row_bytes / 4;
            let total_rows = height * depth;
            let total_words = row_bytes.div_ceil(4).saturating_mul(total_rows);
            let groups = total_words.div_ceil(ROW_PAD_WORKGROUP).max(1);
            let (gx, _) = wgsl::folded(groups);
            let mut bytes = [0u8; 16];
            bytes[0..4].copy_from_slice(&row_bytes.to_le_bytes());
            bytes[4..8].copy_from_slice(&padded_row_words.to_le_bytes());
            bytes[8..12].copy_from_slice(&total_rows.to_le_bytes());
            bytes[12..16].copy_from_slice(&gx.to_le_bytes());
            (Some(bytes), padded_row_bytes, gx)
        };
        let mirror = TextureMirror {
            texture,
            view: view.clone(),
            format,
            dims: (width, height, depth),
            dimension,
            pool: self.mirrors.clone(),
        };
        Ok((
            view,
            Some(MirrorPlan {
                input_index,
                handle,
                mirror,
                width,
                height,
                depth,
                row_bytes,
                pad_params,
                padded_row_bytes,
                fold_x,
            }),
        ))
    }

    /// Encodes a queued buffer-to-texture copy into `encoder` before the
    /// compute pass that samples the texture.
    pub(crate) fn encode_mirror_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        copy: &PendingMirrorCopy,
    ) -> Result<(), ExecutionError> {
        let extent = wgpu::Extent3d {
            width: copy.width,
            height: copy.height,
            depth_or_array_layers: copy.depth,
        };
        if let Some((padded, params_slice, padded_row_bytes, fold_x)) = &copy.padded {
            let pad = self.row_pad_pipeline()?;
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy row pad"),
                layout: &pad.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_slice.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: copy.src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: padded.as_entire_binding(),
                    },
                ],
            });
            let total_rows = copy.height * copy.depth;
            let total_words = copy.row_bytes.div_ceil(4).saturating_mul(total_rows);
            let groups = total_words.div_ceil(ROW_PAD_WORKGROUP).max(1);
            let gy = groups.div_ceil(*fold_x);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("entelechy row pad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pad.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(*fold_x, gy, 1);
            }
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: padded,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(*padded_row_bytes),
                        rows_per_image: (copy.depth > 1).then_some(copy.height),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &copy.dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        } else {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &copy.src,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: (copy.height > 1 || copy.depth > 1)
                            .then_some(copy.row_bytes),
                        rows_per_image: (copy.depth > 1).then_some(copy.height),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &copy.dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        }
        Ok(())
    }
}
