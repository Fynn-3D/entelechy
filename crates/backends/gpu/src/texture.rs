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

use entelechy_abi::{BlockFormat, Domain, Encoding};
use entelechy_ir::{Port, Shape, Type};
use entelechy_rules::ExecutionError;

use crate::batch::UniformSlice;
use crate::failure::Scopes;
use crate::pool::DeviceImage;
use crate::prepare::{storage_entry, texture_entry, uniform_entry};
use crate::{WgpuBackend, wgsl};

/// Maximum number of recycled mirror textures held in [`SpareMirrors`].
const SPARE_MIRRORS: usize = 64;
/// Maximum byte budget for recycled mirror textures (`512 MiB`).
const SPARE_MIRROR_BYTES: u64 = 512 * 1024 * 1024;
/// WebGPU `copy_buffer_to_texture` row alignment requirement.
const ROW_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Native `wgpu::TextureFormat`, view dimension, and bytes per texel for a
/// texture-eligible port (`Float`/`Vec2`/`Vec4` with `Word`/`F16`/`Unorm8`/`Snorm8`).
///
/// `Volume<... as unorm8>` uses [`wgpu::TextureViewDimension::D2Array`] so
/// streamed 2D texture stacks bind hardware block-compressed `Texture2DArray`
/// layers (`Bc7`/`Bc5`/`Bc4`/`Bc1`) natively without decompression.
pub(crate) fn format_for(
    port: &Port,
) -> Option<(wgpu::TextureFormat, wgpu::TextureViewDimension, u32)> {
    let encoding = port.ty.schema.uniform_encoding().unwrap_or(Encoding::Word);
    let view_dim = match port.shape() {
        Shape::Grid => wgpu::TextureViewDimension::D2,
        Shape::Volume if encoding == Encoding::Unorm8 => wgpu::TextureViewDimension::D2Array,
        Shape::Volume => wgpu::TextureViewDimension::D3,
        Shape::Rows => return None,
    };
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

/// Hardware `wgpu::TextureFormat` corresponding to a Host [`BlockFormat`].
pub(crate) fn wgpu_block_format(format: BlockFormat) -> wgpu::TextureFormat {
    match format {
        BlockFormat::Bc1RgbaUnorm => wgpu::TextureFormat::Bc1RgbaUnorm,
        BlockFormat::Bc4RUnorm => wgpu::TextureFormat::Bc4RUnorm,
        BlockFormat::Bc5RgUnorm => wgpu::TextureFormat::Bc5RgUnorm,
        BlockFormat::Bc7RgbaUnorm => wgpu::TextureFormat::Bc7RgbaUnorm,
    }
}

/// Resident hardware block-compressed 2D texture or 2D texture array owned by
/// a [`DeviceImage`], uploaded directly via `queue.write_texture` without any
/// uncompressed storage buffer.
pub(crate) struct CompressedTexture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) view_dim: wgpu::TextureViewDimension,
    pub(crate) format: BlockFormat,
    pub(crate) domain: Domain,
    pub(crate) bytes: u64,
    pub(crate) pool: SpareMirrors,
}

impl Drop for CompressedTexture {
    fn drop(&mut self) {
        self.pool.retire(self.texture.clone());
    }
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
    let w = u64::from(texture.width());
    let h = u64::from(texture.height());
    let d = u64::from(texture.depth_or_array_layers());
    match texture.format() {
        wgpu::TextureFormat::Bc1RgbaUnorm | wgpu::TextureFormat::Bc4RUnorm => {
            w.div_ceil(4) * h.div_ceil(4) * d * 8
        }
        wgpu::TextureFormat::Bc5RgUnorm | wgpu::TextureFormat::Bc7RgbaUnorm => {
            w.div_ceil(4) * h.div_ceil(4) * d * 16
        }
        wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::R8Snorm => w * h * d,
        wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Rg8Snorm
        | wgpu::TextureFormat::R16Float => w * h * d * 2,
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8Snorm
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::R32Float => w * h * d * 4,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg32Float => w * h * d * 8,
        wgpu::TextureFormat::Rgba32Float => w * h * d * 16,
        _ => w * h * d * 16,
    }
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

    pub(crate) fn retire(&self, texture: wgpu::Texture) {
        self.gone
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(texture);
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

/// Compute pipelines that unpack a hardware block-compressed `texture_2d<f32>`
/// or `texture_2d_array<f32>` directly into tightly packed `unorm8` `u32`
/// storage words on the GPU (used only on download, present, or non-textured
/// buffer fallback).
pub(crate) struct DecompressBcPipelines {
    pipe_2d: wgpu::ComputePipeline,
    layout_2d: wgpu::BindGroupLayout,
    pipe_array: wgpu::ComputePipeline,
    layout_array: wgpu::BindGroupLayout,
}

const DECOMPRESS_WORKGROUP: u32 = 64;

fn decompress_bc_wgsl(is_array: bool) -> &'static str {
    if is_array {
        "
struct Params {
    width: u32,
    height: u32,
    depth: u32,
    channels: u32,
    total_words: u32,
    fold_x: u32,
    _p0: u32,
    _p1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn byte_at(byte_idx: u32) -> u32 {
    let elem = byte_idx / p.channels;
    let ch = byte_idx - elem * p.channels;
    let plane = p.width * p.height;
    let z = elem / plane;
    let rem = elem - z * plane;
    let y = rem / p.width;
    let x = rem - y * p.width;
    let v = textureLoad(src, vec2<i32>(i32(x), i32(y)), i32(z), 0);
    let f = select(select(select(v.w, v.z, ch == 2u), v.y, ch == 1u), v.x, ch == 0u);
    return u32(clamp(round(f * 255.0), 0.0, 255.0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * p.fold_x * 64u;
    if (idx >= p.total_words) { return; }
    let total_bytes = p.width * p.height * p.depth * p.channels;
    let base = idx * 4u;
    var word = 0u;
    for (var b = 0u; b < 4u; b = b + 1u) {
        if (base + b < total_bytes) {
            word = word | (byte_at(base + b) << (b * 8u));
        }
    }
    dst[idx] = word;
}
"
    } else {
        "
struct Params {
    width: u32,
    height: u32,
    depth: u32,
    channels: u32,
    total_words: u32,
    fold_x: u32,
    _p0: u32,
    _p1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn byte_at(byte_idx: u32) -> u32 {
    let elem = byte_idx / p.channels;
    let ch = byte_idx - elem * p.channels;
    let y = elem / p.width;
    let x = elem - y * p.width;
    let v = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0);
    let f = select(select(select(v.w, v.z, ch == 2u), v.y, ch == 1u), v.x, ch == 0u);
    return u32(clamp(round(f * 255.0), 0.0, 255.0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * p.fold_x * 64u;
    if (idx >= p.total_words) { return; }
    let total_bytes = p.width * p.height * p.depth * p.channels;
    let base = idx * 4u;
    var word = 0u;
    for (var b = 0u; b < 4u; b = b + 1u) {
        if (base + b < total_bytes) {
            word = word | (byte_at(base + b) << (b * 8u));
        }
    }
    dst[idx] = word;
}
"
    }
}

impl WgpuBackend {
    pub(crate) fn decompress_bc_pipelines(&self) -> Result<&DecompressBcPipelines, ExecutionError> {
        let built = self.decompress_bc.get_or_init(|| {
            let make =
                |is_array: bool| -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), String> {
                    let scopes = Scopes::push(&self.device);
                    let dim = if is_array {
                        wgpu::TextureViewDimension::D2Array
                    } else {
                        wgpu::TextureViewDimension::D2
                    };
                    let entries = [
                        uniform_entry(0),
                        texture_entry(1, dim),
                        storage_entry(2, false),
                    ];
                    let layout =
                        self.device
                            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: Some("entelechy bc unpack"),
                                entries: &entries,
                            });
                    let source = decompress_bc_wgsl(is_array);
                    let shader = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("entelechy bc unpack"),
                            source: wgpu::ShaderSource::Wgsl(source.into()),
                        });
                    let pipeline_layout =
                        self.device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("entelechy bc unpack"),
                                bind_group_layouts: &[Some(&layout)],
                                immediate_size: 0,
                            });
                    let pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("entelechy bc unpack"),
                                layout: Some(&pipeline_layout),
                                module: &shader,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });
                    scopes.pop(self.wait, source).map_err(|e| e.said())?;
                    Ok((pipeline, layout))
                };
            let (pipe_2d, layout_2d) = make(false)?;
            let (pipe_array, layout_array) = make(true)?;
            Ok(DecompressBcPipelines {
                pipe_2d,
                layout_2d,
                pipe_array,
                layout_array,
            })
        });
        built
            .as_ref()
            .map_err(|e| ExecutionError::Device(e.clone()))
    }

    /// Uploads raw hardware-compressed 4x4 BC1/BC4/BC5/BC7 blocks directly
    /// into a native `wgpu::Texture` (2D or 2D array) without allocating an
    /// uncompressed storage buffer.
    pub(crate) fn create_compressed_texture(
        &self,
        domain: Domain,
        format: BlockFormat,
        bytes: &[u8],
    ) -> Result<CompressedTexture, ExecutionError> {
        let w = domain.width();
        let h = domain.height();
        let d = domain.depth();
        let tex_w = w.div_ceil(4) * 4;
        let tex_h = h.div_ceil(4) * 4;
        let limits = self.device.limits();
        if tex_w > limits.max_texture_dimension_2d || tex_h > limits.max_texture_dimension_2d {
            return Err(ExecutionError::Device(format!(
                "compressed texture extent {w}x{h} exceeds max 2D texture dimension {}",
                limits.max_texture_dimension_2d
            )));
        }
        if d > limits.max_texture_array_layers {
            return Err(ExecutionError::Device(format!(
                "compressed texture layer count {d} exceeds max array layers {}",
                limits.max_texture_array_layers
            )));
        }
        let wgpu_format = wgpu_block_format(format);
        let view_dim = match domain.shape() {
            Shape::Grid => wgpu::TextureViewDimension::D2,
            Shape::Volume => wgpu::TextureViewDimension::D2Array,
            Shape::Rows => {
                return Err(ExecutionError::Device(
                    "compressed textures require a Grid or Volume".to_owned(),
                ));
            }
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("entelechy compressed texture"),
            size: wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: d,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let row_bytes = (tex_w / 4) * format.block_bytes() as u32;
        let block_rows = tex_h / 4;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: (d > 1).then_some(block_rows),
            },
            wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: d,
            },
        );
        self.uploaded
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dim),
            ..Default::default()
        });
        Ok(CompressedTexture {
            texture,
            view,
            view_dim,
            format,
            domain,
            bytes: bytes.len() as u64,
            pool: self.mirrors.clone(),
        })
    }

    /// Joins whole layers of hardware block-compressed textures into a single
    /// resident `Texture2DArray` (`Domain::Volume`) via GPU-to-GPU
    /// `copy_texture_to_texture` with zero decompression.
    ///
    /// Returns `Ok(None)` when any part is uncompressed, mixes different block
    /// formats, or is not aligned to whole 2D layers, allowing [`Backend::join`]
    /// to fall back seamlessly to buffer concatenation.
    pub(crate) fn try_join_compressed_layers(
        &self,
        parts: &[(entelechy_abi::Handle, std::ops::Range<usize>)],
        schema: &entelechy_abi::Schema,
        domain: Domain,
        scope: &mut entelechy_abi::Scope<'_>,
    ) -> Result<Option<entelechy_abi::Handle>, ExecutionError> {
        let Domain::Volume(w, h, total_layers) = domain else {
            return Ok(None);
        };
        let Some(layer_elems) = (w as usize).checked_mul(h as usize).filter(|&n| n > 0) else {
            return Ok(None);
        };
        let tex_w = w.div_ceil(4) * 4;
        let tex_h = h.div_ceil(4) * 4;
        let limits = self.device.limits();
        if tex_w > limits.max_texture_dimension_2d
            || tex_h > limits.max_texture_dimension_2d
            || total_layers > limits.max_texture_array_layers
        {
            return Ok(None);
        }

        // Verify all parts are whole-layer slices of the same BlockFormat
        // before uploading or allocating any device texture.
        let mut common_format: Option<BlockFormat> = None;
        let mut sum_layers = 0u32;
        for (handle, range) in parts {
            if range.is_empty()
                || !range.start.is_multiple_of(layer_elems)
                || !range.len().is_multiple_of(layer_elems)
            {
                return Ok(None);
            }
            let part_domain = scope.size(*handle)?;
            if part_domain.width() != w
                || part_domain.height() != h
                || !matches!(part_domain.shape(), Shape::Grid | Shape::Volume)
            {
                return Ok(None);
            }
            let part_fmt = if let Some(dev) = scope.resident(*handle)? {
                let Some(ct) = dev
                    .downcast_ref::<DeviceImage>()
                    .and_then(|img| img.compressed.as_ref())
                else {
                    return Ok(None);
                };
                ct.format
            } else if let Some(blocks) = scope.blocks(*handle)? {
                blocks.format()
            } else {
                return Ok(None);
            };
            match common_format {
                Some(prev) if prev != part_fmt => return Ok(None),
                None => common_format = Some(part_fmt),
                _ => {}
            }
            let layers = u32::try_from(range.len() / layer_elems).unwrap_or(0);
            sum_layers = sum_layers.saturating_add(layers);
        }
        let Some(format) = common_format else {
            return Ok(None);
        };
        if sum_layers != total_layers {
            return Ok(None);
        }
        let Some(total_bytes_usize) = format.byte_len(domain) else {
            return Ok(None);
        };
        let total_bytes = total_bytes_usize as u64;

        let mut copies: Vec<(wgpu::Texture, u32, u32)> = Vec::with_capacity(parts.len());
        for (handle, range) in parts {
            let _ = self.upload_compressed(scope, *handle)?;
            let texture = scope
                .resident(*handle)?
                .and_then(|dev| dev.downcast_ref::<DeviceImage>())
                .and_then(|img| img.compressed.as_ref())
                .map(|ct| ct.texture.clone())
                .ok_or_else(|| {
                    ExecutionError::Device(
                        "compressed layer failed to become resident on GPU".to_owned(),
                    )
                })?;
            let src_layer = (range.start / layer_elems) as u32;
            let layer_count = (range.len() / layer_elems) as u32;
            copies.push((texture, src_layer, layer_count));
        }

        let wgpu_format = wgpu_block_format(format);
        let dst = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("entelechy compressed array"),
            size: wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let batched = self.batch.can_batch(self.measured(), &self.frames);
        if !batched {
            self.batch.flush()?;
        }
        let mut encoder = if batched {
            self.batch.take_or_open()
        } else {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("entelechy compressed join"),
                })
        };
        let mut dst_layer = 0u32;
        for (src_tex, src_layer, layer_count) in &copies {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: src_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: *src_layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: dst_layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: *layer_count,
                },
            );
            dst_layer += *layer_count;
        }
        if batched {
            self.frames.note_batched_run();
            self.batch
                .finish_node(encoder, 0, Vec::new(), Vec::new(), None)?;
        } else {
            self.submit(encoder.finish());
        }

        let view = dst.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let ct = CompressedTexture {
            texture: dst,
            view,
            view_dim: wgpu::TextureViewDimension::D2Array,
            format,
            domain,
            bytes: total_bytes,
            pool: self.mirrors.clone(),
        };
        let handle = scope.adopt_resident_bytes(
            schema.clone(),
            domain,
            total_bytes,
            Box::new(DeviceImage::compressed(ct, Some(self.spare.clone()))),
        )?;
        Ok(Some(handle))
    }

    /// Unpacks a resident hardware-compressed texture into a `unorm8` storage
    /// buffer on the GPU (only called for download, present, or non-textured
    /// buffer access).
    pub(crate) fn decompress_compressed_to_buffer(
        &self,
        ct: &CompressedTexture,
    ) -> Result<wgpu::Buffer, ExecutionError> {
        let pipes = self.decompress_bc_pipelines()?;
        let is_array = ct.view_dim == wgpu::TextureViewDimension::D2Array;
        let (pipeline, layout) = if is_array {
            (&pipes.pipe_array, &pipes.layout_array)
        } else {
            (&pipes.pipe_2d, &pipes.layout_2d)
        };
        let w = ct.domain.width();
        let h = ct.domain.height();
        let d = ct.domain.depth();
        let ch = ct.format.channels() as u32;
        let total_bytes = (w as usize) * (h as usize) * (d as usize) * (ch as usize);
        let total_words = total_bytes.div_ceil(4) as u32;
        let byte_size = (u64::from(total_words) * 4).max(16).div_ceil(16) * 16;
        let dst = self.spare.take(
            &self.device,
            byte_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let groups = total_words.div_ceil(DECOMPRESS_WORKGROUP).max(1);
        let (gx, gy) = wgsl::folded(groups);
        let mut params = [0u8; 32];
        params[0..4].copy_from_slice(&w.to_le_bytes());
        params[4..8].copy_from_slice(&h.to_le_bytes());
        params[8..12].copy_from_slice(&d.to_le_bytes());
        params[12..16].copy_from_slice(&ch.to_le_bytes());
        params[16..20].copy_from_slice(&total_words.to_le_bytes());
        params[20..24].copy_from_slice(&gx.to_le_bytes());
        let uniform = self.take_uniform(&params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy bc unpack"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ct.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
            ],
        });
        let batched = self.batch.can_batch(self.measured(), &self.frames);
        if !batched {
            self.batch.flush()?;
        }
        let mut encoder = if batched {
            self.batch.take_or_open()
        } else {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("entelechy bc unpack"),
                })
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy bc unpack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if batched {
            self.frames.note_batched_run();
            self.batch
                .finish_node(encoder, 0, vec![uniform], Vec::new(), None)?;
        } else {
            self.submit(encoder.finish());
            self.spare.keep(uniform, None);
        }
        Ok(dst)
    }

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
        if let Some(ct) = image.compressed.as_ref()
            && ct.view_dim == view_dim
        {
            return Ok((ct.view.clone(), None));
        }
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
            wgpu::TextureViewDimension::D2Array => {
                if width > limits.max_texture_dimension_2d
                    || height > limits.max_texture_dimension_2d
                    || depth > limits.max_texture_array_layers
                {
                    return Err(ExecutionError::Device(format!(
                        "input `{}` extent {width}x{height}x{depth} exceeds max 2D array limits ({} / {} layers)",
                        port.name, limits.max_texture_dimension_2d, limits.max_texture_array_layers
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
