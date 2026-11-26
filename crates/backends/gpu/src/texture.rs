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
/// `Volume<... as unorm8>` (and any `Volume` sampled with 2D gradients when
/// `force_2d_array` is true) uses [`wgpu::TextureViewDimension::D2Array`] so
/// streamed 2D texture stacks and anisotropic 2D arrays bind natively.
pub(crate) fn format_for(
    port: &Port,
    force_2d_array: bool,
) -> Option<(wgpu::TextureFormat, wgpu::TextureViewDimension, u32)> {
    let encoding = port.ty.schema.uniform_encoding().unwrap_or(Encoding::Word);
    let view_dim = match port.shape() {
        Shape::Grid => wgpu::TextureViewDimension::D2,
        Shape::Volume if encoding == Encoding::Unorm8 || force_2d_array => {
            wgpu::TextureViewDimension::D2Array
        }
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

fn direct_storage_wgsl(format: wgpu::TextureFormat) -> Option<&'static str> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => Some("rgba8unorm"),
        wgpu::TextureFormat::Rgba8Snorm => Some("rgba8snorm"),
        wgpu::TextureFormat::Rgba16Float => Some("rgba16float"),
        wgpu::TextureFormat::Rgba32Float => Some("rgba32float"),
        wgpu::TextureFormat::R32Float => Some("r32float"),
        _ => None,
    }
}

fn format_meta(format: wgpu::TextureFormat) -> (u32, u32, u32) {
    // Returns (channels, encoding_code, texel_bytes).
    // encoding_code: 0 = f32, 1 = f16, 2 = unorm8, 3 = snorm8.
    match format {
        wgpu::TextureFormat::R32Float => (1, 0, 4),
        wgpu::TextureFormat::Rg32Float => (2, 0, 8),
        wgpu::TextureFormat::Rgba32Float => (4, 0, 16),
        wgpu::TextureFormat::R16Float => (1, 1, 2),
        wgpu::TextureFormat::Rg16Float => (2, 1, 4),
        wgpu::TextureFormat::Rgba16Float => (4, 1, 8),
        wgpu::TextureFormat::R8Unorm => (1, 2, 1),
        wgpu::TextureFormat::Rg8Unorm => (2, 2, 2),
        wgpu::TextureFormat::Rgba8Unorm => (4, 2, 4),
        wgpu::TextureFormat::R8Snorm => (1, 3, 1),
        wgpu::TextureFormat::Rg8Snorm => (2, 3, 2),
        wgpu::TextureFormat::Rgba8Snorm => (4, 3, 4),
        _ => (4, 0, 16),
    }
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
    pub(crate) mip_count: u32,
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
    pub(crate) view_dim: wgpu::TextureViewDimension,
    pub(crate) mip_level_count: u32,
    pub(crate) pool: SpareMirrors,
}

impl Drop for TextureMirror {
    fn drop(&mut self) {
        self.pool
            .keep(self.texture.clone(), self.view_dim, self.view.clone());
    }
}

struct HeldMirror {
    texture: wgpu::Texture,
    view_dim: wgpu::TextureViewDimension,
    view: wgpu::TextureView,
    kept: u64,
}

/// Recycled 2D/3D texture mirrors (along with their cached [`wgpu::TextureView`])
/// aged across frame evaluations and destroyed only after the submission
/// referencing them (`bury`). Keeping the view alongside the texture lets
/// [`crate::pool::BindGroupCache`] hit on dynamic per-frame texture mirrors.
#[derive(Clone, Default)]
pub(crate) struct SpareMirrors {
    held: Arc<Mutex<Vec<HeldMirror>>>,
    evaluations: Arc<std::sync::atomic::AtomicU64>,
    gone: Arc<Mutex<Vec<wgpu::Texture>>>,
}

fn mirror_bytes(texture: &wgpu::Texture) -> u64 {
    let w = u64::from(texture.width());
    let h = u64::from(texture.height());
    let d = u64::from(texture.depth_or_array_layers());
    let base = match texture.format() {
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
    };
    if texture.mip_level_count() > 1 {
        base.saturating_mul(4) / 3
    } else {
        base
    }
}

impl SpareMirrors {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        dimension: wgpu::TextureDimension,
        view_dim: wgpu::TextureViewDimension,
        format: wgpu::TextureFormat,
        mip_level_count: u32,
        storage: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let usage = if storage {
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        };
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let matches_tex = |tex: &wgpu::Texture| {
            tex.width() == width
                && tex.height() == height
                && tex.depth_or_array_layers() == depth
                && tex.dimension() == dimension
                && tex.format() == format
                && tex.mip_level_count() == mip_level_count
                && tex.usage() == usage
        };
        if let Some(index) = held
            .iter()
            .rposition(|m| m.view_dim == view_dim && matches_tex(&m.texture))
        {
            let m = held.swap_remove(index);
            return (m.texture, m.view);
        }
        if let Some(index) = held.iter().rposition(|m| matches_tex(&m.texture)) {
            let m = held.swap_remove(index);
            drop(held);
            let view = m.texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(view_dim),
                ..Default::default()
            });
            return (m.texture, view);
        }
        drop(held);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("entelechy texture mirror"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dim),
            ..Default::default()
        });
        (tex, view)
    }

    pub(crate) fn keep(
        &self,
        texture: wgpu::Texture,
        view_dim: wgpu::TextureViewDimension,
        view: wgpu::TextureView,
    ) {
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
        held.push(HeldMirror {
            texture,
            view_dim,
            view,
            kept: now,
        });
        let mut total: u64 = held.iter().map(|m| mirror_bytes(&m.texture)).sum();
        let mut gone = self.gone.lock().unwrap_or_else(|p| p.into_inner());
        while held.len() > SPARE_MIRRORS || total > SPARE_MIRROR_BYTES {
            let evicted = held.remove(0);
            total -= mirror_bytes(&evicted.texture);
            gone.push(evicted.texture);
        }
    }

    pub(crate) fn evaluation_begun(&self) {
        let now = self
            .evaluations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let mut gone = self.gone.lock().unwrap_or_else(|p| p.into_inner());
        held.retain(|m| {
            let fresh = m.kept + 1 >= now;
            if !fresh {
                gone.push(m.texture.clone());
            }
            fresh
        });
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.held
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|m| mirror_bytes(&m.texture))
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

/// One mip-level downsample step queued after mip 0 is copied.
pub(crate) struct PendingMipLevel {
    pub(crate) mip: u32,
    pub(crate) dst_w: u32,
    pub(crate) dst_h: u32,
    pub(crate) dst_d: u32,
    pub(crate) padded_row_bytes: u32,
    pub(crate) groups_x: u32,
    pub(crate) groups_y: u32,
    pub(crate) groups_z: u32,
    pub(crate) params_slice: UniformSlice,
}

/// Planned parameters for generating one mip level `1..mip_level_count`.
pub(crate) struct MipLevelPlan {
    pub(crate) mip: u32,
    pub(crate) dst_w: u32,
    pub(crate) dst_h: u32,
    pub(crate) dst_d: u32,
    pub(crate) padded_row_bytes: u32,
    pub(crate) groups_x: u32,
    pub(crate) groups_y: u32,
    pub(crate) groups_z: u32,
    pub(crate) params: [u8; 48],
}

/// A buffer-to-texture mirror copy queued to run right before a compute pass.
pub(crate) struct PendingMirrorCopy {
    pub(crate) src: wgpu::Buffer,
    pub(crate) dst: wgpu::Texture,
    pub(crate) view_dim: wgpu::TextureViewDimension,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) row_bytes: u32,
    pub(crate) padded: Option<(wgpu::Buffer, UniformSlice, u32, u32)>,
    pub(crate) mip_scratch: Option<wgpu::Buffer>,
    pub(crate) mip_levels: Vec<PendingMipLevel>,
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
    pub(crate) mip_plans: Vec<MipLevelPlan>,
    pub(crate) mip_scratch_bytes: u64,
}

/// GPU compute pipelines that generate 2x2 (2D/2D-array) and 2x2x2 (3D)
/// box-filtered mipmap chains across all 12 hardware texture mirror formats
/// (`R`/`Rg`/`Rgba` x `Unorm8`/`Snorm8`/`F16`/`F32`).
///
/// Core-WebGPU storage-writable formats (`Rgba8Unorm`, `Rgba8Snorm`,
/// `Rgba16Float`, `Rgba32Float`, `R32Float`) write every mip level directly
/// via `textureStore` in one compute pass with zero staging buffer or
/// buffer-to-texture copies (`direct`); remaining narrow formats use the
/// staging-buffer path.
pub(crate) struct MipGenPipelines {
    pipe_2d: wgpu::ComputePipeline,
    layout_2d: wgpu::BindGroupLayout,
    pipe_array: wgpu::ComputePipeline,
    layout_array: wgpu::BindGroupLayout,
    pipe_3d: wgpu::ComputePipeline,
    layout_3d: wgpu::BindGroupLayout,
    direct: Mutex<
        std::collections::HashMap<
            (wgpu::TextureViewDimension, wgpu::TextureFormat),
            (wgpu::ComputePipeline, wgpu::BindGroupLayout),
        >,
    >,
}

fn direct_mip_wgsl(view_dim: wgpu::TextureViewDimension, format: wgpu::TextureFormat) -> String {
    let fmt = direct_storage_wgsl(format).expect("direct storage format checked");
    let sanitize = match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            "let nan = (bitcast<vec4<u32>>(v) & vec4<u32>(0x7FFFFFFFu)) > vec4<u32>(0x7F800000u);\n    let out = select(clamp(v, vec4<f32>(0.0), vec4<f32>(1.0)), vec4<f32>(0.0), nan);"
        }
        wgpu::TextureFormat::Rgba8Snorm => {
            "let nan = (bitcast<vec4<u32>>(v) & vec4<u32>(0x7FFFFFFFu)) > vec4<u32>(0x7F800000u);\n    let out = select(clamp(v, vec4<f32>(-1.0), vec4<f32>(1.0)), vec4<f32>(0.0), nan);"
        }
        _ => "let out = v;",
    };
    match view_dim {
        wgpu::TextureViewDimension::D2 => format!(
            "struct Params {{
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, _p0: u32, _p1: u32,
    _p2: u32, _p3: u32, _p4: u32, _p5: u32,
}}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<{fmt}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) {{ return; }}
    let sx0 = i32(min(gid.x * 2u, p.src_w - 1u));
    let sx1 = i32(min(gid.x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(gid.y * 2u, p.src_h - 1u));
    let sy1 = i32(min(gid.y * 2u + 1u, p.src_h - 1u));
    let v = (textureLoad(src, vec2<i32>(sx0, sy0), 0)
        + textureLoad(src, vec2<i32>(sx1, sy0), 0)
        + textureLoad(src, vec2<i32>(sx0, sy1), 0)
        + textureLoad(src, vec2<i32>(sx1, sy1), 0)) * 0.25;
    {sanitize}
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), out);
}}"
        ),
        wgpu::TextureViewDimension::D2Array => format!(
            "struct Params {{
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, _p0: u32, _p1: u32,
    _p2: u32, _p3: u32, _p4: u32, _p5: u32,
}}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d_array<f32>;
@group(0) @binding(2) var dst: texture_storage_2d_array<{fmt}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= p.dst_w || gid.y >= p.dst_h || gid.z >= p.dst_d) {{ return; }}
    let sx0 = i32(min(gid.x * 2u, p.src_w - 1u));
    let sx1 = i32(min(gid.x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(gid.y * 2u, p.src_h - 1u));
    let sy1 = i32(min(gid.y * 2u + 1u, p.src_h - 1u));
    let lz = i32(gid.z);
    let v = (textureLoad(src, vec2<i32>(sx0, sy0), lz, 0)
        + textureLoad(src, vec2<i32>(sx1, sy0), lz, 0)
        + textureLoad(src, vec2<i32>(sx0, sy1), lz, 0)
        + textureLoad(src, vec2<i32>(sx1, sy1), lz, 0)) * 0.25;
    {sanitize}
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), lz, out);
}}"
        ),
        wgpu::TextureViewDimension::D3 => format!(
            "struct Params {{
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, _p0: u32, _p1: u32,
    _p2: u32, _p3: u32, _p4: u32, _p5: u32,
}}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<{fmt}, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= p.dst_w || gid.y >= p.dst_h || gid.z >= p.dst_d) {{ return; }}
    let sx0 = i32(min(gid.x * 2u, p.src_w - 1u));
    let sx1 = i32(min(gid.x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(gid.y * 2u, p.src_h - 1u));
    let sy1 = i32(min(gid.y * 2u + 1u, p.src_h - 1u));
    let sz0 = i32(min(gid.z * 2u, p.src_d - 1u));
    let sz1 = i32(min(gid.z * 2u + 1u, p.src_d - 1u));
    let a = textureLoad(src, vec3<i32>(sx0, sy0, sz0), 0)
        + textureLoad(src, vec3<i32>(sx1, sy0, sz0), 0)
        + textureLoad(src, vec3<i32>(sx0, sy1, sz0), 0)
        + textureLoad(src, vec3<i32>(sx1, sy1, sz0), 0);
    let b = textureLoad(src, vec3<i32>(sx0, sy0, sz1), 0)
        + textureLoad(src, vec3<i32>(sx1, sy0, sz1), 0)
        + textureLoad(src, vec3<i32>(sx0, sy1, sz1), 0)
        + textureLoad(src, vec3<i32>(sx1, sy1, sz1), 0);
    let v = (a + b) * 0.125;
    {sanitize}
    textureStore(dst, vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)), out);
}}"
        ),
        _ => unreachable!(),
    }
}

impl MipGenPipelines {
    fn direct_pipeline(
        &self,
        device: &wgpu::Device,
        wait: crate::Wait,
        view_dim: wgpu::TextureViewDimension,
        format: wgpu::TextureFormat,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), ExecutionError> {
        let key = (view_dim, format);
        if let Some(cached) = self
            .direct
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }
        let scopes = Scopes::push(device);
        let entries = [
            uniform_entry(0),
            texture_entry(1, view_dim, false),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: view_dim,
                },
                count: None,
            },
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("entelechy mip gen direct"),
            entries: &entries,
        });
        let source = direct_mip_wgsl(view_dim, format);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entelechy mip gen direct"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("entelechy mip gen direct"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("entelechy mip gen direct"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        scopes
            .pop(wait, &source)
            .map_err(|e| ExecutionError::Device(e.said()))?;
        let pair = (pipeline, layout);
        self.direct
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, pair.clone());
        Ok(pair)
    }
}

const MIP_GEN_WORKGROUP: u32 = 64;

fn mip_gen_wgsl(view_dim: wgpu::TextureViewDimension) -> &'static str {
    match view_dim {
        wgpu::TextureViewDimension::D2 => MIP_GEN_2D_WGSL,
        wgpu::TextureViewDimension::D2Array => MIP_GEN_ARRAY_WGSL,
        wgpu::TextureViewDimension::D3 => MIP_GEN_3D_WGSL,
        _ => unreachable!(),
    }
}

const MIP_GEN_BODY_WGSL: &str = "
fn pack_unorm8(x: f32) -> u32 {
    let nan = (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u;
    let c = select(clamp(x, 0.0, 1.0), 0.0, nan);
    return u32(round(c * 255.0));
}
fn pack_snorm8(x: f32) -> u32 {
    let nan = (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u;
    let c = select(clamp(x, -1.0, 1.0), 0.0, nan);
    return bitcast<u32>(i32(round(c * 127.0))) & 0xFFu;
}
fn pack_byte(x: f32, enc: u32) -> u32 {
    if (enc == 2u) { return pack_unorm8(x); }
    return pack_snorm8(x);
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_bytes = p.channels * select(select(1u, 2u, p.encoding == 1u), 4u, p.encoding == 0u);
    let row_words = (p.dst_w * elem_bytes + 3u) >> 2u;
    let total_words = row_words * p.dst_h * p.dst_d;
    let idx = gid.x + gid.y * p.fold_x * 64u;
    if (idx >= total_words) { return; }
    let row = idx / row_words;
    let col_word = idx - row * row_words;
    let z = row / p.dst_h;
    let y = row - z * p.dst_h;
    var word = 0u;
    if (elem_bytes == 1u) {
        let base_x = col_word * 4u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let x = base_x + b;
            if (x < p.dst_w) {
                let v = sample_box(x, y, z);
                word = word | (pack_byte(v.x, p.encoding) << (b * 8u));
            }
        }
    } else if (elem_bytes == 2u) {
        let x0 = col_word * 2u;
        let x1 = x0 + 1u;
        let v0 = sample_box(x0, y, z);
        if (p.encoding == 1u) {
            var s1 = 0.0;
            if (x1 < p.dst_w) { s1 = sample_box(x1, y, z).x; }
            word = pack2x16float(vec2<f32>(v0.x, s1));
        } else {
            word = pack_byte(v0.x, p.encoding) | (pack_byte(v0.y, p.encoding) << 8u);
            if (x1 < p.dst_w) {
                let v1 = sample_box(x1, y, z);
                word = word | (pack_byte(v1.x, p.encoding) << 16u) | (pack_byte(v1.y, p.encoding) << 24u);
            }
        }
    } else if (elem_bytes == 4u) {
        let v = sample_box(col_word, y, z);
        if (p.encoding == 0u) {
            word = bitcast<u32>(v.x);
        } else if (p.encoding == 1u) {
            word = pack2x16float(v.xy);
        } else {
            word = pack_byte(v.x, p.encoding)
                | (pack_byte(v.y, p.encoding) << 8u)
                | (pack_byte(v.z, p.encoding) << 16u)
                | (pack_byte(v.w, p.encoding) << 24u);
        }
    } else if (elem_bytes == 8u) {
        let x = col_word >> 1u;
        let sub = col_word & 1u;
        let v = sample_box(x, y, z);
        if (p.encoding == 0u) {
            word = select(bitcast<u32>(v.y), bitcast<u32>(v.x), sub == 0u);
        } else {
            word = select(pack2x16float(v.zw), pack2x16float(v.xy), sub == 0u);
        }
    } else {
        let x = col_word >> 2u;
        let sub = col_word & 3u;
        let v = sample_box(x, y, z);
        let f = select(select(select(v.w, v.z, sub == 2u), v.y, sub == 1u), v.x, sub == 0u);
        word = bitcast<u32>(f);
    }
    dst[row * p.padded_row_words + col_word] = word;
}
";

const MIP_GEN_2D_WGSL: &str = "
struct Params {
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, padded_row_words: u32, channels: u32,
    encoding: u32, fold_x: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn sample_box(x: u32, y: u32, z: u32) -> vec4<f32> {
    let sx0 = i32(min(x * 2u, p.src_w - 1u));
    let sx1 = i32(min(x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(y * 2u, p.src_h - 1u));
    let sy1 = i32(min(y * 2u + 1u, p.src_h - 1u));
    let _z = z;
    return (textureLoad(src, vec2<i32>(sx0, sy0), 0)
        + textureLoad(src, vec2<i32>(sx1, sy0), 0)
        + textureLoad(src, vec2<i32>(sx0, sy1), 0)
        + textureLoad(src, vec2<i32>(sx1, sy1), 0)) * 0.25;
}
";

const MIP_GEN_ARRAY_WGSL: &str = "
struct Params {
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, padded_row_words: u32, channels: u32,
    encoding: u32, fold_x: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn sample_box(x: u32, y: u32, z: u32) -> vec4<f32> {
    let sx0 = i32(min(x * 2u, p.src_w - 1u));
    let sx1 = i32(min(x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(y * 2u, p.src_h - 1u));
    let sy1 = i32(min(y * 2u + 1u, p.src_h - 1u));
    let lz = i32(z);
    return (textureLoad(src, vec2<i32>(sx0, sy0), lz, 0)
        + textureLoad(src, vec2<i32>(sx1, sy0), lz, 0)
        + textureLoad(src, vec2<i32>(sx0, sy1), lz, 0)
        + textureLoad(src, vec2<i32>(sx1, sy1), lz, 0)) * 0.25;
}
";

const MIP_GEN_3D_WGSL: &str = "
struct Params {
    src_w: u32, src_h: u32, src_d: u32, dst_w: u32,
    dst_h: u32, dst_d: u32, padded_row_words: u32, channels: u32,
    encoding: u32, fold_x: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn sample_box(x: u32, y: u32, z: u32) -> vec4<f32> {
    let sx0 = i32(min(x * 2u, p.src_w - 1u));
    let sx1 = i32(min(x * 2u + 1u, p.src_w - 1u));
    let sy0 = i32(min(y * 2u, p.src_h - 1u));
    let sy1 = i32(min(y * 2u + 1u, p.src_h - 1u));
    let sz0 = i32(min(z * 2u, p.src_d - 1u));
    let sz1 = i32(min(z * 2u + 1u, p.src_d - 1u));
    let a = textureLoad(src, vec3<i32>(sx0, sy0, sz0), 0)
        + textureLoad(src, vec3<i32>(sx1, sy0, sz0), 0)
        + textureLoad(src, vec3<i32>(sx0, sy1, sz0), 0)
        + textureLoad(src, vec3<i32>(sx1, sy1, sz0), 0);
    let b = textureLoad(src, vec3<i32>(sx0, sy0, sz1), 0)
        + textureLoad(src, vec3<i32>(sx1, sy0, sz1), 0)
        + textureLoad(src, vec3<i32>(sx0, sy1, sz1), 0)
        + textureLoad(src, vec3<i32>(sx1, sy1, sz1), 0);
    return (a + b) * 0.125;
}
";

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
                        texture_entry(1, dim, false),
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

    pub(crate) fn mip_gen_pipelines(&self) -> Result<&MipGenPipelines, ExecutionError> {
        let built = self.mip_gen.get_or_init(|| {
            let make = |dim: wgpu::TextureViewDimension| -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), String> {
                let scopes = Scopes::push(&self.device);
                let entries = [
                    uniform_entry(0),
                    texture_entry(1, dim, false),
                    storage_entry(2, false),
                ];
                let layout = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("entelechy mip gen"),
                        entries: &entries,
                    });
                let source = format!("{}{}", mip_gen_wgsl(dim), MIP_GEN_BODY_WGSL);
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("entelechy mip gen"),
                        source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
                    });
                let pipeline_layout =
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("entelechy mip gen"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("entelechy mip gen"),
                            layout: Some(&pipeline_layout),
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                scopes.pop(self.wait, &source).map_err(|e| e.said())?;
                Ok((pipeline, layout))
            };
            let (pipe_2d, layout_2d) = make(wgpu::TextureViewDimension::D2)?;
            let (pipe_array, layout_array) = make(wgpu::TextureViewDimension::D2Array)?;
            let (pipe_3d, layout_3d) = make(wgpu::TextureViewDimension::D3)?;
            Ok(MipGenPipelines {
                pipe_2d,
                layout_2d,
                pipe_array,
                layout_array,
                pipe_3d,
                layout_3d,
                direct: Mutex::new(std::collections::HashMap::new()),
            })
        });
        built
            .as_ref()
            .map_err(|e| ExecutionError::Device(e.clone()))
    }

    /// Uploads raw hardware-compressed 4x4 BC1/BC4/BC5/BC7 blocks (all mip
    /// levels `0..blocks.mip_count()`) directly into a native `wgpu::Texture`
    /// (2D or 2D array) without allocating an uncompressed storage buffer.
    pub(crate) fn create_compressed_texture(
        &self,
        domain: Domain,
        blocks: &entelechy_abi::CompressedBlocks,
    ) -> Result<CompressedTexture, ExecutionError> {
        let format = blocks.format();
        let mip_count = blocks.mip_count();
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
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        for mip in 0..mip_count {
            let md = entelechy_abi::block::mip_domain(domain, mip);
            let mw = md.width().div_ceil(4) * 4;
            let mh = md.height().div_ceil(4) * 4;
            let row_bytes = (mw / 4) * format.block_bytes() as u32;
            let block_rows = mh / 4;
            let slice = blocks.mip_bytes(domain, mip);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                slice,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: (d > 1).then_some(block_rows),
                },
                wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: d,
                },
            );
        }
        let total_bytes = blocks.bytes().len() as u64;
        self.uploaded
            .fetch_add(total_bytes, std::sync::atomic::Ordering::Relaxed);
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
            mip_count,
            bytes: total_bytes,
            pool: self.mirrors.clone(),
        })
    }

    /// Joins whole layers of hardware block-compressed textures into a single
    /// resident `Texture2DArray` (`Domain::Volume`) via GPU-to-GPU
    /// `copy_texture_to_texture` across all common mip levels with zero
    /// decompression.
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
        let mut common_mips = u32::MAX;
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
            let (part_fmt, part_mips) = if let Some(dev) = scope.resident(*handle)? {
                let Some(ct) = dev
                    .downcast_ref::<DeviceImage>()
                    .and_then(|img| img.compressed.as_ref())
                else {
                    return Ok(None);
                };
                (ct.format, ct.mip_count)
            } else if let Some(blocks) = scope.blocks(*handle)? {
                (blocks.format(), blocks.mip_count())
            } else {
                return Ok(None);
            };
            match common_format {
                Some(prev) if prev != part_fmt => return Ok(None),
                None => common_format = Some(part_fmt),
                _ => {}
            }
            common_mips = common_mips.min(part_mips);
            let layers = u32::try_from(range.len() / layer_elems).unwrap_or(0);
            sum_layers = sum_layers.saturating_add(layers);
        }
        let Some(format) = common_format else {
            return Ok(None);
        };
        if sum_layers != total_layers || common_mips == 0 || common_mips == u32::MAX {
            return Ok(None);
        }
        let Some(total_bytes_usize) = format.mip_chain_byte_len(domain, common_mips) else {
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
            mip_level_count: common_mips,
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
        for mip in 0..common_mips {
            let md = entelechy_abi::block::mip_domain(domain, mip);
            let mw = md.width().div_ceil(4) * 4;
            let mh = md.height().div_ceil(4) * 4;
            let mut dst_layer = 0u32;
            for (src_tex, src_layer, layer_count) in &copies {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: src_tex,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: *src_layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &dst,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: dst_layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: mw,
                        height: mh,
                        depth_or_array_layers: *layer_count,
                    },
                );
                dst_layer += *layer_count;
            }
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
            mip_count: common_mips,
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
    /// (with full mipmap chain when `want_mips` is true) and returns a
    /// [`MirrorPlan`] to copy mip 0 and generate mips `1..mip_level_count`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_texture_mirror(
        &self,
        image: &DeviceImage,
        handle: entelechy_abi::Handle,
        port: &Port,
        extent: Domain,
        input_index: usize,
        want_mips: bool,
        force_2d_array: bool,
        blocks_mips: Option<u32>,
    ) -> Result<(wgpu::TextureView, Option<MirrorPlan>), ExecutionError> {
        let (format, view_dim, texel_bytes) =
            format_for(port, force_2d_array).ok_or_else(|| {
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
        let mip_level_count = if want_mips {
            let full = if dimension == wgpu::TextureDimension::D3 {
                entelechy_abi::block::max_3d_mip_count(extent)
            } else {
                entelechy_abi::block::max_mip_count(extent)
            };
            blocks_mips.map_or(full, |m| m.clamp(1, full))
        } else {
            1
        };
        if let Some(existing) = image
            .mirror
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            && existing.format == format
            && existing.dims == (width, height, depth)
            && existing.dimension == dimension
            && existing.view_dim == view_dim
            && existing.mip_level_count >= mip_level_count
        {
            return Ok((existing.view.clone(), None));
        }
        let direct_mips = mip_level_count > 1 && direct_storage_wgsl(format).is_some();
        let (texture, view) = self.mirrors.take(
            &self.device,
            width,
            height,
            depth,
            dimension,
            view_dim,
            format,
            mip_level_count,
            direct_mips,
        );
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
        let mut mip_plans = Vec::new();
        let mut mip_scratch_bytes = 0u64;
        if mip_level_count > 1 {
            let (channels, encoding_code, elem_bytes) = format_meta(format);
            let mut src_w = width;
            let mut src_h = height;
            let mut src_d = depth;
            for mip in 1..mip_level_count {
                let dst_w = (src_w >> 1).max(1);
                let dst_h = (src_h >> 1).max(1);
                let dst_d = if dimension == wgpu::TextureDimension::D3 {
                    (src_d >> 1).max(1)
                } else {
                    depth
                };
                let dst_row_bytes = dst_w * elem_bytes;
                let dst_padded_row_bytes = dst_row_bytes.div_ceil(ROW_ALIGN) * ROW_ALIGN;
                let padded_row_words = dst_padded_row_bytes / 4;
                let (gx, gy, gz) = if direct_mips {
                    match view_dim {
                        wgpu::TextureViewDimension::D2 => (dst_w.div_ceil(8), dst_h.div_ceil(8), 1),
                        wgpu::TextureViewDimension::D2Array => {
                            (dst_w.div_ceil(8), dst_h.div_ceil(8), dst_d)
                        }
                        wgpu::TextureViewDimension::D3 => {
                            (dst_w.div_ceil(4), dst_h.div_ceil(4), dst_d.div_ceil(4))
                        }
                        _ => unreachable!(),
                    }
                } else {
                    let row_words = dst_row_bytes.div_ceil(4);
                    let total_words = row_words.saturating_mul(dst_h).saturating_mul(dst_d);
                    let groups = total_words.div_ceil(MIP_GEN_WORKGROUP).max(1);
                    let (gx, gy) = wgsl::folded(groups);
                    (gx, gy, 1)
                };
                let mut params = [0u8; 48];
                params[0..4].copy_from_slice(&src_w.to_le_bytes());
                params[4..8].copy_from_slice(&src_h.to_le_bytes());
                params[8..12].copy_from_slice(&src_d.to_le_bytes());
                params[12..16].copy_from_slice(&dst_w.to_le_bytes());
                params[16..20].copy_from_slice(&dst_h.to_le_bytes());
                params[20..24].copy_from_slice(&dst_d.to_le_bytes());
                params[24..28].copy_from_slice(&padded_row_words.to_le_bytes());
                params[28..32].copy_from_slice(&channels.to_le_bytes());
                params[32..36].copy_from_slice(&encoding_code.to_le_bytes());
                params[36..40].copy_from_slice(&gx.to_le_bytes());
                if !direct_mips {
                    let level_bytes =
                        (u64::from(dst_padded_row_bytes) * u64::from(dst_h) * u64::from(dst_d))
                            .max(16)
                            .div_ceil(16)
                            * 16;
                    mip_scratch_bytes = mip_scratch_bytes.max(level_bytes);
                }
                mip_plans.push(MipLevelPlan {
                    mip,
                    dst_w,
                    dst_h,
                    dst_d,
                    padded_row_bytes: dst_padded_row_bytes,
                    groups_x: gx,
                    groups_y: gy,
                    groups_z: gz,
                    params,
                });
                src_w = dst_w;
                src_h = dst_h;
                src_d = dst_d;
            }
        }
        let mirror = TextureMirror {
            texture,
            view: view.clone(),
            format,
            dims: (width, height, depth),
            dimension,
            view_dim,
            mip_level_count,
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
                mip_plans,
                mip_scratch_bytes,
            }),
        ))
    }

    /// Encodes a queued buffer-to-texture copy (plus GPU mipmap generation when
    /// `copy.mip_levels` is non-empty) into `encoder` before the compute pass
    /// that samples the texture.
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
        if !copy.mip_levels.is_empty() {
            let pipes = self.mip_gen_pipelines()?;
            if copy.mip_scratch.is_none() {
                // Direct 1-pass `textureStore` mipmap chain (zero staging buffer,
                // zero per-level `copy_buffer_to_texture` copies).
                let (pipeline, layout) = pipes.direct_pipeline(
                    &self.device,
                    self.wait,
                    copy.view_dim,
                    copy.dst.format(),
                )?;
                let mut bind_groups = Vec::with_capacity(copy.mip_levels.len());
                for level in &copy.mip_levels {
                    let src_view = copy.dst.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("entelechy mip gen src"),
                        dimension: Some(copy.view_dim),
                        base_mip_level: level.mip - 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                    let dst_view = copy.dst.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("entelechy mip gen dst"),
                        dimension: Some(copy.view_dim),
                        base_mip_level: level.mip,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                    bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("entelechy mip gen direct"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: level.params_slice.binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&src_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&dst_view),
                            },
                        ],
                    }));
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("entelechy mip gen direct"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                for (level, bg) in copy.mip_levels.iter().zip(&bind_groups) {
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(level.groups_x, level.groups_y, level.groups_z);
                }
            } else if let Some(scratch) = &copy.mip_scratch {
                let (pipeline, layout) = match copy.view_dim {
                    wgpu::TextureViewDimension::D2 => (&pipes.pipe_2d, &pipes.layout_2d),
                    wgpu::TextureViewDimension::D2Array => (&pipes.pipe_array, &pipes.layout_array),
                    wgpu::TextureViewDimension::D3 => (&pipes.pipe_3d, &pipes.layout_3d),
                    _ => unreachable!(),
                };
                for level in &copy.mip_levels {
                    let src_view = copy.dst.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("entelechy mip gen src"),
                        dimension: Some(copy.view_dim),
                        base_mip_level: level.mip - 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("entelechy mip gen"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: level.params_slice.binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&src_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: scratch.as_entire_binding(),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("entelechy mip gen"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(level.groups_x, level.groups_y, 1);
                    }
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: scratch,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(level.padded_row_bytes),
                                rows_per_image: (level.dst_d > 1).then_some(level.dst_h),
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &copy.dst,
                            mip_level: level.mip,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: level.dst_w,
                            height: level.dst_h,
                            depth_or_array_layers: level.dst_d,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}
