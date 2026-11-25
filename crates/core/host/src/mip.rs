//! Deterministic mipmap pyramids and filtered sampling (`at_linear` with LOD
//! or gradients, and `at_repeat`) shared by CPU and JIT backends.
//!
//! Coordinates match [`FieldData`]: `position` is in mip-0 texel units (texel 0
//! centered at `0.5`, texel `w - 1` at `w - 0.5`), `lod` is the mip level (`0.0`
//! = base resolution), and `(ddx, ddy)` are texel-space screen derivatives.

use crate::block::{max_3d_mip_count, max_mip_count, mip_domain};
use crate::{Encoding, FieldData, Scalar, Shape};

/// Maximum hardware-matching anisotropic tap ratio (`16x`).
pub const MAX_ANISOTROPY: f32 = 16.0;

/// One mip level stored as unpacked `f32` scalars (`w * h * d * stride`).
#[derive(Clone, Debug)]
pub struct MipLevel {
    /// Width in texels (`>= 1`).
    pub width: u32,
    /// Height in texels (`>= 1`).
    pub height: u32,
    /// Depth or array layer count (`>= 1`).
    pub depth: u32,
    /// Flat row-major floats (`((z * height + y) * width + x) * stride + c`).
    pub data: Vec<f32>,
}

/// Cached mipmap pyramid built lazily on an all-float [`FieldData`].
#[derive(Clone, Debug)]
pub struct MipPyramid {
    stride: usize,
    base_w: f32,
    base_h: f32,
    base_d: f32,
    /// 2D or 2D-array mip levels (`(w >> m).max(1) x (h >> m).max(1) x d`).
    levels_2d: Vec<MipLevel>,
    /// True 3D mip levels (`(w >> m).max(1) x (h >> m).max(1) x (d >> m).max(1)`),
    /// populated for non-unorm8 3D `Volume` Fields.
    levels_3d: Option<Vec<MipLevel>>,
}

impl MipPyramid {
    /// Builds the mipmap pyramid for an all-float `Grid` or `Volume` Field.
    /// Uses baked compressed mip levels from [`FieldData::blocks`] when present,
    /// and deterministic 2x2 (or 2x2x2) box downsampling otherwise.
    pub(crate) fn from_field(field: &FieldData) -> Option<Self> {
        let domain = field.domain();
        if domain.shape() == Shape::Rows {
            return None;
        }
        let schema = field.schema();
        if !schema.scalars().iter().all(|s| *s == Scalar::Float) {
            return None;
        }
        let stride = schema.value_words();
        let w0 = domain.width();
        let h0 = domain.height();
        let d0 = domain.depth();
        let vals0 = field.values();
        let data0: Vec<f32> = vals0.iter().map(|&w| f32::from_bits(w)).collect();
        let level0 = MipLevel {
            width: w0,
            height: h0,
            depth: d0,
            data: data0,
        };

        let mip_count_2d = field
            .blocks()
            .map_or_else(|| max_mip_count(domain), |b| b.mip_count());
        let mut levels_2d = Vec::with_capacity(mip_count_2d as usize);
        levels_2d.push(level0);

        for mip in 1..mip_count_2d {
            if let Some(blocks) = field.blocks()
                && mip < blocks.mip_count()
            {
                let md = mip_domain(domain, mip);
                let storage = blocks.decode_mip_storage(domain, mip);
                let unpacked = schema.unpack(&storage, md.count());
                let data: Vec<f32> = unpacked.into_iter().map(f32::from_bits).collect();
                levels_2d.push(MipLevel {
                    width: md.width(),
                    height: md.height(),
                    depth: md.depth(),
                    data,
                });
            } else {
                let prev = levels_2d.last().expect("level 0 exists");
                levels_2d.push(downsample_2d(prev, stride));
            }
        }

        let is_unorm8 = schema.uniform_encoding() == Some(Encoding::Unorm8);
        let levels_3d = if domain.shape() == Shape::Volume
            && d0 > 1
            && field.blocks().is_none()
            && !is_unorm8
        {
            let mip_count_3d = max_3d_mip_count(domain);
            let mut chain = Vec::with_capacity(mip_count_3d as usize);
            chain.push(levels_2d[0].clone());
            for _ in 1..mip_count_3d {
                let prev = chain.last().expect("level 0 exists");
                chain.push(downsample_3d(prev, stride));
            }
            Some(chain)
        } else {
            None
        };

        Some(Self {
            stride,
            base_w: w0 as f32,
            base_h: h0 as f32,
            base_d: d0 as f32,
            levels_2d,
            levels_3d,
        })
    }

    /// Number of 2D / 2D-array mip levels held.
    pub fn mip_count_2d(&self) -> u32 {
        self.levels_2d.len() as u32
    }

    /// Trilinear 2D or 2D-array sample at base texel coordinates `pos` and mip
    /// level `lod`.
    pub fn sample_trilinear_2d(&self, pos: [f32; 3], repeat_xy: bool, lod: f32, out: &mut [f32]) {
        let max_mip = (self.levels_2d.len() - 1) as f32;
        let clamped_lod = if lod.is_nan() {
            0.0
        } else {
            lod.clamp(0.0, max_mip)
        };
        let l0 = clamped_lod.floor() as usize;
        let l1 = (l0 + 1).min(self.levels_2d.len() - 1);
        let frac = clamped_lod - l0 as f32;
        let u = if pos[0].is_nan() {
            0.0
        } else {
            pos[0] / self.base_w
        };
        let v = if pos[1].is_nan() {
            0.0
        } else {
            pos[1] / self.base_h
        };
        let (z0, z1, fz) = z_blend(pos[2], self.levels_2d[0].depth);

        if l0 == l1 || frac <= 0.0 {
            sample_level_2d_z(
                &self.levels_2d[l0],
                self.stride,
                u,
                v,
                z0,
                z1,
                fz,
                repeat_xy,
                out,
            );
        } else {
            let mut a = vec![0.0f32; self.stride];
            let mut b = vec![0.0f32; self.stride];
            sample_level_2d_z(
                &self.levels_2d[l0],
                self.stride,
                u,
                v,
                z0,
                z1,
                fz,
                repeat_xy,
                &mut a,
            );
            sample_level_2d_z(
                &self.levels_2d[l1],
                self.stride,
                u,
                v,
                z0,
                z1,
                fz,
                repeat_xy,
                &mut b,
            );
            for i in 0..self.stride {
                out[i] = a[i] + (b[i] - a[i]) * frac;
            }
        }
    }

    /// 3D trilinear mipmap sample for true 3D volumes (`levels_3d`), falling
    /// back to 2D-array trilinear when `levels_3d` is absent.
    pub fn sample_trilinear_3d(&self, pos: [f32; 3], repeat_xy: bool, lod: f32, out: &mut [f32]) {
        let Some(chain) = &self.levels_3d else {
            self.sample_trilinear_2d(pos, repeat_xy, lod, out);
            return;
        };
        let max_mip = (chain.len() - 1) as f32;
        let clamped_lod = if lod.is_nan() {
            0.0
        } else {
            lod.clamp(0.0, max_mip)
        };
        let l0 = clamped_lod.floor() as usize;
        let l1 = (l0 + 1).min(chain.len() - 1);
        let frac = clamped_lod - l0 as f32;
        let u = if pos[0].is_nan() {
            0.0
        } else {
            pos[0] / self.base_w
        };
        let v = if pos[1].is_nan() {
            0.0
        } else {
            pos[1] / self.base_h
        };
        let w = if pos[2].is_nan() {
            0.0
        } else {
            pos[2] / self.base_d
        };

        if l0 == l1 || frac <= 0.0 {
            sample_level_3d(&chain[l0], self.stride, u, v, w, repeat_xy, out);
        } else {
            let mut a = vec![0.0f32; self.stride];
            let mut b = vec![0.0f32; self.stride];
            sample_level_3d(&chain[l0], self.stride, u, v, w, repeat_xy, &mut a);
            sample_level_3d(&chain[l1], self.stride, u, v, w, repeat_xy, &mut b);
            for i in 0..self.stride {
                out[i] = a[i] + (b[i] - a[i]) * frac;
            }
        }
    }

    /// 16x anisotropic + trilinear 2D (or 2D-array layer) sample given
    /// texel-space screen derivatives `(ddx, ddy)`.
    pub fn sample_aniso_2d(
        &self,
        pos: [f32; 3],
        repeat_xy: bool,
        ddx: [f32; 2],
        ddy: [f32; 2],
        out: &mut [f32],
    ) {
        let sanitize = |v: [f32; 2]| {
            [
                if v[0].is_finite() { v[0] } else { 0.0 },
                if v[1].is_finite() { v[1] } else { 0.0 },
            ]
        };
        let ddx = sanitize(ddx);
        let ddy = sanitize(ddy);
        let lx = (ddx[0] * ddx[0] + ddx[1] * ddx[1]).sqrt();
        let ly = (ddy[0] * ddy[0] + ddy[1] * ddy[1]).sqrt();
        let (major, p_max, p_min) = if lx >= ly {
            (ddx, lx, ly)
        } else {
            (ddy, ly, lx)
        };
        let ratio = if p_min > 1e-6 {
            (p_max / p_min).min(MAX_ANISOTROPY)
        } else if p_max > 1.0 {
            p_max.min(MAX_ANISOTROPY)
        } else {
            1.0
        };
        let n_taps = (ratio.ceil() as usize).clamp(1, MAX_ANISOTROPY as usize);
        let lod = if p_max <= 1e-6 {
            0.0
        } else {
            (p_max / (n_taps as f32)).max(1e-6).log2().max(0.0)
        };
        if n_taps == 1 {
            self.sample_trilinear_2d(pos, repeat_xy, lod, out);
            return;
        }
        out.fill(0.0);
        let mut tap = vec![0.0f32; self.stride];
        let inv_n = 1.0 / (n_taps as f32);
        for i in 0..n_taps {
            let t = (i as f32 + 0.5) * inv_n - 0.5;
            let tap_pos = [pos[0] + major[0] * t, pos[1] + major[1] * t, pos[2]];
            self.sample_trilinear_2d(tap_pos, repeat_xy, lod, &mut tap);
            for c in 0..self.stride {
                out[c] += tap[c];
            }
        }
        for v in out.iter_mut().take(self.stride) {
            *v *= inv_n;
        }
    }
}

/// Mip-0 bilinear/trilinear filtered sample directly from raw `storage` words
/// and `schema` (zero allocation, used by both `sample_filtered` and JIT).
pub fn sample_mip0_raw(
    schema: &crate::Schema,
    storage: &[u32],
    domain: crate::Domain,
    pos: [f32; 3],
    repeat_xy: bool,
    out_words: &mut [u32],
) {
    let stride = schema.value_words();
    debug_assert_eq!(out_words.len(), stride);
    let w = domain.width();
    let h = domain.height();
    let d = domain.depth();
    match domain.shape() {
        Shape::Rows => {
            let (x0, x1, fx) = axis_blend(pos[0], w, repeat_xy);
            for (s, slot) in out_words.iter_mut().enumerate().take(stride) {
                let a = f32::from_bits(schema.load(storage, x0 as usize, s));
                let b = f32::from_bits(schema.load(storage, x1 as usize, s));
                *slot = (a + (b - a) * fx).to_bits();
            }
        }
        Shape::Grid => {
            let (x0, x1, fx) = axis_blend(pos[0], w, repeat_xy);
            let (y0, y1, fy) = axis_blend(pos[1], h, repeat_xy);
            let i00 = (y0 * w + x0) as usize;
            let i10 = (y0 * w + x1) as usize;
            let i01 = (y1 * w + x0) as usize;
            let i11 = (y1 * w + x1) as usize;
            for (s, slot) in out_words.iter_mut().enumerate().take(stride) {
                let v00 = f32::from_bits(schema.load(storage, i00, s));
                let v10 = f32::from_bits(schema.load(storage, i10, s));
                let v01 = f32::from_bits(schema.load(storage, i01, s));
                let v11 = f32::from_bits(schema.load(storage, i11, s));
                let top = v00 + (v10 - v00) * fx;
                let bot = v01 + (v11 - v01) * fx;
                *slot = (top + (bot - top) * fy).to_bits();
            }
        }
        Shape::Volume => {
            let (x0, x1, fx) = axis_blend(pos[0], w, repeat_xy);
            let (y0, y1, fy) = axis_blend(pos[1], h, repeat_xy);
            let (z0, z1, fz) = z_blend(pos[2], d);
            let at = |x: u32, y: u32, z: u32| ((z * h + y) * w + x) as usize;
            let n00 = at(x0, y0, z0);
            let n10 = at(x1, y0, z0);
            let n01 = at(x0, y1, z0);
            let n11 = at(x1, y1, z0);
            if z0 == z1 || fz == 0.0 {
                for (s, slot) in out_words.iter_mut().enumerate().take(stride) {
                    let v00 = f32::from_bits(schema.load(storage, n00, s));
                    let v10 = f32::from_bits(schema.load(storage, n10, s));
                    let v01 = f32::from_bits(schema.load(storage, n01, s));
                    let v11 = f32::from_bits(schema.load(storage, n11, s));
                    let top = v00 + (v10 - v00) * fx;
                    let bot = v01 + (v11 - v01) * fx;
                    *slot = (top + (bot - top) * fy).to_bits();
                }
            } else {
                let f00 = at(x0, y0, z1);
                let f10 = at(x1, y0, z1);
                let f01 = at(x0, y1, z1);
                let f11 = at(x1, y1, z1);
                for (s, slot) in out_words.iter_mut().enumerate().take(stride) {
                    let near_top = lerp_load(schema, storage, n00, n10, s, fx);
                    let near_bot = lerp_load(schema, storage, n01, n11, s, fx);
                    let far_top = lerp_load(schema, storage, f00, f10, s, fx);
                    let far_bot = lerp_load(schema, storage, f01, f11, s, fx);
                    let near = near_top + (near_bot - near_top) * fy;
                    let far = far_top + (far_bot - far_top) * fy;
                    *slot = (near + (far - near) * fz).to_bits();
                }
            }
        }
    }
}

/// Samples a prebuilt [`MipPyramid`] with explicit `lod` or 2D gradients
/// `(ddx, ddy)` and writes the resulting `f32` words into `out_words`.
pub fn sample_pyramid_words(
    pyramid: &MipPyramid,
    shape: Shape,
    pos: [f32; 3],
    repeat_xy: bool,
    lod: Option<f32>,
    grad: Option<([f32; 2], [f32; 2])>,
    out_words: &mut [u32],
) {
    let mut floats = vec![0.0f32; pyramid.stride];
    if let Some((ddx, ddy)) = grad {
        pyramid.sample_aniso_2d(pos, repeat_xy, ddx, ddy, &mut floats);
    } else if let Some(lod) = lod {
        if shape == Shape::Volume {
            pyramid.sample_trilinear_3d(pos, repeat_xy, lod, &mut floats);
        } else {
            pyramid.sample_trilinear_2d(pos, repeat_xy, lod, &mut floats);
        }
    }
    for (dst, &v) in out_words.iter_mut().zip(&floats) {
        *dst = v.to_bits();
    }
}

/// Unified reference implementation for `Op::AtFiltered` (`at_linear` 3/4-arg
/// and `at_repeat` 2/3/4-arg), shared by CPU and JIT backends.
pub fn sample_filtered(
    field: &FieldData,
    pos: [f32; 3],
    repeat_xy: bool,
    lod: Option<f32>,
    grad: Option<([f32; 2], [f32; 2])>,
    out_words: &mut [u32],
) {
    let domain = field.domain();
    let schema = field.schema();
    debug_assert_eq!(out_words.len(), schema.value_words());

    // Fast path for mip-0 1D/2D/3D repeat/clamp without allocating a mip pyramid.
    if lod.is_none() && grad.is_none() {
        sample_mip0_raw(schema, field.storage(), domain, pos, repeat_xy, out_words);
        return;
    }

    let Some(pyramid) = field.mip_pyramid() else {
        out_words.fill(0);
        return;
    };
    sample_pyramid_words(
        pyramid,
        domain.shape(),
        pos,
        repeat_xy,
        lod,
        grad,
        out_words,
    );
}

#[inline]
fn lerp_load(schema: &crate::Schema, storage: &[u32], a: usize, b: usize, s: usize, t: f32) -> f32 {
    let va = f32::from_bits(schema.load(storage, a, s));
    let vb = f32::from_bits(schema.load(storage, b, s));
    va + (vb - va) * t
}

#[inline]
fn axis_blend(c: f32, n: u32, repeat: bool) -> (u32, u32, f32) {
    if n <= 1 || c.is_nan() {
        return (0, 0, 0.0);
    }
    let t = c - 0.5;
    if repeat {
        let nf = n as f32;
        let wrapped = t.rem_euclid(nf);
        let lo = (wrapped.floor() as u32).min(n - 1);
        let hi = if lo + 1 >= n { 0 } else { lo + 1 };
        let f = (wrapped - wrapped.floor()).clamp(0.0, 1.0);
        (lo, hi, f)
    } else {
        let max_i = (n - 1) as f32;
        if t <= 0.0 {
            (0, 0, 0.0)
        } else if t >= max_i {
            (n - 1, n - 1, 0.0)
        } else {
            let lo = t.floor() as u32;
            let hi = (lo + 1).min(n - 1);
            let f = (t - t.floor()).clamp(0.0, 1.0);
            (lo, hi, f)
        }
    }
}

#[inline]
fn z_blend(c: f32, d: u32) -> (u32, u32, f32) {
    axis_blend(c, d, false)
}

fn sample_level_2d(
    level: &MipLevel,
    stride: usize,
    u: f32,
    v: f32,
    z: u32,
    repeat_xy: bool,
    out: &mut [f32],
) {
    let (x0, x1, fx) = axis_blend(u * level.width as f32, level.width, repeat_xy);
    let (y0, y1, fy) = axis_blend(v * level.height as f32, level.height, repeat_xy);
    let w = level.width as usize;
    let h = level.height as usize;
    let z = z as usize;
    let b00 = ((z * h + y0 as usize) * w + x0 as usize) * stride;
    let b10 = ((z * h + y0 as usize) * w + x1 as usize) * stride;
    let b01 = ((z * h + y1 as usize) * w + x0 as usize) * stride;
    let b11 = ((z * h + y1 as usize) * w + x1 as usize) * stride;
    let d = &level.data;
    for s in 0..stride {
        let top = d[b00 + s] + (d[b10 + s] - d[b00 + s]) * fx;
        let bot = d[b01 + s] + (d[b11 + s] - d[b01 + s]) * fx;
        out[s] = top + (bot - top) * fy;
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_level_2d_z(
    level: &MipLevel,
    stride: usize,
    u: f32,
    v: f32,
    z0: u32,
    z1: u32,
    fz: f32,
    repeat_xy: bool,
    out: &mut [f32],
) {
    if z0 == z1 || fz <= 0.0 {
        sample_level_2d(level, stride, u, v, z0, repeat_xy, out);
    } else {
        let mut a = vec![0.0f32; stride];
        let mut b = vec![0.0f32; stride];
        sample_level_2d(level, stride, u, v, z0, repeat_xy, &mut a);
        sample_level_2d(level, stride, u, v, z1, repeat_xy, &mut b);
        for s in 0..stride {
            out[s] = a[s] + (b[s] - a[s]) * fz;
        }
    }
}

fn sample_level_3d(
    level: &MipLevel,
    stride: usize,
    u: f32,
    v: f32,
    w_coord: f32,
    repeat_xy: bool,
    out: &mut [f32],
) {
    let (z0, z1, fz) = z_blend(w_coord * level.depth as f32, level.depth);
    sample_level_2d_z(level, stride, u, v, z0, z1, fz, repeat_xy, out);
}

fn downsample_2d(prev: &MipLevel, stride: usize) -> MipLevel {
    let sw = prev.width as usize;
    let sh = prev.height as usize;
    let dw = (sw >> 1).max(1);
    let dh = (sh >> 1).max(1);
    let depth = prev.depth as usize;
    let mut data = vec![0.0f32; dw * dh * depth * stride];
    for z in 0..depth {
        for y in 0..dh {
            let sy0 = (y * 2).min(sh - 1);
            let sy1 = (y * 2 + 1).min(sh - 1);
            for x in 0..dw {
                let sx0 = (x * 2).min(sw - 1);
                let sx1 = (x * 2 + 1).min(sw - 1);
                let i00 = ((z * sh + sy0) * sw + sx0) * stride;
                let i10 = ((z * sh + sy0) * sw + sx1) * stride;
                let i01 = ((z * sh + sy1) * sw + sx0) * stride;
                let i11 = ((z * sh + sy1) * sw + sx1) * stride;
                let out = ((z * dh + y) * dw + x) * stride;
                for s in 0..stride {
                    data[out + s] = (prev.data[i00 + s]
                        + prev.data[i10 + s]
                        + prev.data[i01 + s]
                        + prev.data[i11 + s])
                        * 0.25;
                }
            }
        }
    }
    MipLevel {
        width: dw as u32,
        height: dh as u32,
        depth: prev.depth,
        data,
    }
}

fn downsample_3d(prev: &MipLevel, stride: usize) -> MipLevel {
    let sw = prev.width as usize;
    let sh = prev.height as usize;
    let sd = prev.depth as usize;
    let dw = (sw >> 1).max(1);
    let dh = (sh >> 1).max(1);
    let dd = (sd >> 1).max(1);
    let mut data = vec![0.0f32; dw * dh * dd * stride];
    for z in 0..dd {
        let sz0 = (z * 2).min(sd - 1);
        let sz1 = (z * 2 + 1).min(sd - 1);
        for y in 0..dh {
            let sy0 = (y * 2).min(sh - 1);
            let sy1 = (y * 2 + 1).min(sh - 1);
            for x in 0..dw {
                let sx0 = (x * 2).min(sw - 1);
                let sx1 = (x * 2 + 1).min(sw - 1);
                let out = ((z * dh + y) * dw + x) * stride;
                for s in 0..stride {
                    let mut sum = 0.0f32;
                    for sz in [sz0, sz1] {
                        for sy in [sy0, sy1] {
                            for sx in [sx0, sx1] {
                                sum += prev.data[((sz * sh + sy) * sw + sx) * stride + s];
                            }
                        }
                    }
                    data[out + s] = sum * 0.125;
                }
            }
        }
    }
    MipLevel {
        width: dw as u32,
        height: dh as u32,
        depth: dd as u32,
        data,
    }
}
