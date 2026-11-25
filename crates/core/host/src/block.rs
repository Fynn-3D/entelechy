//! Hardware block-compressed 2D texture storage (BC1, BC4, BC5, BC7).
//!
//! A block-compressed Field keeps its logical [`Schema`] (`Float`, `Vec2`, or
//! `Vec4` as `unorm8`) and [`Domain`] (`Grid` or stacked `Volume`), while
//! holding 4x4 hardware-native compressed blocks in memory and on the GPU
//! (4x to 8x smaller than uncompressed `unorm8`, 16x to 32x smaller than
//! `f32`).
//!
//! Decoders follow the Khronos Data Format Specification §17 / Direct3D BCn
//! rules exactly (all 8 BC7 modes supported) so CPU, JIT, and hardware GPU
//! texture units produce matching texels everywhere.

use std::fmt;
use std::sync::Arc;

use crate::{Domain, Encoding, HostError, Scalar, Schema, Shape};

/// Hardware 4x4 block-compression format carried by a [`CompressedBlocks`]
/// payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockFormat {
    /// 64-bit RGB + 1-bit alpha (`0.5` bytes/texel): `Grid`/`Volume` of `Vec4 as unorm8`.
    Bc1RgbaUnorm,
    /// 64-bit single-channel unsigned normalized (`0.5` bytes/texel): `Float as unorm8`.
    Bc4RUnorm,
    /// 128-bit two-channel unsigned normalized (`1.0` bytes/texel): `Vec2 as unorm8`.
    Bc5RgUnorm,
    /// 128-bit high-quality RGBA unsigned normalized (`1.0` bytes/texel): `Vec4 as unorm8`.
    Bc7RgbaUnorm,
}

impl BlockFormat {
    /// All block formats supported by the Host.
    pub const ALL: [BlockFormat; 4] = [
        BlockFormat::Bc1RgbaUnorm,
        BlockFormat::Bc4RUnorm,
        BlockFormat::Bc5RgUnorm,
        BlockFormat::Bc7RgbaUnorm,
    ];

    /// Short canonical name used in diagnostics and import flags.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bc1RgbaUnorm => "bc1",
            Self::Bc4RUnorm => "bc4",
            Self::Bc5RgUnorm => "bc5",
            Self::Bc7RgbaUnorm => "bc7",
        }
    }

    /// Parses a short format name (`bc1`, `bc4`, `bc5`, `bc7`).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "bc1" | "dxt1" => Some(Self::Bc1RgbaUnorm),
            "bc4" | "ati1" => Some(Self::Bc4RUnorm),
            "bc5" | "ati2" => Some(Self::Bc5RgUnorm),
            "bc7" => Some(Self::Bc7RgbaUnorm),
            _ => None,
        }
    }

    /// Bytes occupied by one 4x4 compressed block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc1RgbaUnorm | Self::Bc4RUnorm => 8,
            Self::Bc5RgUnorm | Self::Bc7RgbaUnorm => 16,
        }
    }

    /// Logical float channels decoded per texel (`1`, `2`, or `4`).
    pub fn channels(self) -> usize {
        match self {
            Self::Bc4RUnorm => 1,
            Self::Bc5RgUnorm => 2,
            Self::Bc1RgbaUnorm | Self::Bc7RgbaUnorm => 4,
        }
    }

    /// Logical Field [`Schema`] exposed to Kernels (`unorm8` scalars).
    pub fn schema(self) -> Schema {
        Schema::new(vec![Scalar::Float; self.channels()])
            .with_encoding(Encoding::Unorm8)
            .expect("unorm8 fits float scalars")
    }

    /// Bytes per row of 4x4 blocks for a texture of width `width`.
    pub fn row_bytes(self, width: u32) -> Option<usize> {
        let bx = (width as usize).div_ceil(4);
        bx.checked_mul(self.block_bytes())
    }

    /// Total compressed byte count for `domain` (`Grid` or `Volume`).
    pub fn byte_len(self, domain: Domain) -> Option<usize> {
        if domain.shape() == Shape::Rows
            || domain.width() == 0
            || domain.height() == 0
            || domain.depth() == 0
        {
            return None;
        }
        let bx = (domain.width() as usize).div_ceil(4);
        let by = (domain.height() as usize).div_ceil(4);
        let layers = domain.depth() as usize;
        bx.checked_mul(by)?
            .checked_mul(layers)?
            .checked_mul(self.block_bytes())
    }

    /// Default compressed format matching a logical channel count (`1` -> BC4,
    /// `2` -> BC5, `4` -> BC7).
    pub fn for_channels(channels: usize) -> Option<Self> {
        match channels {
            1 => Some(Self::Bc4RUnorm),
            2 => Some(Self::Bc5RgUnorm),
            4 => Some(Self::Bc7RgbaUnorm),
            _ => None,
        }
    }
}

impl fmt::Display for BlockFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Immutable hardware-native 4x4 block-compressed bytes for a `Grid` or stacked
/// `Volume` (2D texture array) Field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedBlocks {
    format: BlockFormat,
    bytes: Arc<[u8]>,
}

impl CompressedBlocks {
    /// Wraps raw compressed block bytes, validating exact size for `domain`.
    pub fn new(format: BlockFormat, domain: Domain, bytes: Vec<u8>) -> Result<Self, HostError> {
        let Some(expected) = format.byte_len(domain) else {
            return Err(HostError::ZeroSize);
        };
        if bytes.len() != expected {
            return Err(HostError::WordCount {
                expected: expected.div_ceil(4),
                actual: bytes.len().div_ceil(4),
            });
        }
        Ok(Self {
            format,
            bytes: Arc::from(bytes),
        })
    }

    /// The block format.
    pub fn format(&self) -> BlockFormat {
        self.format
    }

    /// Raw compressed bytes, layer after layer, block-row after block-row.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decodes all compressed blocks into packed `unorm8` little-endian `u32`
    /// storage words matching [`BlockFormat::schema`] over `domain`.
    pub fn decode_storage(&self, domain: Domain) -> Vec<u32> {
        let w = domain.width() as usize;
        let h = domain.height() as usize;
        let d = domain.depth() as usize;
        let ch = self.format.channels();
        let mut out_bytes = vec![0u8; w * h * d * ch];
        let bx_count = w.div_ceil(4);
        let by_count = h.div_ceil(4);
        let bsize = self.format.block_bytes();
        let layer_blocks = bx_count * by_count;

        for z in 0..d {
            for by in 0..by_count {
                for bx in 0..bx_count {
                    let block_idx = z * layer_blocks + by * bx_count + bx;
                    let src = &self.bytes[block_idx * bsize..(block_idx + 1) * bsize];
                    let mut rgba = [[0u8; 4]; 16];
                    match self.format {
                        BlockFormat::Bc1RgbaUnorm => decode_bc1_block(src, &mut rgba),
                        BlockFormat::Bc4RUnorm => {
                            let r = decode_bc4_block(src);
                            for i in 0..16 {
                                rgba[i][0] = r[i];
                            }
                        }
                        BlockFormat::Bc5RgUnorm => {
                            let r = decode_bc4_block(&src[0..8]);
                            let g = decode_bc4_block(&src[8..16]);
                            for i in 0..16 {
                                rgba[i][0] = r[i];
                                rgba[i][1] = g[i];
                            }
                        }
                        BlockFormat::Bc7RgbaUnorm => decode_bc7_block(src, &mut rgba),
                    }
                    for py in 0..4 {
                        let y = by * 4 + py;
                        if y >= h {
                            continue;
                        }
                        for px in 0..4 {
                            let x = bx * 4 + px;
                            if x >= w {
                                continue;
                            }
                            let dst_px = (z * h + y) * w + x;
                            let base = dst_px * ch;
                            let src_px = &rgba[py * 4 + px];
                            out_bytes[base..base + ch].copy_from_slice(&src_px[..ch]);
                        }
                    }
                }
            }
        }

        let word_len = out_bytes.len().div_ceil(4);
        let mut words = Vec::with_capacity(word_len);
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);
            words.push(u32::from_le_bytes(buf));
        }
        words
    }

    /// Compresses packed `unorm8` texel bytes (`w * h * d * channels`) into
    /// `format`.
    pub fn encode_unorm8(
        format: BlockFormat,
        domain: Domain,
        pixels: &[u8],
    ) -> Result<Self, HostError> {
        let Some(expected_out) = format.byte_len(domain) else {
            return Err(HostError::ZeroSize);
        };
        let w = domain.width() as usize;
        let h = domain.height() as usize;
        let d = domain.depth() as usize;
        let ch = format.channels();
        let expected_in = w
            .checked_mul(h)
            .and_then(|v| v.checked_mul(d))
            .and_then(|v| v.checked_mul(ch));
        if expected_in != Some(pixels.len()) {
            return Err(HostError::WordCount {
                expected: expected_in.unwrap_or(0).div_ceil(4),
                actual: pixels.len().div_ceil(4),
            });
        }
        let bx_count = w.div_ceil(4);
        let by_count = h.div_ceil(4);
        let bsize = format.block_bytes();
        let mut out = vec![0u8; expected_out];
        let layer_blocks = bx_count * by_count;

        for z in 0..d {
            for by in 0..by_count {
                for bx in 0..bx_count {
                    let mut tile = [[0u8; 4]; 16];
                    for py in 0..4 {
                        let y = (by * 4 + py).min(h - 1);
                        for px in 0..4 {
                            let x = (bx * 4 + px).min(w - 1);
                            let idx = ((z * h + y) * w + x) * ch;
                            let t = &mut tile[py * 4 + px];
                            t[..ch].copy_from_slice(&pixels[idx..idx + ch]);
                            if ch < 4 {
                                t[3] = 255;
                            }
                        }
                    }
                    let block_idx = z * layer_blocks + by * bx_count + bx;
                    let dst = &mut out[block_idx * bsize..(block_idx + 1) * bsize];
                    match format {
                        BlockFormat::Bc1RgbaUnorm => encode_bc1_block(&tile, dst),
                        BlockFormat::Bc4RUnorm => {
                            let r = tile.map(|p| p[0]);
                            encode_bc4_block(&r, dst);
                        }
                        BlockFormat::Bc5RgUnorm => {
                            let r = tile.map(|p| p[0]);
                            let g = tile.map(|p| p[1]);
                            encode_bc4_block(&r, &mut dst[0..8]);
                            encode_bc4_block(&g, &mut dst[8..16]);
                        }
                        BlockFormat::Bc7RgbaUnorm => encode_bc7_block(&tile, dst),
                    }
                }
            }
        }
        Self::new(format, domain, out)
    }
}

// ---------------------------------------------------------------------------
// BC1 (DXT1)
// ---------------------------------------------------------------------------

fn rgb565_to_rgb8(c: u16) -> [u8; 3] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
    ]
}

fn rgb8_to_rgb565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = ((u16::from(r) * 31 + 127) / 255) & 0x1F;
    let g6 = ((u16::from(g) * 63 + 127) / 255) & 0x3F;
    let b5 = ((u16::from(b) * 31 + 127) / 255) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

fn decode_bc1_block(src: &[u8], out: &mut [[u8; 4]; 16]) {
    let c0 = u16::from_le_bytes([src[0], src[1]]);
    let c1 = u16::from_le_bytes([src[2], src[3]]);
    let bits = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
    let p0 = rgb565_to_rgb8(c0);
    let p1 = rgb565_to_rgb8(c1);
    let palette: [[u8; 4]; 4] = if c0 > c1 {
        [
            [p0[0], p0[1], p0[2], 255],
            [p1[0], p1[1], p1[2], 255],
            [
                ((2 * u16::from(p0[0]) + u16::from(p1[0]) + 1) / 3) as u8,
                ((2 * u16::from(p0[1]) + u16::from(p1[1]) + 1) / 3) as u8,
                ((2 * u16::from(p0[2]) + u16::from(p1[2]) + 1) / 3) as u8,
                255,
            ],
            [
                ((u16::from(p0[0]) + 2 * u16::from(p1[0]) + 1) / 3) as u8,
                ((u16::from(p0[1]) + 2 * u16::from(p1[1]) + 1) / 3) as u8,
                ((u16::from(p0[2]) + 2 * u16::from(p1[2]) + 1) / 3) as u8,
                255,
            ],
        ]
    } else {
        [
            [p0[0], p0[1], p0[2], 255],
            [p1[0], p1[1], p1[2], 255],
            [
                ((u16::from(p0[0]) + u16::from(p1[0])) / 2) as u8,
                ((u16::from(p0[1]) + u16::from(p1[1])) / 2) as u8,
                ((u16::from(p0[2]) + u16::from(p1[2])) / 2) as u8,
                255,
            ],
            [0, 0, 0, 0],
        ]
    };
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = palette[((bits >> (i * 2)) & 3) as usize];
    }
}

fn encode_bc1_block(tile: &[[u8; 4]; 16], dst: &mut [u8]) {
    let has_punch = tile.iter().any(|p| p[3] < 128);
    let mut mean = [0.0f32; 3];
    let mut n = 0.0f32;
    for p in tile {
        if has_punch && p[3] < 128 {
            continue;
        }
        for c in 0..3 {
            mean[c] += f32::from(p[c]);
        }
        n += 1.0;
    }
    if n == 0.0 {
        dst[..8].copy_from_slice(&[0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]);
        return;
    }
    for m in &mut mean {
        *m /= n;
    }
    // Principal RGB direction (handles anti-correlated color gradients cleanly).
    let mut axis = [1.0f32, 1.0, 1.0];
    let mut best_d2 = -1.0f32;
    for p in tile {
        if has_punch && p[3] < 128 {
            continue;
        }
        let d = [
            f32::from(p[0]) - mean[0],
            f32::from(p[1]) - mean[1],
            f32::from(p[2]) - mean[2],
        ];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 > best_d2 {
            best_d2 = d2;
            axis = d;
        }
    }
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    let mut e0 = [0u8; 3];
    let mut e1 = [0u8; 3];
    for p in tile {
        if has_punch && p[3] < 128 {
            continue;
        }
        let proj = (f32::from(p[0]) - mean[0]) * axis[0]
            + (f32::from(p[1]) - mean[1]) * axis[1]
            + (f32::from(p[2]) - mean[2]) * axis[2];
        if proj < t_min {
            t_min = proj;
            e1 = [p[0], p[1], p[2]];
        }
        if proj > t_max {
            t_max = proj;
            e0 = [p[0], p[1], p[2]];
        }
    }
    let mut c0 = rgb8_to_rgb565(e0[0], e0[1], e0[2]);
    let mut c1 = rgb8_to_rgb565(e1[0], e1[1], e1[2]);
    if has_punch {
        if c0 > c1 {
            std::mem::swap(&mut c0, &mut c1);
        }
    } else if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    } else if c0 == c1 {
        let bits = 0u32;
        dst[0..2].copy_from_slice(&c0.to_le_bytes());
        dst[2..4].copy_from_slice(&c1.to_le_bytes());
        dst[4..8].copy_from_slice(&bits.to_le_bytes());
        return;
    }
    let p0 = rgb565_to_rgb8(c0);
    let p1 = rgb565_to_rgb8(c1);
    let pal: [[i32; 3]; 4] = if c0 > c1 {
        [
            [i32::from(p0[0]), i32::from(p0[1]), i32::from(p0[2])],
            [i32::from(p1[0]), i32::from(p1[1]), i32::from(p1[2])],
            [
                (2 * i32::from(p0[0]) + i32::from(p1[0]) + 1) / 3,
                (2 * i32::from(p0[1]) + i32::from(p1[1]) + 1) / 3,
                (2 * i32::from(p0[2]) + i32::from(p1[2]) + 1) / 3,
            ],
            [
                (i32::from(p0[0]) + 2 * i32::from(p1[0]) + 1) / 3,
                (i32::from(p0[1]) + 2 * i32::from(p1[1]) + 1) / 3,
                (i32::from(p0[2]) + 2 * i32::from(p1[2]) + 1) / 3,
            ],
        ]
    } else {
        [
            [i32::from(p0[0]), i32::from(p0[1]), i32::from(p0[2])],
            [i32::from(p1[0]), i32::from(p1[1]), i32::from(p1[2])],
            [
                (i32::from(p0[0]) + i32::from(p1[0])) / 2,
                (i32::from(p0[1]) + i32::from(p1[1])) / 2,
                (i32::from(p0[2]) + i32::from(p1[2])) / 2,
            ],
            [0, 0, 0],
        ]
    };
    let count = if has_punch { 3 } else { 4 };
    let mut bits = 0u32;
    for (i, p) in tile.iter().enumerate() {
        let best = if has_punch && p[3] < 128 {
            3u32
        } else {
            let mut best_idx = 0u32;
            let mut best_err = i32::MAX;
            for (k, c) in pal.iter().enumerate().take(count) {
                let dr = i32::from(p[0]) - c[0];
                let dg = i32::from(p[1]) - c[1];
                let db = i32::from(p[2]) - c[2];
                let err = dr * dr + dg * dg + db * db;
                if err < best_err {
                    best_err = err;
                    best_idx = k as u32;
                }
            }
            best_idx
        };
        bits |= best << (i * 2);
    }
    dst[0..2].copy_from_slice(&c0.to_le_bytes());
    dst[2..4].copy_from_slice(&c1.to_le_bytes());
    dst[4..8].copy_from_slice(&bits.to_le_bytes());
}

// ---------------------------------------------------------------------------
// BC4 / BC5
// ---------------------------------------------------------------------------

fn decode_bc4_block(src: &[u8]) -> [u8; 16] {
    let e0 = src[0];
    let e1 = src[1];
    let mut pal = [0u8; 8];
    pal[0] = e0;
    pal[1] = e1;
    let a0 = u16::from(e0);
    let a1 = u16::from(e1);
    if e0 > e1 {
        for k in 1..=6u16 {
            pal[(k + 1) as usize] = (((7 - k) * a0 + k * a1 + 3) / 7) as u8;
        }
    } else {
        for k in 1..=4u16 {
            pal[(k + 1) as usize] = (((5 - k) * a0 + k * a1 + 2) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut idx_bits = 0u64;
    for (i, &b) in src[2..8].iter().enumerate() {
        idx_bits |= u64::from(b) << (i * 8);
    }
    let mut out = [0u8; 16];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = pal[((idx_bits >> (i * 3)) & 7) as usize];
    }
    out
}

fn encode_bc4_block(vals: &[u8; 16], dst: &mut [u8]) {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for &v in vals {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let mut pal = [0u8; 8];
    let (e0, e1) = (hi, lo);
    pal[0] = e0;
    pal[1] = e1;
    let a0 = u16::from(e0);
    let a1 = u16::from(e1);
    if e0 > e1 {
        for k in 1..=6u16 {
            pal[(k + 1) as usize] = (((7 - k) * a0 + k * a1 + 3) / 7) as u8;
        }
    } else {
        for k in 1..=4u16 {
            pal[(k + 1) as usize] = (((5 - k) * a0 + k * a1 + 2) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut idx_bits = 0u64;
    for (i, &v) in vals.iter().enumerate() {
        let mut best_idx = 0u64;
        let mut best_err = i32::MAX;
        for (k, &c) in pal.iter().enumerate() {
            let err = (i32::from(v) - i32::from(c)).abs();
            if err < best_err {
                best_err = err;
                best_idx = k as u64;
            }
        }
        idx_bits |= best_idx << (i * 3);
    }
    dst[0] = e0;
    dst[1] = e1;
    for i in 0..6 {
        dst[2 + i] = ((idx_bits >> (i * 8)) & 0xFF) as u8;
    }
}

// ---------------------------------------------------------------------------
// BC7 (All 8 modes decoded per Khronos Data Format Specification §17;
// Mode 6 high-precision RGBA encoder with 16-level interpolation).
// ---------------------------------------------------------------------------

const BC7_W2: [u16; 4] = [0, 21, 43, 64];
const BC7_W3: [u16; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const BC7_W4: [u16; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// BC7 2-subset partition table (64 partitions x 16 texels, 1 bit/texel packed
/// LSB-first into `u16`).
const BC7_PART2: [u16; 64] = [
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800,
    0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
    0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA, 0xF0F0, 0x5A5A, 0x33CC,
    0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
    0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C, 0x9336, 0x9CC6, 0x817E, 0xE718,
    0xCCF0, 0x0FCC, 0x7744, 0xEE22,
];

/// BC7 3-subset partition table (64 partitions x 16 texels, 2 bits/texel packed
/// LSB-first into `u32`).
const BC7_PART3: [u32; 64] = [
    0xAA68_5050,
    0x6A5A_5040,
    0x5A5A_4200,
    0x5450_A0A8,
    0xA5A5_0000,
    0xA0A0_5050,
    0x5555_A0A0,
    0x5A5A_5050,
    0xAA55_0000,
    0xAA55_5500,
    0xAAAA_5500,
    0x9090_9090,
    0x9494_9494,
    0xA4A4_A4A4,
    0xA9A5_9450,
    0x2A0A_4250,
    0xA594_5040,
    0x0A42_5054,
    0xA5A5_A500,
    0x55A0_A0A0,
    0xA8A8_5454,
    0x6A6A_4040,
    0xA4A4_5000,
    0x1A1A_0500,
    0x0050_A4A4,
    0xAAA5_9090,
    0x1469_6914,
    0x6969_1400,
    0xA085_85A0,
    0xAA82_1414,
    0x50A4_A450,
    0x6A5A_0200,
    0xA9A5_8000,
    0x5090_A0A8,
    0xA8A0_9050,
    0x2424_2424,
    0x00AA_5500,
    0x2492_4924,
    0x2449_9224,
    0x50A5_0A50,
    0x500A_A550,
    0xAAAA_4444,
    0x6666_0000,
    0xA5A0_A5A0,
    0x50A0_50A0,
    0x6928_6928,
    0x44AA_44AA,
    0x6666_6600,
    0xAA44_4444,
    0x54A8_54A8,
    0x9580_9580,
    0x9696_9600,
    0xA854_54A8,
    0x8095_9580,
    0xAA14_1414,
    0x9696_0000,
    0xAAAA_1414,
    0xA050_50A0,
    0xA0A5_A5A0,
    0x9600_0000,
    0x4080_4080,
    0xA9A8_A9A8,
    0xAAAA_AA44,
    0x2A4A_5254,
];

/// Second anchor index for 2-subset partitions (`0..64`).
const BC7_ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// Second and third anchor indices for 3-subset partitions (`0..64`).
const BC7_ANCHOR3_2: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const BC7_ANCHOR3_3: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

struct BitReader {
    bits: u128,
    pos: u32,
}

impl BitReader {
    fn new(src: &[u8]) -> Self {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&src[..16]);
        Self {
            bits: u128::from_le_bytes(arr),
            pos: 0,
        }
    }

    fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let val = ((self.bits >> self.pos) & ((1u128 << n) - 1)) as u32;
        self.pos += n;
        val
    }
}

struct BitWriter {
    bits: u128,
    pos: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self { bits: 0, pos: 0 }
    }

    fn write(&mut self, val: u32, n: u32) {
        if n == 0 {
            return;
        }
        let mask = (1u128 << n) - 1;
        self.bits |= (u128::from(val) & mask) << self.pos;
        self.pos += n;
    }

    fn finish(self, dst: &mut [u8]) {
        dst[..16].copy_from_slice(&self.bits.to_le_bytes());
    }
}

fn bc7_unquantize(val: u32, bits: u32) -> u8 {
    let shift = 8 - bits;
    let hi = val << shift;
    (hi | (hi >> bits)) as u8
}

fn bc7_interp(e0: u8, e1: u8, idx: usize, idx_bits: u32) -> u8 {
    let w = match idx_bits {
        2 => BC7_W2[idx],
        3 => BC7_W3[idx],
        4 => BC7_W4[idx],
        _ => 0,
    };
    (((64 - w) * u16::from(e0) + w * u16::from(e1) + 32) >> 6) as u8
}

struct Bc7Mode {
    subsets: usize,
    part_bits: u32,
    rot_bits: u32,
    idx_sel_bits: u32,
    col_bits: u32,
    alp_bits: u32,
    ep_pbits: bool,
    sp_pbits: bool,
    idx1_bits: u32,
    idx2_bits: u32,
}

const BC7_MODES: [Bc7Mode; 8] = [
    Bc7Mode {
        subsets: 3,
        part_bits: 4,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 4,
        alp_bits: 0,
        ep_pbits: true,
        sp_pbits: false,
        idx1_bits: 3,
        idx2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        part_bits: 6,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 6,
        alp_bits: 0,
        ep_pbits: false,
        sp_pbits: true,
        idx1_bits: 3,
        idx2_bits: 0,
    },
    Bc7Mode {
        subsets: 3,
        part_bits: 6,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 5,
        alp_bits: 0,
        ep_pbits: false,
        sp_pbits: false,
        idx1_bits: 2,
        idx2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        part_bits: 6,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 7,
        alp_bits: 0,
        ep_pbits: true,
        sp_pbits: false,
        idx1_bits: 2,
        idx2_bits: 0,
    },
    Bc7Mode {
        subsets: 1,
        part_bits: 0,
        rot_bits: 2,
        idx_sel_bits: 1,
        col_bits: 5,
        alp_bits: 6,
        ep_pbits: false,
        sp_pbits: false,
        idx1_bits: 2,
        idx2_bits: 3,
    },
    Bc7Mode {
        subsets: 1,
        part_bits: 0,
        rot_bits: 2,
        idx_sel_bits: 0,
        col_bits: 7,
        alp_bits: 8,
        ep_pbits: false,
        sp_pbits: false,
        idx1_bits: 2,
        idx2_bits: 2,
    },
    Bc7Mode {
        subsets: 1,
        part_bits: 0,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 7,
        alp_bits: 7,
        ep_pbits: true,
        sp_pbits: false,
        idx1_bits: 4,
        idx2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        part_bits: 6,
        rot_bits: 0,
        idx_sel_bits: 0,
        col_bits: 5,
        alp_bits: 5,
        ep_pbits: true,
        sp_pbits: false,
        idx1_bits: 2,
        idx2_bits: 0,
    },
];

fn decode_bc7_block(src: &[u8], out: &mut [[u8; 4]; 16]) {
    let mode_idx = src[0].trailing_zeros() as usize;
    if mode_idx >= 8 {
        out.fill([0, 0, 0, 0]);
        return;
    }
    let m = &BC7_MODES[mode_idx];
    let mut r = BitReader::new(src);
    r.read((mode_idx + 1) as u32);
    let part = r.read(m.part_bits) as usize;
    let rot = r.read(m.rot_bits);
    let idx_sel = r.read(m.idx_sel_bits);

    let ep_count = m.subsets * 2;
    let mut raw = [[0u32; 4]; 6];
    for ch in 0..3 {
        for ep in raw.iter_mut().take(ep_count) {
            ep[ch] = r.read(m.col_bits);
        }
    }
    if m.alp_bits > 0 {
        for ep in raw.iter_mut().take(ep_count) {
            ep[3] = r.read(m.alp_bits);
        }
    }

    let mut ep_u8 = [[255u8; 4]; 6];
    if m.ep_pbits {
        for i in 0..ep_count {
            let p = r.read(1);
            for ch in 0..3 {
                ep_u8[i][ch] = bc7_unquantize((raw[i][ch] << 1) | p, m.col_bits + 1);
            }
            if m.alp_bits > 0 {
                ep_u8[i][3] = bc7_unquantize((raw[i][3] << 1) | p, m.alp_bits + 1);
            }
        }
    } else if m.sp_pbits {
        for s in 0..m.subsets {
            let p = r.read(1);
            for k in 0..2 {
                let i = s * 2 + k;
                for ch in 0..3 {
                    ep_u8[i][ch] = bc7_unquantize((raw[i][ch] << 1) | p, m.col_bits + 1);
                }
            }
        }
    } else {
        for i in 0..ep_count {
            for ch in 0..3 {
                ep_u8[i][ch] = bc7_unquantize(raw[i][ch], m.col_bits);
            }
            if m.alp_bits > 0 {
                ep_u8[i][3] = bc7_unquantize(raw[i][3], m.alp_bits);
            }
        }
    }

    let subset_of = |i: usize| -> usize {
        match m.subsets {
            1 => 0,
            2 => ((BC7_PART2[part] >> i) & 1) as usize,
            _ => ((BC7_PART3[part] >> (i * 2)) & 3) as usize,
        }
    };
    let is_anchor = |i: usize| -> bool {
        if i == 0 {
            return true;
        }
        match m.subsets {
            1 => false,
            2 => i == BC7_ANCHOR2[part] as usize,
            _ => i == BC7_ANCHOR3_2[part] as usize || i == BC7_ANCHOR3_3[part] as usize,
        }
    };

    let mut idx1 = [0usize; 16];
    for (i, slot) in idx1.iter_mut().enumerate() {
        let bits = if is_anchor(i) {
            m.idx1_bits - 1
        } else {
            m.idx1_bits
        };
        *slot = r.read(bits) as usize;
    }
    let mut idx2 = [0usize; 16];
    if m.idx2_bits > 0 {
        for (i, slot) in idx2.iter_mut().enumerate() {
            let bits = if i == 0 { m.idx2_bits - 1 } else { m.idx2_bits };
            *slot = r.read(bits) as usize;
        }
    }

    for (i, dst) in out.iter_mut().enumerate() {
        let s = subset_of(i);
        let e0 = ep_u8[s * 2];
        let e1 = ep_u8[s * 2 + 1];
        let (ci, cb, ai, ab) = if m.idx2_bits == 0 {
            (idx1[i], m.idx1_bits, idx1[i], m.idx1_bits)
        } else if idx_sel == 0 {
            (idx1[i], m.idx1_bits, idx2[i], m.idx2_bits)
        } else {
            (idx2[i], m.idx2_bits, idx1[i], m.idx1_bits)
        };
        let mut px = [
            bc7_interp(e0[0], e1[0], ci, cb),
            bc7_interp(e0[1], e1[1], ci, cb),
            bc7_interp(e0[2], e1[2], ci, cb),
            if m.alp_bits > 0 {
                bc7_interp(e0[3], e1[3], ai, ab)
            } else {
                255
            },
        ];
        match rot {
            1 => px.swap(0, 3),
            2 => px.swap(1, 3),
            3 => px.swap(2, 3),
            _ => {}
        }
        *dst = px;
    }
}

/// Evaluates a candidate BC7 Mode 6 block and returns its encoded bytes + SSE.
fn encode_bc7_mode6_candidate(tile: &[[u8; 4]; 16]) -> ([u8; 16], u64) {
    let mut mean = [0.0f32; 4];
    for p in tile {
        for c in 0..4 {
            mean[c] += f32::from(p[c]);
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }

    // Power-iteration principal direction in 4D RGBA space (seeded from max-deviation pixel).
    let mut axis = [1.0f32, 1.0, 1.0, 0.25];
    let mut best_d2 = -1.0f32;
    for p in tile {
        let mut d = [0.0f32; 4];
        let mut d2 = 0.0f32;
        for c in 0..4 {
            d[c] = f32::from(p[c]) - mean[c];
            d2 += d[c] * d[c];
        }
        if d2 > best_d2 {
            best_d2 = d2;
            axis = d;
        }
    }
    if best_d2 > 1e-4 {
        for _ in 0..3 {
            let mut next = [0.0f32; 4];
            for p in tile {
                let mut proj = 0.0f32;
                for c in 0..4 {
                    proj += (f32::from(p[c]) - mean[c]) * axis[c];
                }
                for c in 0..4 {
                    next[c] += (f32::from(p[c]) - mean[c]) * proj;
                }
            }
            let len =
                (next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3])
                    .sqrt();
            if len > 1e-4 {
                for c in 0..4 {
                    axis[c] = next[c] / len;
                }
            }
        }
    }

    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    let mut e0_f = [
        f32::from(tile[0][0]),
        f32::from(tile[0][1]),
        f32::from(tile[0][2]),
        f32::from(tile[0][3]),
    ];
    let mut e1_f = e0_f;
    for p in tile {
        let proj = (f32::from(p[0]) - mean[0]) * axis[0]
            + (f32::from(p[1]) - mean[1]) * axis[1]
            + (f32::from(p[2]) - mean[2]) * axis[2]
            + (f32::from(p[3]) - mean[3]) * axis[3];
        if proj < t_min {
            t_min = proj;
            e0_f = [
                f32::from(p[0]),
                f32::from(p[1]),
                f32::from(p[2]),
                f32::from(p[3]),
            ];
        }
        if proj > t_max {
            t_max = proj;
            e1_f = [
                f32::from(p[0]),
                f32::from(p[1]),
                f32::from(p[2]),
                f32::from(p[3]),
            ];
        }
    }

    // Two least-squares refinement passes over the 4-bit BC7 weights.
    let mut indices = [0usize; 16];
    for _ in 0..2 {
        let mut dir = [0.0f32; 4];
        let mut len2 = 0.0f32;
        for c in 0..4 {
            dir[c] = e1_f[c] - e0_f[c];
            len2 += dir[c] * dir[c];
        }
        if len2 < 1e-4 {
            indices.fill(0);
            break;
        }
        for (i, p) in tile.iter().enumerate() {
            let dot = (f32::from(p[0]) - e0_f[0]) * dir[0]
                + (f32::from(p[1]) - e0_f[1]) * dir[1]
                + (f32::from(p[2]) - e0_f[2]) * dir[2]
                + (f32::from(p[3]) - e0_f[3]) * dir[3];
            let t = (dot / len2).clamp(0.0, 1.0);
            indices[i] = (t * 15.0).round() as usize;
        }
        let (mut a00, mut a01, mut a11) = (0.0f32, 0.0f32, 0.0f32);
        let mut b0 = [0.0f32; 4];
        let mut b1 = [0.0f32; 4];
        for (i, p) in tile.iter().enumerate() {
            let w1 = f32::from(BC7_W4[indices[i]]) / 64.0;
            let w0 = 1.0 - w1;
            a00 += w0 * w0;
            a01 += w0 * w1;
            a11 += w1 * w1;
            for c in 0..4 {
                b0[c] += w0 * f32::from(p[c]);
                b1[c] += w1 * f32::from(p[c]);
            }
        }
        let det = a00 * a11 - a01 * a01;
        if det > 1e-4 {
            for c in 0..4 {
                e0_f[c] = ((a11 * b0[c] - a01 * b1[c]) / det).clamp(0.0, 255.0);
                e1_f[c] = ((a00 * b1[c] - a01 * b0[c]) / det).clamp(0.0, 255.0);
            }
        }
    }

    let mut q0 = [0u32; 4];
    let mut q1 = [0u32; 4];
    let mut p0_votes = 0i32;
    let mut p1_votes = 0i32;
    for c in 0..4 {
        let v0 = e0_f[c].round() as u32;
        let v1 = e1_f[c].round() as u32;
        q0[c] = v0 >> 1;
        q1[c] = v1 >> 1;
        p0_votes += if (v0 & 1) != 0 { 1 } else { -1 };
        p1_votes += if (v1 & 1) != 0 { 1 } else { -1 };
    }
    let mut p0 = u32::from(p0_votes >= 0);
    let mut p1 = u32::from(p1_votes >= 0);

    let mut u0 = [0u8; 4];
    let mut u1 = [0u8; 4];
    for c in 0..4 {
        u0[c] = bc7_unquantize((q0[c] << 1) | p0, 8);
        u1[c] = bc7_unquantize((q1[c] << 1) | p1, 8);
    }
    let mut pal = [[0i32; 4]; 16];
    for (k, entry) in pal.iter_mut().enumerate() {
        for c in 0..4 {
            entry[c] = i32::from(bc7_interp(u0[c], u1[c], k, 4));
        }
    }
    let mut total_sse = 0u64;
    for (i, p) in tile.iter().enumerate() {
        let mut best_k = 0usize;
        let mut best_err = i32::MAX;
        for (k, entry) in pal.iter().enumerate() {
            let d0 = i32::from(p[0]) - entry[0];
            let d1 = i32::from(p[1]) - entry[1];
            let d2 = i32::from(p[2]) - entry[2];
            let d3 = i32::from(p[3]) - entry[3];
            let err = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            if err < best_err {
                best_err = err;
                best_k = k;
            }
        }
        indices[i] = best_k;
        total_sse += best_err as u64;
    }

    if indices[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut p0, &mut p1);
        for idx in &mut indices {
            *idx = 15 - *idx;
        }
    }

    let mut w = BitWriter::new();
    w.write(1 << 6, 7);
    for c in 0..4 {
        w.write(q0[c], 7);
        w.write(q1[c], 7);
    }
    w.write(p0, 1);
    w.write(p1, 1);
    w.write(indices[0] as u32, 3);
    for &idx in &indices[1..] {
        w.write(idx as u32, 4);
    }
    debug_assert_eq!(w.pos, 128);
    let mut out = [0u8; 16];
    w.finish(&mut out);
    (out, total_sse)
}

/// Evaluates a BC7 Mode 7 (2-subset RGBA, 5.5.5.5 + P-bit endpoints, 2-bit
/// indices) block for partition `part`, returning encoded bytes + SSE.
fn encode_bc7_mode7_candidate(tile: &[[u8; 4]; 16], part: usize) -> ([u8; 16], u64) {
    let mask = BC7_PART2[part];
    let anchor1 = BC7_ANCHOR2[part] as usize;
    let mut q = [[0u32; 4]; 4];
    let mut pb = [0u32; 4];
    let mut indices = [0usize; 16];
    let mut total_sse = 0u64;

    for s in 0..2usize {
        let mut lo = [255u8; 4];
        let mut hi = [0u8; 4];
        for (i, p) in tile.iter().enumerate() {
            if (((mask >> i) & 1) as usize) != s {
                continue;
            }
            for c in 0..4 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        let mut e0_f = [
            f32::from(lo[0]),
            f32::from(lo[1]),
            f32::from(lo[2]),
            f32::from(lo[3]),
        ];
        let mut e1_f = [
            f32::from(hi[0]),
            f32::from(hi[1]),
            f32::from(hi[2]),
            f32::from(hi[3]),
        ];
        // One least-squares refinement step on 2-bit weights.
        let mut dir = [0.0f32; 4];
        let mut len2 = 0.0f32;
        for c in 0..4 {
            dir[c] = e1_f[c] - e0_f[c];
            len2 += dir[c] * dir[c];
        }
        if len2 > 1e-4 {
            let (mut a00, mut a01, mut a11) = (0.0f32, 0.0f32, 0.0f32);
            let mut b0 = [0.0f32; 4];
            let mut b1 = [0.0f32; 4];
            for (i, p) in tile.iter().enumerate() {
                if (((mask >> i) & 1) as usize) != s {
                    continue;
                }
                let dot = (f32::from(p[0]) - e0_f[0]) * dir[0]
                    + (f32::from(p[1]) - e0_f[1]) * dir[1]
                    + (f32::from(p[2]) - e0_f[2]) * dir[2]
                    + (f32::from(p[3]) - e0_f[3]) * dir[3];
                let k = ((dot / len2).clamp(0.0, 1.0) * 3.0).round() as usize;
                let w1 = f32::from(BC7_W2[k]) / 64.0;
                let w0 = 1.0 - w1;
                a00 += w0 * w0;
                a01 += w0 * w1;
                a11 += w1 * w1;
                for c in 0..4 {
                    b0[c] += w0 * f32::from(p[c]);
                    b1[c] += w1 * f32::from(p[c]);
                }
            }
            let det = a00 * a11 - a01 * a01;
            if det > 1e-4 {
                for c in 0..4 {
                    e0_f[c] = ((a11 * b0[c] - a01 * b1[c]) / det).clamp(0.0, 255.0);
                    e1_f[c] = ((a00 * b1[c] - a01 * b0[c]) / det).clamp(0.0, 255.0);
                }
            }
        }
        let mut q0 = [0u32; 4];
        let mut q1 = [0u32; 4];
        let mut v0_p = 0i32;
        let mut v1_p = 0i32;
        for c in 0..4 {
            let v0 = ((e0_f[c] * 63.0 / 255.0).round() as u32).min(63);
            let v1 = ((e1_f[c] * 63.0 / 255.0).round() as u32).min(63);
            q0[c] = v0 >> 1;
            q1[c] = v1 >> 1;
            v0_p += if (v0 & 1) != 0 { 1 } else { -1 };
            v1_p += if (v1 & 1) != 0 { 1 } else { -1 };
        }
        let mut p0 = u32::from(v0_p >= 0);
        let mut p1 = u32::from(v1_p >= 0);
        let mut u0 = [0u8; 4];
        let mut u1 = [0u8; 4];
        for c in 0..4 {
            u0[c] = bc7_unquantize((q0[c] << 1) | p0, 6);
            u1[c] = bc7_unquantize((q1[c] << 1) | p1, 6);
        }
        let mut pal = [[0i32; 4]; 4];
        for (k, entry) in pal.iter_mut().enumerate() {
            for c in 0..4 {
                entry[c] = i32::from(bc7_interp(u0[c], u1[c], k, 2));
            }
        }
        for (i, p) in tile.iter().enumerate() {
            if (((mask >> i) & 1) as usize) != s {
                continue;
            }
            let mut best_k = 0usize;
            let mut best_err = i32::MAX;
            for (k, entry) in pal.iter().enumerate() {
                let d0 = i32::from(p[0]) - entry[0];
                let d1 = i32::from(p[1]) - entry[1];
                let d2 = i32::from(p[2]) - entry[2];
                let d3 = i32::from(p[3]) - entry[3];
                let err = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                if err < best_err {
                    best_err = err;
                    best_k = k;
                }
            }
            indices[i] = best_k;
            total_sse += best_err as u64;
        }
        let anchor = if s == 0 { 0 } else { anchor1 };
        if indices[anchor] >= 2 {
            std::mem::swap(&mut q0, &mut q1);
            std::mem::swap(&mut p0, &mut p1);
            for (i, idx) in indices.iter_mut().enumerate() {
                if (((mask >> i) & 1) as usize) == s {
                    *idx = 3 - *idx;
                }
            }
        }
        q[s * 2] = q0;
        q[s * 2 + 1] = q1;
        pb[s * 2] = p0;
        pb[s * 2 + 1] = p1;
    }

    let mut w = BitWriter::new();
    w.write(1 << 7, 8);
    w.write(part as u32, 6);
    for ch in 0..4 {
        for ep in &q {
            w.write(ep[ch], 5);
        }
    }
    for p in pb {
        w.write(p, 1);
    }
    for (i, &idx) in indices.iter().enumerate() {
        let bits = if i == 0 || i == anchor1 { 1 } else { 2 };
        w.write(idx as u32, bits);
    }
    debug_assert_eq!(w.pos, 128);
    let mut out = [0u8; 16];
    w.finish(&mut out);
    (out, total_sse)
}

/// Multi-mode BC7 encoder: tries Mode 6 (16-level RGBA) and Mode 7 across key
/// horizontal/vertical/diagonal 2-subset partitions when Mode 6 has residual
/// error, keeping the lowest-SSE block.
fn encode_bc7_block(tile: &[[u8; 4]; 16], dst: &mut [u8]) {
    let (mut best_bytes, mut best_sse) = encode_bc7_mode6_candidate(tile);
    if best_sse > 256 {
        // Vertical, horizontal, diagonal, and corner 2-subset partitions.
        for part in [0usize, 13, 2, 14, 27, 28, 3, 7] {
            let (cand, sse) = encode_bc7_mode7_candidate(tile, part);
            if sse < best_sse {
                best_bytes = cand;
                best_sse = sse;
            }
        }
    }
    dst[..16].copy_from_slice(&best_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc4_and_bc5_round_trip_smooth_ramps_tightly() {
        let domain = Domain::Grid(13, 9);
        let w = domain.width() as usize;
        let h = domain.height() as usize;
        let mut gray = vec![0u8; w * h];
        let mut rg = vec![0u8; w * h * 2];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / (w - 1)) as u8;
                let g = ((y * 255) / (h - 1)) as u8;
                gray[y * w + x] = r;
                rg[(y * w + x) * 2] = r;
                rg[(y * w + x) * 2 + 1] = g;
            }
        }
        let bc4 = CompressedBlocks::encode_unorm8(BlockFormat::Bc4RUnorm, domain, &gray).unwrap();
        let bc4_words = bc4.decode_storage(domain);
        let bc4_bytes: Vec<u8> = bc4_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(w * h)
            .collect();
        for (a, b) in gray.iter().zip(&bc4_bytes) {
            assert!((*a as i32 - *b as i32).abs() <= 7, "bc4 diff {a} vs {b}");
        }

        let bc5 = CompressedBlocks::encode_unorm8(BlockFormat::Bc5RgUnorm, domain, &rg).unwrap();
        let bc5_words = bc5.decode_storage(domain);
        let bc5_bytes: Vec<u8> = bc5_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(w * h * 2)
            .collect();
        for (a, b) in rg.iter().zip(&bc5_bytes) {
            assert!((*a as i32 - *b as i32).abs() <= 7, "bc5 diff {a} vs {b}");
        }
    }

    #[test]
    fn bc1_and_bc7_round_trip_color_and_alpha() {
        let domain = Domain::Grid(11, 7);
        let w = domain.width() as usize;
        let h = domain.height() as usize;
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 4;
                rgba[base] = ((x * 240) / (w - 1)) as u8;
                rgba[base + 1] = ((y * 220) / (h - 1)) as u8;
                rgba[base + 2] = 160;
                rgba[base + 3] = 200;
            }
        }
        let bc7 =
            CompressedBlocks::encode_unorm8(BlockFormat::Bc7RgbaUnorm, domain, &rgba).unwrap();
        assert_eq!(bc7.bytes().len(), 3 * 2 * 16);
        let bc7_words = bc7.decode_storage(domain);
        let bc7_bytes: Vec<u8> = bc7_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(w * h * 4)
            .collect();
        let max_err = rgba
            .iter()
            .zip(&bc7_bytes)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(max_err <= 22, "bc7 max error {max_err}");

        // Opaque BC1 check on a smooth 1D color ramp.
        let mut ramp = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 4;
                let t = ((x + y) * 240) / (w + h - 2);
                ramp[base] = t as u8;
                ramp[base + 1] = (255 - t) as u8;
                ramp[base + 2] = (t / 2 + 64) as u8;
                ramp[base + 3] = 255;
            }
        }
        let bc1 =
            CompressedBlocks::encode_unorm8(BlockFormat::Bc1RgbaUnorm, domain, &ramp).unwrap();
        assert_eq!(bc1.bytes().len(), 3 * 2 * 8);
        let bc1_words = bc1.decode_storage(domain);
        let bc1_bytes: Vec<u8> = bc1_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(w * h * 4)
            .collect();
        let max_bc1 = ramp
            .iter()
            .zip(&bc1_bytes)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(max_bc1 <= 16, "bc1 max error {max_bc1}");
    }

    #[test]
    fn bc7_decodes_all_eight_modes_without_panicking_and_matches_known_constants() {
        for mode in 0..8usize {
            let mut block = [0u8; 16];
            block[0] = 1u8 << mode;
            let mut out = [[99u8; 4]; 16];
            decode_bc7_block(&block, &mut out);
            // All-zero endpoints and indices yield R=G=B=0 and alpha=255 for opaque modes 0..=3, 0 for alpha modes 4..=7.
            let expected_a = if mode <= 3 { 255 } else { 0 };
            for px in out {
                assert_eq!(px, [0, 0, 0, expected_a], "mode {mode}");
            }
        }
        // Invalid mode byte (0x00) returns transparent black per Khronos spec.
        let mut out = [[99u8; 4]; 16];
        decode_bc7_block(&[0u8; 16], &mut out);
        assert_eq!(out, [[0, 0, 0, 0]; 16]);
    }
}
