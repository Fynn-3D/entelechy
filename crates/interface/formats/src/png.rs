//! PNG as Fields: decodes to a pixel grid or a `Grid<Float>` of samples
//! (bit depths 1 to 16, every color type, palettes; interlaced files are
//! refused); encodes a pixel grid.

use std::fmt;

use orla_host::{Domain, FieldData, HostError, Scalar, Schema};

use crate::deflate::zlib;
use crate::inflate::{InflateError, unzlib};

/// Why bytes are not a PNG the Host can take.
#[derive(Debug)]
pub enum PngError {
    /// The file is malformed or uses an unsupported feature.
    Format(String),
    /// The compressed data is broken.
    Inflate(InflateError),
    /// The Host refused the Field.
    Host(HostError),
}

impl fmt::Display for PngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(message) => write!(f, "not a PNG the Host can read: {message}"),
            Self::Inflate(error) => write!(f, "{error}"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PngError {}

/// A decoded PNG: samples in `0..=1`, `channels` per pixel (1 gray,
/// 2 gray+alpha, 3 RGB, 4 RGBA).
pub struct Decoded {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Samples per pixel.
    pub channels: usize,
    /// Samples row by row.
    pub samples: Vec<f32>,
}

/// CRC-32 as PNG uses it.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Decodes a PNG to samples.
pub fn decode(bytes: &[u8]) -> Result<Decoded, PngError> {
    decode_within(bytes, usize::MAX)
}

/// [`decode`], refusing a picture of more than `most` pixels before its
/// data is inflated.
pub fn decode_within(bytes: &[u8], most: usize) -> Result<Decoded, PngError> {
    let format = |message: &str| PngError::Format(message.to_owned());
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(format("missing signature"));
    }
    let mut position = 8;
    let mut header: Option<(u32, u32, u8, u8)> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut alpha: Vec<u8> = Vec::new();
    let mut data = Vec::new();
    loop {
        let head = bytes
            .get(position..position + 8)
            .ok_or_else(|| format("chunk header ends early"))?;
        let length = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let kind = &head[4..8];
        let body = bytes
            .get(position + 8..position + 8 + length)
            .ok_or_else(|| format("chunk ends early"))?;
        let crc = bytes
            .get(position + 8 + length..position + 12 + length)
            .ok_or_else(|| format("chunk crc ends early"))?;
        if crc32(&bytes[position + 4..position + 8 + length])
            != u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]])
        {
            return Err(format("chunk crc mismatch"));
        }
        match kind {
            b"IHDR" => {
                if body.len() != 13 {
                    return Err(format("IHDR length"));
                }
                let width = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                let height = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                if body[12] != 0 {
                    return Err(format("interlaced; export it without interlacing"));
                }
                header = Some((width, height, body[8], body[9]));
            }
            b"PLTE" => palette = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            b"tRNS" => alpha = body.to_vec(),
            b"IDAT" => data.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        position += 12 + length;
    }
    let (width, height, depth, color) = header.ok_or_else(|| format("no IHDR"))?;
    let channels = match color {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return Err(format("color type")),
    };
    if !matches!(depth, 1 | 2 | 4 | 8 | 16) || (depth < 8 && !matches!(color, 0 | 3)) {
        return Err(format("bit depth"));
    }
    let pixels = (width as usize).saturating_mul(height as usize);
    if pixels > most {
        return Err(PngError::Format(format!(
            "{width}x{height} pixels, past the room for {most}"
        )));
    }
    let bits_per_pixel = channels * depth as usize;
    let stride = (width as usize * bits_per_pixel).div_ceil(8);
    let bpp = bits_per_pixel.div_ceil(8);
    let expected = (stride + 1) * height as usize;
    let raw = unzlib(&data, expected).map_err(PngError::Inflate)?;
    if raw.len() != expected {
        return Err(format("image data length"));
    }
    let mut rows = vec![0u8; stride * height as usize];
    for y in 0..height as usize {
        let filter = raw[y * (stride + 1)];
        let line = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        for x in 0..stride {
            let a = if x >= bpp {
                rows[y * stride + x - bpp]
            } else {
                0
            };
            let b = if y > 0 { rows[(y - 1) * stride + x] } else { 0 };
            let c = if x >= bpp && y > 0 {
                rows[(y - 1) * stride + x - bpp]
            } else {
                0
            };
            let predicted = match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                4 => {
                    let p = a as i16 + b as i16 - c as i16;
                    let (pa, pb, pc) = (
                        (p - a as i16).abs(),
                        (p - b as i16).abs(),
                        (p - c as i16).abs(),
                    );
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                _ => return Err(format("filter type")),
            };
            rows[y * stride + x] = line[x].wrapping_add(predicted);
        }
    }
    // Samples: palette entries expand to RGB or RGBA; other depths scale.
    let count = width as usize * height as usize;
    let out_channels = if color == 3 {
        if alpha.is_empty() { 3 } else { 4 }
    } else {
        channels
    };
    let mut samples = Vec::with_capacity(count * out_channels);
    let max = ((1u32 << depth) - 1) as f32;
    for y in 0..height as usize {
        let row = &rows[y * stride..(y + 1) * stride];
        for x in 0..width as usize {
            for channel in 0..channels {
                let index = x * channels + channel;
                let value: u32 = match depth {
                    8 => row[index] as u32,
                    16 => u16::from_be_bytes([row[index * 2], row[index * 2 + 1]]) as u32,
                    d => {
                        let bit = index * d as usize;
                        ((row[bit / 8] >> (8 - d as usize - bit % 8)) & ((1 << d) - 1)) as u32
                    }
                };
                if color == 3 {
                    let entry = palette
                        .get(value as usize)
                        .ok_or_else(|| format("palette index"))?;
                    samples.extend(entry.iter().map(|c| *c as f32 / 255.0));
                    if out_channels == 4 {
                        samples
                            .push(alpha.get(value as usize).copied().unwrap_or(255) as f32 / 255.0);
                    }
                } else {
                    samples.push(value as f32 / max);
                }
            }
        }
    }
    Ok(Decoded {
        width,
        height,
        channels: out_channels,
        samples,
    })
}

/// Decodes a PNG as a grid of pixels (`gray` false) or a `Grid<Float>` of
/// its luminance or gray samples (`gray` true), of at most `words` words.
pub fn decode_field(bytes: &[u8], gray: bool, words: usize) -> Result<FieldData, PngError> {
    let decoded = decode_within(bytes, words / if gray { 1 } else { 4 })?;
    let domain = Domain::Grid(decoded.width, decoded.height);
    let pixels = decoded.samples.chunks_exact(decoded.channels);
    if gray {
        let words = pixels
            .map(|p| match decoded.channels {
                1 | 2 => p[0],
                _ => p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722,
            })
            .map(f32::to_bits)
            .collect();
        FieldData::from_storage(Schema::new(vec![Scalar::Float]), domain, words)
            .map_err(PngError::Host)
    } else {
        let words = pixels
            .flat_map(|p| {
                let (r, g, b, a) = match decoded.channels {
                    1 => (p[0], p[0], p[0], 1.0),
                    2 => (p[0], p[0], p[0], p[1]),
                    3 => (p[0], p[1], p[2], 1.0),
                    _ => (p[0], p[1], p[2], p[3]),
                };
                [r.to_bits(), g.to_bits(), b.to_bits(), a.to_bits()]
            })
            .collect();
        FieldData::from_storage(Schema::pixel(), domain, words).map_err(PngError::Host)
    }
}

/// Byte i of a row as PNG's filter `kind` (0 none, 1 sub, 2 up, 3
/// average, 4 Paeth) writes it, against the row above, four bytes a pixel.
fn filtered(kind: u8, row: &[u8], above: &[u8], i: usize) -> u8 {
    let (a, b, c) = if i >= 4 {
        (row[i - 4], above[i], above[i - 4])
    } else {
        (0, above[i], 0)
    };
    let predicted = match kind {
        0 => 0,
        1 => a,
        2 => b,
        3 => ((a as u16 + b as u16) / 2) as u8,
        _ => {
            let p = a as i16 + b as i16 - c as i16;
            let (pa, pb, pc) = (
                (p - a as i16).abs(),
                (p - b as i16).abs(),
                (p - c as i16).abs(),
            );
            if pa <= pb && pa <= pc {
                a
            } else if pb <= pc {
                b
            } else {
                c
            }
        }
    };
    row[i].wrapping_sub(predicted)
}

/// Encodes RGBA pixels (`0..=1`) as a compressed 8-bit PNG: each row
/// filtered as whichever of PNG's filters leaves it the smallest bytes
/// (their sum as signed values), then deflated.
pub fn encode(width: u32, height: u32, pixels: &[[f32; 4]]) -> Vec<u8> {
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    let mut above = vec![0u8; stride];
    for row in pixels.chunks(width as usize) {
        let bytes: Vec<u8> = row
            .iter()
            .flat_map(|pixel| pixel.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect();
        let cost = |kind: u8| -> u64 {
            (0..stride)
                .map(|i| (filtered(kind, &bytes, &above, i) as i8).unsigned_abs() as u64)
                .sum()
        };
        let kind = (0..5u8).min_by_key(|&kind| cost(kind)).unwrap_or(0);
        raw.push(kind);
        raw.extend((0..stride).map(|i| filtered(kind, &bytes, &above, i)));
        above = bytes;
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8], body: &[u8]| {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &zlib(&raw));
    chunk(b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    mod vectors {
        include!("../tests/vectors.rs");
    }

    /// A 1x1 gray PNG whose data inflates to `copies` times 258 zeros: one
    /// zero, then copies of the last 258 bytes, 13 bits each.
    fn bomb(copies: usize) -> Vec<u8> {
        let mut bits: Vec<bool> = Vec::new();
        let mut put = |value: u32, count: u32, reversed: bool| {
            for k in 0..count {
                let bit = if reversed { count - 1 - k } else { k };
                bits.push(value >> bit & 1 == 1);
            }
        };
        put(1, 1, false); // the last block
        put(1, 2, false); // fixed codes
        put(0x30, 8, true); // literal 0
        for _ in 0..copies {
            put(0xC5, 8, true); // length 258
            put(0, 5, true); // distance 1
        }
        put(0, 7, true); // end of block
        let mut stream = vec![0x78, 0x01];
        stream.extend(bits.chunks(8).map(|byte| {
            byte.iter()
                .enumerate()
                .fold(0u8, |acc, (k, bit)| acc | (u8::from(*bit) << k))
        }));
        stream.extend([0, 0, 0, 0]);
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut chunk = |kind: &[u8], body: &[u8]| {
            out.extend_from_slice(&(body.len() as u32).to_be_bytes());
            let start = out.len();
            out.extend_from_slice(kind);
            out.extend_from_slice(body);
            let crc = crc32(&out[start..]);
            out.extend_from_slice(&crc.to_be_bytes());
        };
        chunk(b"IHDR", &[0, 0, 0, 1, 0, 0, 0, 1, 8, 0, 0, 0, 0]);
        chunk(b"IDAT", &stream);
        chunk(b"IEND", &[]);
        out
    }

    #[test]
    fn data_past_the_pictures_size_stops_inflating() {
        // 100 MB of zeros behind a 1x1 picture: inflating stops past the
        // two bytes the picture holds.
        let error = decode(&bomb(400_000)).err().unwrap().to_string();
        assert!(error.contains("more than the 2 byte(s)"), "{error}");
    }

    #[test]
    fn a_picture_past_the_room_is_refused_before_it_inflates() {
        let mut big = bomb(1);
        // 65536x65536 pixels, the header's crc made again.
        big[16..24].copy_from_slice(&[0, 1, 0, 0, 0, 1, 0, 0]);
        let crc = crc32(&big[12..29]);
        big[29..33].copy_from_slice(&crc.to_be_bytes());
        let error = decode_field(&big, true, 1 << 20).unwrap_err().to_string();
        assert!(error.contains("past the room"), "{error}");
    }
    use vectors::*;

    #[test]
    fn decodes_rgb_gray_palette_and_sixteen_bit_files() {
        let rgb = decode(RGB16X16).unwrap();
        assert_eq!((rgb.width, rgb.height, rgb.channels), (16, 16, 3));
        assert_eq!(rgb.samples[(3 * 16 + 5) * 3], (5 * 16) as f32 / 255.0);
        assert_eq!(rgb.samples[(3 * 16 + 5) * 3 + 1], (3 * 16) as f32 / 255.0);
        let gray = decode(GRAY_SUB).unwrap();
        assert_eq!(gray.channels, 1);
        assert_eq!(
            gray.samples,
            [10.0 / 255.0, 15.0 / 255.0, 20.0 / 255.0, 25.0 / 255.0]
        );
        let palette = decode(PALETTE2).unwrap();
        assert_eq!((palette.channels, palette.samples.len()), (3, 12));
        assert_eq!(&palette.samples[3..6], &[0.0, 1.0, 0.0]);
        assert_eq!(&palette.samples[9..12], &[1.0, 1.0, 1.0]);
        let deep = decode(RGBA16).unwrap();
        assert_eq!(deep.channels, 4);
        assert!((deep.samples[1] - 0.5).abs() < 0.001);
        assert!(decode(b"not a png").is_err());
    }

    #[test]
    fn fields_come_out_as_asked_and_encoded_files_decode() {
        let pixels = decode_field(GRAY_SUB, false, usize::MAX).unwrap();
        assert!(pixels.schema().is_pixel());
        assert_eq!(f32::from_bits(pixels.storage()[4]), 15.0 / 255.0);
        let gray = decode_field(RGB16X16, true, usize::MAX).unwrap();
        assert_eq!(gray.schema().value_words(), 1);
        let picture = vec![[1.0, 0.5, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]];
        let decoded = decode(&encode(2, 1, &picture)).unwrap();
        assert_eq!(decoded.channels, 4);
        assert_eq!(decoded.samples[7], 128.0 / 255.0);
        assert_eq!(decoded.samples[2], 0.0);
    }
}
