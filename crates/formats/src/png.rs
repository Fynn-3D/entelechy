//! PNG files as Fields: a picture decodes to a grid of pixels or, when a
//! `Grid<Float>` is wanted, to a grid of samples; a grid of pixels
//! encodes to a PNG anyone can open. Bit depths 1 to 16, every colour
//! type, palettes; interlaced files are refused with a message.

use std::fmt;

use orla_host::{Domain, FieldData, HostError, Scalar, Schema};

use crate::inflate::{InflateError, unzlib, zlib_stored};

/// Why bytes are not a PNG the Host can take.
#[derive(Debug)]
pub enum PngError {
    /// The file is malformed or uses a feature not supported.
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

/// What a decoded PNG holds: samples in `0..=1`, `channels` per pixel,
/// 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA.
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
    let (width, height, depth, colour) = header.ok_or_else(|| format("no IHDR"))?;
    let channels = match colour {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return Err(format("colour type")),
    };
    if !matches!(depth, 1 | 2 | 4 | 8 | 16) || (depth < 8 && !matches!(colour, 0 | 3)) {
        return Err(format("bit depth"));
    }
    let raw = unzlib(&data).map_err(PngError::Inflate)?;
    let bits_per_pixel = channels * depth as usize;
    let stride = (width as usize * bits_per_pixel).div_ceil(8);
    let bpp = bits_per_pixel.div_ceil(8);
    if raw.len() != (stride + 1) * height as usize {
        return Err(format("image data length"));
    }
    // Unfilter row by row.
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
    let out_channels = if colour == 3 {
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
                if colour == 3 {
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
/// its luminance or gray samples (`gray` true).
pub fn decode_field(bytes: &[u8], gray: bool) -> Result<FieldData, PngError> {
    let decoded = decode(bytes)?;
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

/// Encodes RGBA pixels (`0..=1`) as an 8-bit PNG, stored rather than
/// compressed, so no encoder tables are needed and every reader takes it.
pub fn encode(width: u32, height: u32, pixels: &[[f32; 4]]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((width as usize * 4 + 1) * height as usize);
    for row in pixels.chunks(width as usize) {
        raw.push(0);
        for pixel in row {
            raw.extend(
                pixel
                    .iter()
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
            );
        }
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
    chunk(b"IDAT", &zlib_stored(&raw));
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
        let pixels = decode_field(GRAY_SUB, false).unwrap();
        assert!(pixels.schema().is_pixel());
        assert_eq!(f32::from_bits(pixels.storage()[4]), 15.0 / 255.0);
        let gray = decode_field(RGB16X16, true).unwrap();
        assert_eq!(gray.schema().value_words(), 1);
        let picture = vec![[1.0, 0.5, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]];
        let decoded = decode(&encode(2, 1, &picture)).unwrap();
        assert_eq!(decoded.channels, 4);
        assert_eq!(decoded.samples[7], 128.0 / 255.0);
        assert_eq!(decoded.samples[2], 0.0);
    }
}
