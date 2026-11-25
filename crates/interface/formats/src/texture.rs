//! Hardware block-compressed texture containers: Khronos KTX 2.0 (`.ktx2`) and
//! DirectDraw Surface (`.dds`, legacy FourCC + DX10 extended header).
//!
//! Decodes directly into [`FieldData::from_blocks`] with zero CPU texel
//! decompression, preserving 4x4 BC1, BC4, BC5, and BC7 blocks for direct GPU
//! upload and lazy CPU/JIT decoding. Supports both single 2D `Grid` textures
//! and stacked 2D texture arrays (`Volume`).

use entelechy_host::{BlockFormat, CompressedBlocks, Domain, FieldData, Shape};

/// KTX 2.0 12-byte file identifier.
const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// DDS 4-byte file identifier.
const DDS_MAGIC: [u8; 4] = *b"DDS ";

/// Minimum header bytes required to inspect any KTX2 or DDS texture.
pub const HEADER_BYTES: usize = 148;

fn u32_le(bytes: &[u8], at: usize) -> Result<u32, String> {
    let slice = bytes
        .get(at..at + 4)
        .ok_or_else(|| "compressed texture header is truncated".to_owned())?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn u64_le(bytes: &[u8], at: usize) -> Result<u64, String> {
    let slice = bytes
        .get(at..at + 8)
        .ok_or_else(|| "compressed texture header is truncated".to_owned())?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn vk_to_block_format(vk: u32) -> Result<BlockFormat, String> {
    match vk {
        131..=134 => Ok(BlockFormat::Bc1RgbaUnorm),
        139 => Ok(BlockFormat::Bc4RUnorm),
        141 => Ok(BlockFormat::Bc5RgUnorm),
        145 | 146 => Ok(BlockFormat::Bc7RgbaUnorm),
        other => Err(format!(
            "unsupported KTX2 VkFormat {other}; use BC1, BC4, BC5, or BC7"
        )),
    }
}

fn block_format_to_vk(format: BlockFormat) -> u32 {
    match format {
        BlockFormat::Bc1RgbaUnorm => 133, // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
        BlockFormat::Bc4RUnorm => 139,    // VK_FORMAT_BC4_UNORM_BLOCK
        BlockFormat::Bc5RgUnorm => 141,   // VK_FORMAT_BC5_UNORM_BLOCK
        BlockFormat::Bc7RgbaUnorm => 145, // VK_FORMAT_BC7_UNORM_BLOCK
    }
}

fn dxgi_to_block_format(dxgi: u32) -> Result<BlockFormat, String> {
    match dxgi {
        71 | 72 => Ok(BlockFormat::Bc1RgbaUnorm),
        80 => Ok(BlockFormat::Bc4RUnorm),
        83 => Ok(BlockFormat::Bc5RgUnorm),
        98 | 99 => Ok(BlockFormat::Bc7RgbaUnorm),
        other => Err(format!(
            "unsupported DDS DXGI_FORMAT {other}; use BC1, BC4, BC5, or BC7"
        )),
    }
}

fn block_format_to_dxgi(format: BlockFormat) -> u32 {
    match format {
        BlockFormat::Bc1RgbaUnorm => 71, // DXGI_FORMAT_BC1_UNORM
        BlockFormat::Bc4RUnorm => 80,    // DXGI_FORMAT_BC4_UNORM
        BlockFormat::Bc5RgUnorm => 83,   // DXGI_FORMAT_BC5_UNORM
        BlockFormat::Bc7RgbaUnorm => 98, // DXGI_FORMAT_BC7_UNORM
    }
}

fn domain_for(width: u32, height: u32, layers: u32) -> Result<Domain, String> {
    if width == 0 || height == 0 || layers == 0 {
        return Err("compressed texture has a zero extent".to_owned());
    }
    Ok(if layers == 1 {
        Domain::Grid(width, height)
    } else {
        Domain::Volume(width, height, layers)
    })
}

/// Inspects a KTX2 header and returns `(format, domain, level0_offset,
/// level0_byte_len, level0_uncompressed_len, supercompression)`.
fn parse_ktx2_header(bytes: &[u8]) -> Result<(BlockFormat, Domain, u64, u64, u64, u32), String> {
    if bytes.len() < 104 || bytes[..12] != KTX2_MAGIC {
        return Err("not a KTX 2.0 file".to_owned());
    }
    let vk_format = u32_le(bytes, 12)?;
    let format = vk_to_block_format(vk_format)?;
    let width = u32_le(bytes, 20)?;
    let height = u32_le(bytes, 24)?.max(1);
    let depth = u32_le(bytes, 28)?.max(1);
    let layers = u32_le(bytes, 32)?.max(1);
    let faces = u32_le(bytes, 36)?.max(1);
    let supercomp = u32_le(bytes, 44)?;
    let total_layers = depth
        .checked_mul(layers)
        .and_then(|v| v.checked_mul(faces))
        .ok_or_else(|| "KTX2 layer count overflow".to_owned())?;
    let domain = domain_for(width, height, total_layers)?;
    let level0_offset = u64_le(bytes, 80)?;
    let level0_bytes = u64_le(bytes, 88)?;
    let level0_uncomp = u64_le(bytes, 96)?;
    Ok((
        format,
        domain,
        level0_offset,
        level0_bytes,
        level0_uncomp,
        supercomp,
    ))
}

/// Inspects a DDS header and returns `(format, domain, mip_count, body_offset)`.
fn parse_dds_header(bytes: &[u8]) -> Result<(BlockFormat, Domain, u32, usize), String> {
    if bytes.len() < 128 || bytes[..4] != DDS_MAGIC {
        return Err("not a DDS file".to_owned());
    }
    let header_size = u32_le(bytes, 4)?;
    if header_size != 124 {
        return Err(format!("invalid DDS header size {header_size}"));
    }
    let height = u32_le(bytes, 12)?;
    let width = u32_le(bytes, 16)?;
    let mip_count = u32_le(bytes, 28)?.max(1);
    let fourcc = &bytes[84..88];
    let (format, layers, body_offset) = match fourcc {
        b"DXT1" => (BlockFormat::Bc1RgbaUnorm, 1, 128),
        b"ATI1" | b"BC4U" => (BlockFormat::Bc4RUnorm, 1, 128),
        b"ATI2" | b"BC5U" => (BlockFormat::Bc5RgUnorm, 1, 128),
        b"DX10" => {
            if bytes.len() < 148 {
                return Err("DDS DX10 header is truncated".to_owned());
            }
            let dxgi = u32_le(bytes, 128)?;
            let array_size = u32_le(bytes, 140)?.max(1);
            (dxgi_to_block_format(dxgi)?, array_size, 148)
        }
        other => {
            return Err(format!(
                "unsupported DDS FourCC `{}`; use DXT1, ATI1, ATI2, or DX10 (BC1/BC4/BC5/BC7)",
                String::from_utf8_lossy(other)
            ));
        }
    };
    let domain = domain_for(width, height, layers)?;
    Ok((format, domain, mip_count, body_offset))
}

/// Compressed Host budget words required by a `.ktx2` or `.dds` file, read
/// from its header alone before loading its body.
pub fn words(start: &[u8], extension: &str) -> Result<usize, String> {
    let (format, domain) = match extension {
        "ktx2" => {
            let (f, d, ..) = parse_ktx2_header(start)?;
            (f, d)
        }
        "dds" => {
            let (f, d, ..) = parse_dds_header(start)?;
            (f, d)
        }
        other => return Err(format!("not a compressed texture extension `{other}`")),
    };
    let bytes = format
        .byte_len(domain)
        .ok_or_else(|| "compressed texture size overflow".to_owned())?;
    Ok(bytes.div_ceil(4))
}

/// Decodes a KTX 2.0 file directly into a block-compressed [`FieldData`].
pub fn decode_ktx2(bytes: &[u8], max_words: usize) -> Result<FieldData, String> {
    let (format, domain, offset, byte_len, uncomp_len, supercomp) = parse_ktx2_header(bytes)?;
    let expected = format
        .byte_len(domain)
        .ok_or_else(|| "KTX2 byte length overflow".to_owned())?;
    let needed_words = expected.div_ceil(4);
    if needed_words > max_words {
        return Err(entelechy_host::HostError::BudgetExceeded {
            requested: needed_words,
            available: max_words,
        }
        .to_string());
    }
    let start = usize::try_from(offset).map_err(|_| "KTX2 offset overflow".to_owned())?;
    let len = usize::try_from(byte_len).map_err(|_| "KTX2 level length overflow".to_owned())?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| "KTX2 level range overflow".to_owned())?;
    let level_slice = bytes
        .get(start..end)
        .ok_or_else(|| "KTX2 level 0 data is truncated".to_owned())?;
    let raw = match supercomp {
        0 => level_slice.to_vec(),
        3 => {
            let cap = usize::try_from(uncomp_len)
                .unwrap_or(expected)
                .max(expected);
            crate::inflate::unzlib(level_slice, cap)
                .map_err(|e| format!("KTX2 zlib supercompression: {e}"))?
        }
        other => {
            return Err(format!(
                "unsupported KTX2 supercompression scheme {other}; use uncompressed (0) or zlib (3)"
            ));
        }
    };
    if raw.len() != expected {
        return Err(format!(
            "KTX2 level 0 holds {} bytes, expected {expected} for {format} over {domain}",
            raw.len()
        ));
    }
    let blocks = CompressedBlocks::new(format, domain, raw).map_err(|e| e.to_string())?;
    FieldData::from_blocks(domain, blocks).map_err(|e| e.to_string())
}

/// Decodes a DDS file directly into a block-compressed [`FieldData`].
pub fn decode_dds(bytes: &[u8], max_words: usize) -> Result<FieldData, String> {
    let (format, domain, mip_count, body_offset) = parse_dds_header(bytes)?;
    let expected = format
        .byte_len(domain)
        .ok_or_else(|| "DDS byte length overflow".to_owned())?;
    let needed_words = expected.div_ceil(4);
    if needed_words > max_words {
        return Err(entelechy_host::HostError::BudgetExceeded {
            requested: needed_words,
            available: max_words,
        }
        .to_string());
    }
    let body = bytes
        .get(body_offset..)
        .ok_or_else(|| "DDS body is missing".to_owned())?;
    let w = domain.width();
    let h = domain.height();
    let layers = domain.depth() as usize;
    let slice_mip0 = format
        .byte_len(Domain::Grid(w, h))
        .ok_or_else(|| "DDS slice byte length overflow".to_owned())?;
    let raw = if layers == 1 || mip_count <= 1 {
        let slice = body.get(..expected).ok_or_else(|| {
            format!(
                "DDS body holds {} bytes, expected at least {expected}",
                body.len()
            )
        })?;
        slice.to_vec()
    } else {
        // Multi-layer DDS with mipmaps stores all mips of slice 0, then slice 1, ...
        let mut stride = 0usize;
        let (mut mw, mut mh) = (w, h);
        for _ in 0..mip_count {
            stride = stride
                .checked_add(
                    format
                        .byte_len(Domain::Grid(mw, mh))
                        .ok_or_else(|| "DDS mip chain overflow".to_owned())?,
                )
                .ok_or_else(|| "DDS mip chain overflow".to_owned())?;
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
        }
        let mut out = Vec::with_capacity(expected);
        for layer in 0..layers {
            let at = layer
                .checked_mul(stride)
                .ok_or_else(|| "DDS layer offset overflow".to_owned())?;
            let slice = body
                .get(at..at + slice_mip0)
                .ok_or_else(|| "DDS mipmapped array is truncated".to_owned())?;
            out.extend_from_slice(slice);
        }
        out
    };
    let blocks = CompressedBlocks::new(format, domain, raw).map_err(|e| e.to_string())?;
    FieldData::from_blocks(domain, blocks).map_err(|e| e.to_string())
}

/// Encodes [`CompressedBlocks`] over `domain` into a conformant KTX 2.0 byte
/// stream (single level, uncompressed supercompression).
pub fn encode_ktx2(domain: Domain, blocks: &CompressedBlocks) -> Result<Vec<u8>, String> {
    if domain.shape() == Shape::Rows {
        return Err("KTX2 requires a Grid or Volume".to_owned());
    }
    let layer_count = if domain.depth() > 1 {
        domain.depth()
    } else {
        0
    };
    let data = blocks.bytes();
    let header_and_index = 80 + 24;
    // Align level 0 data to 16 bytes per KTX2 §3.11.
    let offset = (header_and_index + 15) & !15;
    let mut out = Vec::with_capacity(offset + data.len());
    out.extend_from_slice(&KTX2_MAGIC);
    out.extend_from_slice(&block_format_to_vk(blocks.format()).to_le_bytes()); // vkFormat
    out.extend_from_slice(&1u32.to_le_bytes()); // typeSize = 1 for block-compressed
    out.extend_from_slice(&domain.width().to_le_bytes());
    out.extend_from_slice(&domain.height().to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth = 0 for 2D / 2D array
    out.extend_from_slice(&layer_count.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // faceCount = 1
    out.extend_from_slice(&1u32.to_le_bytes()); // levelCount = 1
    out.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme = None
    // DFD / KVD / SGD offsets & lengths (all zero).
    out.extend_from_slice(&[0u8; 32]);
    debug_assert_eq!(out.len(), 80);
    // Level 0 index entry (24 bytes).
    out.extend_from_slice(&(offset as u64).to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.resize(offset, 0);
    out.extend_from_slice(data);
    Ok(out)
}

/// Encodes [`CompressedBlocks`] over `domain` into a conformant DDS DX10 byte
/// stream (148-byte header + raw blocks).
pub fn encode_dds(domain: Domain, blocks: &CompressedBlocks) -> Result<Vec<u8>, String> {
    if domain.shape() == Shape::Rows {
        return Err("DDS requires a Grid or Volume".to_owned());
    }
    let data = blocks.bytes();
    let linear_size = blocks
        .format()
        .byte_len(Domain::Grid(domain.width(), domain.height()))
        .ok_or_else(|| "DDS linear size overflow".to_owned())? as u32;
    let mut out = Vec::with_capacity(148 + data.len());
    out.extend_from_slice(&DDS_MAGIC);
    // DDS_HEADER (124 bytes)
    out.extend_from_slice(&124u32.to_le_bytes()); // dwSize
    // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    out.extend_from_slice(&0x0008_1007u32.to_le_bytes()); // dwFlags
    out.extend_from_slice(&domain.height().to_le_bytes());
    out.extend_from_slice(&domain.width().to_le_bytes());
    out.extend_from_slice(&linear_size.to_le_bytes()); // dwPitchOrLinearSize
    out.extend_from_slice(&0u32.to_le_bytes()); // dwDepth
    out.extend_from_slice(&1u32.to_le_bytes()); // dwMipMapCount
    out.extend_from_slice(&[0u8; 44]); // dwReserved1[11]
    // DDS_PIXELFORMAT (32 bytes)
    out.extend_from_slice(&32u32.to_le_bytes()); // dwSize
    out.extend_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
    out.extend_from_slice(b"DX10"); // dwFourCC
    out.extend_from_slice(&[0u8; 20]); // bitcounts & masks
    // dwCaps..dwReserved2 (20 bytes)
    out.extend_from_slice(&0x1000u32.to_le_bytes()); // DDSCAPS_TEXTURE
    out.extend_from_slice(&[0u8; 16]);
    debug_assert_eq!(out.len(), 128);
    // DDS_HEADER_DXT10 (20 bytes)
    out.extend_from_slice(&block_format_to_dxgi(blocks.format()).to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    out.extend_from_slice(&0u32.to_le_bytes()); // miscFlag
    out.extend_from_slice(&domain.depth().to_le_bytes()); // arraySize
    out.extend_from_slice(&0u32.to_le_bytes()); // miscFlags2
    debug_assert_eq!(out.len(), 148);
    out.extend_from_slice(data);
    Ok(out)
}

/// Compresses an arbitrary `Grid`/`Volume` [`FieldData`] into
/// [`CompressedBlocks`] (reusing existing compressed blocks verbatim when
/// already compressed in `format`).
pub fn compress_field(data: &FieldData, format: BlockFormat) -> Result<CompressedBlocks, String> {
    if let Some(existing) = data.blocks()
        && existing.format() == format
    {
        return Ok(existing.clone());
    }
    let domain = data.domain();
    if domain.shape() == Shape::Rows {
        return Err("block compression requires a Grid or Volume".to_owned());
    }
    let target_ch = format.channels();
    let src_ch = data.schema().value_words();
    let count = domain.count();
    let values = data.values();
    let mut pixels = vec![0u8; count * target_ch];
    for i in 0..count {
        let base = i * src_ch;
        for c in 0..target_ch {
            let v = if c < src_ch {
                f32::from_bits(values[base + c])
            } else if c == 3 {
                1.0
            } else {
                0.0
            };
            pixels[i * target_ch + c] = crate::byte(v);
        }
    }
    CompressedBlocks::encode_unorm8(format, domain, &pixels).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ktx2_and_dds_round_trip_all_formats_and_arrays() {
        for format in BlockFormat::ALL {
            for domain in [Domain::Grid(16, 12), Domain::Volume(8, 8, 3)] {
                let ch = format.channels();
                let count = domain.count();
                let pixels: Vec<u8> = (0..count * ch).map(|i| ((i * 37) & 0xFF) as u8).collect();
                let blocks = CompressedBlocks::encode_unorm8(format, domain, &pixels).unwrap();

                let ktx = encode_ktx2(domain, &blocks).unwrap();
                assert_eq!(
                    words(&ktx[..104], "ktx2").unwrap(),
                    blocks.bytes().len().div_ceil(4)
                );
                let from_ktx = decode_ktx2(&ktx, usize::MAX).unwrap();
                assert_eq!(from_ktx.domain(), domain);
                assert_eq!(from_ktx.blocks(), Some(&blocks));

                let dds = encode_dds(domain, &blocks).unwrap();
                assert_eq!(
                    words(&dds[..148], "dds").unwrap(),
                    blocks.bytes().len().div_ceil(4)
                );
                let from_dds = decode_dds(&dds, usize::MAX).unwrap();
                assert_eq!(from_dds.domain(), domain);
                assert_eq!(from_dds.blocks(), Some(&blocks));
            }
        }
    }

    #[test]
    fn ktx2_and_dds_refuse_over_budget_from_header_alone() {
        let domain = Domain::Grid(64, 64);
        let blocks = CompressedBlocks::encode_unorm8(
            BlockFormat::Bc7RgbaUnorm,
            domain,
            &vec![128; 64 * 64 * 4],
        )
        .unwrap();
        let ktx = encode_ktx2(domain, &blocks).unwrap();
        let dds = encode_dds(domain, &blocks).unwrap();
        assert!(decode_ktx2(&ktx, 16).unwrap_err().contains("budget"));
        assert!(decode_dds(&dds, 16).unwrap_err().contains("budget"));
    }
}
