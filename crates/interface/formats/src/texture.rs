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

/// Inspects a KTX2 header and returns `(format, domain, level_count, supercompression)`.
fn parse_ktx2_header(bytes: &[u8]) -> Result<(BlockFormat, Domain, u32, u32), String> {
    if bytes.len() < 80 || bytes[..12] != KTX2_MAGIC {
        return Err("not a KTX 2.0 file".to_owned());
    }
    let vk_format = u32_le(bytes, 12)?;
    let format = vk_to_block_format(vk_format)?;
    let width = u32_le(bytes, 20)?;
    let height = u32_le(bytes, 24)?.max(1);
    let depth = u32_le(bytes, 28)?.max(1);
    let layers = u32_le(bytes, 32)?.max(1);
    let faces = u32_le(bytes, 36)?.max(1);
    let level_count = u32_le(bytes, 40)?.max(1);
    let supercomp = u32_le(bytes, 44)?;
    let total_layers = depth
        .checked_mul(layers)
        .and_then(|v| v.checked_mul(faces))
        .ok_or_else(|| "KTX2 layer count overflow".to_owned())?;
    let domain = domain_for(width, height, total_layers)?;
    let max_mips = entelechy_host::block::max_mip_count(domain);
    if level_count > max_mips {
        return Err(format!(
            "KTX2 declares {level_count} mip levels, exceeding maximum {max_mips} for {domain}"
        ));
    }
    Ok((format, domain, level_count, supercomp))
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
    let raw_mips = u32_le(bytes, 28)?.max(1);
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
    let mip_count = raw_mips.min(entelechy_host::block::max_mip_count(domain));
    Ok((format, domain, mip_count, body_offset))
}

/// Compressed Host budget words required by a `.ktx2` or `.dds` file (across
/// all mip levels), read from its header alone before loading its body.
pub fn words(start: &[u8], extension: &str) -> Result<usize, String> {
    let (format, domain, mip_count) = match extension {
        "ktx2" => {
            let (f, d, m, _) = parse_ktx2_header(start)?;
            (f, d, m)
        }
        "dds" => {
            let (f, d, m, _) = parse_dds_header(start)?;
            (f, d, m)
        }
        other => return Err(format!("not a compressed texture extension `{other}`")),
    };
    let bytes = format
        .mip_chain_byte_len(domain, mip_count)
        .ok_or_else(|| "compressed texture size overflow".to_owned())?;
    Ok(bytes.div_ceil(4))
}

/// Decodes a KTX 2.0 file (all mip levels `0..levelCount`) directly into a
/// block-compressed [`FieldData`].
pub fn decode_ktx2(bytes: &[u8], max_words: usize) -> Result<FieldData, String> {
    let (format, domain, level_count, supercomp) = parse_ktx2_header(bytes)?;
    let total_expected = format
        .mip_chain_byte_len(domain, level_count)
        .ok_or_else(|| "KTX2 byte length overflow".to_owned())?;
    let needed_words = total_expected.div_ceil(4);
    if needed_words > max_words {
        return Err(entelechy_host::HostError::BudgetExceeded {
            requested: needed_words,
            available: max_words,
        }
        .to_string());
    }
    let index_end = 80usize
        .checked_add(level_count as usize * 24)
        .ok_or_else(|| "KTX2 level index overflow".to_owned())?;
    if bytes.len() < index_end {
        return Err("KTX2 level index table is truncated".to_owned());
    }
    let mut all_bytes = Vec::with_capacity(total_expected);
    for mip in 0..level_count {
        let entry = 80 + mip as usize * 24;
        let offset = u64_le(bytes, entry)?;
        let byte_len = u64_le(bytes, entry + 8)?;
        let uncomp_len = u64_le(bytes, entry + 16)?;
        let level_dom = entelechy_host::block::mip_domain(domain, mip);
        let expected = format
            .byte_len(level_dom)
            .ok_or_else(|| "KTX2 mip byte length overflow".to_owned())?;
        let start = usize::try_from(offset).map_err(|_| "KTX2 offset overflow".to_owned())?;
        let len = usize::try_from(byte_len).map_err(|_| "KTX2 level length overflow".to_owned())?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "KTX2 level range overflow".to_owned())?;
        let level_slice = bytes
            .get(start..end)
            .ok_or_else(|| format!("KTX2 mip level {mip} data is truncated"))?;
        match supercomp {
            0 => {
                if level_slice.len() != expected {
                    return Err(format!(
                        "KTX2 mip level {mip} holds {} bytes, expected {expected} for {format} over {level_dom}",
                        level_slice.len()
                    ));
                }
                all_bytes.extend_from_slice(level_slice);
            }
            3 => {
                let cap = usize::try_from(uncomp_len)
                    .unwrap_or(expected)
                    .max(expected);
                let raw = crate::inflate::unzlib(level_slice, cap)
                    .map_err(|e| format!("KTX2 mip level {mip} zlib supercompression: {e}"))?;
                if raw.len() != expected {
                    return Err(format!(
                        "KTX2 mip level {mip} holds {} bytes, expected {expected} for {format} over {level_dom}",
                        raw.len()
                    ));
                }
                all_bytes.extend_from_slice(&raw);
            }
            other => {
                return Err(format!(
                    "unsupported KTX2 supercompression scheme {other}; use uncompressed (0) or zlib (3)"
                ));
            }
        }
    }
    let blocks = CompressedBlocks::new_with_mips(format, domain, level_count, all_bytes)
        .map_err(|e| e.to_string())?;
    FieldData::from_blocks(domain, blocks).map_err(|e| e.to_string())
}

/// Decodes a DDS file (all mip levels `0..dwMipMapCount`) directly into a
/// block-compressed [`FieldData`].
pub fn decode_dds(bytes: &[u8], max_words: usize) -> Result<FieldData, String> {
    let (format, domain, mip_count, body_offset) = parse_dds_header(bytes)?;
    let total_expected = format
        .mip_chain_byte_len(domain, mip_count)
        .ok_or_else(|| "DDS byte length overflow".to_owned())?;
    let needed_words = total_expected.div_ceil(4);
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
    if body.len() < total_expected {
        return Err(format!(
            "DDS body holds {} bytes, expected at least {total_expected}",
            body.len()
        ));
    }
    let layers = domain.depth() as usize;
    let raw = if layers == 1 || mip_count <= 1 {
        body[..total_expected].to_vec()
    } else {
        // Multi-layer DDS stores [layer0: mip0..M][layer1: mip0..M]...;
        // CompressedBlocks stores [mip0: layer0..L][mip1: layer0..L]...
        let w = domain.width();
        let h = domain.height();
        let mut mip_sizes = Vec::with_capacity(mip_count as usize);
        let mut layer_stride = 0usize;
        for m in 0..mip_count {
            let sz = format
                .byte_len(entelechy_host::block::mip_domain(Domain::Grid(w, h), m))
                .ok_or_else(|| "DDS mip chain overflow".to_owned())?;
            mip_sizes.push(sz);
            layer_stride = layer_stride
                .checked_add(sz)
                .ok_or_else(|| "DDS mip chain overflow".to_owned())?;
        }
        let mut out = Vec::with_capacity(total_expected);
        let mut mip_offset_in_layer = 0usize;
        for sz in mip_sizes {
            for layer in 0..layers {
                let at = layer * layer_stride + mip_offset_in_layer;
                out.extend_from_slice(&body[at..at + sz]);
            }
            mip_offset_in_layer += sz;
        }
        out
    };
    let blocks = CompressedBlocks::new_with_mips(format, domain, mip_count, raw)
        .map_err(|e| e.to_string())?;
    FieldData::from_blocks(domain, blocks).map_err(|e| e.to_string())
}

/// Encodes [`CompressedBlocks`] (including all mip levels) over `domain` into a
/// conformant KTX 2.0 byte stream (uncompressed supercompression, 16-byte level
/// alignment per KTX2 §3.11).
pub fn encode_ktx2(domain: Domain, blocks: &CompressedBlocks) -> Result<Vec<u8>, String> {
    if domain.shape() == Shape::Rows {
        return Err("KTX2 requires a Grid or Volume".to_owned());
    }
    let layer_count = if domain.depth() > 1 {
        domain.depth()
    } else {
        0
    };
    let mip_count = blocks.mip_count();
    let header_and_index = 80 + mip_count as usize * 24;
    // Per KTX2 §3.11, store mip data smallest-first with 16-byte alignment,
    // while keeping the level index ordered mip 0 (largest) .. mip N-1.
    let mut offsets = vec![0usize; mip_count as usize];
    let mut lengths = vec![0usize; mip_count as usize];
    let mut cursor = (header_and_index + 15) & !15;
    for mip in (0..mip_count).rev() {
        cursor = (cursor + 15) & !15;
        let len = blocks.mip_bytes(domain, mip).len();
        offsets[mip as usize] = cursor;
        lengths[mip as usize] = len;
        cursor += len;
    }
    let mut out = Vec::with_capacity(cursor);
    out.extend_from_slice(&KTX2_MAGIC);
    out.extend_from_slice(&block_format_to_vk(blocks.format()).to_le_bytes()); // vkFormat
    out.extend_from_slice(&1u32.to_le_bytes()); // typeSize = 1 for block-compressed
    out.extend_from_slice(&domain.width().to_le_bytes());
    out.extend_from_slice(&domain.height().to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth = 0 for 2D / 2D array
    out.extend_from_slice(&layer_count.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // faceCount = 1
    out.extend_from_slice(&mip_count.to_le_bytes()); // levelCount
    out.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme = None
    // DFD / KVD / SGD offsets & lengths (all zero).
    out.extend_from_slice(&[0u8; 32]);
    debug_assert_eq!(out.len(), 80);
    for mip in 0..mip_count as usize {
        out.extend_from_slice(&(offsets[mip] as u64).to_le_bytes());
        out.extend_from_slice(&(lengths[mip] as u64).to_le_bytes());
        out.extend_from_slice(&(lengths[mip] as u64).to_le_bytes());
    }
    for mip in (0..mip_count).rev() {
        out.resize(offsets[mip as usize], 0);
        out.extend_from_slice(blocks.mip_bytes(domain, mip));
    }
    Ok(out)
}

/// Encodes [`CompressedBlocks`] (including all mip levels) over `domain` into a
/// conformant DDS DX10 byte stream (148-byte header + interleaved layer mips).
pub fn encode_dds(domain: Domain, blocks: &CompressedBlocks) -> Result<Vec<u8>, String> {
    if domain.shape() == Shape::Rows {
        return Err("DDS requires a Grid or Volume".to_owned());
    }
    let mip_count = blocks.mip_count();
    let data = blocks.bytes();
    let linear_size = blocks
        .format()
        .byte_len(Domain::Grid(domain.width(), domain.height()))
        .ok_or_else(|| "DDS linear size overflow".to_owned())? as u32;
    let mut out = Vec::with_capacity(148 + data.len());
    out.extend_from_slice(&DDS_MAGIC);
    // DDS_HEADER (124 bytes)
    out.extend_from_slice(&124u32.to_le_bytes()); // dwSize
    // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE [| DDSD_MIPMAPCOUNT]
    let flags = 0x0008_1007u32 | if mip_count > 1 { 0x0002_0000 } else { 0 };
    out.extend_from_slice(&flags.to_le_bytes()); // dwFlags
    out.extend_from_slice(&domain.height().to_le_bytes());
    out.extend_from_slice(&domain.width().to_le_bytes());
    out.extend_from_slice(&linear_size.to_le_bytes()); // dwPitchOrLinearSize
    out.extend_from_slice(&0u32.to_le_bytes()); // dwDepth
    out.extend_from_slice(&mip_count.to_le_bytes()); // dwMipMapCount
    out.extend_from_slice(&[0u8; 44]); // dwReserved1[11]
    // DDS_PIXELFORMAT (32 bytes)
    out.extend_from_slice(&32u32.to_le_bytes()); // dwSize
    out.extend_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
    out.extend_from_slice(b"DX10"); // dwFourCC
    out.extend_from_slice(&[0u8; 20]); // bitcounts & masks
    // dwCaps..dwReserved2 (20 bytes)
    let caps = 0x1000u32 | if mip_count > 1 { 0x0040_0008 } else { 0 };
    out.extend_from_slice(&caps.to_le_bytes()); // DDSCAPS_TEXTURE [| COMPLEX | MIPMAP]
    out.extend_from_slice(&[0u8; 16]);
    debug_assert_eq!(out.len(), 128);
    // DDS_HEADER_DXT10 (20 bytes)
    out.extend_from_slice(&block_format_to_dxgi(blocks.format()).to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    out.extend_from_slice(&0u32.to_le_bytes()); // miscFlag
    out.extend_from_slice(&domain.depth().to_le_bytes()); // arraySize
    out.extend_from_slice(&0u32.to_le_bytes()); // miscFlags2
    debug_assert_eq!(out.len(), 148);
    let layers = domain.depth() as usize;
    if layers == 1 || mip_count <= 1 {
        out.extend_from_slice(data);
    } else {
        // Interleave [mip][layer] into DDS [layer][mip].
        let w = domain.width();
        let h = domain.height();
        for layer in 0..layers {
            for mip in 0..mip_count {
                let mip_slice = blocks.mip_bytes(domain, mip);
                let layer_bytes = blocks
                    .format()
                    .byte_len(entelechy_host::block::mip_domain(Domain::Grid(w, h), mip))
                    .ok_or_else(|| "DDS mip size overflow".to_owned())?;
                out.extend_from_slice(&mip_slice[layer * layer_bytes..(layer + 1) * layer_bytes]);
            }
        }
    }
    Ok(out)
}

/// Compresses an arbitrary `Grid`/`Volume` [`FieldData`] into
/// [`CompressedBlocks`] (single mip level; reuses existing blocks verbatim when
/// already compressed in `format`).
pub fn compress_field(data: &FieldData, format: BlockFormat) -> Result<CompressedBlocks, String> {
    if let Some(existing) = data.blocks()
        && existing.format() == format
    {
        return Ok(existing.clone());
    }
    compress_field_mips(data, format, 1, false)
}

/// Compresses an arbitrary `Grid`/`Volume` [`FieldData`] into
/// [`CompressedBlocks`] with `mip_count` mip levels (`1..=max_mip_count`),
/// applying normal-map renormalization for BC5 when `normal_map` is true.
pub fn compress_field_mips(
    data: &FieldData,
    format: BlockFormat,
    mip_count: u32,
    normal_map: bool,
) -> Result<CompressedBlocks, String> {
    if let Some(existing) = data.blocks()
        && existing.format() == format
        && existing.mip_count() == mip_count
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
    CompressedBlocks::encode_unorm8_mips(format, domain, &pixels, mip_count, normal_map)
        .map_err(|e| e.to_string())
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

                // Full mipmap chain round-trip (single texture + multi-layer array).
                let full_mips = entelechy_host::block::max_mip_count(domain);
                let mip_blocks = CompressedBlocks::encode_unorm8_mips(
                    format,
                    domain,
                    &pixels,
                    full_mips,
                    format == BlockFormat::Bc5RgUnorm,
                )
                .unwrap();
                assert_eq!(mip_blocks.mip_count(), full_mips);

                let ktx_m = encode_ktx2(domain, &mip_blocks).unwrap();
                assert_eq!(
                    words(&ktx_m[..80], "ktx2").unwrap(),
                    mip_blocks.bytes().len().div_ceil(4)
                );
                let from_ktx_m = decode_ktx2(&ktx_m, usize::MAX).unwrap();
                assert_eq!(from_ktx_m.blocks(), Some(&mip_blocks));

                let dds_m = encode_dds(domain, &mip_blocks).unwrap();
                assert_eq!(
                    words(&dds_m[..148], "dds").unwrap(),
                    mip_blocks.bytes().len().div_ceil(4)
                );
                let from_dds_m = decode_dds(&dds_m, usize::MAX).unwrap();
                assert_eq!(from_dds_m.blocks(), Some(&mip_blocks));
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
