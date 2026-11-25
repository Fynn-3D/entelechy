//! Compresses image files (`.png`, `.ppm`, `.pgm`, `.field`) into hardware
//! block-compressed textures (`.ktx2` or `.dds` with BC1, BC4, BC5, or BC7).

use std::path::{Path, PathBuf};

use entelechy_host::BlockFormat;

/// Summary of an imported hardware-compressed texture.
pub struct Imported {
    /// Path written.
    pub file: PathBuf,
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
    /// Array layer count (`1` for a 2D texture).
    pub layers: u32,
    /// Hardware block format written.
    pub format: BlockFormat,
    /// Compressed file size in bytes.
    pub bytes: usize,
}

/// Reads `source` (`.png`, `.ppm`, `.pgm`, `.field`, `.ktx2`, `.dds`) and writes
/// `<stem>.<container>` (`ktx2` or `dds`) compressed as `format` into `out`.
pub fn import(
    source: &Path,
    out: &Path,
    format: Option<BlockFormat>,
    container: &str,
) -> Result<Imported, String> {
    let ext = container.to_ascii_lowercase();
    if ext != "ktx2" && ext != "dds" {
        return Err(format!(
            "--container takes `ktx2` or `dds`, found `{container}`"
        ));
    }
    let (data, _) = entelechy_formats::read(source, None)?;
    let domain = data.domain();
    let chosen = match format {
        Some(fmt) => fmt,
        None => BlockFormat::for_channels(data.schema().value_words()).ok_or_else(|| {
            format!(
                "cannot pick a block format for {} channel(s); pass --format bc7|bc5|bc4|bc1",
                data.schema().value_words()
            )
        })?,
    };
    let blocks = entelechy_formats::texture::compress_field(&data, chosen)?;
    let encoded = if ext == "ktx2" {
        entelechy_formats::texture::encode_ktx2(domain, &blocks)?
    } else {
        entelechy_formats::texture::encode_dds(domain, &blocks)?
    };
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("texture");
    let file = out.join(format!("{stem}.{ext}"));
    std::fs::write(&file, &encoded).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Imported {
        file,
        width: domain.width(),
        height: domain.height(),
        layers: domain.depth(),
        format: chosen,
        bytes: encoded.len(),
    })
}
