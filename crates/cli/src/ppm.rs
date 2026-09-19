//! Binary PPM (P6) import and export for Host images.

use std::fmt;
use std::path::Path;

use orla_runtime::{HostError, Image};

/// Why a PPM file could not be read.
#[derive(Debug)]
pub enum PpmError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The file is not a binary PPM with 8-bit samples.
    Format(String),
    /// The Host rejected the image.
    Host(HostError),
}

impl fmt::Display for PpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Format(message) => write!(f, "not a supported PPM: {message}"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PpmError {}

impl From<std::io::Error> for PpmError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Reads a P6 file into an [`Image`].
pub fn read(path: &Path) -> Result<Image, PpmError> {
    let bytes = std::fs::read(path)?;
    decode(&bytes)
}

/// Decodes P6 bytes.
pub fn decode(bytes: &[u8]) -> Result<Image, PpmError> {
    let mut cursor = 0;
    let mut fields = Vec::new();
    while fields.len() < 4 {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'#' {
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            continue;
        }
        let start = cursor;
        while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if start == cursor {
            return Err(PpmError::Format("header ends early".into()));
        }
        fields.push(String::from_utf8_lossy(&bytes[start..cursor]).into_owned());
    }
    cursor += 1;
    if fields[0] != "P6" {
        return Err(PpmError::Format(format!("magic `{}` is not P6", fields[0])));
    }
    let number = |text: &str| {
        text.parse::<u32>()
            .map_err(|_| PpmError::Format(format!("`{text}` is not a number")))
    };
    let (width, height, maxval) = (
        number(&fields[1])?,
        number(&fields[2])?,
        number(&fields[3])?,
    );
    if maxval != 255 {
        return Err(PpmError::Format(format!("maxval {maxval} is not 255")));
    }
    let count = width as usize * height as usize;
    let data = bytes
        .get(cursor..cursor + count * 3)
        .ok_or_else(|| PpmError::Format("pixel data is truncated".into()))?;
    let pixels = data
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
                1.0,
            ]
        })
        .collect();
    Image::from_pixels(width, height, pixels).map_err(PpmError::Host)
}

/// Encodes an image as P6 bytes, clamping components to `0..=1`.
pub fn encode(image: &Image) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", image.width(), image.height()).into_bytes();
    for pixel in image.pixels() {
        for component in &pixel[..3] {
            bytes.push((component.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    bytes
}

/// Writes an image as a P6 file.
pub fn write(path: &Path, image: &Image) -> Result<(), PpmError> {
    Ok(std::fs::write(path, encode(image))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_with_comments() {
        let image =
            Image::from_pixels(2, 1, vec![[1.0, 0.5, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]).unwrap();
        let mut bytes = b"P6\n# comment\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&encode(&image)[11..]);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.pixels()[0][1], 128.0 / 255.0);
        assert_eq!(decoded.pixels()[1], [0.0, 0.0, 1.0, 1.0]);
        assert!(decode(b"P3 1 1 255 ").is_err());
        assert!(decode(b"P6 2 2 255 abc").is_err());
    }
}
