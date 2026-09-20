//! A Field as a file: how a Field leaves the Host and comes back. The
//! format is the Host's own view of a Field and nothing more: a text
//! header naming the shape, the extent, and the layout, then the words.
//!
//! ```text
//! ORLAFIELD 1
//! Grid 64 16 1 f,f,f,f
//! <width * height * depth * words-per-element little-endian 32-bit words>
//! ```

use std::fmt;

use crate::{Domain, FieldData, HostError, Schema, Shape};

/// Why bytes are not a Field file.
#[derive(Debug)]
pub enum FileError {
    /// The header is missing or malformed.
    Header(String),
    /// The words after the header are not the count the header promises.
    Length {
        /// Words the header promises.
        expected: usize,
        /// Words the file holds.
        found: usize,
    },
    /// The Host refused the Field.
    Host(HostError),
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(message) => write!(f, "not a Field file: {message}"),
            Self::Length { expected, found } => {
                write!(
                    f,
                    "the header promises {expected} words, the file holds {found}"
                )
            }
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FileError {}

const MAGIC: &str = "ORLAFIELD 1";

/// Encodes a Field with its layout and extent.
pub fn encode(schema: &Schema, domain: Domain, words: &[u32]) -> Vec<u8> {
    let mut bytes = format!(
        "{MAGIC}\n{} {} {} {} {schema}\n",
        domain.shape(),
        domain.width(),
        domain.height(),
        domain.depth()
    )
    .into_bytes();
    bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
    bytes
}

/// Encodes Field data.
pub fn encode_data(data: &FieldData) -> Vec<u8> {
    encode(data.schema(), data.domain(), data.words())
}

/// Decodes a Field file.
pub fn decode(bytes: &[u8]) -> Result<FieldData, FileError> {
    let header_end = |from: usize| -> Result<usize, FileError> {
        bytes[from..]
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| from + p)
            .ok_or_else(|| FileError::Header("the header has no end".into()))
    };
    let first = header_end(0)?;
    if &bytes[..first] != MAGIC.as_bytes() {
        return Err(FileError::Header(format!("expected `{MAGIC}` first")));
    }
    let second = header_end(first + 1)?;
    let line = String::from_utf8_lossy(&bytes[first + 1..second]);
    let mut parts = line.split_whitespace();
    let mut next = |what: &str| {
        parts
            .next()
            .ok_or_else(|| FileError::Header(format!("the header lacks {what}")))
    };
    let shape = next("a shape")?;
    let number = |text: &str| -> Result<u32, FileError> {
        text.parse::<u32>()
            .map_err(|_| FileError::Header(format!("`{text}` is not a whole number")))
    };
    let width = number(next("a width")?)?;
    let height = number(next("a height")?)?;
    let depth = number(next("a depth")?)?;
    let schema = next("a layout")?
        .parse::<Schema>()
        .map_err(FileError::Header)?;
    let domain = if shape == Shape::Rows.to_string() {
        Domain::Rows(width)
    } else if shape == Shape::Grid.to_string() {
        Domain::Grid(width, height)
    } else if shape == Shape::Volume.to_string() {
        Domain::Volume(width, height, depth)
    } else {
        return Err(FileError::Header(format!("`{shape}` is not a shape")));
    };
    let body = &bytes[second + 1..];
    let expected = domain.count() * schema.words();
    if body.len() != expected * 4 {
        return Err(FileError::Length {
            expected,
            found: body.len() / 4,
        });
    }
    let words = body
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    FieldData::from_words(schema, domain, words).map_err(FileError::Host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scalar;

    #[test]
    fn a_field_round_trips_through_its_file() {
        let schema = Schema::new(vec![Scalar::Float, Scalar::Int]);
        let words = vec![
            1.5f32.to_bits(),
            7,
            0.0f32.to_bits(),
            u32::MAX,
            2.0f32.to_bits(),
            3,
        ];
        let bytes = encode(&schema, Domain::Volume(1, 3, 1), &words);
        assert!(bytes.starts_with(b"ORLAFIELD 1\nVolume 1 3 1 f,i\n"));
        let data = decode(&bytes).unwrap();
        assert_eq!(data.domain(), Domain::Volume(1, 3, 1));
        assert_eq!(data.schema(), &schema);
        assert_eq!(data.words(), &words[..]);
        assert!(matches!(decode(b"PNG"), Err(FileError::Header(_))));
        assert!(matches!(
            decode(&bytes[..bytes.len() - 4]),
            Err(FileError::Length {
                expected: 6,
                found: 5
            })
        ));
    }
}
