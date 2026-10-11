//! A Field as a file: a text header naming shape, extent and layout,
//! then the elements as kept, little-endian: a word per scalar, fewer
//! bytes where the layout keeps a scalar narrow (`f:unorm8`).
//!
//! ```text
//! ENTELECHYFIELD 1
//! Grid 64 16 1 f,f,f,f
//! <width * height * depth * bytes-per-element bytes>
//! ```

use std::fmt;

use entelechy_host::{Domain, FieldData, HostError, Schema, Shape};

/// Why bytes are not a Field file.
#[derive(Debug)]
pub enum FileError {
    /// The header is missing or malformed.
    Header(String),
    /// The body length differs from the header's count.
    Length {
        /// Bytes the header promises.
        expected: usize,
        /// Bytes the file holds.
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
                    "the header promises {expected} bytes, the file holds {found}"
                )
            }
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FileError {}

const MAGIC: &str = "ENTELECHYFIELD 1";

/// Encodes a Field with its layout and extent, from its storage.
pub fn encode(schema: &Schema, domain: Domain, storage: &[u32]) -> Vec<u8> {
    let mut bytes = format!(
        "{MAGIC}\n{} {} {} {} {schema}\n",
        domain.shape(),
        domain.width(),
        domain.height(),
        domain.depth()
    )
    .into_bytes();
    let length = domain.count() * schema.element_bytes();
    bytes.extend(
        storage
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .take(length),
    );
    bytes
}

/// Encodes Field data.
pub fn encode_data(data: &FieldData) -> Vec<u8> {
    encode(data.schema(), data.domain(), data.storage())
}

/// The words a Field file's elements take, from its first bytes: what a
/// reader checks against its room before it reads the rest.
pub fn words(start: &[u8]) -> Result<usize, FileError> {
    let (schema, domain, _) = header(start)?;
    Ok(domain
        .count()
        .saturating_mul(schema.element_bytes())
        .div_ceil(4))
}

/// The bytes first read for a Field file's header; a header longer than
/// these (a wide layout's) is read further ([`header_whole`]).
pub const HEADER_BYTES: usize = 256;

/// Whether `start`, a Field file's first bytes, holds its header whole.
pub fn header_whole(start: &[u8]) -> bool {
    start.iter().filter(|b| **b == b'\n').take(2).count() == 2
}

/// Decodes a Field file.
pub fn decode(bytes: &[u8]) -> Result<FieldData, FileError> {
    let (schema, domain, start) = header(bytes)?;
    let body = &bytes[start..];
    let expected = domain.count().saturating_mul(schema.element_bytes());
    if body.len() != expected {
        return Err(FileError::Length {
            expected,
            found: body.len(),
        });
    }
    let storage = body
        .chunks(4)
        .map(|w| {
            let mut word = [0u8; 4];
            word[..w.len()].copy_from_slice(w);
            u32::from_le_bytes(word)
        })
        .collect();
    FieldData::from_storage(schema, domain, storage).map_err(FileError::Host)
}

/// A Field file's layout and extent, and where its elements start.
fn header(bytes: &[u8]) -> Result<(Schema, Domain, usize), FileError> {
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
    Ok((schema, domain, second + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scalar;

    #[test]
    fn the_words_a_field_takes_are_read_from_its_header_alone() {
        let schema = Schema::new(vec![Scalar::Float; 2])
            .with_encoding(entelechy_host::Encoding::Unorm8)
            .unwrap();
        let bytes = encode(&schema, Domain::Rows(3), &[0, 0]);
        // Six bytes of elements are two words, known from the header.
        assert_eq!(words(&bytes[..bytes.len() - 6]).unwrap(), 2);
        let big = format!("{MAGIC}\nRows 1000000000 1 1 f\n");
        assert_eq!(words(big.as_bytes()).unwrap(), 1_000_000_000);
        let refused = crate::fits(big.as_bytes(), std::path::Path::new("a.field"), 1 << 20);
        assert!(refused.unwrap_err().contains("requested 1000000000 words"));
    }

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
        assert!(bytes.starts_with(b"ENTELECHYFIELD 1\nVolume 1 3 1 f,i\n"));
        let data = decode(&bytes).unwrap();
        assert_eq!(data.domain(), Domain::Volume(1, 3, 1));
        assert_eq!(data.schema(), &schema);
        assert_eq!(data.storage(), &words[..]);
        assert!(matches!(decode(b"PNG"), Err(FileError::Header(_))));
        assert!(matches!(
            decode(&bytes[..bytes.len() - 4]),
            Err(FileError::Length {
                expected: 24,
                found: 20
            })
        ));
    }

    #[test]
    fn a_narrow_field_keeps_its_bytes_in_its_file() {
        // Three elements of two unorm8 floats: six bytes in two words.
        let schema = Schema::new(vec![Scalar::Float; 2])
            .with_encoding(entelechy_host::Encoding::Unorm8)
            .unwrap();
        let values: Vec<u32> = [0.0f32, 1.0, 0.5, 0.25, 1.0, 0.0]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let data = FieldData::from_values(schema.clone(), Domain::Rows(3), &values).unwrap();
        assert_eq!(data.storage().len(), 2);
        let bytes = encode_data(&data);
        let header = b"ENTELECHYFIELD 1\nRows 3 1 1 f:unorm8,f:unorm8\n";
        assert!(bytes.starts_with(header));
        assert_eq!(&bytes[header.len()..], &[0, 255, 128, 64, 255, 0]);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, data);
        assert_eq!(
            back.element(1).unwrap()[1],
            entelechy_host::Encoding::Unorm8.decode(Scalar::Float, 64)
        );
    }
}
