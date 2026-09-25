//! The `.field` file is a contract other programs keep: Blender's and
//! Godot's bridges, the importer and the exporter, `run --save` and a
//! scene's inputs all write and read it. Its bytes are pinned here, every
//! scalar and a spread of narrow encodings, both ways; and every file a
//! past Orla wrote into this repository still reads. A change that breaks
//! either is a new format, and says so with a new `ORLAFIELD` version.

use std::path::Path;

use orla_formats::file::{decode, encode_data};
use orla_host::{Domain, Encoding, FieldData, Schema};

/// A Field from its layout as a file spells it and its values as words.
fn field(layout: &str, domain: Domain, values: &[u32]) -> FieldData {
    let schema: Schema = layout.parse().unwrap();
    FieldData::from_values(schema, domain, values).unwrap()
}

/// The file's bytes: its header, then the element bytes given.
fn file(header: &str, elements: &[u8]) -> Vec<u8> {
    let mut bytes = format!("ORLAFIELD 1\n{header}\n").into_bytes();
    bytes.extend_from_slice(elements);
    bytes
}

/// Holds `data` to `bytes` both ways: it encodes to exactly these bytes,
/// and these bytes decode to exactly it.
fn pinned(data: FieldData, bytes: Vec<u8>) {
    assert_eq!(
        encode_data(&data),
        bytes,
        "{} encodes differently",
        data.schema()
    );
    assert_eq!(
        decode(&bytes).unwrap(),
        data,
        "{} decodes differently",
        data.schema()
    );
}

#[test]
fn words_are_little_endian_one_scalar_after_another() {
    pinned(
        field(
            "f,i,b",
            Domain::Rows(2),
            &[1.5f32.to_bits(), (-2i32) as u32, 1, 0.0f32.to_bits(), 7, 0],
        ),
        file(
            "Rows 2 1 1 f,i,b",
            &[
                0x00, 0x00, 0xC0, 0x3F, 0xFE, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
        ),
    );
    pinned(
        field("i", Domain::Volume(1, 1, 2), &[5, (-1i32) as u32]),
        file(
            "Volume 1 1 2 i",
            &[0x05, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
    );
}

#[test]
fn a_wide_scalar_is_eight_bytes_low_word_first() {
    let double = 0.1f64.to_bits();
    let long = (-3i64) as u64;
    pinned(
        field(
            "d,l",
            Domain::Rows(1),
            &[
                double as u32,
                (double >> 32) as u32,
                long as u32,
                (long >> 32) as u32,
            ],
        ),
        file(
            "Rows 1 1 1 d,l",
            &[
                0x9A, 0x99, 0x99, 0x99, 0x99, 0x99, 0xB9, 0x3F, //
                0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ],
        ),
    );
}

#[test]
fn a_narrow_scalar_keeps_only_its_bytes() {
    let floats = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    pinned(
        field(
            "f:unorm8,f:unorm8,f:unorm8,f:unorm8",
            Domain::Grid(2, 1),
            &floats(&[1.0, 0.0, 0.5, 1.0, 0.0, 0.25, 1.0, 0.0]),
        ),
        file(
            "Grid 2 1 1 f:unorm8,f:unorm8,f:unorm8,f:unorm8",
            &[0xFF, 0x00, 0x80, 0xFF, 0x00, 0x40, 0xFF, 0x00],
        ),
    );
    pinned(
        field(
            "f:f16,i:i16",
            Domain::Rows(2),
            &[1.0f32.to_bits(), (-2i32) as u32, (-0.5f32).to_bits(), 300],
        ),
        file(
            "Rows 2 1 1 f:f16,i:i16",
            &[0x00, 0x3C, 0xFE, 0xFF, 0x00, 0xB8, 0x2C, 0x01],
        ),
    );
}

#[test]
fn every_encoding_a_file_may_name_is_spelled_as_it_was() {
    // A whole word is what a scalar is when no encoding is named, so a
    // file never spells `word`.
    for name in [
        "f16", "unorm8", "snorm8", "unorm16", "u8", "i8", "u16", "i16",
    ] {
        let encoding = Encoding::parse(name).unwrap_or_else(|| panic!("`{name}` no longer reads"));
        assert_eq!(encoding.name(), name);
    }
    for letter in ["f", "i", "b", "d", "l"] {
        let schema: Schema = letter
            .parse()
            .unwrap_or_else(|e| panic!("`{letter}` no longer reads: {e}"));
        assert_eq!(schema.to_string(), letter);
    }
}

#[test]
fn every_file_a_past_orla_wrote_still_reads() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let mut found = 0;
    let mut dirs = vec![examples];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|ext| ext == "field") {
                let bytes = std::fs::read(&path).unwrap();
                decode(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                found += 1;
            }
        }
    }
    assert!(found >= 10, "found only {found} files");
}
