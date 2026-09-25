//! The file formats Orla reads and writes outside its own: PNG, PPM, WAV
//! and MIDI decoded into Host Fields and encoded out of them, the
//! inflate PNG needs, and the base64 the protocol and glTF carry bytes
//! in. [`read`] and [`encode`] choose among them by a file's extension.
//!
//! Each is pure bytes in and bytes or a Field out, with no session, no
//! runtime and no dependency beyond the Host, so an importer, an
//! exporter and a session share one decoder and none of them owns it.
//!
//! The set is closed on purpose. A format belongs here when it decodes
//! to a numerical Field, needs nothing outside the workspace, and is
//! worth reading live as a scene's input. A format whose meaning is the
//! application's (a level, a ROM, a text) is read as bytes
//! ([`read_bytes`]) and parsed by a Kernel; a format that needs a
//! library or is a document rather than a Field (glTF, MP3, a PDF) is
//! converted outside the runtime by `orla-import` or a bridge.

#![warn(missing_docs)]

pub mod base64;
pub mod file;
pub mod inflate;
pub mod midi;
pub mod png;
pub mod ppm;
pub mod rig;
pub mod wav;

use std::path::Path;

use orla_host::{Domain, FieldData, Scalar, Schema, Shape};

/// The extensions [`read`] and [`encode`] know, as an error names them.
const KNOWN: &str = "use .field, .png, .ppm, .pgm, .wav, or .mid";

/// The frames a second a WAV is written at when none is named.
pub const DEFAULT_RATE: u32 = 48_000;

/// The quarter notes a minute a MIDI file is written at when none is
/// named.
pub const DEFAULT_BEATS: f32 = 120.0;

/// A file's extension, lowercased; empty when it has none.
pub fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// Reads a file as its bytes: `Rows<Int>`, one element a byte, 0 to
/// 255. What a Kernel does with them is the Kernel's business, which is
/// how a scene reads a format nothing here knows.
pub fn read_bytes(path: &Path) -> Result<(FieldData, String), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode_bytes(&bytes)
}

/// Bytes already read, as [`read_bytes`] takes a file's.
pub fn decode_bytes(bytes: &[u8]) -> Result<(FieldData, String), String> {
    if bytes.is_empty() {
        return Err("the file is empty; a Field has no empty extent".into());
    }
    let words: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let data = FieldData::from_storage(
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(words.len() as u32),
        words,
    )
    .map_err(|e| e.to_string())?;
    let detail = format!("{} byte(s)", bytes.len());
    Ok((data, detail))
}

/// Reads a Field file: `.field` as the Host writes it; `.png`, `.ppm`,
/// and `.pgm` as a grid of pixels, or as a `Grid<Float>` of luminance
/// when `want` says so; `.wav` as rows of one or two floats; `.mid` as
/// rows of [`midi::Note`]. Returns the Field and a line saying what
/// came in.
pub fn read(path: &Path, want: Option<&str>) -> Result<(FieldData, String), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode(&bytes, path, want)
}

/// A file's bytes already read, as [`read`] takes them; `path` names
/// the format by its extension.
pub fn decode(
    bytes: &[u8],
    path: &Path,
    want: Option<&str>,
) -> Result<(FieldData, String), String> {
    let gray = want.is_some_and(|ty| without_encoding(ty) == "Grid<Float>");
    let describe = |data: &FieldData| {
        format!(
            "{} {} {}",
            data.domain().shape(),
            data.domain().count(),
            data.schema()
        )
    };
    match extension(path).as_str() {
        "field" => {
            let data = crate::file::decode(bytes).map_err(|e| e.to_string())?;
            let detail = describe(&data);
            Ok((data, detail))
        }
        "png" => {
            let data = png::decode_field(bytes, gray).map_err(|e| e.to_string())?;
            let detail = describe(&data);
            Ok((data, detail))
        }
        "ppm" | "pgm" => {
            let data = ppm::decode_field(bytes).map_err(|e| e.to_string())?;
            let data = if gray && data.schema().is_pixel() {
                let words = data
                    .storage()
                    .chunks_exact(4)
                    .map(|p| {
                        (f32::from_bits(p[0]) * 0.2126
                            + f32::from_bits(p[1]) * 0.7152
                            + f32::from_bits(p[2]) * 0.0722)
                            .to_bits()
                    })
                    .collect();
                FieldData::from_storage(Schema::new(vec![Scalar::Float]), data.domain(), words)
                    .map_err(|e| e.to_string())?
            } else {
                data
            };
            let detail = describe(&data);
            Ok((data, detail))
        }
        "wav" => {
            let sound = wav::decode(bytes).map_err(|e| e.to_string())?;
            let detail = format!("{} at {} Hz", describe(&sound.field), sound.rate);
            Ok((sound.field, detail))
        }
        "mid" | "midi" => {
            let song = midi::read(bytes)?;
            let data = midi::field(&song)?;
            let detail = format!("{} ({} track(s))", describe(&data), song.tracks);
            Ok((data, detail))
        }
        _ => Err(format!("unknown Field file format; {KNOWN}")),
    }
}

/// A `Rows<Int>` of bytes, one to an element, 0 to 255, as those bytes:
/// what `read_bytes` reads, written back.
pub fn encode_bytes(data: &FieldData) -> Result<Vec<u8>, String> {
    if data.schema().scalars() != [Scalar::Int] || data.domain().shape() != Shape::Rows {
        return Err(format!(
            "written as bytes, the output must be a Rows<Int>, not {} {}",
            data.domain().shape(),
            data.schema()
        ));
    }
    data.values()
        .iter()
        .enumerate()
        .map(|(row, word)| {
            u8::try_from(*word as i32).map_err(|_| {
                format!(
                    "row {row} holds {}, which is not a byte (0 to 255)",
                    *word as i32
                )
            })
        })
        .collect()
}

/// A Field's words as the file `extension` names: `rate` is the frames
/// a second of a WAV and the quarter notes a minute of a MIDI file, and
/// is ignored otherwise.
pub fn encode(
    data: &FieldData,
    extension: &str,
    rate: Option<u32>,
) -> Result<Vec<u8>, EncodeError> {
    let unfit = |why: &'static str| Err(EncodeError::Unfit(why));
    let domain = data.domain();
    let floats = |words: &[u32]| -> Vec<f32> { words.iter().map(|w| f32::from_bits(*w)).collect() };
    match extension {
        "field" => Ok(crate::file::encode_data(data)),
        "ppm" => {
            if !data.schema().is_pixel() || domain.shape() != Shape::Grid {
                return unfit("not a grid of pixels");
            }
            let Ok(image) = data.to_image() else {
                return unfit("not a grid of pixels");
            };
            Ok(ppm::encode(&image))
        }
        "pgm" => {
            if data.schema().value_words() != 1 || domain.shape() != Shape::Grid {
                return unfit("not a Grid<Float>");
            }
            Ok(ppm::encode_gray(
                domain.width(),
                domain.height(),
                &floats(&data.values()),
            ))
        }
        "png" => {
            if domain.shape() != Shape::Grid {
                return unfit("not a grid");
            }
            let values = floats(&data.values());
            let pixels: Vec<[f32; 4]> = if data.schema().is_pixel() {
                values
                    .chunks_exact(4)
                    .map(|p| [p[0], p[1], p[2], p[3]])
                    .collect()
            } else if data.schema().value_words() == 1 {
                values.iter().map(|v| [*v, *v, *v, 1.0]).collect()
            } else {
                return unfit("neither pixels nor a Grid<Float>");
            };
            Ok(png::encode(domain.width(), domain.height(), &pixels))
        }
        "wav" => {
            let channels = data.schema().value_words();
            if domain.shape() != Shape::Rows || channels > 2 || data.schema().is_pixel() {
                return unfit("not rows of one or two floats");
            }
            Ok(wav::encode(
                &data.values(),
                channels,
                rate.unwrap_or(DEFAULT_RATE),
            ))
        }
        "mid" | "midi" => {
            if domain.shape() != Shape::Rows || data.schema() != &midi::schema() {
                return unfit("not rows of a note (start, duration, pitch, velocity, channel)");
            }
            let notes: Vec<midi::Note> = data
                .values()
                .chunks_exact(midi::schema().value_words())
                .map(midi::note)
                .collect();
            let beats = rate.map_or(DEFAULT_BEATS, |beats| beats as f32);
            Ok(midi::encode(&notes, beats))
        }
        _ => Err(EncodeError::Unknown),
    }
}

/// Why a Field was not encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// The extension names no format here.
    Unknown,
    /// The Field does not fit the format; the words say what it is not,
    /// and follow "it is".
    Unfit(&'static str),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "unknown Field file format; {KNOWN}"),
            Self::Unfit(why) => f.write_str(why),
        }
    }
}

/// A port type with its storage encoding left out: `Grid<Float as
/// unorm8>` reads as `Grid<Float>`.
fn without_encoding(ty: &str) -> String {
    match ty.split_once(" as ") {
        Some((head, tail)) => format!("{head}{}", tail.find('>').map_or("", |at| &tail[at..])),
        None => ty.to_owned(),
    }
}
