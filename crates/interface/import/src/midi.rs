//! A MIDI file brought in as the `.field` of notes a scene reads. The
//! reading is [`orla_formats::midi`], which a scene also reaches live
//! when an input names a `.mid` itself; this writes what it read beside
//! the other imports, and says what was left out.

use std::path::{Path, PathBuf};

pub use orla_formats::midi::{Note, Song, read, schema};

/// What a MIDI import produced.
pub struct Imported {
    /// The Field written.
    pub file: PathBuf,
    /// The notes, in order of when they start.
    pub notes: Vec<Note>,
    /// How long the song is, in seconds.
    pub seconds: f32,
    /// The tracks the file holds.
    pub tracks: usize,
    /// What was met and left out, in words.
    pub left_out: Vec<String>,
}

/// Reads `path` and writes `<stem>.notes.field` into `out_dir`.
pub fn import(path: &Path, out_dir: &Path) -> Result<Imported, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let song = read(&bytes)?;
    let data = orla_formats::midi::field(&song).map_err(|e| format!("{}: {e}", path.display()))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("song")
        .to_owned();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let file_path = out_dir.join(format!("{stem}.notes.field"));
    std::fs::write(&file_path, orla_host::file::encode_data(&data))
        .map_err(|e| format!("{}: {e}", file_path.display()))?;
    let seconds = song
        .notes
        .iter()
        .map(|note| note.start + note.duration)
        .fold(0.0f32, f32::max);
    Ok(Imported {
        file: file_path,
        notes: song.notes,
        seconds,
        tracks: song.tracks,
        left_out: song.left_out,
    })
}
