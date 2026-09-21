//! A MIDI file as rows the Host reads: every note with when it starts,
//! how long it lasts, which key it is, how hard it was struck and on
//! which channel. The file's ticks and its tempo map become seconds
//! here, so a Kernel plays a score against `time()` and knows nothing
//! about MIDI.
//!
//! A standard MIDI file is chunks: `MThd` with the format, the track
//! count and the division, then a `MTrk` for each track, each a run of
//! events prefixed by a delta time. Nothing outside is needed to read
//! one, so this takes no dependency.

use std::path::{Path, PathBuf};

use orla_runtime::{Domain, Scalar, Schema, file};

/// One note, as a scene reads it:
/// `struct Note { start: Float; duration: Float; pitch: Int; velocity: Float; channel: Int; }`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// Seconds from the start of the song.
    pub start: f32,
    /// Seconds it sounds for.
    pub duration: f32,
    /// The key, 0 to 127; 60 is middle C.
    pub pitch: i32,
    /// How hard it was struck, 0 to 1.
    pub velocity: f32,
    /// The channel it played on, 0 to 15; 9 is the drum channel.
    pub channel: i32,
}

/// The Field a score is written as: five words a note.
pub fn schema() -> Schema {
    Schema::new(vec![
        Scalar::Float,
        Scalar::Float,
        Scalar::Int,
        Scalar::Float,
        Scalar::Int,
    ])
}

/// The words of a note, in the schema's order.
pub fn words(note: &Note) -> [u32; 5] {
    [
        note.start.to_bits(),
        note.duration.to_bits(),
        note.pitch as u32,
        note.velocity.to_bits(),
        note.channel as u32,
    ]
}

/// A note read back from its words.
pub fn note(words: &[u32]) -> Note {
    Note {
        start: f32::from_bits(words[0]),
        duration: f32::from_bits(words[1]),
        pitch: words[2] as i32,
        velocity: f32::from_bits(words[3]),
        channel: words[4] as i32,
    }
}

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
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("song")
        .to_owned();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let file_path = out_dir.join(format!("{stem}.notes.field"));
    let all: Vec<u32> = song.notes.iter().flat_map(words).collect();
    if song.notes.is_empty() {
        return Err(format!(
            "{}: the file holds no note; a Field has no empty extent",
            path.display()
        ));
    }
    std::fs::write(
        &file_path,
        file::encode(&schema(), Domain::Rows(song.notes.len() as u32), &all),
    )
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

/// The notes a file holds, and what was left out of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Song {
    /// The notes, by when they start.
    pub notes: Vec<Note>,
    /// How many tracks the file held.
    pub tracks: usize,
    /// What was met and left out.
    pub left_out: Vec<String>,
}

/// A note while it is still sounding: the second it began, taken at the
/// tempo then in force, and how hard it was struck.
struct Sounding {
    start: f64,
    velocity: f32,
}

/// Reads a standard MIDI file: the header, then every track, with the
/// tempo map applied so the notes come out in seconds.
pub fn read(bytes: &[u8]) -> Result<Song, String> {
    let mut at = 0usize;
    let (kind, length) = chunk(bytes, &mut at)?;
    if kind != *b"MThd" || length < 6 {
        return Err("not a MIDI file: it does not begin with a header chunk".into());
    }
    let header = &bytes[at..at + length];
    at += length;
    let format = u16::from_be_bytes([header[0], header[1]]);
    let tracks = u16::from_be_bytes([header[2], header[3]]) as usize;
    let division = i16::from_be_bytes([header[4], header[5]]);
    if division <= 0 {
        return Err("SMPTE timing is not read; this takes ticks to the quarter note".into());
    }
    let per_quarter = division as u64;
    let mut left_out = Vec::new();
    if format == 2 {
        left_out.push("format 2, whose tracks are separate songs: they are read as one".into());
    }
    // Every track's events by tick, so one tempo map covers them all.
    let mut events: Vec<(u64, usize, Event)> = Vec::new();
    let mut read_tracks = 0usize;
    while at < bytes.len() {
        let (kind, length) = chunk(bytes, &mut at)?;
        let end = at + length;
        if end > bytes.len() {
            return Err("a track runs past the end of the file".into());
        }
        if kind != *b"MTrk" {
            at = end;
            left_out.push(format!(
                "a `{}` chunk, which is not a track",
                String::from_utf8_lossy(&kind)
            ));
            continue;
        }
        track(&bytes[at..end], read_tracks, &mut events, &mut left_out)?;
        read_tracks += 1;
        at = end;
    }
    if read_tracks != tracks {
        left_out.push(format!(
            "the header says {tracks} track(s) and the file holds {read_tracks}"
        ));
    }
    // In tick order, and within a tick a note off before a note on, so a
    // key struck again at the moment it is released pairs correctly.
    events.sort_by_key(|(tick, track, event)| (*tick, event.order(), *track));
    let mut seconds_at = 0.0f64;
    let mut tick_of_change = 0u64;
    let mut micros_per_quarter = 500_000.0f64; // 120 to the minute, the default
    let seconds = |tick: u64, tick_of_change: u64, seconds_at: f64, micros: f64| -> f64 {
        seconds_at + (tick - tick_of_change) as f64 * micros / (per_quarter as f64 * 1_000_000.0)
    };
    let mut sounding: std::collections::HashMap<(i32, i32), Vec<Sounding>> =
        std::collections::HashMap::new();
    let mut notes: Vec<Note> = Vec::new();
    for (tick, _, event) in events {
        // Where this tick falls, at the tempo in force since the last
        // change: a note remembers its own second, since the tempo may
        // change again while it sounds.
        let at = seconds(tick, tick_of_change, seconds_at, micros_per_quarter);
        match event {
            Event::Tempo(micros) => {
                seconds_at = at;
                tick_of_change = tick;
                micros_per_quarter = micros as f64;
            }
            Event::On {
                channel,
                pitch,
                velocity,
            } => sounding
                .entry((channel, pitch))
                .or_default()
                .push(Sounding {
                    start: at,
                    velocity,
                }),
            Event::Off { channel, pitch } => {
                let Some(open) = sounding.get_mut(&(channel, pitch)) else {
                    continue;
                };
                if open.is_empty() {
                    continue;
                }
                let began = open.remove(0);
                notes.push(Note {
                    start: began.start as f32,
                    duration: (at - began.start).max(0.0) as f32,
                    pitch,
                    velocity: began.velocity,
                    channel,
                });
            }
        }
    }
    let still_sounding: usize = sounding.values().map(Vec::len).sum();
    if still_sounding > 0 {
        left_out.push(format!(
            "{still_sounding} note(s) never released, which end where the song does"
        ));
        let last = notes
            .iter()
            .map(|note| note.start + note.duration)
            .fold(0.0f32, f32::max);
        for ((channel, pitch), open) in &sounding {
            for began in open {
                let start = began.start as f32;
                notes.push(Note {
                    start,
                    duration: (last - start).max(0.0),
                    pitch: *pitch,
                    velocity: began.velocity,
                    channel: *channel,
                });
            }
        }
    }
    // In the order a score reads: by when they start, then by key.
    notes.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.pitch.cmp(&b.pitch))
            .then(a.channel.cmp(&b.channel))
    });
    Ok(Song {
        notes,
        tracks: read_tracks,
        left_out,
    })
}

/// What a track says that a score keeps.
enum Event {
    On {
        channel: i32,
        pitch: i32,
        velocity: f32,
    },
    Off {
        channel: i32,
        pitch: i32,
    },
    Tempo(u32),
}

impl Event {
    /// Within one tick: the tempo first, then releases, then strikes.
    fn order(&self) -> u8 {
        match self {
            Self::Tempo(_) => 0,
            Self::Off { .. } => 1,
            Self::On { .. } => 2,
        }
    }
}

/// A chunk's kind and the length of what follows it.
fn chunk(bytes: &[u8], at: &mut usize) -> Result<([u8; 4], usize), String> {
    if *at + 8 > bytes.len() {
        return Err("a chunk header runs past the end of the file".into());
    }
    let kind = [bytes[*at], bytes[*at + 1], bytes[*at + 2], bytes[*at + 3]];
    let length = u32::from_be_bytes([
        bytes[*at + 4],
        bytes[*at + 5],
        bytes[*at + 6],
        bytes[*at + 7],
    ]) as usize;
    *at += 8;
    Ok((kind, length))
}

/// A variable-length quantity: seven bits a byte, the high bit saying
/// another follows.
fn varint(bytes: &[u8], at: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for _ in 0..4 {
        let byte = *bytes
            .get(*at)
            .ok_or_else(|| "a number runs past the end of a track".to_string())?;
        *at += 1;
        value = (value << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("a number is longer than four bytes".into())
}

/// One track's events, added to `events` at their ticks.
fn track(
    bytes: &[u8],
    track: usize,
    events: &mut Vec<(u64, usize, Event)>,
    left_out: &mut Vec<String>,
) -> Result<(), String> {
    let mut at = 0usize;
    let mut tick = 0u64;
    let mut status = 0u8;
    while at < bytes.len() {
        tick += varint(bytes, &mut at)?;
        let mut byte = *bytes
            .get(at)
            .ok_or_else(|| "a track ends inside an event".to_string())?;
        if byte >= 0x80 {
            status = byte;
            at += 1;
            byte = *bytes.get(at).copied().get_or_insert(0);
        }
        let _ = byte;
        match status & 0xF0 {
            0x80 | 0x90 => {
                let (pitch, velocity) = (two(bytes, &mut at)?, two(bytes, &mut at)?);
                let channel = i32::from(status & 0x0F);
                let pitch = i32::from(pitch);
                // A strike with no force is a release, as the format says.
                if status & 0xF0 == 0x90 && velocity > 0 {
                    events.push((
                        tick,
                        track,
                        Event::On {
                            channel,
                            pitch,
                            velocity: f32::from(velocity) / 127.0,
                        },
                    ));
                } else {
                    events.push((tick, track, Event::Off { channel, pitch }));
                }
            }
            0xA0 | 0xB0 | 0xE0 => {
                two(bytes, &mut at)?;
                two(bytes, &mut at)?;
            }
            0xC0 | 0xD0 => {
                two(bytes, &mut at)?;
            }
            _ => match status {
                0xFF => {
                    let kind = two(bytes, &mut at)?;
                    let length = varint(bytes, &mut at)? as usize;
                    let data = bytes
                        .get(at..at + length)
                        .ok_or_else(|| "a meta event runs past the end of a track".to_string())?;
                    at += length;
                    match kind {
                        0x51 if length == 3 => events.push((
                            tick,
                            track,
                            Event::Tempo(
                                (u32::from(data[0]) << 16)
                                    | (u32::from(data[1]) << 8)
                                    | u32::from(data[2]),
                            ),
                        )),
                        0x2F => return Ok(()),
                        _ => {}
                    }
                }
                0xF0 | 0xF7 => {
                    let length = varint(bytes, &mut at)? as usize;
                    at += length;
                    left_out.push("a system exclusive message".into());
                }
                other => {
                    return Err(format!(
                        "a track holds {other:#04x}, which is not an event a file may carry"
                    ));
                }
            },
        }
    }
    Ok(())
}

/// One data byte of an event.
fn two(bytes: &[u8], at: &mut usize) -> Result<u8, String> {
    let byte = *bytes
        .get(*at)
        .ok_or_else(|| "a track ends inside an event".to_string())?;
    *at += 1;
    Ok(byte)
}
