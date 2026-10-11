//! A standard MIDI file as rows of notes: start, duration, key,
//! velocity, channel. Ticks and the tempo map become seconds here, so a
//! Kernel plays a score against `time()`. Encoding writes one track of
//! strikes and releases at the caller's tempo.

use std::collections::VecDeque;

use entelechy_host::{Domain, FieldData, Scalar, Schema};

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

/// A song's notes as rows of [`schema`]; a song needs at least one note.
pub fn field(song: &Song) -> Result<FieldData, String> {
    if song.notes.is_empty() {
        return Err("the file holds no note; a Field has no empty extent".into());
    }
    let words: Vec<u32> = song.notes.iter().flat_map(words).collect();
    FieldData::from_storage(schema(), Domain::Rows(song.notes.len() as u32), words)
        .map_err(|e| e.to_string())
}

/// Ticks to the quarter note in a written file.
pub const DIVISION: u16 = 480;

/// A MIDI file of these notes at `beats` quarter notes a minute: one
/// track, the tempo, then strikes and releases in time order.
pub fn encode(notes: &[Note], beats: f32) -> Vec<u8> {
    let beats = if beats > 0.0 { beats } else { 120.0 };
    let micros_per_quarter = (60_000_000.0 / beats).round().clamp(1.0, 16_777_215.0) as u32;
    let seconds_per_tick = 60.0 / (beats * f32::from(DIVISION));
    let tick = |seconds: f32| (seconds.max(0.0) / seconds_per_tick).round() as u64;
    // Releases precede strikes at a tick, so a re-struck key reads back
    // as two notes.
    let mut events: Vec<(u64, u8, [u8; 3])> = Vec::new();
    for note in notes {
        let channel = note.channel.clamp(0, 15) as u8;
        let pitch = note.pitch.clamp(0, 127) as u8;
        let velocity = (note.velocity * 127.0).round().clamp(1.0, 127.0) as u8;
        let start = tick(note.start);
        let end = start + tick(note.duration).max(1);
        events.push((start, 1, [0x90 | channel, pitch, velocity]));
        events.push((end, 0, [0x80 | channel, pitch, 0]));
    }
    events.sort_by_key(|(tick, order, event)| (*tick, *order, event[1]));
    let mut body: Vec<u8> = Vec::new();
    body.extend(quantity(0));
    body.extend([0xFF, 0x51, 0x03]);
    body.extend(&micros_per_quarter.to_be_bytes()[1..]);
    let mut at = 0u64;
    for (tick, _, event) in events {
        body.extend(quantity(tick - at));
        body.extend(event);
        at = tick;
    }
    body.extend(quantity(0));
    body.extend([0xFF, 0x2F, 0x00]);
    let mut out: Vec<u8> = Vec::new();
    out.extend(b"MThd");
    out.extend(6u32.to_be_bytes());
    out.extend(0u16.to_be_bytes()); // format 0: one track holds it all
    out.extend(1u16.to_be_bytes());
    out.extend(DIVISION.to_be_bytes());
    out.extend(b"MTrk");
    out.extend((body.len() as u32).to_be_bytes());
    out.extend(body);
    out
}

/// Writes a variable-length quantity: seven bits a byte, high bit set
/// while another follows.
fn quantity(mut value: u64) -> Vec<u8> {
    let mut out = vec![(value & 0x7F) as u8];
    value >>= 7;
    while value > 0 {
        out.insert(0, ((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out
}

/// The notes a file holds, and what was left out of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Song {
    /// The notes, by when they start.
    pub notes: Vec<Note>,
    /// How many tracks the file held.
    pub tracks: usize,
    /// What was left out.
    pub left_out: Vec<String>,
}

/// A sounding note: its start second and velocity.
struct Sounding {
    start: f64,
    velocity: f32,
}

/// Reads a standard MIDI file, notes in seconds by the tempo map.
pub fn read(bytes: &[u8]) -> Result<Song, String> {
    let mut at = 0usize;
    let not_midi = || "not a MIDI file: it does not begin with a header chunk".to_owned();
    if !bytes.starts_with(b"MThd") {
        return Err(not_midi());
    }
    let (_, header) = chunk(bytes, &mut at)?;
    if header.len() < 6 {
        return Err(not_midi());
    }
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
    // All tracks' events merged by tick under one tempo map.
    let mut events: Vec<(u64, usize, Event)> = Vec::new();
    let mut read_tracks = 0usize;
    while at < bytes.len() {
        let (kind, body) = chunk(bytes, &mut at)?;
        if kind != *b"MTrk" {
            left_out.push(format!(
                "a `{}` chunk, which is not a track",
                String::from_utf8_lossy(&kind)
            ));
            continue;
        }
        track(body, read_tracks, &mut events, &mut left_out)?;
        read_tracks += 1;
    }
    if read_tracks != tracks {
        left_out.push(format!(
            "the header says {tracks} track(s) and the file holds {read_tracks}"
        ));
    }
    // Tick order; within a tick, note off before note on.
    events.sort_by_key(|(tick, track, event)| (*tick, event.order(), *track));
    let mut seconds_at = 0.0f64;
    let mut tick_of_change = 0u64;
    let mut micros_per_quarter = 500_000.0f64; // 120 to the minute, the default
    let seconds = |tick: u64, tick_of_change: u64, seconds_at: f64, micros: f64| -> f64 {
        seconds_at + (tick - tick_of_change) as f64 * micros / (per_quarter as f64 * 1_000_000.0)
    };
    // Each key's sounding notes, the earliest first; keys in order, so
    // what is never released is found in one order.
    let mut sounding: std::collections::BTreeMap<(i32, i32), VecDeque<Sounding>> =
        std::collections::BTreeMap::new();
    let mut notes: Vec<Note> = Vec::new();
    for (tick, _, event) in events {
        // The tick's second at the tempo in force; a note keeps its own
        // start second across later tempo changes.
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
                .push_back(Sounding {
                    start: at,
                    velocity,
                }),
            Event::Off { channel, pitch } => {
                let Some(began) = sounding
                    .get_mut(&(channel, pitch))
                    .and_then(VecDeque::pop_front)
                else {
                    continue;
                };
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
    let still_sounding: usize = sounding.values().map(VecDeque::len).sum();
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
    // By start, then by key.
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

/// A track event a score keeps.
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
fn chunk<'b>(bytes: &'b [u8], at: &mut usize) -> Result<([u8; 4], &'b [u8]), String> {
    let Some(head) = bytes.get(*at..).and_then(|rest| rest.get(..8)) else {
        return Err("a chunk header runs past the end of the file".into());
    };
    let kind = [head[0], head[1], head[2], head[3]];
    let length = u32::from_be_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let Some(body) = bytes[*at + 8..].get(..length) else {
        return Err(format!(
            "a `{}` chunk runs past the end of the file",
            String::from_utf8_lossy(&kind)
        ));
    };
    *at += 8 + length;
    Ok((kind, body))
}

/// Reads a variable-length quantity.
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
                // A note on of velocity zero is a release.
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
