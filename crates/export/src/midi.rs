//! A score as a standard MIDI file, from the rows the importer reads:
//! every note becomes a strike and a release, the seconds become ticks
//! at a tempo the caller names, and what a music editor opens is what
//! comes out. Nothing outside is needed to write one.

/// One note, as the Field holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// Seconds from the start of the song.
    pub start: f32,
    /// Seconds it sounds for.
    pub duration: f32,
    /// The key, 0 to 127.
    pub pitch: i32,
    /// How hard it was struck, 0 to 1.
    pub velocity: f32,
    /// The channel, 0 to 15.
    pub channel: i32,
}

/// Ticks to the quarter note: fine enough that a millisecond lands on
/// one at any usual tempo, small enough to read.
pub const DIVISION: u16 = 480;

/// A MIDI file of these notes at `beats` quarter notes a minute: one
/// track, the tempo first, then every strike and release in time order.
pub fn midi(notes: &[Note], beats: f32) -> Vec<u8> {
    let beats = if beats > 0.0 { beats } else { 120.0 };
    let micros_per_quarter = (60_000_000.0 / beats).round().clamp(1.0, 16_777_215.0) as u32;
    let seconds_per_tick = 60.0 / (beats * f32::from(DIVISION));
    let tick = |seconds: f32| (seconds.max(0.0) / seconds_per_tick).round() as u64;
    // A release before a strike at the same tick, so a key struck again
    // where it was released reads back as two notes.
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
    body.extend(varint(0));
    body.extend([0xFF, 0x51, 0x03]);
    body.extend(&micros_per_quarter.to_be_bytes()[1..]);
    let mut at = 0u64;
    for (tick, _, event) in events {
        body.extend(varint(tick - at));
        body.extend(event);
        at = tick;
    }
    body.extend(varint(0));
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

/// A variable-length quantity: seven bits a byte, the high bit saying
/// another follows.
fn varint(mut value: u64) -> Vec<u8> {
    let mut out = vec![(value & 0x7F) as u8];
    value >>= 7;
    while value > 0 {
        out.insert(0, ((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out
}
