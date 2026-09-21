//! A MIDI file read as notes, and the same notes written back: the
//! file here is built byte by byte from the format's own definition, so
//! what is tested is the reader, not the writer.

use orla_export::midi as write;
use orla_import::midi as read;

/// A variable-length quantity, as the format writes one.
fn varint(mut value: u32) -> Vec<u8> {
    let mut out = vec![(value & 0x7F) as u8];
    value >>= 7;
    while value > 0 {
        out.insert(0, ((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out
}

fn track(events: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (delta, event) in events {
        body.extend(varint(*delta));
        body.extend(event);
    }
    body.extend(varint(0));
    body.extend([0xFF, 0x2F, 0x00]);
    let mut out = b"MTrk".to_vec();
    out.extend((body.len() as u32).to_be_bytes());
    out.extend(body);
    out
}

fn file(division: u16, tracks: &[Vec<u8>]) -> Vec<u8> {
    let mut out = b"MThd".to_vec();
    out.extend(6u32.to_be_bytes());
    out.extend(1u16.to_be_bytes());
    out.extend((tracks.len() as u16).to_be_bytes());
    out.extend(division.to_be_bytes());
    for track in tracks {
        out.extend(track);
    }
    out
}

#[test]
fn a_file_reads_as_notes_in_seconds() {
    let division = 480u16;
    let tempo = track(&[(0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])]); // 500000: 120 a minute
    // A quarter note, then a rest, then two notes at once; the second
    // uses running status, which a file may leave the status byte out of.
    let notes = track(&[
        (0, vec![0x90, 60, 100]),
        (480, vec![0x80, 60, 0]),
        (480, vec![0x90, 64, 64]),
        (0, vec![67, 64]), // running status: another strike
        (960, vec![0x80, 64, 0]),
        (0, vec![0x80, 67, 0]),
    ]);
    let song = read::read(&file(division, &[tempo, notes])).unwrap();
    assert_eq!(song.tracks, 2);
    assert_eq!(song.notes.len(), 3);
    let first = song.notes[0];
    assert_eq!(first.pitch, 60);
    assert!((first.start - 0.0).abs() < 1e-6, "{first:?}");
    assert!((first.duration - 0.5).abs() < 1e-6, "{first:?}");
    assert!((first.velocity - 100.0 / 127.0).abs() < 1e-6, "{first:?}");
    assert_eq!(first.channel, 0);
    // The two that sound together start a beat after the first ended.
    for note in &song.notes[1..] {
        assert!((note.start - 1.0).abs() < 1e-6, "{note:?}");
        assert!((note.duration - 1.0).abs() < 1e-6, "{note:?}");
    }
    assert_eq!(song.notes[1].pitch, 64);
    assert_eq!(song.notes[2].pitch, 67);
}

#[test]
fn a_tempo_change_moves_the_notes_after_it() {
    let division = 96u16;
    // Half way through, the tempo doubles: 250000 microseconds a quarter.
    let events = track(&[
        (0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]),
        (0, vec![0x90, 60, 127]),
        (96, vec![0x80, 60, 0]),
        (0, vec![0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]),
        (0, vec![0x90, 62, 127]),
        (96, vec![0x80, 62, 0]),
    ]);
    let song = read::read(&file(division, &[events])).unwrap();
    assert_eq!(song.notes.len(), 2);
    assert!(
        (song.notes[0].duration - 0.5).abs() < 1e-6,
        "{:?}",
        song.notes[0]
    );
    assert!(
        (song.notes[1].start - 0.5).abs() < 1e-6,
        "{:?}",
        song.notes[1]
    );
    assert!(
        (song.notes[1].duration - 0.25).abs() < 1e-6,
        "{:?}",
        song.notes[1]
    );
}

#[test]
fn what_is_written_reads_back_the_same() {
    let notes: Vec<write::Note> = vec![
        write::Note {
            start: 0.0,
            duration: 0.5,
            pitch: 60,
            velocity: 100.0 / 127.0,
            channel: 0,
        },
        write::Note {
            start: 0.5,
            duration: 0.25,
            pitch: 67,
            velocity: 80.0 / 127.0,
            channel: 1,
        },
        write::Note {
            start: 0.5,
            duration: 1.0,
            pitch: 48,
            velocity: 40.0 / 127.0,
            channel: 1,
        },
    ];
    let bytes = write::midi(&notes, 120.0);
    let song = read::read(&bytes).unwrap();
    assert_eq!(song.notes.len(), notes.len());
    // A score reads back in its own order: by when a note starts, then
    // by its key.
    let mut expected = notes.clone();
    expected.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.pitch.cmp(&b.pitch)));
    for (read, written) in song.notes.iter().zip(&expected) {
        assert_eq!(read.pitch, written.pitch);
        assert_eq!(read.channel, written.channel);
        assert!((read.start - written.start).abs() < 1e-3, "{read:?}");
        assert!((read.duration - written.duration).abs() < 1e-3, "{read:?}");
        assert!((read.velocity - written.velocity).abs() < 1e-3, "{read:?}");
    }
    // At another tempo the ticks differ and the seconds do not.
    let slow = read::read(&write::midi(&notes, 75.0)).unwrap();
    for (read, written) in slow.notes.iter().zip(&expected) {
        assert!((read.start - written.start).abs() < 1e-3, "{read:?}");
        assert!((read.duration - written.duration).abs() < 1e-3, "{read:?}");
    }
}

#[test]
fn what_is_not_a_midi_file_says_so() {
    let error = read::read(b"not a midi file at all").unwrap_err();
    assert!(error.contains("does not begin with a header"), "{error}");
    let mut smpte = b"MThd".to_vec();
    smpte.extend(6u32.to_be_bytes());
    smpte.extend(0u16.to_be_bytes());
    smpte.extend(1u16.to_be_bytes());
    smpte.extend((-25i16).to_be_bytes());
    let error = read::read(&smpte).unwrap_err();
    assert!(error.contains("SMPTE"), "{error}");
}
