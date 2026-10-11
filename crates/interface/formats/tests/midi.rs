//! A MIDI file read as notes and written back. The file is built byte by
//! byte from the format's definition, so the reader is what is tested.

use entelechy_formats::midi::{self as read, Note, encode};

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
    // A quarter note, a rest, then two notes at once, the second in
    // running status.
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
    // The two together start a beat after the first ends.
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
    let notes: Vec<Note> = vec![
        Note {
            start: 0.0,
            duration: 0.5,
            pitch: 60,
            velocity: 100.0 / 127.0,
            channel: 0,
        },
        Note {
            start: 0.5,
            duration: 0.25,
            pitch: 67,
            velocity: 80.0 / 127.0,
            channel: 1,
        },
        Note {
            start: 0.5,
            duration: 1.0,
            pitch: 48,
            velocity: 40.0 / 127.0,
            channel: 1,
        },
    ];
    let bytes = encode(&notes, 120.0);
    let song = read::read(&bytes).unwrap();
    assert_eq!(song.notes.len(), notes.len());
    // A score reads back ordered by start, then key.
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
    let slow = read::read(&encode(&notes, 75.0)).unwrap();
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

#[test]
fn a_score_read_live_is_the_score_imported() {
    // A scene naming the `.mid` reads the same rows as the `.field` the
    // file imports to.
    let tempo = track(&[(0, vec![0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])]);
    let notes = track(&[
        (0, vec![0x90, 60, 100]),
        (480, vec![0x80, 60, 0]),
        (0, vec![0x91, 64, 80]),
        (0, vec![67, 70]),
        (240, vec![0x81, 64, 0]),
        (240, vec![0x81, 67, 0]),
    ]);
    let bytes = file(480, &[tempo, notes]);
    let dir = std::env::temp_dir().join(format!("entelechy-midi-score-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let song = dir.join("song.mid");
    std::fs::write(&song, &bytes).unwrap();
    let imported = dir.join("song.notes.field");
    let rows = read::field(&read::read(&bytes).unwrap()).unwrap();
    std::fs::write(&imported, entelechy_formats::file::encode_data(&rows)).unwrap();
    let (live, detail) = entelechy_formats::read(&song, None).unwrap();
    let (from_file, _) = entelechy_formats::read(&imported, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(live, from_file, "{detail}");
    assert_eq!(live.domain(), entelechy_host::Domain::Rows(3));
    // Written out again, it reads back as the same rows.
    let written = entelechy_formats::encode(&live, "mid", None).unwrap();
    let again = entelechy_formats::midi::field(&read::read(&written).unwrap()).unwrap();
    assert_eq!(again.domain(), live.domain());
}

#[test]
fn a_field_that_does_not_fit_a_format_says_what_it_is_not() {
    use entelechy_formats::EncodeError;
    use entelechy_host::{Domain, FieldData, Scalar, Schema};
    let rows = FieldData::from_storage(
        Schema::new(vec![Scalar::Float]),
        Domain::Rows(2),
        vec![0.5f32.to_bits(); 2],
    )
    .unwrap();
    assert_eq!(
        entelechy_formats::encode(&rows, "png", None),
        Err(EncodeError::Unfit("not a grid"))
    );
    assert_eq!(
        entelechy_formats::encode(&rows, "mid", None),
        Err(EncodeError::Unfit(
            "not rows of a note (start, duration, pitch, velocity, channel)"
        ))
    );
    assert_eq!(
        entelechy_formats::encode(&rows, "pdf", None),
        Err(EncodeError::Unknown)
    );
    assert!(entelechy_formats::encode(&rows, "wav", Some(8000)).is_ok());
}

#[test]
fn a_header_longer_than_the_file_is_an_error() {
    // A header chunk that says it runs 4 GB, in a 14-byte file.
    let mut file = b"MThd".to_vec();
    file.extend(u32::MAX.to_be_bytes());
    file.extend([0, 0, 0, 1, 0, 0x60]);
    let error = read::read(&file).unwrap_err();
    assert!(error.contains("past the end"), "{error}");
}

#[test]
fn many_notes_on_one_key_read_in_linear_time() {
    // Strikes of one key all sounding at once, then as many releases:
    // each release ends the earliest strike.
    let count = 400_000u32;
    let mut events = Vec::new();
    for _ in 0..count {
        events.extend([0, 0x90, 60, 100]);
    }
    for _ in 0..count {
        events.extend([1, 0x80, 60, 0]);
    }
    events.extend([0, 0xFF, 0x2F, 0]);
    let mut file = b"MThd".to_vec();
    file.extend(6u32.to_be_bytes());
    file.extend([0, 0, 0, 1, 0, 0x60]);
    file.extend(b"MTrk");
    file.extend((events.len() as u32).to_be_bytes());
    file.extend(events);
    let started = std::time::Instant::now();
    let song = read::read(&file).unwrap();
    assert_eq!(song.notes.len(), count as usize);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}
