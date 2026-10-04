//! A scene output written `to` a file lands whole, only in an allowed
//! directory, only after a changing evaluation commits; the scene reads
//! it back through an input that falls back on a node while it is absent.

use std::path::{Path, PathBuf};

use orla_kernel::{Context, Value};
use orla_session::{Event, Session};

const KERNELS: &str = "
kernel t.start@1 { requires { field.write; } output result: Rows<Int>; compute { return 5; } }
kernel t.next@1 { requires { field.read; field.write; } input last: Rows<Int>; output result: Rows<Int>; compute { return sample(last) + 1; } }
kernel t.over@1 { requires { field.read; field.write; } input x: Rows<Int>; param by: Int = 1; output result: Rows<Int>; compute { return sample(x) / by; } }
kernel t.bytes@1 { requires { field.write; } param top: Int = 255; output result: Rows<Int>; compute { return if element() == 3 { top } else { element() * 7 }; } }
";

fn scratch(name: &str, scene: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-writes-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("saves")).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.osn"), scene).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

fn wrote(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::Wrote { .. }))
        .count()
}

fn refused(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::WriteRefused { why, .. } => Some(why.clone()),
            _ => None,
        })
        .collect()
}

fn saved(path: &Path) -> Vec<u32> {
    orla_formats::read(path, None)
        .unwrap()
        .0
        .values()
        .into_owned()
}

const SAVING: &str = "
use \"k.orla\";
param by: Int = 1;
input saved: Rows<Int> = \"saves/n.field\" else start.result;
node start: t.start@1 { domain = 1; }
node next: t.next@1 { last <- saved; }
node over: t.over@1 { x <- next.result; by = by; }
output shown <- over.result;
output kept <- next.result to \"saves/n.field\";
";

/// A single file the person allowed (`--out`, `--save`) is for that
/// command's write: the scene's `to` writes only into allowed
/// directories, even for that file, and says how to allow it.
#[test]
fn a_file_allowed_for_the_caller_is_not_the_scenes_to_write() {
    let dir = scratch("caller-file", SAVING);
    let file = dir.join("saves/n.field");
    let mut session = open(&dir);
    session.allow_write_to(&file).unwrap();
    session.allow_write_to(&dir.join("output.ppm")).unwrap();
    session.evaluate(&Context::default()).unwrap();
    let events = session.sync();
    assert_eq!(wrote(&events), 0, "{events:?}");
    assert!(
        refused(&events)
            .iter()
            .any(|why| why.contains("writing files is not allowed; run with `--allow file.write=")),
        "{events:?}"
    );
    assert!(!file.exists());
    // The caller's own write to it lands.
    assert!(session.allowed_write(&file).is_ok());
}

#[test]
fn an_output_is_written_where_allowed_once_it_changed_and_read_back() {
    let dir = scratch("saving", SAVING);
    let file = dir.join("saves/n.field");
    let mut session = open(&dir);
    let context = Context::default();

    // Nothing allowed: nothing is written, and the reason is reported.
    session.evaluate(&context).unwrap();
    let events = session.sync();
    let why = refused(&events);
    assert!(
        why.iter().any(|why| why.contains("--allow file.write")),
        "{events:?}"
    );
    assert!(!file.exists());

    // Allowed: the start plus one lands.
    session.allow_writes(&dir.join("saves")).unwrap();
    session.evaluate(&context).unwrap();
    assert_eq!(saved(&file), [6]);
    // The file is read back: the input leaves its fallback, and the next
    // evaluation writes one more.
    let events = session.sync();
    assert_eq!(wrote(&events), 1, "{events:?}");
    session.evaluate(&context).unwrap();
    assert_eq!(saved(&file), [7]);
    // Nothing changed: nothing is written.
    session.evaluate(&context).unwrap();
    let events = session.sync();
    assert_eq!(wrote(&events), 1, "the write of 7 alone: {events:?}");
    session.evaluate(&context).unwrap();
    assert_eq!(wrote(&session.sync()), 1, "8, read back from 7");
    assert_eq!(saved(&file), [8]);

    // A failed evaluation commits and writes nothing: the file says 8,
    // the next would be 9, and a division by zero fails the frame.
    session.sync();
    session.set_param("by", Value::Int(0)).unwrap();
    assert!(session.evaluate(&context).is_err());
    assert_eq!(saved(&file), [8], "a failed evaluation wrote");

    // A new session reads what the last one wrote.
    let mut again = open(&dir);
    again.allow_writes(&dir.join("saves")).unwrap();
    let report = again.evaluate(&context).unwrap();
    let kept = report
        .outputs
        .iter()
        .find(|(name, _)| name == "kept")
        .unwrap()
        .1;
    assert_eq!(
        again.runtime().host().data(kept).unwrap().values().to_vec(),
        [9]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_write_never_leaves_the_allowed_directory() {
    let dir = scratch(
        "outside",
        "use \"k.orla\";
node start: t.start@1 { domain = 1; }
output kept <- start.result to \"saves/../outside.field\";
",
    );
    let mut session = open(&dir);
    session.allow_writes(&dir.join("saves")).unwrap();
    session.evaluate(&Context::default()).unwrap();
    let events = session.sync();
    let why = refused(&events);
    assert!(
        why.iter()
            .any(|why| why.contains("outside every directory")),
        "{events:?}"
    );
    assert!(!dir.join("outside.field").exists());
    // An allowance names an existing directory.
    assert!(session.allow_writes(&dir.join("nowhere")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bytes_are_written_as_bytes_and_only_bytes_are() {
    let dir = scratch(
        "bytes",
        "use \"k.orla\";
param top: Int = 255;
input back: Rows<Int> = bytes \"saves/b.bin\" else made.result;
node made: t.bytes@1 { top = top; domain = 5; }
output kept <- made.result to bytes \"saves/b.bin\";
output read <- back;
",
    );
    let mut session = open(&dir);
    session.allow_writes(&dir.join("saves")).unwrap();
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        std::fs::read(dir.join("saves/b.bin")).unwrap(),
        [0, 7, 14, 255, 28]
    );
    // A value that is not a byte is not written, and its row is named.
    session.set_param("top", Value::Int(256)).unwrap();
    session.evaluate(&Context::default()).unwrap();
    let why = refused(&session.sync());
    assert!(
        why.iter().any(|why| why.contains("row 3 holds 256")),
        "{why:?}"
    );
    assert_eq!(
        std::fs::read(dir.join("saves/b.bin")).unwrap(),
        [0, 7, 14, 255, 28]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Field of a wide layout: its header runs past the bytes first read.
const WIDE: &str = "
struct Wide { n: Int; rest: [Int; 200]; }
kernel w.start@1 { requires { field.write; } output result: Rows<Wide>; compute { return Wide { n: 5, rest: [each 200 { index() }] }; } }
kernel w.next@1 { requires { field.read; field.write; } input last: Rows<Wide>; output result: Rows<Wide>; compute { let w = sample(last); return Wide { ..w, n: w.n + 1 }; } }
";

/// A file whose header is longer than the bytes first read for it is
/// read whole: a wide layout written `to` a file is read back.
#[test]
fn a_wide_layout_written_is_read_back() {
    let dir = scratch(
        "wide",
        "use \"w.orla\";
input saved: Rows<Wide> = \"saves/w.field\" else start.result;
node start: w.start@1 { domain = 1; }
node next: w.next@1 { last <- saved; }
output kept <- next.result to \"saves/w.field\";
",
    );
    std::fs::write(dir.join("w.orla"), WIDE).unwrap();
    let file = dir.join("saves/w.field");
    let mut session = open(&dir);
    session.allow_writes(&dir.join("saves")).unwrap();
    let context = Context::default();
    session.evaluate(&context).unwrap();
    assert_eq!(saved(&file)[0], 6);
    let bytes = std::fs::read(&file).unwrap();
    let header_end = bytes
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'\n')
        .nth(1)
        .unwrap()
        .0;
    assert!(
        header_end > orla_formats::file::HEADER_BYTES,
        "{header_end}"
    );
    let events = session.sync();
    assert_eq!(wrote(&events), 1, "{events:?}");
    session.evaluate(&context).unwrap();
    assert_eq!(saved(&file)[0], 7, "the file was read back");
    assert_eq!(saved(&file)[200], 199);
}
