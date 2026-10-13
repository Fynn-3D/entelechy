//! The caller places a scene's file-backed inputs and outputs
//! (`--input`, `--output`, [`Overrides`]): the inputs read there, the
//! outputs write there.

use std::path::{Path, PathBuf};

use entelechy_formats::directory::WRITE;
use entelechy_kernel::Context;
use entelechy_session::overrides::Overrides;
use entelechy_session::{Event, Session};

const SCENE: &str = "
kernel t.copy@1 { requires { field.read; field.write; } input words: Rows<Int>; output result: Rows<Int>; compute { return sample(words); } }
input disk: Rows<Int> = files \"project/\";
input note: Rows<Int> = bytes \"note.txt\";
input asked: Rows<Int> = bytes \"ops.bin\";
node listed: t.copy@1 { words <- disk; }
node noted: t.copy@1 { words <- note; }
node ops: t.copy@1 { words <- asked; }
output result <- listed.result;
output said <- noted.result;
output done <- ops.result to files \"project/\";
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-placed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for project in ["project", "other"] {
        std::fs::create_dir_all(dir.join(project)).unwrap();
    }
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    std::fs::write(dir.join("project/main.ent"), "m").unwrap();
    std::fs::write(dir.join("other/elsewhere.txt"), "e").unwrap();
    std::fs::write(dir.join("note.txt"), "n").unwrap();
    std::fs::write(dir.join("other.txt"), "o").unwrap();
    ask(&dir, &[]);
    dir
}

/// Writes the operations the scene reads, a byte a word.
fn ask(dir: &Path, ops: &[(i32, &str, &[u8])]) {
    let mut words = vec![ops.len() as u8];
    for (kind, path, data) in ops {
        words.extend([*kind as u8, path.len() as u8, data.len() as u8]);
        words.extend(path.bytes());
        words.extend(*data);
    }
    std::fs::write(dir.join("ops.bin"), words).unwrap();
}

fn text(session: &mut Session, output: &str) -> String {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some(output)).unwrap();
    session
        .field(handle)
        .unwrap()
        .values()
        .iter()
        .map(|w| char::from_u32(*w).unwrap_or('?'))
        .collect()
}

/// The paths under the `files` input, in order.
fn listed(session: &mut Session) -> Vec<String> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let words: Vec<u32> = session.field(handle).unwrap().values().into_owned();
    let mut at = 1;
    let mut found = Vec::new();
    for _ in 0..words[0] {
        let (path_len, size) = (words[at] as usize, words[at + 1] as usize);
        found.push(
            words[at + 2..at + 2 + path_len]
                .iter()
                .map(|w| char::from_u32(*w).unwrap())
                .collect(),
        );
        at += 2 + path_len + size;
    }
    found
}

fn opened(dir: &Path, overrides: Overrides) -> (Session, Vec<Event>) {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.place(overrides).unwrap();
    let events = session.sync();
    (session, events)
}

#[test]
fn inputs_and_outputs_read_and_write_where_the_caller_places_them() {
    let dir = scratch("launch");
    let mut overrides = Overrides::default();
    overrides.inputs.insert("disk".into(), dir.join("other"));
    overrides
        .inputs
        .insert("note".into(), dir.join("other.txt"));
    overrides.outputs.insert("done".into(), dir.join("other/"));
    let (mut session, events) = opened(&dir, overrides);
    assert!(session.graph().is_some(), "{events:?}");
    assert_eq!(listed(&mut session), vec!["elsewhere.txt".to_owned()]);
    assert_eq!(text(&mut session, "said"), "o");
    // The file read follows its changes where it is placed.
    std::fs::write(dir.join("other.txt"), "o2").unwrap();
    session.sync();
    assert_eq!(text(&mut session, "said"), "o2");
    // The output writes where it is placed, with the grant there.
    session.allow_writes(&dir.join("other")).unwrap();
    ask(&dir, &[(WRITE, "made.txt", b"hi")]);
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    session.sync();
    assert_eq!(std::fs::read(dir.join("other/made.txt")).unwrap(), b"hi");
    assert!(!dir.join("project/made.txt").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_name_the_scene_does_not_read_from_a_file_is_refused() {
    let dir = scratch("unknown");
    let mut overrides = Overrides::default();
    overrides.inputs.insert("nothing".into(), dir.join("other"));
    let (session, events) = opened(&dir, overrides);
    assert!(session.graph().is_none());
    let said: Vec<String> = events.iter().map(Event::to_string).collect();
    assert!(
        said.iter()
            .any(|line| line.contains("--input nothing") && line.contains("note, asked, disk")),
        "{said:?}"
    );
    let mut outputs = Overrides::default();
    outputs.outputs.insert("result".into(), dir.join("x.field"));
    let (session, events) = opened(&dir, outputs);
    assert!(session.graph().is_none());
    let said: Vec<String> = events.iter().map(Event::to_string).collect();
    assert!(
        said.iter().any(|line| line.contains("--output result")),
        "{said:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
