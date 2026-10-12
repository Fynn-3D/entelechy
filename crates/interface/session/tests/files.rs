//! A `files` input holds every file under a directory, its path and its
//! bytes, read again when one of them, or which there are, changes. An
//! output `to files` is a list of operations on the files under a
//! directory, done once each time the list changes: writes where writes
//! are allowed, removals and renames where removals are too.

use std::path::{Path, PathBuf};

use entelechy_formats::directory::{REMOVE, RENAME, WRITE};
use entelechy_kernel::Context;
use entelechy_session::{Event, Session};

const SCENE: &str = "
kernel t.copy@1 { requires { field.read; field.write; } input words: Rows<Int>; output result: Rows<Int>; compute { return sample(words); } }
input disk: Rows<Int> = files \"project/\";
input asked: Rows<Int> = bytes \"ops.bin\";
node listed: t.copy@1 { words <- disk; }
node ops: t.copy@1 { words <- asked; }
output result <- listed.result;
output done <- ops.result to files \"project/\";
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-files-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("project/lib")).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    std::fs::write(dir.join("project/main.ent"), "m").unwrap();
    std::fs::write(dir.join("project/lib/warm.ent"), "w1").unwrap();
    std::fs::write(dir.join("project/.hidden"), "h").unwrap();
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

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// The files the input holds, each its path and its bytes as text.
fn listed(session: &mut Session) -> Vec<(String, String)> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let words: Vec<u32> = session.field(handle).unwrap().values().into_owned();
    let mut at = 1;
    let mut found = Vec::new();
    for _ in 0..words[0] {
        let (path_len, size) = (words[at] as usize, words[at + 1] as usize);
        let text = |from: usize, n: usize| -> String {
            words[from..from + n]
                .iter()
                .map(|w| char::from_u32(*w).unwrap())
                .collect()
        };
        found.push((text(at + 2, path_len), text(at + 2 + path_len, size)));
        at += 2 + path_len + size;
    }
    found
}

/// Evaluates, then syncs: the operations done, and the events said.
fn step(session: &mut Session) -> Vec<String> {
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    session.sync().iter().map(Event::to_string).collect()
}

fn pairs(found: &[(&str, &str)]) -> Vec<(String, String)> {
    found
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect()
}

#[test]
fn a_files_input_holds_the_directory_and_follows_it() {
    let dir = scratch("follows");
    let mut session = open(&dir);
    // In path order, nested ones by their path, a hidden one left out.
    assert_eq!(
        listed(&mut session),
        pairs(&[("lib/warm.ent", "w1"), ("main.ent", "m")])
    );
    // Changed, added and removed outside the session, it follows.
    std::fs::write(dir.join("project/lib/warm.ent"), "w22").unwrap();
    std::fs::write(dir.join("project/notes.md"), "n").unwrap();
    std::fs::remove_file(dir.join("project/main.ent")).unwrap();
    session.sync();
    assert_eq!(
        listed(&mut session),
        pairs(&[("lib/warm.ent", "w22"), ("notes.md", "n")])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn operations_write_remove_and_rename_where_allowed() {
    let dir = scratch("operations");
    let mut session = open(&dir);
    session.allow_writes(&dir.join("project")).unwrap();
    // A write makes the file and the directories it needs.
    ask(&dir, &[(WRITE, "new/a.txt", b"hi")]);
    let said = step(&mut session);
    assert_eq!(
        std::fs::read(dir.join("project/new/a.txt")).unwrap(),
        b"hi",
        "{said:?}"
    );
    // Removing needs its own allowance, and says how to give it.
    ask(&dir, &[(REMOVE, "new/a.txt", b"")]);
    let said = step(&mut session);
    assert!(
        said.iter().any(|line| line.contains("--allow file.delete")),
        "{said:?}"
    );
    assert!(dir.join("project/new/a.txt").exists());
    // Allowed, the file goes, and the directory it left empty.
    session.allow_deletes(&dir.join("project")).unwrap();
    ask(
        &dir,
        &[
            (REMOVE, "new/a.txt", b""),
            (RENAME, "main.ent", b"lib/main.ent"),
        ],
    );
    let said = step(&mut session);
    assert!(!dir.join("project/new").exists(), "{said:?}");
    assert!(!dir.join("project/main.ent").exists(), "{said:?}");
    assert_eq!(
        std::fs::read(dir.join("project/lib/main.ent")).unwrap(),
        b"m"
    );
    assert!(said.iter().any(|line| line.contains("renamed")), "{said:?}");
    // A rename onto a file that is there is refused; the file stays.
    ask(&dir, &[(RENAME, "lib/main.ent", b"lib/warm.ent")]);
    let said = step(&mut session);
    assert!(
        said.iter().any(|line| line.contains("is there already")),
        "{said:?}"
    );
    assert_eq!(
        std::fs::read(dir.join("project/lib/warm.ent")).unwrap(),
        b"w1"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_list_is_done_once_and_a_malformed_one_not_at_all() {
    let dir = scratch("once");
    let mut session = open(&dir);
    session.allow_writes(&dir.join("project")).unwrap();
    ask(&dir, &[(WRITE, "main.ent", b"mine")]);
    step(&mut session);
    assert_eq!(
        std::fs::read(dir.join("project/main.ent")).unwrap(),
        b"mine"
    );
    // Changed outside, the file is not written again by the same list.
    std::fs::write(dir.join("project/main.ent"), "theirs").unwrap();
    step(&mut session);
    step(&mut session);
    assert_eq!(
        std::fs::read(dir.join("project/main.ent")).unwrap(),
        b"theirs"
    );
    // A path out of the directory refuses the whole list.
    ask(
        &dir,
        &[(WRITE, "main.ent", b"x"), (WRITE, "../out.txt", b"x")],
    );
    let said = step(&mut session);
    assert!(
        said.iter().any(|line| line.contains("not a path under")),
        "{said:?}"
    );
    assert_eq!(
        std::fs::read(dir.join("project/main.ent")).unwrap(),
        b"theirs"
    );
    assert!(!dir.join("out.txt").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_directorys_files_are_words() {
    let dir = scratch("typed");
    std::fs::write(
        dir.join("scene.esn"),
        "input disk: Rows<Float> = files \"project/\";\noutput result <- disk;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let said: Vec<String> = session.sync().iter().map(Event::to_string).collect();
    assert!(
        said.iter().any(|line| line.contains("`Rows<Int>`")),
        "{said:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
