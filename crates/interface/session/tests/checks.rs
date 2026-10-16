//! A `checks` input holds what the language finds wrong in the documents
//! of a directory, as text, a line a problem, the file relative to the
//! directory; the documents checked again when one changes.

use std::path::{Path, PathBuf};

use entelechy_kernel::Context;
use entelechy_session::Session;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-checks-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src/lib")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.copy@1 { requires { field.read; field.write; } input text: Rows<Int>; output result: Rows<Int>; compute { return sample(text); } }
input problems: Rows<Int> = checks \"src/\";
node copy: t.copy@1 { text <- problems; }
output result <- copy.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "fn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0);\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/ramp.ent"),
        "use \"lib/warm.ent\";\nkernel q.ramp@1 { requires { field.write; } output result: Image; compute { return vec4(warm(1.0), 1.0); } }\n",
    )
    .unwrap();
    dir
}

fn text(session: &mut Session) -> String {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    session
        .field(handle)
        .unwrap()
        .values()
        .iter()
        .map(|w| char::from_u32(*w).unwrap())
        .collect()
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

#[test]
fn a_checks_input_holds_the_problems_found_and_follows_the_documents() {
    let dir = scratch("follows");
    let mut session = open(&dir);
    // The fault is in the library, where it is written, once, however many
    // documents meet it.
    assert_eq!(
        text(&mut session),
        "lib/warm.ent:2:12: `vec3` takes 3 components, found 2 in (Float, Float)\n"
    );
    // Mended, nothing is found: a newline alone.
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "fn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0, 0.0);\n}\n",
    )
    .unwrap();
    session.sync();
    assert_eq!(text(&mut session), "\n");
    // A new document with a fault of its own is found as it appears.
    std::fs::write(dir.join("src/bad.ent"), "fn f() -> Int { return oops; }\n").unwrap();
    session.sync();
    let found = text(&mut session);
    assert!(found.starts_with("bad.ent:1:"), "{found}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_used_document_that_does_not_parse_is_said_by_its_own_path() {
    let dir = scratch("used");
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "fn warm(x: Float) -> Vec3 {\n    let y = x\n    return vec3(y, 0.0, 0.0);\n}\n",
    )
    .unwrap();
    let mut session = open(&dir);
    let found = text(&mut session);
    assert!(found.starts_with("lib/warm.ent:3:5: "), "{found}");
    assert_eq!(found.lines().count(), 1, "{found}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_a_check_finds_is_text() {
    let dir = scratch("typed");
    std::fs::write(
        dir.join("scene.esn"),
        "input problems: Rows<Float> = checks \"src/\";\noutput result <- problems;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    let said: Vec<String> = events.iter().map(|event| event.to_string()).collect();
    assert!(
        said.iter().any(|line| line.contains("`Rows<Int>`")),
        "{said:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// What `session` holds once a check made off the frames is taken in:
/// synced until the text is `want`, or what it was when time ran out.
fn text_once(session: &mut Session, want: &str) -> String {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        session.sync();
        let now = text(session);
        if now == want || std::time::Instant::now() > until {
            return now;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn checks_after_the_first_are_made_off_the_frames_while_the_session_prepares_in_the_background() {
    let dir = scratch("aside");
    let mut session = open(&dir);
    session.set_background(true);
    // The first check is taken in at once: the scene never reads it absent.
    assert_eq!(
        text(&mut session),
        "lib/warm.ent:2:12: `vec3` takes 3 components, found 2 in (Float, Float)\n"
    );
    // Mended, the check is made off the frames: the sync that sees the
    // change does not wait for it, and the text held is the last one until
    // a later sync takes the new one in.
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "fn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0, 0.0);\n}\n",
    )
    .unwrap();
    session.sync();
    assert_eq!(
        text(&mut session),
        "lib/warm.ent:2:12: `vec3` takes 3 components, found 2 in (Float, Float)\n"
    );
    assert_eq!(text_once(&mut session, "\n"), "\n");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fault_moved_down_its_document_is_said_where_it_now_is() {
    let dir = scratch("moved");
    let mut session = open(&dir);
    assert!(text(&mut session).starts_with("lib/warm.ent:2:12: "));
    // Two lines above the function: its body is as it was, its fault two
    // lines down.
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "// a note\n// another\nfn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0);\n}\n",
    )
    .unwrap();
    session.sync();
    assert!(text(&mut session).starts_with("lib/warm.ent:4:12: "));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_a_session_keeps_between_checks_finds_what_a_fresh_one_finds() {
    let dir = scratch("kept");
    std::fs::write(
        dir.join("src/glow.ent"),
        "use \"lib/warm.ent\";\nkernel q.glow@1 { requires { field.write; } output result: Image; compute { return vec4(warm(2.0), oops); } }\n",
    )
    .unwrap();
    let mut session = open(&dir);
    let edits: [(&str, &str); 4] = [
        (
            "src/lib/warm.ent",
            "fn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0, 0.0);\n}\n",
        ),
        (
            "src/glow.ent",
            "use \"lib/warm.ent\";\n\nkernel q.glow@1 { requires { field.write; } output result: Image; compute { return vec4(warm(2.0), 1.0); } }\n",
        ),
        (
            "src/lib/warm.ent",
            "// moved\nfn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0);\n}\n",
        ),
        (
            "src/ramp.ent",
            "use \"lib/warm.ent\";\nkernel q.ramp@1 { requires { field.write; } output result: Image; compute { return vec4(warm(1.0), nope); } }\n",
        ),
    ];
    for (file, written) in edits {
        // A stamp is a modification time and a length: a write lands in a
        // later moment.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join(file), written).unwrap();
        session.sync();
        let kept = text(&mut session);
        let fresh = text(&mut open(&dir));
        assert_eq!(kept, fresh, "after writing {file}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
