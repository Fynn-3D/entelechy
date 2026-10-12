//! An `outline` input holds what the documents of a directory define, as
//! text, a line a definition, the file relative to the directory; read
//! again when one changes.

use std::path::{Path, PathBuf};

use entelechy_kernel::Context;
use entelechy_session::{Event, Session};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-outline-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src/lib")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.copy@1 { requires { field.read; field.write; } input text: Rows<Int>; output result: Rows<Int>; compute { return sample(text); } }
input defined: Rows<Int> = outline \"src/\";
node copy: t.copy@1 { text <- defined; }
output result <- copy.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "struct Glow { at: Vec2; hue: Float; }\n\nfn warm(x: Float) -> Vec3 {\n    return vec3(x, 0.0, 0.0);\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/ramp.esn"),
        "use \"lib/warm.ent\";\nkernel q.ramp@1 { requires { field.write; } param k: Float = 1.0; output result: Image; compute { return vec4(warm(k), 1.0); } }\nnode ramp: q.ramp@1 { domain = vec2(4, 4); }\noutput shown <- ramp.result;\n",
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
fn an_outline_input_holds_what_the_documents_define_and_follows_them() {
    let dir = scratch("follows");
    let mut session = open(&dir);
    assert_eq!(
        text(&mut session),
        "lib/warm.ent:1:8: struct Glow: { at: Vec2; hue: Float }
lib/warm.ent:3:4: fn warm: (x: Float) -> Vec3
ramp.esn:2:8: kernel q.ramp@1: param k: Float; out result: Image
ramp.esn:3:6: node ramp: q.ramp@1
ramp.esn:4:8: output shown
"
    );
    // A new function is found as it is written; a document that does not
    // parse defines nothing.
    std::fs::write(
        dir.join("src/lib/warm.ent"),
        "fn warm(x: Float) -> Vec3 { return vec3(x, 0.0, 0.0); }\nfn cold(x: Float) -> Vec3 { return vec3(0.0, 0.0, x); }\n",
    )
    .unwrap();
    std::fs::write(dir.join("src/ramp.esn"), "node oops\n").unwrap();
    session.sync();
    assert_eq!(
        text(&mut session),
        "lib/warm.ent:1:4: fn warm: (x: Float) -> Vec3
lib/warm.ent:2:4: fn cold: (x: Float) -> Vec3
"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_an_outline_finds_is_text() {
    let dir = scratch("typed");
    std::fs::write(
        dir.join("scene.esn"),
        "input defined: Rows<Float> = outline \"src/\";\noutput result <- defined;\n",
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
