//! A `font` node gives the glyphs of the characters its `codes` held when
//! last evaluated: a row a code, in order, a code the font lacks `-1`;
//! read again only when the codes or the font change.

use std::path::PathBuf;

use entelechy_kernel::{Context, Value};
use entelechy_session::Session;

const SCENE: &str = "
struct Letterform { code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2; }
kernel t.codes@1 { requires { field.write; } param which: Int = 0; output result: Rows<Int>;
    compute { return if which == 0 { [111, 20013, 73][element()] } else { [73, 73, 0][element()] }; } }
kernel t.copy@1 { requires { field.read; field.write; } input glyphs: Rows<Letterform>; output result: Rows<Letterform>; compute { return sample(glyphs); } }
struct Measures { units: Int; ascent: Float; descent: Float; gap: Float; }
kernel t.measured@1 { requires { field.read; field.write; } input metrics: Rows<Measures>; output result: Rows<Measures>; compute { return sample(metrics); } }
kernel t.ends@1 { requires { field.read; field.write; } input outlines: Rows<Vec4>; output result: Rows<Vec4>; compute { return sample(outlines); } }
param which: Int = 0;
node codes: t.codes@1 { which = which; domain = 3; }
node g: font \"fonts/sample.ttf\" { codes <- codes.result; }
node copy: t.copy@1 { glyphs <- g.glyphs; }
node ends: t.ends@1 { outlines <- g.outlines; }
node measured: t.measured@1 { metrics <- g.metrics; }
output result <- copy.result;
output outlines <- ends.result;
output metrics <- measured.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-font-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("fonts")).unwrap();
    std::fs::write(
        dir.join("fonts/sample.ttf"),
        entelechy_import::font::sample::latin(),
    )
    .unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn open(dir: &std::path::Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// The codes of the glyph rows the scene's `result` holds.
fn codes(session: &mut Session) -> Vec<i32> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let words = session.field(handle).unwrap().values().to_vec();
    words.chunks(8).map(|row| row[0] as i32).collect()
}

#[test]
fn the_glyphs_follow_the_codes_a_frame_later() {
    let dir = scratch("follow");
    let mut session = open(&dir);
    // Before any codes are evaluated, the missing glyph alone.
    assert_eq!(codes(&mut session), vec![-1]);
    // Then the three asked, the one the font lacks -1.
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    let metrics = session.output(Some("metrics")).unwrap();
    assert_eq!(session.field(metrics).unwrap().values()[0], 1000);
    let outlines = session.output(Some("outlines")).unwrap();
    assert!(session.field(outlines).unwrap().values().len() > 8);
    // Other codes: other glyphs, a frame later; `I` drawn once for both.
    session.set_param("which", Value::Int(1)).unwrap();
    codes(&mut session);
    assert_eq!(codes(&mut session), vec![73, 73, -1]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_same_codes_leave_the_glyphs_as_they_were() {
    let dir = scratch("same");
    let mut session = open(&dir);
    codes(&mut session);
    codes(&mut session);
    let report = session.evaluate(&Context::default()).unwrap();
    assert!(
        report.reused.iter().any(|node| node.0 == "copy"),
        "{:?}",
        report.executed
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_font_node_names_its_one_input() {
    let dir = scratch("input");
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("{ codes <- codes.result; }", "{ chars <- codes.result; }"),
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(said.iter().any(|line| line.contains("`codes`")), "{said:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
