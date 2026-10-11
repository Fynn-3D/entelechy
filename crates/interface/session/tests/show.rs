//! A replay's `show` prints what a node's output holds, a struct field by
//! field, and checks nothing.

use std::path::{Path, PathBuf};

use entelechy_session::{Session, replay};

const SCENE: &str = "struct Spot { at: Vec2; hits: Int; }
kernel t.spots@1 {
    requires { field.write; }
    output result: Rows<Spot>;
    compute { let i = element(); return Spot { at: vec2(float(i), 0.5), hits: i * 2 }; }
}
node spots: t.spots@1 { domain = 3; }
output result <- spots.result;
";

fn scene(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-show-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn ran(dir: &Path, text: &str) -> Result<replay::Outcome, String> {
    let replay = replay::parse(text, dir).map_err(|e| e.to_string())?;
    let runtime = Session::runtime_for("cpu", entelechy_session::Limits::default()).unwrap();
    replay::run(&replay, runtime)
}

#[test]
fn show_prints_an_element_and_a_field_and_checks_nothing() {
    let dir = scene("prints");
    let outcome = ran(
        &dir,
        "scene scene.esn\nframe 0\nshow spots.result[2]\nshow spots.result[2].hits\nexpect spots.result[2].hits = 4\n",
    )
    .unwrap();
    assert_eq!(
        outcome.shown,
        [
            "line 3: spots.result[2] is { at: 2 0.5, hits: 4 }",
            "line 4: spots.result[2].hits is 4",
        ]
    );
    assert_eq!(outcome.checks, 1);
    assert!(outcome.failures.is_empty(), "{:?}", outcome.failures);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_before_a_frame_is_refused() {
    let dir = scene("early");
    let error = ran(&dir, "scene scene.esn\nshow spots.result[0]\n").unwrap_err();
    assert!(error.contains("line 2: show before any frame"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

const ARRAYS: &str = "struct Tally { counts: [Int; 3]; }
kernel t.tally@1 {
    requires { field.write; }
    output result: Rows<Tally>;
    compute { let i = element(); return Tally { counts: [each 3 { i * 10 + index() }] }; }
}
node tally: t.tally@1 { domain = 2; }
output result <- tally.result;
";

#[test]
fn an_array_element_is_a_field_named_by_its_index() {
    let dir = scene("arrays");
    std::fs::write(dir.join("scene.esn"), ARRAYS).unwrap();
    let outcome = ran(
        &dir,
        "scene scene.esn\nframe 0\nshow tally.result[1].counts\nexpect tally.result[1].counts.2 = 12\nexpect tally.result[1].counts.0 < 11\n",
    )
    .unwrap();
    assert_eq!(
        outcome.shown,
        ["line 3: tally.result[1].counts is { 0: 10, 1: 11, 2: 12 }"]
    );
    assert_eq!(outcome.checks, 2);
    assert!(outcome.failures.is_empty(), "{:?}", outcome.failures);
    // Written as an index in brackets, the refusal says how it is named.
    let error = ran(
        &dir,
        "scene scene.esn\nframe 0\nexpect tally.result[1].counts[2] = 12\n",
    )
    .unwrap_err();
    assert!(
        error.contains("an array's element is named by its index: `counts.2`"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
