//! A scene holding more than the Host's budget lets go of cached nodes to
//! make room and says how many in its report; one within its budget lets
//! go of none.

use std::path::PathBuf;

use entelechy_session::Session;

const SCENE: &str = "
kernel t.big@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()); } }
kernel t.next@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { return sample(a) + 1.0; } }
kernel t.sum@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { return at(a, 0); } }
node a: t.big@1 { domain = 400000; }
node b: t.next@1 { a <- a.result; }
node c: t.next@1 { a <- b.result; }
node sum: t.sum@1 { a <- c.result; domain = 1; }
output result <- sum.result;
";

fn scene(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-memory-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir.join("scene.esn")
}

fn let_go(budget_words: usize, name: &str) -> (usize, usize) {
    let path = scene(name);
    let mut session = Session::open(&path, None, budget_words);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.set_fusion(false);
    let context = entelechy_kernel::Context::default();
    let first = session.evaluate(&context).unwrap();
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    (first.let_go, first.executed.len())
}

#[test]
fn a_scene_past_its_budget_says_what_it_let_go() {
    // A chain of three outputs of 400 000 words: within four million
    // words every one stays.
    assert_eq!(let_go(1 << 22, "wide"), (0, 4));
    // Within a million the third has no room until the first, which no
    // later run reads, is let go; the report says so.
    let (gone, ran) = let_go(1 << 20, "narrow");
    assert_eq!(ran, 4);
    assert!(gone > 0, "none let go");
}
