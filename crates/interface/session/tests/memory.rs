//! A scene holding more than the Host's budget lets go of cached nodes to
//! make room and says how many in its report; one within its budget lets
//! go of none. Outputs made once, which nothing reads again until
//! something changes, are let go cold after their frame: passed by unrun
//! while unchanged, made again for a reader that runs, or to be read.

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

fn opened(budget_words: usize, name: &str, text: &str) -> (Session, PathBuf) {
    let path = scene(name);
    std::fs::write(&path, text).unwrap();
    let mut session = Session::open(&path, None, budget_words);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.set_fusion(false);
    (session, path)
}

fn first_word(session: &Session) -> f32 {
    let handle = session.output(None).unwrap();
    f32::from_bits(session.runtime().host().element(handle, 0).unwrap()[0])
}

fn names(nodes: &[entelechy_graph::NodeId]) -> Vec<&str> {
    nodes.iter().map(|node| node.0.as_str()).collect()
}

#[test]
fn outputs_made_once_are_let_go_cold_and_passed_by() {
    let (mut session, path) = opened(1 << 24, "once", SCENE);
    let context = entelechy_kernel::Context::default();
    let first = session.evaluate(&context).unwrap();
    assert_eq!(first.executed.len(), 4);
    // The chain's three outputs of 400 000 words are made once: let go,
    // the output kept.
    let mut cold = session.cold_nodes();
    cold.sort();
    assert_eq!(names(&cold), ["a", "b", "c"]);
    assert!(
        session.runtime().host().used_words() < 400_000,
        "{} words held",
        session.runtime().host().used_words()
    );
    // Unchanged, nothing runs again.
    let second = session.evaluate(&context).unwrap();
    assert!(second.executed.is_empty(), "{:?}", second.executed);
    assert_eq!(first_word(&session), 2.0);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn a_reader_that_runs_has_what_it_reads_made_again() {
    let (mut session, path) = opened(1 << 24, "reader", SCENE);
    let context = entelechy_kernel::Context::default();
    session.evaluate(&context).unwrap();
    assert_eq!(session.cold_nodes().len(), 3);
    // The reader's Kernel changes: the chain it reads is made again.
    std::fs::write(&path, SCENE.replace("return at(a, 0);", "return at(a, 5);")).unwrap();
    session.sync();
    let after = session.evaluate(&context).unwrap();
    assert_eq!(names(&after.executed), ["a", "b", "c", "sum"]);
    assert_eq!(first_word(&session), 7.0);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn a_reader_of_the_clock_keeps_what_it_reads() {
    // `sum` reads the clock and runs each frame: `c`, which it reads,
    // stays; what only `c`'s makers read is let go.
    let text = SCENE
        .replace(
            "kernel t.sum@1 { requires { field.read; field.write; }",
            "kernel t.sum@1 { requires { field.read; field.write; time.read; }",
        )
        .replace("return at(a, 0);", "return at(a, 0) + time() * 0.0;");
    let (mut session, path) = opened(1 << 24, "clock", &text);
    let mut context = entelechy_kernel::Context::default();
    session.evaluate(&context).unwrap();
    let mut cold = session.cold_nodes();
    cold.sort();
    assert_eq!(names(&cold), ["a", "b"]);
    context.time = 1.0;
    let next = session.evaluate(&context).unwrap();
    assert_eq!(names(&next.executed), ["sum"]);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn past_its_budget_a_scene_made_once_is_not_made_again_each_frame() {
    let (mut session, path) = opened(1 << 20, "tight", SCENE);
    let context = entelechy_kernel::Context::default();
    let first = session.evaluate(&context).unwrap();
    assert!(first.let_go > 0, "none let go");
    // What was let go to make room is passed by while unchanged.
    for _ in 0..3 {
        let next = session.evaluate(&context).unwrap();
        assert!(next.executed.is_empty(), "{:?}", next.executed);
    }
    assert_eq!(first_word(&session), 2.0);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn an_output_let_go_cold_is_made_again_to_be_read() {
    let (mut session, path) = opened(1 << 24, "read", SCENE);
    let context = entelechy_kernel::Context::default();
    session.evaluate(&context).unwrap();
    let b = entelechy_graph::NodeId::from("b");
    assert!(session.cached(&b, "result").is_none());
    session.warm(&b, "result").unwrap();
    let handle = session.cached(&b, "result").expect("made again");
    let host = session.runtime().host();
    assert_eq!(f32::from_bits(host.element(handle, 9).unwrap()[0]), 10.0);
    // Reading commits nothing: the next frame runs nothing.
    let next = session.evaluate(&context).unwrap();
    assert!(next.executed.is_empty(), "{:?}", next.executed);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn a_node_that_may_sleep_keeps_what_it_gave_and_sleeps_on_it() {
    // `c` reads the clock, so its outputs are new frame after frame: such
    // outputs are let go once read, but not one that may sleep. Asleep on
    // a row of 0, it is not run again, nor what only it reads.
    let text = "
kernel t.tick@1 { requires { field.write; time.read; } output result: Rows<Float>; compute { return time(); } }
kernel t.next@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { return sample(a) + 1.0; } }
kernel t.gate@1 { requires { field.write; } param on: Int = 1; output result: Rows<Int>; compute { return on; } }
param on: Int = 1;
node gate: t.gate@1 { on = on; domain = 1; }
node tick: t.tick@1 { domain = 4; }
node c: t.next@1 { a <- tick.result; awake = value(gate.result); }
node d: t.next@1 { a <- c.result; }
output result <- d.result;
";
    let (mut session, path) = opened(1 << 24, "asleep", text);
    let mut context = entelechy_kernel::Context::default();
    for time in 0..12 {
        context.time = f64::from(time);
        session.evaluate(&context).unwrap();
    }
    assert!(!names(&session.cold_nodes()).contains(&"c"));
    session
        .set_param("on", entelechy_abi::Value::Int(0))
        .unwrap();
    // The row the last frame left turns 0 on the frame after this one.
    context.time = 12.0;
    session.evaluate(&context).unwrap();
    for time in 13..16 {
        context.time = f64::from(time);
        let next = session.evaluate(&context).unwrap();
        assert!(next.executed.is_empty(), "{:?}", names(&next.executed));
        assert_eq!(first_word(&session), 12.0 + 2.0);
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}
