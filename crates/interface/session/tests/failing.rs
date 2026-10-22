//! A frame that fails keeps nothing: a read of a value it kept from being
//! made is told the failure, not only that the node has not run.

use entelechy_kernel::{Context, Runtime};
use entelechy_session::Session;

const LIB: &str = "kernel t.fill@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()); } }
kernel t.far@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { let zero = int(sample(a)) - int(sample(a)); return float(7 / zero); } }
kernel t.twice@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { return sample(a) * 2.0; } }
";

const SCENE: &str = "use \"lib.ent\";
node a: t.fill@1 { domain = 3; }
node bad: t.far@1 { a <- a.result; }
node shown: t.twice@1 { a <- bad.result; }
output result <- shown.result;
";

#[test]
fn a_read_after_a_failed_frame_is_told_why_the_frame_failed() {
    let dir = std::env::temp_dir().join(format!("entelechy-failing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.ent"), LIB).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    let runtime = Runtime::with_backends(1 << 20, Box::new(entelechy_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    let shown = entelechy_graph::NodeId("shown".into());
    let before = session.why_no_value(&shown, "result");
    assert!(
        before.contains("has not run since it was placed; evaluate"),
        "{before}"
    );
    let failed = session.evaluate(&Context::default());
    let error = failed.expect_err("`bad` divides by zero").to_string();
    let why = session.why_no_value(&shown, "result");
    assert!(why.contains("the last evaluation failed"), "{why}");
    assert!(why.contains(&error), "the failure itself is said: {why}");
    assert!(why.contains("division by zero"), "and why it failed: {why}");
    // `bad` runs fused into `shown`, which the failure names: the read
    // names `bad` too, where the fault is.
    assert!(why.contains("`bad`"), "the node fused in is named: {why}");
    let _ = std::fs::remove_dir_all(&dir);
}
