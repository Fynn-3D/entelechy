//! A frame runs what something reads: an output, a state, a pin, and what
//! they read in turn. A node nothing reads computes nothing anyone sees,
//! so it does not run, and a check says so once, at the top of its chain.

use entelechy_kernel::{Context, Runtime};
use entelechy_session::Session;

const LIB: &str = "kernel t.fill@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()); } }
kernel t.twice@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { return sample(a) * 2.0; } }
kernel t.add@1 { requires { field.read; field.write; } input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) + 1.0; } }
kernel t.far@1 { requires { field.read; field.write; } input a: Rows<Float>; output result: Rows<Float>; compute { let zero = int(sample(a)) - int(sample(a)); return float(7 / zero); } }
";

const SCENE: &str = "use \"lib.ent\";
node a: t.fill@1 { domain = 3; }
node shown: t.twice@1 { a <- a.result; }
node zero: t.fill@1 { domain = 3; }
node tally: t.add@1 { previous <~ tally.result from zero.result; }
node above: t.twice@1 { a <- a.result; }
node lonely: t.far@1 { a <- above.result; domain = 3; }
output result <- shown.result;
";

fn opened(name: &str) -> (Session, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("entelechy-unread-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.ent"), LIB).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    let runtime = Runtime::with_backends(1 << 20, Box::new(entelechy_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    (session, dir)
}

#[test]
fn a_frame_runs_what_an_output_a_state_or_a_pin_reads_and_nothing_else() {
    let (mut session, dir) = opened("frame");
    // `lonely` divides by zero: run, it fails.
    let report = session
        .evaluate(&Context::default())
        .expect("nothing that fails is read");
    let ran: Vec<String> = report.executed.iter().map(ToString::to_string).collect();
    for node in ["a", "shown", "zero", "tally"] {
        assert!(ran.contains(&node.to_string()), "{node} is read: {ran:?}");
    }
    for node in ["above", "lonely"] {
        assert!(
            !ran.contains(&node.to_string()),
            "{node} is read by nothing: {ran:?}"
        );
    }
    // State advances though nothing reads it.
    session.evaluate(&Context::default()).unwrap();
    let why = session.why_no_value(&entelechy_graph::NodeId("lonely".into()), "result");
    assert!(why.contains("nothing reads node `lonely`"), "{why}");
    // Pinned, it is read, so it runs, and fails the frame.
    session.pin(entelechy_graph::NodeId("lonely".into()));
    let failed = session.evaluate(&Context::default());
    assert!(failed.is_err(), "a pinned node runs: {failed:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_check_says_a_node_nothing_reads_once_at_the_top_of_its_chain() {
    let (_, dir) = opened("check");
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let notes: Vec<&String> = checked
        .lines
        .iter()
        .filter(|(_, line)| line.contains("nothing reads"))
        .map(|(_, line)| line)
        .collect();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("node `lonely`"), "{notes:?}");
    assert!(
        checked
            .lines
            .iter()
            .all(|(problem, line)| !line.contains("nothing reads") || !*problem),
        "a note, not a problem"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
