//! A placed graph's outputs are named as the placing graph names them,
//! `node.output`, wherever a node's port is: in a replay's expectations,
//! in a save, in a read, in a pin.

use std::path::PathBuf;

use orla_kernel::Context;
use orla_session::Session;

const KERNELS: &str = "kernel t.start@1 { requires { field.write; } output result: Rows<Int>; compute { return element() + 1; } }
kernel t.twice@1 { requires { field.read; field.write; } input counts: Rows<Int>; output result: Rows<Int>; compute { return sample(counts) * 2; } }
";

const PRESET: &str = "use \"k.orla\";
input counts: Rows<Int>;
node once: t.twice@1 { counts <- counts; }
node again: t.twice@1 { counts <- once.result; }
output quad <- again.result;
";

const SCENE: &str = "use \"k.orla\";
node start: t.start@1 { domain = 3; }
node q: graph \"quad.osn\" { counts <- start.result; }
output shown <- q.quad;
";

const REPLAY: &str = "scene scene.osn
frame 0
expect q.quad[0] = 4
expect q.quad[2] = 12
expect count q.quad = 3
";

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-placed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("quad.osn"), PRESET).unwrap();
    std::fs::write(dir.join("scene.osn"), SCENE).unwrap();
    std::fs::write(dir.join("t.replay"), REPLAY).unwrap();
    dir
}

#[test]
fn a_placed_graphs_output_is_named_as_its_placing_graph_names_it() {
    let dir = scratch();
    let replay = orla_session::replay::parse(REPLAY, &dir).unwrap();
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let outcome = orla_session::replay::run(&replay, runtime).unwrap();
    assert_eq!(
        (outcome.checks, outcome.failures.len()),
        (3, 0),
        "{outcome:?}"
    );

    // Pinned before the scene is read, the name still holds once it is.
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.pin(orla_graph::NodeId("q".into()));
    session.sync();
    session.allow_writes(&dir).unwrap();
    session.evaluate(&Context::default()).unwrap();
    assert!(
        session
            .cached(&orla_graph::NodeId("q".into()), "quad")
            .is_some()
    );
    assert_eq!(
        session.producer(&orla_graph::NodeId("q".into()), "quad").0,
        orla_graph::NodeId("q.again".into())
    );
    session
        .save(
            &orla_graph::NodeId("q".into()),
            "quad",
            &dir.join("quad.field"),
        )
        .unwrap();
    assert!(dir.join("quad.field").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
