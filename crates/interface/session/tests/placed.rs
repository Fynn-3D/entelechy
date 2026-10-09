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

/// A graph whose Kernel is written in place, placed twice and placed
/// `times` over, defines that Kernel once.
#[test]
fn a_graph_placed_more_than_once_defines_its_inline_kernel_once() {
    let dir = std::env::temp_dir().join(format!("orla-placed-twice-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("inc.osn"),
        "input value: Rows<Float>;
node plus: kernel {
    requires { field.read; field.write; }
    input v: Rows<Float> <- value;
    output result: Rows<Float>;
    compute { return sample(v) + 1.0; }
}
output result <- plus.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "node seed: Rows<Float> { domain = 2; value = 1.0; }
node a: graph \"inc.osn\" { value <- seed.result; }
node b: graph \"inc.osn\" { value <- a.result; }
node c: graph \"inc.osn\" { times = 2; carry value <- result; value <- b.result; }
output result <- c.result;
",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&Context::default()).unwrap();
    let output = session.output(None).unwrap();
    let words = session.field(output).unwrap().values().into_owned();
    assert_eq!(
        f32::from_bits(words[0]),
        5.0,
        "four placements, each adding one"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A graph is the file it is in: placed by the scene and by a graph the
/// scene places, its path written two ways, its Kernels (an inline one
/// and a filled Field) are one set, published once, and each placement
/// runs.
#[test]
fn a_graph_placed_from_two_levels_is_one_document() {
    let dir = std::env::temp_dir().join(format!("orla-placed-levels-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("g")).unwrap();
    std::fs::write(
        dir.join("inc.osn"),
        "input value: Rows<Float>;
node one: Rows<Float> { domain = 2; value = 1.0; }
node plus: kernel {
    requires { field.read; field.write; }
    input v: Rows<Float> <- value;
    input w: Rows<Float> <- one.result;
    output result: Rows<Float>;
    compute { return sample(v) + sample(w); }
}
output result <- plus.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("g/wrap.osn"),
        "input value: Rows<Float>;
node inner: graph \"../inc.osn\" { value <- value; }
output result <- inner.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "node seed: Rows<Float> { domain = 2; value = 1.0; }
node a: graph \"inc.osn\" { value <- seed.result; }
node b: graph \"g/wrap.osn\" { value <- a.result; }
output result <- b.result;
",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&Context::default()).unwrap();
    let output = session.output(None).unwrap();
    let words = session.field(output).unwrap().values().into_owned();
    assert_eq!(
        f32::from_bits(words[0]),
        3.0,
        "two placements, each adding one"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A node placed `times = N` is named, wherever a node's port is, as the
/// scene names it: `name.port` is its last time's, a placed graph's
/// output through the node inside it.
#[test]
fn a_repeated_nodes_name_is_its_last_time() {
    let dir = std::env::temp_dir().join(format!("orla-repeated-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("quad.osn"), PRESET).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"k.orla\";
node start: t.start@1 { domain = 3; }
node grow: t.twice@1 { times = 3; carry counts <- result; counts <- start.result; }
node r: graph \"quad.osn\" { times = 2; carry counts <- quad; counts <- start.result; }
output shown <- grow.result;
",
    )
    .unwrap();
    let replay = "scene scene.osn
frame 0
expect grow.result[2] = 24
expect r.quad[0] = 16
expect count grow.result = 3
";
    let parsed = orla_session::replay::parse(replay, &dir).unwrap();
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let outcome = orla_session::replay::run(&parsed, runtime).unwrap();
    assert_eq!(
        (outcome.checks, outcome.failures.len()),
        (3, 0),
        "{outcome:?}"
    );

    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.pin(orla_graph::NodeId("r".into()));
    session.sync();
    session.allow_writes(&dir).unwrap();
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        session
            .producer(&orla_graph::NodeId("grow".into()), "result")
            .0,
        orla_graph::NodeId("grow/3".into())
    );
    assert_eq!(
        session.producer(&orla_graph::NodeId("r".into()), "quad").0,
        orla_graph::NodeId("r/2.again".into())
    );
    assert!(
        session
            .cached(&orla_graph::NodeId("r".into()), "quad")
            .is_some()
    );
    session
        .save(
            &orla_graph::NodeId("grow".into()),
            "result",
            &dir.join("grow.field"),
        )
        .unwrap();
    assert!(dir.join("grow.field").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
