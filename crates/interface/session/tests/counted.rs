//! A node placed as many times as a row says: `times = value(node.port,
//! row);`. The count is the row's as the last frame left it: none before
//! a frame has made it, and a frame behind the row from then on; the
//! document is lowered again only when the count changes.

use std::path::PathBuf;

use entelechy_kernel::Context;
use entelechy_session::Session;

/// `grown` counts the frames up to `most`; `step` adds one to what it is
/// given, and is placed as many times as `grown` said.
const KERNELS: &str = "kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }
kernel t.grown@1 { requires { field.read; field.write; } input previous: Rows<Int>; param most: Int = 3; output result: Rows<Int>; compute { return min(at(previous, 0) + 1, most); } }
kernel t.step@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 1; } }
kernel t.told@1 { requires { field.write; time.read; } output result: Rows<Int>; compute { return min(int(time() * 10.0 + 0.5), 2); } }
kernel t.half@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.5; } }
kernel t.less@1 { requires { field.write; } output result: Rows<Int>; compute { return -2; } }
";

const SCENE: &str = "use \"k.ent\";
node zero: t.zero@1 { domain = 1; }
node grown: t.grown@1 { previous <~ grown.result from zero.result; domain = 1; }
node step: t.step@1 { times = value(grown.result); carry previous <- result; previous <- zero.result; domain = 1; }
output result <- step.result;
";

fn scratch(name: &str, scene: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-counted-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.ent"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.esn"), scene).unwrap();
    dir
}

/// A frame, then a sync: what the scene showed, how many nodes it held as
/// the frame ran, and what the sync said.
fn frame(session: &mut Session, at: u32) -> (i32, usize, Vec<String>) {
    let context = Context {
        time: f64::from(at) * 0.1,
        ..Context::default()
    };
    let report = session.evaluate(&context).unwrap();
    let shown = session.field(report.output).unwrap().values()[0] as i32;
    let nodes = session.graph().unwrap().nodes.len();
    let said = session
        .sync()
        .iter()
        .map(|event| format!("{event:?}"))
        .collect();
    (shown, nodes, said)
}

#[test]
fn a_node_is_placed_as_many_times_as_its_row_said_when_the_last_frame_ended() {
    let dir = scratch("grows", SCENE);
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    // No frame has made the row: `step` is placed no times, and its
    // carried output is what it was first bound to.
    assert_eq!(session.graph().unwrap().nodes.len(), 2);
    let mut seen = Vec::new();
    for at in 0..6 {
        let (shown, nodes, _) = frame(&mut session, at);
        seen.push((shown, nodes));
    }
    // Each frame shows the chain as long as `grown` was a frame before:
    // 0, then 1, 2 and 3, where it stays; the nodes are `zero`, `grown`
    // and the chain.
    assert_eq!(
        seen,
        [(0, 2), (1, 3), (2, 4), (3, 5), (3, 5), (3, 5)],
        "{seen:?}"
    );
    // A count that stays lowers nothing again.
    let (_, _, said) = frame(&mut session, 6);
    assert!(said.is_empty(), "{said:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn frames_follow_the_row_with_no_sync_between_them() {
    // A run of frames nobody syncs between, as a replay's `frame 0 to 1`
    // is: the session lowers the scene again itself before the frame a
    // new count runs on.
    let dir = scratch("unsynced", SCENE);
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    let seen: Vec<i32> = (0..5)
        .map(|at| {
            let context = Context {
                time: f64::from(at) * 0.1,
                ..Context::default()
            };
            let report = session.evaluate(&context).unwrap();
            session.field(report.output).unwrap().values()[0] as i32
        })
        .collect();
    assert_eq!(seen, [0, 1, 2, 3, 3], "{seen:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_row_nothing_else_reads_is_still_made_and_followed() {
    // `told` holds no state and feeds nothing: its row is made because
    // `step` is placed by it.
    let scene = SCENE.replace(
        "node grown: t.grown@1 { previous <~ grown.result from zero.result; domain = 1; }",
        "node grown: t.told@1 { domain = 1; }",
    );
    let dir = scratch("unread", &scene);
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    let seen: Vec<i32> = (0..5).map(|at| frame(&mut session, at).0).collect();
    // The row is 0, 1, 2, 2, 2 as the frames pass; the chain a frame behind.
    assert_eq!(seen, [0, 0, 1, 2, 2], "{seen:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_row_that_is_no_count_is_said_and_the_node_stays_as_it_is_placed() {
    for (kernel, why) in [
        ("t.half@1", "a count is an `Int`"),
        ("t.less@1", "it holds -2, less than 0"),
    ] {
        let scene = SCENE
            .replace(
                "node grown: t.grown@1 { previous <~ grown.result from zero.result; domain = 1; }",
                &format!("node grown: {kernel} {{ domain = 1; }}"),
            )
            .replace(
                "output result <- step.result;",
                "output result <- step.result;\noutput count <- grown.result;",
            );
        let dir = scratch("refused", &scene);
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
        let events = session.sync();
        assert!(session.graph().is_some(), "{events:?}");
        let (shown, nodes, said) = frame(&mut session, 0);
        assert_eq!((shown, nodes), (0, 2));
        assert!(
            said.iter().any(|event| event.contains(why)),
            "{kernel}: {said:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_row_asked_past_its_end_or_of_no_node_is_refused_where_it_is_written() {
    let refused = |scene: &str| {
        let dir = scratch("unwritten", scene);
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
        let said = format!("{:?}", session.sync());
        let _ = std::fs::remove_dir_all(&dir);
        said
    };
    let said = refused(&SCENE.replace("value(grown.result)", "value(grew.result)"));
    assert!(
        said.contains("`grew` is not another node of this scene"),
        "{said}"
    );
    let said = refused(&SCENE.replace("value(grown.result)", "value(step.result)"));
    assert!(
        said.contains("`step` is not another node of this scene"),
        "{said}"
    );
    let said = refused(&SCENE.replace("value(grown.result)", "value(grown.result, -1)"));
    assert!(said.contains("`times`'s row is a whole number"), "{said}");
}
