//! Where replacements prepare off the frames, so are changed library
//! documents read and their Kernels compiled: a sync hands them to
//! another thread and returns, the running Kernels keep the frames, and a
//! later sync publishes them, or says why not. The scene is read once
//! they land, so it never names a Kernel they are about to publish.

use orla_kernel::{Context, Runtime};
use orla_session::{Event, Session};

fn kernel(id: &str, value: &str) -> String {
    format!(
        "kernel {id} {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value}; }} }}\n"
    )
}

fn value(session: &mut Session, node: &str) -> f32 {
    session.evaluate(&Context::default()).unwrap();
    let handle = session
        .cached(&orla_graph::NodeId(node.into()), "result")
        .unwrap();
    f32::from_bits(session.field(handle).unwrap().values()[0])
}

#[test]
fn a_changed_library_compiles_off_the_frame_and_publishes_at_a_later_sync() {
    let dir = std::env::temp_dir().join(format!("orla-compile-ahead-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.orla"), kernel("t.a@1", "1.0")).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"lib.orla\";\nnode a: t.a@1 { domain = 4; }\noutput result <- a.result;\n",
    )
    .unwrap();
    let runtime = Runtime::with_backends(1 << 20, Box::new(orla_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    session.sync();
    assert_eq!(value(&mut session, "a"), 1.0);
    session.set_background(true);

    // The library changes one Kernel and adds another, which the scene
    // starts to use.
    std::fs::write(
        dir.join("lib.orla"),
        kernel("t.a@1", "2.0") + &kernel("t.b@1", "5.0"),
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"lib.orla\";\nnode a: t.a@1 { domain = 4; }\nnode b: t.b@1 { domain = 4; }\noutput result <- a.result;\noutput other <- b.result;\n",
    )
    .unwrap();
    let events = session.sync();
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::Published { .. } | Event::SceneRejected(_) | Event::Rejected { .. }
        )),
        "handed off, nothing said yet: {events:?}"
    );
    assert_eq!(value(&mut session, "a"), 1.0, "the running Kernel runs");

    let events = session.sync_settled();
    assert!(
        !events.iter().any(|e| matches!(e, Event::SceneRejected(_))),
        "{events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneLoaded)),
        "{events:?}"
    );
    assert_eq!(value(&mut session, "a"), 2.0);
    assert_eq!(value(&mut session, "b"), 5.0);

    // A fault is said when it lands, and the running Kernel stays.
    std::fs::write(dir.join("lib.orla"), "kernel t.a@1 { nonsense }\n").unwrap();
    let events = session.sync_settled();
    assert!(
        events.iter().any(|e| matches!(e, Event::Rejected { .. })),
        "{events:?}"
    );
    assert_eq!(value(&mut session, "a"), 2.0);
    let _ = std::fs::remove_dir_all(&dir);
}
