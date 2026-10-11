//! A Kernel is compiled again when what it reaches changes, and only
//! then: a function it never calls, added to or changed in a document it
//! uses, leaves it running as it was; the function it calls, changed,
//! publishes it again.

use entelechy_kernel::{Context, KernelId};
use entelechy_session::Session;

fn scratch() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-reach-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn library(used: &str, unused: &str) -> String {
    format!(
        "fn used() -> Float {{ return {used}; }}\nfn unused() -> Float {{ return {unused}; }}\n"
    )
}

fn generation(session: &Session) -> u64 {
    let id: KernelId = "t.k@1".parse().unwrap();
    session.runtime().kernel(&id).unwrap().generation().0
}

#[test]
fn a_kernel_is_compiled_again_only_when_what_it_reaches_changes() {
    let dir = scratch();
    std::fs::write(dir.join("lib.ent"), library("1.0", "2.0")).unwrap();
    std::fs::write(
        dir.join("k.ent"),
        "use \"lib.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return used(); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"k.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&Context::default()).unwrap();
    let first = generation(&session);
    // What it does not call: changed, and a function added.
    std::fs::write(
        dir.join("lib.ent"),
        format!(
            "{}fn added() -> Float {{ return 4.0; }}\n",
            library("1.0", "3.0")
        ),
    )
    .unwrap();
    session.sync();
    assert_eq!(generation(&session), first);
    // What it calls, changed: published again, and it answers so.
    std::fs::write(dir.join("lib.ent"), library("5.0", "3.0")).unwrap();
    session.sync();
    assert!(generation(&session) > first);
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let value = f32::from_bits(session.field(handle).unwrap().values()[0]);
    assert_eq!(value, 5.0);
    let _ = std::fs::remove_dir_all(&dir);
}
