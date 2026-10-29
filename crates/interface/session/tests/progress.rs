//! An input `= progress` holds the Kernels the session watches preparing
//! and the scene they are for, as words: its own, or another's given;
//! read again each evaluation.

use entelechy_ir::KernelId;
use entelechy_kernel::{Context, Progress};
use entelechy_session::Session;

const SCENE: &str = "
kernel t.copy@1 { requires { field.read; field.write; } input work: Rows<Int>; output result: Rows<Int>; compute { return sample(work); } }
input work: Rows<Int> = progress;
node seen: t.copy@1 { work <- work; }
output result <- seen.result;
";

fn open(scene: &str, name: &str) -> (Session, Vec<entelechy_session::Event>) {
    let dir =
        std::env::temp_dir().join(format!("entelechy-progress-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), scene).unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    (session, events)
}

fn words(session: &mut Session) -> Vec<u32> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    session.field(handle).unwrap().values().to_vec()
}

#[test]
fn an_input_holds_the_kernels_watched_preparing_as_words() {
    let (mut session, events) = open(SCENE, "words");
    assert!(session.graph().is_some(), "{events:?}");
    // Its own: one node on its one Kernel, built, nothing running.
    let own = words(&mut session);
    assert_eq!(&own[..7], &[1, 0, 1, 1, 0, 0, 4], "{own:?}");
    assert_eq!(own.last(), Some(&0));
    // Another's, as it changes: three nodes, two Kernels not being
    // prepared (built), linked in a line.
    let other = Progress::default();
    session.watch(other.clone());
    let id = |name: &str| KernelId::new(name, 1).unwrap();
    other.shape(vec![id("o.a"), id("o.b"), id("o.a")], [(1, 0), (2, 1)]);
    let seen = words(&mut session);
    assert_eq!(seen, vec![3, 2, 0, 0, 0, 0, 4, 4, 4, 0, 1, 1, 2, 0]);
    assert_eq!(entelechy_session::preparing::HEAD, 6);
}

#[test]
fn progress_is_read_as_words() {
    let (session, events) = open(
        "input work: Rows<Float> = progress;\noutput result <- work;\n",
        "typed",
    );
    assert!(session.graph().is_none());
    let said = format!("{events:?}");
    assert!(
        said.contains("`= progress` gives words, a `Rows<Int>`"),
        "{said}"
    );
    assert!(said.contains("not `Rows<Float>`"), "{said}");
}
