//! A replay starts from the same files whoever has played before it: what
//! a scene saved for itself on disk is not read, until the run writes it.

use std::sync::Arc;

use entelechy_kernel::Context;
use entelechy_session::replay::{Resident, parse, run_fused};
use entelechy_session::{Disk, Event, Limits, Session, Warm};

const SCENE: &str = "
node fresh: kernel { requires { field.write; } output result: Rows<Int>; domain = 1; compute { return 7; } }
input saved: Rows<Int> = \"saves/n.field\" else fresh.result;
node count: kernel {
    requires { field.read; field.write; }
    input previous: Rows<Int> <~ count.result from saved;
    output result: Rows<Int>;
    domain = 1;
    compute { return at(previous, 0) + 1; }
}
output kept <- count.result to \"saves/n.field\";
";

fn limits() -> Limits {
    Limits {
        words: 1 << 20,
        work: 1 << 30,
        small: 0,
    }
}

/// A scene that has been played: its save on disk.
fn played(name: &str, saved: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("own_saves_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("saves")).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    std::fs::write(dir.join("saves/n.field"), saved).unwrap();
    dir
}

/// The save a run of `scene` leaves: each frame's count is one more.
fn saved_by(scene: &str, name: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("own_saves_{}_by_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("saves")).unwrap();
    std::fs::write(dir.join("scene.esn"), scene).unwrap();
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let runtime = Session::runtime_for("cpu", limits()).unwrap();
    let mut session = Session::open_in(store, &dir.join("scene.esn"), None, runtime);
    session.sync();
    session.allow_writes(&dir.join("saves")).unwrap();
    for frame in 0..3 {
        session
            .evaluate(&Context {
                time: f64::from(frame) * 0.1,
                ..Context::default()
            })
            .unwrap();
        session.sync();
    }
    std::fs::read(dir.join("saves/n.field")).unwrap()
}

#[test]
fn a_replay_does_not_read_what_a_person_saved() {
    let dir = played("disk", &saved_by(SCENE, "disk"));
    let replay = parse(
        "scene scene.esn\nframe 0.0\nexpect count.result[0] = 8\n",
        &dir,
    )
    .unwrap();
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let runtime = Session::runtime_for("cpu", limits()).unwrap();
    let run = run_fused(&replay, store, runtime, false).unwrap();
    assert!(run.failures.is_empty(), "{:?}", run.failures);
}

#[test]
fn a_resident_replay_does_not_either() {
    let dir = played("resident", &saved_by(SCENE, "resident"));
    let replay = parse(
        "scene scene.esn\nframe 0.0\nexpect count.result[0] = 8\n",
        &dir,
    )
    .unwrap();
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let warm = Warm::open("cpu", limits()).unwrap();
    let run = Resident::new(warm, store, false)
        .play_as(None, &replay, None)
        .unwrap();
    assert!(run.failures.is_empty(), "{:?}", run.failures);
}

#[test]
fn a_replay_reads_back_what_it_wrote_itself() {
    let saved = saved_by(SCENE, "back");
    let dir = played("back", &saved);
    let replay = parse(
        "scene scene.esn\nallow file.write saves\nframe 0.0\nexpect count.result[0] = 8\nframe 0.1\nreset\nframe 0.2\nexpect count.result[0] > 8\n",
        &dir,
    )
    .unwrap();
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let runtime = Session::runtime_for("cpu", limits()).unwrap();
    let run = run_fused(&replay, store, runtime, false).unwrap();
    assert!(run.failures.is_empty(), "{:?}", run.failures);
    assert_eq!(
        std::fs::read(dir.join("saves/n.field")).unwrap(),
        saved,
        "a run changes nothing on disk"
    );
}

/// A save in a layout the scene has left behind is refused and named; the
/// scene carries on from its fallback instead of stopping.
#[test]
fn a_save_of_another_layout_falls_back_and_says_which_file() {
    let old = SCENE
        .replace("Rows<Int>", "Rows<Float>")
        .replace("return 7;", "return 7.0;")
        .replace("+ 1;", "+ 1.0;");
    let dir = played("layout", &saved_by(&old, "layout"));
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let runtime = Session::runtime_for("cpu", limits()).unwrap();
    let mut session = Session::open_in(store, &dir.join("scene.esn"), None, runtime);
    let events = session.sync();
    let told = events
        .iter()
        .any(|e| matches!(e, Event::Rejected { path, .. } if path.ends_with("saves/n.field")));
    assert!(told, "{events:?}");
    session.evaluate(&Context::default()).unwrap();
}
