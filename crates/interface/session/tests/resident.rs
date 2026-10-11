//! Replays played one after another on one kept session see what a fresh
//! session would show each: its state, its parameters and its pins start
//! over, and a document changed between them is read again.

use std::sync::Arc;

use entelechy_session::replay::{Replay, Resident, parse, run_fused};
use entelechy_session::{Disk, Limits, Session, Warm};

const SCENE: &str = "
param step: Float = 1.0;
node zero: kernel { requires { field.write; } output result: Rows<Float>; domain = 1; compute { return 0.0; } }
node count: kernel {
    requires { field.read; field.write; }
    param step: Float = step;
    input previous: Rows<Float> <~ count.result from zero.result;
    output result: Rows<Float>;
    domain = 1;
    compute { return at(previous, 0) + step; }
}
node baked: kernel { requires { field.write; } param step: Float = step; output result: Rows<Float>; domain = 1; compute { return step * 100.0; } }
output result <- count.result;
";

fn replay(dir: &std::path::Path, text: &str) -> Replay {
    parse(text, dir).unwrap()
}

fn limits() -> Limits {
    Limits {
        words: 1 << 20,
        work: 1 << 30,
        small: 0,
    }
}

/// What a replay shows, run on a session of its own.
fn fresh(replay: &Replay) -> Vec<String> {
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let runtime = Session::runtime_for("cpu", limits()).unwrap();
    run_fused(replay, store, runtime, true).unwrap().shown
}

#[test]
fn replays_on_one_kept_session_see_what_fresh_ones_would() {
    let dir = std::env::temp_dir().join(format!("entelechy-resident-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    let three = replay(
        &dir,
        "scene scene.esn\nparam step 3\nframe 0 to 0.2 every 0.1\nshow count.result[0]\nshow baked.result[0]\n",
    );
    let plain = replay(
        &dir,
        "scene scene.esn\nframe 0 to 0.1 every 0.1\nshow count.result[0]\nshow baked.result[0]\n",
    );
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let mut resident = Resident::new(Warm::open("cpu", limits()).unwrap(), store, true);
    for replay in [&three, &plain, &three, &plain] {
        let kept = resident.play(replay, None).unwrap();
        assert!(kept.failures.is_empty(), "{:?}", kept.failures);
        assert_eq!(kept.shown, fresh(replay));
    }
    // The scene changed between replays: read again, as a fresh one reads it.
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("step * 100.0", "step * 7.0"),
    )
    .unwrap();
    let kept = resident.play(&plain, None).unwrap();
    assert_eq!(kept.shown, fresh(&plain));
    assert!(
        kept.shown.iter().any(|line| line.ends_with(" 7")),
        "{:?}",
        kept.shown
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A change reaches a replay through the nodes its checks read, now or as
/// committed state: one that reads only the count is not reached by the
/// baked node's Kernel changing, and one that reads the baked node is.
#[test]
fn a_change_reaches_the_replays_whose_checks_read_it() {
    let dir = std::env::temp_dir().join(format!("entelechy-reached-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    let counting = replay(
        &dir,
        "scene scene.esn\nframe 0 to 0.2 every 0.1\nshow count.result[0]\n",
    );
    let baking = replay(&dir, "scene scene.esn\nframe 0\nshow baked.result[0]\n");
    let store: Arc<dyn entelechy_session::Store> = Arc::new(Disk::default());
    let mut resident = Resident::new(Warm::open("cpu", limits()).unwrap(), store, true);
    // Played in either order, each knows what it reads.
    let baked_first = resident.play(&baking, None).unwrap().touched;
    let counted = resident.play(&counting, None).unwrap().touched;
    let baked = resident.play(&baking, None).unwrap().touched;
    assert_eq!(baked, baked_first);
    assert!(counted.contains("count") && counted.contains("zero") && !counted.contains("baked"));
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("step * 100.0", "step * 7.0"),
    )
    .unwrap();
    let changed = resident.changed();
    let scene = counting.scene.clone();
    assert!(!resident.affects(&changed, &scene, &counted));
    assert!(resident.affects(&changed, &scene, &baked));
    let _ = std::fs::remove_dir_all(&dir);
}
