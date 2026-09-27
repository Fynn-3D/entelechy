//! Fusion changes how a scene runs, never what it computes: every
//! example's replay, played with fusion and without, gives every output
//! of the scene the same words at every frame.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_session::Session;
use orla_session::replay::{self, Replayer};

fn replays() -> Vec<PathBuf> {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let mut out = Vec::new();
    for example in std::fs::read_dir(&examples).unwrap().flatten() {
        let tests = example.path().join("tests");
        for entry in std::fs::read_dir(tests).into_iter().flatten().flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "replay") {
                out.push(entry.path());
            }
        }
    }
    out.sort();
    out
}

/// A session over the replay's scene, with fusion on or off.
fn opened(replay: &replay::Replay, fusion: bool) -> Session {
    let runtime = Session::runtime_for("jit", orla_session::Limits::default()).unwrap();
    let mut session = Session::open_in(
        Arc::new(orla_session::Disk::default()),
        &replay.scene,
        replay.kernels.as_deref(),
        runtime,
    );
    session.set_fusion(fusion);
    let _ = session.sync();
    session
}

/// Every output of the scene by name, as words.
fn outputs(session: &mut Session, report: &orla_graph::Report) -> Vec<(String, Vec<u32>)> {
    report
        .outputs
        .iter()
        .map(|(name, handle)| {
            let words = session.field(*handle).unwrap().values().into_owned();
            (name.clone(), words)
        })
        .collect()
}

/// What `path`'s replay found different with fusion, and how many frames
/// it compared.
fn compared(path: &Path) -> (Vec<String>, usize) {
    let mut wrong = Vec::new();
    let mut frames = 0;
    let text = std::fs::read_to_string(path).unwrap();
    let replay = replay::parse(&text, path.parent().unwrap()).unwrap();
    let mut fused = opened(&replay, true);
    let mut apart = opened(&replay, false);
    let mut with = Replayer::start(&replay, &mut fused);
    let mut without = Replayer::start(&replay, &mut apart);
    loop {
        let (Some(a), Some(b)) = (
            with.next_frame(&mut fused).unwrap(),
            without.next_frame(&mut apart).unwrap(),
        ) else {
            break;
        };
        let (Ok(one), Ok(other)) = (fused.evaluate(&a.context), apart.evaluate(&b.context)) else {
            wrong.push(format!(
                "{}: a frame failed on one side only, or both",
                path.display()
            ));
            break;
        };
        frames += 1;
        let (x, y) = (outputs(&mut fused, &one), outputs(&mut apart, &other));
        for ((name, a), (_, b)) in x.iter().zip(&y) {
            if a != b {
                let at = a.iter().zip(b).position(|(p, q)| p != q);
                wrong.push(format!(
                    "{}: at {} `{name}` differs fused, first at word {at:?}",
                    path.display(),
                    a.len()
                ));
            }
        }
        with.evaluated(one);
        without.evaluated(other);
    }
    (wrong, frames)
}

#[test]
fn fusion_changes_no_output_of_any_example() {
    let paths = replays();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(path) = paths.get(at) else { break };
                    let found = compared(path);
                    results.lock().unwrap().push(found);
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let frames: usize = results.iter().map(|(_, frames)| frames).sum();
    let wrong: Vec<String> = results.into_iter().flat_map(|(wrong, _)| wrong).collect();
    assert!(frames > 0, "no frame was compared");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
