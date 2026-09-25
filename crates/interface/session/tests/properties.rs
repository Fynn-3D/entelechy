//! A session under a random life: evaluations, Kernels republished with
//! the state's layout changed and back, broken documents, a check that
//! fails at a count, resets. Whatever the order, three things hold:
//!
//! - an evaluation that succeeds advances the state by exactly one;
//! - one that fails leaves it as it was, a migration it ran included, so
//!   the next success counts on from where the last one stopped;
//! - what the Host holds does not grow with the number of steps.
//!
//! The steps come from a seed; `ORLA_FUZZ=<n>` takes more of them.

use std::path::Path;

use orla_conformance::random::Rng;
use orla_runtime::Context;
use orla_session::Session;

fn steps() -> usize {
    std::env::var("ORLA_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(400)
}

/// The library: the step in one of two layouts of its state, the
/// migrations between them, the zero, and a check that fails where the
/// count is `fail_at`.
fn library(layout: u32, fail_at: i32, broken: bool) -> String {
    let tally = if layout == 1 {
        "struct Tally { count: Int; }"
    } else {
        "struct Tally { count: Int; doubled: Int; }"
    };
    let next = if layout == 1 {
        "Tally { count: t.count + 1 }"
    } else {
        "Tally { count: t.count + 1, doubled: (t.count + 1) * 2 }"
    };
    let zero = if layout == 1 {
        "Tally { count: 0 }"
    } else {
        "Tally { count: 0, doubled: 0 }"
    };
    let broken = if broken { "nope" } else { "t.count" };
    format!(
        "kernel t.zero@1 {{ requires {{ field.write; }} {tally} output result: Rows<Tally>; compute {{ return {zero}; }} }}
kernel t.step@1 {{ requires {{ field.read; field.write; }} {tally} input previous: Rows<Tally>; output result: Rows<Tally>; compute {{ let t = sample(previous); return {next}; }} }}
kernel t.check@1 {{ requires {{ field.read; field.write; }} {tally} input tally: Rows<Tally>; output result: Rows<Int>; compute {{ let t = sample(tally); return 100 / ({broken} - ({fail_at})); }} }}
kernel t.up@1 {{ requires {{ field.read; field.write; }} struct Old {{ count: Int; }} struct Tally {{ count: Int; doubled: Int; }} input old: Rows<Old>; output result: Rows<Tally>; compute {{ let o = sample(old); return Tally {{ count: o.count, doubled: o.count * 2 }}; }} }}
kernel t.down@1 {{ requires {{ field.read; field.write; }} struct Wide {{ count: Int; doubled: Int; }} struct Tally {{ count: Int; }} input old: Rows<Wide>; output result: Rows<Tally>; compute {{ let o = sample(old); return Tally {{ count: o.count }}; }} }}
"
    )
}

/// The scene: the count carried by `<~` and checked every frame. Into
/// the wide layout the count migrates; into the narrow one nothing
/// carries it and it starts again, so which layout the committed state
/// is in shows in the count that follows.
fn scene(layout: u32) -> String {
    let migrate = if layout == 1 { "" } else { " migrate t.up@1" };
    format!(
        "use \"lib.orla\";
node seed: t.zero@1 {{ domain = 1; }}
node count: t.step@1 {{ previous <~ count.result from seed.result{migrate}; }}
node check: t.check@1 {{ tally <- count.result; }}
output result <- check.result;
output count <- count.result;
"
    )
}

/// Rewrites a file so its stamp moves even within the same second.
fn write(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    let file = std::fs::File::options().write(true).open(path).unwrap();
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(1);
    let _ = file.set_modified(later);
}

#[test]
fn a_random_life_keeps_the_state_and_the_host_honest() {
    let dir = std::env::temp_dir().join(format!("orla-properties-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, root) = (dir.join("lib.orla"), dir.join("scene.osn"));
    let (mut layout, mut fail_at) = (1u32, 1_000_000i32);
    write(&lib, &library(layout, fail_at, false));
    write(&root, &scene(layout));
    let runtime = Session::runtime_for("jit", Default::default()).unwrap();
    let mut session = Session::open_with(&root, None, runtime);
    session.sync();
    let count_node = orla_graph::NodeId("count".into());
    // What the model says the committed count is; `None` before the
    // first evaluation, when the seed's zero is what the step reads.
    let mut committed = 0i32;
    // The layout the committed state is in; `None` while it is the seed.
    let mut kept_as: Option<u32> = None;
    // A snapshot taken, and what the model said of the world then.
    let mut taken: Option<(orla_session::Snapshot, i32, Option<u32>)> = None;
    let mut restored = 0;
    let mut rng = Rng::new(0x57a7e);
    let mut most_held = 0usize;
    let mut held_after_warmup = None;
    let mut log = Vec::new();
    let (mut succeeded, mut failed, mut migrated) = (0, 0, 0);
    for step in 0..steps() {
        let choice = rng.below(13);
        match choice {
            // Evaluate, most of the time.
            0..=4 => {
                let outcome = session.evaluate(&Context::default());
                match outcome {
                    Ok(report) => {
                        succeeded += 1;
                        migrated += report.migrated.len();
                        let (_, words) = session
                            .element_field(&count_node, "result", 0, &["count".into()])
                            .unwrap();
                        let now = words[0] as i32;
                        // Into the narrow layout nothing carries the count.
                        let from = match kept_as {
                            Some(kept) if kept != layout && layout == 1 => 0,
                            _ => committed,
                        };
                        assert_eq!(
                            now,
                            from + 1,
                            "step {step}: a success advances by one from what is kept\n{}",
                            log.join("\n")
                        );
                        kept_as = Some(layout);
                        committed = now;
                        log.push(format!("{step}: evaluated, count {now}"));
                    }
                    Err(error) => {
                        failed += 1;
                        log.push(format!("{step}: failed: {error}"));
                    }
                }
            }
            // The step's layout changes, with or without a broken check.
            5 | 6 => {
                layout = 3 - layout;
                let broken = rng.below(4) == 0;
                write(&lib, &library(layout, fail_at, broken));
                write(&root, &scene(layout));
                session.sync();
                log.push(format!(
                    "{step}: layout {layout}{}",
                    if broken { ", broken" } else { "" }
                ));
                if broken {
                    // A broken library publishes nothing: put it right,
                    // and the layout it was meant to bring comes.
                    write(&lib, &library(layout, fail_at, false));
                    session.sync();
                }
            }
            // The check fails at the next count, or never.
            7 => {
                fail_at = if rng.below(2) == 0 {
                    committed + 1
                } else {
                    1_000_000
                };
                write(&lib, &library(layout, fail_at, false));
                session.sync();
                log.push(format!("{step}: check fails at {fail_at}"));
            }
            // A broken library alone: nothing of it is published.
            8 => {
                write(&lib, &library(layout, fail_at, true));
                session.sync();
                write(&lib, &library(layout, fail_at, false));
                session.sync();
                log.push(format!("{step}: broken and mended"));
            }
            // A snapshot of the world as it stands.
            10 => {
                let snapshot = session.snapshot().unwrap();
                log.push(format!("{step}: snapshot at {committed}"));
                taken = Some((snapshot, committed, kept_as));
            }
            // A restore under the laws that run: it stands only if its
            // trial frame runs, and then the world is the snapshot's, its
            // state migrated or restarted as the current layout takes it;
            // refused, the world is as it was.
            11 | 12 => {
                let Some((snapshot, count, kept)) = &taken else {
                    continue;
                };
                match session.restore(snapshot, orla_session::Laws::Current, &Context::default()) {
                    Ok(_) => {
                        restored += 1;
                        (committed, kept_as) = match kept {
                            // Into the narrow layout nothing carries it.
                            Some(old) if *old != layout && layout == 1 => (0, None),
                            Some(_) => (*count, Some(layout)),
                            None => (0, None),
                        };
                        log.push(format!("{step}: restored {count} as {committed}"));
                    }
                    Err(why) => log.push(format!("{step}: restore refused: {why}")),
                }
            }
            // A reset: the state starts again from the seed.
            _ => {
                session.reset_state();
                committed = 0;
                kept_as = None;
                log.push(format!("{step}: reset"));
            }
        }
        let held = session.runtime().host().used_words();
        most_held = most_held.max(held);
        if step == steps() / 4 {
            held_after_warmup = Some(most_held);
        }
    }
    // The life went through each of what it is about.
    assert!(
        succeeded > steps() / 10 && failed > 0 && migrated > 0 && restored > 0,
        "{succeeded} succeeded, {failed} failed, {migrated} migrated, {restored} restored"
    );
    // A leak grows with the steps; what the scene holds does not.
    let warm = held_after_warmup.unwrap();
    assert!(
        most_held <= warm * 2 + 64,
        "the Host held {warm} words a quarter of the way and {most_held} by the end"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
