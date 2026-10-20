//! A law born inside the world: a Kernel writes a law's document through
//! an output `to files`, where the person running it allowed, and the
//! next sync publishes it in the law's place.

use std::path::PathBuf;

use entelechy_kernel::Context;
use entelechy_session::Session;

const SEED: &str = "kernel world.rule@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }\n";
const BORN: &str = "kernel world.rule@1 { requires { field.write; } output result: Rows<Float>; compute { return 2.0; } }\n";

/// The words of a `to files` list writing `text` at `path`.
fn writing(path: &str, text: &str) -> Vec<u32> {
    let mut words = vec![1, 0, path.len() as u32, text.len() as u32];
    words.extend(path.bytes().map(u32::from));
    words.extend(text.bytes().map(u32::from));
    words
}

/// A scene whose author writes, once its count reaches each threshold,
/// the text paired with it: the last threshold reached wins.
fn scene(stages: &[(i32, &str)]) -> String {
    let lists: Vec<Vec<u32>> = stages
        .iter()
        .map(|(_, text)| writing("rule.ent", text))
        .collect();
    let n = lists.iter().map(Vec::len).max().unwrap_or(1);
    let mut chosen = String::from("0");
    for ((at, _), list) in stages.iter().zip(&lists) {
        let mut words = list.clone();
        words.resize(n, 0);
        let words = words
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        chosen = format!("if at(count, 0) >= {at} {{ [{words}][element()] }} else {{ {chosen} }}");
    }
    format!(
        "use \"laws/rule.ent\";
kernel t.count@1 {{ requires {{ field.read; field.write; }} input before: Rows<Int>; output result: Rows<Int>; compute {{ return sample(before) + 1; }} }}
kernel t.zero@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ return 0; }} }}
kernel t.author@1 {{ requires {{ field.read; field.write; }} input count: Rows<Int>; output result: Rows<Int>; compute {{ return {chosen}; }} }}
kernel t.copy@1 {{ requires {{ field.read; field.write; }} input words: Rows<Int>; output result: Rows<Int>; compute {{ return sample(words); }} }}
input problems: Rows<Int> = checks \"laws/\";
node zero: t.zero@1 {{ domain = 1; }}
node count: t.count@1 {{ before <~ count.result from zero.result; }}
node author: t.author@1 {{ count <- count.result; domain = {n}; }}
node rule: world.rule@1 {{ domain = 1; }}
node seen: t.copy@1 {{ words <- problems; }}
output rule <- rule.result;
output problems <- seen.result;
output laws <- author.result to files \"laws/\";
"
    )
}

fn scratch(name: &str, stages: &[(i32, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-genesis-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("laws")).unwrap();
    std::fs::write(dir.join("laws/rule.ent"), SEED).unwrap();
    std::fs::write(dir.join("scene.esn"), scene(stages)).unwrap();
    dir
}

#[test]
fn a_law_a_kernel_writes_is_published_in_its_place() {
    let dir = scratch("born", &[(3, BORN)]);
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.allow_writes(&dir.join("laws")).unwrap();
    let mut seen = Vec::new();
    for _ in 0..8 {
        let events = session.sync();
        session.evaluate(&Context::default()).unwrap();
        let handle = session.output(Some("rule")).unwrap();
        let value = f32::from_bits(session.field(handle).unwrap().values()[0]);
        seen.push((
            value,
            events.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ));
    }
    let values: Vec<f32> = seen.iter().map(|(value, _)| *value).collect();
    assert_eq!(values, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0], "{seen:?}");
    assert_eq!(
        std::fs::read_to_string(dir.join("laws/rule.ent")).unwrap(),
        BORN
    );
    // The law is the world's: its output `laws` wrote it.
    let origins: Vec<_> = session
        .changes_since(0)
        .into_iter()
        .filter(|record| matches!(&record.change, entelechy_session::Change::Published { id, .. } if id.to_string() == "world.rule@1"))
        .map(|record| record.origin.clone())
        .collect();
    assert_eq!(
        origins.last(),
        Some(&entelechy_session::Origin::Output("laws".into())),
        "{origins:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Each frame: the law's value, and the problems the world reads in its
/// laws, as text.
fn frames(session: &mut Session, count: usize) -> Vec<(f32, String)> {
    (0..count)
        .map(|_| {
            session.sync();
            session.evaluate(&Context::default()).unwrap();
            let handle = session.output(Some("rule")).unwrap();
            let value = f32::from_bits(session.field(handle).unwrap().values()[0]);
            let handle = session.output(Some("problems")).unwrap();
            let text: String = session
                .field(handle)
                .unwrap()
                .values()
                .iter()
                .take_while(|&&c| c != 0)
                .filter_map(|&c| char::from_u32(c))
                .collect();
            (value, text)
        })
        .collect()
}

#[test]
fn a_law_the_world_gets_wrong_leaves_the_last_one_running_and_tells_it_why() {
    let broken = "kernel world.rule@1 { requires { field.write; } output result: Rows<Float>; compute { return nope; } }\n";
    let dir = scratch("wrong", &[(3, broken), (5, BORN)]);
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    assert!(session.graph().is_none());
    session.sync();
    session.allow_writes(&dir.join("laws")).unwrap();
    let seen = frames(&mut session, 8);
    let values: Vec<f32> = seen.iter().map(|(value, _)| *value).collect();
    // The broken law is refused and the seed runs on; the one written
    // after it is published.
    assert_eq!(values, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "{seen:?}");
    // The world reads why while its law is wrong, and nothing once mended.
    assert!(
        seen[3].1.contains("rule.ent") && seen[3].1.contains("nope"),
        "{seen:?}"
    );
    assert_eq!(seen[7].1.trim(), "", "{seen:?}");
    // The refusal is the world's own.
    let refused: Vec<_> = session
        .attempts_since(0)
        .into_iter()
        .map(|attempt| attempt.origin)
        .collect();
    assert!(
        refused.contains(&entelechy_session::Origin::Output("laws".into())),
        "{refused:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_replay_that_allows_writes_sees_the_law_born_and_leaves_the_disk_as_it_was() {
    let dir = scratch("replay", &[(3, BORN)]);
    let text = "scene scene.esn
allow file.write laws/
frame 0.0
frame 0.1
frame 0.2
expect rule.result[0] = 1.0
frame 0.3
expect rule.result[0] = 2.0
";
    std::fs::write(dir.join("born.replay"), text).unwrap();
    let replay = entelechy_session::replay::parse(text, &dir).unwrap();
    for _ in 0..2 {
        // Every run starts from the seed: what the last one wrote was its own.
        let outcome = entelechy_session::replay::run_in(
            &replay,
            std::sync::Arc::new(entelechy_session::Disk::default()),
            entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(entelechy_cpu::CpuBackend)),
        );
        assert_eq!(outcome.map(|outcome| outcome.failures), Ok(Vec::new()));
    }
    assert_eq!(
        std::fs::read_to_string(dir.join("laws/rule.ent")).unwrap(),
        SEED
    );
    // Without the allowance, nothing is written and the seed runs on.
    let unallowed =
        entelechy_session::replay::parse(&text.replace("allow file.write laws/\n", ""), &dir)
            .unwrap();
    let outcome = entelechy_session::replay::run_in(
        &unallowed,
        std::sync::Arc::new(entelechy_session::Disk::default()),
        entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(entelechy_cpu::CpuBackend)),
    );
    assert_eq!(
        outcome.unwrap().failures,
        ["line 7: rule.result[0] is 1, expected 2.0"],
        "the law is not born"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_resident_replay_that_allows_writes_sees_the_law_born() {
    let dir = scratch("resident", &[(3, BORN)]);
    let text = "scene scene.esn
allow file.write laws/
frame 0.0
frame 0.1
frame 0.2
frame 0.3
expect rule.result[0] = 2.0
frame 0.4
expect rule.result[0] = 2.0
frame 0.5
expect rule.result[0] = 2.0
frame 0.6
expect rule.result[0] = 2.0
";
    let replay = entelechy_session::replay::parse(text, &dir).unwrap();
    let store: std::sync::Arc<dyn entelechy_session::Store> =
        std::sync::Arc::new(entelechy_session::Disk::with_packages(Vec::new()));
    let warm = entelechy_session::Warm::open("cpu", entelechy_session::Limits::default()).unwrap();
    let mut resident = entelechy_session::replay::Resident::new(warm, store, true);
    let outcome = resident.play(&replay, None).unwrap();
    assert_eq!(outcome.failures, Vec::<String>::new());
    let _ = std::fs::remove_dir_all(&dir);
}
