//! Hostile input: mutations of the examples' documents and the protocol's
//! requests, fed to the parser, the compiler, the scene lowering, a live
//! session and the protocol handler. Every answer is a value or an error,
//! in bounded time, without a panic. Mutations come from a seed;
//! `ORLA_FUZZ=<n>` runs more.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use orla_conformance::random::Rng;
use orla_session::Session;

fn count() -> usize {
    std::env::var("ORLA_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300)
}

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples")
}

/// Every `.orla` and `.osn` document the examples ship, with its path.
fn documents() -> Vec<(PathBuf, String)> {
    let mut found = Vec::new();
    let mut stack = vec![examples()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| ext == "orla" || ext == "osn")
            {
                found.push((path.clone(), std::fs::read_to_string(&path).unwrap()));
            }
        }
    }
    found.sort();
    found
}

/// Language pieces a mutation inserts, to reach rare parser, checker and
/// lowering paths.
const TOKENS: &[&str] = &[
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ";",
    ",",
    ".",
    "=",
    "<-",
    "<~",
    "@",
    "::",
    "\"",
    "//",
    "kernel",
    "node",
    "input",
    "output",
    "param",
    "struct",
    "fn",
    "use",
    "graph",
    "times",
    "carry",
    "domain",
    "loop",
    "with",
    "until",
    "if",
    "else",
    "match",
    "each",
    "scatter",
    "prefix",
    "scan",
    "merge",
    "emit",
    "return",
    "let",
    "requires",
    "field.read",
    "Rows<Int>",
    "Grid<Vec4>",
    "Volume<Float>",
    "[Int; 4096]",
    "4294967295",
    "-2147483648",
    "1e39",
    "0x",
    "999999999999999999999",
    "vec3(",
    "window()",
    "value(",
    "budget 0",
    "\u{0}",
    "\u{feff}",
    "é",
    "∞",
];

/// One mutation of `text`: a range cut or doubled, a token inserted, a
/// region nested deep, or bytes swapped.
fn mutate(rng: &mut Rng, text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1);
    let at = rng.below(n as u64) as usize;
    let len = (rng.below(40) as usize).min(n - at.min(n));
    let s: String = match rng.below(7) {
        0 => chars[..at]
            .iter()
            .chain(&chars[(at + len).min(chars.len())..])
            .collect(),
        1 => chars[..at + len].iter().chain(&chars[at..]).collect(),
        2 | 3 => {
            let token = *rng.pick(TOKENS);
            chars[..at].iter().collect::<String>()
                + " "
                + token
                + " "
                + &chars[at..].iter().collect::<String>()
        }
        4 => {
            let depth = 1 + rng.below(400) as usize;
            let (open, close) = *rng.pick(&[("(", ")"), ("{", "}"), ("[", "]"), ("if x { ", " }")]);
            chars[..at].iter().collect::<String>()
                + &open.repeat(depth)
                + &chars[at..].iter().collect::<String>()
                + &close.repeat(depth)
        }
        5 => {
            let mut c = chars.clone();
            let other = rng.below(n as u64) as usize;
            if at < c.len() && other < c.len() {
                c.swap(at, other);
            }
            c.into_iter().collect()
        }
        _ => {
            // A number in the text made enormous, negative or zero.
            let big = *rng.pick(&["0", "-1", "2147483647", "4294967296", "1000000000", "65536"]);
            let mut out = String::new();
            let mut replaced = false;
            for (k, c) in chars.iter().enumerate() {
                if !replaced && k >= at && c.is_ascii_digit() {
                    out.push_str(big);
                    replaced = true;
                    continue;
                }
                out.push(*c);
            }
            out
        }
    };
    s
}

/// Runs `work`, reporting a panic or an overlong run.
fn guarded(what: &str, work: impl FnOnce()) -> Option<String> {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(work));
    let took = started.elapsed();
    if outcome.is_err() {
        return Some(format!("{what}: panicked"));
    }
    // Generous for a loaded debug build: it catches blow-ups.
    (took > Duration::from_secs(20)).then(|| format!("{what}: took {took:?}"))
}

#[test]
fn mutated_documents_are_refused_or_taken_never_crash_the_front_end() {
    let documents = documents();
    let mut rng = Rng::new(0x4057);
    let mut failures = Vec::new();
    for case in 0..count() {
        let (path, text) = &documents[rng.below(documents.len() as u64) as usize];
        let mut mutated = text.clone();
        for _ in 0..1 + rng.below(3) {
            mutated = mutate(&mut rng, &mutated);
        }
        let name = format!("case {case} from {}", path.display());
        let parsed = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: parse"), move || {
            let _ = orla_language::parse_document(&parsed);
        }) {
            failures.push((failure, mutated.clone()));
            continue;
        }
        let compiled = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: compile"), move || {
            if let Ok(document) = orla_language::parse_document(&compiled) {
                for kernel in &document.kernels {
                    let _ = orla_compiler::compile_kernel(kernel, None);
                }
            }
        }) {
            failures.push((failure, mutated.clone()));
            continue;
        }
        let lowered = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: lower"), move || {
            let _ = orla_scene::parse(&lowered);
        }) {
            failures.push((failure, mutated.clone()));
        }
    }
    let said: Vec<String> = failures
        .iter()
        .take(5)
        .map(|(failure, text)| {
            format!("{failure}\n{}", text.chars().take(1500).collect::<String>())
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {}:\n{}",
        failures.len(),
        count(),
        said.join("\n\n")
    );
}

/// A live session given mutated scenes by `compose`, evaluated after each:
/// every scene is taken or refused, every evaluation succeeds or fails,
/// and the session answers the next request.
#[test]
fn a_live_session_survives_mutated_scenes_and_requests() {
    // Examples reading nothing outside their directory, copied whole so
    // `use "kernels/"` resolves; one session each.
    let chosen: Vec<PathBuf> = std::fs::read_dir(examples())
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|dir| {
            std::fs::read_to_string(dir.join("scene.osn"))
                .is_ok_and(|text| !text.contains("../") && !text.contains("= \""))
        })
        .collect();
    assert!(
        chosen.len() > 5,
        "found {} examples to mutate",
        chosen.len()
    );
    let mut sessions = Vec::new();
    let mut scenes = Vec::new();
    for (k, example) in chosen.iter().enumerate() {
        let dir = std::env::temp_dir().join(format!("orla-hostile-{}-{k}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        copy_dir(example, &dir);
        let scene = dir.join("scene.osn");
        scenes.push(std::fs::read_to_string(&scene).unwrap());
        let runtime = Session::runtime_for(
            "jit",
            orla_session::Limits {
                words: 1 << 22,
                ..Default::default()
            },
        )
        .unwrap();
        let mut session = Session::open_with(&scene, None, runtime);
        session.restrict_reads().unwrap();
        session.sync();
        sessions.push((dir, session));
    }
    let mut rng = Rng::new(0x5e55);
    let mut failures = Vec::new();
    let ops = [
        "status", "kernels", "graph", "document", "evaluate", "read", "trace", "help", "compose",
        "publish", "prepare", "retire", "param", "write", "reset", "sync", "nonsense",
    ];
    for case in 0..count() / 3 {
        let which = rng.below(scenes.len() as u64) as usize;
        let base = &scenes[which];
        let session = &mut sessions[which].1;
        let mut text = base.clone();
        for _ in 0..rng.below(3) {
            text = mutate(&mut rng, &text);
        }
        let compose = serde_json::json!({ "id": case, "op": "compose", "text": text }).to_string();
        let mut lines = vec![compose];
        for _ in 0..3 {
            let op = *rng.pick(&ops);
            let mut request = serde_json::json!({ "id": case, "op": op });
            // Fields an agent might send, of the right kind or not.
            for (key, values) in [
                (
                    "frames",
                    &[
                        serde_json::json!(1),
                        serde_json::json!(0),
                        serde_json::json!(4000000000u64),
                    ][..],
                ),
                (
                    "time",
                    &[
                        serde_json::json!(0.5),
                        serde_json::json!(-1e30),
                        serde_json::json!("x"),
                    ][..],
                ),
                (
                    "node",
                    &[
                        serde_json::json!("show"),
                        serde_json::json!(""),
                        serde_json::json!(7),
                    ][..],
                ),
                (
                    "source",
                    &[
                        serde_json::json!(mutate(&mut rng, base)),
                        serde_json::json!(null),
                    ][..],
                ),
                (
                    "name",
                    &[serde_json::json!("level"), serde_json::json!("x")][..],
                ),
                (
                    "value",
                    &[serde_json::json!([1, 2]), serde_json::json!(1e300)][..],
                ),
                (
                    "format",
                    &[
                        serde_json::json!("ppm"),
                        serde_json::json!("field"),
                        serde_json::json!("??"),
                    ][..],
                ),
                (
                    "at",
                    &[
                        serde_json::json!([0, 0, 0]),
                        serde_json::json!([4294967295u64, 0]),
                    ][..],
                ),
            ] {
                if rng.below(3) == 0 {
                    request[key] = rng.pick(values).clone();
                }
            }
            let mut line = request.to_string();
            if rng.below(8) == 0 {
                line = mutate(&mut rng, &line);
            }
            lines.push(line);
        }
        for line in lines {
            let shown = line.chars().take(200).collect::<String>();
            if let Some(failure) = guarded(&format!("case {case}: {shown}"), || {
                let _ = orla_protocol::respond_line(session, &line);
            }) {
                failures.push(failure);
            }
        }
        // The session still answers.
        let (status, _) = orla_protocol::respond_line(session, r#"{"id":0,"op":"status"}"#);
        if !status.contains("\"ok\"") {
            failures.push(format!("case {case}: status no longer answers: {status}"));
        }
    }
    for (dir, _) in sessions {
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(
        failures.is_empty(),
        "{}:\n{}",
        failures.len(),
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            copy_dir(&path, &to.join(entry.file_name()));
        } else {
            std::fs::copy(&path, to.join(entry.file_name())).unwrap();
        }
    }
}
