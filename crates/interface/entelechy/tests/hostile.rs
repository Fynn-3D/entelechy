//! Hostile input: mutations of the documents below and the protocol's
//! requests, fed to the parser, the compiler, the scene lowering, a live
//! session and the protocol handler. Every answer is a value or an error,
//! in bounded time, without a panic. Mutations come from a seed;
//! `ENTELECHY_FUZZ=<n>` runs more.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use entelechy_conformance::random::Rng;
use entelechy_session::Session;

fn count() -> usize {
    std::env::var("ENTELECHY_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300)
}

/// A library of Kernels in every mode, a seed of the front end's
/// mutations.
const LIBRARY: &str = r#"//! Kernels of every mode.
struct Particle { position: Vec2; velocity: Vec2; }
type Swarm = Rows<Particle>;

fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; }

/// A picture turned gray.
kernel t.gray@1 {
    requires { field.read; field.write; }
    input source: Image;
    param amount: Float = 1.0;
    output result: Image;
    compute { let c = sample(source); let g = luma(c); return mix(c, vec4(g, g, g, c.a), amount); }
}

kernel t.stats@1 {
    requires { field.read; field.write; }
    input source: Image;
    output result: Rows<Light>;
    struct Light { sum: Float; max: Float; count: Int; }
    compute over source { let l = luma(sample(source)); return Light { sum: l, max: l, count: 1 }; }
    merge(a, b) { return Light { sum: a.sum + b.sum, max: max(a.max, b.max), count: a.count + b.count }; }
}

kernel t.lowpass@1 {
    requires { field.read; field.write; }
    input samples: Rows<Float>;
    param k: Float = 0.1;
    output result: Rows<Float>;
    scan over samples with y = 0.0 { return y + (sample(samples) - y) * k; }
}

kernel t.machine@1 {
    requires { field.read; field.write; }
    input ticks: Rows<Int>;
    memory ram: Rows<Int>;
    output result: Rows<Int>;
    scan over ticks with pc = 0 { let op = at(ram, pc); put(ram, pc + 100, op * 2); return pc + 1; }
}

kernel t.splat@1 {
    requires { field.read; field.write; }
    input particles: Swarm;
    input canvas: Grid<Vec3>;
    output result: Grid<Vec3>;
    struct Splat { cell: Vec2; heat: Vec3; }
    scatter over particles into canvas with sum {
        let p = sample(particles);
        return Splat { cell: floor(p.position * size(canvas)), heat: vec3(p.velocity.x, p.velocity.y, 1.0) };
    }
}

kernel t.offsets@1 {
    requires { field.read; field.write; }
    input flags: Rows<Int>;
    output result: Rows<Int>;
    prefix over flags with sum { return if sample(flags) != 0 { 1 } else { 0 }; }
}

kernel t.orbit@1 {
    requires { field.write; time.read; }
    param steps: Int = 16;
    output result: Image;
    compute {
        let p = coord() / domain() - vec2(0.5, 0.5);
        let z = loop steps with z = vec2(0.0, 0.0) until dot(z, z) > 4.0 { vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + p };
        let shade = match int(time()) % 3 { 0 => 0.2, 1 | 2 => 0.6, _ => 1.0 };
        let taps = [each 4 { float(index()) * 0.25 }];
        return vec4(z.x, taps[2], shade, 1.0);
    }
}
"#;

/// A scene placing the library's Kernels, a graph and a saved output, a
/// seed of the front end's mutations.
const PLACED: &str = r#"//! A scene of placed Kernels and a placed graph.
use "library.ent";

input photo: Image;
param amount: Float = 1.0;
require gentle: amount <= 4;

node gray: t.gray@1 { source <- photo; amount = amount; }
node steps: graph "step.esn" { times = 3; carry cells <- result; cells <~ steps.result from gray.result; }
node show: t.stats@1 { source <- steps.result; }

output result <- gray.result;
output kept <- show.result to "saves/stats.field";
"#;

/// Scenes whole in one document: seeds of the front end's mutations, and
/// each run live by a session.
const SCENES: &[&str] = &[
    // A picture made, turned gray and cut into blocks.
    r#"param level: Float = 0.5;
require gentle: level <= 4;
node source: kernel { requires { field.write; } param depth: Float = 0.5; output result: Image; domain = vec2(32, 24);
    compute { let uv = coord() / domain(); return vec4(uv.x, uv.y, depth, 1.0); } }
node gray: kernel { requires { field.read; field.write; } input source: Image <- source.result; param amount: Float = level; output result: Image;
    fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; }
    compute { let c = sample(source); let g = luma(c); return mix(c, vec4(g, g, g, c.a), amount); } }
node show: kernel { requires { field.read; field.write; } input source: Image <- gray.result; param size: Int = 4; output result: Image;
    compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }
output result <- show.result;
"#,
    // A trail fed back from the evaluation before.
    r#"node seed: kernel { requires { field.write; time.read; } output result: Image; domain = vec2(16, 16);
    compute { return vec4(fract(coord().x / 16.0 + time()), 0.0, 0.0, 1.0); } }
node show: kernel { requires { field.read; field.write; } input current: Image <- seed.result; input previous: Image <~ show.result from seed.result;
    param decay: Float = 0.9; output result: Image;
    compute { return sample(current) * 0.1 + sample(previous) * decay; } }
output result <- show.result;
"#,
    // Rows ranked by a prefix, smoothed by a scan, summed by a merge.
    r#"param level: Int = 3;
node ticks: kernel { requires { field.write; } param level: Int = level; output result: Rows<Int>; domain = 64; compute { return element() % level; } }
node ranks: kernel { requires { field.read; field.write; } input flags: Rows<Int> <- ticks.result; output result: Rows<Int>;
    prefix over flags with sum { return if sample(flags) != 0 { 1 } else { 0 }; } }
node smooth: kernel { requires { field.read; field.write; } input ranks: Rows<Int> <- ranks.result; param k: Float = 0.25; output result: Rows<Float>;
    scan over ranks with y = 0.0 { return y + (float(sample(ranks)) - y) * k; } }
node show: kernel { requires { field.read; field.write; } input values: Rows<Float> <- smooth.result; output result: Rows<Float>;
    compute over values { return sample(values); } merge(a, b) { return a + b; } }
output result <- show.result;
"#,
    // A machine stepping over a memory it carries.
    r#"kernel t.machine@1 {
    requires { field.read; field.write; }
    input ticks: Rows<Int>;
    memory ram: Rows<Int>;
    output result: Rows<Int>;
    scan over ticks with pc = 0 { let op = at(ram, pc); put(ram, pc + 100, op * 2); return pc + 1; }
}
node zeros: kernel { requires { field.write; } output result: Rows<Int>; domain = 256; compute { return element(); } }
node ticks: kernel { requires { field.write; } output result: Rows<Int>; domain = 8; compute { return 0; } }
node show: t.machine@1 { ticks <- ticks.result; ram <~ show.ram from zeros.result; }
output result <- show.result;
"#,
    // A loop, a match and an unrolled array, sized by the window.
    r#"param level: Int = 16;
node show: kernel { requires { field.write; time.read; input.read; } param steps: Int = level; output result: Image; domain = window();
    compute {
        let p = coord() / domain() - vec2(0.5, 0.5);
        let z = loop steps with z = vec2(0.0, 0.0) until dot(z, z) > 4.0 { vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + p };
        let shade = match int(time()) % 3 { 0 => 0.2, 1 | 2 => 0.6, _ => 1.0 };
        let taps = [each 4 { float(index()) * 0.25 }];
        return vec4(z.x, taps[2], shade, 1.0);
    } }
output result <- show.result;
"#,
    // Records carried from frame to frame.
    r#"node seed: kernel { requires { field.write; } struct Tally { count: Int; total: Float; } output result: Rows<Tally>; domain = 4;
    compute { return Tally { count: 0, total: 0.0 }; } }
node show: kernel { requires { field.read; field.write; } struct Tally { count: Int; total: Float; } input previous: Rows<Tally> <~ show.result from seed.result; output result: Rows<Tally>;
    compute { let t = sample(previous); return Tally { count: t.count + 1, total: t.total + float(element()) }; } }
output result <- show.result;
"#,
];

/// Every seed document, by name.
fn documents() -> Vec<(String, String)> {
    let mut found = vec![
        ("library.ent".to_owned(), LIBRARY.to_owned()),
        ("placed.esn".to_owned(), PLACED.to_owned()),
    ];
    for (k, scene) in SCENES.iter().enumerate() {
        found.push((format!("scene {k}"), (*scene).to_owned()));
    }
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
    // Every seed is a document the front end takes whole, and the
    // library's Kernels all publish.
    for (name, text) in &documents {
        if let Err(error) = entelechy_language::parse_document(text) {
            panic!("{name}: {error}");
        }
    }
    let dir =
        std::env::temp_dir().join(format!("entelechy-hostile-library-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("library.ent"), LIBRARY).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"library.ent\";\nnode source: kernel { requires { field.write; } output result: Image; domain = vec2(4, 4); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\nnode show: t.gray@1 { source <- source.result; }\noutput result <- show.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(
        events.contains(&entelechy_session::Event::SceneLoaded)
            && events
                .iter()
                .filter(|event| matches!(
                    event,
                    entelechy_session::Event::Published { id, .. } if id.name().starts_with("t.")
                ))
                .count()
                == 7,
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng::new(0x4057);
    let mut failures = Vec::new();
    for case in 0..count() {
        let (path, text) = &documents[rng.below(documents.len() as u64) as usize];
        let mut mutated = text.clone();
        for _ in 0..1 + rng.below(3) {
            mutated = mutate(&mut rng, &mutated);
        }
        let name = format!("case {case} from {path}");
        let parsed = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: parse"), move || {
            let _ = entelechy_language::parse_document(&parsed);
        }) {
            failures.push((failure, mutated.clone()));
            continue;
        }
        let compiled = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: compile"), move || {
            if let Ok(document) = entelechy_language::parse_document(&compiled) {
                for kernel in &document.kernels {
                    let _ = entelechy_compiler::compile_kernel(kernel, None);
                }
            }
        }) {
            failures.push((failure, mutated.clone()));
            continue;
        }
        let lowered = mutated.clone();
        if let Some(failure) = guarded(&format!("{name}: lower"), move || {
            let _ = entelechy_scene::parse(&lowered);
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
    let mut sessions = Vec::new();
    let mut scenes = Vec::new();
    for (k, text) in SCENES.iter().enumerate() {
        let dir =
            std::env::temp_dir().join(format!("entelechy-hostile-{}-{k}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let scene = dir.join("scene.esn");
        std::fs::write(&scene, text).unwrap();
        scenes.push((*text).to_owned());
        let runtime = Session::runtime_for(
            "jit",
            entelechy_session::Limits {
                words: 1 << 22,
                ..Default::default()
            },
        )
        .unwrap();
        let mut session = Session::open_with(&scene, None, runtime);
        session.restrict_reads().unwrap();
        let events = session.sync();
        assert!(
            events.contains(&entelechy_session::Event::SceneLoaded),
            "scene {k} loads: {events:?}"
        );
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
                let _ = entelechy_protocol::respond_line(session, &line);
            }) {
                failures.push(failure);
            }
        }
        // The session still answers.
        let (status, _) = entelechy_protocol::respond_line(session, r#"{"id":0,"op":"status"}"#);
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
