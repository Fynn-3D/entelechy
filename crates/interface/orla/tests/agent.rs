//! Live agents: clients choosing each request from the last answer,
//! through `orla serve` alone and no Rust API of Orla's.
//!
//! One repairs a draft with typos from diagnostics and `help`, publishes,
//! composes it into the scene, and holds the picture to its inverse.
//!
//! The other changes a running state's layout: a first migration with the
//! wrong layout is refused and the state kept; repaired from the state's
//! layout, the count goes on.

use std::io::{BufRead, BufReader, Lines, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value as Json, json};

/// A client of one served session.
struct Agent {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    next: u64,
}

impl Agent {
    fn start(scene: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
            .args(["serve", "--backend", "cpu"])
            .arg(scene)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Self {
            child,
            stdin,
            lines,
            next: 1,
        }
    }

    /// One request: `Ok` with its answer, or `Err` with the error object.
    fn ask(&mut self, op: &str, fields: Json) -> Result<Json, Json> {
        let mut request = fields;
        request["op"] = json!(op);
        request["id"] = json!(self.next);
        self.next += 1;
        writeln!(self.stdin, "{request}").unwrap();
        self.stdin.flush().unwrap();
        let line = self
            .lines
            .next()
            .unwrap_or_else(|| panic!("`orla serve` stopped answering at {request}"))
            .unwrap();
        let answer: Json = serde_json::from_str(&line).unwrap();
        assert_eq!(answer["id"], request["id"], "answers come in order");
        match answer.get("ok") {
            Some(ok) => Ok(ok.clone()),
            None => Err(answer["error"].clone()),
        }
    }

    /// A node's current layout, as a file spells it, and its words.
    fn state(&mut self, node: &str) -> (String, Vec<u32>) {
        let read = self
            .ask("read", json!({ "node": node, "format": "words" }))
            .expect("the node's state reads");
        let bytes = orla_formats::base64::unbase64(read["data"].as_str().unwrap()).unwrap();
        let words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        (read["schema"].as_str().unwrap().to_owned(), words)
    }

    /// The scene's output `name` as RGBA floats.
    fn picture(&mut self, name: &str) -> Vec<[f32; 4]> {
        self.ask("evaluate", json!({ "time": 0.0 }))
            .expect("the scene evaluates");
        let read = self
            .ask("read", json!({ "output": name, "format": "rgba32f" }))
            .expect("the output reads");
        let bytes = orla_formats::base64::unbase64(read["data"].as_str().unwrap()).unwrap();
        bytes
            .chunks_exact(16)
            .map(|pixel| {
                std::array::from_fn(|k| {
                    f32::from_le_bytes(pixel[k * 4..k * 4 + 4].try_into().unwrap())
                })
            })
            .collect()
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{}", json!({ "op": "quit" }));
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

/// The word of `source` at a diagnostic's 1-based line and column.
fn word_at(source: &str, line: u64, column: u64) -> (usize, String) {
    let start: usize = source
        .lines()
        .take(line as usize - 1)
        .map(|line| line.len() + 1)
        .sum::<usize>()
        + column as usize
        - 1;
    let word: String = source[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (start, word)
}

/// Prepares `draft` until it compiles, replacing the word each diagnostic
/// points at with `help`'s first suggestion. Returns the compiled source
/// and the repairs as `(wrong, right)`.
fn repair(agent: &mut Agent, draft: &str) -> (String, Vec<(String, String)>) {
    let mut source = draft.to_owned();
    let mut repairs = Vec::new();
    for _ in 0..8 {
        let error = match agent.ask("prepare", json!({ "source": source })) {
            Ok(prepared) => {
                assert!(prepared["id"].is_string(), "{prepared}");
                return (source, repairs);
            }
            Err(error) => error,
        };
        let (Some(line), Some(column)) = (error["line"].as_u64(), error["column"].as_u64()) else {
            panic!("a compile error the agent cannot place: {error}");
        };
        let (at, word) = word_at(&source, line, column);
        let message = error["message"].as_str().unwrap();
        assert!(
            message.contains(&format!("`{word}`")),
            "{line}:{column} points at `{word}`, and the diagnostic is about something else: {message}"
        );
        let help = agent
            .ask("help", json!({ "name": word }))
            .expect_err("a word the compiler does not know is not in `help` either");
        let right = help["similar"][0]
            .as_str()
            .unwrap_or_else(|| panic!("`help` suggests nothing for `{word}`: {help}"))
            .to_owned();
        source.replace_range(at..at + word.len(), &right);
        repairs.push((word, right));
    }
    panic!("the draft did not compile after eight repairs: {repairs:?}");
}

/// The intended Kernel with first-draft typos: a misspelt `sample` and a
/// misspelt `clamp`, twice.
const DRAFT: &str = "\
// The picture with every color turned to its opposite.
kernel agent.invert@1 {
    requires { field.read; field.write; }
    input source: Image;
    output result: Image;
    compute {
        let c = sampel(source);
        return vec4(clmap(1.0 - c.r, 0.0, 1.0), clmap(1.0 - c.g, 0.0, 1.0), 1.0 - c.b, c.a);
    }
}
";

/// A gradient, turned gray, in blocks of eight texels.
const GRAYSCALE: &[(&str, &str)] = &[
    (
        "scene.osn",
        "use \"kernels/\";\n\
         node source: image.gradient@1 { domain = vec2(32, 32); }\n\
         node gray: image.grayscale@1 { source <- source.result; amount = 1.0; }\n\
         node blocks: image.pixelate@1 { source <- gray.result; size = 8; }\n\
         output result <- blocks.result;\n",
    ),
    (
        "kernels/image.orla",
        "kernel image.gradient@1 {\n\
         \x20   requires { field.write; }\n\
         \x20   output result: Image;\n\
         \x20   compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.5, 1.0); }\n\
         }\n\
         kernel image.grayscale@1 {\n\
         \x20   requires { field.read; field.write; }\n\
         \x20   input source: Image;\n\
         \x20   param amount: Float = 1.0;\n\
         \x20   output result: Image;\n\
         \x20   compute { let c = sample(source); let g = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return mix(c, vec4(g, g, g, c.a), amount); }\n\
         }\n\
         kernel image.pixelate@1 {\n\
         \x20   requires { field.read; field.write; }\n\
         \x20   input source: Image;\n\
         \x20   param size: Int = 8;\n\
         \x20   output result: Image;\n\
         \x20   compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); }\n\
         }\n",
    ),
];

/// A tally counting evaluations, its state fed back through the graph.
const TALLY: &[(&str, &str)] = &[
    (
        "scene.osn",
        "use \"kernels/\";\n\
         node seed: tally.zero@1 { domain = 1; }\n\
         node count: tally.step@1 { previous <~ count.result from seed.result; }\n\
         output result <- count.result;\n",
    ),
    (
        "kernels/tally.orla",
        "struct Tally { count: Int; }\n\
         \n\
         kernel tally.zero@1 {\n\
         \x20   requires { field.write; }\n\
         \x20   output result: Rows<Tally>;\n\
         \x20   compute { return Tally { count: 0 }; }\n\
         }\n\
         \n\
         kernel tally.step@1 {\n\
         \x20   requires { field.read; field.write; }\n\
         \x20   input previous: Rows<Tally>;\n\
         \x20   output result: Rows<Tally>;\n\
         \x20   compute {\n\
         \x20       let t = sample(previous);\n\
         \x20       return Tally { count: t.count + 1 };\n\
         \x20   }\n\
         }\n",
    ),
];

/// `files` written into a directory of their own, for the agent to
/// rewrite; the path of its scene.
fn scene(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-agent-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for (path, text) in files {
        std::fs::write(dir.join(path), text).unwrap();
    }
    dir.join("scene.osn")
}

#[test]
fn an_agent_repairs_publishes_composes_and_meets_its_goal() {
    let scene = scene("grayscale", GRAYSCALE);
    let mut agent = Agent::start(&scene);

    // The versions it speaks, and what the scene shows.
    let status = agent.ask("status", json!({})).unwrap();
    assert!(status["versions"]["api"].is_u64(), "{status}");
    assert_eq!(
        status["versions"]["orla"],
        env!("CARGO_PKG_VERSION"),
        "{status}"
    );
    let graph = agent.ask("graph", json!({})).unwrap();
    let output = graph["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|output| output["name"] == "result")
        .expect("the scene shows an output named `result`");
    let shown = output["source"].as_str().unwrap().to_owned();
    let before = agent.picture("result");

    // An unknown operation is answered with the ones there are.
    let guess = agent.ask("compile", json!({})).unwrap_err();
    assert!(
        guess["operations"]
            .as_array()
            .unwrap()
            .contains(&json!("prepare")),
        "{guess}"
    );

    // Propose, and repair from the answers alone.
    let (source, repairs) = repair(&mut agent, DRAFT);
    assert_eq!(
        repairs,
        [
            ("sampel".to_owned(), "sample".to_owned()),
            ("clmap".to_owned(), "clamp".to_owned()),
            ("clmap".to_owned(), "clamp".to_owned()),
        ],
        "each slip repaired once, in the order the compiler met them"
    );

    // Publish where nothing is, then learn what is running.
    let published = agent
        .ask("publish", json!({ "source": source, "expect": "absent" }))
        .unwrap();
    let generation = published["generation"].as_u64().unwrap();

    // A second writer expecting nothing is refused; a broken replacement
    // leaves the running version.
    agent
        .ask("publish", json!({ "source": source, "expect": "absent" }))
        .expect_err("publishing over a Kernel that exists, expecting none, is refused");
    let broken = source.replace("clamp(1.0 - c.r", "clamp(1.0 - c.nope");
    agent
        .ask("publish", json!({ "source": broken, "expect": generation }))
        .expect_err("a replacement that does not compile is refused");
    let kernels = agent.ask("kernels", json!({})).unwrap();
    let running = kernels["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|kernel| kernel["id"] == "agent.invert@1")
        .expect("the published Kernel is running");
    assert_eq!(running["generation"].as_u64(), Some(generation));

    // Compose it between the scene's picture and its output.
    let document = agent.ask("document", json!({})).unwrap();
    let text = document["text"].as_str().unwrap();
    let line = format!("output result <- {shown};");
    assert!(
        text.contains(&line),
        "the document spells its output as the graph says"
    );
    let composed = text.replace(
        &line,
        &format!(
            "node invert: agent.invert@1 {{\n    source <- {shown};\n}}\n\noutput result <- invert.result;"
        ),
    );
    let events = agent.ask("compose", json!({ "text": composed })).unwrap();
    let kinds: Vec<&str> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|event| event["kind"].as_str())
        .collect();
    assert!(
        kinds.contains(&"scene_loaded") && !kinds.iter().any(|kind| kind.contains("rejected")),
        "{events}"
    );

    // The goal: every color turned to its opposite.
    let after = agent.picture("result");
    assert_eq!(after.len(), before.len());
    let mut changed = false;
    for (was, is) in before.iter().zip(&after) {
        for k in 0..3 {
            assert!(
                (is[k] - (1.0 - was[k]).clamp(0.0, 1.0)).abs() < 1e-5,
                "{was:?} became {is:?}"
            );
        }
        assert_eq!(is[3], was[3]);
        changed |= is != was;
    }
    assert!(changed, "the picture did not change");
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// A Kernel type for one scalar of a layout as a file spells it.
fn scalar_type(letter: &str) -> &'static str {
    match letter {
        "i" => "Int",
        "b" => "Bool",
        _ => "Float",
    }
}

/// A migration from the believed layout, `fields` of `(name, type)`, to
/// the same fields and a `doubled` Int.
fn widen(fields: &[(String, String)], generation: &str) -> String {
    let old: Vec<String> = fields
        .iter()
        .map(|(name, ty)| format!("{name}: {ty};"))
        .collect();
    let kept: Vec<String> = fields
        .iter()
        .map(|(name, ty)| match ty.as_str() {
            "Int" => format!("{name}: o.{name}"),
            _ => format!("{name}: int(o.{name})"),
        })
        .collect();
    let first = &fields[0].0;
    let doubled = match fields[0].1.as_str() {
        "Int" => format!("o.{first} * 2"),
        _ => format!("int(o.{first}) * 2"),
    };
    format!(
        "// The state as it was, carried into the state as it is: every field\n\
         // kept, and `doubled` made from the first. ({generation})\n\
         kernel agent.widen@1 {{\n    requires {{ field.read; field.write; }}\n    \
         struct Old {{ {} }}\n    struct Tally {{ count: Int; doubled: Int; }}\n    \
         input old: Rows<Old>;\n    output result: Rows<Tally>;\n    \
         compute {{ let o = sample(old); return Tally {{ {}, doubled: {doubled} }}; }}\n}}\n",
        old.join(" "),
        kept.join(", ")
    )
}

#[test]
fn an_agent_changes_what_a_running_state_carries_and_keeps_it() {
    let scene = scene("migrate", TALLY);
    let mut agent = Agent::start(&scene);

    // The state: a node whose port reads what it committed before.
    let graph = agent.ask("graph", json!({})).unwrap();
    let (node, port, kernel) = graph["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|node| {
            node["inputs"]
                .as_object()?
                .iter()
                .find_map(|(port, binding)| {
                    (binding["kind"] == "previous").then(|| {
                        (
                            node["id"].as_str().unwrap().to_owned(),
                            port.clone(),
                            node["kernel"].as_str().unwrap().to_owned(),
                        )
                    })
                })
        })
        .expect("the scene carries state");

    // The state's fields, by name, from tracing one element.
    let manifest = agent.ask("kernel", json!({ "kernel": kernel })).unwrap();
    let element = manifest["outputs"][0]["type"]
        .as_str()
        .unwrap()
        .trim_start_matches("Rows<")
        .trim_end_matches('>')
        .to_owned();
    let trace = agent
        .ask("trace", json!({ "node": node, "element": 0, "time": 0.0 }))
        .unwrap();
    let result = trace["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["type"] == element)
        .expect("the trace shows the state's element");
    let names: Vec<String> = result["value"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(names, ["count"], "{result}");

    // Run frame by frame until the count passes four.
    for _ in 0..100 {
        agent.ask("evaluate", json!({ "time": 0.0 })).unwrap();
        if agent.state(&node).1[0] > 4 {
            break;
        }
    }
    let (layout, words) = agent.state(&node);
    assert_eq!((layout.as_str(), words.as_slice()), ("i", &[5][..]));

    // The new step, the running source edited, carries `doubled`.
    let step = manifest["source"].as_str().unwrap();
    let counted = "Tally { count: t.count + 1 }";
    assert!(
        step.contains(counted),
        "the step counts as the agent read it: {step}"
    );
    let step = step
        .replacen(
            "requires { field.read; field.write; }",
            "requires { field.read; field.write; }\n    struct Tally { count: Int; doubled: Int; }",
            1,
        )
        .replace(
            counted,
            "Tally { count: t.count + 1, doubled: (t.count + 1) * 2 }",
        );

    // The first draft types every field as Float, as the trace's JSON
    // shows every number alike.
    let guessed: Vec<(String, String)> = names
        .iter()
        .map(|name| (name.clone(), "Float".to_owned()))
        .collect();
    let first = agent
        .ask(
            "publish",
            json!({ "source": widen(&guessed, "first draft"), "expect": "absent" }),
        )
        .unwrap();
    agent
        .ask("publish", json!({ "source": step, "expect": "any" }))
        .unwrap();
    let document = agent.ask("document", json!({})).unwrap();
    let text = document["text"].as_str().unwrap();
    let binding = format!("{port} <~ {node}.result from seed.result;");
    assert!(text.contains(&binding), "{text}");
    let composed = text.replace(
        &binding,
        &format!("{port} <~ {node}.result from seed.result migrate agent.widen@1;"),
    );
    agent.ask("compose", json!({ "text": composed })).unwrap();

    // The evaluation refuses the draft, naming state and migration, and
    // keeps the state.
    let refused = agent
        .ask("evaluate", json!({ "time": 1.0 }))
        .expect_err("a migration that takes the wrong layout fails the evaluation");
    assert_eq!(refused["node"], node.as_str(), "{refused}");
    assert_eq!(refused["port"], port.as_str(), "{refused}");
    assert_eq!(refused["migration"], "agent.widen@1", "{refused}");
    let (layout, words) = agent.state(&node);
    assert_eq!(
        (layout.as_str(), words.as_slice()),
        ("i", &[5][..]),
        "a refused migration leaves the state as it was"
    );

    // Repair from the state's layout.
    let typed: Vec<(String, String)> = names
        .iter()
        .zip(layout.split(','))
        .map(|(name, letter)| (name.clone(), scalar_type(letter).to_owned()))
        .collect();
    agent
        .ask(
            "publish",
            json!({ "source": widen(&typed, "typed from the state"), "expect": first["generation"] }),
        )
        .unwrap();
    let migrated = agent.ask("evaluate", json!({ "time": 1.0 })).unwrap();
    assert!(
        migrated["migrated"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["node"] == node.as_str() && entry["kernel"] == "agent.widen@1"),
        "{migrated}"
    );

    // The count went on from where it was, and carries its double.
    let (layout, words) = agent.state(&node);
    assert_eq!((layout.as_str(), words.as_slice()), ("i,i", &[6, 12][..]));
    for _ in 0..3 {
        agent.ask("evaluate", json!({ "time": 0.0 })).unwrap();
    }
    let (_, words) = agent.state(&node);
    assert_eq!(words, [9, 18]);
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}
