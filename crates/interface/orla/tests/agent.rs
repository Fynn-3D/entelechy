//! An agent at work, live: not a record played back but a client that
//! decides each request from the answer to the one before, through
//! `orla serve` and nothing else. It is given a goal, the running picture
//! inverted, and a first draft of a Kernel with the slips a writer makes.
//! It finds what the scene outputs, repairs the draft from the positioned
//! diagnostics and `help`'s `similar` alone, publishes against the
//! generation it expects, is refused where it should be and sees the
//! running Kernel survive, composes the Kernel into the scene, and holds
//! the picture to the goal.
//!
//! Any agent in any language speaks this; the test is written in Rust
//! because the workspace's tests are, and it touches no Rust API of
//! Orla's.

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

    /// One request, and its answer: `Ok` with what came back, or `Err`
    /// with the error object.
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

/// The word of `source` at a diagnostic's line and column, both from one.
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

/// Prepares `draft` until it compiles, each time taking the word the
/// diagnostic points at, asking `help` what it might have meant, and
/// putting the first suggestion in its place. Returns the source that
/// compiled and the repairs made, as `(wrong, right)`.
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

/// The Kernel the agent means to write, with the slips of a first draft:
/// a misspelt `sample` and a misspelt `clamp`, twice.
const DRAFT: &str = "\
// The picture with every colour turned to its opposite.
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

/// A copy of the grayscale example the agent may rewrite.
fn scene() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-agent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/grayscale");
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for entry in std::fs::read_dir(example.join("kernels"))
        .unwrap()
        .flatten()
    {
        std::fs::copy(entry.path(), dir.join("kernels").join(entry.file_name())).unwrap();
    }
    std::fs::copy(example.join("scene.osn"), dir.join("scene.osn")).unwrap();
    dir.join("scene.osn")
}

#[test]
fn an_agent_repairs_publishes_composes_and_meets_its_goal() {
    let scene = scene();
    let mut agent = Agent::start(&scene);

    // Look before touching: the versions it speaks, and what the scene
    // shows the person.
    let status = agent.ask("status", json!({})).unwrap();
    assert!(status["versions"]["api"].is_u64(), "{status}");
    let graph = agent.ask("graph", json!({})).unwrap();
    let output = graph["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|output| output["name"] == "result")
        .expect("the scene shows an output named `result`");
    let shown = output["source"].as_str().unwrap().to_owned();
    let before = agent.picture("result");

    // An operation it guessed at is answered with the ones there are.
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

    // A second writer expecting nothing is refused, and a broken
    // replacement leaves the running version where it was.
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

    // Compose it between what the scene showed and the person.
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

    // Hold the picture to the goal: every colour turned to its opposite.
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
