//! What help shows of the language, the graph and the authority is
//! proved by its canonical examples ([`orla_protocol::help::examples`]):
//! every item is shown by one, every item one claims is an item and is
//! written in it, and every example loads and runs (a package's
//! manifest where an example uses it).

use std::path::{Path, PathBuf};

use orla_kernel::Context;
use orla_session::Session;

/// The sections the examples prove.
const PROVED: [&str; 5] = ["types", "builtins", "kernel", "scene", "authority"];

fn help_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../protocol/help")
}

/// Every item of a proved section, by name.
fn items() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for section in PROVED {
        let answer = orla_protocol::help::section(section).unwrap();
        for item in answer["items"].as_array().unwrap() {
            out.push((
                section.to_owned(),
                item["name"].as_str().unwrap().to_owned(),
            ));
        }
    }
    out
}

/// Where an item is written in a source: a form by the text that writes
/// it, a call by its name and a parenthesis, anything else as a word.
fn written(text: &str, section: &str, name: &str) -> bool {
    // What is written is code, not what its comments say.
    let code: String = text
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(|line| format!("{line}\n"))
        .collect();
    let text = code.as_str();
    let word = |w: &str| {
        text.match_indices(w).any(|(at, _)| {
            let before = text[..at].chars().next_back();
            let after = text[at + w.len()..].chars().next();
            let apart = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
            apart(before) && apart(after)
        })
    };
    match (section, name) {
        ("kernel", "and") => text.contains("&&"),
        // Every operator the item lists, written between spaces.
        ("kernel", "operators") => orla_compiler::help::forms()
            .iter()
            .find(|form| form.name == "operators")
            .expect("the operators are a form")
            .shape
            .split_whitespace()
            .filter(|op| *op != "a" && *op != "b")
            .all(|op| text.contains(&format!(" {op} "))),
        // The reserved words are shown by the ones an example writes.
        ("kernel", "reserved") => orla_language::RESERVED.iter().any(|w| word(w)),
        ("kernel", "array") => text.contains("; 4]") || text.contains("[1.0"),
        ("kernel", "struct") => text.contains("struct "),
        ("kernel", "each") => text.contains("[each "),
        ("kernel", "merge") => text.contains("merge("),
        ("kernel", "scan") => text.contains("scan with"),
        ("kernel", "prefix") => text.contains("prefix with"),
        ("kernel", "scatter") => text.contains("scatter into"),
        ("scene", "inline") => text.contains(": kernel {"),
        ("scene", "fill") => ["Rows<", "Grid<", "Volume<"].iter().any(|shape| {
            text.lines()
                .any(|l| l.starts_with("node ") && l.contains(&format!(": {shape}")))
        }),
        ("scene", "graph") => text.contains("graph \""),
        ("scene", "previous") => text.contains("<~"),
        ("scene", "times") => text.contains("times ="),
        ("builtins", _) => text.contains(&format!("{name}(")),
        _ => word(name),
    }
}

#[test]
fn every_item_is_shown_by_an_example_that_writes_it() {
    let examples = orla_protocol::help::examples();
    let items = items();
    let mut wrong = Vec::new();
    for example in &examples {
        let shows = example.shows();
        assert!(!shows.is_empty(), "{} shows nothing", example.file);
        for name in shows {
            let sections: Vec<&str> = items
                .iter()
                .filter(|(_, n)| n == name)
                .map(|(s, _)| s.as_str())
                .collect();
            if sections.is_empty() {
                wrong.push(format!(
                    "{} shows `{name}`, which no proved section holds",
                    example.file
                ));
            } else if !sections.iter().any(|s| written(example.text, s, name)) {
                wrong.push(format!(
                    "{} shows `{name}` but does not write it",
                    example.file
                ));
            }
        }
    }
    for (section, name) in &items {
        if !examples.iter().any(|e| e.shows().contains(&name.as_str())) {
            wrong.push(format!("{section} `{name}` is shown by no example"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn every_example_loads_and_runs() {
    let context = Context {
        window: [64.0, 48.0],
        scale: 1.0,
        ..Context::default()
    };
    let mut wrong = Vec::new();
    let examples = orla_protocol::help::examples();
    for example in &examples {
        let path = help_dir().join(example.file);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            example.text,
            "{} is the file help carries",
            example.file
        );
        // A package runs where a scene uses it.
        if example.is_package() {
            let package = Path::new(example.file).parent().unwrap().to_string_lossy();
            if !examples
                .iter()
                .any(|e| !e.is_package() && e.text.contains(&format!("use \"{package}\"")))
            {
                wrong.push(format!("{} is used by no example", example.file));
            }
            continue;
        }
        let mut session = Session::open(&path, None, 1 << 24);
        let events = session.sync();
        if session.graph().is_none() {
            wrong.push(format!("{} does not load: {events:?}", example.file));
            continue;
        }
        if let Err(error) = session.evaluate(&context) {
            wrong.push(format!("{} does not run: {error}", example.file));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A copy of `help/` in a directory of its own: a scenario may rewrite
/// its scene.
fn help_copy(name: &str) -> PathBuf {
    fn tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                tree(&path, &to.join(entry.file_name()));
            } else {
                std::fs::copy(&path, to.join(entry.file_name())).unwrap();
            }
        }
    }
    let dir = std::env::temp_dir().join(format!("orla-help-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    tree(&help_dir(), &dir);
    dir
}

/// `"$n"` anywhere in `value` replaced by answer `n`.
fn filled(value: &serde_json::Value, answers: &[serde_json::Value]) -> serde_json::Value {
    use serde_json::Value as Json;
    match value {
        Json::String(text) => text
            .strip_prefix('$')
            .and_then(|n| n.parse::<usize>().ok())
            .and_then(|n| answers.get(n).cloned())
            .unwrap_or_else(|| value.clone()),
        Json::Array(items) => Json::Array(items.iter().map(|v| filled(v, answers)).collect()),
        Json::Object(fields) => Json::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), filled(v, answers)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Every operation but `quit` has one scenario; its last request is that
/// operation, and every request of it is answered, not refused, by a
/// session serving its scene.
#[test]
fn every_operation_is_shown_by_a_scenario_that_is_answered() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    let scenarios = orla_protocol::help::scenarios();
    let mut wrong = Vec::new();
    for op in orla_protocol::ops::names()
        .into_iter()
        .filter(|op| *op != "quit")
    {
        let mine: Vec<&serde_json::Value> = scenarios.iter().filter(|s| s["op"] == op).collect();
        if mine.len() != 1 {
            wrong.push(format!("`{op}` has {} scenarios", mine.len()));
        }
    }
    for scenario in &scenarios {
        let op = scenario["op"].as_str().unwrap();
        if scenario["send"]["op"] != op {
            wrong.push(format!(
                "`{op}`'s scenario sends {}",
                scenario["send"]["op"]
            ));
        }
        let dir = help_copy(op);
        let scene = dir.join(scenario["scene"].as_str().unwrap_or("scene.osn"));
        let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("serve")
            .arg(&scene)
            .args(["--window", "64x48"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let setup = scenario["setup"].as_array().cloned().unwrap_or_default();
        let mut answers: Vec<serde_json::Value> = Vec::new();
        for request in setup.iter().chain([&scenario["send"]]) {
            writeln!(stdin, "{}", filled(request, &answers)).unwrap();
            stdin.flush().unwrap();
            let answer: serde_json::Value =
                serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
            if answer.get("ok").is_none() {
                wrong.push(format!("`{op}`: {request} was refused: {answer}"));
                break;
            }
            answers.push(answer["ok"].clone());
        }
        drop(stdin);
        child.wait().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
