//! What agents found wrong stays fixed. Each file in `findings/` is a
//! session over a copy of an example: its first line names the example
//! (`{"scene": "examples/life"}`, with `"installed": true` when the copied
//! packages are the ones installed, and `"text"` to start from that text
//! in its place), and every other line sends a request
//! and may hold its answer to what was found wanting:
//!
//! - `"finding"`: what an agent met, as a sentence;
//! - `"send"`: the request; `{scene}` in a string is the example's scene
//!   text, as copied;
//! - `"expect"`: `ok` (answered, or refused), `at` (a JSON pointer into the
//!   answer and the value there), `has` (pointers to something not null
//!   and not empty), `lacks` (pointers to nothing), `says` and `not`
//!   (text the answer, or a string in it, holds or does not).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value as Json;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A copy of `example` (its scene and Kernels) at the place it has in the
/// repository, beside a copy of the packages, in a directory of its own;
/// the copied scene's path.
fn copy(example: &str, name: &str) -> PathBuf {
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
    let dir = std::env::temp_dir().join(format!("orla-findings-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    tree(&root().join("packages"), &dir.join("packages"));
    let at = dir.join(example);
    std::fs::create_dir_all(&at).unwrap();
    std::fs::copy(root().join(example).join("scene.osn"), at.join("scene.osn")).unwrap();
    if root().join(example).join("kernels").is_dir() {
        tree(&root().join(example).join("kernels"), &at.join("kernels"));
    }
    at.join("scene.osn")
}

/// `{scene}` in every string of `value` replaced by `scene`.
fn filled(value: &Json, scene: &str) -> Json {
    match value {
        Json::String(text) => Json::String(text.replace("{scene}", scene)),
        Json::Array(items) => Json::Array(items.iter().map(|v| filled(v, scene)).collect()),
        Json::Object(fields) => Json::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), filled(v, scene)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Each request answered by one `orla serve` of `scene`, in order;
/// `installed`, the packages it finds by name.
fn talk(scene: &Path, installed: Option<&Path>, requests: &[Json]) -> Vec<Json> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_orla"));
    if let Some(dir) = installed {
        command.env("ORLA_PACKAGES", dir);
    }
    let mut child = command
        .arg("serve")
        .arg(scene)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut answers = Vec::new();
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        let line = lines
            .next()
            .unwrap_or_else(|| panic!("`orla serve` answered nothing to {request}"))
            .unwrap();
        answers.push(serde_json::from_str(&line).unwrap());
    }
    drop(stdin);
    child.wait().unwrap();
    answers
}

/// What `answer` falls short of in `expect`, one line each.
fn wanting(answer: &Json, expect: &Json) -> Vec<String> {
    fn strings(value: &Json, out: &mut String) {
        match value {
            Json::String(text) => {
                out.push_str(text);
                out.push('\n');
            }
            Json::Array(items) => items.iter().for_each(|v| strings(v, out)),
            Json::Object(fields) => fields.values().for_each(|v| strings(v, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    // The answer as sent, and every string in it as written.
    let mut text = answer.to_string();
    strings(answer, &mut text);
    let empty = |v: &Json| {
        v.is_null()
            || v.as_str().is_some_and(str::is_empty)
            || v.as_array().is_some_and(Vec::is_empty)
            || v.as_object().is_some_and(serde_json::Map::is_empty)
    };
    if let Some(ok) = expect.get("ok").and_then(Json::as_bool)
        && answer.get("ok").is_some() != ok
    {
        out.push(format!(
            "expected {}",
            if ok { "an answer" } else { "a refusal" }
        ));
    }
    for (pointer, value) in expect
        .get("at")
        .and_then(Json::as_object)
        .into_iter()
        .flatten()
    {
        if answer.pointer(pointer) != Some(value) {
            out.push(format!(
                "{pointer} is {:?}, not {value}",
                answer.pointer(pointer)
            ));
        }
    }
    for pointer in expect
        .get("has")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let pointer = pointer.as_str().unwrap_or_default();
        if answer.pointer(pointer).is_none_or(empty) {
            out.push(format!("{pointer} is empty"));
        }
    }
    for pointer in expect
        .get("lacks")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let pointer = pointer.as_str().unwrap_or_default();
        if answer.pointer(pointer).is_some() {
            out.push(format!("{pointer} is there"));
        }
    }
    for words in expect
        .get("says")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let words = words.as_str().unwrap_or_default();
        if !text.contains(words) {
            out.push(format!("does not say {words:?}"));
        }
    }
    for words in expect
        .get("not")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let words = words.as_str().unwrap_or_default();
        if text.contains(words) {
            out.push(format!("says {words:?}"));
        }
    }
    out
}

#[test]
fn what_agents_found_wanting_stays_fixed() {
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(root().join("crates/interface/orla/tests/findings"))
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
    files.sort();
    assert!(!files.is_empty());
    let mut wrong = Vec::new();
    let mut held = 0;
    for file in &files {
        let name = file.file_stem().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(file).unwrap();
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let head: Json = serde_json::from_str(lines.next().unwrap()).unwrap();
        let example = head["scene"]
            .as_str()
            .expect("the first line names a scene");
        let copied = copy(example, &name);
        if let Some(text) = head["text"].as_str() {
            std::fs::write(&copied, text).unwrap();
        }
        let scene = std::fs::read_to_string(&copied).unwrap();
        let steps: Vec<Json> = lines
            .map(|line| {
                serde_json::from_str(line).unwrap_or_else(|e| panic!("{name}: {e}: {line}"))
            })
            .collect();
        let requests: Vec<Json> = steps
            .iter()
            .enumerate()
            .map(|(id, step)| {
                let mut request = filled(&step["send"], &scene);
                request["id"] = Json::from(id);
                request
            })
            .collect();
        let packages = std::env::temp_dir()
            .join(format!("orla-findings-{name}-{}", std::process::id()))
            .join("packages");
        let installed = head["installed"].as_bool() == Some(true);
        let answers = talk(&copied, installed.then_some(packages.as_path()), &requests);
        for (step, answer) in steps.iter().zip(&answers) {
            let Some(expect) = step.get("expect") else {
                continue;
            };
            held += 1;
            let short = wanting(answer, expect);
            if !short.is_empty() {
                wrong.push(format!(
                    "{name}: {}\n  {}\n  answer: {}",
                    step["finding"].as_str().unwrap_or("(no finding)"),
                    short.join("\n  "),
                    answer.to_string().chars().take(600).collect::<String>()
                ));
            }
        }
        let _ = std::fs::remove_dir_all(
            std::env::temp_dir().join(format!("orla-findings-{name}-{}", std::process::id())),
        );
    }
    assert!(held > 0);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
