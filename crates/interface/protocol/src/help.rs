//! What Orla answers about itself, gathered from where each part is kept:
//! the builtins and the forms from the compiler, the replay forms from the
//! replayer, the operations from [`crate::ops`]. An index first, then a
//! section's one line an item, then one name in full.

use serde_json::{Value as Json, json};

use orla_compiler::help::{Builtin, Form};
use orla_session::replay;

/// The sections, each with what it holds.
const SECTIONS: [(&str, &str); 5] = [
    ("builtins", "what a Kernel may call"),
    (
        "kernel",
        "how a Kernel is written: its declarations and expressions",
    ),
    (
        "scene",
        "how a scene is written: nodes, bindings, inputs and packages",
    ),
    ("replay", "the lines of a replay test"),
    ("protocol", "the operations a session answers"),
];

fn builtin_json(builtin: &Builtin) -> Json {
    json!({
        "section": "builtins",
        "name": builtin.name,
        "kind": builtin.kind,
        "signatures": builtin.signatures,
        "doc": builtin.doc,
        "requires": builtin.requires,
        "entry_only": builtin.entry_only,
    })
}

/// The section a form is written in.
fn form_section(form: &Form) -> &'static str {
    match form.kind {
        "graph" | "document" => "scene",
        _ => "kernel",
    }
}

fn form_json(form: &Form) -> Json {
    json!({
        "section": form_section(form),
        "name": form.name,
        "kind": form.kind,
        "shape": form.shape,
        "doc": form.doc,
    })
}

fn replay_json(form: &replay::Form) -> Json {
    json!({
        "section": "replay",
        "name": form.keyword,
        "shape": form.shape,
        "doc": form.doc,
        "examples": form.examples,
    })
}

fn op_json(op: &crate::ops::Operation) -> Json {
    let mut entry = op.json();
    entry["section"] = json!("protocol");
    entry["name"] = json!(op.name);
    entry
}

/// Every item of every section, in full.
fn items() -> Vec<Json> {
    let mut out: Vec<Json> = orla_compiler::help::builtins()
        .iter()
        .map(builtin_json)
        .collect();
    out.extend(orla_compiler::help::forms().iter().map(form_json));
    out.extend(replay::forms().iter().map(replay_json));
    out.extend(crate::ops::operations().iter().map(op_json));
    out
}

/// The index: the sections, how many items each holds, the versions an
/// answer is good for, and how to ask further.
pub fn index() -> Json {
    let all = items();
    let sections: Vec<Json> = SECTIONS
        .iter()
        .map(|(name, about)| {
            json!({
                "section": name,
                "about": about,
                "count": all.iter().filter(|item| item["section"] == *name).count(),
            })
        })
        .collect();
    json!({
        "sections": sections,
        "versions": crate::inspect::versions(),
        "ask": "`help` with `section` for one line an item, with `name` for one item in full; the answer holds while `versions` does. `search` finds the capabilities that exist: Kernels and preset graphs, by words, types and capability",
    })
}

/// One section: each item's name and one line.
pub fn section(name: &str) -> Result<Json, Json> {
    if !SECTIONS.iter().any(|(section, _)| *section == name) {
        let names = SECTIONS.map(|(section, _)| section);
        return Err(json!({
            "message": format!(
                "no section `{name}`; the sections are {}{}",
                names.join(", "),
                orla_language::did_you_mean(name, names)
            ),
            "sections": names,
        }));
    }
    let items: Vec<Json> = items()
        .into_iter()
        .filter(|item| item["section"] == name)
        .map(|item| {
            let line = item
                .get("summary")
                .or_else(|| item.get("shape"))
                .or_else(|| item.get("doc"))
                .cloned()
                .unwrap_or(Json::Null);
            json!({ "name": item["name"], "line": line })
        })
        .collect();
    Ok(json!({ "section": name, "items": items }))
}

/// Every item named `name`, in full: a name may be a builtin, a form, a
/// replay step and an operation at once (`window`, `reset`).
pub fn named(name: &str) -> Result<Json, Json> {
    let all = items();
    let found: Vec<Json> = all
        .iter()
        .filter(|item| item["name"] == name)
        .cloned()
        .collect();
    if found.is_empty() {
        let mut names: Vec<&str> = all
            .iter()
            .filter_map(|item| item["name"].as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        let near = orla_language::nearest(name, names.iter().copied());
        return Err(json!({
            "message": format!(
                "nothing named `{name}`; `help` lists the sections{}",
                orla_language::suggestion(near)
            ),
            "similar": near.into_iter().collect::<Vec<_>>(),
        }));
    }
    Ok(json!({ "name": name, "found": found }))
}
