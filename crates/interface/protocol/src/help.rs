//! Orla's syntax and API, layer by layer, gathered from the tables the
//! implementation executes: the builtins and the forms from the compiler,
//! the reserved words from the parser, what a `domain` may call from the
//! scene's lowering, what a capability grants from the ABI, a package's
//! parts from the session, the operations from [`crate::ops`], the wire
//! from [`crate::envelope`], the replay lines from the replayer. The whole
//! of it as one text, or one name in full.

use serde_json::{Value as Json, json};

use orla_compiler::help::{Builtin, Form};
use orla_kernel::Capability;
use orla_session::replay;

/// The sections in the order the reference gives them, each with what it
/// holds.
const SECTIONS: [(&str, &str); 9] = [
    ("types", "the types a port, a param or a value has"),
    ("builtins", "what a Kernel may call"),
    (
        "kernel",
        "how a Kernel is written: its declarations and expressions",
    ),
    (
        "scene",
        "how a scene is written: nodes, bindings, inputs and packages",
    ),
    (
        "authority",
        "the capabilities a Kernel declares in `requires`",
    ),
    (
        "packages",
        "what a package holds and how a scene reaches one",
    ),
    ("protocol", "the operations a session answers"),
    (
        "wire",
        "a request and an answer as they cross a door, and the doors",
    ),
    ("replay", "the lines of a replay test"),
];

fn builtin_json(builtin: &Builtin) -> Json {
    json!({
        "section": "builtins",
        "name": builtin.name,
        "kind": builtin.kind,
        "signatures": builtin.signatures,
        "doc": builtin.doc,
        "requires": builtin.requires.map(Capability::name),
        "entry_only": builtin.entry_only,
    })
}

/// The section a form is written in.
fn form_section(form: &Form) -> &'static str {
    match form.kind {
        "type" => "types",
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

fn authority_json(capability: Capability, builtins: &[Builtin]) -> Json {
    json!({
        "section": "authority",
        "name": capability.name(),
        "doc": capability.doc(),
        "required_by": builtins
            .iter()
            .filter(|b| b.requires == Some(capability))
            .map(|b| b.name)
            .collect::<Vec<_>>(),
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
    let builtins = orla_compiler::help::builtins();
    let mut out: Vec<Json> = builtins.iter().map(builtin_json).collect();
    out.extend(
        Capability::ALL
            .iter()
            .map(|capability| authority_json(*capability, &builtins)),
    );
    out.extend(orla_compiler::help::forms().iter().map(form_json));
    // The parser's own table of the words no name may be.
    out.push(json!({
        "section": "kernel",
        "name": "reserved",
        "kind": "document",
        "shape": orla_language::RESERVED.join(" "),
        "doc": "The words the language keeps for itself: a Kernel, node, port, param, function, struct or binding may not be named one.",
    }));
    out.extend(orla_scene::domain::functions().iter().map(|f| {
        json!({
            "section": "scene",
            "name": f.name,
            "kind": "domain",
            "shape": f.shape,
            "doc": f.doc,
            "examples": [format!("domain = {};", f.example)],
        })
    }));
    out.extend(orla_session::package::LAYOUT.iter().map(|part| {
        json!({
            "section": "packages",
            "name": part.name,
            "kind": "package",
            "shape": part.shape,
            "doc": part.doc,
        })
    }));
    out.extend(crate::ops::operations().iter().map(op_json));
    out.extend(crate::envelope::PARTS.iter().map(|part| {
        json!({
            "section": "wire",
            "name": part.name,
            "kind": part.within,
            "shape": format!("{}: `{}`: {}", part.within, part.name, part.ty),
            "doc": part.doc,
        })
    }));
    out.extend(crate::envelope::DOORS.iter().map(|door| {
        json!({
            "section": "wire",
            "name": door.name,
            "kind": "door",
            "shape": door.opened,
            "doc": door.doc,
        })
    }));
    out.extend(replay::forms().iter().map(replay_json));
    out
}

/// The whole of it as one Markdown text: each section, an item a line or
/// two.
pub fn reference() -> String {
    let all = items();
    let text = |item: &Json, key: &str| item[key].as_str().unwrap_or_default().to_owned();
    let mut out = String::from(
        "# Orla\n\nKernels (`.orla`) compute; scenes (`.osn`) compose them into a running world. Whole programs are in `examples/`.\n",
    );
    for (section, what) in SECTIONS {
        out.push_str(&format!("\n## {section}: {what}\n\n"));
        for item in all.iter().filter(|item| item["section"] == section) {
            let name = text(item, "name");
            let line = match section {
                "builtins" => {
                    let signatures: Vec<&str> = item["signatures"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Json::as_str)
                        .collect();
                    let needs = item["requires"]
                        .as_str()
                        .map(|c| format!(" Needs `{c}`."))
                        .unwrap_or_default();
                    format!("`{}` {}{needs}", signatures.join("`; `"), text(item, "doc"))
                }
                "protocol" => {
                    let fields: Vec<String> = item["fields"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|f| {
                            format!(
                                "`{}`{}: {}",
                                f["name"].as_str().unwrap_or_default(),
                                if f["required"] == true { "" } else { "?" },
                                f["type"].as_str().unwrap_or_default()
                            )
                        })
                        .collect();
                    format!(
                        "`{name}` {{{}}} {} Answers {}.",
                        fields.join(", "),
                        text(item, "summary"),
                        text(item, "answer")
                    )
                }
                "authority" => format!("`{name}` {}", text(item, "doc")),
                _ => format!("`{}` {}", text(item, "shape"), text(item, "doc")),
            };
            out.push_str(&format!("- **{name}**: {line}\n"));
        }
    }
    out
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
            "message": format!("nothing named `{name}`{}", orla_language::suggestion(near)),
            "similar": near.into_iter().collect::<Vec<_>>(),
        }));
    }
    Ok(json!({ "name": name, "found": found }))
}
