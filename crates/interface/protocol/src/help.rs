//! What Orla answers about itself, gathered from where each part is kept
//! and laid out as the architecture gives AI to create: the creation
//! surfaces (the Kernel Language, the Graph API, the Capability API) and
//! the doors that reach them. The builtins and the forms come from the
//! compiler, what a `domain` may call from the scene's lowering, what a
//! capability grants from the ABI, the replay forms from the replayer, the
//! operations from [`crate::ops`], the loop an agent works in from
//! [`crate::workflow`]. An index first, then a section's one line an item,
//! then one name in full.

use serde_json::{Value as Json, json};

use orla_compiler::help::{Builtin, Form};
use orla_kernel::Capability;
use orla_session::replay;

/// What the sections stand under: the creation surfaces, then the doors.
/// The Capability API is what the world grants (the `authority` section)
/// and what it already knows (the Kernels and preset graphs `search`
/// finds).
pub const UNDER: [(&str, &str); 4] = [
    ("language", "what a Kernel may write"),
    ("graph", "how a scene composes Kernels into a running world"),
    (
        "capabilities",
        "what the world grants a Kernel, and what it already knows (`search`)",
    ),
    ("doors", "how an agent reaches the surfaces"),
];

/// A canonical example: a whole scene the tests load and run, the help
/// items it shows (its `//! shows:` line), and its text. What help shows of
/// the language, the graph and the authority is proved by these.
pub struct Example {
    /// Its file, in `help/` beside this crate's sources.
    pub file: &'static str,
    /// Its text.
    pub text: &'static str,
}

impl Example {
    /// Whether it is a package's manifest, which a scene uses rather than
    /// runs.
    pub fn is_package(&self) -> bool {
        self.text.lines().any(|line| line.starts_with("package "))
    }

    /// The help items it shows.
    pub fn shows(&self) -> Vec<&'static str> {
        self.text
            .lines()
            .find_map(|line| line.strip_prefix("//! shows:"))
            .map(|names| names.split(',').map(str::trim).collect())
            .unwrap_or_default()
    }
}

/// The canonical examples, smallest first.
pub fn examples() -> Vec<Example> {
    macro_rules! example {
        ($file:literal) => {
            Example {
                file: $file,
                text: include_str!(concat!("../help/", $file)),
            }
        };
    }
    let mut all = vec![
        example!("scene.osn"),
        example!("values.osn"),
        example!("types.osn"),
        example!("math.osn"),
        example!("loops.osn"),
        example!("fields.osn"),
        example!("modes.osn"),
        example!("graph.osn"),
        example!("context.osn"),
        example!("tally.orlapkg/package.osn"),
    ];
    all.sort_by_key(|example| example.text.len());
    all
}

/// Each operation's scenario, as a test sends it: the scene it runs on
/// (`help/scene.osn` when not given), the requests that set it up, and the
/// request that shows the operation. A `"$n"` in a request stands for the
/// answer to set-up request `n`.
pub fn scenarios() -> Vec<Json> {
    include_str!("../help/protocol.jsonl")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("help/protocol.jsonl holds JSON lines"))
        .collect()
}

/// The sections, each under a surface or the doors, with what it holds.
const SECTIONS: [(&str, &str, &str); 9] = [
    (
        "types",
        "language",
        "the types a port, a param or a value has",
    ),
    ("builtins", "language", "what a Kernel may call"),
    (
        "kernel",
        "language",
        "how a Kernel is written: its declarations and expressions",
    ),
    (
        "scene",
        "graph",
        "how a scene is written: nodes, bindings, inputs and packages",
    ),
    (
        "authority",
        "capabilities",
        "the capabilities a Kernel declares in `requires`, each what it grants",
    ),
    (
        "packages",
        "capabilities",
        "what a package holds and how a scene reaches one; `search` finds what they hold",
    ),
    ("protocol", "doors", "the operations a session answers"),
    (
        "wire",
        "doors",
        "a request and an answer as they cross a door, and the doors",
    ),
    ("replay", "doors", "the lines of a replay test"),
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
    // The doors in the index's order: the protocol, its wire, replay.
    out.extend(replay::forms().iter().map(replay_json));
    out
}

/// The index: what the sections stand under, the sections and how many
/// items each holds, the loop an agent works in, the versions an answer is
/// good for, and how to ask further.
pub fn index() -> Json {
    let all = items();
    let under: Vec<Json> = UNDER
        .iter()
        .map(|(name, about)| json!({ "name": name, "about": about }))
        .collect();
    let sections: Vec<Json> = SECTIONS
        .iter()
        .map(|(name, under, about)| {
            json!({
                "section": name,
                "under": under,
                "about": about,
                "count": all.iter().filter(|item| item["section"] == *name).count(),
            })
        })
        .collect();
    let workflow: Vec<Json> = crate::workflow::WORKFLOW
        .iter()
        .map(|step| json!({ "step": step.name, "about": step.about, "ops": step.ops }))
        .collect();
    json!({
        "under": under,
        "sections": sections,
        "workflow": workflow,
        "versions": crate::inspect::versions(),
        "ask": "`help` with `section` for one line an item, with `name` for one item in full; the answer holds while `versions` does. `search` finds what the world already knows: Kernels and preset graphs, by words, types and capability",
    })
}

/// Everything help answers as one Markdown text, for an agent to read
/// once: the loop, then each section under its surface or door, an item a
/// line or two, then every canonical example whole, each once.
pub fn reference() -> String {
    let all = items();
    let text = |item: &Json, key: &str| item[key].as_str().unwrap_or_default().to_owned();
    let mut out = String::from(
        "# Orla\n\nKernels (`.orla`) compute; scenes (`.osn`) compose them into a running world.\n\n## The loop\n\n",
    );
    for step in &crate::workflow::WORKFLOW {
        let ops: Vec<String> = step.ops.iter().map(|op| format!("`{op}`")).collect();
        out.push_str(&format!(
            "- {} ({}): {}\n",
            step.name,
            ops.join(", "),
            step.about
        ));
    }
    for (under, about) in UNDER {
        out.push_str(&format!("\n## {under}: {about}\n"));
        for (section, _, what) in SECTIONS.iter().filter(|(_, u, _)| *u == under) {
            out.push_str(&format!("\n### {section}: {what}\n\n"));
            for item in all.iter().filter(|item| item["section"] == *section) {
                let name = text(item, "name");
                let line = match *section {
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
    }
    out.push_str(
        "\n## Examples\n\nWhole scenes the tests load and run; each says what it shows.\n",
    );
    for example in examples() {
        let language = example.file.rsplit('.').next().unwrap_or_default();
        out.push_str(&format!(
            "\n### {}\n\n```{language}\n{}```\n",
            example.file, example.text
        ));
    }
    out
}

/// One section: each item's name and one line.
pub fn section(name: &str) -> Result<Json, Json> {
    if !SECTIONS.iter().any(|(section, ..)| *section == name) {
        let names = SECTIONS.map(|(section, ..)| section);
        let near = orla_language::nearest(name, names);
        let next = near.map_or_else(
            || json!({ "op": "help" }),
            |near| json!({ "op": "help", "section": near }),
        );
        return Err(crate::wire::tried(
            json!({
                "message": format!(
                    "no section `{name}`; the sections are {}{}",
                    names.join(", "),
                    orla_language::suggestion(near)
                ),
                "sections": names,
            }),
            [next],
        ));
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
    // Each item with the smallest canonical example that shows it.
    let example = examples()
        .into_iter()
        .find(|example| example.shows().contains(&name));
    // A scene shown once, on the first item it proves.
    let mut shown = false;
    let found: Vec<Json> = all
        .iter()
        .filter(|item| item["name"] == name)
        .cloned()
        .map(|mut item| {
            let proved = matches!(
                item["section"].as_str(),
                Some("types" | "builtins" | "kernel" | "scene" | "authority")
            );
            if let (true, false, Some(example)) = (proved, shown, &example) {
                item["example"] = json!({ "file": example.file, "text": example.text });
                shown = true;
            }
            if item["section"] == "protocol"
                && let Some(scenario) = scenarios().into_iter().find(|s| s["op"] == name)
            {
                item["example"] = scenario;
            }
            item
        })
        .collect();
    if found.is_empty() {
        let mut names: Vec<&str> = all
            .iter()
            .filter_map(|item| item["name"].as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        let near = orla_language::nearest(name, names.iter().copied());
        let next = near.map_or_else(|| json!({ "op": "help" }), crate::wire::ask_help);
        return Err(crate::wire::tried(
            json!({
                "message": format!(
                    "nothing named `{name}`; `help` lists the sections{}",
                    orla_language::suggestion(near)
                ),
                "similar": near.into_iter().collect::<Vec<_>>(),
            }),
            [next],
        ));
    }
    Ok(json!({ "name": name, "found": found }))
}
