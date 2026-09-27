//! `orla help`: the protocol's `help` ([`orla_protocol::help`]) with the
//! command line's own section beside it, as text or, with `--json`, as the
//! protocol answers.

use serde_json::{Value as Json, json};

use crate::commands;

const COMMANDS: &str = "commands";

fn command_json(command: &commands::Command) -> Json {
    json!({
        "section": COMMANDS,
        "name": command.name,
        "synopsis": format!("{} {}", command.name, command.args).trim_end(),
        "summary": command.summary,
        "doc": command.doc,
        "flags": command.flags.iter().map(|flag| json!({
            "name": flag.name,
            "value": flag.value,
            "repeats": flag.repeats,
            "doc": flag.doc,
        })).collect::<Vec<_>>(),
    })
}

/// The answer to `orla help [<section> | <name>]`: the index, a section
/// (the protocol's or `commands`), or every item of a name.
pub fn answer(asked: Option<&str>) -> Result<Json, Json> {
    let all = commands::commands();
    match asked {
        None => {
            let mut index = orla_protocol::help::index();
            if let Some(sections) = index["sections"].as_array_mut() {
                sections.push(json!({
                    "section": COMMANDS,
                    "under": "doors",
                    "about": "the orla command line",
                    "count": all.len(),
                }));
            }
            index["ask"] = json!(
                "`orla help <section>` for one line an item, `orla help <name>` for one item in full, `--json` for the protocol's answer"
            );
            Ok(index)
        }
        Some(COMMANDS) => Ok(json!({
            "section": COMMANDS,
            "items": all.iter().map(|c| json!({ "name": c.name, "line": c.summary })).collect::<Vec<_>>(),
        })),
        Some(name) => {
            // A section's name may also name items (`replay`, the command
            // and the section): the answer is both.
            if let Ok(mut section) = orla_protocol::help::section(name) {
                let found: Vec<Json> = all
                    .iter()
                    .filter(|c| c.name == name)
                    .map(command_json)
                    .collect();
                if !found.is_empty() {
                    section["found"] = json!(found);
                }
                return Ok(section);
            }
            let mut found: Vec<Json> = match orla_protocol::help::named(name) {
                Ok(named) => named["found"].as_array().cloned().unwrap_or_default(),
                Err(_) => Vec::new(),
            };
            found.extend(all.iter().filter(|c| c.name == name).map(command_json));
            if found.is_empty() {
                // The protocol's refusal names the near miss among its own.
                let mut refusal = orla_protocol::help::named(name).unwrap_err();
                if refusal["similar"].as_array().is_none_or(Vec::is_empty) {
                    let near = orla_language::nearest(name, all.iter().map(|c| c.name));
                    refusal["similar"] = json!(near.into_iter().collect::<Vec<_>>());
                    refusal["message"] = json!(format!(
                        "nothing named `{name}`; `orla help` lists the sections{}",
                        orla_language::suggestion(near)
                    ));
                }
                return Err(refusal);
            }
            Ok(json!({ "name": name, "found": found }))
        }
    }
}

/// The whole reference ([`orla_protocol::help::reference`]) with the
/// command line's commands beside the protocol's doors.
pub fn full() -> String {
    let mut out = orla_protocol::help::reference();
    out.push_str("\n## commands: the orla command line\n\n");
    for command in commands::commands() {
        out.push_str(&format!(
            "- **{}**: `orla {} {}` {}\n",
            command.name, command.name, command.args, command.summary
        ));
    }
    out
}

/// The answer as a person reads it.
pub fn text(answer: &Json) -> String {
    let mut out = String::new();
    if let Some(sections) = answer["sections"].as_array() {
        let versions = &answer["versions"];
        out.push_str(&format!(
            "orla {} (api {}, language {}, ir {}, package {})\n",
            versions["orla"].as_str().unwrap_or_default(),
            versions["api"],
            versions["language"],
            versions["ir"],
            versions["package"]
        ));
        // The sections under each surface, then under the doors.
        for under in answer["under"].as_array().into_iter().flatten() {
            let name = under["name"].as_str().unwrap_or_default();
            out.push_str(&format!(
                "{name}: {}\n",
                under["about"].as_str().unwrap_or_default()
            ));
            for section in sections.iter().filter(|s| s["under"] == name) {
                out.push_str(&format!(
                    "  {:<10} {:>4}  {}\n",
                    section["section"].as_str().unwrap_or_default(),
                    section["count"],
                    section["about"].as_str().unwrap_or_default()
                ));
            }
        }
        out.push_str("the loop:\n");
        for step in answer["workflow"].as_array().into_iter().flatten() {
            let ops: Vec<&str> = step["ops"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Json::as_str)
                .collect();
            out.push_str(&format!(
                "  {:<10} {}\n",
                step["step"].as_str().unwrap_or_default(),
                ops.join(", ")
            ));
        }
        out.push_str(answer["ask"].as_str().unwrap_or_default());
        out.push('\n');
    } else if answer["items"].is_array() {
        out.push_str(&items_text(answer));
        if let Some(found) = answer["found"].as_array() {
            for item in found {
                out.push('\n');
                out.push_str(&item_text(item));
            }
        }
    } else if let Some(found) = answer["found"].as_array() {
        for (index, item) in found.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            out.push_str(&item_text(item));
        }
    }
    out
}

/// A section's items, one line each.
fn items_text(answer: &Json) -> String {
    let mut out = String::new();
    if let Some(items) = answer["items"].as_array() {
        let width = items
            .iter()
            .filter_map(|item| item["name"].as_str())
            .map(str::len)
            .max()
            .unwrap_or(0);
        for item in items {
            out.push_str(&format!(
                "  {:<width$}  {}\n",
                item["name"].as_str().unwrap_or_default(),
                item["line"].as_str().unwrap_or_default()
            ));
        }
    }
    out
}

fn item_text(item: &Json) -> String {
    let mut out = item_body(item);
    let Some(example) = item.get("example") else {
        return out;
    };
    if let Some(send) = example.get("send") {
        // An operation's scenario: what sets it up, then the request.
        out.push_str(&format!(
            "e.g. on help/{}:\n",
            example["scene"].as_str().unwrap_or("scene.osn")
        ));
        for request in example["setup"].as_array().into_iter().flatten() {
            out.push_str(&format!("  {request}\n"));
        }
        out.push_str(&format!("  {send}\n"));
    } else {
        out.push_str(&format!(
            "Shown in {}:\n",
            example["file"].as_str().unwrap_or_default()
        ));
        for line in example["text"].as_str().unwrap_or_default().lines() {
            out.push_str(&format!("  {line}\n"));
        }
    }
    out
}

fn item_body(item: &Json) -> String {
    let text = |key: &str| item[key].as_str().unwrap_or_default().to_owned();
    let mut out = String::new();
    match item["section"].as_str().unwrap_or_default() {
        "builtins" => {
            out.push_str(&format!("{} ({})\n", text("name"), text("kind")));
            for signature in item["signatures"].as_array().into_iter().flatten() {
                out.push_str(&format!("  {}\n", signature.as_str().unwrap_or_default()));
            }
            out.push_str(&format!("{}\n", text("doc")));
            if let Some(capability) = item["requires"].as_str() {
                out.push_str(&format!(
                    "Needs `{capability}` in the Kernel's `requires`.\n"
                ));
            }
            if item["entry_only"] == true {
                out.push_str("Only the entry (`compute`, `scan`, `scatter`) may call it; a function takes its value as an argument.\n");
            }
        }
        "replay" => {
            out.push_str(&format!(
                "{} (replay)\n  {}\n{}\n",
                text("name"),
                text("shape"),
                text("doc")
            ));
            for example in item["examples"].as_array().into_iter().flatten() {
                out.push_str(&format!(
                    "  e.g. {}\n",
                    example.as_str().unwrap_or_default()
                ));
            }
        }
        "protocol" => {
            out.push_str(&format!(
                "{} (protocol)\n{}\n",
                text("name"),
                text("summary")
            ));
            for field in item["fields"].as_array().into_iter().flatten() {
                out.push_str(&format!(
                    "  {}{}: {}  {}\n",
                    field["name"].as_str().unwrap_or_default(),
                    if field["required"] == true { "" } else { "?" },
                    field["type"].as_str().unwrap_or_default(),
                    field["doc"].as_str().unwrap_or_default()
                ));
            }
            out.push_str(&format!("Answers {}.\n", text("answer")));
        }
        "authority" => {
            out.push_str(&format!("{} (authority)\n{}\n", text("name"), text("doc")));
            let by: Vec<&str> = item["required_by"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Json::as_str)
                .collect();
            if !by.is_empty() {
                out.push_str(&format!("Needed to call {}.\n", by.join(", ")));
            }
        }
        COMMANDS => {
            out.push_str(&format!(
                "orla {} (command)\n{}\n",
                text("synopsis"),
                text("summary")
            ));
            for flag in item["flags"].as_array().into_iter().flatten() {
                out.push_str(&format!(
                    "  {}{}  {}\n",
                    flag["name"].as_str().unwrap_or_default(),
                    flag["value"]
                        .as_str()
                        .map(|v| format!(" {v}"))
                        .unwrap_or_default(),
                    flag["doc"].as_str().unwrap_or_default()
                ));
            }
        }
        section => {
            out.push_str(&format!(
                "{} ({section}, {})\n  {}\n{}\n",
                text("name"),
                text("kind"),
                text("shape"),
                text("doc")
            ));
        }
    }
    out
}
