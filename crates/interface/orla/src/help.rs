//! `orla help`: the protocol's `help` ([`orla_protocol::help`]) with the
//! command line's own commands beside it, as text or, with `--json`, as
//! the protocol answers.

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

/// Every item named `name`, the protocol's and the commands'.
pub fn answer(name: &str) -> Result<Json, Json> {
    let all = commands::commands();
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
                "nothing named `{name}`{}",
                orla_language::suggestion(near)
            ));
        }
        return Err(refusal);
    }
    Ok(json!({ "name": name, "found": found }))
}

/// The whole reference ([`orla_protocol::help::reference`]) with the
/// command line's commands.
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

/// The items found, as a person reads them.
pub fn text(answer: &Json) -> String {
    let mut out = String::new();
    for (index, item) in answer["found"].as_array().into_iter().flatten().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&item_body(item));
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
