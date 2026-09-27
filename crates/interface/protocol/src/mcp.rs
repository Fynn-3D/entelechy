//! The protocol as an MCP server on standard input and output: JSON-RPC
//! 2.0, one message a line. Each operation is a tool whose input schema
//! is its request's ([`crate::ops::Operation::schema`]); a call is the
//! operation's request, answered by the session as any other, its answer
//! the tool's result.

use serde_json::{Value as Json, json};

use crate::ops;

/// The MCP versions this server speaks, newest first.
const VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// What the server tells an agent it has just met: what Orla is, and the
/// loop it works in ([`crate::workflow`]), step by step.
fn instructions() -> String {
    format!(
        "Orla is a runtime where Kernels (`.orla`) compute and scenes (`.osn`) compose them into a running world; this session serves one scene. Work in its loop: {} `help` with no arguments answers an index of everything Orla answers; `help` with `section` or `name` goes further.",
        crate::workflow::text()
    )
}

/// What one line from the client comes to.
pub enum Step {
    /// A message answered here at once: its response line.
    Answer(String),
    /// A notification, which is not answered.
    Quiet,
    /// A tool call: the operation's request, answered by the session, and
    /// the id its result goes back under.
    Call {
        /// The JSON-RPC id.
        id: Json,
        /// The protocol request.
        request: Json,
    },
}

fn response(id: &Json, result: Json) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
}

fn error(id: &Json, code: i64, message: &str) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }).to_string()
}

/// Every operation but `quit` as a tool: the client ends the session.
pub fn tools() -> Json {
    let tools: Vec<Json> = ops::operations()
        .iter()
        .filter(|op| op.name != "quit")
        .map(|op| {
            json!({
                "name": op.name,
                "description": format!("{} Answers {}.", op.summary, op.answer),
                "inputSchema": op.schema(),
            })
        })
        .collect();
    json!({ "tools": tools })
}

/// Reads one line from the client.
pub fn step(line: &str) -> Step {
    let Ok(message) = serde_json::from_str::<Json>(line) else {
        return Step::Answer(error(&Json::Null, -32700, "not JSON"));
    };
    let id = message.get("id").cloned();
    let method = message
        .get("method")
        .and_then(Json::as_str)
        .unwrap_or_default();
    let Some(id) = id else {
        // A notification (`notifications/initialized` and the like).
        return Step::Quiet;
    };
    let params = message.get("params").cloned().unwrap_or(json!({}));
    match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Json::as_str);
            let version = asked
                .filter(|v| VERSIONS.contains(v))
                .unwrap_or(VERSIONS[0]);
            Step::Answer(response(
                &id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "orla", "version": env!("CARGO_PKG_VERSION") },
                    "instructions": instructions(),
                }),
            ))
        }
        "ping" => Step::Answer(response(&id, json!({}))),
        "tools/list" => Step::Answer(response(&id, tools())),
        "tools/call" => {
            let name = params
                .get("name")
                .and_then(Json::as_str)
                .unwrap_or_default();
            if name == "quit" || ops::operation(name).is_none() {
                return Step::Answer(error(
                    &id,
                    -32602,
                    &format!("no tool `{name}`; `tools/list` names them"),
                ));
            }
            let mut request = match params.get("arguments") {
                Some(Json::Object(arguments)) => Json::Object(arguments.clone()),
                None | Some(Json::Null) => json!({}),
                Some(_) => {
                    return Step::Answer(error(&id, -32602, "a tool's `arguments` is an object"));
                }
            };
            request["op"] = json!(name);
            Step::Call { id, request }
        }
        other => Step::Answer(error(&id, -32601, &format!("no method `{other}`"))),
    }
}

/// The tool result for a protocol answer line (`{"id", "ok"}` or
/// `{"id", "error"}`), under `id`: the answer as text, and as structured
/// content when it is an object, a PNG picture as an image; a refusal is
/// an error result, so the agent reads why.
pub fn result(id: &Json, answer: &str) -> String {
    let answer: Json = serde_json::from_str(answer).unwrap_or(Json::Null);
    let (value, failed) = match (answer.get("ok"), answer.get("error")) {
        (Some(ok), _) => (ok.clone(), false),
        (None, Some(error)) => (error.clone(), true),
        (None, None) => (answer, true),
    };
    // A picture answered as PNG is shown as the picture; the rest of the
    // answer stays text.
    let mut value = value;
    let mut content = Vec::new();
    if value["format"] == "png"
        && let Some(Json::String(data)) = value.as_object_mut().and_then(|o| o.remove("data"))
    {
        content.push(json!({ "type": "image", "data": data, "mimeType": "image/png" }));
    }
    content.insert(0, json!({ "type": "text", "text": value.to_string() }));
    let mut result = json!({ "content": content, "isError": failed });
    if value.is_object() {
        result["structuredContent"] = value;
    }
    response(id, result)
}
