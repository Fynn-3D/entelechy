//! One request and one answer as they cross a door: a request is one
//! JSON object a line, an answer one line back. What each carries, and
//! the doors a session is reached through, are the table [`PARTS`] and
//! [`DOORS`]; [`parse_request`] reads the one and [`envelope`] writes the
//! other; `help` answers them.

use serde_json::{Value as Json, json};

use crate::wire::failure;

/// One part of a request, an answer or an error.
pub struct Part {
    /// Its key.
    pub name: &'static str,
    /// Where it stands: `request`, `answer` or `error`.
    pub within: &'static str,
    /// Its type, as the operations' fields spell one.
    pub ty: &'static str,
    /// What it says, one sentence.
    pub doc: &'static str,
}

const fn part(
    name: &'static str,
    within: &'static str,
    ty: &'static str,
    doc: &'static str,
) -> Part {
    Part {
        name,
        within,
        ty,
        doc,
    }
}

/// Every part of the envelope: what a request carries beside its
/// operation's fields, what an answer carries, and what an error may.
pub const PARTS: [Part; 22] = [
    part(
        "op",
        "request",
        "string",
        "The operation: one of the protocol section's, each with its own fields.",
    ),
    part(
        "id",
        "request",
        "value",
        "Anything; the answer carries it back as it was sent, to pair them.",
    ),
    part(
        "id",
        "answer",
        "value",
        "The request's `id`, or null when it sent none or was not JSON.",
    ),
    part(
        "ok",
        "answer",
        "object",
        "What the operation answered, as its entry in the protocol section says.",
    ),
    part(
        "error",
        "answer",
        "object",
        "Why the request was refused, and what acts on it as data; the world is as it was.",
    ),
    part(
        "message",
        "error",
        "string",
        "Why, one or more sentences for a person or an agent to read.",
    ),
    part(
        "line",
        "error",
        "integer",
        "Where the fault is, from 1: in the text the request carried, or in `path`.",
    ),
    part("column", "error", "integer", "Where on that line, from 1."),
    part(
        "similar",
        "error",
        "[string]",
        "The names nearest an unknown one.",
    ),
    part(
        "fix",
        "error",
        "object",
        "The change that mends the fault, when one does: an item of `compose`'s `documents` that takes its document's place, handed back as it is; `path` with `text`, the refused change's own document with a `use` item added, or with `edits` adding it to the document on disk. A `rejected` event carries it too.",
    ),
    part(
        "fields",
        "error",
        "[string]",
        "The fields the operation takes.",
    ),
    part("field", "error", "string", "The field at fault."),
    part("expected", "error", "string", "The type that field is."),
    part(
        "operations",
        "error",
        "[string]",
        "Every operation, for an unknown one.",
    ),
    part(
        "version",
        "error",
        "string",
        "The scene document's version as it stands, for a compose refused.",
    ),
    part(
        "path",
        "error",
        "string",
        "The scene document a refused compose would have written, or the document a failed element's operation is written in.",
    ),
    part(
        "written",
        "error",
        "bool",
        "Whether a compose wrote its document: false when refused.",
    ),
    part(
        "text",
        "error",
        "string",
        "The document an `edits` compose made, which `line` counts in.",
    ),
    part(
        "node",
        "error",
        "string",
        "The node an evaluation failed at.",
    ),
    part(
        "port",
        "error",
        "string",
        "The state port whose migration failed.",
    ),
    part(
        "migration",
        "error",
        "string",
        "The Kernel that failed to migrate that state.",
    ),
    part(
        "frame",
        "error",
        "integer",
        "The frame a run failed at, counted from its first.",
    ),
];

/// A way a session is reached.
pub struct Door {
    /// Its name.
    pub name: &'static str,
    /// How it is opened.
    pub opened: &'static str,
    /// What crosses it, one sentence.
    pub doc: &'static str,
}

/// Every door of a served session; all answer the same operations.
pub const DOORS: [Door; 3] = [
    Door {
        name: "stdin",
        opened: "entelechy serve <scene.esn>",
        doc: "Request lines on standard input, answer lines on standard output; `quit` ends the process.",
    },
    Door {
        name: "socket",
        opened: "entelechy serve <scene.esn> --listen <host:port>",
        doc: "The same lines over each TCP connection to the address; `quit` ends the connection.",
    },
    Door {
        name: "websocket",
        opened: "entelechy serve <scene.esn> --listen <host:port>",
        doc: "The same lines as WebSocket text messages to the same address, one request a message.",
    },
];

/// Parses one request line into its echoed id and the request, or an
/// error envelope's payload.
pub fn parse_request(line: &str) -> Result<(Json, Json), Json> {
    let request = serde_json::from_str::<Json>(line)
        .map_err(|error| failure(format!("request is not JSON: {error}")))?;
    let id = request.get("id").cloned().unwrap_or(Json::Null);
    Ok((id, request))
}

/// The response line for a result.
pub fn envelope(id: &Json, response: Result<Json, Json>) -> String {
    match response {
        Ok(ok) => json!({ "id": id, "ok": ok }).to_string(),
        Err(error) => json!({ "id": id, "error": error }).to_string(),
    }
}

/// The envelope as a Markdown table, as the documents render it.
pub fn table() -> String {
    let mut out = String::from("| part | in | type | says |\n| --- | --- | --- | --- |\n");
    for part in &PARTS {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            part.name, part.within, part.ty, part.doc
        ));
    }
    out.push_str("\n| door | opened by | carries |\n| --- | --- | --- |\n");
    for door in &DOORS {
        out.push_str(&format!(
            "| {} | `{}` | {} |\n",
            door.name, door.opened, door.doc
        ));
    }
    out
}
