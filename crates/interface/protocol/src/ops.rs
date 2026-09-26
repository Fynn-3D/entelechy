//! Every operation the protocol answers, as data: what it does, the fields
//! it takes and what it answers. The server checks each request against
//! it, `help` answers from it, and `docs/PROTOCOL.md`'s table is its
//! rendering.

use serde_json::{Value as Json, json};

use crate::wire::failure;

/// One operation.
#[derive(Clone, Debug)]
pub struct Operation {
    /// Its name, the request's `op`.
    pub name: &'static str,
    /// What it does, in one sentence.
    pub summary: &'static str,
    /// The fields it takes, beside `op` and `id`.
    pub fields: Vec<Field>,
    /// What it answers.
    pub answer: &'static str,
}

/// One field of a request.
#[derive(Clone, Debug)]
pub struct Field {
    /// Its name.
    pub name: &'static str,
    /// The JSON it holds.
    pub ty: &'static str,
    /// Whether a request must carry it.
    pub required: bool,
    /// What it means.
    pub doc: &'static str,
}

fn needs(name: &'static str, ty: &'static str, doc: &'static str) -> Field {
    Field {
        name,
        ty,
        required: true,
        doc,
    }
}

fn may(name: &'static str, ty: &'static str, doc: &'static str) -> Field {
    Field {
        name,
        ty,
        required: false,
        doc,
    }
}

/// The context an evaluation runs under, which every operation that
/// evaluates takes.
fn context() -> Vec<Field> {
    vec![
        may(
            "time",
            "number",
            "session time in seconds; 0 when not given",
        ),
        may(
            "rate",
            "number",
            "the sound clock, frames a second, which Kernels read as rate()",
        ),
        may(
            "pointer",
            "[x, y]",
            "the pointer in pixels from the window's top left",
        ),
        may("down", "bool", "whether the pointer is pressed"),
        may(
            "buttons",
            "[integer]",
            "buttons held: 0 left, 1 right, 2 middle, then a pad's",
        ),
        may("wheel", "[x, y]", "the scroll for this frame"),
        may(
            "sticks",
            "[[x, y]]",
            "a pad's left stick, right stick and triggers, up to three pairs",
        ),
        may("keys", "[integer]", "keys held, by the browser's codes"),
        may(
            "window",
            "[w, h]",
            "the window in pixels; else the session's",
        ),
        may(
            "scale",
            "number",
            "the window's scale factor, with `window`; 1 when not given",
        ),
    ]
}

fn with_context(mut fields: Vec<Field>) -> Vec<Field> {
    fields.extend(context());
    fields
}

fn op(
    name: &'static str,
    summary: &'static str,
    fields: Vec<Field>,
    answer: &'static str,
) -> Operation {
    Operation {
        name,
        summary,
        fields,
        answer,
    }
}

/// Every operation, in the order an agent meets them.
pub fn operations() -> Vec<Operation> {
    vec![
        op(
            "status",
            "The session at a glance.",
            vec![],
            "backend, scene, watched, kernels (published Kernels), natives (the backend's native operations), nodes, revision, frame, fusion, grants, versions (`orla`, `api`, `language`, `ir`, `package`)",
        ),
        op(
            "kernels",
            "Every published Kernel, with its generation.",
            vec![],
            "a list of manifests, each with its generation, which `publish`'s `expect` names",
        ),
        op(
            "kernel",
            "One Kernel's manifest and source.",
            vec![
                may(
                    "kernel",
                    "string",
                    "the Kernel's id, `name@version`; else `id` names it",
                ),
                may(
                    "generation",
                    "integer",
                    "one the session remembers; the current when not given",
                ),
            ],
            "manifest, generation, source, as `publish` takes it",
        ),
        op(
            "graph",
            "The running scene's structure.",
            vec![],
            "inputs, params, nodes, outputs",
        ),
        op(
            "document",
            "The scene document's text.",
            vec![may(
                "version",
                "string",
                "one the session remembers; the current when not given",
            )],
            "path, text, and version, which `compose`'s `expect` names",
        ),
        op(
            "history",
            "What changed the world, and what was refused.",
            vec![may(
                "since",
                "integer",
                "a revision; everything when not given",
            )],
            "revision, frame, the changes after `since` and the attempts refused since, each with its origin",
        ),
        op(
            "compose",
            "Changes the scene document, checked before it is written; answered once the scene stands.",
            vec![
                may("text", "string", "the whole document; or `edits`"),
                may(
                    "edits",
                    "[{statement | member, text}]",
                    "each names a statement, or a node's `member`, with its new `text` or null to remove it",
                ),
                may(
                    "expect",
                    "string",
                    "the version `document` answered; refused when the scene changed since",
                ),
            ],
            "events, version",
        ),
        op(
            "sync",
            "Reads every changed document again.",
            vec![],
            "events",
        ),
        op(
            "prepare",
            "Compiles a Kernel and says what is wrong with it; publishes nothing.",
            vec![needs(
                "source",
                "string",
                "a document with one Kernel; it inherits the scene's declarations",
            )],
            "id, manifest; or an error at a line and column",
        ),
        op(
            "publish",
            "Compiles, validates and prepares a Kernel, then swaps it in whole; answered once it stands. The running world changes, not a file: the document it came from is left as it is.",
            vec![
                needs("source", "string", "a document with one Kernel"),
                may(
                    "expect",
                    "\"absent\" | \"any\" | integer",
                    "`absent` for a new Kernel, a generation `kernels` answered, or `any`; `any` when not given",
                ),
            ],
            "id, generation",
        ),
        op(
            "retire",
            "Removes a published Kernel.",
            vec![may(
                "kernel",
                "string",
                "the Kernel's id; else `id` names it",
            )],
            "retired",
        ),
        op(
            "param",
            "Sets a parameter the scene declares.",
            vec![
                needs("name", "string", "a param the scene declares"),
                needs(
                    "value",
                    "value",
                    "a literal of the param's type: a number, a bool, or an array",
                ),
            ],
            "name, value, events; a node's own value is refused, and changes by `compose`",
        ),
        op(
            "evaluate",
            "Runs one frame.",
            with_context(vec![
                may(
                    "nodes",
                    "[string]",
                    "evaluate only these nodes, the state and never-produced outputs",
                ),
                may("settle", "bool", "first wait for every `async` run to land"),
            ]),
            "executed, reused, committed, outputs",
        ),
        op(
            "advance",
            "Runs many frames, until a count or a value says to stop.",
            with_context(vec![
                needs("frames", "integer", "the most frames to run"),
                may(
                    "fps",
                    "number",
                    "the clock's frames a second; 60 when not given",
                ),
                may(
                    "until",
                    "{target, below | above | equals}",
                    "stops when `node.port[i].field` passes the bound",
                ),
                may(
                    "save",
                    "{node.port: path}",
                    "files written after the last frame, where `--allow file.write` allows",
                ),
            ]),
            "frames, stopped, target, value, saved, millis, report",
        ),
        op(
            "bench",
            "Times the scene's frames.",
            with_context(vec![
                may("frames", "integer", "frames a run; 30 when not given"),
                may("fps", "number", "the clock's frames a second"),
                may(
                    "repeat",
                    "integer",
                    "runs, the least kept; 3 when not given",
                ),
                may("nodes", "bool", "time every node too"),
            ]),
            "backend, frames, repeat, best_ms, median_ms, nodes, ran",
        ),
        op(
            "window",
            "Sets the window a headless session reports to evaluations that name none.",
            vec![
                may("size", "[w, h] | null", "in pixels; null for none"),
                may("scale", "number", "the scale factor; 1 when not given"),
            ],
            "the window",
        ),
        op(
            "read",
            "Reads an output or a node's port.",
            vec![
                may(
                    "output",
                    "string",
                    "a scene output; the first when neither it nor `node` is given",
                ),
                may("node", "string", "a node, with `port`"),
                may("port", "string", "the node's port; `result` when not given"),
                may(
                    "format",
                    "\"rgba32f\" | \"ppm\" | \"field\"",
                    "a picture's encoding, or `field` for any Field as the Host's file",
                ),
            ],
            "width, height, format, data (base64); words for a Field that is not pixels",
        ),
        op(
            "write",
            "Replaces a node's committed state.",
            vec![
                needs("node", "string", "the node"),
                may(
                    "port",
                    "string",
                    "its port, read by a `<~` binding; `result` when not given",
                ),
                needs(
                    "data",
                    "string",
                    "base64 words, or a `field` file with `format`",
                ),
                may(
                    "format",
                    "\"words\" | \"field\"",
                    "how `data` is written; words when not given",
                ),
            ],
            "the state replaced",
        ),
        op(
            "snapshot",
            "The whole world: its Kernels, its scene and every committed state.",
            vec![may(
                "to",
                "string",
                "a file to write it to, where `--allow file.write` allows",
            )],
            "the world's revision and frame, every Kernel with its generation and source, every port's committed state; or the file written",
        ),
        op(
            "restore",
            "Puts a snapshot's world back, after a trial frame that commits nothing.",
            with_context(vec![
                may("snapshot", "object", "as `snapshot` gave it; or `from`"),
                may("from", "string", "a file `snapshot` wrote"),
                needs(
                    "laws",
                    "\"current\" | \"snapshot\"",
                    "which Kernels the restored world runs under",
                ),
                may(
                    "structure",
                    "\"current\" | \"snapshot\"",
                    "which scene; `current` when not given",
                ),
            ]),
            "revision, frame, report",
        ),
        op(
            "reset",
            "Forgets committed state; `<~` readers start from their initial sources.",
            vec![
                may("node", "string", "one node; every node when not given"),
                may("port", "string", "one port of it; `result` when not given"),
            ],
            "what was reset",
        ),
        op(
            "help",
            "What Orla answers: this index, a section, or one item by name.",
            vec![
                may("section", "string", "a section the index names"),
                may(
                    "name",
                    "string",
                    "one builtin, form, operation or replay step",
                ),
            ],
            "the index; a section's items; or one item in full. An unknown name is an error with `similar`",
        ),
        op(
            "search",
            "The capabilities that exist: published Kernels, and every Kernel and preset graph of the packages this session reaches.",
            vec![
                may(
                    "text",
                    "string",
                    "words in a name, package, doc or port name, any case",
                ),
                may(
                    "takes",
                    "string",
                    "a type one of its inputs takes, as written: `Rows<Int>`",
                ),
                may("gives", "string", "a type one of its outputs gives"),
                may("requires", "string", "a capability it requires"),
                may("limit", "integer", "the most to answer; 20 when not given"),
            ],
            "capabilities, nearest first, each with kind, name, doc, package, reach (what a scene writes to reach it), inputs, params, outputs (name, type, doc), requires, granted, published; and count, how many matched",
        ),
        op(
            "trace",
            "Runs one element of a node's Kernel and shows every step of it.",
            with_context(vec![
                needs("node", "string", "the node"),
                may(
                    "at",
                    "[x, y, z]",
                    "the element in a grid or a volume; or `element`",
                ),
                may(
                    "element",
                    "integer",
                    "the element in rows, or a scan's step",
                ),
            ]),
            "kernel, at, bindings, results, calls",
        ),
        op(
            "pin",
            "Keeps a node materialized so `read` can see it.",
            vec![needs("node", "string", "the node")],
            "pinned",
        ),
        op(
            "unpin",
            "Lets fusion have a pinned node back.",
            vec![needs("node", "string", "the node")],
            "unpinned",
        ),
        op(
            "quit",
            "Ends the connection, or on standard input the process.",
            vec![],
            "bye",
        ),
    ]
}

/// The operation named `name`.
pub fn operation(name: &str) -> Option<Operation> {
    operations().into_iter().find(|op| op.name == name)
}

/// Every operation's name, in order.
pub fn names() -> Vec<&'static str> {
    operations().iter().map(|op| op.name).collect()
}

/// Refuses a request that carries a field its operation does not take, or
/// lacks one it needs, naming the fields it takes. An unknown operation is
/// left to [`crate::handle`].
pub(crate) fn check(request: &Json) -> Result<(), Json> {
    let Some(object) = request.as_object() else {
        return Err(failure("a request is a JSON object"));
    };
    let Some(op) = object.get("op").and_then(Json::as_str).and_then(operation) else {
        return Ok(());
    };
    let takes: Vec<&str> = op.fields.iter().map(|field| field.name).collect();
    let listed = || {
        if takes.is_empty() {
            "no fields".to_owned()
        } else {
            takes
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };
    for key in object.keys() {
        if key == "op" || key == "id" || takes.contains(&key.as_str()) {
            continue;
        }
        let near = orla_language::nearest(key, takes.iter().copied());
        return Err(json!({
            "message": format!(
                "`{}` takes {}; `{key}` is not one of them{}",
                op.name,
                listed(),
                orla_language::suggestion(near)
            ),
            "fields": takes,
            "similar": near.into_iter().collect::<Vec<_>>(),
        }));
    }
    if let Some(missing) = op
        .fields
        .iter()
        .find(|field| field.required && !object.contains_key(field.name))
    {
        return Err(json!({
            "message": format!("`{}` needs `{}` ({}); it takes {}", op.name, missing.name, missing.doc, listed()),
            "fields": takes,
        }));
    }
    Ok(())
}

impl Operation {
    /// The operation in full, as `help` answers it.
    pub fn json(&self) -> Json {
        json!({
            "op": self.name,
            "summary": self.summary,
            "fields": self.fields.iter().map(|field| json!({
                "name": field.name,
                "type": field.ty,
                "required": field.required,
                "doc": field.doc,
            })).collect::<Vec<_>>(),
            "answer": self.answer,
        })
    }
}

/// The operations as `docs/PROTOCOL.md` shows them: one row each.
pub fn table() -> String {
    let mut out = String::from("| op | fields | answer |\n| --- | --- | --- |\n");
    for op in operations() {
        let fields = op
            .fields
            .iter()
            .map(|field| {
                format!(
                    "`{}`{}: {}",
                    field.name,
                    if field.required { "" } else { "?" },
                    field.ty.replace('|', "\\|")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "| `{}` | {} | {} |\n",
            op.name,
            fields,
            op.answer.replace('|', "\\|")
        ));
    }
    out
}
