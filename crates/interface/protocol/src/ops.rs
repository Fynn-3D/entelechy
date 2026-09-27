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
    /// The JSON it holds, which the server checks.
    pub ty: Ty,
    /// Whether a request must carry it.
    pub required: bool,
    /// What it means.
    pub doc: &'static str,
}

/// The JSON a field holds: what the server checks a request against,
/// what `help` shows and what a schema says.
#[derive(Clone, Debug)]
pub enum Ty {
    /// A string.
    Str,
    /// A whole number.
    Int,
    /// A number.
    Num,
    /// true or false.
    Bool,
    /// null.
    Null,
    /// Any value: a param's literal, which the operation reads by the
    /// param's own type.
    Value,
    /// Any object, as an operation answered it: a snapshot.
    Object,
    /// One of these strings.
    Words(&'static [&'static str]),
    /// An array of numbers, one a name: `[x, y]`.
    Numbers(&'static [&'static str]),
    /// An array of values of one type.
    List(Box<Ty>),
    /// An object of these fields and no others.
    Record(Vec<Field>),
    /// An object whose every key (the name says what one is) holds a
    /// value of one type.
    Map(&'static str, Box<Ty>),
    /// One of these.
    Either(Vec<Ty>),
}

impl Ty {
    /// As `help` and the protocol document show it.
    pub fn describe(&self) -> String {
        match self {
            Self::Str => "string".into(),
            Self::Int => "integer".into(),
            Self::Num => "number".into(),
            Self::Bool => "bool".into(),
            Self::Null => "null".into(),
            Self::Value => "value".into(),
            Self::Object => "object".into(),
            Self::Words(words) => words
                .iter()
                .map(|w| format!("\"{w}\""))
                .collect::<Vec<_>>()
                .join(" | "),
            Self::Numbers(names) => format!("[{}]", names.join(", ")),
            Self::List(item) => format!("[{}]", item.describe()),
            Self::Record(fields) => format!(
                "{{{}}}",
                fields
                    .iter()
                    .map(|f| format!(
                        "{}{}: {}",
                        f.name,
                        if f.required { "" } else { "?" },
                        f.ty.describe()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Map(key, value) => format!("{{{key}: {}}}", value.describe()),
            Self::Either(tys) => tys.iter().map(Ty::describe).collect::<Vec<_>>().join(" | "),
        }
    }

    /// Whether `value` is of this type; else what is wrong, naming where
    /// in it by `at`.
    pub fn check(&self, value: &Json, at: &str) -> Result<(), String> {
        let is = |ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(format!("`{at}` is {}, not {value}", self.describe()))
            }
        };
        match self {
            Self::Str => is(value.is_string()),
            Self::Int => is(value.as_i64().is_some() || value.as_u64().is_some()),
            Self::Num => is(value.is_number()),
            Self::Bool => is(value.is_boolean()),
            Self::Null => is(value.is_null()),
            Self::Value => Ok(()),
            Self::Object => is(value.is_object()),
            Self::Words(words) => is(value.as_str().is_some_and(|w| words.contains(&w))),
            Self::Numbers(names) => is(value.as_array().is_some_and(|items| {
                items.len() == names.len() && items.iter().all(Json::is_number)
            })),
            Self::List(item) => {
                let items = value
                    .as_array()
                    .ok_or_else(|| format!("`{at}` is {}, not {value}", self.describe()))?;
                for (index, element) in items.iter().enumerate() {
                    item.check(element, &format!("{at}[{index}]"))?;
                }
                Ok(())
            }
            Self::Record(fields) => {
                let object = value
                    .as_object()
                    .ok_or_else(|| format!("`{at}` is {}, not {value}", self.describe()))?;
                for key in object.keys() {
                    if !fields.iter().any(|f| f.name == key) {
                        return Err(format!(
                            "`{at}` is {}; `{key}` is none of its fields",
                            self.describe()
                        ));
                    }
                }
                for field in fields {
                    match object.get(field.name) {
                        Some(inner) => field.ty.check(inner, &format!("{at}.{}", field.name))?,
                        None if field.required => {
                            return Err(format!("`{at}` needs `{}` ({})", field.name, field.doc));
                        }
                        None => {}
                    }
                }
                Ok(())
            }
            Self::Map(_, inner) => {
                let object = value
                    .as_object()
                    .ok_or_else(|| format!("`{at}` is {}, not {value}", self.describe()))?;
                for (key, element) in object {
                    inner.check(element, &format!("{at}.{key}"))?;
                }
                Ok(())
            }
            Self::Either(tys) => is(tys.iter().any(|ty| ty.check(value, at).is_ok())),
        }
    }

    /// As JSON Schema.
    pub fn schema(&self) -> Json {
        match self {
            Self::Str => json!({ "type": "string" }),
            Self::Int => json!({ "type": "integer" }),
            Self::Num => json!({ "type": "number" }),
            Self::Bool => json!({ "type": "boolean" }),
            Self::Null => json!({ "type": "null" }),
            Self::Value => json!({}),
            Self::Object => json!({ "type": "object" }),
            Self::Words(words) => json!({ "type": "string", "enum": words }),
            Self::Numbers(names) => json!({
                "type": "array",
                "items": { "type": "number" },
                "minItems": names.len(),
                "maxItems": names.len(),
                "description": format!("[{}]", names.join(", ")),
            }),
            Self::List(item) => json!({ "type": "array", "items": item.schema() }),
            Self::Record(fields) => fields_schema(fields),
            Self::Map(key, value) => json!({
                "type": "object",
                "additionalProperties": value.schema(),
                "description": format!("keys: {key}"),
            }),
            Self::Either(tys) => json!({ "anyOf": tys.iter().map(Ty::schema).collect::<Vec<_>>() }),
        }
    }
}

/// An object of `fields` and no others, as JSON Schema.
fn fields_schema(fields: &[Field]) -> Json {
    let mut properties = serde_json::Map::new();
    for field in fields {
        let mut schema = field.ty.schema();
        schema["description"] = json!(field.doc);
        properties.insert(field.name.to_owned(), schema);
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": fields.iter().filter(|f| f.required).map(|f| f.name).collect::<Vec<_>>(),
        "additionalProperties": false,
    })
}

/// `advance`'s `until`: a value to watch and a bound on it.
fn until() -> Ty {
    Ty::Record(vec![
        needs("target", Ty::Str, "the value watched: `node.port[i].field`"),
        may("below", Ty::Num, "stop once it is below this"),
        may("above", Ty::Num, "stop once it is above this"),
        may("equals", Ty::Num, "stop once it equals this"),
    ])
}

/// One of `compose`'s `edits`: a statement by its kind and name, or a
/// node's member, and its new text.
fn edit() -> Ty {
    let named = |kind: &'static str, doc: &'static str| may(kind, Ty::Str, doc);
    Ty::Record(vec![
        named("node", "a node, by name"),
        named("output", "an output, by name"),
        named("param", "a param, by name"),
        named("input", "an input, by name"),
        named("kernel", "a Kernel, by id"),
        named("struct", "a struct, by name"),
        named("type", "a type alias, by name"),
        named("fn", "a function, by name"),
        named("use", "a use, by its path"),
        may(
            "member",
            Ty::Str,
            "a member of the node named: one binding or setting",
        ),
        needs(
            "text",
            Ty::Either(vec![Ty::Str, Ty::Null]),
            "the new text; null removes it",
        ),
    ])
}

fn needs(name: &'static str, ty: Ty, doc: &'static str) -> Field {
    Field {
        name,
        ty,
        required: true,
        doc,
    }
}

fn may(name: &'static str, ty: Ty, doc: &'static str) -> Field {
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
        may("time", Ty::Num, "session time in seconds; 0 when not given"),
        may(
            "rate",
            Ty::Num,
            "the sound clock, frames a second, which Kernels read as rate()",
        ),
        may(
            "pointer",
            Ty::Numbers(&["x", "y"]),
            "the pointer in the output's pixels from its top left, as a Kernel's `pointer()` reads it",
        ),
        may(
            "down",
            Ty::Bool,
            "whether the pointer is pressed, what `pointer_down()` reads",
        ),
        may(
            "buttons",
            Ty::List(Box::new(Ty::Int)),
            "buttons held: 0 left, 1 right, 2 middle, then a pad's",
        ),
        may(
            "wheel",
            Ty::Numbers(&["x", "y"]),
            "the scroll for this frame",
        ),
        may(
            "sticks",
            Ty::List(Box::new(Ty::Numbers(&["x", "y"]))),
            "a pad's left stick, right stick and triggers, up to three pairs",
        ),
        may(
            "keys",
            Ty::List(Box::new(Ty::Int)),
            "keys held, by the browser's codes",
        ),
        may(
            "window",
            Ty::Numbers(&["w", "h"]),
            "the window in pixels; else the session's",
        ),
        may(
            "scale",
            Ty::Num,
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
            "backend, scene, watched, kernels (published Kernels), natives (the backend's native operations), nodes, loaded (whether a scene is), refused (why not, when not), revision (moved by every change but running), frame (the frames committed since that revision), fusion, grants, versions (`orla`, `api`, `language`, `ir`, `package`)",
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
                    Ty::Str,
                    "the Kernel's id, `name@version`; else `id` names it",
                ),
                may(
                    "generation",
                    Ty::Int,
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
                Ty::Str,
                "one the session remembers; the current when not given",
            )],
            "path, text, and version, which `compose`'s `expect` names",
        ),
        op(
            "history",
            "What changed the world, and what was refused.",
            vec![may(
                "since",
                Ty::Int,
                "a revision; everything when not given",
            )],
            "revision, frame (the frames committed since that revision), the changes after `since` and the attempts refused since, each with its origin",
        ),
        op(
            "compose",
            "Changes the scene: rewrites its document on disk, the file at `path` (`document` and `status` name it), once the text is checked; answered once the scene stands. A refused text writes nothing and is an error: why, at `line` and `column`, the document's `version`, and for `edits` the `text` they made, which `line` counts in.",
            vec![
                may("text", Ty::Str, "the whole document; or `edits`"),
                may(
                    "edits",
                    Ty::List(Box::new(edit())),
                    "each names a statement, or a node's `member`, with its new `text` or null to remove it",
                ),
                may(
                    "expect",
                    Ty::Str,
                    "the version `document` answered; refused when the scene changed since",
                ),
            ],
            "events, version, path (the document), written",
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
                Ty::Str,
                "a document with one Kernel; it inherits the scene's declarations",
            )],
            "id, manifest; or an error at a line and column",
        ),
        op(
            "publish",
            "Compiles, validates and prepares a Kernel, then swaps it in whole; answered once it stands. The running world changes, not a file: the document it came from is left as it is.",
            vec![
                needs("source", Ty::Str, "a document with one Kernel"),
                may(
                    "expect",
                    Ty::Either(vec![Ty::Words(&["absent", "any"]), Ty::Int]),
                    "`absent` for a new Kernel, a generation `kernels` answered, or `any`; `any` when not given",
                ),
            ],
            "id, generation; and when nodes running it start state, `keeps` (those `node.port`s, which keep what they committed) and `reset` (the requests that start them from the new law)",
        ),
        op(
            "retire",
            "Removes a published Kernel.",
            vec![may(
                "kernel",
                Ty::Str,
                "the Kernel's id; else `id` names it",
            )],
            "retired",
        ),
        op(
            "param",
            "Sets a parameter the scene declares.",
            vec![
                needs("name", Ty::Str, "a param the scene declares"),
                needs(
                    "value",
                    Ty::Value,
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
                    Ty::List(Box::new(Ty::Str)),
                    "evaluate only these nodes, the state and never-produced outputs",
                ),
                may(
                    "settle",
                    Ty::Bool,
                    "first wait for every `async` run to land",
                ),
            ]),
            "executed, reused, committed, outputs",
        ),
        op(
            "advance",
            "Runs many frames, until a count or a value says to stop.",
            with_context(vec![
                needs("frames", Ty::Int, "the most frames to run"),
                may(
                    "fps",
                    Ty::Num,
                    "the clock's frames a second; 60 when not given",
                ),
                may(
                    "until",
                    until(),
                    "stops when `node.port[i].field` passes the bound",
                ),
                may(
                    "save",
                    Ty::Map("node.port", Box::new(Ty::Str)),
                    "files written after the last frame, where `--allow file.write` allows",
                ),
            ]),
            "frames, stopped, target, value, saved, millis, report",
        ),
        op(
            "bench",
            "Times the scene's frames.",
            with_context(vec![
                may("frames", Ty::Int, "frames a run; 30 when not given"),
                may("fps", Ty::Num, "the clock's frames a second"),
                may("repeat", Ty::Int, "runs, the least kept; 3 when not given"),
                may("nodes", Ty::Bool, "time every node too"),
            ]),
            "backend, frames, repeat, best_ms, median_ms, nodes, ran",
        ),
        op(
            "window",
            "Sets the window a headless session reports to evaluations that name none.",
            vec![
                may(
                    "size",
                    Ty::Either(vec![Ty::Numbers(&["w", "h"]), Ty::Null]),
                    "in pixels; null for none",
                ),
                may("scale", Ty::Num, "the scale factor; 1 when not given"),
            ],
            "the window",
        ),
        op(
            "read",
            "Reads an output or a node's port.",
            vec![
                may(
                    "output",
                    Ty::Str,
                    "a scene output; the first when neither it nor `node` is given",
                ),
                may("node", Ty::Str, "a node, with `port`"),
                may("port", Ty::Str, "the node's port; `result` when not given"),
                may(
                    "format",
                    Ty::Words(&["png", "rgba32f", "ppm", "summary", "words", "field"]),
                    "a picture (a Grid of pixels) as `png` (when not given), `rgba32f` or `ppm`; any Field as its `summary` (when not given and not a picture), its `words` or the Host's `field` file",
                ),
                may(
                    "from",
                    Ty::Int,
                    "the first element to answer as numbers, counted along the Field, x fastest, then y, then z; 0 when not given",
                ),
                may(
                    "count",
                    Ty::Int,
                    "how many elements from `from` to answer as numbers, at most 4096; 1 when not given",
                ),
            ],
            "a picture: width, height, format, data (base64; over MCP the picture itself). Any other Field: shape, extent, count, schema, and its summary (each number's min, max, sum, mean, how many were not zero, and how many were not finite), or `values` from `from` (a number, or a list for an element of several), or format and data (base64)",
        ),
        op(
            "write",
            "Replaces a node's committed state.",
            vec![
                needs("node", Ty::Str, "the node"),
                may(
                    "port",
                    Ty::Str,
                    "its port, read by a `<~` binding; `result` when not given",
                ),
                needs(
                    "data",
                    Ty::Str,
                    "base64 words, or a `field` file with `format`",
                ),
                may(
                    "format",
                    Ty::Words(&["words", "field"]),
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
                Ty::Str,
                "a file to write it to, where `--allow file.write` allows",
            )],
            "the world's revision and frame (the frames committed since it), every Kernel with its generation and source, every port's committed state; or the file written",
        ),
        op(
            "restore",
            "Puts a snapshot's world back, after a trial frame that commits nothing: the state is the snapshot's, which the next frame reads; until then `read` shows the trial frame's outputs.",
            with_context(vec![
                may("snapshot", Ty::Object, "as `snapshot` gave it; or `from`"),
                may("from", Ty::Str, "a file `snapshot` wrote"),
                needs(
                    "laws",
                    Ty::Words(&["current", "snapshot"]),
                    "which Kernels the restored world runs under",
                ),
                may(
                    "structure",
                    Ty::Words(&["current", "snapshot"]),
                    "which scene; `current` when not given",
                ),
            ]),
            "revision, frame, report",
        ),
        op(
            "reset",
            "Forgets committed state; `<~` readers start from their initial sources.",
            vec![
                may("node", Ty::Str, "one node; every node when not given"),
                may("port", Ty::Str, "one port of it; `result` when not given"),
            ],
            "what was reset",
        ),
        op(
            "help",
            "What Orla answers: this index, a section, one item by name, or with `full` everything at once.",
            vec![
                may("section", Ty::Str, "a section the index names"),
                may(
                    "full",
                    Ty::Bool,
                    "the whole reference in one answer, every item and every canonical example: the fastest start",
                ),
                may(
                    "name",
                    Ty::Str,
                    "one builtin, form, operation or replay step",
                ),
            ],
            "the index; a section's items; one item in full; or with `full`, `reference`, the whole of it as one Markdown text. An unknown name is an error with `similar`",
        ),
        op(
            "search",
            "The capabilities that exist: published Kernels, and every Kernel and preset graph of the packages this session reaches.",
            vec![
                may(
                    "text",
                    Ty::Str,
                    "words in a name, package, doc or port name, any case",
                ),
                may(
                    "takes",
                    Ty::Str,
                    "a type one of its inputs takes, as written: `Rows<Int>`",
                ),
                may("gives", Ty::Str, "a type one of its outputs gives"),
                may("requires", Ty::Str, "a capability it requires"),
                may("limit", Ty::Int, "the most to answer; 20 when not given"),
            ],
            "capabilities, nearest first, each with kind, name, doc, package, reach (the line a scene writes to use it: an installed package by its name, any other path relative to `base`), inputs, params, outputs (name, type, doc), requires, granted, published; base, the scene document's directory; count, how many matched",
        ),
        op(
            "trace",
            "Runs one element of a node's Kernel and shows every step of it.",
            with_context(vec![
                needs("node", Ty::Str, "the node"),
                may(
                    "at",
                    Ty::Either(vec![
                        Ty::Numbers(&["x", "y"]),
                        Ty::Numbers(&["x", "y", "z"]),
                    ]),
                    "the element in a grid or a volume; or `element`",
                ),
                may("element", Ty::Int, "the element in rows, or a scan's step"),
            ]),
            "kernel, at, bindings, results, calls",
        ),
        op(
            "pin",
            "Keeps a node materialized so `read` can see it.",
            vec![needs("node", Ty::Str, "the node")],
            "pinned",
        ),
        op(
            "unpin",
            "Lets fusion have a pinned node back.",
            vec![needs("node", Ty::Str, "the node")],
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
        // The field it was meant as: the one named nearest, or else the
        // only field the request lacks that would hold its value, one the
        // operation needs before one it may take.
        let value = &object[key];
        let holds: Vec<&Field> = op
            .fields
            .iter()
            .filter(|field| !object.contains_key(field.name))
            .filter(|field| field.ty.check(value, field.name).is_ok())
            .collect();
        let needed: Vec<&Field> = holds
            .iter()
            .copied()
            .filter(|field| field.required)
            .collect();
        let only = match (needed.as_slice(), holds.as_slice()) {
            ([one], _) | ([], [one]) => Some(one.name),
            _ => None,
        };
        let similar = orla_language::nearest(key, takes.iter().copied());
        let near = similar.or(only);
        // The request with the field named as the operation takes it, or
        // without it; offered only when the operation would take it, and
        // otherwise `help` for the operation.
        let mut fixed = object.clone();
        fixed.remove("id");
        if let Some(value) = fixed.remove(key)
            && let Some(near) = near
        {
            fixed.insert(near.to_owned(), value);
        }
        let fixed = Json::Object(fixed);
        let next = if check(&fixed).is_ok() {
            fixed
        } else {
            crate::wire::ask_help(op.name)
        };
        return Err(crate::wire::tried(
            json!({
                "message": format!(
                    "`{}` takes {}; `{key}` is not one of them{}",
                    op.name,
                    listed(),
                    orla_language::suggestion(near)
                ),
                "fields": takes,
                "similar": similar.into_iter().collect::<Vec<_>>(),
            }),
            [next],
        ));
    }
    if let Some(missing) = op
        .fields
        .iter()
        .find(|field| field.required && !object.contains_key(field.name))
    {
        return Err(crate::wire::tried(
            json!({
                "message": format!("`{}` needs `{}` ({}); it takes {}", op.name, missing.name, missing.doc, listed()),
                "fields": takes,
            }),
            [crate::wire::ask_help(op.name)],
        ));
    }
    for field in &op.fields {
        if let Some(value) = object.get(field.name)
            && let Err(why) = field.ty.check(value, field.name)
        {
            return Err(crate::wire::tried(
                json!({
                    "message": format!("`{}`: {why}", op.name),
                    "fields": takes,
                    "field": field.name,
                    "expected": field.ty.describe(),
                }),
                [crate::wire::ask_help(op.name)],
            ));
        }
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
                "type": field.ty.describe(),
                "required": field.required,
                "doc": field.doc,
            })).collect::<Vec<_>>(),
            "answer": self.answer,
            "schema": self.schema(),
        })
    }

    /// Its request's fields, beside `op` and `id`, as JSON Schema.
    pub fn schema(&self) -> Json {
        fields_schema(&self.fields)
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
                    field.ty.describe().replace('|', "\\|")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `try` a misnamed field is answered with is a request the
    /// operation takes.
    #[test]
    fn a_misnamed_field_is_corrected_to_a_request_the_operation_takes() {
        for op in operations() {
            for value in [json!("image.blur@1"), json!(3), json!({ "x": 1 })] {
                for key in ["name", "zzq"] {
                    let request = json!({ "op": op.name, key: value });
                    if check(&request).is_ok() {
                        continue;
                    }
                    let refused = check(&request).unwrap_err();
                    for next in refused["try"].as_array().into_iter().flatten() {
                        assert!(
                            check(next).is_ok(),
                            "{request} tries {next}, which is refused"
                        );
                    }
                }
            }
        }
        let refused = check(&json!({ "op": "kernel", "name": "image.blur@1" })).unwrap_err();
        assert_eq!(
            refused["try"],
            json!([{ "op": "kernel", "kernel": "image.blur@1" }])
        );
    }
}
