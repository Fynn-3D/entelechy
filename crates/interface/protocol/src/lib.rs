//! The creation protocol: JSON lines over stdin and stdout.
//!
//! Any agent, in any language, drives a live session through the same
//! operations the CLI and the Player use: discover Kernels and their
//! signatures, inspect the graph, prepare source and receive positioned
//! diagnostics, publish against an expected generation, compose the scene,
//! set parameters, evaluate, and read results. Transport is a detail; the
//! contract is these operations and the authority rules of the session.
//!
//! Every request is one JSON object per line with an `op` and an optional
//! `id` that is echoed back; every response is one line, `{"id", "ok": ...}`
//! or `{"id", "error": {"message", "line"?, "column"?}}`. An error says what
//! an agent can act on as data as well as in words: an unknown operation
//! lists the `operations`, and an unknown name in `help` the `similar` ones.
//!
//! | op | request fields | ok |
//! | --- | --- | --- |
//! | `status` | | backend, scene, watched, kernels, nodes |
//! | `kernels` | | list of manifests with generations |
//! | `kernel` | `kernel` (the Kernel's id) | manifest, generation, source, as `publish` takes it |
//! | `graph` | | inputs, params, nodes, outputs |
//! | `document` | | path, text of the scene |
//! | `compose` | `text` | writes the scene, syncs; events |
//! | `sync` | | events |
//! | `prepare` | `source` (a document with one Kernel; it inherits the scene's declarations) | id, manifest (nothing published) |
//! | `publish` | `source`, `expect` = `"absent"` / `"any"` / generation | id, generation |
//! | `retire` | `kernel` (the Kernel's id) | retired |
//! | `param` | `name`, `value` | ok |
//! | `evaluate` | `time`?, `rate`?, `pointer`?, `down`?, `buttons`?, `wheel`?, `keys`?, `window`? (`[w, h]` in pixels), `scale`?, `nodes`? (evaluate only these, the state, and never-produced outputs) | executed, reused, committed, outputs |
//! | `advance` | `frames`, `fps`?, `time`?, the context `evaluate` takes, `until`? (`{target, below / above / equals}`), `save`? | frames, stopped, target, value, saved, millis, report |
//! | `bench` | `frames`?, `fps`?, `repeat`?, `nodes`? | backend, frames, repeat, best_ms, median_ms, nodes, ran |
//! | `window` | `size` (`[w, h]` in pixels, or null for none), `scale`? | the window a headless session reports to evaluations that name none |
//! | `read` | `output`? or `node`+`port`, `format`? = `"rgba32f"` / `"ppm"` / `"field"` | width, height, format, data (base64); words for a Field that is not pixels; `field` is the Host's file format for any Field |
//! | `write` | `node`, `port`?, `data` (base64 words, or a `field` file with `format`) | replaces the node's committed state; the port must be read by a `<~` binding |
//! | `reset` | | ok |
//! | `help` | `name`? | every builtin and form, or the one named; an unknown name is an error with `similar` |
//! | `trace` | `node`, `at` or `element`, the context `evaluate` takes | kernel, at, bindings, results, calls |
//! | `pin` / `unpin` | `node` | keeps a node materialized so `read` can see it |
//! | `quit` | | bye |
//!
//! Each operation is a function in the module for what it may do, and is
//! given only that: [`inspect`] reads the session and cannot change it,
//! [`create`] changes the scene and its Kernels, [`run`] runs them, and
//! [`fields`] reads and writes a node's Fields. [`wire`] says how values
//! and results look as JSON, and [`transport`] carries the lines.

#![warn(missing_docs)]

mod create;
mod fields;
mod inspect;
mod request;
mod run;
mod transport;
pub mod websocket;
mod wire;

use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::failure;

pub use transport::{Incoming, Reply, listen, serve, stdin_into};
pub use wire::{
    context_of, report_json, trace_json, type_name, value_from_json, value_json, words_json,
    words_text,
};

/// Answers one incoming request; returns whether the process should end
/// (a `quit` on standard input). A `quit` on a socket answers, then
/// closes that connection.
pub fn answer(session: &mut Session, incoming: Incoming) -> bool {
    let (response, quit) = respond_line(session, &incoming.line);
    incoming.reply.send(response);
    if quit && !incoming.local {
        incoming.reply.send(String::new());
    }
    quit && incoming.local
}

/// Parses one request line and returns the request and its echoed id, or
/// an error envelope's payload.
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

/// Answers one request line; the flag says whether it asked to quit.
pub fn respond_line(session: &mut Session, line: &str) -> (String, bool) {
    match parse_request(line) {
        Ok((id, request)) => {
            let quit = request.get("op").and_then(Json::as_str) == Some("quit");
            (envelope(&id, handle(session, &request)), quit)
        }
        Err(error) => (envelope(&Json::Null, Err(error)), false),
    }
}

/// Every operation the protocol answers, in the order the table above
/// lists them. A request naming another is refused with this list, so an
/// agent that guessed wrong learns what there is.
pub const OPERATIONS: [&str; 23] = [
    "status", "kernels", "kernel", "graph", "document", "compose", "sync", "prepare", "publish",
    "retire", "param", "evaluate", "advance", "bench", "window", "read", "write", "reset", "help",
    "trace", "pin", "unpin", "quit",
];

/// Handles one request: the operation it names, answered by the function
/// that does it, with what that function may touch.
pub fn handle(session: &mut Session, request: &Json) -> Result<Json, Json> {
    let request = Request::new(request)?;
    match request.op() {
        "status" => inspect::status(session),
        "kernels" => inspect::kernels(session),
        "kernel" => inspect::kernel(session, &request),
        "graph" => inspect::graph(session),
        "document" => inspect::document(session),
        "help" => inspect::help(&request),
        "compose" => create::compose(session, &request),
        "sync" => create::sync(session),
        "prepare" => create::prepare(session, &request),
        "publish" => create::publish(session, &request),
        "retire" => create::retire(session, &request),
        "param" => create::param(session, &request),
        "evaluate" => run::evaluate(session, &request),
        "advance" => run::advance(session, &request),
        "bench" => run::bench(session, &request),
        "reset" => run::reset(session),
        "window" => run::window(session, &request),
        "read" => fields::read(session, &request),
        "write" => fields::write(session, &request),
        "trace" => fields::trace(session, &request),
        "pin" => fields::pin(session, &request),
        "unpin" => fields::unpin(session, &request),
        "quit" => Ok(json!({ "bye": true })),
        other => Err(json!({
            "message": format!("unknown op `{other}`; the operations are {}", OPERATIONS.join(", ")),
            "operations": OPERATIONS,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::positioned;
    use orla_runtime::Value;

    /// The table at the top of this module is the protocol's contract.
    /// It names every operation the server answers, the server answers
    /// every operation it names, and a name that is neither is told the
    /// list: an operation cannot be added, dropped or renamed without the
    /// contract saying so.
    #[test]
    fn the_table_is_every_operation_and_every_operation_is_answered() {
        let documented: Vec<&str> = include_str!("lib.rs")
            .lines()
            .take_while(|line| line.starts_with("//!"))
            .filter_map(|line| line.strip_prefix("//! | `"))
            .flat_map(|row| {
                let cell = row.split(" |").next().unwrap_or_default();
                cell.split(" / ").map(|op| op.trim_matches('`'))
            })
            .collect();
        assert_eq!(documented, OPERATIONS, "the table and `OPERATIONS` differ");
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        session.sync();
        for op in OPERATIONS {
            // Asked with nothing else, an operation may refuse for want of
            // a field; what it may not do is not know its own name.
            if let Err(error) = handle(&mut session, &json!({ "op": op })) {
                assert!(
                    error.get("operations").is_none(),
                    "`{op}` is in the table but the server does not answer it: {error}"
                );
            }
        }
        let error = handle(&mut session, &json!({ "op": "compile" })).unwrap_err();
        assert_eq!(error["operations"], json!(OPERATIONS));
    }

    #[test]
    fn an_unknown_name_in_help_lists_the_similar_ones_as_data() {
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        let error = handle(&mut session, &json!({ "op": "help", "name": "clmap" })).unwrap_err();
        assert_eq!(error["similar"][0], "clamp", "{error}");
    }

    #[test]
    fn values_round_trip_through_json() {
        for value in [
            Value::Int(3),
            Value::Float(1.5),
            Value::Bool(true),
            Value::Vec2([1.0, 2.0]),
            Value::Vec3([1.0, 2.0, 3.0]),
            Value::Vec4([1.0, 2.0, 3.0, 4.0]),
            Value::IVec2([3, -4]),
            Value::Mat2([1.0, 0.0, 0.5, 1.0]),
            Value::Record(vec![Value::Vec3([0.0, 1.0, 0.0]), Value::Int(2)]),
            Value::Double(500000.123456789),
            Value::Double(0.1),
            Value::Long(5),
            Value::Long(-(1 << 60) - 3),
            Value::DVec3([0.1, -4000000.987654321, 2.0]),
            Value::LVec2([1 << 62, -7]),
            Value::DMat2([1.0, 0.1, -0.1, 1.0]),
            Value::Record(vec![Value::DVec2([0.25, 0.5]), Value::Long(1 << 40)]),
        ] {
            // JSON is untyped; what the value is read as is its shape.
            let back = value_from_json(&value_json(&value)).unwrap();
            assert_eq!(back.shaped_like(&value), Some(value));
        }
        // A long a double cannot hold travels as its digits.
        assert_eq!(
            value_json(&Value::Long((1 << 53) + 1)),
            json!("9007199254740993")
        );
        assert_eq!(
            value_json(&Value::Long(1 << 53)),
            json!(9007199254740992i64)
        );
        // A number with a fraction is exact until a width is asked of it.
        assert_eq!(
            value_from_json(&json!(0.1))
                .unwrap()
                .shaped_like(&Value::Float(0.0)),
            Some(Value::Float(0.1))
        );
        assert!(value_from_json(&json!("no")).is_err());
    }

    #[test]
    fn positions_are_lifted_out_of_messages() {
        let json = positioned("4:20: `oops` is not defined".into());
        assert_eq!(json["line"], 4);
        assert_eq!(json["column"], 20);
        assert_eq!(json["message"], "`oops` is not defined");
        assert!(positioned("plain".into())["line"].is_null());
    }
}
