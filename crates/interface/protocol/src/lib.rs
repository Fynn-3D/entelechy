//! The creation protocol: JSON lines over stdin and stdout.
//!
//! Any agent drives a live session through the operations the CLI and the
//! Player use. The contract is these operations and the session's
//! authority rules.
//!
//! The envelope, and every operation with its fields and answer, are in
//! `docs/PROTOCOL.md`, *Operations*, rendered from [`ops`], which the
//! server checks every request against.
//!
//! Each operation lives in the module for what it may do: [`inspect`]
//! reads the session, [`create`] changes the scene and its Kernels, [`run`]
//! runs them, [`fields`] reads and writes a node's Fields. [`wire`] is the
//! JSON shape of values and results; [`transport`] carries the lines.

#![warn(missing_docs)]

mod answers;
mod create;
mod fields;
pub mod help;
mod inspect;
pub mod mcp;
pub mod ops;
mod request;
mod run;
mod transport;
pub mod websocket;
mod wire;

use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::failure;

pub use answers::{Answers, Served};
pub use transport::{Framing, Incoming, Reply, listen, serve, stdin_into, stdin_lines};
pub use wire::{
    capability_json, context_of, report_json, trace_json, type_name, value_from_json, value_json,
    words_json, words_text,
};

/// Answers one incoming request; true when the process should end (a
/// `quit` on standard input). A `quit` on a socket closes that connection.
pub fn answer(session: &mut Session, incoming: Incoming) -> bool {
    let (response, quit) = respond_from(session, &incoming);
    incoming.reply.send(response);
    if quit && !incoming.local {
        incoming.reply.send(String::new());
    }
    quit && incoming.local
}

/// Answers as [`respond_line`] does, recording changes as from the
/// request's channel.
pub fn respond_from(session: &mut Session, incoming: &Incoming) -> (String, bool) {
    let before = session.act_for(orla_session::Origin::Protocol(incoming.from.clone()));
    let answered = respond_line(session, &incoming.line);
    session.act_for(before);
    answered
}

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

/// The operations that run or read the scene.
const RUNS_THE_SCENE: [&str; 6] = ["evaluate", "advance", "bench", "read", "write", "trace"];

/// Handles one request by the function for the operation it names.
pub fn handle(session: &mut Session, request: &Json) -> Result<Json, Json> {
    ops::check(request)?;
    let request = Request::new(request)?;
    // What runs or reads the scene is refused while none is loaded,
    // saying why and what to do.
    if RUNS_THE_SCENE.contains(&request.op())
        && let Some(why) = session.why_no_scene()
    {
        return Err(wire::failure(format!(
            "no scene is loaded: {why}; `compose` one"
        )));
    }
    match request.op() {
        "status" => inspect::status(session),
        "kernels" => inspect::kernels(session),
        "kernel" => inspect::kernel(session, &request),
        "graph" => inspect::graph(session),
        "document" => inspect::document(session, &request),
        "history" => inspect::history(session, &request),
        "help" => inspect::help(&request),
        "search" => inspect::search(session, &request),
        "compose" => create::compose(session, &request),
        "sync" => create::sync(session),
        "prepare" => create::prepare(session, &request),
        "publish" => create::publish(session, &request),
        "retire" => create::retire(session, &request),
        "param" => create::param(session, &request),
        "evaluate" => run::evaluate(session, &request),
        "advance" => run::advance(session, &request),
        "bench" => run::bench(session, &request),
        "reset" => run::reset(session, &request),
        "window" => run::window(session, &request),
        "read" => fields::read(session, &request),
        "snapshot" => fields::snapshot(session, &request),
        "restore" => fields::restore(session, &request),
        "write" => fields::write(session, &request),
        "trace" => fields::trace(session, &request),
        "pin" => fields::pin(session, &request),
        "unpin" => fields::unpin(session, &request),
        "quit" => Ok(json!({ "bye": true })),
        other => Err(json!({
            "message": format!(
                "unknown op `{other}`; the operations are {}{}",
                ops::names().join(", "),
                orla_language::did_you_mean(other, ops::names())
            ),
            "operations": ops::names(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::diagnostic_json;
    use orla_kernel::Value;

    /// `docs/PROTOCOL.md`'s table is the operations' rendering, and every
    /// operation is answered; `ORLA_BLESS=1` writes the table again.
    #[test]
    fn the_document_shows_every_operation_and_every_operation_is_answered() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/PROTOCOL.md");
        let text = std::fs::read_to_string(&path).unwrap();
        let start = text
            .find("\n| op |")
            .expect("PROTOCOL.md has the operations table")
            + 1;
        let end = start + text[start..].find("\n\n").expect("the table ends") + 1;
        let table = ops::table();
        if text[start..end] != table {
            if std::env::var_os("ORLA_BLESS").is_some() {
                std::fs::write(&path, format!("{}{table}{}", &text[..start], &text[end..]))
                    .unwrap();
            } else {
                panic!(
                    "docs/PROTOCOL.md's table is not the operations'; write it again with \
                     ORLA_BLESS=1 cargo test -p orla-protocol\nexpected:\n{table}"
                );
            }
        }
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        session.sync();
        for op in ops::names() {
            // A bare request may lack a field; its name is still known.
            if let Err(error) = handle(&mut session, &json!({ "op": op })) {
                assert!(
                    error.get("operations").is_none(),
                    "`{op}` is in the table but the server does not answer it: {error}"
                );
            }
        }
        let error = handle(&mut session, &json!({ "op": "compile" })).unwrap_err();
        assert_eq!(error["operations"], json!(ops::names()));
    }

    /// A field an operation does not take, or a missing one it needs, is
    /// refused with the fields it takes; the near miss is named.
    #[test]
    fn a_request_is_held_to_its_operations_fields() {
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        session.sync();
        let error = handle(&mut session, &json!({ "op": "publish", "sorce": "x" })).unwrap_err();
        assert_eq!(error["fields"], json!(["source", "expect"]), "{error}");
        assert_eq!(error["similar"], json!(["source"]), "{error}");
        let error = handle(&mut session, &json!({ "op": "publish" })).unwrap_err();
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .starts_with("`publish` needs `source`"),
            "{error}"
        );
        let error = handle(&mut session, &json!({ "op": "status", "verbose": true })).unwrap_err();
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("takes no fields"),
            "{error}"
        );
        assert!(handle(&mut session, &json!({ "op": "status", "id": 7 })).is_ok());
    }

    #[test]
    fn an_unknown_name_in_help_lists_the_similar_ones_as_data() {
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        let error = handle(&mut session, &json!({ "op": "help", "name": "clmap" })).unwrap_err();
        assert_eq!(error["similar"][0], "clamp", "{error}");
    }

    /// A field of the wrong type is refused before its operation reads it,
    /// naming the type it is; nested values are named by where they are.
    #[test]
    fn a_field_is_held_to_its_type() {
        let scene = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/grayscale/scene.osn");
        let mut session = Session::open(&scene, None, orla_session::DEFAULT_BUDGET_WORDS);
        session.sync();
        let mut refused = |request: Json| {
            let error = handle(&mut session, &request).unwrap_err();
            error["message"].as_str().unwrap().to_owned()
        };
        assert!(
            refused(json!({ "op": "evaluate", "time": "0" }))
                .contains("`time` is number, not \"0\"")
        );
        assert!(
            refused(json!({ "op": "publish", "source": "x", "expect": "anyy" }))
                .contains("`expect` is \"absent\" | \"any\" | integer")
        );
        assert!(
            refused(json!({ "op": "read", "format": "jpeg" }))
                .contains("\"png\" | \"rgba32f\" | \"ppm\" | \"summary\" | \"words\" | \"field\"")
        );
        assert!(
            refused(json!({ "op": "evaluate", "pointer": [1] })).contains("`pointer` is [x, y]")
        );
        assert!(
            refused(json!({ "op": "evaluate", "keys": [65, "B"] }))
                .contains("`keys[1]` is integer")
        );
        assert!(
            refused(json!({ "op": "advance", "frames": 2, "until": { "target": 3 } }))
                .contains("`until.target` is string")
        );
        // The type is the operation's to take further: a wrong one never
        // reaches it.
        assert!(
            handle(
                &mut session,
                &json!({ "op": "evaluate", "time": 0.5, "pointer": [1, 2] })
            )
            .is_ok()
        );
    }

    /// Every operation's request is an object of its fields and no others,
    /// as JSON Schema, the needed ones required.
    #[test]
    fn every_operation_has_a_schema_of_its_fields() {
        for op in ops::operations() {
            let schema = op.schema();
            assert_eq!(schema["type"], "object", "{}", op.name);
            assert_eq!(schema["additionalProperties"], false, "{}", op.name);
            let properties = schema["properties"].as_object().unwrap();
            assert_eq!(properties.len(), op.fields.len(), "{}", op.name);
            for field in &op.fields {
                assert!(
                    properties[field.name]["description"].is_string(),
                    "{}.{}",
                    op.name,
                    field.name
                );
            }
        }
        let publish = ops::operation("publish").unwrap().schema();
        assert_eq!(publish["required"], json!(["source"]));
        assert_eq!(
            publish["properties"]["expect"]["anyOf"][0]["enum"],
            json!(["absent", "any"])
        );
        let evaluate = ops::operation("evaluate").unwrap().schema();
        assert_eq!(evaluate["properties"]["pointer"]["maxItems"], 2);
    }

    /// A float answers as the decimal written for it.
    #[test]
    fn a_float_is_its_shortest_decimal() {
        assert_eq!(value_json(&Value::Float(0.3)).to_string(), "0.3");
        assert_eq!(
            value_json(&Value::Vec2([0.1, 2.5])).to_string(),
            "[0.1,2.5]"
        );
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
    fn a_diagnostic_carries_its_place_as_data() {
        let json = diagnostic_json(&orla_session::Diagnostic {
            message: "`oops` is not defined".into(),
            at: Some((4, 20)),
        });
        assert_eq!(json["line"], 4);
        assert_eq!(json["column"], 20);
        assert_eq!(json["message"], "`oops` is not defined");
        // A message starting with numbers carries no place.
        let plain = diagnostic_json(&orla_session::Diagnostic::new("3:4 is a ratio"));
        assert!(plain["line"].is_null());
        assert_eq!(plain["message"], "3:4 is a ratio");
    }
}
