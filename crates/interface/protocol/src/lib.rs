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
pub mod envelope;
mod fields;
pub mod help;
mod inspect;
pub mod ops;
mod request;
mod run;
mod transport;
pub mod websocket;
mod wire;

use serde_json::{Value as Json, json};

use entelechy_session::Session;

use crate::request::Request;

pub use answers::{Answers, Served};
pub use envelope::{envelope, parse_request};
pub use run::{evaluation_json, measure_asked};
pub use transport::{Incoming, Reply, listen, serve, stdin_into, stdin_lines};
pub use wire::{
    CONTEXT_FIELDS, context_given, context_of, report_json, trace_json, type_name, value_from_json,
    value_json, words_json, words_text,
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
    let before = session.act_for(entelechy_session::Origin::Protocol(incoming.from.clone()));
    let answered = respond_parsed(session, &incoming.parsed);
    session.act_for(before);
    answered
}

/// Answers one request line; the flag says whether it asked to quit.
pub fn respond_line(session: &mut Session, line: &str) -> (String, bool) {
    respond_parsed(session, &parse_request(line))
}

/// Answers a request line already read ([`parse_request`]).
fn respond_parsed(session: &mut Session, parsed: &Result<(Json, Json), Json>) -> (String, bool) {
    match parsed {
        Ok((id, request)) => {
            let quit = request.get("op").and_then(Json::as_str) == Some("quit");
            (envelope(id, handle(session, request)), quit)
        }
        Err(error) => (envelope(&Json::Null, Err(error.clone())), false),
    }
}

/// The operations that run or read the scene.
const RUNS_THE_SCENE: [&str; 4] = ["evaluate", "read", "write", "trace"];

/// Handles one request by the function for the operation it names.
pub fn handle(session: &mut Session, request: &Json) -> Result<Json, Json> {
    ops::check(request)?;
    let request = Request::new(request)?;
    // What runs or reads the scene is refused while none is loaded,
    // saying why and what to do.
    if RUNS_THE_SCENE.contains(&request.op())
        && let Some(why) = session.why_no_scene()
    {
        // The inline form's example is the smallest whole scene.
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
        "compose" => create::compose(session, &request),
        "sync" => create::sync(session),
        "prepare" => create::prepare(session, &request),
        "publish" => create::publish(session, &request),
        "retire" => create::retire(session, &request),
        "param" => create::param(session, &request),
        "evaluate" => run::evaluate(session, &request),
        "reset" => run::reset(session, &request),
        "read" => fields::read(session, &request),
        "snapshot" => fields::snapshot(session, &request),
        "restore" => fields::restore(session, &request),
        "write" => fields::write(session, &request),
        "trace" => fields::trace(session, &request),
        "pin" => fields::pin(session, &request),
        "unpin" => fields::unpin(session, &request),
        "quit" => Ok(json!({ "bye": true })),
        other => {
            let near = entelechy_language::nearest(other, ops::names());
            Err(json!({
                "message": format!(
                    "unknown op `{other}`; the operations are {}{}",
                    ops::names().join(", "),
                    entelechy_language::suggestion(near)
                ),
                "operations": ops::names(),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::diagnostic_json;
    use entelechy_kernel::Value;

    /// A picture of 8 by 8 texels, the whole scene a request is put to.
    const SCENE: &str = "node gradient: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0); } }\noutput result <- gradient.result;\n";

    /// A session over [`SCENE`], written into a directory of its own.
    fn session(name: &str) -> (std::path::PathBuf, Session) {
        let dir =
            std::env::temp_dir().join(format!("entelechy-protocol-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
        let session = Session::open(
            &dir.join("scene.esn"),
            None,
            entelechy_session::DEFAULT_BUDGET_WORDS,
        );
        (dir, session)
    }

    /// The context a request gives is named field by field, as a window
    /// that reads its own input says what it did not read.
    #[test]
    fn the_context_a_request_gives_is_named() {
        let given = crate::context_given(
            &json!({ "op": "evaluate", "keys": [37], "time": 1.0, "measure": true }),
        );
        assert_eq!(given, vec!["time", "keys"]);
        assert!(crate::context_given(&json!({ "op": "evaluate" })).is_empty());
    }

    /// `docs/PROTOCOL.md`'s table is the operations' rendering, and every
    /// operation is answered; `ENTELECHY_BLESS=1` writes the table again.
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
            if std::env::var_os("ENTELECHY_BLESS").is_some() {
                std::fs::write(&path, format!("{}{table}{}", &text[..start], &text[end..]))
                    .unwrap();
            } else {
                panic!(
                    "docs/PROTOCOL.md's table is not the operations'; write it again with \
                     ENTELECHY_BLESS=1 cargo test -p entelechy-protocol\nexpected:\n{table}"
                );
            }
        }
        let (dir, mut session) = session("table");
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A field an operation does not take, or a missing one it needs, is
    /// refused with the fields it takes; the near miss is named.
    #[test]
    fn a_request_is_held_to_its_operations_fields() {
        let (dir, mut session) = session("fields");
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_name_in_help_lists_the_similar_ones_as_data() {
        let (dir, mut session) = session("help");
        let error = handle(&mut session, &json!({ "op": "help", "name": "clmap" })).unwrap_err();
        assert_eq!(error["similar"][0], "clamp", "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A field of the wrong type is refused before its operation reads it,
    /// naming the type it is; nested values are named by where they are.
    #[test]
    fn a_field_is_held_to_its_type() {
        let (dir, mut session) = session("types");
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
        assert!(refused(json!({ "op": "evaluate", "text": 7 })).contains("`text` is string"));
        // An evaluation holds 32 characters typed.
        assert!(
            refused(json!({ "op": "evaluate", "text": "a".repeat(33) }))
                .contains("`text` holds at most 32 characters")
        );
        assert!(
            refused(json!({ "op": "compose", "edits": [{ "node": 3, "text": null }] }))
                .contains("`edits[0].node` is string")
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
        let _ = std::fs::remove_dir_all(&dir);
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
        let json = diagnostic_json(&entelechy_session::Diagnostic {
            message: "`oops` is not defined".into(),
            at: Some((4, 20)),
        });
        assert_eq!(json["line"], 4);
        assert_eq!(json["column"], 20);
        assert_eq!(json["message"], "`oops` is not defined");
        // A message starting with numbers carries no place.
        let plain = diagnostic_json(&entelechy_session::Diagnostic::new("3:4 is a ratio"));
        assert!(plain["line"].is_null());
        assert_eq!(plain["message"], "3:4 is a ratio");
    }

    /// What a document renders between `<!-- rendered: <what> -->` and
    /// `<!-- end -->`: the wire, or the whole reference.
    fn rendering(what: &str) -> Option<String> {
        match what {
            "wire" => Some(crate::envelope::table()),
            "reference" => Some(crate::help::reference()),
            _ => None,
        }
    }

    /// Every rendered region of the documents is its rendering;
    /// `ENTELECHY_BLESS=1` writes the regions again.
    #[test]
    fn the_documents_render_what_they_show() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let mut files = vec![root.join("README.md")];
        for entry in std::fs::read_dir(root.join("docs")).unwrap().flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "md") {
                files.push(entry.path());
            }
        }
        let bless = std::env::var_os("ENTELECHY_BLESS").is_some();
        let mut wrong = Vec::new();
        let mut regions = 0;
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            // Each region, replaced by its rendering.
            let mut out = String::new();
            let mut rest = text.as_str();
            while let Some(at) = rest.find("<!-- rendered: ") {
                let open_end = at + rest[at..].find("-->").expect("a marker closes") + 3;
                let what = rest[at + 15..open_end - 3].trim().to_owned();
                let body_start = open_end + 1;
                let close = body_start
                    + rest[body_start..]
                        .find("<!-- end -->")
                        .unwrap_or_else(|| panic!("{name}: `{what}` has no `<!-- end -->`"));
                out.push_str(&rest[..body_start]);
                match rendering(&what) {
                    Some(rendered) => {
                        if rest[body_start..close] != rendered {
                            wrong.push(format!("{name}: `{what}` is not its rendering"));
                        }
                        out.push_str(&rendered);
                    }
                    None => {
                        wrong.push(format!("{name}: nothing renders `{what}`"));
                        out.push_str(&rest[body_start..close]);
                    }
                }
                regions += 1;
                rest = &rest[close..];
            }
            out.push_str(rest);
            if bless && out != text {
                std::fs::write(&path, out).unwrap();
            }
        }
        assert!(regions > 0, "no document renders anything");
        if !bless {
            assert!(
                wrong.is_empty(),
                "{}\nwrite the regions again with ENTELECHY_BLESS=1 cargo test -p entelechy-protocol",
                wrong.join("\n")
            );
        }
    }
}
