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
mod trial;
pub mod websocket;
mod wire;

use serde_json::{Value as Json, json};

use entelechy_session::Session;

use crate::request::Request;

pub use answers::{Answers, Served};
pub use envelope::{envelope, parse_request};
pub use run::{evaluation_json, measure_asked, step_for_window, steps_asked};
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
        "catalog" => inspect::catalog(session, &request),
        "kernel" => inspect::kernel(session, &request),
        "graph" => inspect::graph(session),
        "document" => inspect::document(session, &request),
        "history" => inspect::history(session, &request),
        "help" => inspect::help(&request),
        "compose" => create::compose(session, &request),
        "trial" => trial::trial(session, &request),
        "diagnose" => trial::diagnose(session, &request),
        "sync" => create::sync(session),
        "prepare" => create::prepare(session, &request),
        "publish" => create::publish(session, &request),
        "retire" => create::retire(session, &request),
        "param" => create::param(session, &request),
        "evaluate" => run::evaluate(session, &request),
        "reset" => run::reset(session, &request),
        // A window's clock is its own (the Player answers this); a serve's
        // is its caller's.
        "pause" => Err(wire::failure(
            "`pause` holds a window's clock, and here the caller gives the time: `evaluate` with the same `time` holds it, a later one steps it".to_owned(),
        )),
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
            Value::DVec3(Box::new([0.1, -4000000.987654321, 2.0])),
            Value::LVec2([1 << 62, -7]),
            Value::DMat2(Box::new([1.0, 0.1, -0.1, 1.0])),
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
            fix: None,
        });
        assert_eq!(json["line"], 4);
        assert_eq!(json["column"], 20);
        assert_eq!(json["message"], "`oops` is not defined");
        assert!(json["fix"].is_null());
        // One that a `use` mends carries it as an item of compose's `documents`.
        let mended = diagnostic_json(&entelechy_session::Diagnostic {
            message: "`deep` is declared in c.ent, which this document does not use".into(),
            at: Some((2, 9)),
            fix: Some(Box::new(entelechy_session::Fix {
                document: std::path::PathBuf::from("scene/a.ent"),
                using: "c.ent".into(),
                text: None,
            })),
        });
        assert_eq!(
            mended["fix"],
            json!({ "path": "scene/a.ent", "edits": [{ "use": "c.ent", "text": "use \"c.ent\";" }] })
        );
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

    #[test]
    fn warp_advances_state_skips_unasked_outputs_and_stops_on_until_settled_or_nonfinite() {
        let dir =
            std::env::temp_dir().join(format!("entelechy-protocol-warp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // `view` divides by zero (`1 / 0`) if ever executed while `gate == 0`,
        // proving `warp` never evaluates unasked viewport outputs!
        let scene = "\
kernel w.seed@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }
kernel w.step@1 {
    requires { field.read; field.write; }
    input prev: Rows<Int>;
    param cap: Int = 1000;
    output result: Rows<Int>;
    compute {
        let v = at(prev, 0);
        if v < cap { return v + 1; }
        return v;
    }
}
kernel w.done@1 {
    requires { field.read; field.write; }
    input count: Rows<Int>;
    param target: Int = 25;
    output result: Rows<Bool>;
    compute { return at(count, 0) >= target; }
}
kernel w.boom@1 {
    requires { field.read; field.write; }
    input count: Rows<Int>;
    output result: Rows<Float>;
    compute {
        if at(count, 0) >= 30 { return 0.0 / 0.0; }
        return 0.0;
    }
}
kernel w.view@1 {
    requires { field.read; field.write; }
    input count: Rows<Int>;
    param gate: Int = 0;
    output result: Image;
    compute {
        let safe = 1 / gate;
        return vec4(float(at(count, 0) + safe) / 100.0, 0.0, 0.0, 1.0);
    }
}
param cap: Int = 1000;
param target: Int = 25;
param gate: Int = 0;
node seed: w.seed@1 { domain = 1; }
node counter: w.step@1 { prev <~ counter.result from seed.result; cap = cap; domain = 1; }
node reached: w.done@1 { count <- counter.result; target = target; domain = 1; }
node unstable: w.boom@1 { count <- counter.result; domain = 1; }
node view: w.view@1 { count <- counter.result; gate = gate; domain = vec2(8, 8); }
output result <- view.result;
";
        std::fs::write(dir.join("scene.esn"), scene).unwrap();
        let mut session = Session::open(
            &dir.join("scene.esn"),
            None,
            entelechy_session::DEFAULT_BUDGET_WORDS,
        );
        session.sync();

        // 1. Step 10 ticks without `until`: `view` (`1 / 0`) is never run.
        let w1 = handle(
            &mut session,
            &json!({ "op": "evaluate", "steps": 10, "dt": 0.1 }),
        )
        .unwrap();
        assert_eq!(w1["steps"], 10);
        assert_eq!(w1["active"], 10);
        assert_eq!(w1["stopped"], "steps");
        assert!((w1["time"].as_f64().unwrap() - 0.9).abs() < 1e-9);
        assert!(!w1["executed"].as_array().unwrap().contains(&json!("view")));

        // 2. Next step run without `time` continues from 1.0 s and stops when `reached` becomes true (at count == 25, 15 ticks later).
        let w2 = handle(
            &mut session,
            &json!({ "op": "evaluate", "steps": 100, "dt": 0.1, "until": "reached" }),
        )
        .unwrap();
        assert_eq!(w2["steps"], 15, "{w2}");
        assert_eq!(w2["stopped"], "until", "{w2}");
        assert_eq!(w2["until"], 1.0, "{w2}");
        assert!((w2["time"].as_f64().unwrap() - 2.4).abs() < 1e-9);
        let c_summary = handle(&mut session, &json!({ "op": "read", "node": "counter" })).unwrap();
        assert_eq!(c_summary["summary"][0]["max"], 25.0);

        // 3. Stepping with `until: "unstable"` stops immediately on NaN at count == 30 (5 ticks later).
        let w3 = handle(
            &mut session,
            &json!({ "op": "evaluate", "steps": 100, "dt": 0.1, "until": "unstable" }),
        )
        .unwrap();
        assert_eq!(w3["steps"], 5, "{w3}");
        assert_eq!(w3["stopped"], "nonfinite", "{w3}");
        assert_eq!(w3["until"], "NaN", "{w3}");

        // 4. Cap the counter at 32 in a clockless scene: stepping 500 ticks stops early once settled!
        handle(
            &mut session,
            &json!({ "op": "param", "name": "cap", "value": 32 }),
        )
        .unwrap();
        let w4 = handle(
            &mut session,
            &json!({ "op": "evaluate", "steps": 500, "dt": 0.1 }),
        )
        .unwrap();
        assert_eq!(w4["stopped"], "settled", "{w4}");
        assert!(w4["steps"].as_u64().unwrap() <= 4, "{w4}");

        // 5. Open `gate = 1` and run a Porch `evaluate`: viewport renders count == 32 cleanly.
        handle(
            &mut session,
            &json!({ "op": "param", "name": "gate", "value": 1 }),
        )
        .unwrap();
        let ev = handle(&mut session, &json!({ "op": "evaluate", "time": 4.0 })).unwrap();
        assert!(ev["executed"].as_array().unwrap().contains(&json!("view")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn summary_reports_rms_extrema_coordinates_and_nonfinite_places_for_trace() {
        let dir =
            std::env::temp_dir().join(format!("entelechy-protocol-summary-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let scene = "\
kernel s.grid@1 {
    requires { field.write; }
    output result: Grid<Vec2>;
    compute {
        let c = coord();
        let x = float(c.x - 1) * 3.0 + float(c.y) * 4.0;
        if c.x == 2 && c.y == 1 {
            return vec2(x, 0.0 / 0.0);
        }
        if c.x == 3 && c.y == 1 {
            return vec2(x, 1.0 / 0.0);
        }
        return vec2(x, -2.0);
    }
}
kernel s.pic@1 {
    requires { field.read; field.write; }
    input g: Grid<Vec2>;
    output result: Image;
    compute { return vec4(sample(g).x, 0.0, 0.0, 1.0); }
}
node g: s.grid@1 { domain = vec2(4, 2); }
node pic: s.pic@1 { g <- g.result; domain = vec2(4, 2); }
output result <- pic.result;
";
        std::fs::write(dir.join("scene.esn"), scene).unwrap();
        let mut session = Session::open(
            &dir.join("scene.esn"),
            None,
            entelechy_session::DEFAULT_BUDGET_WORDS,
        );
        session.sync();
        handle(&mut session, &json!({ "op": "pin", "node": "g" })).unwrap();
        handle(&mut session, &json!({ "op": "evaluate", "time": 0.0 })).unwrap();

        let digest = handle(
            &mut session,
            &json!({ "op": "read", "node": "g", "format": "summary" }),
        )
        .unwrap();
        let ch0 = &digest["summary"][0];
        // At (0, 0): -3; at (3, 1): (3-1)*3 + 1*4 = 10.
        assert_eq!(ch0["min"], -3.0);
        assert_eq!(ch0["min_at"], json!([0, 0]));
        assert_eq!(ch0["max"], 10.0);
        assert_eq!(ch0["max_at"], json!([3, 1]));
        assert!(ch0["rms"].as_f64().unwrap() > 0.0);
        assert!(ch0.get("nonfinite").is_none());

        let ch1 = &digest["summary"][1];
        assert_eq!(ch1["nonfinite"], 2);
        assert_eq!(ch1["nan"], 1);
        assert_eq!(ch1["inf"], 1);
        assert_eq!(ch1["first_nonfinite"], json!([2, 1]));
        assert!((ch1["rms"].as_f64().unwrap() - 2.0).abs() < 1e-9);

        // Feed `first_nonfinite` straight into `trace`!
        let traced = handle(
            &mut session,
            &json!({ "op": "trace", "node": "g", "at": ch1["first_nonfinite"] }),
        )
        .unwrap();
        assert_eq!(traced["at"], json!([2, 1, 0]));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trial_rejects_bad_edits_without_touching_disk_and_commits_clean_ones() {
        let (dir, mut session) = session("trial_gates");
        session.sync();
        // Establish baseline live evaluation.
        handle(&mut session, &json!({ "op": "evaluate", "time": 0.0 })).unwrap();
        let disk_before = std::fs::read_to_string(dir.join("scene.esn")).unwrap();

        // 1) Compile error is caught cleanly with zero disk changes.
        let bad_syntax = handle(
            &mut session,
            &json!({
                "op": "trial",
                "text": "node gradient: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return nope; } }\noutput result <- gradient.result;\n",
                "commit": true
            }),
        )
        .unwrap();
        assert_eq!(bad_syntax["passed"], false);
        assert_eq!(bad_syntax["committed"], false);
        assert_eq!(bad_syntax["written"], false);
        assert_eq!(bad_syntax["gates"][0]["name"], "compile");
        assert_eq!(bad_syntax["gates"][0]["passed"], false);
        assert!(
            bad_syntax["fault"]["message"]
                .as_str()
                .unwrap()
                .contains("nope")
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("scene.esn")).unwrap(),
            disk_before
        );

        // 2) NaN output is rejected by the `finite` gate with automatic element `trace`
        //    even when `commit: true`, leaving disk and live session untouched!
        let nan_scene = "node gradient: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { let bad = if icoord().x == 3 && icoord().y == 2 { 0.0 / 0.0 } else { 0.5 }; return vec4(bad, 0.2, 0.4, 1.0); } }\noutput result <- gradient.result;\n";
        let nan_trial = handle(
            &mut session,
            &json!({
                "op": "trial",
                "text": nan_scene,
                "forbid_nonfinite": true,
                "commit": true
            }),
        )
        .unwrap();
        assert_eq!(nan_trial["passed"], false);
        assert_eq!(nan_trial["committed"], false);
        assert_eq!(nan_trial["written"], false);
        let finite_gate = nan_trial["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["name"] == "finite")
            .unwrap();
        assert_eq!(finite_gate["passed"], false);
        assert_eq!(finite_gate["nonfinite"], 1);
        assert_eq!(nan_trial["trace"]["at"], json!([3, 2, 0]));
        assert_eq!(
            std::fs::read_to_string(dir.join("scene.esn")).unwrap(),
            disk_before
        );

        // 3) Valid edit passes dry-run trial (`commit: false`), computes diff vs baseline,
        //    and leaves disk untouched.
        let bright_scene = "node gradient: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(1.0, coord().y / 8.0, 0.5, 1.0); } }\noutput result <- gradient.result;\n";
        let dry = handle(
            &mut session,
            &json!({
                "op": "trial",
                "text": bright_scene,
                "steps": 2,
                "include_png": true,
                "commit": false
            }),
        )
        .unwrap();
        assert_eq!(dry["passed"], true);
        assert_eq!(dry["committed"], false);
        assert_eq!(dry["written"], false);
        assert_eq!(dry["steps"], 2);
        assert!(dry["outputs"][0]["diff"]["changed"].as_u64().unwrap() > 0);
        assert!(dry["outputs"][0]["png"].as_str().is_some());
        assert_eq!(
            std::fs::read_to_string(dir.join("scene.esn")).unwrap(),
            disk_before
        );

        // 4) Valid edit with `commit: true` atomically writes and syncs into live session.
        let promoted = handle(
            &mut session,
            &json!({
                "op": "trial",
                "text": bright_scene,
                "commit": true
            }),
        )
        .unwrap();
        assert_eq!(promoted["passed"], true);
        assert_eq!(promoted["committed"], true);
        assert_eq!(promoted["written"], true);
        assert_eq!(
            std::fs::read_to_string(dir.join("scene.esn")).unwrap(),
            bright_scene
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn diagnose_and_catalog_report_structured_world_state() {
        let (dir, mut session) = session("diagnose_catalog");
        // Add a local kernel file beside the scene for `catalog`.
        std::fs::write(
            dir.join("helpers.ent"),
            "/// Doubles a float row.\nkernel local.double@1 {\n    requires { field.read; field.write; }\n    input source: Rows<Float>;\n    output result: Rows<Float>;\n    compute { return sample(source) * 2.0; }\n}\n",
        )
        .unwrap();
        session.sync();

        let cat = handle(
            &mut session,
            &json!({ "op": "catalog", "package": "local", "query": "double" }),
        )
        .unwrap();
        assert_eq!(cat["total"], 1);
        assert_eq!(cat["entries"][0]["id"], "local.double@1");
        assert_eq!(cat["entries"][0]["mode"], "map");

        let diag = handle(&mut session, &json!({ "op": "diagnose", "measure": true })).unwrap();
        assert_eq!(diag["healthy"], true);
        assert_eq!(diag["loaded"], true);
        assert!(!diag["hotspots"].as_array().unwrap().is_empty());
        assert_eq!(diag["transfers"]["downloaded_bytes"], 0);
        assert_eq!(diag["outputs"][0]["nonfinite"], 0);
        assert!(diag["memory"]["budget"].as_u64().unwrap() > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
