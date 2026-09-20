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
//! or `{"id", "error": {"message", "line"?, "column"?}}`.
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
//! | `window` | `size` (`[w, h]` in pixels, or null for none), `scale`? | the window a headless session reports to evaluations that name none |
//! | `read` | `output`? or `node`+`port`, `format`? = `"rgba32f"` / `"ppm"` / `"field"` | width, height, format, data (base64); words for a Field that is not pixels; `field` is the Host's file format for any Field |
//! | `write` | `node`, `port`?, `data` (base64 words, or a `field` file with `format`) | replaces the node's committed state; the port must be read by a `<~` binding |
//! | `reset` | | ok |
//! | `pin` / `unpin` | `node` | keeps a node materialized so `read` can see it |
//! | `quit` | | bye |

use std::io::{BufRead, Write};

use orla_graph::{Binding, ParamValue, Source};
use orla_ir::{Type, Validated};
use orla_runtime::{Context, Expect, Generation, InputState, KernelId, Manifest, Shape, Value};
use serde_json::{Value as Json, json};

use crate::ppm;
use crate::session::{Event, Session};

/// Serves requests from `input` until EOF or `quit`, writing responses to
/// `output`.
pub fn serve(
    session: &mut Session,
    input: impl BufRead,
    mut output: impl Write,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (response, quit) = respond_line(session, &line);
        writeln!(output, "{response}")?;
        output.flush()?;
        if quit {
            break;
        }
    }
    Ok(())
}

/// One request line and where its answer goes: the process's standard
/// output, or the socket it came in on. The transports below feed these
/// to whoever serves the session, a loop or a window.
pub struct Incoming {
    /// The request line.
    pub line: String,
    /// Where the response line goes.
    pub reply: std::sync::mpsc::Sender<String>,
    /// Whether it came on standard input, where `quit` ends the process;
    /// on a socket, `quit` ends that connection.
    pub local: bool,
}

/// Feeds standard input's lines to `sink` and answers on standard
/// output, from threads of its own.
pub fn stdin_into(sink: std::sync::mpsc::Sender<Incoming>) {
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdout = std::io::stdout();
        for line in answers {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
    });
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let incoming = Incoming {
                line,
                reply: reply.clone(),
                local: true,
            };
            if sink.send(incoming).is_err() {
                break;
            }
        }
    });
}

/// Listens on `address` (`127.0.0.1:7788`, or port 0 for any) and feeds
/// every connection's lines to `sink`; each connection's answers go back
/// on its socket, one line each, from a thread of its own. Returns the
/// address bound. The protocol is the same on a socket as on standard
/// input, so a tool in another program, Blender say, talks to a running
/// window. A connection that opens with an HTTP request is a browser:
/// it is upgraded to a WebSocket and the same lines travel as text
/// frames ([`crate::websocket`]).
pub fn listen(
    address: &str,
    sink: std::sync::mpsc::Sender<Incoming>,
) -> std::io::Result<std::net::SocketAddr> {
    let listener = std::net::TcpListener::bind(address)?;
    let bound = listener.local_addr()?;
    std::thread::spawn(move || {
        for stream in listener.incoming().map_while(Result::ok) {
            let sink = sink.clone();
            std::thread::spawn(move || {
                let mut opening = [0u8; 4];
                let websocket = matches!(stream.peek(&mut opening), Ok(4) if &opening == b"GET ");
                if websocket {
                    serve_websocket(stream, sink);
                } else {
                    serve_lines(stream, sink);
                }
            });
        }
    });
    Ok(bound)
}

/// One connection of JSON lines.
fn serve_lines(stream: std::net::TcpStream, sink: std::sync::mpsc::Sender<Incoming>) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in answers {
            // An empty line is the sign to close: sent after a quit's
            // answer, it ends the connection from this side.
            if line.is_empty()
                || writeln!(writer, "{line}")
                    .and_then(|_| writer.flush())
                    .is_err()
            {
                break;
            }
        }
        let _ = writer.shutdown(std::net::Shutdown::Both);
    });
    let reader = std::io::BufReader::new(stream);
    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let incoming = Incoming {
            line,
            reply: reply.clone(),
            local: false,
        };
        if sink.send(incoming).is_err() {
            break;
        }
    }
}

/// One connection of WebSocket frames: the handshake, then a text frame
/// per request line and one per answer; pings are answered, a close
/// frame or a quit ends it.
fn serve_websocket(stream: std::net::TcpStream, sink: std::sync::mpsc::Sender<Incoming>) {
    use crate::websocket::{self, Message};
    let Ok(writer) = stream.try_clone() else {
        return;
    };
    let writer = std::sync::Arc::new(std::sync::Mutex::new(writer));
    let mut reader = std::io::BufReader::new(stream);
    let opened = {
        let mut guard = writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        websocket::handshake(&mut reader, &mut *guard)
    };
    if !matches!(opened, Ok(true)) {
        let guard = writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = guard.shutdown(std::net::Shutdown::Both);
        return;
    }
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    let framer = writer.clone();
    std::thread::spawn(move || {
        for line in answers {
            let mut guard = framer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let written = if line.is_empty() {
                websocket::write_close(&mut *guard).and(Err(std::io::Error::other("closed")))
            } else {
                websocket::write_text(&mut *guard, &line)
            };
            if written.is_err() {
                let _ = guard.shutdown(std::net::Shutdown::Both);
                break;
            }
        }
    });
    loop {
        match websocket::read_message(&mut reader) {
            Ok(Some(Message::Text(line))) => {
                if line.trim().is_empty() {
                    continue;
                }
                let incoming = Incoming {
                    line,
                    reply: reply.clone(),
                    local: false,
                };
                if sink.send(incoming).is_err() {
                    break;
                }
            }
            Ok(Some(Message::Ping(payload))) => {
                let mut guard = writer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if websocket::write_pong(&mut *guard, &payload).is_err() {
                    break;
                }
            }
            Ok(Some(Message::Close)) | Ok(None) | Err(_) => {
                let _ = reply.send(String::new());
                break;
            }
        }
    }
}

/// Answers one incoming request; returns whether the process should end
/// (a `quit` on standard input). A `quit` on a socket answers, then
/// closes that connection.
pub fn answer(session: &mut Session, incoming: Incoming) -> bool {
    let (response, quit) = respond_line(session, &incoming.line);
    let _ = incoming.reply.send(response);
    if quit && !incoming.local {
        let _ = incoming.reply.send(String::new());
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

/// The context an `evaluate` or `trace` request describes: time, rate,
/// the pointer, buttons, wheel and keys, and the window (the request's,
/// else the session's, else none).
pub fn context_of(session: &Session, request: &Json) -> Context {
    let number_at = |list: &Vec<Json>, index: usize| {
        list.get(index).and_then(Json::as_f64).unwrap_or(0.0) as f32
    };
    // The window: the request's, else the session's, else none.
    let (window, scale) = match request.get("window").and_then(Json::as_array) {
        Some(w) => (
            [number_at(w, 0), number_at(w, 1)],
            request.get("scale").and_then(Json::as_f64).unwrap_or(1.0) as f32,
        ),
        None => session.window().unwrap_or(([0.0, 0.0], 1.0)),
    };
    let mut context = Context {
        time: request.get("time").and_then(Json::as_f64).unwrap_or(0.0) as f32,
        rate: request.get("rate").and_then(Json::as_f64).unwrap_or(0.0) as f32,
        input: InputState::default(),
        window,
        scale,
    };
    if let Some(pointer) = request.get("pointer").and_then(Json::as_array) {
        context.input.pointer = [
            pointer.first().and_then(Json::as_f64).unwrap_or(0.0) as f32,
            pointer.get(1).and_then(Json::as_f64).unwrap_or(0.0) as f32,
        ];
    }
    context.input.pointer_down = request.get("down").and_then(Json::as_bool).unwrap_or(false);
    if let Some(buttons) = request.get("buttons").and_then(Json::as_array) {
        for button in buttons.iter().filter_map(Json::as_i64) {
            match button {
                0 => context.input.pointer_down = true,
                1 => context.input.pointer_right = true,
                2 => context.input.pointer_middle = true,
                _ => {}
            }
        }
    }
    if let Some(wheel) = request.get("wheel").and_then(Json::as_array) {
        context.input.wheel = [
            wheel.first().and_then(Json::as_f64).unwrap_or(0.0) as f32,
            wheel.get(1).and_then(Json::as_f64).unwrap_or(0.0) as f32,
        ];
    }
    if let Some(keys) = request.get("keys").and_then(Json::as_array) {
        for key in keys.iter().filter_map(Json::as_i64) {
            context.input.set_key(key as i32, true);
        }
    }
    context
}

/// The JSON form of an evaluation report.
pub fn report_json(session: &Session, report: &orla_graph::Report) -> Json {
    let host = session.runtime().host();
    let outputs = report
        .outputs
        .iter()
        .map(|(name, handle)| {
            let (width, height) = host.image_size(*handle).unwrap_or((0, 0));
            json!({ "name": name, "width": width, "height": height })
        })
        .collect::<Vec<_>>();
    json!({
        "executed": report.executed.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "reused": report.reused.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "committed": report.committed.iter().map(|(node, port)| json!([node.to_string(), port])).collect::<Vec<_>>(),
        "fused": report.fused.iter().map(|(node, absorbed)| json!({ "node": node.to_string(), "absorbed": absorbed.iter().map(ToString::to_string).collect::<Vec<_>>() })).collect::<Vec<_>>(),
        "outputs": outputs,
        "timings": report.timings.iter().map(|(node, took)| json!({ "node": node.to_string(), "millis": took.as_secs_f64() * 1000.0 })).collect::<Vec<_>>(),
        "migrated": report.migrated.iter().map(|(node, port, kernel)| json!({ "node": node.to_string(), "port": port, "kernel": kernel.to_string() })).collect::<Vec<_>>(),
        "reset": report.reset.iter().map(|(node, port, reason)| json!({ "node": node.to_string(), "port": port, "reason": reason })).collect::<Vec<_>>(),
    })
}

/// Words a value of `ty` takes.
fn words_of(module: &Validated, ty: Type) -> usize {
    match ty {
        Type::Float | Type::Int | Type::Bool => 1,
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 | Type::Mat2 => 4,
        Type::IVec2 => 2,
        Type::IVec3 => 3,
        Type::Mat3 => 9,
        Type::Mat4 => 16,
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .map(|field| words_of(module, field.ty))
            .sum(),
    }
}

/// A type's name as the source spells it: a struct by its name.
pub fn type_name(module: &Validated, ty: Type) -> String {
    match ty {
        Type::Struct(index) => module.structs[index as usize].name.clone(),
        other => other.to_string(),
    }
}

/// A traced value's words as JSON: numbers, booleans, vectors as arrays,
/// structs as objects by field name.
pub fn words_json(module: &Validated, ty: Type, words: &[u32]) -> Json {
    match ty {
        Type::Float => json!(f32::from_bits(words[0])),
        Type::Int => json!(words[0] as i32),
        Type::Bool => json!(words[0] != 0),
        Type::Vec2 | Type::Vec3 | Type::Vec4 => {
            json!(words.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>())
        }
        Type::IVec2 | Type::IVec3 => json!(words.iter().map(|w| *w as i32).collect::<Vec<_>>()),
        Type::Mat2 | Type::Mat3 | Type::Mat4 => {
            let n = usize::from(ty.order().unwrap_or(0));
            json!(
                words
                    .chunks(n)
                    .map(|column| column
                        .iter()
                        .map(|w| f32::from_bits(*w))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            )
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut cursor = 0;
            if let Some(element) = def.element {
                let n = words_of(module, element);
                return json!(
                    (0..def.fields.len())
                        .map(|k| words_json(module, element, &words[k * n..(k + 1) * n]))
                        .collect::<Vec<_>>()
                );
            }
            let mut object = serde_json::Map::new();
            for field in &def.fields {
                let n = words_of(module, field.ty);
                object.insert(
                    field.name.clone(),
                    words_json(module, field.ty, &words[cursor..cursor + n]),
                );
                cursor += n;
            }
            Json::Object(object)
        }
    }
}

/// A traced value's words as text, the way the source would write them;
/// an `Int` of sixteen or more also in hex.
pub fn words_text(module: &Validated, ty: Type, words: &[u32]) -> String {
    match ty {
        Type::Float => format!("{}", f32::from_bits(words[0])),
        Type::Int => {
            let v = words[0] as i32;
            if v.unsigned_abs() >= 16 {
                format!("{v} (0x{:X})", words[0])
            } else {
                v.to_string()
            }
        }
        Type::Bool => (words[0] != 0).to_string(),
        Type::Vec2 | Type::Vec3 | Type::Vec4 => format!(
            "vec{}({})",
            words.len(),
            words
                .iter()
                .map(|w| f32::from_bits(*w).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::IVec2 | Type::IVec3 => format!(
            "ivec{}({})",
            words.len(),
            words
                .iter()
                .map(|w| (*w as i32).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Mat2 | Type::Mat3 | Type::Mat4 => {
            let n = usize::from(ty.order().unwrap_or(0));
            let column = ty.column().unwrap_or(Type::Float);
            format!(
                "mat{n}({})",
                words
                    .chunks(n)
                    .map(|c| words_text(module, column, c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut cursor = 0;
            if let Some(element) = def.element {
                let n = words_of(module, element);
                let elements: Vec<String> = (0..def.fields.len())
                    .map(|k| words_text(module, element, &words[k * n..(k + 1) * n]))
                    .collect();
                return format!("[{}]", elements.join(", "));
            }
            let fields: Vec<String> = def
                .fields
                .iter()
                .map(|field| {
                    let n = words_of(module, field.ty);
                    let text = format!(
                        "{}: {}",
                        field.name,
                        words_text(module, field.ty, &words[cursor..cursor + n])
                    );
                    cursor += n;
                    text
                })
                .collect();
            format!("{} {{ {} }}", def.name, fields.join(", "))
        }
    }
}

/// The JSON form of a trace: the node, its Kernel, the element, every
/// binding and every result with its type and value.
pub fn trace_json(
    module: &Validated,
    node: &str,
    kernel: &KernelId,
    trace: &orla_cpu::Trace,
) -> Json {
    let values = |list: &[orla_cpu::TraceValue]| {
        list.iter()
            .map(|value| json!({ "name": value.name, "type": type_name(module, value.ty), "value": words_json(module, value.ty, &value.words), "ran": value.ran }))
            .collect::<Vec<_>>()
    };
    json!({
        "node": node,
        "kernel": kernel.to_string(),
        "at": trace.at,
        "bindings": values(&trace.bindings),
        "calls": trace.calls.iter().map(|call| json!({ "function": call.function, "bindings": values(&call.bindings), "results": values(&call.results) })).collect::<Vec<_>>(),
        "calls_truncated": trace.calls_truncated,
        "results": values(&trace.results),
    })
}

/// The Kernel a request names: in `kernel`, or, for clients from before
/// the field had its own name, in `id`, which is also the request's own.
fn kernel_field(request: &Json) -> Result<KernelId, Json> {
    let named = request
        .get("kernel")
        .or_else(|| request.get("id"))
        .and_then(Json::as_str)
        .ok_or_else(|| failure("`kernel` names the Kernel, for example `image.blur@1`"))?;
    named
        .parse::<KernelId>()
        .map_err(|e| failure(e.to_string()))
}

fn failure(message: impl Into<String>) -> Json {
    json!({ "message": message.into() })
}

/// A diagnostic with a position when the message starts with `line:col:`.
fn positioned(message: String) -> Json {
    let mut parts = message.splitn(3, ':');
    if let (Some(line), Some(column), Some(rest)) = (parts.next(), parts.next(), parts.next())
        && let (Ok(line), Ok(column)) = (line.trim().parse::<u32>(), column.trim().parse::<u32>())
    {
        return json!({ "message": rest.trim(), "line": line, "column": column });
    }
    failure(message)
}

/// Handles one request.
pub fn handle(session: &mut Session, request: &Json) -> Result<Json, Json> {
    let op = request
        .get("op")
        .and_then(Json::as_str)
        .ok_or_else(|| failure("request needs an `op`"))?;
    let field = |name: &str| {
        request
            .get(name)
            .ok_or_else(|| failure(format!("`{op}` needs `{name}`")))
    };
    let text_field = |name: &str| {
        field(name)?
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| failure(format!("`{name}` must be a string")))
    };
    match op {
        "status" => Ok(json!({
            "versions": { "api": orla_runtime::ABI_VERSION, "language": orla_language::LANGUAGE_VERSION, "ir": orla_ir::IR_VERSION, "package": crate::session::PACKAGE_VERSION },
            "backend": session.runtime().backend_name(),
            "scene": session.root().display().to_string(),
            "watched": session.watched().iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "kernels": session.runtime().kernels().count(),
            "nodes": session.graph().map_or(0, |graph| graph.nodes.len()),
            "fusion": session.fusion(),
        })),
        "kernels" => Ok(json!({
            "kernels": session
                .runtime()
                .kernels()
                .map(|version| manifest_json(version.manifest(), version.generation()))
                .collect::<Vec<_>>()
        })),
        "kernel" => {
            let id = kernel_field(request)?;
            let version = session
                .runtime()
                .kernel(&id)
                .ok_or_else(|| failure(format!("{id} is not published")))?;
            let mut json = manifest_json(version.manifest(), version.generation());
            json["source"] = Json::String(version.source().to_owned());
            Ok(json)
        }
        "graph" => {
            let graph = session
                .graph()
                .ok_or_else(|| failure("no scene is loaded"))?;
            Ok(graph_json(graph, session.params()))
        }
        "document" => {
            let text = std::fs::read_to_string(session.root())
                .map_err(|e| failure(format!("{}: {e}", session.root().display())))?;
            Ok(json!({ "path": session.root().display().to_string(), "text": text }))
        }
        "compose" => {
            let text = text_field("text")?;
            std::fs::write(session.root(), text)
                .map_err(|e| failure(format!("{}: {e}", session.root().display())))?;
            Ok(json!({ "events": events_json(&session.sync()) }))
        }
        "sync" => Ok(json!({ "events": events_json(&session.sync()) })),
        "prepare" => {
            let source = text_field("source")?;
            let candidate = session.prepare_source(&source).map_err(positioned)?;
            let current = session
                .runtime()
                .kernel(candidate.id())
                .map(|version| version.generation().0);
            let mut json = manifest_json(candidate.manifest(), Generation(0));
            json["generation"] = Json::Null;
            json["current"] = current.map_or(Json::Null, Json::from);
            Ok(json)
        }
        "publish" => {
            let source = text_field("source")?;
            let expect = match request.get("expect") {
                None | Some(Json::Null) => Expect::Any,
                Some(Json::String(s)) if s == "any" => Expect::Any,
                Some(Json::String(s)) if s == "absent" => Expect::Absent,
                Some(Json::Number(n)) => Expect::Generation(Generation(
                    n.as_u64()
                        .ok_or_else(|| failure("`expect` generation must be a whole number"))?,
                )),
                Some(_) => {
                    return Err(failure(
                        "`expect` is \"absent\", \"any\", or a generation number",
                    ));
                }
            };
            let version = session
                .publish_source(&source, expect)
                .map_err(positioned)?;
            Ok(json!({ "id": version.id().to_string(), "generation": version.generation().0 }))
        }
        "retire" => {
            let id = kernel_field(request)?;
            Ok(json!({ "retired": session.retire_kernel(&id) }))
        }
        "param" => {
            let name = text_field("name")?;
            let value = value_from_json(field("value")?)?;
            session.set_param(&name, value);
            // A param that sizes the graph lowers the document again now.
            let events = session.sync();
            Ok(json!({ "name": name, "value": value_json(&value), "events": events_json(&events) }))
        }
        "evaluate" => {
            let context = context_of(session, request);
            let report = match request.get("nodes").and_then(Json::as_array) {
                Some(nodes) => {
                    let demand = nodes
                        .iter()
                        .filter_map(Json::as_str)
                        .map(|node| orla_graph::NodeId(node.to_owned()))
                        .collect();
                    session.evaluate_for(&context, &demand)
                }
                None => session.evaluate(&context),
            }
            .map_err(|e| failure(e.to_string()))?;
            Ok(report_json(session, &report))
        }
        "help" => {
            let entry = |builtin: &orla_compiler::help::Builtin| {
                json!({
                    "name": builtin.name,
                    "kind": builtin.kind,
                    "signatures": builtin.signatures,
                    "doc": builtin.doc,
                    "requires": builtin.requires,
                    "entry_only": builtin.entry_only,
                })
            };
            match request.get("name").and_then(Json::as_str) {
                Some(name) => match orla_compiler::help::builtin(name) {
                    Some(builtin) => Ok(entry(&builtin)),
                    None => Err(failure(format!(
                        "no builtin named `{name}`; similar: {}",
                        orla_compiler::help::similar(name).join(", ")
                    ))),
                },
                None => Ok(json!({
                    "builtins": orla_compiler::help::builtins().iter().map(entry).collect::<Vec<_>>()
                })),
            }
        }
        "trace" => {
            let node = text_field("node")?;
            let at = match (
                request.get("at").and_then(Json::as_array),
                request.get("element").and_then(Json::as_u64),
            ) {
                (Some(list), _) => {
                    let axis = |i: usize| list.get(i).and_then(Json::as_u64).unwrap_or(0) as u32;
                    [axis(0), axis(1), axis(2)]
                }
                (None, Some(element)) => [element as u32, 0, 0],
                (None, None) => {
                    return Err(failure(
                        "`trace` names the element as `at`: [x, y, z] or `element`: n",
                    ));
                }
            };
            let context = context_of(session, request);
            let id = orla_graph::NodeId(node.clone());
            let (trace, kernel) = session.trace(&id, at, &context).map_err(failure)?;
            let version = session
                .runtime()
                .kernel(&kernel)
                .ok_or_else(|| failure(format!("`{kernel}` is not published")))?;
            let module = version
                .module()
                .ok_or_else(|| failure("a native operation has no trace"))?;
            Ok(trace_json(module, &node, &kernel, &trace))
        }
        "write" => {
            let node = text_field("node")?;
            let port = request
                .get("port")
                .and_then(Json::as_str)
                .unwrap_or("result")
                .to_owned();
            let data = text_field("data")?;
            let bytes = unbase64(&data).map_err(failure)?;
            let words: Vec<u32> = if request.get("format").and_then(Json::as_str) == Some("field") {
                // A Field file: its layout and extent must be the port's.
                let data =
                    orla_runtime::file::decode(&bytes).map_err(|e| failure(e.to_string()))?;
                let current = session
                    .cached(&orla_graph::NodeId(node.clone()), &port)
                    .ok_or_else(|| {
                        failure(format!("`{node}.{port}` has no value yet; evaluate first"))
                    })?;
                let host = session.runtime().host();
                let (schema, domain) = (
                    host.schema(current)
                        .map_err(|e| failure(e.to_string()))?
                        .clone(),
                    host.domain(current).map_err(|e| failure(e.to_string()))?,
                );
                if data.schema() != &schema || data.domain() != domain {
                    return Err(failure(format!(
                        "the file is {} {} {}, but `{node}.{port}` is {} {} {}",
                        data.domain().shape(),
                        data.domain().count(),
                        data.schema(),
                        domain.shape(),
                        domain.count(),
                        schema
                    )));
                }
                data.words().to_vec()
            } else {
                if bytes.len() % 4 != 0 {
                    return Err(failure("`data` must be whole 32-bit words"));
                }
                bytes
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            };
            let count = words.len();
            session
                .write(&orla_graph::NodeId(node.clone()), &port, words)
                .map_err(|e| failure(e.to_string()))?;
            Ok(json!({ "node": node, "port": port, "words": count }))
        }
        "read" => {
            let graph = session
                .graph()
                .ok_or_else(|| failure("no scene is loaded"))?;
            let handle = if let Some(node) = request.get("node").and_then(Json::as_str) {
                let port = request
                    .get("port")
                    .and_then(Json::as_str)
                    .unwrap_or("result");
                session
                    .cached(&orla_graph::NodeId(node.to_owned()), port)
                    .ok_or_else(|| {
                        failure(format!(
                            "node `{node}` has no materialized `{port}`; pin it and evaluate again"
                        ))
                    })?
            } else {
                let name = request
                    .get("output")
                    .and_then(Json::as_str)
                    .map(str::to_owned)
                    .or_else(|| graph.outputs.first().map(|(name, _)| name.clone()))
                    .ok_or_else(|| failure("the graph has no output"))?;
                let (_, source) = graph
                    .outputs
                    .iter()
                    .find(|(output, _)| *output == name)
                    .ok_or_else(|| failure(format!("no output named `{name}`")))?;
                match source {
                    Source::Node { node, port } => session
                        .cached(node, port)
                        .ok_or_else(|| failure("evaluate before reading"))?,
                    Source::External(_) => return Err(failure("output is an external image")),
                }
            };
            // A Field of pixels reads as an image; anything else reads as
            // words with its shape and layout.
            let (domain, schema) = {
                let host = session.runtime().host();
                (
                    host.domain(handle).map_err(|e| failure(e.to_string()))?,
                    host.schema(handle)
                        .map_err(|e| failure(e.to_string()))?
                        .clone(),
                )
            };
            let as_file = request.get("format").and_then(Json::as_str) == Some("field");
            if as_file || !schema.is_pixel() || domain.shape() == Shape::Volume {
                session
                    .runtime_mut()
                    .materialize(handle)
                    .map_err(|e| failure(e.to_string()))?;
                let data = session
                    .runtime()
                    .host()
                    .data(handle)
                    .map_err(|e| failure(e.to_string()))?;
                let bytes: Vec<u8> = if as_file {
                    orla_runtime::file::encode_data(data)
                } else {
                    data.words().iter().flat_map(|w| w.to_le_bytes()).collect()
                };
                return Ok(json!({
                    "shape": domain.shape().to_string(),
                    "extent": [domain.width(), domain.height(), domain.depth()],
                    "count": domain.count(),
                    "schema": schema.to_string(),
                    "format": if as_file { "field" } else { "words" },
                    "data": base64(&bytes),
                }));
            }
            let image = session
                .runtime_mut()
                .image(handle)
                .map_err(|e| failure(e.to_string()))?;
            let format = request
                .get("format")
                .and_then(Json::as_str)
                .unwrap_or("rgba32f");
            let bytes = match format {
                "rgba32f" => image
                    .pixels()
                    .iter()
                    .flat_map(|pixel| pixel.iter().flat_map(|v| v.to_le_bytes()))
                    .collect::<Vec<u8>>(),
                "ppm" => ppm::encode(&image),
                other => {
                    return Err(failure(format!(
                        "unknown format `{other}`; use rgba32f or ppm"
                    )));
                }
            };
            Ok(json!({
                "width": image.width(),
                "height": image.height(),
                "format": format,
                "data": base64(&bytes),
            }))
        }
        "reset" => {
            session.reset_state();
            Ok(json!({ "reset": true }))
        }
        "window" => {
            let window = match request.get("size") {
                None | Some(Json::Null) => None,
                Some(Json::Array(size)) if size.len() == 2 => {
                    let side = |index: usize| {
                        size.get(index)
                            .and_then(Json::as_f64)
                            .filter(|v| *v >= 1.0)
                            .map(|v| v as f32)
                    };
                    let scale = request
                        .get("scale")
                        .map_or(Some(1.0), Json::as_f64)
                        .filter(|s| *s > 0.0)
                        .ok_or_else(|| failure("`scale` must be a positive number"))?;
                    match (side(0), side(1)) {
                        (Some(w), Some(h)) => Some(([w, h], scale as f32)),
                        _ => {
                            return Err(failure(
                                "`size` is `[width, height]` in whole pixels, each at least one",
                            ));
                        }
                    }
                }
                Some(_) => {
                    return Err(failure(
                        "`size` is `[width, height]` in pixels, or null for no window",
                    ));
                }
            };
            session.set_window(window);
            Ok(match window {
                Some((size, scale)) => json!({ "window": size, "scale": scale }),
                None => json!({ "window": Json::Null }),
            })
        }
        "pin" => {
            let node = text_field("node")?;
            session.pin(orla_graph::NodeId(node.clone()));
            Ok(json!({ "pinned": node }))
        }
        "unpin" => {
            let node = text_field("node")?;
            session.unpin(&orla_graph::NodeId(node.clone()));
            Ok(json!({ "unpinned": node }))
        }
        "quit" => Ok(json!({ "bye": true })),
        other => Err(failure(format!("unknown op `{other}`"))),
    }
}

fn manifest_json(manifest: &Manifest, generation: Generation) -> Json {
    json!({
        "id": manifest.id.to_string(),
        "generation": generation.0,
        "mode": manifest.mode.to_string(),
        "requires": manifest.requires.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        "inputs": manifest.inputs.iter().map(|p| json!({ "name": p.name, "type": p.ty.to_string() })).collect::<Vec<_>>(),
        "params": manifest.params.iter().map(|p| json!({ "name": p.name, "type": p.default.ty().to_string(), "default": value_json(&p.default) })).collect::<Vec<_>>(),
        "outputs": manifest.outputs.iter().map(|p| json!({ "name": p.name, "type": p.ty.to_string() })).collect::<Vec<_>>(),
        "memories": manifest.memories,
    })
}

fn graph_json(
    graph: &orla_graph::Graph,
    overrides: &std::collections::BTreeMap<String, Value>,
) -> Json {
    let source = |source: &Source| source.to_string();
    json!({
        "name": graph.name,
        "inputs": graph.externals,
        "params": graph.params.iter().map(|p| json!({
            "name": p.name,
            "type": p.default.ty().to_string(),
            "default": value_json(&p.default),
            "value": value_json(overrides.get(&p.name).unwrap_or(&p.default)),
        })).collect::<Vec<_>>(),
        "nodes": graph.nodes.iter().map(|node| json!({
            "id": node.id.to_string(),
            "kernel": node.kernel.to_string(),
            "inputs": node.inputs.iter().map(|(port, binding)| (port.clone(), match binding {
                Binding::Now(s) => json!({ "kind": "now", "source": source(s) }),
                Binding::Previous { source: s, initial, migration } => json!({ "kind": "previous", "source": source(s), "initial": source(initial), "migrate": migration.as_ref().map(ToString::to_string) }),
            })).collect::<serde_json::Map<_, _>>(),
            "params": node.params.iter().map(|(name, value)| (name.clone(), match value {
                ParamValue::Literal(v) => value_json(v),
                ParamValue::Graph(p) => json!({ "param": p }),
            })).collect::<serde_json::Map<_, _>>(),
            "domain": node.domain.as_ref().map(|d| match d {
                orla_graph::DomainSpec::Literal(orla_runtime::Domain::Rows(n)) => json!([n]),
                orla_graph::DomainSpec::Literal(orla_runtime::Domain::Grid(w, h)) => json!([w, h]),
                orla_graph::DomainSpec::Literal(orla_runtime::Domain::Volume(w, h, d)) => json!([w, h, d]),
                orla_graph::DomainSpec::Expr(expr) => json!(expr.to_string()),
            }),
        })).collect::<Vec<_>>(),
        "outputs": graph.outputs.iter().map(|(name, s)| json!({ "name": name, "source": source(s) })).collect::<Vec<_>>(),
    })
}

fn events_json(events: &[Event]) -> Vec<Json> {
    events
        .iter()
        .map(|event| match event {
            Event::Preparing { path, id } => json!({ "kind": "preparing", "path": path.display().to_string(), "id": id.to_string() }),
            Event::Published {
                path,
                id,
                generation,
            } => json!({ "kind": "published", "path": path.display().to_string(), "id": id.to_string(), "generation": generation.0 }),
            Event::Rejected { path, error } => {
                let mut json = positioned(error.trim_start_matches(' ').to_owned());
                json["kind"] = json!("rejected");
                json["path"] = json!(path.display().to_string());
                json
            }
            Event::Retired { path, id } => json!({ "kind": "retired", "path": path.display().to_string(), "id": id.to_string() }),
            Event::SceneLoaded => json!({ "kind": "scene_loaded" }),
            Event::Loaded { name, path, detail } => json!({ "kind": "loaded", "name": name, "path": path.display().to_string(), "detail": detail }),
            Event::Text { name, chars } => json!({ "kind": "text", "name": name, "chars": chars }),
            Event::SceneRejected(error) => {
                let mut json = positioned(error.clone());
                json["kind"] = json!("scene_rejected");
                json
            }
        })
        .collect()
}

/// Values cross the protocol as JSON numbers, booleans, and float arrays.
pub fn value_json(value: &Value) -> Json {
    match value {
        Value::Float(v) => json!(v),
        Value::Int(v) => json!(v),
        Value::Bool(v) => json!(v),
        Value::Vec2(v) => json!(v),
        Value::Vec3(v) => json!(v),
        Value::Vec4(v) => json!(v),
    }
}

/// Reads a value: integers become `Int`, other numbers `Float`, booleans
/// `Bool`, arrays of two, three, or four numbers vectors.
pub fn value_from_json(json: &Json) -> Result<Value, Json> {
    match json {
        Json::Bool(v) => Ok(Value::Bool(*v)),
        Json::Number(n) => {
            if let Some(i) = n.as_i64().filter(|i| i32::try_from(*i).is_ok()) {
                Ok(Value::Int(i as i32))
            } else {
                Ok(Value::Float(n.as_f64().unwrap_or(0.0) as f32))
            }
        }
        Json::Array(items) => {
            let floats: Vec<f32> = items
                .iter()
                .map(|item| item.as_f64().map(|v| v as f32))
                .collect::<Option<_>>()
                .ok_or_else(|| failure("vector components must be numbers"))?;
            match floats.as_slice() {
                [x, y] => Ok(Value::Vec2([*x, *y])),
                [x, y, z] => Ok(Value::Vec3([*x, *y, *z])),
                [x, y, z, w] => Ok(Value::Vec4([*x, *y, *z, *w])),
                _ => Err(failure("a vector has two, three, or four components")),
            }
        }
        _ => Err(failure("a value is a number, a boolean, or a vector array")),
    }
}

/// Decodes standard base64, with or without padding.
pub fn unbase64(text: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0;
    for byte in text.bytes() {
        if byte == b'=' || byte.is_ascii_whitespace() {
            continue;
        }
        let value = ALPHABET
            .iter()
            .position(|c| *c == byte)
            .ok_or_else(|| format!("`{}` is not base64", byte as char))?;
        acc = (acc << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Standard base64 with padding.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut word = 0u32;
        for (index, byte) in chunk.iter().enumerate() {
            word |= (*byte as u32) << (16 - 8 * index);
        }
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[((word >> (18 - 6 * index)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_what_it_encodes() {
        for sample in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            &[0, 255, 7, 128],
        ] {
            assert_eq!(unbase64(&base64(sample)).unwrap(), sample);
        }
        assert_eq!(unbase64("Zm9v").unwrap(), b"foo");
        assert!(unbase64("Zm9*").is_err());
    }

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
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
        ] {
            assert_eq!(value_from_json(&value_json(&value)).unwrap(), value);
        }
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
