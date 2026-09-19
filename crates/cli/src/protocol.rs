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
//! | `kernel` | `id` | manifest, generation, source |
//! | `graph` | | inputs, params, nodes, outputs |
//! | `document` | | path, text of the scene |
//! | `compose` | `text` | writes the scene, syncs; events |
//! | `sync` | | events |
//! | `prepare` | `source` | id, manifest (nothing published) |
//! | `publish` | `source`, `expect` = `"absent"` / `"any"` / generation | id, generation |
//! | `retire` | `id` | retired |
//! | `param` | `name`, `value` | ok |
//! | `evaluate` | `time`?, `pointer`?, `down`?, `keys`? | executed, reused, committed, outputs |
//! | `read` | `output`? or `node`+`port`, `format`? = `"rgba32f"` / `"ppm"` | width, height, format, data (base64) |
//! | `reset` | | ok |
//! | `pin` / `unpin` | `node` | keeps a node materialized so `read` can see it |
//! | `quit` | | bye |

use std::io::{BufRead, Write};

use orla_graph::{Binding, ParamValue, Source};
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
    })
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
            let id = text_field("id")?
                .parse::<KernelId>()
                .map_err(|e| failure(e.to_string()))?;
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
            let candidate = session
                .runtime()
                .prepare(&source)
                .map_err(|e| positioned(e.to_string()))?;
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
            let id = text_field("id")?
                .parse::<KernelId>()
                .map_err(|e| failure(e.to_string()))?;
            Ok(json!({ "retired": session.retire_kernel(&id) }))
        }
        "param" => {
            let name = text_field("name")?;
            let value = value_from_json(field("value")?)?;
            session.set_param(&name, value);
            Ok(json!({ "name": name, "value": value_json(&value) }))
        }
        "evaluate" => {
            let mut context = Context {
                time: request.get("time").and_then(Json::as_f64).unwrap_or(0.0) as f32,
                input: InputState::default(),
            };
            if let Some(pointer) = request.get("pointer").and_then(Json::as_array) {
                context.input.pointer = [
                    pointer.first().and_then(Json::as_f64).unwrap_or(0.0) as f32,
                    pointer.get(1).and_then(Json::as_f64).unwrap_or(0.0) as f32,
                ];
            }
            context.input.pointer_down =
                request.get("down").and_then(Json::as_bool).unwrap_or(false);
            if let Some(keys) = request.get("keys").and_then(Json::as_array) {
                for key in keys.iter().filter_map(Json::as_i64) {
                    context.input.set_key(key as i32, true);
                }
            }
            let report = session
                .evaluate(&context)
                .map_err(|e| failure(e.to_string()))?;
            Ok(report_json(session, &report))
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
            if !schema.is_pixel() || domain.shape() == Shape::Volume {
                session
                    .runtime_mut()
                    .materialize(handle)
                    .map_err(|e| failure(e.to_string()))?;
                let data = session
                    .runtime()
                    .host()
                    .data(handle)
                    .map_err(|e| failure(e.to_string()))?;
                let bytes: Vec<u8> = data.words().iter().flat_map(|w| w.to_le_bytes()).collect();
                return Ok(json!({
                    "shape": domain.shape().to_string(),
                    "extent": [domain.width(), domain.height(), domain.depth()],
                    "count": domain.count(),
                    "schema": schema.to_string(),
                    "format": "words",
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
                Binding::Previous { source: s, initial } => json!({ "kind": "previous", "source": source(s), "initial": source(initial) }),
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
