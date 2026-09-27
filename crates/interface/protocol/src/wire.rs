//! The JSON wire forms: values, a request's context, evaluation reports,
//! traces, manifests, the Graph, session events and positioned errors.

use orla_graph::{Binding, ParamValue, Source};
use orla_ir::{Type, Validated};
use orla_kernel::{Context, Generation, InputState, KernelId, Manifest, Value};
use serde_json::{Value as Json, json};

use orla_session::{Event, Session};

/// The context an `evaluate`, `trace` or `restore` request describes:
/// time, rate, pointer, buttons, wheel, sticks, keys, and the window.
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
            context.input.set_button(button as i32, true);
        }
    }
    if let Some(wheel) = request.get("wheel").and_then(Json::as_array) {
        context.input.wheel = [
            wheel.first().and_then(Json::as_f64).unwrap_or(0.0) as f32,
            wheel.get(1).and_then(Json::as_f64).unwrap_or(0.0) as f32,
        ];
    }
    // A pad's axes: `"sticks": [[lx, ly], [rx, ry], [lt, rt]]`, up to
    // three pairs.
    if let Some(sticks) = request.get("sticks").and_then(Json::as_array) {
        for (slot, pair) in sticks.iter().take(3).enumerate() {
            if let Some(pair) = pair.as_array() {
                context.input.sticks[slot] = [
                    pair.first().and_then(Json::as_f64).unwrap_or(0.0) as f32,
                    pair.get(1).and_then(Json::as_f64).unwrap_or(0.0) as f32,
                ];
            }
        }
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
        "placements": report.placements.iter().map(|placed| json!({ "node": placed.node.to_string(), "backend": placed.backend, "routed": placed.routed, "uploaded": placed.uploaded, "downloaded": placed.downloaded })).collect::<Vec<_>>(),
        "migrated": report.migrated.iter().map(|(node, port, kernel)| json!({ "node": node.to_string(), "port": port, "kernel": kernel.to_string() })).collect::<Vec<_>>(),
        "reset": report.reset.iter().map(|(node, port, reason)| json!({ "node": node.to_string(), "port": port, "reason": reason })).collect::<Vec<_>>(),
        "pending": report.pending.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "landed": report.landed.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

/// Words a value of `ty` takes.
fn words_of(module: &Validated, ty: Type) -> usize {
    match ty {
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .map(|field| words_of(module, field.ty))
            .sum(),
        other => other.words().unwrap_or(1),
    }
}

/// A value of a type that is not a struct, from its words.
fn plain_value(ty: Type, words: &[u32]) -> Value {
    orla_ir::template(ty, &[]).rebuilt(&mut words.iter().copied())
}

/// A type's source name; a struct by its name.
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
        other => value_json(&plain_value(other, words)),
    }
}

/// A traced value's words as source text; an `Int` of magnitude 16 or
/// more also in hex.
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
        other => plain_value(other, words).to_string(),
    }
}

/// The JSON form of a trace: the node, its Kernel, the element, every
/// binding and every result with its type and value.
pub fn trace_json(
    module: &Validated,
    node: &str,
    kernel: &KernelId,
    trace: &orla_session::Trace,
) -> Json {
    let values = |list: &[orla_session::TraceValue]| {
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

pub(crate) fn failure(message: impl Into<String>) -> Json {
    json!({ "message": message.into() })
}

/// A failed evaluation as data: the message, its node if any, for a
/// failed state migration the port and migration Kernel, and a run's
/// `frame`.
pub(crate) fn evaluation_error(error: &orla_graph::EvaluateError, frame: Option<u64>) -> Json {
    let message = match frame {
        Some(frame) => format!("frame {frame}: {error}"),
        None => error.to_string(),
    };
    let mut json = failure(message);
    if let Some(node) = error.node() {
        json["node"] = json!(node.to_string());
    }
    if let orla_graph::EvaluateError::Migration { port, kernel, .. } = error {
        json["port"] = json!(port);
        json["migration"] = json!(kernel.to_string());
    }
    if let Some(frame) = frame {
        json["frame"] = json!(frame);
    }
    json
}

/// A diagnostic as data: its message, and line and column when placed.
pub(crate) fn diagnostic_json(diagnostic: &orla_session::Diagnostic) -> Json {
    let mut json = failure(diagnostic.message.clone());
    if let Some((line, column)) = diagnostic.at {
        json["line"] = json!(line);
        json["column"] = json!(column);
    }
    json
}

pub(crate) fn manifest_json(
    manifest: &Manifest,
    generation: Generation,
    structs: &[orla_ir::StructDef],
) -> Json {
    let port = |ty: &orla_ir::PortType| orla_ir::port_type_name(ty, structs);
    json!({
        "id": manifest.id.to_string(),
        "generation": generation.0,
        "mode": manifest.mode.to_string(),
        "requires": manifest.requires.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        "inputs": manifest.inputs.iter().map(|p| json!({ "name": p.name, "type": port(&p.ty) })).collect::<Vec<_>>(),
        "params": manifest.params.iter().map(|p| json!({ "name": p.name, "type": orla_ir::type_name(p.ty, structs), "default": value_json(&p.default) })).collect::<Vec<_>>(),
        "outputs": manifest.outputs.iter().map(|p| json!({ "name": p.name, "type": port(&p.ty) })).collect::<Vec<_>>(),
        "memories": manifest.memories,
        "budget": manifest.budget,
    })
}

pub(crate) fn graph_json(
    graph: &orla_graph::Graph,
    overrides: &std::collections::BTreeMap<String, Value>,
    placements: &[(String, orla_kernel::Placement)],
) -> Json {
    let source = |source: &Source| source.to_string();
    let placed = |node: &str| placements.iter().find(|(id, _)| id == node).map(|(_, p)| p);
    json!({
        "name": graph.name,
        "inputs": graph.externals,
        "params": graph.params.iter().map(|p| json!({
            "name": p.name,
            "type": p.ty,
            "default": value_json(&p.default),
            "value": value_json(overrides.get(&p.name).unwrap_or(&p.default)),
        })).collect::<Vec<_>>(),
        "nodes": graph.nodes.iter().map(|node| json!({
            "id": node.id.to_string(),
            "kernel": node.kernel.to_string(),
            "backend": placed(&node.id.to_string()).map(|p| p.backend),
            "routed": placed(&node.id.to_string()).and_then(|p| p.routed.clone()),
            "inputs": node.inputs.iter().map(|(port, binding)| (port.clone(), match binding {
                Binding::Now(s) => json!({ "kind": "now", "source": source(s) }),
                Binding::Previous { source: s, initial, migration } => json!({ "kind": "previous", "source": source(s), "initial": source(initial), "migrate": migration.as_ref().map(ToString::to_string) }),
            })).collect::<serde_json::Map<_, _>>(),
            "params": node.params.iter().map(|(name, value)| (name.clone(), match value {
                ParamValue::Literal(v) => value_json(v),
                ParamValue::Graph(p) => json!({ "param": p }),
            })).collect::<serde_json::Map<_, _>>(),
            "domain": node.domain.as_ref().map(|d| match d {
                orla_graph::DomainSpec::Literal(orla_kernel::Domain::Rows(n)) => json!([n]),
                orla_graph::DomainSpec::Literal(orla_kernel::Domain::Grid(w, h)) => json!([w, h]),
                orla_graph::DomainSpec::Literal(orla_kernel::Domain::Volume(w, h, d)) => json!([w, h, d]),
                orla_graph::DomainSpec::Expr(expr) => json!(expr.to_string()),
            }),
            "async": node.is_async,
        })).collect::<Vec<_>>(),
        "outputs": graph.outputs.iter().map(|(name, s)| json!({ "name": name, "source": source(s) })).collect::<Vec<_>>(),
    })
}

pub(crate) fn events_json(events: &[Event]) -> Vec<Json> {
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
                let mut json = diagnostic_json(error);
                json["kind"] = json!("rejected");
                json["path"] = json!(path.display().to_string());
                json
            }
            Event::Retired { path, id } => json!({ "kind": "retired", "path": path.display().to_string(), "id": id.to_string() }),
            Event::SceneLoaded => json!({ "kind": "scene_loaded" }),
            Event::Loaded { name, path, detail } => json!({ "kind": "loaded", "name": name, "path": path.display().to_string(), "detail": detail }),
            Event::Text { name, chars } => json!({ "kind": "text", "name": name, "chars": chars }),
            Event::SceneRejected(error) => {
                let mut json = diagnostic_json(error);
                json["kind"] = json!("scene_rejected");
                json
            }
            Event::Routed { node, backend, why } => {
                json!({ "kind": "routed", "node": node, "backend": backend, "why": why })
            }
            Event::Absent { name, path } => json!({ "kind": "absent", "name": name, "path": path.display().to_string() }),
            Event::Wrote { output, path, bytes } => json!({ "kind": "wrote", "output": output, "path": path.display().to_string(), "bytes": bytes }),
            Event::WriteRefused { output, path, why } => json!({ "kind": "write_refused", "output": output, "path": path.display().to_string(), "why": why }),
        })
        .collect()
}

/// A Float as the shortest decimal that reads back as it: `0.3`, not the
/// double nearest the float (`0.30000001192092896`).
fn float_json(v: f32) -> Json {
    v.to_string()
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Json::Null, Json::Number)
}

fn floats_json(v: &[f32]) -> Json {
    Json::Array(v.iter().map(|x| float_json(*x)).collect())
}

/// A value as JSON: numbers and booleans, a vector as its components, a
/// matrix as its columns, a struct or array as its members. A float is
/// the shortest decimal that reads back as it, a double the number that
/// reads back exactly, a long a number up to 2^53 in magnitude and a
/// string of digits beyond.
pub fn value_json(value: &Value) -> Json {
    let columns = |v: &[f32], n: usize| Json::Array(v.chunks(n).map(floats_json).collect());
    let dcolumns = |v: &[f64], n: usize| json!(v.chunks(n).collect::<Vec<_>>());
    let long = |v: i64| {
        if v.unsigned_abs() <= 1 << 53 {
            json!(v)
        } else {
            json!(v.to_string())
        }
    };
    match value {
        Value::Double(v) => json!(v),
        Value::Long(v) => long(*v),
        Value::DVec2(v) => json!(v),
        Value::DVec3(v) => json!(v),
        Value::DVec4(v) => json!(v),
        Value::LVec2(v) => json!(v.iter().map(|x| long(*x)).collect::<Vec<_>>()),
        Value::LVec3(v) => json!(v.iter().map(|x| long(*x)).collect::<Vec<_>>()),
        Value::DMat2(v) => dcolumns(v, 2),
        Value::DMat3(v) => dcolumns(v, 3),
        Value::DMat4(v) => dcolumns(v, 4),
        Value::Float(v) => float_json(*v),
        Value::Int(v) => json!(v),
        Value::Bool(v) => json!(v),
        Value::Vec2(v) => floats_json(v),
        Value::Vec3(v) => floats_json(v),
        Value::Vec4(v) => floats_json(v),
        Value::IVec2(v) => json!(v),
        Value::IVec3(v) => json!(v),
        Value::Mat2(v) => columns(v, 2),
        Value::Mat3(v) => columns(v, 3),
        Value::Mat4(v) => columns(v, 4),
        Value::Record(members) => json!(members.iter().map(value_json).collect::<Vec<_>>()),
    }
}

/// Reads a value: an integer is an `Int` (a `Long` beyond 32 bits, as is
/// a string of digits); another number stays exact until the receiving
/// parameter gives its width; a boolean is a `Bool`; an array is a record
/// the parameter shapes: `[1, 2.5]` a `Vec2`, `[3, 4]` an `IVec2` or
/// `Vec2`, `[[1, 0], [0, 1]]` a `Mat2`, `[[0, 1, 0], 2]` a struct of a
/// `Vec3` and a `Float`.
pub fn value_from_json(json: &Json) -> Result<Value, Json> {
    match json {
        Json::Bool(v) => Ok(Value::Bool(*v)),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i32::try_from(i).map_or(Value::Long(i), Value::Int))
            } else {
                Ok(Value::Double(n.as_f64().unwrap_or(0.0)))
            }
        }
        Json::String(digits) if digits.parse::<i64>().is_ok() => {
            Ok(Value::Long(digits.parse().unwrap_or(0)))
        }
        Json::Array(items) if !items.is_empty() => Ok(Value::Record(
            items
                .iter()
                .map(value_from_json)
                .collect::<Result<_, _>>()?,
        )),
        _ => Err(failure(
            "a value is a number, a boolean, or an array of values",
        )),
    }
}
