//! Operations that run the scene: one evaluation, many frames up to a
//! count or a condition, a measurement, and the state and window runs see.

use orla_kernel::Context;
use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{context_of, evaluation_error, failure, report_json};

/// The most frames one `advance` (or `bench`) runs.
const MAX_ADVANCE: u64 = 100_000;

/// A Float, Int or Bool value as a number; other types are an error.
fn scalar((ty, words): (orla_ir::Type, Vec<u32>)) -> Result<f64, String> {
    match (ty, words.first()) {
        (orla_ir::Type::Float, Some(word)) => Ok(f64::from(f32::from_bits(*word))),
        (orla_ir::Type::Int, Some(word)) => Ok(f64::from(*word as i32)),
        (orla_ir::Type::Bool, Some(word)) => Ok(f64::from(u8::from(*word != 0))),
        (ty, _) => Err(format!(
            "`until` compares one number; name a field that is a Float, an Int or a Bool, not a {ty}"
        )),
    }
}

/// Evaluates the scene once.
pub(crate) fn evaluate(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let context = context_of(session, request.json());
    if request.get("settle").and_then(Json::as_bool) == Some(true) {
        session.settle().map_err(|e| evaluation_error(&e, None))?;
    }
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
    .map_err(|e| evaluation_error(&e, None))?;
    Ok(report_json(session, &report))
}

/// Measures the same frames several times, as `orla bench` does. The
/// clock advances as in `advance`, so a measurement also runs the scene.
pub(crate) fn bench(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let whole = |name: &str, default: u64| -> Result<u64, Json> {
        match request.get(name) {
            None | Some(Json::Null) => Ok(default),
            Some(value) => value
                .as_u64()
                .filter(|count| *count >= 1)
                .ok_or_else(|| failure(format!("`{name}` is a whole number of at least 1"))),
        }
    };
    let frames = whole("frames", 30)?;
    let repeat = whole("repeat", 3)?;
    if frames
        .checked_mul(repeat)
        .is_none_or(|runs| runs > MAX_ADVANCE)
    {
        return Err(failure(format!(
            "`frames` times `repeat` is at most {MAX_ADVANCE}; measure fewer frames or fewer runs"
        )));
    }
    let fps = request.get("fps").and_then(Json::as_f64).unwrap_or(60.0);
    if fps <= 0.0 {
        return Err(failure(
            "`fps` is the frames a second of the clock; it is positive",
        ));
    }
    let nodes = request
        .get("nodes")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let backend = session.runtime().backend_name();
    let measured = orla_session::bench::measure(
        session,
        backend,
        frames as u32,
        repeat as u32,
        fps as f32,
        context_of(session, request.json()),
        nodes,
    )
    .map_err(failure)?;
    Ok(json!({
        "backend": measured.backend,
        "frames": frames,
        "repeat": repeat,
        "best_ms": measured.best,
        "median_ms": measured.median,
        "ran": measured.ran,
        "nodes": measured.nodes.iter().map(|time| json!({ "node": time.node, "ms": time.ms, "backend": time.backend, "routed": time.routed, "uploaded": time.uploaded, "downloaded": time.downloaded })).collect::<Vec<_>>(),
    }))
}

/// Runs many frames in one request, until a count or a value says to stop.
pub(crate) fn advance(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let frames = request
        .get("frames")
        .and_then(Json::as_u64)
        .ok_or_else(|| failure("`advance` takes `frames`, a whole number of at least 1"))?;
    if frames == 0 || frames > MAX_ADVANCE {
        return Err(failure(format!(
            "`frames` is 1 to {MAX_ADVANCE}; a longer run is several `advance` requests"
        )));
    }
    let fps = request.get("fps").and_then(Json::as_f64).unwrap_or(60.0);
    if fps <= 0.0 {
        return Err(failure(
            "`fps` is the frames a second of the clock; it is positive",
        ));
    }
    let base = context_of(session, request.json());
    // `until` watches one value of one node and stops the run early.
    let until = match request.get("until") {
        None | Some(Json::Null) => None,
        Some(Json::Object(test)) => {
            let target = test
                .get("target")
                .and_then(Json::as_str)
                .ok_or_else(|| failure("`until` names a `target`, as `node.port[0].field`"))?;
            let (node, port, index, path) = orla_session::parse_target(target).map_err(failure)?;
            let (kind, bound) = ["below", "above", "equals"]
                .iter()
                .find_map(|kind| {
                    test.get(*kind)
                        .and_then(Json::as_f64)
                        .map(|bound| (*kind, bound))
                })
                .ok_or_else(|| {
                    failure("`until` takes `below`, `above` or `equals` and a number")
                })?;
            Some((
                target.to_owned(),
                orla_graph::NodeId(node),
                port,
                index,
                path,
                kind,
                bound,
            ))
        }
        Some(_) => return Err(failure("`until` is an object")),
    };
    // `save` writes node outputs when the run stops, as `run --save`
    // does; a relative path is beside the scene.
    let mut saves: Vec<(orla_graph::NodeId, String, std::path::PathBuf)> = Vec::new();
    match request.get("save") {
        None | Some(Json::Null) => {}
        Some(Json::Object(items)) => {
            for (target, path) in items {
                let (node, port) = orla_graph::split_port(target)
                    .ok_or_else(|| failure("`save` names `node.port`"))?;
                let path = path
                    .as_str()
                    .ok_or_else(|| failure("`save` takes a file path"))?;
                let resolved = std::path::Path::new(path);
                let resolved = if resolved.is_absolute() {
                    resolved.to_path_buf()
                } else {
                    session
                        .root()
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new("."))
                        .join(resolved)
                };
                // Writes follow the session's write grant, as a scene's
                // outputs do.
                let resolved = session
                    .allowed_write(&resolved)
                    .map_err(|why| failure(format!("cannot save {target}: {why}")))?;
                saves.push((node, port, resolved));
            }
        }
        Some(_) => return Err(failure("`save` is an object of `node.port` to a path")),
    }
    // Watched and saved nodes stay materialized and are the demand;
    // with none, the whole scene runs.
    let mut watched: Vec<orla_graph::NodeId> = Vec::new();
    if let Some((_, node, ..)) = &until {
        watched.push(node.clone());
    }
    watched.extend(saves.iter().map(|(node, _, _)| node.clone()));
    let mut pinned = Vec::new();
    for node in &watched {
        session.pin(node.clone());
        pinned.push(node.clone());
    }
    let demand: std::collections::BTreeSet<orla_graph::NodeId> = watched.iter().cloned().collect();
    let started = orla_kernel::clock::now();
    let outcome = (|| -> Result<Json, Json> {
        let mut ran = 0u64;
        let mut stopped = "frames";
        let mut value: Option<f64> = None;
        let mut last = None;
        for frame in 0..frames {
            let context = Context {
                time: base.time + frame as f32 / fps as f32,
                ..base
            };
            let report = if demand.is_empty() {
                session.evaluate(&context)
            } else {
                session.evaluate_for(&context, &demand)
            }
            .map_err(|e| evaluation_error(&e, Some(frame)))?;
            session
                .settle()
                .map_err(|e| evaluation_error(&e, Some(frame)))?;
            ran += 1;
            last = Some(report);
            if let Some((_, node, port, index, path, kind, bound)) = &until {
                let read = session
                    .element_field(node, port, *index, path)
                    .map_err(|e| failure(format!("frame {frame}: {e}")))?;
                let now = scalar(read).map_err(failure)?;
                value = Some(now);
                let holds = match *kind {
                    "below" => now < *bound,
                    "above" => now > *bound,
                    _ => now == *bound,
                };
                if holds {
                    stopped = "until";
                    break;
                }
            }
        }
        let mut saved = Vec::new();
        for (node, port, path) in &saves {
            session
                .save(node, port, path)
                .map_err(|e| failure(format!("cannot save {node}.{port}: {e}")))?;
            saved.push(path.display().to_string());
        }
        let report = last
            .as_ref()
            .map(|report| report_json(session, report))
            .unwrap_or(Json::Null);
        Ok(json!({
            "frames": ran,
            "stopped": stopped,
            "target": until.as_ref().map(|(target, ..)| target.clone()),
            "value": value,
            "saved": saved,
            "millis": orla_kernel::clock::since(started).as_secs_f64() * 1000.0,
            "report": report,
        }))
    })();
    for node in pinned {
        session.unpin(&node);
    }
    outcome
}

/// Forgets all committed state, or one port's with `node` (and `port`,
/// default `result`).
pub(crate) fn reset(session: &mut Session, request: &Request) -> Result<Json, Json> {
    if request.get("node").is_none() {
        if request.get("port").is_some() {
            return Err(failure("`port` names a port of `node`; name the node too"));
        }
        session.reset_state();
        return Ok(json!({ "reset": true }));
    }
    let node = request.text("node")?;
    let port = match request.get("port") {
        None => "result",
        Some(port) => port
            .as_str()
            .ok_or_else(|| failure("`port` must be a string"))?,
    };
    session
        .reset_port(&orla_graph::NodeId(node.clone()), port)
        .map_err(|e| failure(e.to_string()))?;
    Ok(json!({ "reset": true, "node": node, "port": port }))
}

/// Sets the window a headless session reports.
pub(crate) fn window(session: &mut Session, request: &Request) -> Result<Json, Json> {
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
