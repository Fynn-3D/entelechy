//! Operations that run the scene: one evaluation, multi-tick headless
//! warps, and the state runs see.

use serde_json::{Value as Json, json};

use entelechy_session::{Session, WarpError, WarpOutcome};

use crate::request::Request;
use crate::wire::{context_of, evaluation_error, failure, place_json, report_json};

/// Evaluates the scene once.
pub(crate) fn evaluate(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let context = context_of(session, request.json())?;
    if request.get("settle").and_then(Json::as_bool) == Some(true) {
        session.settle().map_err(|e| evaluation_error(&e, None))?;
    }
    // Clocked, the device times the frames as they run, from this one on.
    if let Some(clock) = request.get("clock").and_then(Json::as_bool) {
        session.set_clocked(clock);
    }
    // Measured, each node waits for the device, this evaluation alone.
    let measured = measure_asked(request.json());
    session.set_measured(measured);
    let report = match request.get("nodes").and_then(Json::as_array) {
        Some(nodes) => {
            let demand = nodes
                .iter()
                .filter_map(Json::as_str)
                .map(|node| entelechy_graph::NodeId(node.to_owned().into()))
                .collect();
            session.evaluate_for(&context, &demand)
        }
        None => session.evaluate(&context),
    };
    if measured {
        session.set_measured(false);
    }
    let report = report.map_err(|e| {
        let mut json = evaluation_error(&e, None);
        // Where in the Kernel an element failed, traced again.
        if let Some(place) = session.failure_place(&e, &context) {
            place_json(&mut json, &place);
        }
        json
    })?;
    Ok(evaluation_json(session, &report, measured))
}

/// Whether an `evaluate` request asks each node's time by the device's
/// clock (`measure: true`).
pub fn measure_asked(request: &Json) -> bool {
    request.get("measure").and_then(Json::as_bool) == Some(true)
}

/// What an `evaluate` answers: its report, whether it was measured, and
/// the latest clocked frame the device has done.
pub fn evaluation_json(
    session: &Session,
    report: &entelechy_graph::Report,
    measured: bool,
) -> Json {
    let mut answer = report_json(session, report);
    answer["measured"] = json!(measured);
    answer["device"] = device_json(session.device_times());
    answer
}

/// The latest clocked frame the device has done, node by node, or null.
fn device_json(times: Option<entelechy_session::DeviceTimes>) -> Json {
    let millis = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    match times {
        None => Json::Null,
        Some(times) => json!({
            "frame": times.frame,
            "busy": millis(times.busy),
            "between": millis(times.between),
            "nodes": times.nodes.iter().map(|(node, busy)| json!({ "node": node.to_string(), "millis": millis(*busy) })).collect::<Vec<_>>(),
        }),
    }
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
        .reset_port(&entelechy_graph::NodeId(node.clone().into()), port)
        .map_err(|e| failure(e.to_string()))?;
    Ok(json!({ "reset": true, "node": node, "port": port }))
}

/// Steps causal state forward across many ticks without evaluating unasked
/// outputs.
pub(crate) fn warp(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let context = context_of(session, request.json())?;
    let time_given = request.get("time").is_some();
    warp_with_context(session, request.json(), &context, time_given).map(|(answer, _)| answer)
}

/// Runs a `warp` request under `base_context` (used directly by `serve` and
/// by a served Player window, which also receives `(steps, dt)` to advance
/// its own clock).
pub fn warp_for_window(
    session: &mut Session,
    request: &Json,
    window_context: &entelechy_kernel::Context,
) -> Result<(Json, f64), Json> {
    crate::ops::check(request)?;
    let mut base = *window_context;
    let time_given = request.get("time").is_some();
    if let Some(t) = request.get("time").and_then(Json::as_f64) {
        base.time = t;
    }
    warp_with_context(session, request, &base, time_given)
}

fn warp_with_context(
    session: &mut Session,
    request: &Json,
    context: &entelechy_kernel::Context,
    time_given: bool,
) -> Result<(Json, f64), Json> {
    let steps_i64 = request
        .get("steps")
        .and_then(Json::as_i64)
        .ok_or_else(|| failure("`warp` needs `steps` as a positive integer"))?;
    if steps_i64 <= 0 {
        return Err(failure(format!(
            "`steps` must be between 1 and 100000, got {steps_i64}"
        )));
    }
    let steps = steps_i64 as u64;
    let dt = request
        .get("dt")
        .and_then(Json::as_f64)
        .unwrap_or(1.0 / 60.0);
    if request.get("settle").and_then(Json::as_bool) == Some(true) {
        session.settle().map_err(|e| evaluation_error(&e, None))?;
    }
    let nodes: Vec<entelechy_graph::NodeId> = request
        .get("nodes")
        .and_then(Json::as_array)
        .map(|list| {
            list.iter()
                .filter_map(Json::as_str)
                .map(|node| entelechy_graph::NodeId(node.to_owned().into()))
                .collect()
        })
        .unwrap_or_default();
    let until = request.get("until").and_then(Json::as_str);
    let outcome = session
        .warp(context, time_given, steps, dt, &nodes, until)
        .map_err(|err| match err {
            WarpError::Message(msg) => failure(msg),
            WarpError::Evaluate {
                step,
                context: step_ctx,
                error,
            } => {
                let mut json = evaluation_error(&error, None);
                if step > 0 {
                    json["step"] = json!(step);
                    json["time"] = json!(step_ctx.time);
                }
                if let Some(place) = session.failure_place(&error, &step_ctx) {
                    place_json(&mut json, &place);
                }
                json
            }
        })?;
    let advanced_seconds = (outcome.steps as f64) * dt;
    Ok((warp_json(&outcome), advanced_seconds))
}

fn warp_json(outcome: &WarpOutcome) -> Json {
    let mut answer = json!({
        "steps": outcome.steps,
        "active": outcome.active,
        "time": outcome.time,
        "millis": outcome.elapsed.as_secs_f64() * 1000.0,
        "stopped": outcome.stopped.as_str(),
        "executed": outcome.last.executed.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "reused": outcome.last.reused.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "committed": outcome.last.committed.iter().map(|(node, port)| json!([node.to_string(), port])).collect::<Vec<_>>(),
        "let_go": outcome.last.let_go,
    });
    if let Some(value) = outcome.until_value {
        answer["until"] = if value.is_finite() {
            json!(value)
        } else if value.is_nan() {
            json!("NaN")
        } else if value.is_sign_positive() {
            json!("Infinity")
        } else {
            json!("-Infinity")
        };
    }
    answer
}
