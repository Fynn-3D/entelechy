//! Operations that run the scene: one evaluation, and the state runs
//! see.

use serde_json::{Value as Json, json};

use entelechy_session::Session;

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
                .map(|node| entelechy_graph::NodeId(node.to_owned()))
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
        .reset_port(&entelechy_graph::NodeId(node.clone()), port)
        .map_err(|e| failure(e.to_string()))?;
    Ok(json!({ "reset": true, "node": node, "port": port }))
}
