//! Operations that run the scene: one evaluation, and the state runs
//! see.

use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{context_of, evaluation_error, failure, place_json, report_json};

/// Evaluates the scene once.
pub(crate) fn evaluate(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let context = context_of(session, request.json());
    if request.get("settle").and_then(Json::as_bool) == Some(true) {
        session.settle().map_err(|e| evaluation_error(&e, None))?;
    }
    // Measured, each node waits for the device, this evaluation alone.
    let measured = request.get("measure").and_then(Json::as_bool) == Some(true);
    session.set_measured(measured);
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
    let mut answer = report_json(session, &report);
    answer["measured"] = json!(measured);
    Ok(answer)
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
