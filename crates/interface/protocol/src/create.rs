//! Operations that create: the scene document written, Kernels prepared,
//! published and retired, parameters set. Publication is checked against
//! the generation the agent expects, and a rejected change leaves the
//! running version alive.

use orla_runtime::{Expect, Generation};
use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{events_json, failure, manifest_json, positioned, value_from_json, value_json};

/// Writes the scene document and syncs.
pub(crate) fn compose(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let text = request.text("text")?;
    std::fs::write(session.root(), text)
        .map_err(|e| failure(format!("{}: {e}", session.root().display())))?;
    Ok(json!({ "events": events_json(&session.sync()) }))
}

/// Reads again what changed on disk.
pub(crate) fn sync(session: &mut Session) -> Result<Json, Json> {
    Ok(json!({ "events": events_json(&session.sync()) }))
}

/// Prepares a Kernel without publishing it.
pub(crate) fn prepare(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let source = request.text("source")?;
    let candidate = session.prepare_source(&source).map_err(positioned)?;
    let current = session
        .runtime()
        .kernel(candidate.id())
        .map(|version| version.generation().0);
    let mut json = manifest_json(
        candidate.manifest(),
        Generation(0),
        &candidate.module().structs,
    );
    json["generation"] = Json::Null;
    json["current"] = current.map_or(Json::Null, Json::from);
    Ok(json)
}

/// Publishes a Kernel against the generation expected.
pub(crate) fn publish(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let source = request.text("source")?;
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

/// Retires a published Kernel.
pub(crate) fn retire(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let id = request.kernel()?;
    Ok(json!({ "retired": session.retire_kernel(&id) }))
}

/// Sets a graph parameter.
pub(crate) fn param(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let name = request.text("name")?;
    let given = value_from_json(request.field("value")?)?;
    session.set_param(&name, given).map_err(failure)?;
    let value = session.params().get(&name).cloned().expect("just set");
    // A param that sizes the graph lowers the document again now.
    let events = session.sync();
    Ok(json!({ "name": name, "value": value_json(&value), "events": events_json(&events) }))
}
