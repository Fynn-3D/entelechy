//! Operations that create: the scene document written, Kernels prepared,
//! published and retired, parameters set. Publication is checked against
//! the generation the agent expects, and a rejected change leaves the
//! running version alive.

use orla_runtime::{Expect, Generation};
use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{
    diagnostic_json, events_json, failure, manifest_json, value_from_json, value_json,
};

/// Replaces the scene document and syncs: checked first, so a text that
/// would be rejected never reaches the disk, and with `expect`, only if
/// the document is still the version the agent read. The new text is
/// given whole (`text`), or as `edits` of the one on disk, each naming a
/// statement or a node's member and only its bytes changing.
pub(crate) fn compose(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let root = session.root().to_path_buf();
    let written = session.store().read_text(&root);
    let current = written
        .as_ref()
        .map(|text| document_version(text))
        .unwrap_or_default();
    if let Some(expect) = request.get("expect") {
        let expect = expect
            .as_str()
            .ok_or_else(|| failure("`expect` is the `version` `document` answered"))?;
        if current != expect {
            let mut refused = failure(format!(
                "the scene is at version {current}, not {expect}: it changed since it was read"
            ));
            refused["version"] = json!(current);
            return Err(refused);
        }
    }
    let text = match (request.get("text"), request.get("edits")) {
        (Some(_), Some(_)) => {
            return Err(failure("`compose` takes `text` or `edits`, not both"));
        }
        (None, Some(edits)) => {
            let written = written.map_err(|e| failure(format!("{}: {e}", root.display())))?;
            orla_scene::edit::edit(&written, &edits_from_json(edits)?)
                .map_err(|e| failure(e.to_string()))?
        }
        _ => request.text("text")?,
    };
    // Rejected as a sync would reject it, and the document on disk, and
    // so the scene running, left as they were.
    if let Err(why) = session.check_scene(&text) {
        return Ok(json!({
            "events": events_json(&[orla_session::Event::SceneRejected(why)]),
            "version": current,
        }));
    }
    // Replaced whole: a reader, the session's own watcher included, never
    // sees half of it.
    session
        .store()
        .write(&root, text.as_bytes())
        .map_err(|e| failure(format!("{}: {e}", root.display())))?;
    // The scene this reads is this request's change.
    session.claim_scene();
    Ok(json!({
        "events": events_json(&session.sync()),
        "version": document_version(&text),
    }))
}

/// `edits` as the scene takes them: each an object naming its target by
/// its kind (`node`, `output`, `param`, `input`, `kernel`, `struct`,
/// `type`, `fn`, `use`), with `member` for a member of a node's block,
/// and `text`, the statement or member whole, or null to remove it.
fn edits_from_json(edits: &Json) -> Result<Vec<orla_scene::edit::Edit>, Json> {
    use orla_scene::edit::{Edit, StatementKind, Target};
    const KINDS: [(&str, StatementKind); 9] = [
        ("node", StatementKind::Node),
        ("output", StatementKind::Output),
        ("param", StatementKind::Param),
        ("input", StatementKind::Input),
        ("kernel", StatementKind::Kernel),
        ("struct", StatementKind::Struct),
        ("type", StatementKind::Type),
        ("fn", StatementKind::Fn),
        ("use", StatementKind::Use),
    ];
    let edits = edits
        .as_array()
        .ok_or_else(|| failure("`edits` is an array of edits"))?;
    edits
        .iter()
        .enumerate()
        .map(|(index, edit)| {
            let bad = |message: String| failure(format!("edit {index}: {message}"));
            let object = edit
                .as_object()
                .ok_or_else(|| bad("an edit is an object".into()))?;
            let named: Vec<(StatementKind, &str)> = KINDS
                .iter()
                .filter_map(|(key, kind)| {
                    object.get(*key).map(|name| (*kind, name.as_str().unwrap_or("")))
                })
                .collect();
            let [(kind, name)] = named.as_slice() else {
                return Err(bad(
                    "an edit names one thing: `node`, `output`, `param`, `input`, `kernel`, `struct`, `type`, `fn`, or `use`, and its name".into(),
                ));
            };
            if name.is_empty() {
                return Err(bad(format!("`{}` is a name", kind.keyword())));
            }
            let target = match object.get("member") {
                None => Target::Statement {
                    kind: *kind,
                    name: (*name).to_owned(),
                },
                Some(member) if *kind == StatementKind::Node => Target::Member {
                    node: (*name).to_owned(),
                    member: member
                        .as_str()
                        .filter(|m| !m.is_empty())
                        .ok_or_else(|| bad("`member` is a name".into()))?
                        .to_owned(),
                },
                Some(_) => return Err(bad("only a node has members".into())),
            };
            let text = match object.get("text") {
                Some(Json::Null) => None,
                Some(Json::String(text)) => Some(text.clone()),
                Some(_) => return Err(bad("`text` is a string, or null to remove".into())),
                None => {
                    return Err(bad(
                        "`text` is what stands there from now on, or null to remove it".into(),
                    ));
                }
            };
            Ok(Edit { target, text })
        })
        .collect()
}

/// A document's version: a digest of its text
/// ([`orla_session::history::digest`]), the same in every build and on
/// every platform.
pub(crate) fn document_version(text: &str) -> String {
    format!("{:016x}", orla_session::history::digest(text))
}

/// Reads again what changed on disk.
pub(crate) fn sync(session: &mut Session) -> Result<Json, Json> {
    Ok(json!({ "events": events_json(&session.sync()) }))
}

/// Prepares a Kernel without publishing it.
pub(crate) fn prepare(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let source = request.text("source")?;
    let candidate = session
        .prepare_source(&source)
        .map_err(|e| diagnostic_json(&e))?;
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
        .map_err(|e| diagnostic_json(&e))?;
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
    // A param that sizes the graph lowers the document again now, as
    // this request's change.
    session.claim_scene();
    let events = session.sync();
    Ok(json!({ "name": name, "value": value_json(&value), "events": events_json(&events) }))
}
