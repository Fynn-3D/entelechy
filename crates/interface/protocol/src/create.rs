//! Operations that create: writing the scene, preparing, publishing and
//! retiring Kernels, setting parameters. A publish is checked against the
//! expected generation; a rejected change leaves the running version.

use entelechy_kernel::{Expect, Generation};
use serde_json::{Value as Json, json};

use entelechy_session::Session;

use crate::request::Request;
use crate::wire::{
    diagnostic_json, events_json, failure, manifest_json, value_from_json, value_json,
};

/// Replaces the scene document and syncs. The text is checked before it
/// reaches disk; with `expect`, the document must still be that version.
/// The text comes whole (`text`) or as `edits` of the one on disk, each
/// changing only the bytes of the statement or node member it names.
pub(crate) fn compose(session: &mut Session, request: &Request) -> Result<Json, Json> {
    compose_with(session, request, true)
}

/// [`compose`]; with `settle`, answered once the scene lands, otherwise
/// with the sync's events while the scene may still wait for its Kernels
/// ([`crate::Answers`] answers it once it lands).
pub(crate) fn compose_with(
    session: &mut Session,
    request: &Request,
    settle: bool,
) -> Result<Json, Json> {
    if request.get("documents").is_some() || request.get("path").is_some() {
        return compose_documents(session, request, settle);
    }
    let root = session.root().to_path_buf();
    let written = session.scene_text();
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
            // The document as it stands, to compose against.
            return Err(refused);
        }
    }
    let text = match (request.get("text"), request.get("edits")) {
        (Some(_), Some(_)) => {
            return Err(failure("`compose` takes `text` or `edits`, not both"));
        }
        (None, Some(edits)) => {
            let written = written.map_err(|e| failure(format!("{}: {e}", root.display())))?;
            entelechy_scene::edit::edit(&written, &edits_from_json(edits)?)
                .map_err(|e| failure(e.to_string()))?
        }
        _ => request.text("text")?,
    };
    // A text a sync rejects leaves the disk and the running scene as they
    // are; an accepted one replaces the file whole, so every reader,
    // the session's watcher included, sees one version.
    // A refusal is an error, as a refused publish is: why and where, the
    // document untouched at its version, and for `edits` the text they
    // made, which `line` counts in.
    if let Err(why) = session
        .write_scene(&text)
        .map_err(|e| failure(e.to_string()))?
    {
        let mut refused = diagnostic_json(&why);
        refused["version"] = json!(current);
        refused["path"] = json!(root.display().to_string());
        refused["written"] = json!(false);
        if request.get("edits").is_some() {
            refused["text"] = json!(text);
        }
        return Err(refused);
    }
    // The scene read here is this request's change: its Kernels published
    // and the scene replaced, or refused, as one.
    session.claim_scene();
    let events = if settle {
        session.sync_settled()
    } else {
        session.sync()
    };
    Ok(json!({
        "events": events_json(&events),
        "version": document_version(&text),
        "path": root.display().to_string(),
        "written": true,
    }))
}

/// [`compose`] of documents of the scene other than its root, or of
/// several at once: `path` names one (from the scene's directory), or
/// `documents` lists each with its own `path` (none for the scene),
/// `text` or `edits`, and `expect`. All are checked together before any
/// is written, and all are written or none.
fn compose_documents(session: &mut Session, request: &Request, settle: bool) -> Result<Json, Json> {
    let scene_name = session
        .root()
        .file_name()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| session.root().to_path_buf());
    let items: Vec<&serde_json::Map<String, Json>> = match request.get("documents") {
        Some(documents) => {
            if ["path", "text", "edits", "expect"]
                .iter()
                .any(|key| request.get(key).is_some())
            {
                return Err(failure(
                    "`documents` carries each document's own `path`, `text` or `edits`, and `expect`; they stand in each, not beside it",
                ));
            }
            let list = documents
                .as_array()
                .ok_or_else(|| failure("`documents` is an array, one object a document"))?;
            if list.is_empty() {
                return Err(failure("`documents` names at least one document"));
            }
            list.iter()
                .map(|item| {
                    item.as_object()
                        .ok_or_else(|| failure("each of `documents` is an object"))
                })
                .collect::<Result<_, _>>()?
        }
        None => vec![
            request
                .json()
                .as_object()
                .ok_or_else(|| failure("a request is an object"))?,
        ],
    };
    struct Change {
        path: std::path::PathBuf,
        text: String,
        version: String,
        edited: bool,
    }
    let mut changes = Vec::new();
    for item in items {
        let named = match item.get("path") {
            None => scene_name.clone(),
            Some(Json::String(path)) => std::path::PathBuf::from(path),
            Some(_) => {
                return Err(failure(
                    "`path` is a document's path, from the scene's directory",
                ));
            }
        };
        let (path, current) = session.composed_document(&named).map_err(|why| {
            let mut refused = failure(why);
            refused["path"] = json!(named.display().to_string());
            refused["written"] = json!(false);
            refused
        })?;
        let version = current.as_deref().map(document_version).unwrap_or_default();
        let at_fault = |message: String| {
            let mut refused = failure(message);
            refused["path"] = json!(path.display().to_string());
            refused["version"] = json!(version);
            refused["written"] = json!(false);
            refused
        };
        if let Some(expect) = item.get("expect") {
            let expect = expect
                .as_str()
                .ok_or_else(|| at_fault("`expect` is the `version` `document` answered".into()))?;
            if version != expect {
                return Err(at_fault(format!(
                    "`{}` is at version {}, not {expect}: it changed since it was read",
                    path.display(),
                    if version.is_empty() {
                        "none (it is not written)"
                    } else {
                        &version
                    }
                )));
            }
        }
        let text = match (item.get("text"), item.get("edits")) {
            (Some(_), Some(_)) => {
                return Err(at_fault(
                    "a document takes `text` or `edits`, not both".into(),
                ));
            }
            (None, Some(edits)) => entelechy_scene::edit::edit(
                current.as_deref().unwrap_or(""),
                &edits_from_json(edits)?,
            )
            .map_err(|e| at_fault(e.to_string()))?,
            (Some(Json::String(text)), None) => text.clone(),
            (Some(_), None) => return Err(at_fault("`text` is the document's whole text".into())),
            (None, None) => return Err(at_fault("a document changes by `text` or `edits`".into())),
        };
        changes.push(Change {
            path,
            text,
            version,
            edited: item.get("edits").is_some(),
        });
    }
    let texts: Vec<(std::path::PathBuf, String)> = changes
        .iter()
        .map(|c| (c.path.clone(), c.text.clone()))
        .collect();
    if let Err(refusal) = session
        .write_documents(&texts)
        .map_err(|e| failure(e.to_string()))?
    {
        let mut refused = diagnostic_json(&refusal.diagnostic);
        refused["path"] = json!(refusal.path.display().to_string());
        refused["written"] = json!(false);
        if let Some(change) = changes.iter().find(|c| c.path == refusal.path) {
            refused["version"] = json!(change.version);
            if change.edited {
                refused["text"] = json!(change.text);
            }
        }
        return Err(refused);
    }
    session.claim_scene();
    let events = if settle {
        session.sync_settled()
    } else {
        session.sync()
    };
    let root = session.root().display().to_string();
    let scene = session
        .scene_text()
        .map(|text| document_version(&text))
        .unwrap_or_default();
    Ok(json!({
        "events": events_json(&events),
        "version": scene,
        "path": root,
        "written": true,
        "documents": changes.iter().map(|c| json!({ "path": c.path.display().to_string(), "version": document_version(&c.text) })).collect::<Vec<_>>(),
    }))
}

/// `edits` as the scene takes them: each object names its target by kind
/// (`node`, `output`, `param`, `require`, `input`, `kernel`, `struct`,
/// `type`, `fn`, `use`), optionally a node's `member`, and `text`, the whole statement
/// or member, or null to remove it.
fn edits_from_json(edits: &Json) -> Result<Vec<entelechy_scene::edit::Edit>, Json> {
    use entelechy_scene::edit::{Edit, StatementKind, Target};
    const KINDS: [(&str, StatementKind); 10] = [
        ("node", StatementKind::Node),
        ("output", StatementKind::Output),
        ("param", StatementKind::Param),
        ("require", StatementKind::Require),
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
                    "an edit names one thing: `node`, `output`, `param`, `require`, `input`, `kernel`, `struct`, `type`, `fn`, or `use`, and its name".into(),
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

/// A document's version: the digest of its text
/// ([`entelechy_session::history::digest`]), stable across builds and platforms.
pub(crate) fn document_version(text: &str) -> String {
    format!("{:016x}", entelechy_session::history::digest(text))
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
    let publishing = publish_begin(session, request)?;
    loop {
        if let Some(published) = session.publishing(&publishing) {
            return published_json(session, published);
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// A begun `publish`: its source compiled, and prepared off the frame
/// where the session does so.
pub(crate) fn publish_begin(
    session: &mut Session,
    request: &Request,
) -> Result<entelechy_session::Publishing, Json> {
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
    session
        .publish_later(&source, expect)
        .map_err(|e| diagnostic_json(&e))
}

/// What a finished `publish` answers.
pub(crate) fn published_json(
    session: &Session,
    published: Result<
        std::sync::Arc<entelechy_kernel::KernelVersion>,
        entelechy_session::Diagnostic,
    >,
) -> Result<Json, Json> {
    let version = published.map_err(|e| diagnostic_json(&e))?;
    let mut answer =
        json!({ "id": version.id().to_string(), "generation": version.generation().0 });
    // State this Kernel starts keeps what it committed; reset starts it
    // from the new law.
    let kept = session.started_by(version.id());
    if !kept.is_empty() {
        let resets: Vec<Json> = kept
            .iter()
            .filter_map(|at| entelechy_graph::split_port(at))
            .map(|(node, port)| json!({ "op": "reset", "node": node.0, "port": port }))
            .collect();
        answer["keeps"] = json!(kept);
        answer["reset"] = Json::Array(resets);
    }
    Ok(answer)
}

/// Retires a published Kernel.
pub(crate) fn retire(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let id = request.kernel()?;
    if session
        .runtime()
        .kernel(&id)
        .is_some_and(|version| version.native().is_some())
    {
        return Err(failure(format!(
            "`{id}` is a native operation of the Host: it is neither retired nor replaced"
        )));
    }
    Ok(json!({ "retired": session.retire_kernel(&id) }))
}

/// Sets a graph parameter.
pub(crate) fn param(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let name = request.text("name")?;
    // The name is judged before the value, so a value for a param that
    // lives elsewhere is answered with where it lives.
    let ty = session.param_type(&name).map_err(failure)?;
    let given = value_from_json(request.field("value")?).map_err(|mut refused| {
        if let (Some(ty), Some(message)) = (&ty, refused["message"].as_str()) {
            refused["message"] = json!(format!("param `{name}` is of type {ty}; {message}"));
        }
        refused
    })?;
    // A value of another shape is refused as the caller wrote it.
    session.set_param(&name, given).map_err(|why| match &ty {
        Some(ty) => failure(format!(
            "param `{name}` is of type {ty}; {} is not one",
            request.get("value").cloned().unwrap_or_default()
        )),
        None => failure(why),
    })?;
    let value = session.params().get(&name).cloned().expect("just set");
    // A param that sizes the graph lowers the document now, as this
    // request's change.
    session.claim_scene();
    let events = session.sync();
    Ok(json!({ "name": name, "value": value_json(&value), "events": events_json(&events) }))
}
