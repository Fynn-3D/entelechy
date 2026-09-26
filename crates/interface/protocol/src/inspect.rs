//! Read-only operations: the session, published Kernels, the scene and the
//! language's builtins. Each takes the session by shared reference.

use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{failure, graph_json, manifest_json};

/// The session at a glance, and the versions it speaks.
pub(crate) fn status(session: &Session) -> Result<Json, Json> {
    Ok(json!({
        "versions": { "orla": env!("CARGO_PKG_VERSION"), "api": orla_kernel::ABI_VERSION, "language": orla_language::LANGUAGE_VERSION, "ir": orla_ir::IR_VERSION, "package": orla_session::PACKAGE_VERSION },
        "backend": session.runtime().backend_name(),
        "scene": session.root().display().to_string(),
        "watched": session.watched().iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "kernels": session.runtime().kernels().count(),
        "nodes": session.graph().map_or(0, |graph| graph.nodes.len()),
        "fusion": session.fusion(),
        "revision": session.position().0,
        "frame": session.position().1,
        "grants": session.runtime().grants().iter().map(|c| c.name()).collect::<Vec<_>>(),
    }))
}

/// A published version's manifest with its structs, from a Kernel's
/// module or a native operation's table.
fn version_json(version: &orla_kernel::KernelVersion) -> Json {
    let native = version.native().map(|op| op.structs());
    let structs = match (version.module(), &native) {
        (Some(module), _) => &module.structs[..],
        (None, Some(structs)) => &structs[..],
        (None, None) => &[][..],
    };
    manifest_json(version.manifest(), version.generation(), structs)
}

/// Every published Kernel and native operation, by its manifest.
pub(crate) fn kernels(session: &Session) -> Result<Json, Json> {
    Ok(json!({
        "kernels": session
            .runtime()
            .kernels()
            .map(|version| version_json(version))
            .collect::<Vec<_>>()
    }))
}

/// One Kernel's manifest and the source `publish` takes.
pub(crate) fn kernel(session: &Session, request: &Request) -> Result<Json, Json> {
    let id = request.kernel()?;
    // A generation the registry released comes from the session's
    // record of the change that published it.
    if let Some(generation) = request.get("generation").and_then(Json::as_u64) {
        let source = session
            .source_at(&id, orla_kernel::Generation(generation))
            .ok_or_else(|| {
                failure(format!(
                    "{id} generation {generation} is not one the session remembers"
                ))
            })?;
        return Ok(json!({ "id": id.to_string(), "generation": generation, "source": source }));
    }
    let version = session
        .runtime()
        .kernel(&id)
        .ok_or_else(|| failure(format!("{id} is not published")))?;
    let mut json = version_json(&version);
    json["source"] = Json::String(version.source().to_owned());
    Ok(json)
}

/// The changes that made revisions after `since` and the attempts refused
/// since, each with its origin, as far back as the session remembers.
pub(crate) fn history(session: &Session, request: &Request) -> Result<Json, Json> {
    let since = request.get("since").and_then(Json::as_u64).unwrap_or(0);
    let (revision, frame) = session.position();
    let origin = |origin: &orla_session::Origin| match origin {
        orla_session::Origin::Document(path) => json!({ "document": path.display().to_string() }),
        orla_session::Origin::Protocol(channel) => json!({ "protocol": channel }),
        orla_session::Origin::Local => json!("local"),
    };
    let generation = |g: &Option<orla_kernel::Generation>| g.map(|g| g.0);
    let changes: Vec<Json> = session
        .changes_since(since)
        .iter()
        .map(|record| {
            let mut json = match &record.change {
                orla_session::Change::Published {
                    id,
                    from,
                    to,
                    source,
                } => json!({
                    "kind": "published", "kernel": id.to_string(), "from": generation(from),
                    "to": to.0, "source": format!("{source:016x}"),
                }),
                orla_session::Change::Retired { id, from } => {
                    json!({ "kind": "retired", "kernel": id.to_string(), "from": generation(from) })
                }
                orla_session::Change::Scene { document } => {
                    json!({ "kind": "scene", "version": format!("{document:016x}") })
                }
                orla_session::Change::Param { name } => json!({ "kind": "param", "name": name }),
                orla_session::Change::Written { node, port } => {
                    json!({ "kind": "written", "node": node, "port": port })
                }
                orla_session::Change::Reset => json!({ "kind": "reset" }),
                orla_session::Change::ResetPort { node, port } => {
                    json!({ "kind": "reset", "node": node, "port": port })
                }
                orla_session::Change::Restored { revision, laws } => {
                    json!({ "kind": "restored", "snapshot_revision": revision, "laws": if *laws { "snapshot" } else { "current" } })
                }
            };
            json["revision"] = json!(record.revision);
            json["after_frames"] = json!(record.after_frames);
            json["wall"] = json!(record.wall);
            json["origin"] = origin(&record.origin);
            json
        })
        .collect();
    let attempts: Vec<Json> = session
        .attempts_since(since)
        .iter()
        .map(|attempt| {
            json!({
                "revision": attempt.revision, "frame": attempt.frame, "wall": attempt.wall,
                "origin": origin(&attempt.origin), "subject": attempt.subject, "why": attempt.why,
                "source": attempt.source.map(|s| format!("{s:016x}")),
            })
        })
        .collect();
    Ok(json!({ "revision": revision, "frame": frame, "changes": changes, "attempts": attempts }))
}

/// The Graph the scene lowered to.
pub(crate) fn graph(session: &Session) -> Result<Json, Json> {
    let graph = session
        .graph()
        .ok_or_else(|| failure("no scene is loaded"))?;
    Ok(graph_json(graph, session.params(), &session.placements()))
}

/// The scene document on disk, or with `version`, a version the session
/// remembers from a change that loaded it.
pub(crate) fn document(session: &Session, request: &Request) -> Result<Json, Json> {
    if let Some(version) = request.get("version").and_then(Json::as_str) {
        let text = u64::from_str_radix(version, 16)
            .ok()
            .and_then(|digest| session.scene_at(digest))
            .ok_or_else(|| {
                failure(format!(
                    "version {version} is not one the session remembers"
                ))
            })?;
        return Ok(
            json!({ "path": session.root().display().to_string(), "text": text, "version": version }),
        );
    }
    let text = session
        .scene_text()
        .map_err(|e| failure(format!("{}: {e}", session.root().display())))?;
    let version = crate::create::document_version(&text);
    Ok(json!({ "path": session.root().display().to_string(), "text": text, "version": version }))
}

/// Every builtin and form, or the one named.
pub(crate) fn help(request: &Request) -> Result<Json, Json> {
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
    let shape = |form: &orla_compiler::help::Form| {
        json!({
            "name": form.name,
            "kind": form.kind,
            "shape": form.shape,
            "doc": form.doc,
        })
    };
    match request.get("name").and_then(Json::as_str) {
        Some(name) => match orla_compiler::help::builtin(name) {
            Some(builtin) => Ok(entry(&builtin)),
            None => match orla_compiler::help::form_named(name) {
                Some(form) => Ok(shape(&form)),
                None => {
                    let similar = orla_compiler::help::similar(name);
                    Err(json!({
                        "message": format!("nothing named `{name}`; similar: {}", similar.join(", ")),
                        "similar": similar,
                    }))
                }
            },
        },
        None => Ok(json!({
            "builtins": orla_compiler::help::builtins().iter().map(entry).collect::<Vec<_>>(),
            "forms": orla_compiler::help::forms().iter().map(shape).collect::<Vec<_>>(),
        })),
    }
}
