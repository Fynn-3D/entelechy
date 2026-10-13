//! Read-only operations: the session, published Kernels, the scene and the
//! language's builtins. Each takes the session by shared reference.

use serde_json::{Value as Json, json};

use entelechy_session::Session;

use crate::request::Request;
use crate::wire::{failure, graph_json, manifest_json};

/// The session at a glance, and the versions it speaks.
pub(crate) fn status(session: &Session) -> Result<Json, Json> {
    Ok(json!({
        "versions": versions(),
        "backend": session.runtime().backend_name(),
        "scene": session.root().display().to_string(),
        "watched": session.watched().iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "kernels": session.runtime().kernels().filter(|v| v.native().is_none()).count(),
        "natives": session.runtime().kernels().filter(|v| v.native().is_some()).count(),
        "nodes": session.graph().map_or(0, |graph| graph.nodes.len()),
        "loaded": session.graph().is_some(),
        "refused": session.why_no_scene(),
        "fusion": session.fusion(),
        "revision": session.position().0,
        "frame": session.position().1,
        "grants": session.runtime().grants().iter().map(|c| c.name()).collect::<Vec<_>>(),
    }))
}

/// A published version's manifest with its structs, from a Kernel's
/// module or a native operation's table.
fn version_json(version: &entelechy_kernel::KernelVersion) -> Json {
    let native = version.native().map(|op| op.structs());
    let structs = match (version.module(), &native) {
        (Some(module), _) => &module.structs[..],
        (None, Some(structs)) => &structs[..],
        (None, None) => &[][..],
    };
    let mut json = manifest_json(version.manifest(), version.generation(), structs);
    with_docs(&mut json, version.source());
    json
}

/// Adds what the Kernel's source says of it and of its ports and params,
/// its `///` lines, as `doc`.
fn with_docs(json: &mut Json, source: &str) {
    let docs = entelechy_language::doc::kernel_docs(source);
    if let Some(doc) = docs.doc {
        json["doc"] = Json::String(doc);
    }
    for list in ["inputs", "params", "outputs"] {
        if let Some(entries) = json[list].as_array_mut() {
            for entry in entries {
                if let Some((_, doc)) = docs.ports.iter().find(|(name, _)| entry["name"] == *name) {
                    entry["doc"] = Json::String(doc.clone());
                }
            }
        }
    }
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
            .source_at(&id, entelechy_kernel::Generation(generation))
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
    // What it calls or builds that its document declares beside it.
    if let Some((document, shared)) = session.shared_with(&id)
        && !shared.is_empty()
    {
        json["document"] = json!(document.display().to_string());
        json["shared"] = json!(shared);
    }
    Ok(json)
}

/// The changes that made revisions after `since` and the attempts refused
/// since, each with its origin, as far back as the session remembers.
pub(crate) fn history(session: &Session, request: &Request) -> Result<Json, Json> {
    let since = request.get("since").and_then(Json::as_u64).unwrap_or(0);
    let (revision, frame) = session.position();
    let origin = |origin: &entelechy_session::Origin| match origin {
        entelechy_session::Origin::Document(path) => {
            json!({ "document": path.display().to_string() })
        }
        entelechy_session::Origin::Protocol(channel) => json!({ "protocol": channel }),
        entelechy_session::Origin::Local => json!("local"),
    };
    let generation = |g: &Option<entelechy_kernel::Generation>| g.map(|g| g.0);
    let versions = |documents: &[(std::path::PathBuf, u64)]| {
        documents
            .iter()
            .map(|(path, version)| json!({ "path": path.display().to_string(), "version": format!("{version:016x}") }))
            .collect::<Vec<_>>()
    };
    let changes: Vec<Json> = session
        .changes_since(since)
        .iter()
        .map(|record| {
            let mut json = match &record.change {
                entelechy_session::Change::Published {
                    id,
                    from,
                    to,
                    source,
                } => json!({
                    "kind": "published", "kernel": id.to_string(), "from": generation(from),
                    "to": to.0, "source": format!("{source:016x}"),
                }),
                entelechy_session::Change::Retired { id, from } => {
                    json!({ "kind": "retired", "kernel": id.to_string(), "from": generation(from) })
                }
                entelechy_session::Change::Scene { document, documents } => json!({
                    "kind": "scene",
                    "version": format!("{document:016x}"),
                    "documents": versions(documents),
                }),
                entelechy_session::Change::Documents { written } => json!({
                    "kind": "documents",
                    "documents": versions(written),
                }),
                entelechy_session::Change::Param { name } => json!({ "kind": "param", "name": name }),
                entelechy_session::Change::Written { node, port } => {
                    json!({ "kind": "written", "node": node, "port": port })
                }
                entelechy_session::Change::Reset => json!({ "kind": "reset" }),
                entelechy_session::Change::ResetPort { node, port } => {
                    json!({ "kind": "reset", "node": node, "port": port })
                }
                entelechy_session::Change::Restored { revision, laws } => {
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
    Ok(graph_json(
        graph,
        session.params(),
        &session.placements(),
        session.fonts(),
    ))
}

/// The scene document on disk, or with `version`, a version the session
/// remembers from a change that loaded it; with `path`, the document of
/// the scene there (from the scene's directory), as a compose names it.
pub(crate) fn document(session: &Session, request: &Request) -> Result<Json, Json> {
    if let Some(path) = request.get("path") {
        let path = path
            .as_str()
            .ok_or_else(|| failure("`path` is a document's path, from the scene's directory"))?;
        if let Some(version) = request.get("version").and_then(Json::as_str) {
            let digest = u64::from_str_radix(version, 16)
                .map_err(|_| failure(format!("version {version} is not one `document` answers")))?;
            let text = session
                .document_at(std::path::Path::new(path), digest)
                .map_err(failure)?;
            let (landed, _) = session
                .composed_document(std::path::Path::new(path))
                .map_err(failure)?;
            return Ok(
                json!({ "path": landed.display().to_string(), "text": text, "version": version }),
            );
        }
        let (landed, text) = session
            .composed_document(std::path::Path::new(path))
            .map_err(failure)?;
        let text = text.ok_or_else(|| {
            failure(format!(
                "`{}` is not written yet: a compose may make it",
                landed.display()
            ))
        })?;
        let version = crate::create::document_version(&text);
        return Ok(
            json!({ "path": landed.display().to_string(), "text": text, "version": version }),
        );
    }
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

/// The versions an answer about Entelechy holds for.
pub(crate) fn versions() -> Json {
    json!({
        "entelechy": env!("CARGO_PKG_VERSION"),
        "api": entelechy_kernel::ABI_VERSION,
        "language": entelechy_language::LANGUAGE_VERSION,
        "ir": entelechy_ir::IR_VERSION,
        "package": entelechy_session::PACKAGE_VERSION,
    })
}

/// The whole reference, or the items named; see [`crate::help`].
pub(crate) fn help(request: &Request) -> Result<Json, Json> {
    match request.get("name").and_then(Json::as_str) {
        Some(name) => crate::help::named(name),
        None => Ok(json!({
            "reference": crate::help::reference(),
            "versions": versions(),
        })),
    }
}
