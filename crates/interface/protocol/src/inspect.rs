//! Operations that look and change nothing: what the session is, what is
//! published, what the scene is, and what the language offers. Each is
//! given the session to read and nothing to write with.

use serde_json::{Value as Json, json};

use orla_session::Session;

use crate::request::Request;
use crate::wire::{failure, graph_json, manifest_json};

/// The session at a glance, and the versions it speaks.
pub(crate) fn status(session: &Session) -> Result<Json, Json> {
    Ok(json!({
        "versions": { "api": orla_runtime::ABI_VERSION, "language": orla_language::LANGUAGE_VERSION, "ir": orla_ir::IR_VERSION, "package": orla_session::PACKAGE_VERSION },
        "backend": session.runtime().backend_name(),
        "scene": session.root().display().to_string(),
        "watched": session.watched().iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "kernels": session.runtime().kernels().count(),
        "nodes": session.graph().map_or(0, |graph| graph.nodes.len()),
        "fusion": session.fusion(),
    }))
}

/// A published version's manifest, its structs named: a Kernel's from
/// its module, a native operation's from its own table.
fn version_json(version: &orla_runtime::KernelVersion) -> Json {
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
    let version = session
        .runtime()
        .kernel(&id)
        .ok_or_else(|| failure(format!("{id} is not published")))?;
    let mut json = version_json(&version);
    json["source"] = Json::String(version.source().to_owned());
    Ok(json)
}

/// The Graph the scene lowered to.
pub(crate) fn graph(session: &Session) -> Result<Json, Json> {
    let graph = session
        .graph()
        .ok_or_else(|| failure("no scene is loaded"))?;
    Ok(graph_json(graph, session.params(), &session.placements()))
}

/// The scene document as it is on disk.
pub(crate) fn document(session: &Session) -> Result<Json, Json> {
    let text = std::fs::read_to_string(session.root())
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
