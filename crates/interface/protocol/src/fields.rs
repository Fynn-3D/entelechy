//! Operations on a node's Fields: read what it holds, write its committed
//! state, trace one element through its Kernel, and keep it materialized
//! for reading.

use orla_graph::Source;
use orla_runtime::Shape;
use serde_json::{Value as Json, json};

use orla_formats::base64::{base64, unbase64};
use orla_formats::ppm;
use orla_session::Session;

use crate::request::Request;
use crate::wire::{context_of, failure, trace_json};

/// Refuses a name that is not a node of the loaded scene, naming the
/// near miss where there is one. `pin` used to take any word and say
/// `ok`, which a caller only found out about one request later, as a
/// read that told it to pin.
fn known_node(session: &Session, node: &str) -> Result<(), Json> {
    let Some(graph) = session.graph() else {
        return Err(failure("no scene is loaded"));
    };
    if graph.nodes.iter().any(|n| n.id.0 == node) {
        return Ok(());
    }
    let known: Vec<&str> = graph.nodes.iter().map(|n| n.id.0.as_str()).collect();
    Err(failure(format!(
        "`{node}` is not a node of this scene{}",
        orla_language::did_you_mean(node, known)
    )))
}

/// Runs one element of a node on the interpreter, every binding shown.
pub(crate) fn trace(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let node = request.text("node")?;
    let at = match (
        request.get("at").and_then(Json::as_array),
        request.get("element").and_then(Json::as_u64),
    ) {
        (Some(list), _) => {
            let axis = |i: usize| list.get(i).and_then(Json::as_u64).unwrap_or(0) as u32;
            [axis(0), axis(1), axis(2)]
        }
        (None, Some(element)) => [element as u32, 0, 0],
        (None, None) => {
            return Err(failure(
                "`trace` names the element as `at`: [x, y, z] or `element`: n",
            ));
        }
    };
    let context = context_of(session, request.json());
    let id = orla_graph::NodeId(node.clone());
    let (trace, kernel) = session.trace(&id, at, &context).map_err(failure)?;
    let version = session
        .runtime()
        .kernel(&kernel)
        .ok_or_else(|| failure(format!("`{kernel}` is not published")))?;
    let module = version
        .module()
        .ok_or_else(|| failure("a native operation has no trace"))?;
    Ok(trace_json(module, &node, &kernel, &trace))
}

/// Replaces a node's committed state.
pub(crate) fn write(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let node = request.text("node")?;
    let port = request
        .get("port")
        .and_then(Json::as_str)
        .unwrap_or("result")
        .to_owned();
    let data = request.text("data")?;
    let bytes = unbase64(&data).map_err(failure)?;
    let words: Vec<u32> = if request.get("format").and_then(Json::as_str) == Some("field") {
        // A Field file: its layout and extent must be the port's.
        let data = orla_runtime::file::decode(&bytes).map_err(|e| failure(e.to_string()))?;
        let current = session
            .cached(&orla_graph::NodeId(node.clone()), &port)
            .ok_or_else(|| failure(format!("`{node}.{port}` has no value yet; evaluate first")))?;
        let host = session.runtime().host();
        let (schema, domain) = (
            host.schema(current)
                .map_err(|e| failure(e.to_string()))?
                .clone(),
            host.domain(current).map_err(|e| failure(e.to_string()))?,
        );
        if data.schema() != &schema || data.domain() != domain {
            return Err(failure(format!(
                "the file is {} {} {}, but `{node}.{port}` is {} {} {}",
                data.domain().shape(),
                data.domain().count(),
                data.schema(),
                domain.shape(),
                domain.count(),
                schema
            )));
        }
        data.values().into_owned()
    } else {
        if bytes.len() % 4 != 0 {
            return Err(failure("`data` must be whole 32-bit words"));
        }
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let count = words.len();
    session
        .write(&orla_graph::NodeId(node.clone()), &port, words)
        .map_err(|e| failure(e.to_string()))?;
    Ok(json!({ "node": node, "port": port, "words": count }))
}

/// Reads an output or a node's port: pixels, words, or a Field file.
pub(crate) fn read(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let graph = session
        .graph()
        .ok_or_else(|| failure("no scene is loaded"))?;
    let handle = if let Some(node) = request.get("node").and_then(Json::as_str) {
        let port = request
            .get("port")
            .and_then(Json::as_str)
            .unwrap_or("result");
        session
            .cached(&orla_graph::NodeId(node.to_owned()), port)
            .ok_or_else(|| {
                // A name that is not a node at all is the likelier
                // mistake, and pinning it again will not help.
                if !graph.nodes.iter().any(|n| n.id.0 == node) {
                    let known: Vec<&str> = graph.nodes.iter().map(|n| n.id.0.as_str()).collect();
                    return failure(format!(
                        "`{node}` is not a node of this scene{}",
                        orla_language::did_you_mean(node, known)
                    ));
                }
                failure(format!(
                    "node `{node}` has no materialized `{port}`; pin it and evaluate again"
                ))
            })?
    } else {
        let name = request
            .get("output")
            .and_then(Json::as_str)
            .map(str::to_owned)
            .or_else(|| graph.outputs.first().map(|(name, _)| name.clone()))
            .ok_or_else(|| failure("the graph has no output"))?;
        let (_, source) = graph
            .outputs
            .iter()
            .find(|(output, _)| *output == name)
            .ok_or_else(|| failure(format!("no output named `{name}`")))?;
        match source {
            Source::Node { node, port } => session
                .cached(node, port)
                .ok_or_else(|| failure("evaluate before reading"))?,
            Source::External(_) => return Err(failure("output is an external image")),
        }
    };
    // A Field of pixels reads as an image; anything else reads as
    // words with its shape and layout.
    let (domain, schema) = {
        let host = session.runtime().host();
        (
            host.domain(handle).map_err(|e| failure(e.to_string()))?,
            host.schema(handle)
                .map_err(|e| failure(e.to_string()))?
                .clone(),
        )
    };
    let as_file = request.get("format").and_then(Json::as_str) == Some("field");
    if as_file || !schema.is_pixel() || domain.shape() == Shape::Volume {
        session
            .runtime_mut()
            .materialize(handle)
            .map_err(|e| failure(e.to_string()))?;
        let data = session
            .runtime()
            .host()
            .data(handle)
            .map_err(|e| failure(e.to_string()))?;
        let bytes: Vec<u8> = if as_file {
            orla_runtime::file::encode_data(data)
        } else {
            data.values().iter().flat_map(|w| w.to_le_bytes()).collect()
        };
        return Ok(json!({
            "shape": domain.shape().to_string(),
            "extent": [domain.width(), domain.height(), domain.depth()],
            "count": domain.count(),
            "schema": schema.to_string(),
            "format": if as_file { "field" } else { "words" },
            "data": base64(&bytes),
        }));
    }
    let image = session
        .runtime_mut()
        .image(handle)
        .map_err(|e| failure(e.to_string()))?;
    let format = request
        .get("format")
        .and_then(Json::as_str)
        .unwrap_or("rgba32f");
    let bytes = match format {
        "rgba32f" => image
            .pixels()
            .iter()
            .flat_map(|pixel| pixel.iter().flat_map(|v| v.to_le_bytes()))
            .collect::<Vec<u8>>(),
        "ppm" => ppm::encode(&image),
        other => {
            return Err(failure(format!(
                "unknown format `{other}`; use rgba32f or ppm"
            )));
        }
    };
    Ok(json!({
        "width": image.width(),
        "height": image.height(),
        "format": format,
        "data": base64(&bytes),
    }))
}

/// Keeps a node materialized so `read` can see it.
pub(crate) fn pin(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let node = request.text("node")?;
    known_node(session, &node)?;
    session.pin(orla_graph::NodeId(node.clone()));
    Ok(json!({ "pinned": node }))
}

/// Lets a node be fused away again.
pub(crate) fn unpin(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let node = request.text("node")?;
    known_node(session, &node)?;
    session.unpin(&orla_graph::NodeId(node.clone()));
    Ok(json!({ "unpinned": node }))
}
