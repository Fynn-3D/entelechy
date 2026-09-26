//! Operations on a node's Fields: read what it holds, write its committed
//! state, trace one element through its Kernel, and keep it materialized
//! for reading.

use orla_graph::Source;
use orla_kernel::Shape;
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
            // A coordinate past 32 bits stays past the end, rather than wrapping
            // onto an element that is there.
            let axis = |i: usize| {
                list.get(i)
                    .and_then(Json::as_u64)
                    .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
            };
            [axis(0), axis(1), axis(2)]
        }
        (None, Some(element)) => [u32::try_from(element).unwrap_or(u32::MAX), 0, 0],
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
        let data = orla_formats::file::decode(&bytes).map_err(|e| failure(e.to_string()))?;
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
        let data = session.field(handle).map_err(|e| failure(e.to_string()))?;
        let bytes: Vec<u8> = if as_file {
            orla_formats::file::encode_data(data)
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
        .picture(handle)
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

/// The version of a snapshot's format: what a reader of one checks.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The world as it stands, whole: the position, every published Kernel
/// with its generation and source, every port's committed state with its
/// layout, extent and words. Given back, or written to `to` where the
/// session was started with `--allow file.write` for it; either way the
/// session keeps nothing of it. Only committed state is taken.
pub(crate) fn snapshot(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let snapshot = session.snapshot().map_err(failure)?;
    let laws: Vec<Json> = snapshot
        .laws
        .iter()
        .map(|(id, generation, source)| {
            json!({ "kernel": id.to_string(), "generation": generation.0, "source": source })
        })
        .collect();
    let states: Vec<Json> = snapshot
        .states
        .iter()
        .map(|state| {
            let bytes: Vec<u8> = state.words.iter().flat_map(|w| w.to_le_bytes()).collect();
            json!({
                "node": state.node,
                "port": state.port,
                "layout": state.schema.to_string(),
                "shape": state.domain.shape().to_string(),
                "extent": [state.domain.width(), state.domain.height(), state.domain.depth()],
                "data": base64(&bytes),
            })
        })
        .collect();
    let json = json!({
        "snapshot": SNAPSHOT_VERSION,
        "revision": snapshot.revision,
        "frame": snapshot.frame,
        "laws": laws,
        "states": states,
        "scene": snapshot.scene.as_ref().map(|text| json!({
            "version": crate::create::document_version(text),
            "text": text,
        })),
    });
    let Some(to) = request.get("to") else {
        return Ok(json);
    };
    let to = to.as_str().ok_or_else(|| failure("`to` is a file path"))?;
    let path = std::path::Path::new(to);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        session
            .root()
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path)
    };
    // Written where the person running the session allowed, whole.
    let target = session
        .allowed_write(&path)
        .map_err(|why| failure(format!("cannot write the snapshot: {why}")))?;
    session
        .store()
        .write(&target, json.to_string().as_bytes())
        .map_err(|e| failure(format!("{}: {e}", target.display())))?;
    Ok(json!({
        "path": target.display().to_string(),
        "revision": snapshot.revision,
        "frame": snapshot.frame,
    }))
}

/// Restores a snapshot's world: given inline, as `snapshot` gave it, or
/// read `from` a file where the session may read. `laws` says whether
/// the snapshot's Kernels come back with it or its state goes under the
/// laws that run; there is no default, so a restore says which. The
/// world stands only if a trial frame, under the context given as
/// `evaluate` takes it, succeeds, and that frame commits nothing.
pub(crate) fn restore(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let laws = match request.get("laws").and_then(Json::as_str) {
        Some("current") => orla_session::Laws::Current,
        Some("snapshot") => orla_session::Laws::Snapshot,
        _ => {
            return Err(failure(
                "`laws` says what the restored world runs under: \"current\" or \"snapshot\"",
            ));
        }
    };
    let read;
    let json = match (request.get("snapshot"), request.get("from")) {
        (Some(json), None) => json,
        (None, Some(Json::String(from))) => {
            let path = std::path::Path::new(from);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                session
                    .root()
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(path)
            };
            session
                .allowed_read(&path)
                .map_err(|why| failure(format!("cannot read the snapshot: {why}")))?;
            let text = session
                .store()
                .read_text(&path)
                .map_err(|e| failure(format!("{}: {e}", path.display())))?;
            read = serde_json::from_str::<Json>(&text)
                .map_err(|e| failure(format!("{}: not a snapshot: {e}", path.display())))?;
            &read
        }
        _ => return Err(failure("`restore` takes a `snapshot`, or a file `from`")),
    };
    // The structure is the scene as it stands unless the restore asks
    // for the snapshot's.
    let structure = match request.get("structure").and_then(Json::as_str) {
        None | Some("current") => orla_session::Structure::Current,
        Some("snapshot") => orla_session::Structure::Snapshot,
        Some(other) => {
            return Err(failure(format!(
                "`structure` is \"current\" or \"snapshot\", not `{other}`"
            )));
        }
    };
    let snapshot = snapshot_from(json)?;
    let context = context_of(session, request.json());
    let report = session
        .restore(&snapshot, laws, structure, &context)
        .map_err(failure)?;
    let (revision, frame) = session.position();
    Ok(json!({
        "revision": revision,
        "frame": frame,
        "report": crate::report_json(session, &report),
    }))
}

/// A snapshot read back from what `snapshot` wrote.
fn snapshot_from(json: &Json) -> Result<orla_session::Snapshot, Json> {
    let bad = |what: &str| failure(format!("not a snapshot: {what}"));
    if json.get("snapshot").and_then(Json::as_u64) != Some(u64::from(SNAPSHOT_VERSION)) {
        return Err(bad(&format!(
            "`snapshot` is the format's version, {SNAPSHOT_VERSION}"
        )));
    }
    let number = |key: &str| json.get(key).and_then(Json::as_u64).ok_or_else(|| bad(key));
    let mut laws = Vec::new();
    for law in json
        .get("laws")
        .and_then(Json::as_array)
        .ok_or_else(|| bad("laws"))?
    {
        let id = law
            .get("kernel")
            .and_then(Json::as_str)
            .and_then(|id| id.parse().ok())
            .ok_or_else(|| bad("a law's kernel"))?;
        let generation = law
            .get("generation")
            .and_then(Json::as_u64)
            .ok_or_else(|| bad("a law's generation"))?;
        let source = law
            .get("source")
            .and_then(Json::as_str)
            .ok_or_else(|| bad("a law's source"))?;
        laws.push((id, orla_kernel::Generation(generation), source.to_owned()));
    }
    let mut states = Vec::new();
    for state in json
        .get("states")
        .and_then(Json::as_array)
        .ok_or_else(|| bad("states"))?
    {
        let text = |key: &str| {
            state
                .get(key)
                .and_then(Json::as_str)
                .ok_or_else(|| bad(&format!("a state's {key}")))
        };
        let schema: orla_kernel::Schema = text("layout")?
            .parse()
            .map_err(|e: String| bad(&format!("a state's layout: {e}")))?;
        // Three whole sides, each one a Field may have: nothing dropped,
        // nothing truncated.
        let extent: Option<Vec<u32>> = state.get("extent").and_then(Json::as_array).map(|sides| {
            sides
                .iter()
                .map(|side| side.as_u64().and_then(|side| u32::try_from(side).ok()))
                .collect::<Option<Vec<u32>>>()
                .unwrap_or_default()
        });
        let Some([w, h, d]) = extent.as_deref().and_then(|e| <[u32; 3]>::try_from(e).ok()) else {
            return Err(bad("a state's extent is three whole sides"));
        };
        let domain = match text("shape")? {
            "Rows" => orla_kernel::Domain::Rows(w),
            "Grid" => orla_kernel::Domain::Grid(w, h),
            "Volume" => orla_kernel::Domain::Volume(w, h, d),
            other => return Err(bad(&format!("`{other}` is no shape"))),
        };
        let bytes = unbase64(text("data")?).map_err(|e| bad(&format!("a state's data: {e}")))?;
        if bytes.len() % 4 != 0 {
            return Err(bad("a state's data is whole words"));
        }
        let words = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        states.push(orla_session::StateCopy {
            node: text("node")?.to_owned(),
            port: text("port")?.to_owned(),
            schema,
            domain,
            words,
        });
    }
    let scene = json
        .get("scene")
        .and_then(|scene| scene.get("text"))
        .and_then(Json::as_str)
        .map(str::to_owned);
    Ok(orla_session::Snapshot {
        revision: number("revision")?,
        frame: number("frame")?,
        laws,
        states,
        scene,
    })
}
