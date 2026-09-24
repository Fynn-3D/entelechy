//! What a caller observes of a session without changing what it
//! computes: an output saved as a file, a pixel, an element's field, an
//! element traced through its Kernel, a picture exported.
//!
//! Nothing here sees a session. Each observation is given the parts it
//! reads — the Graph, the evaluator's cache, the runtime — and nothing
//! else, so an observation cannot reach the ledger, the inputs or the
//! scene's documents by accident.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use orla_formats::ppm;
use orla_graph::{Evaluator, Graph, Node, NodeId};
use orla_runtime::{Context, Handle, KernelVersion, Runtime, Validated, Value};

use crate::SessionError;

/// A vector's component by name, as a Kernel spells it (`x` or `r` and
/// the rest): which word it is, and what type it holds.
fn component_of(ty: orla_ir::Type, name: &str) -> Option<(usize, orla_ir::Type)> {
    use orla_ir::Type;
    let (lanes, element) = match ty {
        Type::Vec2 => (2, Type::Float),
        Type::Vec3 => (3, Type::Float),
        Type::Vec4 => (4, Type::Float),
        Type::IVec2 => (2, Type::Int),
        Type::IVec3 => (3, Type::Int),
        _ => return None,
    };
    let at = ["xyzw", "rgba"]
        .iter()
        .find_map(|alphabet| alphabet.find(name))?;
    (name.len() == 1 && at < lanes).then_some((at, element))
}

/// `node.port[index].field.field` as a replay writes it and the
/// protocol takes it: the node, the port, the element, and the path
/// through the struct to a value.
pub fn parse_target(text: &str) -> Result<(String, String, u32, Vec<String>), String> {
    let (head, rest) = text
        .split_once('[')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (index, tail) = rest
        .split_once(']')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (node, port) = head
        .rsplit_once('.')
        .ok_or_else(|| "name `node.port`".to_string())?;
    let index = index
        .parse::<u32>()
        .map_err(|_| format!("element index `{index}` is not a whole number"))?;
    let path: Vec<String> = tail
        .split('.')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect();
    Ok((node.to_owned(), port.to_owned(), index, path))
}

/// What a scene output of this name is wired to, when the name is an
/// output rather than a node.
fn output_source(graph: Option<&Graph>, node: &NodeId, port: &str) -> Option<(NodeId, String)> {
    let graph = graph?;
    if graph.nodes.iter().any(|n| n.id == *node) {
        return None;
    }
    let (name, _) = graph
        .outputs
        .iter()
        .find(|(name, _)| name == &node.0 && port == "result")?;
    match &graph.outputs.iter().find(|(o, _)| o == name)?.1 {
        orla_graph::Source::Node { node, port } => Some((node.clone(), port.clone())),
        orla_graph::Source::External(_) => None,
    }
}

/// Why a node's output is not there to be saved: the name may be a
/// placed graph, which is its nodes by the time the graph runs, or a
/// node of one written without its repetition, or no node at all.
fn why_not_saved(graph: Option<&Graph>, node: &NodeId, port: &str) -> String {
    let Some(graph) = graph else {
        return format!("`{node}.{port}` has no value; no scene is loaded");
    };
    let named: Vec<&str> = graph
        .nodes
        .iter()
        .map(|n| n.id.0.as_str())
        .filter(|id| {
            id.starts_with(&format!("{node}."))
                || id.starts_with(&format!("{node}/"))
                || id.split('/').next() == Some(node.0.as_str())
        })
        .collect();
    if !graph.nodes.iter().any(|n| n.id == *node) {
        if let Some(last) = named.last() {
            return format!(
                "`{node}` is not a node: it is placed as {} of them, the last `{last}`, which is what a `--save` names (or give the scene an `output`)",
                named.len()
            );
        }
        return format!("`{node}` is not a node of this scene");
    }
    format!("`{node}.{port}` has no value; pin it and evaluate")
}

/// Writes a node's output as a file; see [`crate::Session::save`].
pub(crate) fn save(
    graph: Option<&Graph>,
    evaluator: &Evaluator,
    runtime: &mut Runtime,
    node: &NodeId,
    port: &str,
    path: &Path,
) -> Result<(), String> {
    // A scene's own output name stands for what it is wired to, so
    // `--save settled=...` works with the names the scene reads by.
    let (node, port) = match output_source(graph, node, port) {
        Some((node, port)) => (node, port),
        None => (node.clone(), port.to_owned()),
    };
    let (node, port) = (&node, port.as_str());
    let Some(handle) = evaluator.cached(node, port) else {
        return Err(why_not_saved(graph, node, port));
    };
    runtime.materialize(handle).map_err(|e| e.to_string())?;
    let data = runtime.host().data(handle).map_err(|e| e.to_string())?;
    let bytes = orla_formats::encode(data, &orla_formats::extension(path), None).map_err(
        |why| match why {
            orla_formats::EncodeError::Unknown => why.to_string(),
            orla_formats::EncodeError::Unfit(what) => {
                format!("`{node}.{port}` is {what}; save it as .field")
            }
        },
    )?;
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// How many elements a node's output holds; see
/// [`crate::Session::element_count`].
pub(crate) fn element_count(
    evaluator: &Evaluator,
    runtime: &Runtime,
    node: &NodeId,
    port: &str,
) -> Result<usize, String> {
    let handle = evaluator
        .cached(node, port)
        .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
    runtime
        .host()
        .domain(handle)
        .map(|domain| domain.count())
        .map_err(|e| e.to_string())
}

/// The node a trace runs and the version it runs: refused for a node
/// the scene does not have, a Kernel not published, or a native
/// operation, which has no source to walk.
pub(crate) fn trace_target(
    graph: Option<&Graph>,
    runtime: &Runtime,
    node: &NodeId,
) -> Result<(Node, Arc<KernelVersion>), String> {
    let graph = graph.ok_or("no scene is loaded")?;
    let target = graph
        .nodes
        .iter()
        .find(|n| &n.id == node)
        .ok_or_else(|| format!("no node named `{node}`"))?
        .clone();
    let version = runtime
        .kernel(&target.kernel)
        .ok_or_else(|| format!("`{}` is not published", target.kernel))?;
    if version.module().is_none() {
        return Err(format!(
            "`{}` is a native operation; there is nothing to trace",
            target.kernel
        ));
    }
    Ok((target, version))
}

/// The nodes whose Fields a trace of `target` reads, which must be
/// kept for it.
pub(crate) fn trace_sources(target: &Node) -> Vec<NodeId> {
    let mut sources = Vec::new();
    for binding in target.inputs.values() {
        let read = match binding {
            orla_graph::Binding::Now(source) => vec![source],
            orla_graph::Binding::Previous {
                source, initial, ..
            } => vec![source, initial],
        };
        for source in read {
            if let orla_graph::Source::Node { node, .. } = source {
                sources.push(node.clone());
            }
        }
    }
    sources.sort();
    sources.dedup();
    sources
}

/// Runs element `at` of `target` on the interpreter over the invocation
/// an evaluation would give it now. `written` is the Kernel as its
/// document wrote it, compiled without the optimizer, when a document
/// owns it; the running module answers otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_element(
    graph: &Graph,
    evaluator: &Evaluator,
    runtime: &mut Runtime,
    inputs: &BTreeMap<String, Handle>,
    params: &BTreeMap<String, Value>,
    written: Option<Validated>,
    (target, version): (&Node, &Arc<KernelVersion>),
    context: &Context,
    at: [u32; 3],
) -> Result<orla_cpu::Trace, String> {
    let invocation = evaluator
        .invocation(graph, target, version, runtime, inputs, context, params)
        .map_err(|error| error.to_string())?;
    for handle in &invocation.inputs {
        runtime
            .materialize(*handle)
            .map_err(|error| error.to_string())?;
    }
    // A trace walks the Kernel as written: the same source without the
    // optimizer, so its functions are still called and its `let`s are
    // all there.
    let module = written
        .as_ref()
        .unwrap_or_else(|| version.module().expect("a traced Kernel has a module"));
    // It runs with its work accounted, as a backend runs it.
    let module = orla_ir::account(module).map_err(|error| error.to_string())?;
    // What it requires, of what the session grants, as an execution.
    let granted: orla_runtime::Capabilities = module
        .manifest
        .requires
        .iter()
        .filter(|capability| runtime.grants().contains(*capability))
        .collect();
    let mut scope = orla_runtime::Scope::new(runtime.host_mut(), granted);
    orla_cpu::trace(&module, &invocation, &mut scope, at).map_err(|error| error.to_string())
}

/// One element field of a node's output; see
/// [`crate::Session::element_field`].
pub(crate) fn element_field(
    graph: Option<&Graph>,
    evaluator: &Evaluator,
    runtime: &mut Runtime,
    (node, port): (&NodeId, &str),
    index: u32,
    path: &[String],
) -> Result<(orla_ir::Type, Vec<u32>), String> {
    use orla_ir::Type;
    let kernel = graph
        .and_then(|graph| graph.nodes.iter().find(|n| &n.id == node))
        .map(|n| n.kernel.clone())
        .ok_or_else(|| format!("no node `{node}` in the scene"))?;
    let version = runtime
        .kernel(&kernel)
        .ok_or_else(|| format!("`{kernel}` is not published"))?;
    let output = version
        .manifest()
        .outputs
        .iter()
        .find(|o| o.name == port)
        .ok_or_else(|| format!("`{kernel}` has no output `{port}`"))?;
    // A Kernel names its structs in its module; a native operation, in
    // its own table.
    let structs: Vec<orla_ir::StructDef> = match (version.module(), version.native()) {
        (Some(module), _) => module.structs.clone(),
        (None, Some(op)) => op.structs(),
        (None, None) => Vec::new(),
    };
    // Follow the field path through the struct table to a word offset
    // and a type.
    let mut ty = output.element();
    let mut offset = 0usize;
    let words_of = |ty: Type| orla_ir::schema_of(ty, &structs).map_or(1, |s| s.value_words());
    for field in path {
        // A struct's field by name, or a vector's component by the
        // letter the language spells it with.
        if let Some((component, element)) = component_of(ty, field) {
            offset += component;
            ty = element;
            continue;
        }
        let Type::Struct(id) = ty else {
            return Err(format!(
                "`{field}` names a field or a component, but the value is a {ty}"
            ));
        };
        let def = structs
            .get(id as usize)
            .ok_or_else(|| format!("`{kernel}` does not name the fields of its `{port}`"))?;
        let mut found = None;
        for f in &def.fields {
            if &f.name == field {
                found = Some(f.ty);
                break;
            }
            offset += words_of(f.ty);
        }
        ty = found.ok_or_else(|| format!("struct `{}` has no field `{field}`", def.name))?;
    }
    let handle = evaluator
        .cached(node, port)
        .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
    runtime.materialize(handle).map_err(|e| e.to_string())?;
    let element = runtime
        .host()
        .element(handle, index)
        .map_err(|e| e.to_string())?;
    Ok((ty, element[offset..offset + words_of(ty)].to_vec()))
}

/// Writes a cached image to a PPM file; see [`crate::Session::export`].
pub(crate) fn export(
    runtime: &mut Runtime,
    image: Handle,
    path: &Path,
) -> Result<(), SessionError> {
    let image = runtime
        .image(image)
        .map_err(|e| SessionError::Observe(e.to_string()))?;
    ppm::write(path, &image).map_err(|e| SessionError::Image(path.to_path_buf(), e))
}

/// One pixel of a cached image; see [`crate::Session::read_pixel`].
pub(crate) fn read_pixel(
    runtime: &mut Runtime,
    image: Handle,
    x: u32,
    y: u32,
) -> Result<orla_runtime::Pixel, SessionError> {
    // Four words out of the Field, not a copy of the picture: a replay
    // checks a handful of pixels a frame, and the picture it checks them
    // in is thirty-three megabytes at 1920x1080. The Field comes to the
    // Host once; every pixel after that is an index.
    runtime
        .materialize(image)
        .map_err(|e| SessionError::Observe(e.to_string()))?;
    let host = runtime.host();
    let domain = host.domain(image).map_err(SessionError::Host)?;
    if !host.schema(image).map_err(SessionError::Host)?.is_pixel() {
        return Err(SessionError::Observe(
            "the output is not a picture".to_owned(),
        ));
    }
    let (width, height) = (domain.width(), domain.height() * domain.depth());
    if x >= width || y >= height {
        return Err(SessionError::Host(orla_runtime::HostError::OutOfBounds {
            x,
            y,
            width,
            height,
        }));
    }
    let words = host
        .element(image, y * width + x)
        .map_err(SessionError::Host)?;
    Ok([
        f32::from_bits(words[0]),
        f32::from_bits(words[1]),
        f32::from_bits(words[2]),
        f32::from_bits(words[3]),
    ])
}
