//! Observations that leave computation unchanged: a saved output, a
//! pixel, an element's field, an element trace, an exported picture.
//!
//! Each observation receives only the parts it reads (the Graph, the
//! evaluator's cache, the runtime), never the session, so it cannot
//! reach the ledger, the inputs or the scene's documents.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use orla_graph::{Evaluator, Graph, Node, NodeId};
use orla_kernel::{Context, Handle, KernelVersion, Runtime, Validated, Value};

use crate::SessionError;

/// A vector component by letter (`xyzw` or `rgba`): its word and type.
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

/// Parses `node.port[index].field.field` (replays and the protocol)
/// into node, port, element index and field path.
pub fn parse_target(text: &str) -> Result<(String, String, u32, Vec<String>), String> {
    let (head, rest) = text
        .split_once('[')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (index, tail) = rest
        .split_once(']')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (node, port) =
        orla_graph::split_port(head).ok_or_else(|| "name `node.port`".to_string())?;
    let index = index
        .parse::<u32>()
        .map_err(|_| format!("element index `{index}` is not a whole number"))?;
    let path: Vec<String> = tail
        .split('.')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect();
    Ok((node.0, port, index, path))
}

/// Why a node's output has no value: the name is a placed graph or
/// repeated node, no node, a node folded into its reader, or one that has
/// not run since it was placed.
pub(crate) fn why_no_value(
    graph: Option<&Graph>,
    evaluator: &Evaluator,
    node: &NodeId,
    port: &str,
) -> String {
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
    match evaluator.fused_into(node) {
        Some(reader) => format!(
            "node `{node}` runs fused into `{reader}` and keeps no `{port}` of its own; pin it and evaluate again"
        ),
        None => format!(
            "node `{node}` has no `{port}` yet: it has not run since it was placed; evaluate"
        ),
    }
}

/// Writes a node's output as a file; see [`crate::Session::save`].
pub(crate) fn save(
    store: &dyn crate::Store,
    graph: Option<&Graph>,
    evaluator: &Evaluator,
    runtime: &mut Runtime,
    node: &NodeId,
    port: &str,
    path: &Path,
) -> Result<(), String> {
    // A placed graph's output, or a scene output, stands for the node
    // that gives it.
    let (node, port) = graph
        .and_then(|graph| graph.producer(node, port))
        .unwrap_or_else(|| (node.clone(), port.to_owned()));
    let (node, port) = (&node, port.as_str());
    let Some(handle) = evaluator.cached(node, port) else {
        return Err(why_no_value(graph, evaluator, node, port));
    };
    write(store, runtime, handle, path).map_err(|why| match why {
        Written::Unfit(what) => format!("`{node}.{port}` is {what}; save it as .field"),
        other => other.to_string(),
    })
}

/// Why a Field was not written to a file.
pub(crate) enum Written {
    /// Reading the Field back from the device failed.
    Observe(String),
    /// The file's extension names no format.
    Unknown(orla_formats::EncodeError),
    /// The Field does not fit the format the extension names.
    Unfit(&'static str),
    /// The store refused it.
    Io(std::path::PathBuf, std::io::Error),
}

impl std::fmt::Display for Written {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observe(why) => f.write_str(why),
            Self::Unknown(why) => write!(f, "{why}"),
            Self::Unfit(what) => write!(f, "it is {what}"),
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
        }
    }
}

/// Materializes a Field and writes it to `path` in the format its
/// extension names, through `store`: the session's one Field-to-file
/// path, for `save` and `export`.
pub(crate) fn write(
    store: &dyn crate::Store,
    runtime: &mut Runtime,
    handle: Handle,
    path: &Path,
) -> Result<(), Written> {
    runtime
        .materialize(handle)
        .map_err(|e| Written::Observe(e.to_string()))?;
    let data = runtime
        .host()
        .data(handle)
        .map_err(|e| Written::Observe(e.to_string()))?;
    let bytes = orla_formats::encode(data, &orla_formats::extension(path), None).map_err(
        |why| match why {
            orla_formats::EncodeError::Unfit(what) => Written::Unfit(what),
            unknown => Written::Unknown(unknown),
        },
    )?;
    store
        .write(path, &bytes)
        .map_err(|e| Written::Io(path.to_path_buf(), e))
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

/// The node and Kernel version a trace runs; a trace requires a scene
/// node whose published Kernel has a module (not a native operation).
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

/// The nodes whose Fields a trace of `target` reads and so keeps.
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

/// Runs element `at` of `target` on the interpreter with the current
/// invocation. `written` is the owning document's unoptimized module;
/// without it the running module is traced.
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
    // Unoptimized, so the trace keeps the source's calls and `let`s.
    let module = written
        .as_ref()
        .unwrap_or_else(|| version.module().expect("a traced Kernel has a module"));
    // Work-accounted, as a backend runs it.
    let module = orla_ir::account(module).map_err(|error| error.to_string())?;
    // Only the required capabilities the session grants.
    let granted: orla_kernel::Capabilities = module
        .manifest
        .requires
        .iter()
        .filter(|capability| runtime.grants().contains(*capability))
        .collect();
    let mut scope = orla_kernel::Scope::new(runtime.host_mut(), granted);
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
    // Struct table: the module's, or a native operation's own.
    let structs: Vec<orla_ir::StructDef> = match (version.module(), version.native()) {
        (Some(module), _) => module.structs.clone(),
        (None, Some(op)) => op.structs(),
        (None, None) => Vec::new(),
    };
    // Follow the path to a word offset and type.
    let mut ty = output.element();
    let mut offset = 0usize;
    let words_of = |ty: Type| orla_ir::schema_of(ty, &structs).map_or(1, |s| s.value_words());
    for field in path {
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

/// One pixel of a cached image; see [`crate::Session::read_pixel`].
pub(crate) fn read_pixel(
    runtime: &mut Runtime,
    image: Handle,
    x: u32,
    y: u32,
) -> Result<orla_kernel::Pixel, SessionError> {
    // Reads four words of the materialized Field; the picture is not
    // copied.
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
        return Err(SessionError::Host(orla_kernel::HostError::OutOfBounds {
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
