//! Cached, dirty-tracked evaluation of a graph against a runtime.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use orla_abi::{Capability, Domain, ImageHandle, Invocation, Type, Value};
use orla_runtime::{ExecutionError, Generation, HostError, KernelId, KernelVersion, Runtime};

use crate::{Graph, GraphError, Node, NodeId, Source};

/// Why evaluation stopped. Nodes evaluated before the failure keep their new
/// results; the failing node keeps its last successful result, if any.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluateError {
    /// The graph is malformed.
    Graph(GraphError),
    /// A node binds a Kernel that is not published.
    UnknownKernel {
        /// The node.
        node: NodeId,
        /// The missing Kernel.
        kernel: KernelId,
    },
    /// The caller did not supply a declared external image.
    MissingExternal {
        /// The node that needs it.
        node: NodeId,
        /// The external name.
        name: String,
    },
    /// A manifest input has no binding.
    UnboundInput {
        /// The node.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A binding or parameter names something the manifest does not have.
    UnknownPort {
        /// The node.
        node: NodeId,
        /// The port or parameter.
        port: String,
    },
    /// A parameter override has the wrong type.
    ParamType {
        /// The node.
        node: NodeId,
        /// The parameter.
        name: String,
        /// Declared type.
        expected: Type,
        /// Supplied type.
        actual: Type,
    },
    /// A source refers to an output port the producer does not declare.
    UnknownOutput {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A node has no image inputs and no explicit domain.
    NoDomain(NodeId),
    /// The Kernel failed while executing.
    Execution {
        /// The node.
        node: NodeId,
        /// The failure.
        error: ExecutionError,
    },
    /// The Host rejected a resource operation.
    Host(HostError),
}

impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "{error}"),
            Self::UnknownKernel { node, kernel } => {
                write!(f, "node `{node}` binds `{kernel}`, which is not published")
            }
            Self::MissingExternal { node, name } => {
                write!(
                    f,
                    "node `{node}` needs external image `{name}`, which was not supplied"
                )
            }
            Self::UnboundInput { node, port } => {
                write!(f, "node `{node}` does not bind input `{port}`")
            }
            Self::UnknownPort { node, port } => {
                write!(
                    f,
                    "node `{node}` refers to `{port}`, which its kernel does not declare"
                )
            }
            Self::ParamType {
                node,
                name,
                expected,
                actual,
            } => write!(
                f,
                "node `{node}` sets `{name}` to {actual}, but the kernel expects {expected}"
            ),
            Self::UnknownOutput { node, port } => {
                write!(f, "node `{node}` has no output `{port}`")
            }
            Self::NoDomain(node) => write!(
                f,
                "node `{node}` has no image inputs; add `domain <width> <height>`"
            ),
            Self::Execution { node, error } => write!(f, "node `{node}` failed: {error}"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EvaluateError {}

impl From<GraphError> for EvaluateError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<HostError> for EvaluateError {
    fn from(error: HostError) -> Self {
        Self::Host(error)
    }
}

/// What one evaluation did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// The graph output image, owned by the evaluator's cache.
    pub output: ImageHandle,
    /// Nodes that executed, in evaluation order.
    pub executed: Vec<NodeId>,
    /// Nodes whose cached outputs were reused.
    pub reused: Vec<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
enum InputKey {
    External(ImageHandle),
    Node(NodeId, usize, u64),
}

#[derive(Clone, Debug, PartialEq)]
struct CacheKey {
    generation: Generation,
    params: Vec<Value>,
    inputs: Vec<InputKey>,
    domain: Domain,
    /// Present only for Kernels that read the clock, so pure nodes stay cached
    /// across frames.
    time: Option<u32>,
}

#[derive(Debug)]
struct Cached {
    key: CacheKey,
    outputs: Vec<ImageHandle>,
    output_names: Vec<String>,
    epoch: u64,
}

/// Evaluates graphs and owns the cached node outputs.
#[derive(Debug, Default)]
pub struct Evaluator {
    cache: BTreeMap<NodeId, Cached>,
    epoch: u64,
}

impl Evaluator {
    /// An evaluator with an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Releases every cached image.
    pub fn clear(&mut self, runtime: &mut Runtime) {
        for (_, cached) in std::mem::take(&mut self.cache) {
            release(runtime, &cached.outputs);
        }
    }

    /// The cached image for a node's output port, if it was evaluated.
    pub fn cached(&self, node: &NodeId, port: &str) -> Option<ImageHandle> {
        let cached = self.cache.get(node)?;
        let index = cached.output_names.iter().position(|name| name == port)?;
        Some(cached.outputs[index])
    }

    /// Evaluates `graph` at session time `time`, executing only nodes whose
    /// inputs changed. Nodes that declare `time.read` re-execute whenever
    /// `time` changes.
    pub fn evaluate(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, ImageHandle>,
        time: f32,
    ) -> Result<Report, EvaluateError> {
        let order = graph.order()?;
        // Nodes that left the graph release their images.
        let stale: Vec<NodeId> = self
            .cache
            .keys()
            .filter(|id| graph.node(id).is_none())
            .cloned()
            .collect();
        for id in stale {
            if let Some(cached) = self.cache.remove(&id) {
                release(runtime, &cached.outputs);
            }
        }
        let mut executed = Vec::new();
        let mut reused = Vec::new();
        for index in order {
            let node = &graph.nodes[index];
            let version =
                runtime
                    .kernel(&node.kernel)
                    .ok_or_else(|| EvaluateError::UnknownKernel {
                        node: node.id.clone(),
                        kernel: node.kernel.clone(),
                    })?;
            let (invocation, key) = self.bind(node, &version, runtime, externals, time)?;
            if self
                .cache
                .get(&node.id)
                .is_some_and(|cached| cached.key == key)
            {
                reused.push(node.id.clone());
                continue;
            }
            let outputs = runtime.execute(&version, &invocation).map_err(|error| {
                EvaluateError::Execution {
                    node: node.id.clone(),
                    error,
                }
            })?;
            self.epoch += 1;
            let fresh = Cached {
                key,
                outputs,
                output_names: version
                    .manifest()
                    .outputs
                    .iter()
                    .map(|port| port.name.clone())
                    .collect(),
                epoch: self.epoch,
            };
            if let Some(previous) = self.cache.insert(node.id.clone(), fresh) {
                release(runtime, &previous.outputs);
            }
            executed.push(node.id.clone());
        }
        let (node, port) = &graph.output;
        let output = self
            .cached(node, port)
            .ok_or_else(|| EvaluateError::UnknownOutput {
                node: node.clone(),
                port: port.clone(),
            })?;
        Ok(Report {
            output,
            executed,
            reused,
        })
    }

    fn bind(
        &self,
        node: &Node,
        version: &Arc<KernelVersion>,
        runtime: &Runtime,
        externals: &BTreeMap<String, ImageHandle>,
        time: f32,
    ) -> Result<(Invocation, CacheKey), EvaluateError> {
        let manifest = version.manifest();
        for port in node.inputs.keys().chain(node.params.keys()) {
            let declared = manifest.inputs.iter().any(|input| &input.name == port)
                || manifest.params.iter().any(|param| &param.name == port);
            if !declared {
                return Err(EvaluateError::UnknownPort {
                    node: node.id.clone(),
                    port: port.clone(),
                });
            }
        }
        let mut inputs = Vec::with_capacity(manifest.inputs.len());
        let mut input_keys = Vec::with_capacity(manifest.inputs.len());
        for input in &manifest.inputs {
            let source =
                node.inputs
                    .get(&input.name)
                    .ok_or_else(|| EvaluateError::UnboundInput {
                        node: node.id.clone(),
                        port: input.name.clone(),
                    })?;
            let (handle, key) = match source {
                Source::External(name) => {
                    let handle =
                        *externals
                            .get(name)
                            .ok_or_else(|| EvaluateError::MissingExternal {
                                node: node.id.clone(),
                                name: name.clone(),
                            })?;
                    (handle, InputKey::External(handle))
                }
                Source::Node {
                    node: producer,
                    port,
                } => {
                    let cached = self.cache.get(producer).ok_or_else(|| {
                        // The producer precedes this node in topological
                        // order, so a missing entry means it never
                        // produced this port.
                        EvaluateError::UnknownOutput {
                            node: producer.clone(),
                            port: port.clone(),
                        }
                    })?;
                    let index = cached
                        .output_names
                        .iter()
                        .position(|name| name == port)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: producer.clone(),
                            port: port.clone(),
                        })?;
                    (
                        cached.outputs[index],
                        InputKey::Node(producer.clone(), index, cached.epoch),
                    )
                }
            };
            inputs.push(handle);
            input_keys.push(key);
        }
        let mut params = Vec::with_capacity(manifest.params.len());
        for param in &manifest.params {
            let value = match node.params.get(&param.name) {
                None => param.default,
                Some(Value::Int(v)) if param.default.ty() == Type::Float => Value::Float(*v as f32),
                Some(value) if value.ty() == param.default.ty() => *value,
                Some(value) => {
                    return Err(EvaluateError::ParamType {
                        node: node.id.clone(),
                        name: param.name.clone(),
                        expected: param.default.ty(),
                        actual: value.ty(),
                    });
                }
            };
            params.push(value);
        }
        let domain = match (node.domain, inputs.first()) {
            (Some(domain), _) => domain,
            (None, Some(first)) => {
                let (width, height) = runtime.host().image_size(*first)?;
                Domain { width, height }
            }
            (None, None) => return Err(EvaluateError::NoDomain(node.id.clone())),
        };
        let reads_clock = manifest.requires.contains(Capability::TimeRead);
        let key = CacheKey {
            generation: version.generation(),
            params: params.clone(),
            inputs: input_keys,
            domain,
            time: reads_clock.then_some(time.to_bits()),
        };
        Ok((
            Invocation {
                inputs,
                params,
                domain,
                time,
            },
            key,
        ))
    }
}

fn release(runtime: &mut Runtime, outputs: &[ImageHandle]) {
    for output in outputs {
        // Cached handles were created by executions this evaluator owns; a
        // failure here means the interface released them behind our back,
        // which is the only case where ignoring the error is correct.
        let _ = runtime.host_mut().release_image(*output);
    }
}
