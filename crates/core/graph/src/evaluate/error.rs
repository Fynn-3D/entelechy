//! Why an evaluation stopped, as data an interface hands on: the node,
//! the port and the Kernel where it belongs to one.

use super::*;

/// Why evaluation stopped. Nodes evaluated before the failure keep their new
/// results; the failing node keeps its last successful result, if any; no
/// state is committed.
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
        /// The published id it was probably meant to be.
        near: Option<String>,
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
        /// The port or parameter it was probably meant to be; see
        /// [`orla_abi::names::nearest`].
        near: Option<String>,
    },
    /// A parameter override has the wrong type.
    ParamType {
        /// The node.
        node: NodeId,
        /// The parameter.
        name: String,
        /// Declared type, as the Kernel names it.
        expected: String,
        /// The value given.
        actual: String,
    },
    /// A source refers to an output port the producer does not declare.
    UnknownOutput {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
        /// The output it was probably meant to be.
        near: Option<String>,
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
    /// A reduction node was given a `domain`; a reduction produces one
    /// element and maps over its first input.
    ReduceDomain(NodeId),
    /// A scatter node wrote a `domain`; it takes the extent of its target.
    /// A node's `domain` expression could not be resolved.
    Domain {
        /// The node.
        node: NodeId,
        /// What is wrong.
        message: String,
    },
    /// A write or a reset named an output no `<~` binding reads, so it
    /// holds no state.
    NotState {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A write supplied the wrong number of words for the port's layout
    /// and extent.
    WordCount {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
        /// Words the port holds.
        expected: usize,
        /// Words supplied.
        found: usize,
    },
    /// A port is bound to a resource of the other kind, or to a Field with
    /// a different element layout.
    PortKind {
        /// The node.
        node: NodeId,
        /// The port.
        port: String,
        /// What is wrong.
        message: String,
    },
    /// A `<~` port's committed state no longer has the port's layout and
    /// the binding's migration Kernel could not carry it over.
    Migration {
        /// The node reading the state.
        node: NodeId,
        /// The port.
        port: String,
        /// The migration Kernel.
        kernel: KernelId,
        /// What went wrong.
        message: String,
    },
    /// A fused group could not be prepared by the backend.
    Fusion {
        /// The consuming node.
        node: NodeId,
        /// The failure.
        message: String,
    },
    /// The Host rejected a resource operation.
    Host(HostError),
}

impl EvaluateError {
    /// The node the failure belongs to, where it belongs to one, so a
    /// caller that has the graph can say where that node was written.
    pub fn node(&self) -> Option<&NodeId> {
        match self {
            Self::UnknownKernel { node, .. }
            | Self::MissingExternal { node, .. }
            | Self::UnboundInput { node, .. }
            | Self::UnknownPort { node, .. }
            | Self::ParamType { node, .. }
            | Self::UnknownOutput { node, .. }
            | Self::NoDomain(node)
            | Self::Execution { node, .. }
            | Self::Domain { node, .. }
            | Self::NotState { node, .. }
            | Self::WordCount { node, .. }
            | Self::Fusion { node, .. }
            | Self::ReduceDomain(node)
            | Self::PortKind { node, .. }
            | Self::Migration { node, .. } => Some(node),
            Self::Graph(_) | Self::Host(_) => None,
        }
    }
}

impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "{error}"),
            Self::UnknownKernel { node, kernel, near } => write!(
                f,
                "node `{node}` binds `{kernel}`, which is not published{}",
                suggest(near)
            ),
            Self::MissingExternal { node, name } => write!(
                f,
                "node `{node}` needs external image `{name}`, which was not supplied"
            ),
            Self::UnboundInput { node, port } => {
                write!(f, "node `{node}` does not bind input `{port}`")
            }
            Self::UnknownPort { node, port, near } => write!(
                f,
                "node `{node}` refers to `{port}`, which its kernel does not declare{}",
                suggest(near)
            ),
            Self::ParamType {
                node,
                name,
                expected,
                actual,
            } => write!(
                f,
                "node `{node}` sets `{name}` to {actual}, but the kernel expects {expected}"
            ),
            Self::UnknownOutput { node, port, near } => {
                write!(f, "node `{node}` has no output `{port}`{}", suggest(near))
            }
            Self::NoDomain(node) => write!(
                f,
                "node `{node}` has no image inputs; add `domain = vec2(width, height);`"
            ),
            Self::Execution { node, error } => write!(f, "node `{node}` failed: {error}"),
            Self::Fusion { node, message } => {
                write!(f, "fused group at `{node}` was rejected: {message}")
            }
            Self::Domain { node, message } => write!(f, "node `{node}`: {message}"),
            Self::NotState { node, port } => {
                write!(f, "`{node}.{port}` is not state: no `<~` binding reads it")
            }
            Self::WordCount {
                node,
                port,
                expected,
                found,
            } => write!(
                f,
                "`{node}.{port}` holds {expected} words; {found} were written"
            ),
            Self::ReduceDomain(node) => write!(
                f,
                "node `{node}` is a reduction, which produces one element; remove its `domain`"
            ),
            Self::PortKind {
                node,
                port,
                message,
            } => write!(f, "node `{node}` port `{port}`: {message}"),
            Self::Migration {
                node,
                port,
                kernel,
                message,
            } => write!(
                f,
                "node `{node}` port `{port}`: its state migration `{kernel}` {message}; the state stays as it was"
            ),
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
