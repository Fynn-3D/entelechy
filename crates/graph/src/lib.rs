#![warn(missing_docs)]
//! Graph Runtime (L6).
//!
//! A Graph is a program: nodes bound to stable [`KernelId`]s, typed
//! connections between their ports, and one designated output. The Graph
//! owns composition; each Kernel owns its computation. The [`Evaluator`]
//! executes a Graph against a [`Runtime`], caches every node's outputs, and
//! re-executes a node only when its Kernel generation, parameters, domain,
//! or upstream results changed.
//!
//! Graphs are written in the small declarative `.osn` format:
//!
//! ```text
//! graph demo
//! input photo
//! node gray = image.grayscale@1
//!   source <- photo
//!   amount = 1.0
//! node blocks = image.pixelate@1
//!   source <- gray.result
//!   size = 8
//! output blocks.result
//! ```

mod document;
mod evaluate;

use std::collections::BTreeMap;
use std::fmt;

pub use document::DocumentError;
pub use evaluate::{EvaluateError, Evaluator, Report};
pub use orla_abi::{Context, Domain, InputState, Value};
pub use orla_ir::KernelId;

/// Name of a node within a graph.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where an input port receives its image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// An image supplied by the caller under this name.
    External(String),
    /// An output port of another node.
    Node {
        /// The producing node.
        node: NodeId,
        /// The output port name.
        port: String,
    },
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::External(name) => f.write_str(name),
            Self::Node { node, port } => write!(f, "{node}.{port}"),
        }
    }
}

/// How an input port is fed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// The source's value from this evaluation.
    Now(Source),
    /// The source's value as committed by the previous successful
    /// evaluation; `initial` feeds the port until a value has been
    /// committed, and again after a state reset. This is how state lives in
    /// the Graph while Kernels stay pure.
    Previous {
        /// A node output; read from the last commit.
        source: Source,
        /// What to read before the first commit.
        initial: Source,
    },
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Now(source) => write!(f, "<- {source}"),
            Self::Previous { source, initial } => write!(f, "<~ {source} from {initial}"),
        }
    }
}

/// A node: a Kernel binding plus its connections and parameter overrides.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Node name.
    pub id: NodeId,
    /// The Kernel this node executes.
    pub kernel: KernelId,
    /// Input port bindings.
    pub inputs: BTreeMap<String, Binding>,
    /// Parameter overrides; unlisted parameters use manifest defaults.
    pub params: BTreeMap<String, Value>,
    /// Explicit output domain; defaults to the first image input's size.
    pub domain: Option<Domain>,
}

/// A graph document.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Graph name.
    pub name: String,
    /// External image names the caller must supply.
    pub externals: Vec<String>,
    /// Nodes in document order.
    pub nodes: Vec<Node>,
    /// The node and port whose image is the graph result.
    pub output: (NodeId, String),
}

/// Structural problems with a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// Two nodes or externals share a name.
    DuplicateName(String),
    /// A port or the output refers to a name that is neither a node nor an external.
    UnknownSource {
        /// The node whose port is bound, or the graph output.
        node: NodeId,
        /// The port.
        port: String,
        /// The missing name.
        source: Source,
    },
    /// Nodes depend on each other in a cycle.
    Cycle(Vec<NodeId>),
    /// The output refers to an unknown node.
    UnknownOutput(NodeId),
    /// A `<~` binding reads an external image; only node outputs carry
    /// state between evaluations.
    FeedbackNeedsNode {
        /// The node whose port is bound.
        node: NodeId,
        /// The port.
        port: String,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "`{name}` is declared twice"),
            Self::UnknownSource { node, port, source } => {
                write!(
                    f,
                    "node `{node}` port `{port}` refers to unknown source `{source}`"
                )
            }
            Self::Cycle(nodes) => {
                let names = nodes
                    .iter()
                    .map(|node| node.0.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ");
                write!(f, "nodes form a cycle: {names}")
            }
            Self::UnknownOutput(node) => write!(f, "output refers to unknown node `{node}`"),
            Self::FeedbackNeedsNode { node, port } => write!(
                f,
                "node `{node}` port `{port}` reads previous state from an external image; use a node output"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

impl Graph {
    /// Parses a `.osn` document.
    pub fn parse(text: &str) -> Result<Self, DocumentError> {
        document::parse(text)
    }

    /// Looks up a node.
    pub fn node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| &node.id == id)
    }

    /// Checks names and dependencies and returns node indices in an order
    /// where every producer precedes its consumers.
    pub fn order(&self) -> Result<Vec<usize>, GraphError> {
        let mut names = std::collections::BTreeSet::new();
        for name in self
            .externals
            .iter()
            .chain(self.nodes.iter().map(|node| &node.id.0))
        {
            if !names.insert(name) {
                return Err(GraphError::DuplicateName(name.clone()));
            }
        }
        let index_of = |id: &NodeId| self.nodes.iter().position(|node| &node.id == id);
        for node in &self.nodes {
            for (port, binding) in &node.inputs {
                let sources: Vec<&Source> = match binding {
                    Binding::Now(source) => vec![source],
                    Binding::Previous { source, initial } => {
                        if let Source::External(_) = source {
                            return Err(GraphError::FeedbackNeedsNode {
                                node: node.id.clone(),
                                port: port.clone(),
                            });
                        }
                        vec![source, initial]
                    }
                };
                for source in sources {
                    let known = match source {
                        Source::External(name) => self.externals.contains(name),
                        Source::Node { node, .. } => index_of(node).is_some(),
                    };
                    if !known {
                        return Err(GraphError::UnknownSource {
                            node: node.id.clone(),
                            port: port.clone(),
                            source: source.clone(),
                        });
                    }
                }
            }
        }
        index_of(&self.output.0).ok_or_else(|| GraphError::UnknownOutput(self.output.0.clone()))?;

        // Depth-first topological sort with cycle detection.
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            None,
            Active,
            Done,
        }
        let mut marks = vec![Mark::None; self.nodes.len()];
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for root in 0..self.nodes.len() {
            if marks[root] != Mark::None {
                continue;
            }
            marks[root] = Mark::Active;
            stack.push((root, 0));
            while let Some((current, next_input)) = stack.last().copied() {
                // `Previous` sources read committed state and add no
                // dependency; their initial sources do.
                let dependencies: Vec<usize> = self.nodes[current]
                    .inputs
                    .values()
                    .filter_map(|binding| match binding {
                        Binding::Now(Source::Node { node, .. })
                        | Binding::Previous {
                            initial: Source::Node { node, .. },
                            ..
                        } => index_of(node),
                        _ => None,
                    })
                    .collect();
                if next_input < dependencies.len() {
                    stack.last_mut().expect("stack is non-empty").1 += 1;
                    let dependency = dependencies[next_input];
                    match marks[dependency] {
                        Mark::None => {
                            marks[dependency] = Mark::Active;
                            stack.push((dependency, 0));
                        }
                        Mark::Active => {
                            let start = stack
                                .iter()
                                .position(|(index, _)| *index == dependency)
                                .unwrap_or(0);
                            let cycle = stack[start..]
                                .iter()
                                .map(|(index, _)| self.nodes[*index].id.clone())
                                .chain(std::iter::once(self.nodes[dependency].id.clone()))
                                .collect();
                            return Err(GraphError::Cycle(cycle));
                        }
                        Mark::Done => {}
                    }
                } else {
                    marks[current] = Mark::Done;
                    order.push(current);
                    stack.pop();
                }
            }
        }
        Ok(order)
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph {}", self.name)?;
        for external in &self.externals {
            writeln!(f, "input {external}")?;
        }
        for node in &self.nodes {
            writeln!(f, "node {} = {}", node.id, node.kernel)?;
            for (port, binding) in &node.inputs {
                writeln!(f, "  {port} {binding}")?;
            }
            for (name, value) in &node.params {
                write!(f, "  {name} =")?;
                match value {
                    Value::Float(v) => writeln!(f, " {v:?}")?,
                    Value::Int(v) => writeln!(f, " {v}")?,
                    Value::Bool(v) => writeln!(f, " {v}")?,
                    Value::Vec2([x, y]) => writeln!(f, " {x:?} {y:?}")?,
                    Value::Vec3([x, y, z]) => writeln!(f, " {x:?} {y:?} {z:?}")?,
                    Value::Vec4([x, y, z, w]) => writeln!(f, " {x:?} {y:?} {z:?} {w:?}")?,
                }
            }
            if let Some(domain) = node.domain {
                writeln!(f, "  domain {} {}", domain.width, domain.height)?;
            }
        }
        writeln!(f, "output {}.{}", self.output.0, self.output.1)
    }
}
