#![warn(missing_docs)]
//! Graph Runtime (L6).
//!
//! A Graph is a program: nodes bound to stable [`KernelId`]s, typed
//! connections between their ports, and named outputs. The Graph owns
//! composition; each Kernel owns its computation. The [`Evaluator`] runs a
//! Graph against a `Runtime`, caches node outputs, re-executes a node
//! only when its inputs changed, and commits state between evaluations.
//! The Graph binds nodes by id; it never sees source or Kernel bodies.

mod domain;
mod evaluate;

pub use domain::{DomainError, DomainExpr, DomainOp, DomainSpec};
pub use evaluate::{EvaluateError, Evaluator, Placed, PlanEntry, Report};
pub use orla_abi::{Context, Domain, InputState, Value};
pub use orla_ir::KernelId;
use std::collections::BTreeMap;
use std::fmt;

/// Name of a node within a graph.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Splits `node.port` at the last dot (placed node ids contain dots:
/// `fx.gray.out`); `None` when the text names no port.
pub fn split_port(text: &str) -> Option<(NodeId, String)> {
    let (node, port) = text.rsplit_once('.')?;
    if node.is_empty() || port.is_empty() {
        return None;
    }
    Some((NodeId(node.to_owned()), port.to_owned()))
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
    /// The source's value committed by the previous successful evaluation;
    /// `initial` feeds the port before the first commit and after a state
    /// reset. State lives in the Graph; Kernels stay pure.
    Previous {
        /// A node output; read from the last commit.
        source: Source,
        /// What to read before the first commit.
        initial: Source,
        /// The Kernel that maps committed state into the port's layout
        /// when a republished Kernel changed it; without one, state
        /// restarts from `initial` when the layouts disagree.
        migration: Option<KernelId>,
    },
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Now(source) => write!(f, "<- {source}"),
            Self::Previous {
                source,
                initial,
                migration,
            } => {
                write!(f, "<~ {source} from {initial}")?;
                if let Some(kernel) = migration {
                    write!(f, " migrate {kernel}")?;
                }
                Ok(())
            }
        }
    }
}

/// A parameter value on a node: a literal, or a graph parameter by name.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// A fixed value.
    Literal(Value),
    /// The graph parameter with this name, resolved at evaluation from the
    /// caller's overrides or the graph's defaults.
    Graph(String),
}

/// A graph parameter: part of the graph's interface, like a Kernel's.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphParam {
    /// Parameter name.
    pub name: String,
    /// Its type as written; a record is read back by it.
    pub ty: String,
    /// Default value, of that type; a value given for it takes its shape.
    pub default: Value,
}

/// A node: a Kernel binding plus its connections and parameter overrides.
#[derive(Clone, Debug)]
pub struct Node {
    /// Node name; a placed sub-graph's nodes carry a dotted prefix.
    pub id: NodeId,
    /// The Kernel this node executes.
    pub kernel: KernelId,
    /// Input port bindings.
    pub inputs: BTreeMap<String, Binding>,
    /// Parameter overrides; unlisted parameters use manifest defaults.
    pub params: BTreeMap<String, ParamValue>,
    /// Output extent; defaults to the first input's extent.
    pub domain: Option<DomainSpec>,
    /// `async = true;`: the node runs on a thread off the frame. Until a
    /// run lands, readers take its last output and the state it feeds
    /// stays. Its first run is on the frame.
    pub is_async: bool,
    /// Where the node was written, `file:line:column`, for diagnostics.
    pub at: Option<String>,
}

/// Equality ignores `at`: where a node was written is provenance, not
/// identity.
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.kernel == other.kernel
            && self.inputs == other.inputs
            && self.params == other.params
            && self.domain == other.domain
            && self.is_async == other.is_async
    }
}

/// A graph document.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Graph name.
    pub name: String,
    /// External Field names the caller must supply.
    pub externals: Vec<String>,
    /// Text externals: name and text, supplied as `Rows<Int>` of the
    /// characters' codes.
    pub texts: Vec<(String, String)>,
    /// Parameters the caller may override.
    pub params: Vec<GraphParam>,
    /// Parameters that size the graph (a node's `times`); changing one
    /// lowers the document again.
    pub times_params: Vec<String>,
    /// Nodes in document order, sub-graphs flattened in place.
    pub nodes: Vec<Node>,
    /// Named results in document order; interfaces present the first.
    pub outputs: Vec<(String, Source)>,
    /// Fallbacks: an external's name and the source it reads while the
    /// caller supplies nothing (its file absent or unreadable).
    pub fallbacks: Vec<(String, Source)>,
    /// Each placed graph's outputs, by the id it was placed under, each
    /// wired to the node that gives it.
    pub placed: Vec<(String, Vec<(String, Source)>)>,
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
        /// The node it was probably meant to be.
        near: Option<String>,
    },
    /// Nodes depend on each other in a cycle.
    Cycle(Vec<NodeId>),
    /// A graph output refers to an unknown source.
    UnknownOutput(Source),
    /// The graph declares no output.
    NoOutput,
    /// A `<~` binding reads an external; only node outputs carry state.
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
            Self::UnknownSource {
                node,
                port,
                source,
                near,
            } => {
                let hint = orla_abi::names::suggestion(near.as_deref());
                write!(
                    f,
                    "node `{node}` port `{port}` refers to unknown source `{source}`{hint}"
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
            Self::UnknownOutput(source) => write!(f, "output refers to unknown source `{source}`"),
            Self::NoOutput => write!(f, "the graph declares no output"),
            Self::FeedbackNeedsNode { node, port } => write!(
                f,
                "node `{node}` port `{port}` reads previous state from an external image; use a node output"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

impl Graph {
    /// The node and port that produce what `node.port` names: a node's
    /// own port; a placed graph's output (`kept.count`), by the node inside
    /// it that gives it; or the graph's own output named `node`, as
    /// `node.result`. `None` when the name is none of these.
    pub fn producer(&self, node: &NodeId, port: &str) -> Option<(NodeId, String)> {
        if self.node(node).is_some() {
            return Some((node.clone(), port.to_owned()));
        }
        let wired = |source: &Source| match source {
            Source::Node { node, port } => Some((node.clone(), port.clone())),
            Source::External(_) => None,
        };
        if let Some((_, outputs)) = self.placed.iter().find(|(id, _)| *id == node.0) {
            let (_, source) = outputs.iter().find(|(name, _)| name == port)?;
            return wired(source);
        }
        if port == "result" {
            let (_, source) = self.outputs.iter().find(|(name, _)| *name == node.0)?;
            return wired(source);
        }
        None
    }

    /// The first declared output.
    pub fn primary_output(&self) -> Option<&(String, Source)> {
        self.outputs.first()
    }

    /// Looks up a node.
    pub fn node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| &node.id == id)
    }

    /// Node index by name; an evaluation builds it once for lookups.
    pub fn index(&self) -> BTreeMap<&NodeId, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(at, node)| (&node.id, at))
            .collect()
    }

    /// What an external reads while the caller supplies nothing for it.
    pub fn fallback(&self, external: &str) -> Option<&Source> {
        self.fallbacks
            .iter()
            .find(|(name, _)| name == external)
            .map(|(_, source)| source)
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
        let index = self.index();
        let index_of = |id: &NodeId| index.get(id).copied();
        for node in &self.nodes {
            for (port, binding) in &node.inputs {
                let sources: Vec<&Source> = match binding {
                    Binding::Now(source) => vec![source],
                    Binding::Previous {
                        source, initial, ..
                    } => {
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
                        // Every name, for the closest-match suggestion.
                        let near = match source {
                            Source::Node { node, .. } => orla_abi::names::nearest(
                                &node.0,
                                self.nodes.iter().map(|n| n.id.0.as_str()),
                            ),
                            Source::External(name) => orla_abi::names::nearest(
                                name,
                                self.externals.iter().map(String::as_str),
                            ),
                        }
                        .map(str::to_owned);
                        return Err(GraphError::UnknownSource {
                            node: node.id.clone(),
                            port: port.clone(),
                            source: source.clone(),
                            near,
                        });
                    }
                }
            }
        }
        for (name, source) in &self.fallbacks {
            let known = self.externals.contains(name)
                && match source {
                    Source::External(other) => self.externals.contains(other),
                    Source::Node { node, .. } => index_of(node).is_some(),
                };
            if !known {
                return Err(GraphError::UnknownOutput(source.clone()));
            }
        }
        if self.outputs.is_empty() {
            return Err(GraphError::NoOutput);
        }
        for (_, source) in &self.outputs {
            let known = match source {
                Source::External(name) => self.externals.contains(name),
                Source::Node { node, .. } => index_of(node).is_some(),
            };
            if !known {
                return Err(GraphError::UnknownOutput(source.clone()));
            }
        }

        // Depth-first topological sort with cycle detection.
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            None,
            Active,
            Done,
        }
        // What each node waits for, computed once. An external with a
        // fallback waits for the node the fallback reads, supplied or not.
        let producer = |source: &Source| match source {
            Source::Node { node, .. } => index_of(node),
            Source::External(name) => match self.fallback(name) {
                Some(Source::Node { node, .. }) => index_of(node),
                _ => None,
            },
        };
        let waits_for: Vec<Vec<usize>> = self
            .nodes
            .iter()
            .map(|node| {
                node.inputs
                    .values()
                    .filter_map(|binding| match binding {
                        Binding::Now(source)
                        | Binding::Previous {
                            initial: source, ..
                        } => producer(source),
                    })
                    .collect()
            })
            .collect();
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
                let dependencies = &waits_for[current];
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
