#![warn(missing_docs)]
//! Graph Runtime (L6).
//!
//! A Graph is a program: nodes bound to stable [`KernelId`]s, typed
//! connections between their ports, and named outputs. The Graph owns
//! composition; each Kernel owns its computation. The [`Evaluator`]
//! executes a Graph against a [`Runtime`], caches every node's outputs,
//! re-executes a node only when its Kernel generation, parameters, domain,
//! context, or upstream results changed, and commits state between
//! evaluations.
//!
//! Graphs are written in `.osn` documents, which `orla-scene` parses and
//! lowers into this model; the Graph Runtime never sees source. Inline
//! Kernels are reported back to the loader for publication; the Graph
//! itself binds every node by id and never holds Kernel bodies.

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
        /// The Kernel that carries the committed state into the port's
        /// layout when a republished Kernel changed it: a map over the
        /// old elements to the new. Without one, the state restarts from
        /// `initial` when the layouts disagree.
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
    /// Its type as the document writes it, which a record needs to be
    /// read back as what it is.
    pub ty: String,
    /// Default value, of that type; a value given for it takes its shape.
    pub default: Value,
}

/// A node: a Kernel binding plus its connections and parameter overrides.
#[derive(Clone, Debug)]
pub struct Node {
    /// Node name; nodes of a placed sub-graph carry its node name as a
    /// dotted prefix.
    pub id: NodeId,
    /// The Kernel this node executes.
    pub kernel: KernelId,
    /// Input port bindings.
    pub inputs: BTreeMap<String, Binding>,
    /// Parameter overrides; unlisted parameters use manifest defaults.
    pub params: BTreeMap<String, ParamValue>,
    /// Explicit output extent, literal or computed from the node's inputs
    /// and parameters; defaults to the first input's extent.
    pub domain: Option<DomainSpec>,
    /// Written `async = true;`: the node runs off the frame. An
    /// evaluation starts its execution on a thread and goes on; until
    /// the run lands, the node's readers take its last output, and the
    /// state it feeds stays as it was. Its first run is on the frame, so
    /// its readers have something to read.
    pub is_async: bool,
    /// Where the node was written, as `file:line:column`, when the
    /// graph came from a document. A Kernel's diagnostics have always
    /// said where; a scene's said only which node, and a scene of a
    /// placed sub-graph has nodes whose names nobody wrote.
    pub at: Option<String>,
}

/// Where a node was written is provenance and not identity: the same
/// graph written twice, or printed and read back, is the same graph.
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
    /// Externals the document writes as text: the name and the text,
    /// which the session supplies as `Rows<Int>` of the characters' codes.
    pub texts: Vec<(String, String)>,
    /// Parameters the caller may override.
    pub params: Vec<GraphParam>,
    /// Parameters that size the graph (a node's `times`): a change to one
    /// of these means lowering the document again.
    pub times_params: Vec<String>,
    /// Nodes in document order, sub-graphs flattened in place.
    pub nodes: Vec<Node>,
    /// Named results in document order; interfaces present the first.
    pub outputs: Vec<(String, Source)>,
    /// What an external read from a file reads while the caller supplies
    /// nothing for it (the file absent or unreadable): the external's
    /// name and the source.
    pub fallbacks: Vec<(String, Source)>,
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
            Self::UnknownSource {
                node,
                port,
                source,
                near,
            } => {
                let hint = match near {
                    Some(name) => format!("; did you mean `{name}`?"),
                    None => String::new(),
                };
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
    /// The first declared output.
    pub fn primary_output(&self) -> Option<&(String, Source)> {
        self.outputs.first()
    }

    /// Looks up a node.
    pub fn node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| &node.id == id)
    }

    /// Where every node is, by name. Finding one by name is a walk of
    /// the whole list, so anything that does it once a node is a walk
    /// of the list squared: an evaluation builds this once and looks
    /// nodes up in it. A graph of a thousand nodes spent most of a
    /// frame on those walks.
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
                        // Every node and external there is, so a name
                        // one letter out is named.
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
        // What each node waits for, worked out once. It used to be
        // gathered again on every step of the walk below, which for a
        // graph of any size is most of what the sort cost.
        // An external with a fallback waits for the node the fallback
        // reads, whether or not the caller supplies it this time.
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

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// {}", self.name)?;
        for external in &self.externals {
            match self.texts.iter().find(|(name, _)| name == external) {
                Some((_, text)) => writeln!(f, "input {external}: Rows<Int> = text \"{text}\";")?,
                None => writeln!(f, "input {external}: Image;")?,
            }
        }
        for param in &self.params {
            writeln!(
                f,
                "param {}: {} = {};",
                param.name,
                param.ty,
                literal_text(&param.default)
            )?;
        }
        for node in &self.nodes {
            writeln!(f)?;
            writeln!(f, "node {}: {} {{", node.id, node.kernel)?;
            for (port, binding) in &node.inputs {
                writeln!(f, "    {port} {binding};")?;
            }
            for (name, value) in &node.params {
                match value {
                    ParamValue::Literal(value) => {
                        writeln!(f, "    {name} = {};", literal_text(value))?
                    }
                    ParamValue::Graph(param) => writeln!(f, "    {name} = {param};")?,
                }
            }
            if let Some(domain) = &node.domain {
                writeln!(f, "    domain = {domain};")?;
            }
            if node.is_async {
                writeln!(f, "    async = true;")?;
            }
            writeln!(f, "}}")?;
        }
        writeln!(f)?;
        for (name, source) in &self.outputs {
            writeln!(f, "output {name} <- {source};")?;
        }
        Ok(())
    }
}

/// Renders a value in the literal grammar shared with `.orla`.
pub fn literal_text(value: &Value) -> String {
    // A number a scene wrote is exact and has no width until a Kernel's
    // param gives it one, so it reads back as written, without `d`.
    match value {
        Value::Double(v) => format!("{v:?}"),
        Value::Long(v) => v.to_string(),
        Value::Record(members) => format!(
            "[{}]",
            members
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}
