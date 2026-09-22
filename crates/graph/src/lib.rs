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
//! Graphs are written in `.osn` documents, parsed by the language frontend
//! and lowered here by [`Graph::from_document`]. Inline Kernels are reported
//! back to the loader for publication; the Graph itself binds every node by
//! id and never holds Kernel bodies.

mod document;
mod domain;
mod evaluate;

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

pub use document::{Declare, DocumentError, InlineKernel, Loader, Lowered, anonymous_id};
pub use domain::{DomainError, DomainExpr, DomainOp, DomainSpec};
pub use evaluate::{EvaluateError, Evaluator, PlanEntry, Report};
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
    /// Default value; its type is the parameter type.
    pub default: Value,
}

/// A node: a Kernel binding plus its connections and parameter overrides.
#[derive(Clone, Debug, PartialEq)]
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
}

/// An external a document reads from a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asset {
    /// The input's name.
    pub name: String,
    /// The file, resolved against the document's directory.
    pub path: PathBuf,
    /// The type as written, which tells a picture file whether to come
    /// in as pixels or as a `Grid<Float>`.
    pub ty: String,
    /// Whether the file comes in as its bytes, one to an element of a
    /// `Rows<Int>`, rather than as a Field of its own format.
    pub bytes: bool,
}

/// A graph document.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Graph name.
    pub name: String,
    /// External Field names the caller must supply.
    pub externals: Vec<String>,
    /// Externals the document reads from files; the Host loads and
    /// watches them.
    pub assets: Vec<Asset>,
    /// Externals the document writes as text: the name and the text,
    /// which the session supplies as `Rows<Int>` of the characters' codes.
    pub texts: Vec<(String, String)>,
    /// Externals the document reads frame by frame from a directory: the
    /// name, the directory resolved against the document, the type as
    /// written, and the frames a second of the session's time; the
    /// session supplies the file whose turn it is at each evaluation.
    pub sequences: Vec<(String, PathBuf, String, u32)>,
    /// Parameters the caller may override.
    pub params: Vec<GraphParam>,
    /// Parameters that size the graph (a node's `times`): a change to one
    /// of these means lowering the document again.
    pub times_params: Vec<String>,
    /// Nodes in document order, sub-graphs flattened in place.
    pub nodes: Vec<Node>,
    /// Named results in document order; interfaces present the first.
    pub outputs: Vec<(String, Source)>,
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
    /// Parses a `.osn` document that references published Kernels only.
    /// Loaders that also publish inline Kernels or place sub-graphs use
    /// [`Graph::load`].
    pub fn parse(text: &str) -> Result<Self, DocumentError> {
        let document = orla_language::parse_document(text).map_err(DocumentError::from)?;
        let mut loader = |path: &std::path::Path| -> Result<String, String> {
            Err(format!(
                "sub-graph `{}` cannot be loaded here",
                path.display()
            ))
        };
        let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
        let lowered = document::lower(
            &document,
            text,
            "graph",
            std::path::Path::new("."),
            &mut loader,
            &mut none,
            &BTreeMap::new(),
        )?;
        if let Some(inline) = lowered.inline.first() {
            return Err(DocumentError {
                span: inline
                    .kernel
                    .id
                    .as_ref()
                    .map(|id| id.span)
                    .unwrap_or_default(),
                message: format!(
                    "node `{}` carries an inline kernel; load this document through a session",
                    inline.node
                ),
            });
        }
        Ok(lowered.graph)
    }

    /// Lowers a parsed document that places no sub-graphs into a Graph named
    /// `name`, reporting inline Kernels for the loader to publish. Anonymous
    /// inline Kernels are named by [`anonymous_id`].
    pub fn from_document(
        document: &orla_language::Document,
        name: &str,
    ) -> Result<Lowered, DocumentError> {
        let mut loader = |path: &std::path::Path| -> Result<String, String> {
            Err(format!(
                "sub-graph `{}` cannot be loaded here; use Graph::load with a loader",
                path.display()
            ))
        };
        let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
        document::lower(
            document,
            "",
            name,
            std::path::Path::new("."),
            &mut loader,
            &mut none,
            &BTreeMap::new(),
        )
    }

    /// Lowers a parsed document into a Graph, loading the sub-graphs it
    /// places through `loader` (paths are resolved against `base`) and
    /// flattening them in place: their nodes are prefixed with the placing
    /// node's name, their inputs rewired to the parent's bindings, their
    /// params set from the parent's values.
    pub fn load(
        document: &orla_language::Document,
        text: &str,
        name: &str,
        base: &std::path::Path,
        loader: &mut Loader<'_>,
    ) -> Result<Lowered, DocumentError> {
        let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
        document::lower(
            document,
            text,
            name,
            base,
            loader,
            &mut none,
            &BTreeMap::new(),
        )
    }

    /// Loads like [`Graph::load`] and also resolves every document's `use`
    /// items to shared struct and type declarations through `declare`,
    /// which every Kernel of that document then sees.
    pub fn load_with(
        document: &orla_language::Document,
        text: &str,
        name: &str,
        base: &std::path::Path,
        loader: &mut Loader<'_>,
        declare: &mut Declare<'_>,
    ) -> Result<Lowered, DocumentError> {
        document::lower(
            document,
            text,
            name,
            base,
            loader,
            declare,
            &BTreeMap::new(),
        )
    }

    /// As [`Graph::load_with`], with parameter overrides that a node's
    /// `times` may read, so the caller's values size the graph.
    pub fn load_with_params(
        document: &orla_language::Document,
        text: &str,
        name: &str,
        base: &std::path::Path,
        loader: &mut Loader<'_>,
        declare: &mut Declare<'_>,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<Lowered, DocumentError> {
        document::lower(document, text, name, base, loader, declare, overrides)
    }

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
                        return Err(GraphError::UnknownSource {
                            node: node.id.clone(),
                            port: port.clone(),
                            source: source.clone(),
                        });
                    }
                }
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
        let waits_for: Vec<Vec<usize>> = self
            .nodes
            .iter()
            .map(|node| {
                node.inputs
                    .values()
                    .filter_map(|binding| match binding {
                        Binding::Now(Source::Node { node, .. })
                        | Binding::Previous {
                            initial: Source::Node { node, .. },
                            ..
                        } => index_of(node),
                        _ => None,
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
                param.default.ty(),
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
pub(crate) fn literal_text(value: &Value) -> String {
    match value {
        Value::Float(v) => format!("{v:?}"),
        Value::Int(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2({x:?}, {y:?})"),
        Value::Vec3([x, y, z]) => format!("vec3({x:?}, {y:?}, {z:?})"),
        Value::Vec4([x, y, z, w]) => format!("vec4({x:?}, {y:?}, {z:?}, {w:?})"),
    }
}
