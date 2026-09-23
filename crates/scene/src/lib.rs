#![warn(missing_docs)]
//! Scene documents: a parsed `.osn` lowered into the Graph model, with
//! sub-graphs flattened in place.
//!
//! This is the frontend half of composition. The Graph Runtime
//! (`orla-graph`) evaluates a [`Graph`] and never sees source; this crate
//! reads what was written — nodes, wiring, domains, `times`, placed
//! sub-graphs, inline Kernels — checks it where it was written, and
//! hands the loader the inline Kernels to publish. Where the files come
//! from is the caller's: every function takes a [`Loader`] or refuses.

mod domain;

pub use domain::domain_expr;

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use orla_abi::Value;
use orla_ir::KernelId;
use orla_language::NodeDecl;
use orla_language::{
    BindingDecl, Document, Expr, ExprKind, Kernel, NodeKind, NodeMember, ParseError, SourceRef,
    Span, Spanned,
};

use orla_graph::{
    Asset, Binding, DomainSpec, Graph, GraphParam, Node, NodeId, ParamValue, Source, literal_text,
};

/// The port type an alias stands for, following aliases as far as they
/// go; a name that is not one is itself.
fn resolved(ty: &str, aliases: &[orla_language::TypeAlias]) -> String {
    let mut ty = ty.to_owned();
    for _ in 0..8 {
        match aliases.iter().find(|alias| alias.name.value == ty) {
            Some(alias) => ty = alias.target.value.clone(),
            None => break,
        }
    }
    ty
}

/// The value a Field of this port type is filled with when the node names
/// none: the zero of its element. A struct or an integer vector is not a
/// parameter's type, so a Field of one is a Kernel's work, not a fill.
fn zero_of(ty: &str) -> Option<Value> {
    if ty == "Image" {
        return Some(Value::Vec4([0.0; 4]));
    }
    let element = ty
        .split_once('<')
        .and_then(|(_, rest)| rest.strip_suffix('>'))?;
    // The value is the element's, however the Field keeps it.
    let element = element.split(" as ").next().unwrap_or(element);
    match element {
        "Float" => Some(Value::Float(0.0)),
        "Int" => Some(Value::Int(0)),
        "Bool" => Some(Value::Bool(false)),
        "Vec2" => Some(Value::Vec2([0.0; 2])),
        "Vec3" => Some(Value::Vec3([0.0; 3])),
        "Vec4" => Some(Value::Vec4([0.0; 4])),
        _ => None,
    }
}

/// A document does not parse or does not describe a valid graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentError {
    /// Where the problem is.
    pub span: Span,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.span, self.message)
    }
}

impl std::error::Error for DocumentError {}

impl From<ParseError> for DocumentError {
    fn from(error: ParseError) -> Self {
        Self {
            span: error.span,
            message: error.message,
        }
    }
}

/// Reads the text of a document at a resolved path.
pub type Loader<'a> = dyn FnMut(&Path) -> Result<String, String> + 'a;

/// Resolves a document's `use` items to the structs and type aliases they
/// declare, given the document's directory, so every Kernel the document
/// holds sees them. The caller reads the files; the graph only asks.
pub type Declare<'a> = dyn FnMut(
        &Path,
        &[orla_language::Use],
    ) -> Result<
        (
            Vec<orla_language::StructDecl>,
            Vec<orla_language::TypeAlias>,
            Vec<orla_language::FnDecl>,
        ),
        String,
    > + 'a;

/// A Kernel written inside a node, for the loader to publish before the
/// graph runs.
#[derive(Clone, Debug, PartialEq)]
pub struct InlineKernel {
    /// The flattened node that carries it.
    pub node: NodeId,
    /// The id it publishes as: its own, or an anonymous one.
    pub id: KernelId,
    /// The parsed Kernel, with graph-param defaults resolved to literals.
    pub kernel: Kernel,
    /// Its source text.
    pub text: String,
    /// The document it was written in.
    pub document: PathBuf,
}

/// A lowered document: the graph plus everything the loader must publish
/// and watch.
#[derive(Clone, Debug, PartialEq)]
pub struct Lowered {
    /// The graph.
    pub graph: Graph,
    /// Inline Kernels from every level, in document order.
    pub inline: Vec<InlineKernel>,
    /// `use` targets from every level, resolved, with whether the path
    /// named a directory.
    pub uses: Vec<(PathBuf, bool)>,
    /// Sub-graph documents that were loaded, resolved.
    pub documents: Vec<PathBuf>,
}

/// An inline Kernel's text as `publish` takes it: the node's Kernel as
/// written, with what only a node may write taken out (its `domain`
/// statement and the wiring on its inputs), a default that named a graph
/// param replaced by the param's value (`edits`, as byte ranges of the
/// document and the text to put there), and the id the node gave it
/// written in when it had none of its own. What `kernel` returns over
/// the protocol republishes as is.
pub fn publishable_text(
    document: &str,
    kernel: &Kernel,
    id: &KernelId,
    mut edits: Vec<(Range<usize>, String)>,
) -> String {
    let Some(mut text) = document.get(kernel.range.clone()).map(str::to_owned) else {
        return String::new();
    };
    let base = kernel.range.start;
    if let Some(range) = &kernel.domain_range {
        let (a, b) = (range.start - base, range.end - base);
        // Take the whole line when the statement stands alone on it.
        let line_start = text[..a].rfind('\n').map_or(0, |i| i + 1);
        let line_end = text[b..].find('\n').map_or(text.len(), |i| b + i + 1);
        let alone = text[line_start..a].trim().is_empty() && text[b..line_end].trim().is_empty();
        let cut = if alone { line_start..line_end } else { a..b };
        edits.push((cut.start + base..cut.end + base, String::new()));
    }
    for port in &kernel.inputs {
        if let Some(range) = &port.binding_range {
            // The space before the arrow goes with it: `input x: Image;`.
            let mut start = range.start;
            while start > base && document.as_bytes()[start - 1] == b' ' {
                start -= 1;
            }
            edits.push((start..range.end, String::new()));
        }
    }
    // Later ranges first, so earlier offsets stay right.
    edits.sort_by(|a, b| b.0.start.cmp(&a.0.start));
    for (range, replacement) in edits {
        text.replace_range(range.start - base..range.end - base, &replacement);
    }
    if kernel.id.is_none() {
        text = format!("kernel {id}{}", &text["kernel".len()..]);
    }
    text
}

/// The id of an anonymous inline Kernel: the document name and the node
/// name, lowercased with unsupported characters replaced, at version 1.
pub fn anonymous_id(document: &str, node: &str) -> KernelId {
    let segment = |text: &str| {
        let mut out: String = text
            .chars()
            .map(|c| match c.to_ascii_lowercase() {
                c @ ('a'..='z' | '0'..='9' | '_') => c,
                _ => '_',
            })
            .collect();
        if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
            out.insert(0, 'n');
        }
        out
    };
    KernelId::new(&format!("{}.{}", segment(document), segment(node)), 1)
        .expect("sanitized segments form a valid id")
}

fn error(span: Span, message: impl Into<String>) -> DocumentError {
    DocumentError {
        span,
        message: message.into(),
    }
}

/// A node's `domain` as written: a literal resolves now, an expression
/// over the node's inputs and parameters resolves when the node binds.
fn domain_spec(expr: &orla_language::Expr) -> Result<DomainSpec, DocumentError> {
    let domain = domain_expr(expr)?;
    if domain.is_constant() {
        let literal = domain
            .resolve(&|_| None, &|_| None, &|_| None, &|_, _| None)
            .map_err(|e| error(expr.span, e.message))?;
        Ok(DomainSpec::Literal(literal))
    } else {
        Ok(DomainSpec::Expr(domain))
    }
}

/// A document's nodes and outputs after `times` expansion.
type Expanded = (
    Vec<NodeDecl>,
    Vec<(orla_language::Spanned<String>, SourceRef)>,
);

/// What a placed sub-graph receives from its parent.
struct Placement {
    /// Node-name prefix, e.g. `fx.`.
    prefix: String,
    /// Parent bindings for the sub-graph's inputs, already rewritten into
    /// the parent's flattened space.
    inputs: BTreeMap<String, Binding>,
    /// Parent values for the sub-graph's params, already resolved.
    params: BTreeMap<String, ParamValue>,
}

/// State shared across one lowering.
struct Lowering<'a, 'b> {
    loader: &'a mut Loader<'b>,
    declare: &'a mut Declare<'b>,
    /// The caller's parameter values, which a root node's `times` reads.
    overrides: &'a BTreeMap<String, Value>,
    out: Lowered,
    /// Documents on the current placement path, to refuse recursion.
    stack: Vec<PathBuf>,
    /// The declarations of the document being lowered, its own and the
    /// ones it inherits: a filled Field's element is resolved through the
    /// aliases, and a synthesized Kernel inherits all of them as the
    /// document's own Kernels do.
    aliases: Vec<orla_language::TypeAlias>,
    structs: Vec<orla_language::StructDecl>,
    functions: Vec<orla_language::FnDecl>,
}

/// Where something in a document was written, in the `file:line:column`
/// a Kernel's diagnostics use.
fn written_at(path: &Path, span: orla_language::Span) -> String {
    format!("{}:{}:{}", path.display(), span.line, span.column)
}

fn lower(
    document: &Document,
    text: &str,
    name: &str,
    base: &Path,
    loader: &mut Loader<'_>,
    declare: &mut Declare<'_>,
    overrides: &BTreeMap<String, Value>,
) -> Result<Lowered, DocumentError> {
    let mut document = document.clone();
    let (structs, types, functions) =
        declare(base, &document.uses).map_err(|message| error(Span::default(), message))?;
    document.inherit(&structs, &types, &functions);
    let document = &document;
    let mut lowering = Lowering {
        loader,
        declare,
        overrides,
        out: Lowered {
            graph: Graph {
                name: name.to_owned(),
                externals: Vec::new(),
                assets: Vec::new(),
                texts: Vec::new(),
                sequences: Vec::new(),
                params: Vec::new(),
                times_params: Vec::new(),
                nodes: Vec::new(),
                outputs: Vec::new(),
            },
            inline: Vec::new(),
            uses: Vec::new(),
            documents: Vec::new(),
        },
        aliases: types.iter().chain(&document.types).cloned().collect(),
        structs: structs.iter().chain(&document.structs).cloned().collect(),
        functions: functions
            .iter()
            .chain(&document.functions)
            .cloned()
            .collect(),
        stack: Vec::new(),
    };
    let level = lowering.level(document, text, name, base, &PathBuf::from(name), None)?;
    lowering.out.graph.externals = level.externals;
    lowering.out.graph.assets = level.assets;
    lowering.out.graph.texts = level.texts;
    lowering.out.graph.sequences = level.sequences;
    lowering.out.graph.params = level.params;
    lowering.out.graph.outputs = level.outputs;
    Ok(lowering.out)
}

/// What one document level exposes to its parent.
struct Level {
    externals: Vec<String>,
    assets: Vec<Asset>,
    texts: Vec<(String, String)>,
    sequences: Vec<(String, PathBuf, String, u32)>,
    params: Vec<GraphParam>,
    /// Outputs in declaration order; the first is the primary.
    outputs: Vec<(String, Source)>,
}

impl Lowering<'_, '_> {
    /// The structs a literal of this document may name, and their fields
    /// in order: the document's own and those it uses.
    fn known_structs(&self) -> Vec<(String, Vec<String>)> {
        self.structs
            .iter()
            .map(|decl| {
                let fields = decl
                    .fields
                    .iter()
                    .map(|(field, _)| field.value.clone())
                    .collect();
                (decl.name.value.clone(), fields)
            })
            .collect()
    }

    /// A node with `times = N;` stands for `N` nodes `name/1` to `name/N`,
    /// each wired as written, except that a `carry port <- output;` feeds
    /// the port from the previous time's output from the second time on;
    /// `name.port` anywhere means the last time's. `N` is a literal or a
    /// param of this document, resolved from the placing graph, the
    /// caller's overrides, or the default; a root param that sizes the
    /// graph is remembered so a change to it lowers the document again.
    fn expand_times(
        &mut self,
        document: &Document,
        placement: Option<&Placement>,
        params: &[GraphParam],
    ) -> Result<Expanded, DocumentError> {
        let mut nodes: Vec<NodeDecl> = Vec::new();
        let mut renames: BTreeMap<String, String> = BTreeMap::new();
        for decl in &document.nodes {
            let times = decl.members.iter().find_map(|member| match member {
                NodeMember::Set(name, expr) if name.value == "times" => Some((name, expr)),
                _ => None,
            });
            let Some((times_name, expr)) = times else {
                nodes.push(decl.clone());
                continue;
            };
            let value = match &expr.kind {
                ExprKind::Name(param) => {
                    if !params.iter().any(|p| &p.name == param) {
                        return Err(error(
                            expr.span,
                            format!(
                                "`{param}` is not a param of this graph; `times` is a literal or a graph param"
                            ),
                        ));
                    }
                    match placement.and_then(|p| p.params.get(param)) {
                        Some(ParamValue::Literal(v)) => v.clone(),
                        Some(ParamValue::Graph(root)) => self
                            .overrides
                            .get(root)
                            .cloned()
                            .or_else(|| {
                                self.out
                                    .graph
                                    .params
                                    .iter()
                                    .find(|p| &p.name == root)
                                    .map(|p| p.default.clone())
                            })
                            .unwrap_or_else(|| {
                                params
                                    .iter()
                                    .find(|p| &p.name == param)
                                    .map(|p| p.default.clone())
                                    .unwrap()
                            }),
                        None => {
                            if placement.is_none() {
                                self.out.graph.times_params.push(param.clone());
                            }
                            self.overrides
                                .get(param)
                                .cloned()
                                .filter(|_| placement.is_none())
                                .unwrap_or_else(|| {
                                    params
                                        .iter()
                                        .find(|p| &p.name == param)
                                        .map(|p| p.default.clone())
                                        .unwrap()
                                })
                        }
                    }
                }
                _ => literal(expr, &self.known_structs())?,
            };
            let count = match value {
                Value::Int(n) if n >= 1 => n as usize,
                other => {
                    return Err(error(
                        times_name.span,
                        format!("`times` must be a whole number of at least 1, found {other:?}"),
                    ));
                }
            };
            let carries: Vec<(String, String)> = decl
                .members
                .iter()
                .filter_map(|member| match member {
                    NodeMember::Carry(port, output) => {
                        Some((port.value.clone(), output.value.clone()))
                    }
                    _ => None,
                })
                .collect();
            for (port, _) in &carries {
                let bound = decl.members.iter().any(|member| match member {
                    NodeMember::Bind(p, _) | NodeMember::Previous { port: p, .. } => {
                        &p.value == port
                    }
                    _ => false,
                });
                if !bound {
                    return Err(error(
                        times_name.span,
                        format!("carried port `{port}` needs a binding for the first time"),
                    ));
                }
            }
            let base_name = decl.name.value.clone();
            if count == 1 {
                // One time is the node itself, under its own name.
                let mut copy = decl.clone();
                copy.members.retain(|member| {
                    !matches!(member, NodeMember::Carry(..))
                        && !matches!(member, NodeMember::Set(name, _) if name.value == "times")
                });
                nodes.push(copy);
                continue;
            }
            renames.insert(base_name.clone(), format!("{base_name}/{count}"));
            for k in 1..=count {
                let mut copy = decl.clone();
                copy.name.value = format!("{base_name}/{k}");
                copy.members = decl
                    .members
                    .iter()
                    .filter_map(|member| match member {
                        NodeMember::Set(name, _) if name.value == "times" => None,
                        NodeMember::Carry(..) => None,
                        NodeMember::Bind(port, _) | NodeMember::Previous { port, .. }
                            if k > 1 && carries.iter().any(|(p, _)| p == &port.value) =>
                        {
                            let output = &carries.iter().find(|(p, _)| p == &port.value).unwrap().1;
                            Some(NodeMember::Bind(
                                port.clone(),
                                SourceRef {
                                    node: Some(format!("{base_name}/{}", k - 1)),
                                    name: output.clone(),
                                    span: port.span,
                                },
                            ))
                        }
                        other => Some(other.clone()),
                    })
                    .collect();
                nodes.push(copy);
            }
        }
        // `name.port` written anywhere now means the last time's.
        let rename = |reference: &mut SourceRef| {
            if let Some(node) = &reference.node
                && let Some(last) = renames.get(node)
            {
                reference.node = Some(last.clone());
            }
        };
        if !renames.is_empty() {
            for node in &mut nodes {
                for member in &mut node.members {
                    match member {
                        NodeMember::Bind(_, reference) => rename(reference),
                        NodeMember::Previous {
                            source, initial, ..
                        } => {
                            rename(source);
                            rename(initial);
                        }
                        NodeMember::Set(..) | NodeMember::Carry(..) => {}
                    }
                }
                if let NodeKind::Inline(kernel) = &mut node.kind {
                    for input in &mut kernel.inputs {
                        match &mut input.binding {
                            Some(BindingDecl::Now(reference)) => rename(reference),
                            Some(BindingDecl::Previous {
                                source, initial, ..
                            }) => {
                                rename(source);
                                rename(initial);
                            }
                            None => {}
                        }
                    }
                }
            }
        }
        let mut outputs = document.outputs.clone();
        for (_, reference) in &mut outputs {
            rename(reference);
        }
        Ok((nodes, outputs))
    }

    /// Lowers one document whose source is `text`. `placement` is `None`
    /// for the root.
    fn level(
        &mut self,
        document: &Document,
        text: &str,
        name: &str,
        base: &Path,
        path: &Path,
        placement: Option<&Placement>,
    ) -> Result<Level, DocumentError> {
        let prefix = placement.map(|p| p.prefix.as_str()).unwrap_or("");
        for item in &document.uses {
            self.out
                .uses
                .push((base.join(&item.path.value), item.path.value.ends_with('/')));
        }
        let mut externals = Vec::new();
        let mut assets = Vec::new();
        let mut texts = Vec::new();
        let mut sequences = Vec::new();
        for input in &document.inputs {
            // The type is documentation for the reader and the placing
            // graph; what arrives is checked by shape and layout where it
            // binds. A caller can supply images from files; a parent graph
            // wires anything.
            if input.binding.is_some() {
                return Err(error(
                    input.name.span,
                    "a graph input is supplied by the caller and cannot be wired",
                ));
            }
            if let Some(file) = &input.file {
                if placement.is_some() {
                    return Err(error(
                        file.span,
                        "a placed graph's input is wired by the placing graph; it cannot name a file",
                    ));
                }
                assets.push(Asset {
                    name: input.name.value.clone(),
                    path: base.join(&file.value),
                    ty: resolved(&input.ty.value, &self.aliases),
                    bytes: false,
                });
            }
            if let Some(bytes) = &input.bytes {
                if placement.is_some() {
                    return Err(error(
                        bytes.span,
                        "a placed graph's input is wired by the placing graph; it cannot name a file",
                    ));
                }
                assets.push(Asset {
                    name: input.name.value.clone(),
                    path: base.join(&bytes.value),
                    ty: resolved(&input.ty.value, &self.aliases),
                    bytes: true,
                });
            }
            if let Some(text) = &input.text {
                if placement.is_some() {
                    return Err(error(
                        text.span,
                        "a placed graph's input is wired by the placing graph; it cannot hold text",
                    ));
                }
                if text.value.is_empty() {
                    return Err(error(
                        text.span,
                        "an input's text needs at least one character; a Field has no empty extent",
                    ));
                }
                texts.push((input.name.value.clone(), text.value.clone()));
            }
            if let Some(frames) = &input.frames {
                if placement.is_some() {
                    return Err(error(
                        frames.dir.span,
                        "a placed graph's input is wired by the placing graph; it cannot read frames",
                    ));
                }
                sequences.push((
                    input.name.value.clone(),
                    base.join(&frames.dir.value),
                    resolved(&input.ty.value, &self.aliases),
                    frames.rate,
                ));
            }
            externals.push(input.name.value.clone());
        }
        let mut params = Vec::new();
        for decl in &document.params {
            let default = literal(&decl.default, &self.known_structs())?;
            if params
                .iter()
                .any(|p: &GraphParam| p.name == decl.name.value)
            {
                return Err(error(
                    decl.name.span,
                    format!("param `{}` is declared twice", decl.name.value),
                ));
            }
            params.push(GraphParam {
                name: decl.name.value.clone(),
                ty: resolved(&decl.ty.value, &self.aliases),
                default,
            });
        }
        // The root's params, before anything placed under it is walked. A
        // sub-graph's `times` names one of them through its placement and
        // is structural, so it is resolved here rather than at run time;
        // without this the lookup found an empty list and fell back to the
        // sub-graph's own default, which is a scene's parameter silently
        // doing nothing.
        if placement.is_none() {
            self.out.graph.params = params.clone();
        }
        if let Some(placement) = placement {
            for input in placement.inputs.keys() {
                if !externals.contains(input) {
                    return Err(error(
                        Span::default(),
                        format!("sub-graph `{}` has no input `{input}`", path.display()),
                    ));
                }
            }
            for external in &externals {
                if !placement.inputs.contains_key(external) {
                    return Err(error(
                        Span::default(),
                        format!(
                            "sub-graph `{}` input `{external}` must be wired where it is placed",
                            path.display()
                        ),
                    ));
                }
            }
            for param in placement.params.keys() {
                if !params.iter().any(|p| &p.name == param) {
                    return Err(error(
                        Span::default(),
                        format!("sub-graph `{}` has no param `{param}`", path.display()),
                    ));
                }
            }
        }
        // Resolves a value written in this document: a literal, or a graph
        // param of this document, which in a placed sub-graph becomes what
        // the parent passed (or the default).
        let known = self.known_structs();
        let resolve_value = |expr: &Expr| -> Result<ParamValue, DocumentError> {
            if let ExprKind::Name(param) = &expr.kind {
                let declared = params.iter().find(|p| &p.name == param).ok_or_else(|| {
                    error(
                        expr.span,
                        format!("`{param}` is not a param of this graph; values are literals or graph params"),
                    )
                })?;
                return Ok(match placement.and_then(|p| p.params.get(param)) {
                    Some(from_parent) => from_parent.clone(),
                    None if placement.is_some() => ParamValue::Literal(declared.default.clone()),
                    None => ParamValue::Graph(param.clone()),
                });
            }
            let value = literal(expr, &known)?;
            Ok(ParamValue::Literal(value))
        };
        let default_of = |value: &ParamValue| -> Value {
            match value {
                ParamValue::Literal(v) => v.clone(),
                ParamValue::Graph(name) => params
                    .iter()
                    .find(|p| &p.name == name)
                    .map(|p| p.default.clone())
                    .unwrap_or(Value::Float(0.0)),
            }
        };
        // Rewrites a source written in this document into the flattened
        // space: prefixes node names, replaces this document's inputs with
        // the parent's bindings (as a source when they are `Now`).
        let rewrite_now =
            |reference: &SourceRef| -> Result<Binding, DocumentError> {
                match &reference.node {
                    Some(node) => Ok(Binding::Now(Source::Node {
                        node: NodeId(format!("{prefix}{node}")),
                        port: reference.name.clone(),
                    })),
                    None => match placement {
                        Some(placement) => placement
                            .inputs
                            .get(&reference.name)
                            .cloned()
                            .ok_or_else(|| {
                                error(
                                    reference.span,
                                    format!("`{}` is not an input of this graph", reference.name),
                                )
                            }),
                        None => Ok(Binding::Now(Source::External(reference.name.clone()))),
                    },
                }
            };
        let rewrite_previous = |source: &SourceRef,
                                initial: &SourceRef,
                                migration: Option<&Spanned<String>>|
         -> Result<Binding, DocumentError> {
            let migration =
                match migration {
                    Some(named) => Some(named.value.parse::<KernelId>().map_err(|e| {
                        error(named.span, format!("`migrate` names a kernel id: {e}"))
                    })?),
                    None => None,
                };
            let Some(node) = &source.node else {
                return Err(error(
                    source.span,
                    "`<~` reads a node output; graph inputs carry no state",
                ));
            };
            let initial = match rewrite_now(initial)? {
                Binding::Now(source) => source,
                Binding::Previous { .. } => {
                    return Err(error(
                        initial.span,
                        "the `from` source of a `<~` binding must be a current value",
                    ));
                }
            };
            Ok(Binding::Previous {
                source: Source::Node {
                    node: NodeId(format!("{prefix}{node}")),
                    port: source.name.clone(),
                },
                initial,
                migration,
            })
        };
        let mut aliases: BTreeMap<String, Vec<(String, Source)>> = BTreeMap::new();
        let first_node = self.out.graph.nodes.len();
        let (nodes, declared_outputs) = self.expand_times(document, placement, &params)?;
        for decl in &nodes {
            let id = NodeId(format!("{prefix}{}", decl.name.value));
            let mut inputs = BTreeMap::new();
            let mut node_params = BTreeMap::new();
            let mut domain = None;
            let mut is_async = false;
            let mut bind = |port: &orla_language::Spanned<String>, binding: Binding| {
                if inputs.insert(port.value.clone(), binding).is_some() {
                    return Err(error(
                        port.span,
                        format!("port `{}` is bound twice", port.value),
                    ));
                }
                Ok(())
            };
            for member in &decl.members {
                match member {
                    NodeMember::Set(name, expr) if name.value == "domain" => {
                        if domain.replace(domain_spec(expr)?).is_some() {
                            return Err(error(name.span, "`domain` is set twice"));
                        }
                    }
                    NodeMember::Set(name, expr) if name.value == "async" => {
                        let ExprKind::Bool(value) = &expr.kind else {
                            return Err(error(expr.span, "`async` is `true` or `false`"));
                        };
                        is_async = *value;
                    }
                    NodeMember::Set(name, expr) => {
                        if node_params
                            .insert(name.value.clone(), resolve_value(expr)?)
                            .is_some()
                        {
                            return Err(error(
                                name.span,
                                format!("parameter `{}` is set twice", name.value),
                            ));
                        }
                    }
                    NodeMember::Bind(port, reference) => bind(port, rewrite_now(reference)?)?,
                    NodeMember::Previous {
                        port,
                        source,
                        initial,
                        migration,
                    } => bind(port, rewrite_previous(source, initial, migration.as_ref())?)?,
                    NodeMember::Carry(port, _) => {
                        return Err(error(port.span, "`carry` needs `times = N;` on the node"));
                    }
                }
            }
            match &decl.kind {
                NodeKind::Reference(kernel) => {
                    let kernel_id = kernel
                        .value
                        .parse::<KernelId>()
                        .map_err(|e| error(kernel.span, e.to_string()))?;
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                    });
                }
                NodeKind::Inline(kernel) => {
                    let kernel_id = match &kernel.id {
                        Some(written) => written
                            .value
                            .parse::<KernelId>()
                            .map_err(|e| error(written.span, e.to_string()))?,
                        None => anonymous_id(name, &decl.name.value),
                    };
                    for input in &kernel.inputs {
                        match &input.binding {
                            Some(BindingDecl::Now(reference)) => {
                                bind(&input.name, rewrite_now(reference)?)?;
                            }
                            Some(BindingDecl::Previous {
                                source,
                                initial,
                                migration,
                            }) => {
                                bind(
                                    &input.name,
                                    rewrite_previous(source, initial, migration.as_ref())?,
                                )?;
                            }
                            None => {
                                return Err(error(
                                    input.name.span,
                                    format!(
                                        "inline kernel input `{}` must be wired: `input {}: Image <- source;`",
                                        input.name.value, input.name.value
                                    ),
                                ));
                            }
                        }
                    }
                    if let Some(expr) = &kernel.domain {
                        domain = Some(domain_spec(expr)?);
                    }
                    // Param defaults that name a graph param become node
                    // values; the compiled Kernel keeps a literal default.
                    let mut compiled = (**kernel).clone();
                    let mut edits = Vec::new();
                    for param in &mut compiled.params {
                        if let ExprKind::Name(_) = &param.default.kind {
                            let value = resolve_value(&param.default)?;
                            let literal = default_of(&value);
                            edits.push((param.default_range.clone(), literal.to_string()));
                            param.default = literal_expr(literal, param.default.span);
                            node_params.insert(param.name.value.clone(), value);
                        }
                    }
                    let text = publishable_text(text, kernel, &kernel_id, edits);
                    self.out.inline.push(InlineKernel {
                        node: id.clone(),
                        id: kernel_id.clone(),
                        kernel: compiled,
                        text,
                        document: path.to_path_buf(),
                    });
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                    });
                }
                NodeKind::Fill(ty) => {
                    if domain.is_none() {
                        return Err(error(
                            ty.span,
                            "a filled Field needs a `domain`: `domain = 65;`, `domain = vec2(96, 64);`",
                        ));
                    }
                    let kernel_id = anonymous_id(name, &decl.name.value);
                    // The fill is the node's `value` parameter: its
                    // literal is the Kernel's default, so the node with
                    // no `value` is the zero of the element.
                    let spelled = resolved(&ty.value, &self.aliases);
                    let value = match node_params.get("value") {
                        Some(given) => default_of(given),
                        None => zero_of(&spelled).ok_or_else(|| {
                            error(
                                ty.span,
                                format!(
                                    "a filled Field holds Float, Int, Bool, Vec2, Vec3 or Vec4 (and `Image` holds a pixel); `{}` holds something else, so a Kernel fills it",
                                    ty.value
                                ),
                            )
                        })?,
                    };
                    let text = format!(
                        "kernel {kernel_id} {{\n    requires {{ field.write; }}\n    param value: {} = {};\n    output result: {};\n    compute {{ return value; }}\n}}\n",
                        value
                            .ty()
                            .map_or_else(|| ty.value.clone(), |t| t.to_string()),
                        literal_text(&value),
                        ty.value
                    );
                    let mut kernel = orla_language::parse(&text).map_err(|e| {
                        error(ty.span, format!("a filled Field of `{}`: {e}", ty.value))
                    })?;
                    // It inherits what the document's own Kernels do, so
                    // its port type may be an alias from a used document.
                    kernel.structs = self.structs.clone();
                    kernel.types = self.aliases.clone();
                    kernel.functions = self.functions.clone();
                    self.out.inline.push(InlineKernel {
                        node: id.clone(),
                        id: kernel_id.clone(),
                        kernel,
                        text,
                        document: path.to_path_buf(),
                    });
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                    });
                }
                NodeKind::SubGraph(sub_path) => {
                    if domain.is_some() {
                        return Err(error(
                            sub_path.span,
                            "a sub-graph takes no `domain`; its nodes set their own",
                        ));
                    }
                    if is_async {
                        return Err(error(
                            sub_path.span,
                            "a sub-graph takes no `async`; its nodes say which of them run off the frame",
                        ));
                    }
                    let resolved = base.join(&sub_path.value);
                    if self.stack.contains(&resolved) {
                        return Err(error(
                            sub_path.span,
                            format!("sub-graph `{}` places itself", sub_path.value),
                        ));
                    }
                    let text = (self.loader)(&resolved).map_err(|message| {
                        error(
                            sub_path.span,
                            format!("cannot load `{}`: {message}", sub_path.value),
                        )
                    })?;
                    let mut sub_document = orla_language::parse_document(&text)
                        .map_err(|e| error(sub_path.span, format!("{}:{e}", sub_path.value)))?;
                    let sub_base = resolved.parent().unwrap_or_else(|| Path::new("."));
                    let (structs, types, functions) = (self.declare)(sub_base, &sub_document.uses)
                        .map_err(|m| error(sub_path.span, format!("{}: {m}", sub_path.value)))?;
                    sub_document.inherit(&structs, &types, &functions);
                    let outer = (
                        std::mem::replace(
                            &mut self.aliases,
                            types
                                .iter()
                                .chain(&sub_document.types)
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                        std::mem::replace(
                            &mut self.structs,
                            structs
                                .iter()
                                .chain(&sub_document.structs)
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                        std::mem::replace(
                            &mut self.functions,
                            functions
                                .iter()
                                .chain(&sub_document.functions)
                                .cloned()
                                .collect::<Vec<_>>(),
                        ),
                    );
                    let sub_name = resolved
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("graph")
                        .to_owned();
                    let sub_base = resolved.parent().unwrap_or(base).to_path_buf();
                    self.out.documents.push(resolved.clone());
                    self.stack.push(resolved.clone());
                    let placed = Placement {
                        prefix: format!("{}.", id.0),
                        inputs,
                        params: node_params,
                    };
                    let level = self
                        .level(
                            &sub_document,
                            &text,
                            &sub_name,
                            &sub_base,
                            &resolved,
                            Some(&placed),
                        )
                        .map_err(|e| error(sub_path.span, format!("{}:{e}", sub_path.value)));
                    self.stack.pop();
                    (self.aliases, self.structs, self.functions) = outer;
                    let level = level?;
                    aliases.insert(id.0.clone(), level.outputs);
                }
            }
        }
        // Sources that named a placed sub-graph now point at its outputs.
        let resolve = |source: &mut Source| -> Result<(), DocumentError> {
            if let Source::Node { node, port } = source
                && let Some(outputs) = aliases.get(&node.0)
            {
                *source = outputs
                    .iter()
                    .find(|(name, _)| name == port)
                    .map(|(_, source)| source.clone())
                    .ok_or_else(|| {
                        error(
                            Span::default(),
                            format!("sub-graph `{node}` has no output `{port}`"),
                        )
                    })?;
            }
            Ok(())
        };
        for node in &mut self.out.graph.nodes[first_node..] {
            for binding in node.inputs.values_mut() {
                match binding {
                    Binding::Now(source) => resolve(source)?,
                    Binding::Previous {
                        source, initial, ..
                    } => {
                        resolve(source)?;
                        resolve(initial)?;
                    }
                }
            }
        }
        let mut outputs: Vec<(String, Source)> = Vec::new();
        for (output, reference) in &declared_outputs {
            let mut source = match rewrite_now(reference)? {
                Binding::Now(source) => source,
                Binding::Previous { .. } => {
                    return Err(error(reference.span, "an output must be a current value"));
                }
            };
            resolve(&mut source)?;
            if outputs.iter().any(|(name, _)| name == &output.value) {
                return Err(error(
                    output.span,
                    format!("output `{}` is declared twice", output.value),
                ));
            }
            outputs.push((output.value.clone(), source));
        }
        Ok(Level {
            externals,
            assets,
            texts,
            sequences,
            params,
            outputs,
        })
    }
}

/// A literal as a scene writes it: any value a Kernel can hold, a struct
/// literal naming one of `known` (a struct the document declares or uses).
fn literal(expr: &Expr, known: &[(String, Vec<String>)]) -> Result<Value, DocumentError> {
    let fields_of = |name: &str| {
        known
            .iter()
            .find(|(struct_name, _)| struct_name == name)
            .map(|(_, fields)| fields.clone())
    };
    orla_compiler::literal_in(expr, &fields_of).map_err(|e| error(expr.span, e.to_string()))
}

/// A literal expression carrying a value, for rewriting param defaults:
/// the value's own literal, read back.
fn literal_expr(value: Value, span: Span) -> Expr {
    let mut expr =
        orla_language::parse_literal(&value.to_string()).expect("a value's literal reads back");
    expr.span = span;
    expr
}

/// Parses a `.osn` document that references published Kernels only.
/// Loaders that also publish inline Kernels or place sub-graphs use
/// [`load`].
pub fn parse(text: &str) -> Result<Graph, DocumentError> {
    let document = orla_language::parse_document(text).map_err(DocumentError::from)?;
    let mut loader = |path: &std::path::Path| -> Result<String, String> {
        Err(format!(
            "sub-graph `{}` cannot be loaded here",
            path.display()
        ))
    };
    let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
    let lowered = lower(
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
            "sub-graph `{}` cannot be loaded here; use orla_scene::load with a loader",
            path.display()
        ))
    };
    let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
    lower(
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
    lower(
        document,
        text,
        name,
        base,
        loader,
        &mut none,
        &BTreeMap::new(),
    )
}

/// Loads like [`load`] and also resolves every document's `use`
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
    lower(
        document,
        text,
        name,
        base,
        loader,
        declare,
        &BTreeMap::new(),
    )
}

/// As [`load_with`], with parameter overrides that a node's
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
    lower(document, text, name, base, loader, declare, overrides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orla_abi::Domain;

    const DOCUMENT: &str = "
// a comment
input photo: Image;
param amount: Float = 0.5;

node gray: image.grayscale@1 {
    source <- photo;
    amount = amount;
}
node blocks: image.pixelate@1 {
    source <- gray.result;
    size = 8;
    tint = vec4(1.0, 0.5, 0.25, 1.0);
    domain = vec2(64, 32);
}
output result <- blocks.result;
";

    #[test]
    fn documents_round_trip() {
        let graph = parse(DOCUMENT).unwrap();
        assert_eq!(graph.externals, ["photo"]);
        assert_eq!(graph.params[0].default, Value::Float(0.5));
        assert_eq!(
            graph.nodes[0].inputs["source"],
            Binding::Now(Source::External("photo".into()))
        );
        assert_eq!(
            graph.nodes[0].params["amount"],
            ParamValue::Graph("amount".into())
        );
        assert_eq!(
            graph.nodes[1].params["size"],
            ParamValue::Literal(Value::Int(8))
        );
        assert_eq!(
            graph.nodes[1].domain,
            Some(DomainSpec::Literal(Domain::Grid(64, 32)))
        );
        let text = graph.to_string();
        assert_eq!(parse(&text).unwrap(), graph);
    }

    #[test]
    fn inline_kernels_are_reported_with_their_wiring() {
        let source = "input photo: Image;
             param decay: Float = 0.9;
             node trail: kernel {
                 requires { field.read; field.write; }
                 input current: Image <- photo;
                 input previous: Image <~ trail.result from photo;
                 param fade: Float = decay;
                 output result: Image;
                 domain = vec2(4, 4);
                 compute { return sample(current) * fade + sample(previous); }
             }
             node named: kernel fx.glow@2 {
                 requires { field.read; field.write; }
                 input source: Image <- trail.result;
                 output result: Image;
                 compute { return sample(source); }
             }
             output result <- named.result;";
        let document = orla_language::parse_document(source).unwrap();
        let mut none = |path: &Path| -> Result<String, String> { Err(path.display().to_string()) };
        let lowered = load(&document, source, "My-Scene", Path::new("."), &mut none).unwrap();
        // The text is publishable as it is: the node's id written in, the
        // node's domain left out, and it parses on its own.
        assert!(
            !lowered.inline[0].text.contains("domain"),
            "{}",
            lowered.inline[0].text
        );
        assert!(
            orla_language::parse(&lowered.inline[0].text).is_ok(),
            "{}",
            lowered.inline[0].text
        );
        assert!(
            lowered.inline[0]
                .text
                .starts_with("kernel my_scene.trail@1 {"),
            "{}",
            lowered.inline[0].text
        );
        assert_eq!(lowered.inline.len(), 2);
        assert_eq!(lowered.inline[0].id.to_string(), "my_scene.trail@1");
        assert_eq!(lowered.inline[1].id.to_string(), "fx.glow@2");
        let trail = &lowered.graph.nodes[0];
        assert_eq!(trail.kernel.to_string(), "my_scene.trail@1");
        assert!(matches!(trail.inputs["previous"], Binding::Previous { .. }));
        assert_eq!(trail.domain, Some(DomainSpec::Literal(Domain::Grid(4, 4))));
        assert_eq!(trail.params["fade"], ParamValue::Graph("decay".into()));
        let compiled =
            orla_compiler::compile_kernel(&lowered.inline[0].kernel, Some(&lowered.inline[0].id))
                .unwrap();
        assert_eq!(compiled.manifest.params[0].default, Value::Float(0.9));
    }

    #[test]
    fn sub_graphs_flatten_with_prefixed_nodes_wired_inputs_and_passed_params() {
        let display = "use \"lib/\";
            input source: Image;
            param exposure: Float = 1.0;
            node map: image.tonemap@1 { source <- source; exposure = exposure; }
            node fade: image.fade@1 { previous <~ fade.result from map.result; }
            output result <- fade.result;";
        let root = orla_language::parse_document(
            "param strength: Float = 2.0;
             node scene: image.gradient@1 { domain = vec2(8, 8); }
             node display: graph \"shared/display.osn\" { source <- scene.result; exposure = strength; }
             node again: graph \"shared/display.osn\" { source <- display.result; }
             output result <- again.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<String, String> {
            assert_eq!(path, Path::new("root/shared/display.osn"));
            Ok(display.to_owned())
        };
        let lowered = load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
        let graph = &lowered.graph;
        let ids: Vec<&str> = graph.nodes.iter().map(|n| n.id.0.as_str()).collect();
        assert_eq!(
            ids,
            [
                "scene",
                "display.map",
                "display.fade",
                "again.map",
                "again.fade"
            ]
        );
        assert_eq!(
            graph.nodes[1].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("scene".into()),
                port: "result".into()
            })
        );
        assert_eq!(
            graph.nodes[1].params["exposure"],
            ParamValue::Graph("strength".into())
        );
        assert_eq!(
            graph.nodes[3].params["exposure"],
            ParamValue::Literal(Value::Float(1.0))
        );
        assert_eq!(
            graph.nodes[3].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("display.fade".into()),
                port: "result".into()
            })
        );
        assert!(
            matches!(&graph.nodes[2].inputs["previous"], Binding::Previous { source: Source::Node { node, .. }, .. } if node.0 == "display.fade")
        );
        assert_eq!(
            graph.outputs[0].1,
            Source::Node {
                node: NodeId("again.fade".into()),
                port: "result".into()
            }
        );
        assert_eq!(
            lowered.uses,
            vec![(PathBuf::from("root/shared/lib/"), true); 2]
        );
        assert_eq!(
            lowered.documents,
            vec![PathBuf::from("root/shared/display.osn"); 2]
        );
        assert!(graph.order().is_ok());
    }

    #[test]
    fn errors_name_the_position() {
        let error = parse("node a: Bad {\n}\noutput r <- a.r;").unwrap_err();
        assert_eq!(error.span.line, 1);
        let error = parse("node a: t.k@1 { x <- y; x <- z; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("bound twice"), "{error}");
        let error = parse("node a: t.k@1 { domain = true; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("`domain` is a count"), "{error}");
        let error = parse("node a: kernel { requires { field.write; } input s: Image; output result: Image; compute { return sample(s); } }\noutput r <- a.result;").unwrap_err();
        assert!(error.message.contains("must be wired"), "{error}");
        let error = parse("node a: t.k@1 { y = 1.0 + 2.0; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("literal"), "{error}");
        let error = parse("node a: t.k@1 { y = nope; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("not a param"), "{error}");
        let error = parse("node a: graph \"x.osn\" { }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("cannot load"), "{error}");
    }
}

#[cfg(test)]
mod times_tests {
    use super::*;

    const REPEATED: &str = "
param level: Int = 2;
node seed: test.zero@1 { domain = 4; }
node step: test.inc@1 { times = level; carry previous <- result; previous <- seed.result; }
node show: test.show@1 { source <- step.result; }
output result <- show.result;
";

    fn ids(graph: &Graph) -> Vec<String> {
        graph.nodes.iter().map(|n| n.id.0.clone()).collect()
    }

    #[test]
    fn times_expands_a_node_and_carries_between_times() {
        let graph = parse(REPEATED).unwrap();
        assert_eq!(ids(&graph), ["seed", "step/1", "step/2", "show"]);
        assert_eq!(graph.times_params, ["level"]);
        let first = &graph.nodes[1];
        assert_eq!(
            first.inputs["previous"],
            Binding::Now(Source::Node {
                node: NodeId("seed".into()),
                port: "result".into()
            })
        );
        let second = &graph.nodes[2];
        assert_eq!(
            second.inputs["previous"],
            Binding::Now(Source::Node {
                node: NodeId("step/1".into()),
                port: "result".into()
            })
        );
        assert_eq!(
            graph.nodes[3].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("step/2".into()),
                port: "result".into()
            }),
            "the name means the last time"
        );
        // The caller's value sizes the graph; one time keeps the plain name.
        let document = orla_language::parse_document(REPEATED).unwrap();
        let mut none: Box<Declare<'_>> = Box::new(|_, _| Ok((Vec::new(), Vec::new(), Vec::new())));
        let mut loader: Box<Loader<'_>> = Box::new(|_| Err("no files".into()));
        let overrides = BTreeMap::from([("level".to_owned(), Value::Int(4))]);
        let lowered = load_with_params(
            &document,
            REPEATED,
            "g",
            Path::new("."),
            &mut loader,
            &mut none,
            &overrides,
        )
        .unwrap();
        assert_eq!(
            ids(&lowered.graph),
            ["seed", "step/1", "step/2", "step/3", "step/4", "show"]
        );
        let overrides = BTreeMap::from([("level".to_owned(), Value::Int(1))]);
        let lowered = load_with_params(
            &document,
            REPEATED,
            "g",
            Path::new("."),
            &mut loader,
            &mut none,
            &overrides,
        )
        .unwrap();
        assert_eq!(ids(&lowered.graph), ["seed", "step", "show"]);
    }

    #[test]
    fn times_and_carry_are_checked() {
        let zero = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = 0; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(zero.message.contains("at least 1"), "{zero}");
        let unbound = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = 2; carry previous <- result; }\noutput result <- r.result;").unwrap_err();
        assert!(
            unbound
                .message
                .contains("needs a binding for the first time"),
            "{unbound}"
        );
        let orphan = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { carry previous <- result; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(orphan.message.contains("needs `times"), "{orphan}");
        let unknown = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = n; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(unknown.message.contains("not a param"), "{unknown}");
    }
}
