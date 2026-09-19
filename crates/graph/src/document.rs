//! Lowering of parsed `.osn` documents into the Graph model, with
//! sub-graphs flattened in place.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use orla_abi::Value;
use orla_ir::KernelId;
use orla_language::{
    BindingDecl, Document, Expr, ExprKind, Kernel, NodeKind, NodeMember, ParseError, SourceRef,
    Span,
};

use crate::{Binding, DomainExpr, DomainSpec, Graph, GraphParam, Node, NodeId, ParamValue, Source};

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
    let domain = DomainExpr::from_expr(expr).map_err(|e| error(expr.span, e.message))?;
    if domain.is_constant() {
        let literal = domain
            .resolve(&|_| None, &|_| None)
            .map_err(|e| error(expr.span, e.message))?;
        Ok(DomainSpec::Literal(literal))
    } else {
        Ok(DomainSpec::Expr(domain))
    }
}

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
    out: Lowered,
    /// Documents on the current placement path, to refuse recursion.
    stack: Vec<PathBuf>,
}

pub(crate) fn lower(
    document: &Document,
    text: &str,
    name: &str,
    base: &Path,
    loader: &mut Loader<'_>,
) -> Result<Lowered, DocumentError> {
    let mut lowering = Lowering {
        loader,
        out: Lowered {
            graph: Graph {
                name: name.to_owned(),
                externals: Vec::new(),
                params: Vec::new(),
                nodes: Vec::new(),
                outputs: Vec::new(),
            },
            inline: Vec::new(),
            uses: Vec::new(),
            documents: Vec::new(),
        },
        stack: Vec::new(),
    };
    let level = lowering.level(document, text, name, base, &PathBuf::from(name), None)?;
    lowering.out.graph.externals = level.externals;
    lowering.out.graph.params = level.params;
    lowering.out.graph.outputs = level.outputs;
    Ok(lowering.out)
}

/// What one document level exposes to its parent.
struct Level {
    externals: Vec<String>,
    params: Vec<GraphParam>,
    /// Outputs in declaration order; the first is the primary.
    outputs: Vec<(String, Source)>,
}

impl Lowering<'_, '_> {
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
            externals.push(input.name.value.clone());
        }
        let mut params = Vec::new();
        for decl in &document.params {
            let default = orla_compiler::literal(&decl.default)
                .map_err(|e| error(decl.default.span, e.to_string()))?;
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
                default,
            });
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
                    None if placement.is_some() => ParamValue::Literal(declared.default),
                    None => ParamValue::Graph(param.clone()),
                });
            }
            let value =
                orla_compiler::literal(expr).map_err(|e| error(expr.span, e.to_string()))?;
            Ok(ParamValue::Literal(value))
        };
        let default_of = |value: &ParamValue| -> Value {
            match value {
                ParamValue::Literal(v) => *v,
                ParamValue::Graph(name) => params
                    .iter()
                    .find(|p| &p.name == name)
                    .map(|p| p.default)
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
        let rewrite_previous =
            |source: &SourceRef, initial: &SourceRef| -> Result<Binding, DocumentError> {
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
                })
            };
        let mut aliases: BTreeMap<String, Vec<(String, Source)>> = BTreeMap::new();
        let first_node = self.out.graph.nodes.len();
        for decl in &document.nodes {
            let id = NodeId(format!("{prefix}{}", decl.name.value));
            let mut inputs = BTreeMap::new();
            let mut node_params = BTreeMap::new();
            let mut domain = None;
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
                    } => bind(port, rewrite_previous(source, initial)?)?,
                }
            }
            match &decl.kind {
                NodeKind::Reference(kernel) => {
                    let kernel_id = kernel
                        .value
                        .parse::<KernelId>()
                        .map_err(|e| error(kernel.span, e.to_string()))?;
                    self.out.graph.nodes.push(Node {
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
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
                            Some(BindingDecl::Previous { source, initial }) => {
                                bind(&input.name, rewrite_previous(source, initial)?)?;
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
                    for param in &mut compiled.params {
                        if let ExprKind::Name(_) = &param.default.kind {
                            let value = resolve_value(&param.default)?;
                            let literal = default_of(&value);
                            param.default = literal_expr(literal, param.default.span);
                            node_params.insert(param.name.value.clone(), value);
                        }
                    }
                    let text = text
                        .get(kernel.range.clone())
                        .unwrap_or_default()
                        .to_owned();
                    self.out.inline.push(InlineKernel {
                        node: id.clone(),
                        id: kernel_id.clone(),
                        kernel: compiled,
                        text,
                        document: path.to_path_buf(),
                    });
                    self.out.graph.nodes.push(Node {
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                    });
                }
                NodeKind::SubGraph(sub_path) => {
                    if domain.is_some() {
                        return Err(error(
                            sub_path.span,
                            "a sub-graph takes no `domain`; its nodes set their own",
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
                    let sub_document = orla_language::parse_document(&text)
                        .map_err(|e| error(sub_path.span, format!("{}:{e}", sub_path.value)))?;
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
                    Binding::Previous { source, initial } => {
                        resolve(source)?;
                        resolve(initial)?;
                    }
                }
            }
        }
        let mut outputs: Vec<(String, Source)> = Vec::new();
        for (output, reference) in &document.outputs {
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
            params,
            outputs,
        })
    }
}

/// A literal expression carrying a value, for rewriting param defaults.
fn literal_expr(value: Value, span: Span) -> Expr {
    let float = |v: f32| Expr {
        kind: ExprKind::Float(v),
        span,
    };
    let kind = match value {
        Value::Float(v) => ExprKind::Float(v),
        Value::Int(v) => ExprKind::Int(v),
        Value::Bool(v) => ExprKind::Bool(v),
        Value::Vec2([x, y]) => ExprKind::Call("vec2".into(), vec![float(x), float(y)]),
        Value::Vec3([x, y, z]) => ExprKind::Call("vec3".into(), vec![float(x), float(y), float(z)]),
        Value::Vec4([x, y, z, w]) => {
            ExprKind::Call("vec4".into(), vec![float(x), float(y), float(z), float(w)])
        }
    };
    Expr { kind, span }
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
        let graph = Graph::parse(DOCUMENT).unwrap();
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
        assert_eq!(Graph::parse(&text).unwrap(), graph);
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
        let lowered =
            Graph::load(&document, source, "My-Scene", Path::new("."), &mut none).unwrap();
        assert!(
            lowered.inline[0].text.starts_with("kernel {"),
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
        let lowered = Graph::load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
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
        let error = Graph::parse("node a: Bad {\n}\noutput r <- a.r;").unwrap_err();
        assert_eq!(error.span.line, 1);
        let error =
            Graph::parse("node a: t.k@1 { x <- y; x <- z; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("bound twice"), "{error}");
        let error = Graph::parse("node a: t.k@1 { domain = true; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("`domain` is a count"), "{error}");
        let error = Graph::parse("node a: kernel { requires { field.write; } input s: Image; output result: Image; compute { return sample(s); } }\noutput r <- a.result;").unwrap_err();
        assert!(error.message.contains("must be wired"), "{error}");
        let error = Graph::parse("node a: t.k@1 { y = 1.0 + 2.0; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("literal"), "{error}");
        let error = Graph::parse("node a: t.k@1 { y = nope; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("not a param"), "{error}");
        let error = Graph::parse("node a: graph \"x.osn\" { }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("cannot load"), "{error}");
    }
}
