//! Lowering of a parsed `.osn` document into the Graph model.

use std::collections::BTreeMap;
use std::fmt;

use orla_abi::{Domain, Value};
use orla_ir::KernelId;
use orla_language::{
    BindingDecl, Document, Kernel, NodeDecl, NodeKind, NodeMember, ParseError, SourceRef, Span,
};

use crate::{Binding, Graph, Node, NodeId, Source};

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

/// A Kernel written inside a node, for the loader to publish before the
/// graph runs.
#[derive(Clone, Debug, PartialEq)]
pub struct InlineKernel<'a> {
    /// The node that carries it.
    pub node: NodeId,
    /// The id it publishes as: its own, or an anonymous one.
    pub id: KernelId,
    /// The parsed Kernel, whose `range` locates its text in the document.
    pub kernel: &'a Kernel,
}

/// A lowered document: the graph plus the inline Kernels it needs.
#[derive(Clone, Debug, PartialEq)]
pub struct Lowered<'a> {
    /// The graph.
    pub graph: Graph,
    /// Inline Kernels in document order.
    pub inline: Vec<InlineKernel<'a>>,
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

fn source(reference: &SourceRef) -> Source {
    match &reference.node {
        Some(node) => Source::Node {
            node: NodeId(node.clone()),
            port: reference.name.clone(),
        },
        None => Source::External(reference.name.clone()),
    }
}

fn domain_value(value: Value, span: Span) -> Result<Domain, DocumentError> {
    let Value::Vec2([width, height]) = value else {
        return Err(error(
            span,
            "`domain` needs `vec2(width, height)` with positive whole numbers",
        ));
    };
    let side = |v: f32| {
        (v > 0.0 && v.fract() == 0.0 && v <= u32::MAX as f32)
            .then_some(v as u32)
            .ok_or_else(|| error(span, "`domain` sides must be positive whole numbers"))
    };
    Ok(Domain {
        width: side(width)?,
        height: side(height)?,
    })
}

pub(crate) fn lower<'a>(document: &'a Document, name: &str) -> Result<Lowered<'a>, DocumentError> {
    let externals: Vec<String> = document
        .inputs
        .iter()
        .map(|input| {
            if input.ty.value != "Image" {
                return Err(error(
                    input.ty.span,
                    format!(
                        "graph input `{}` must be an Image in this version",
                        input.name.value
                    ),
                ));
            }
            if input.binding.is_some() {
                return Err(error(
                    input.name.span,
                    "a graph input is supplied by the caller and cannot be wired",
                ));
            }
            Ok(input.name.value.clone())
        })
        .collect::<Result<_, _>>()?;
    let mut nodes = Vec::with_capacity(document.nodes.len());
    let mut inline = Vec::new();
    for decl in &document.nodes {
        let (node, kernel) = lower_node(decl, name)?;
        if let Some(kernel) = kernel {
            inline.push(InlineKernel {
                node: node.id.clone(),
                id: node.kernel.clone(),
                kernel,
            });
        }
        nodes.push(node);
    }
    let outputs = document
        .outputs
        .iter()
        .map(|(name, reference)| (name.value.clone(), source(reference)))
        .collect();
    Ok(Lowered {
        graph: Graph {
            name: name.to_owned(),
            externals,
            nodes,
            outputs,
        },
        inline,
    })
}

fn lower_node<'a>(
    decl: &'a NodeDecl,
    document: &str,
) -> Result<(Node, Option<&'a Kernel>), DocumentError> {
    let id = NodeId(decl.name.value.clone());
    let mut inputs = BTreeMap::new();
    let mut params = BTreeMap::new();
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
    match &decl.kind {
        NodeKind::Reference(kernel) => {
            let kernel_id = kernel
                .value
                .parse::<KernelId>()
                .map_err(|e| error(kernel.span, e.to_string()))?;
            for member in &decl.members {
                match member {
                    NodeMember::Set(name, expr) if name.value == "domain" => {
                        let value = orla_compiler::literal(expr)
                            .map_err(|e| error(expr.span, e.to_string()))?;
                        if domain.replace(domain_value(value, expr.span)?).is_some() {
                            return Err(error(name.span, "`domain` is set twice"));
                        }
                    }
                    NodeMember::Set(name, expr) => {
                        let value = orla_compiler::literal(expr)
                            .map_err(|e| error(expr.span, e.to_string()))?;
                        if params.insert(name.value.clone(), value).is_some() {
                            return Err(error(
                                name.span,
                                format!("parameter `{}` is set twice", name.value),
                            ));
                        }
                    }
                    NodeMember::Bind(port, reference) => {
                        bind(port, Binding::Now(source(reference)))?;
                    }
                    NodeMember::Previous {
                        port,
                        source: previous,
                        initial,
                    } => bind(
                        port,
                        Binding::Previous {
                            source: source(previous),
                            initial: source(initial),
                        },
                    )?,
                }
            }
            Ok((
                Node {
                    id,
                    kernel: kernel_id,
                    inputs,
                    params,
                    domain,
                },
                None,
            ))
        }
        NodeKind::Inline(kernel) => {
            let kernel_id = match &kernel.id {
                Some(written) => written
                    .value
                    .parse::<KernelId>()
                    .map_err(|e| error(written.span, e.to_string()))?,
                None => anonymous_id(document, &decl.name.value),
            };
            for input in &kernel.inputs {
                match &input.binding {
                    Some(BindingDecl::Now(reference)) => {
                        bind(&input.name, Binding::Now(source(reference)))?;
                    }
                    Some(BindingDecl::Previous {
                        source: previous,
                        initial,
                    }) => bind(
                        &input.name,
                        Binding::Previous {
                            source: source(previous),
                            initial: source(initial),
                        },
                    )?,
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
                let value =
                    orla_compiler::literal(expr).map_err(|e| error(expr.span, e.to_string()))?;
                domain = Some(domain_value(value, expr.span)?);
            }
            Ok((
                Node {
                    id,
                    kernel: kernel_id,
                    inputs,
                    params,
                    domain,
                },
                Some(kernel),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT: &str = "
// a comment
input photo: Image;

node gray: image.grayscale@1 {
    source <- photo;
    amount = 0.5;
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
        assert_eq!(
            graph.nodes[0].inputs["source"],
            Binding::Now(Source::External("photo".into()))
        );
        assert_eq!(graph.nodes[0].params["amount"], Value::Float(0.5));
        assert_eq!(graph.nodes[1].params["size"], Value::Int(8));
        assert_eq!(
            graph.nodes[1].params["tint"],
            Value::Vec4([1.0, 0.5, 0.25, 1.0])
        );
        assert_eq!(
            graph.nodes[1].domain,
            Some(Domain {
                width: 64,
                height: 32
            })
        );
        assert_eq!(
            graph.outputs,
            vec![(
                "result".to_string(),
                Source::Node {
                    node: NodeId("blocks".into()),
                    port: "result".into()
                }
            )]
        );
        let text = graph.to_string();
        assert_eq!(Graph::parse(&text).unwrap(), graph);
    }

    #[test]
    fn inline_kernels_are_reported_with_their_wiring() {
        let document = orla_language::parse_document(
            "input photo: Image;
             node trail: kernel {
                 requires { image.read; image.write; }
                 input current: Image <- photo;
                 input previous: Image <~ trail.result from photo;
                 param decay: Float = 0.9;
                 output result: Image;
                 domain = vec2(4, 4);
                 compute { return sample(current) * decay + sample(previous); }
             }
             node named: kernel fx.glow@2 {
                 requires { image.read; image.write; }
                 input source: Image <- trail.result;
                 output result: Image;
                 compute { return sample(source); }
             }
             output result <- named.result;",
        )
        .unwrap();
        let lowered = Graph::from_document(&document, "My-Scene").unwrap();
        assert_eq!(lowered.inline.len(), 2);
        assert_eq!(lowered.inline[0].id.to_string(), "my_scene.trail@1");
        assert_eq!(lowered.inline[1].id.to_string(), "fx.glow@2");
        let trail = &lowered.graph.nodes[0];
        assert_eq!(trail.kernel.to_string(), "my_scene.trail@1");
        assert!(matches!(trail.inputs["previous"], Binding::Previous { .. }));
        assert_eq!(
            trail.domain,
            Some(Domain {
                width: 4,
                height: 4
            })
        );
        assert!(trail.params.is_empty(), "inline defaults are the values");
    }

    #[test]
    fn errors_name_the_position() {
        let error = Graph::parse("node a: Bad {\n}\noutput r <- a.r;").unwrap_err();
        assert_eq!(error.span.line, 1);
        let error =
            Graph::parse("node a: t.k@1 { x <- y; x <- z; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("bound twice"), "{error}");
        let error = Graph::parse("node a: t.k@1 { domain = 3; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("vec2(width, height)"), "{error}");
        let error = Graph::parse("node a: kernel { requires { image.write; } input s: Image; output result: Image; compute { return sample(s); } }\noutput r <- a.result;").unwrap_err();
        assert!(error.message.contains("must be wired"), "{error}");
        let error = Graph::parse("node a: t.k@1 { y = 1.0 + 2.0; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("literal"), "{error}");
    }
}
