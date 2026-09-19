//! Parser for the `.osn` graph document format.

use std::collections::BTreeMap;
use std::fmt;

use orla_abi::{Domain, Value};
use orla_ir::KernelId;

use crate::{Binding, Graph, Node, NodeId, Source};

/// A document does not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentError {
    /// 1-based line number.
    pub line: u32,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for DocumentError {}

pub(crate) fn parse(text: &str) -> Result<Graph, DocumentError> {
    let mut name = None;
    let mut externals = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut output = None;
    for (index, raw) in text.lines().enumerate() {
        let line = index as u32 + 1;
        let error = |message: String| DocumentError { line, message };
        let content = raw.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let words: Vec<&str> = content.split_whitespace().collect();
        match words.as_slice() {
            ["graph", graph_name] => {
                if name.replace(graph_name.to_string()).is_some() {
                    return Err(error("`graph` is declared twice".into()));
                }
            }
            ["input", external] => externals.push(external.to_string()),
            ["node", node_name, "=", kernel] => {
                let kernel = kernel
                    .parse::<KernelId>()
                    .map_err(|e| error(e.to_string()))?;
                nodes.push(Node {
                    id: NodeId(node_name.to_string()),
                    kernel,
                    inputs: BTreeMap::new(),
                    params: BTreeMap::new(),
                    domain: None,
                });
            }
            ["output", target] => {
                let (node, port) = target.split_once('.').ok_or_else(|| {
                    error(format!("output must be `node.port`, found `{target}`"))
                })?;
                if output
                    .replace((NodeId(node.to_string()), port.to_string()))
                    .is_some()
                {
                    return Err(error("`output` is declared twice".into()));
                }
            }
            [port, "<-", source] => {
                let node = nodes.last_mut().ok_or_else(|| {
                    error(format!("`{port} <- {source}` appears before any node"))
                })?;
                let binding = Binding::Now(parse_source(source));
                if node.inputs.insert(port.to_string(), binding).is_some() {
                    return Err(error(format!("port `{port}` is bound twice")));
                }
            }
            [port, "<~", source, "from", initial] => {
                let node = nodes.last_mut().ok_or_else(|| {
                    error(format!("`{port} <~ {source}` appears before any node"))
                })?;
                let binding = Binding::Previous {
                    source: parse_source(source),
                    initial: parse_source(initial),
                };
                if node.inputs.insert(port.to_string(), binding).is_some() {
                    return Err(error(format!("port `{port}` is bound twice")));
                }
            }
            [port, "<~", ..] => {
                return Err(error(format!(
                    "`{port} <~` needs `<~ node.port from source`: the previous value and where it starts"
                )));
            }
            [param, "=", numbers @ ..] if !numbers.is_empty() => {
                let node = nodes
                    .last_mut()
                    .ok_or_else(|| error(format!("`{param} = ...` appears before any node")))?;
                let value = parse_value(numbers).ok_or_else(|| {
                    error(format!(
                        "`{param}` needs one number, `true`/`false`, or two, three, or four floats, found `{}`",
                        numbers.join(" ")
                    ))
                })?;
                if node.params.insert(param.to_string(), value).is_some() {
                    return Err(error(format!("parameter `{param}` is set twice")));
                }
            }
            ["domain", width, height] => {
                let node = nodes
                    .last_mut()
                    .ok_or_else(|| error("`domain` appears before any node".into()))?;
                let parse = |text: &str| text.parse::<u32>().ok().filter(|v| *v > 0);
                let (Some(width), Some(height)) = (parse(width), parse(height)) else {
                    return Err(error("`domain` needs two positive integers".into()));
                };
                node.domain = Some(Domain { width, height });
            }
            _ => return Err(error(format!("cannot read `{content}`"))),
        }
    }
    Ok(Graph {
        name: name.ok_or(DocumentError {
            line: 1,
            message: "document must start with `graph <name>`".into(),
        })?,
        externals,
        nodes,
        output: output.ok_or(DocumentError {
            line: text.lines().count() as u32,
            message: "document must end with `output <node>.<port>`".into(),
        })?,
    })
}

fn parse_source(text: &str) -> Source {
    match text.split_once('.') {
        Some((producer, output)) => Source::Node {
            node: NodeId(producer.to_string()),
            port: output.to_string(),
        },
        None => Source::External(text.to_string()),
    }
}

fn parse_value(words: &[&str]) -> Option<Value> {
    let floats = || {
        words
            .iter()
            .map(|word| word.parse::<f32>().ok())
            .collect::<Option<Vec<f32>>>()
    };
    match words {
        ["true"] => Some(Value::Bool(true)),
        ["false"] => Some(Value::Bool(false)),
        [single] => {
            if single.contains('.') {
                single.parse::<f32>().ok().map(Value::Float)
            } else {
                single.parse::<i32>().ok().map(Value::Int)
            }
        }
        [_, _] => floats().map(|v| Value::Vec2([v[0], v[1]])),
        [_, _, _] => floats().map(|v| Value::Vec3([v[0], v[1], v[2]])),
        [_, _, _, _] => floats().map(|v| Value::Vec4([v[0], v[1], v[2], v[3]])),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT: &str = "
# a comment
graph demo
input photo
node gray = image.grayscale@1
  source <- photo
  amount = 0.5
node blocks = image.pixelate@1
  source <- gray.result
  size = 8
  tint = 1.0 0.5 0.25 1.0
  domain 64 32
output blocks.result
";

    #[test]
    fn documents_round_trip() {
        let graph = Graph::parse(DOCUMENT).unwrap();
        assert_eq!(graph.name, "demo");
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
        assert_eq!(graph.output, (NodeId("blocks".into()), "result".into()));
        let text = graph.to_string();
        assert_eq!(Graph::parse(&text).unwrap(), graph);
    }

    #[test]
    fn errors_name_the_line() {
        let error = Graph::parse("graph g\nnode a = Bad\noutput a.r").unwrap_err();
        assert_eq!(error.line, 2);
        let error = Graph::parse("graph g\n  x <- y\noutput a.r").unwrap_err();
        assert!(error.message.contains("before any node"), "{error}");
        let error = Graph::parse("graph g\nnode a = t.k@1\noutput a").unwrap_err();
        assert!(error.message.contains("node.port"), "{error}");
        let error = Graph::parse("node a = t.k@1\noutput a.r").unwrap_err();
        assert!(error.message.contains("graph <name>"), "{error}");
    }
}
