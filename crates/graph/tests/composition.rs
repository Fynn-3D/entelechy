use std::collections::BTreeMap;

use orla_graph::{EvaluateError, Evaluator, Graph, GraphError, NodeId};
use orla_runtime::{ExecutionError, Runtime};

const GRADIENT: &str = "kernel image.gradient@1 { requires { image.write; } output result: Image; compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.0, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { image.read; image.write; } input source: Image; param amount: Float = 1.0; output result: Image; compute { let c = sample(source); let g = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return mix(c, vec4(g, g, g, c.a), amount); } }";
const INVERT: &str = "kernel image.invert@1 { requires { image.read; image.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a); } }";
const DOCUMENT: &str = "
graph demo
node source = image.gradient@1
  domain 3 3
node gray = image.grayscale@1
  source <- source.result
node final = image.invert@1
  source <- gray.result
output final.result
";

fn ids(names: &[&str]) -> Vec<NodeId> {
    names.iter().map(|n| NodeId((*n).into())).collect()
}

fn near(actual: [f32; 4], expected: [f32; 4]) {
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-4, "{actual:?} != {expected:?}");
    }
}

fn runtime() -> Runtime {
    let mut runtime = Runtime::new(1024);
    for source in [GRADIENT, GRAYSCALE, INVERT] {
        runtime.load(source).unwrap();
    }
    runtime
}

#[test]
fn graphs_bind_nodes_to_kernel_ids_and_reuse_unchanged_results() {
    let graph = Graph::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();
    assert_eq!(report.executed, ids(&["source", "gray", "final"]));
    let pixel = runtime.host().read_pixel(report.output, 2, 0).unwrap();
    near(pixel, [1.0 - 0.2126, 1.0 - 0.2126, 1.0 - 0.2126, 1.0]);
    assert_eq!(runtime.host().live_images(), 3);

    let again = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();
    assert!(again.executed.is_empty());
    assert_eq!(again.reused, ids(&["source", "gray", "final"]));
    assert_eq!(again.output, report.output);
}

#[test]
fn republishing_a_kernel_invalidates_only_its_dependents() {
    let graph = Graph::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();

    runtime
        .load(
            &GRAYSCALE
                .replace("0.2126", "1.0")
                .replace("0.7152", "0.0")
                .replace("0.0722", "0.0"),
        )
        .unwrap();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();
    assert_eq!(report.reused, ids(&["source"]));
    assert_eq!(report.executed, ids(&["gray", "final"]));
    near(
        runtime.host().read_pixel(report.output, 2, 0).unwrap(),
        [0.0, 0.0, 0.0, 1.0],
    );
    assert_eq!(
        runtime.host().live_images(),
        3,
        "replaced outputs were released"
    );
}

#[test]
fn parameter_and_structure_changes_are_tracked() {
    let mut graph = Graph::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();

    graph.nodes[1]
        .params
        .insert("amount".into(), orla_graph::Value::Int(0));
    let report = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();
    assert_eq!(report.executed, ids(&["gray", "final"]));
    near(
        runtime.host().read_pixel(report.output, 2, 0).unwrap(),
        [0.0, 1.0, 1.0, 1.0],
    );

    graph.nodes.pop();
    graph.output = (NodeId("gray".into()), "result".into());
    let report = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();
    assert!(report.executed.is_empty());
    assert_eq!(
        runtime.host().live_images(),
        2,
        "removed node released its image"
    );

    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live_images(), 0);
}

#[test]
fn externals_feed_graph_inputs() {
    let graph = Graph::parse(
        "graph g\ninput photo\nnode inv = image.invert@1\n  source <- photo\noutput inv.result",
    )
    .unwrap();
    let mut runtime = runtime();
    let photo = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(photo, 0, 0, [1.0, 0.0, 0.5, 1.0])
        .unwrap();
    let mut evaluator = Evaluator::new();
    let error = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::MissingExternal { .. }),
        "{error}"
    );
    let externals = BTreeMap::from([("photo".to_string(), photo)]);
    let report = evaluator
        .evaluate(&graph, &mut runtime, &externals)
        .unwrap();
    near(
        runtime.host().read_pixel(report.output, 0, 0).unwrap(),
        [0.0, 1.0, 0.5, 1.0],
    );
}

#[test]
fn failures_are_attributed_and_last_good_results_survive() {
    let graph = Graph::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    let first = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap();

    runtime
        .load("kernel image.invert@1 { requires { image.read; image.write; } input source: Image; param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }")
        .unwrap();
    let error = evaluator
        .evaluate(&graph, &mut runtime, &BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(&error, EvaluateError::Execution { node, error: ExecutionError::Arithmetic { .. } } if node.0 == "final"),
        "{error}"
    );
    assert_eq!(
        evaluator.cached(&NodeId("final".into()), "result"),
        Some(first.output)
    );
    assert!(runtime.host().read_pixel(first.output, 0, 0).is_ok());

    let missing =
        Graph::parse("graph g\nnode a = image.missing@1\n  domain 1 1\noutput a.result").unwrap();
    let error = evaluator
        .evaluate(&missing, &mut runtime, &BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::UnknownKernel { .. }),
        "{error}"
    );
}

#[test]
fn structural_errors_are_reported_before_execution() {
    let cyclic = Graph::parse("graph g\nnode a = image.invert@1\n  source <- b.result\nnode b = image.invert@1\n  source <- a.result\noutput a.result").unwrap();
    assert!(matches!(cyclic.order(), Err(GraphError::Cycle(_))));
    let dangling =
        Graph::parse("graph g\nnode a = image.invert@1\n  source <- nowhere\noutput a.result")
            .unwrap();
    assert!(matches!(
        dangling.order(),
        Err(GraphError::UnknownSource { .. })
    ));
    let no_domain = Graph::parse("graph g\nnode a = image.gradient@1\noutput a.result").unwrap();
    let mut runtime = runtime();
    let error = Evaluator::new()
        .evaluate(&no_domain, &mut runtime, &BTreeMap::new())
        .unwrap_err();
    assert_eq!(error, EvaluateError::NoDomain(NodeId("a".into())));
    let bad_port = Graph::parse(
        "graph g\nnode a = image.gradient@1\n  domain 1 1\n  gain = 2\noutput a.result",
    )
    .unwrap();
    let error = Evaluator::new()
        .evaluate(&bad_port, &mut runtime, &BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::UnknownPort { .. }),
        "{error}"
    );
}
