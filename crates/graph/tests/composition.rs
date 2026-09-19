use std::collections::BTreeMap;

use orla_graph::{Context, EvaluateError, Evaluator, Graph, GraphError, NodeId};
use orla_runtime::{ExecutionError, Runtime};

fn at(time: f32) -> Context {
    Context {
        time,
        ..Context::default()
    }
}

const GRADIENT: &str = "kernel image.gradient@1 { requires { image.write; } output result: Image; compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.0, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { image.read; image.write; } input source: Image; param amount: Float = 1.0; output result: Image; compute { let c = sample(source); let g = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return mix(c, vec4(g, g, g, c.a), amount); } }";
const INVERT: &str = "kernel image.invert@1 { requires { image.read; image.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a); } }";
const DOCUMENT: &str = "
node source: image.gradient@1 { domain = vec2(3, 3); }
node gray: image.grayscale@1 { source <- source.result; }
node final: image.invert@1 { source <- gray.result; }
output result <- final.result;
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, ids(&["source", "gray", "final"]));
    let pixel = runtime.host().read_pixel(report.output, 2, 0).unwrap();
    near(pixel, [1.0 - 0.2126, 1.0 - 0.2126, 1.0 - 0.2126, 1.0]);
    assert_eq!(runtime.host().live_images(), 3);
    assert_eq!(report.outputs[0].0, "result");

    let again = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();

    graph.nodes[1].params.insert(
        "amount".into(),
        orla_graph::ParamValue::Literal(orla_graph::Value::Int(0)),
    );
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, ids(&["gray", "final"]));
    near(
        runtime.host().read_pixel(report.output, 2, 0).unwrap(),
        [0.0, 1.0, 1.0, 1.0],
    );

    graph.nodes.pop();
    graph.outputs = vec![(
        "result".into(),
        orla_graph::Source::Node {
            node: NodeId("gray".into()),
            port: "result".into(),
        },
    )];
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
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
        "input photo: Image;\nnode inv: image.invert@1 { source <- photo; }\noutput result <- inv.result;",
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::MissingExternal { .. }),
        "{error}"
    );
    let externals = BTreeMap::from([("photo".to_string(), photo)]);
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &externals,
            &Context::default(),
            &BTreeMap::new(),
        )
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
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();

    runtime
        .load("kernel image.invert@1 { requires { image.read; image.write; } input source: Image; param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }")
        .unwrap();
    let error = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
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
        Graph::parse("node a: image.missing@1 { domain = vec2(1, 1); }\noutput r <- a.result;")
            .unwrap();
    let error = evaluator
        .evaluate(
            &missing,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::UnknownKernel { .. }),
        "{error}"
    );
}

#[test]
fn structural_errors_are_reported_before_execution() {
    let cyclic = Graph::parse("node a: image.invert@1 { source <- b.result; }\nnode b: image.invert@1 { source <- a.result; }\noutput r <- a.result;").unwrap();
    assert!(matches!(cyclic.order(), Err(GraphError::Cycle(_))));
    let dangling =
        Graph::parse("node a: image.invert@1 { source <- nowhere; }\noutput r <- a.result;")
            .unwrap();
    assert!(matches!(
        dangling.order(),
        Err(GraphError::UnknownSource { .. })
    ));
    let no_output = Graph::parse("node a: image.gradient@1 { domain = vec2(1, 1); }").unwrap();
    assert!(matches!(no_output.order(), Err(GraphError::NoOutput)));
    let no_domain = Graph::parse("node a: image.gradient@1 { }\noutput r <- a.result;").unwrap();
    let mut runtime = runtime();
    let error = Evaluator::new()
        .evaluate(
            &no_domain,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert_eq!(error, EvaluateError::NoDomain(NodeId("a".into())));
    let bad_port = Graph::parse(
        "node a: image.gradient@1 { domain = vec2(1, 1); gain = 2; }\noutput r <- a.result;",
    )
    .unwrap();
    let error = Evaluator::new()
        .evaluate(
            &bad_port,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert!(
        matches!(error, EvaluateError::UnknownPort { .. }),
        "{error}"
    );
}

#[test]
fn only_clock_reading_nodes_re_execute_when_time_advances() {
    let mut runtime = runtime();
    runtime
        .load("kernel image.pulse@1 { requires { image.read; image.write; time.read; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(c.r * fract(time()), c.g, c.b, c.a); } }")
        .unwrap();
    let graph = Graph::parse("node source: image.gradient@1 { domain = vec2(2, 2); }\nnode pulse: image.pulse@1 { source <- source.result; }\noutput r <- pulse.result;").unwrap();
    let mut evaluator = Evaluator::new();
    let first = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.25),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(first.executed, ids(&["source", "pulse"]));
    let second = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.5),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(second.reused, ids(&["source"]));
    assert_eq!(second.executed, ids(&["pulse"]));
    near(
        runtime.host().read_pixel(second.output, 1, 0).unwrap(),
        [0.5, 0.0, 0.0, 1.0],
    );
    let third = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.5),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(third.executed.is_empty(), "same time, nothing re-executes");
}

const ZERO: &str = "kernel test.zero@1 { requires { image.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }";
const INCREMENT: &str = "kernel test.increment@1 { requires { image.read; image.write; } input previous: Image; param by: Float = 1.0; output result: Image; compute { let c = sample(previous); return vec4(c.r + by, c.g, c.b, 1.0); } }";
const COUNTER: &str = "
node seed: test.zero@1 { domain = vec2(2, 2); }
node count: test.increment@1 { previous <~ count.result from seed.result; }
output result <- count.result;
";

fn red(runtime: &Runtime, handle: orla_runtime::ImageHandle) -> f32 {
    runtime.host().read_pixel(handle, 1, 1).unwrap()[0]
}

#[test]
fn feedback_bindings_read_committed_state_and_advance_per_evaluation() {
    let mut runtime = Runtime::new(1024);
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    let graph = Graph::parse(COUNTER).unwrap();
    assert_eq!(Graph::parse(&graph.to_string()).unwrap(), graph);
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    for expected in 1..=3 {
        let report = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &ctx,
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(red(&runtime, report.output), expected as f32);
        assert_eq!(
            report.committed,
            vec![(NodeId("count".into()), "result".into())]
        );
        assert_eq!(
            runtime.host().live_images(),
            2,
            "seed and the committed frame only"
        );
    }

    // A replaced Kernel keeps the state it inherits.
    runtime
        .load(&INCREMENT.replace("param by: Float = 1.0", "param by: Float = 10.0"))
        .unwrap();
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(red(&runtime, report.output), 13.0);

    // Reset restarts from the initial source.
    evaluator.reset_state(&mut runtime);
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(red(&runtime, report.output), 10.0);
    assert_eq!(runtime.host().live_images(), 2);

    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live_images(), 0);
}

#[test]
fn a_failed_evaluation_commits_nothing() {
    let mut runtime = Runtime::new(1024);
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    runtime
        .load("kernel test.gate@1 { requires { image.read; image.write; } input source: Image; param d: Int = 1; output result: Image; compute { let q = 1 / d; return sample(source) * float(q); } }")
        .unwrap();
    let mut graph = Graph::parse(&format!(
        "{COUNTER}node gate: test.gate@1 {{ source <- count.result; d = 1; }}\n"
    ))
    .unwrap();
    graph.outputs = vec![(
        "result".into(),
        orla_graph::Source::Node {
            node: NodeId("gate".into()),
            port: "result".into(),
        },
    )];
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(red(&runtime, report.output), 1.0);

    graph.nodes[2].params.insert(
        "d".into(),
        orla_graph::ParamValue::Literal(orla_graph::Value::Int(0)),
    );
    for _ in 0..2 {
        let error = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &ctx,
                &BTreeMap::new(),
            )
            .unwrap_err();
        assert!(matches!(error, EvaluateError::Execution { .. }), "{error}");
    }
    graph.nodes[2].params.insert(
        "d".into(),
        orla_graph::ParamValue::Literal(orla_graph::Value::Int(1)),
    );
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(
        red(&runtime, report.output),
        2.0,
        "state advanced exactly once past the failures"
    );
    assert!(
        runtime.host().live_images() <= 4,
        "no leaked frames: {}",
        runtime.host().live_images()
    );
}

#[test]
fn feedback_edges_are_not_dependencies_and_must_name_nodes() {
    let ring = Graph::parse("node a: test.increment@1 { previous <~ b.result from seed.result; }\nnode b: test.increment@1 { previous <~ a.result from seed.result; }\nnode seed: test.zero@1 { domain = vec2(1, 1); }\noutput r <- a.result;").unwrap();
    assert!(ring.order().is_ok(), "mutual feedback is not a cycle");
    let error = Graph::parse("input photo: Image;\nnode a: test.increment@1 { previous <~ photo from photo; }\noutput r <- a.result;").unwrap_err();
    assert!(error.message.contains("carry no state"), "{error}");
    let error =
        Graph::parse("node a: test.increment@1 { previous <~ a.result; }\noutput r <- a.result;")
            .unwrap_err();
    assert!(error.message.contains("`from`"), "{error}");
}

#[test]
fn input_reading_nodes_re_execute_when_input_changes() {
    let mut runtime = runtime();
    runtime
        .load("kernel image.paint@1 { requires { image.read; image.write; input.read; } input source: Image; output result: Image; compute { let c = sample(source); let near = length(coord() - pointer()) < 1.5; let hit = if near && pointer_down() { 1.0 } else { 0.0 }; return vec4(hit, float(key(65)), c.b, c.a); } }")
        .unwrap();
    let graph = Graph::parse("node source: image.gradient@1 { domain = vec2(4, 4); }\nnode paint: image.paint@1 { source <- source.result; }\noutput r <- paint.result;").unwrap();
    let mut evaluator = Evaluator::new();
    let mut context = Context::default();
    let first = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &context,
            &BTreeMap::new(),
        )
        .unwrap();
    near(
        runtime.host().read_pixel(first.output, 2, 2).unwrap(),
        [0.0, 0.0, 0.0, 1.0],
    );
    let again = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &context,
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(again.executed.is_empty());

    context.input.pointer = [2.0, 2.0];
    context.input.pointer_down = true;
    context.input.set_key(65, true);
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &context,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, ids(&["paint"]));
    near(
        runtime.host().read_pixel(report.output, 2, 2).unwrap(),
        [1.0, 1.0, 0.0, 1.0],
    );
    near(
        runtime.host().read_pixel(report.output, 0, 0).unwrap(),
        [0.0, 1.0, 0.0, 1.0],
    );
}

#[test]
fn graph_params_feed_nodes_and_can_be_overridden() {
    let mut runtime = runtime();
    let graph = Graph::parse(
        "param amount: Float = 1.0;\nnode source: image.gradient@1 { domain = vec2(2, 2); }\nnode gray: image.grayscale@1 { source <- source.result; amount = amount; }\noutput r <- gray.result;",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    let first = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    let gray = runtime.host().read_pixel(first.output, 1, 0).unwrap();
    assert!(
        (gray[0] - gray[1]).abs() < 1e-5,
        "default 1.0 fully desaturates"
    );
    let overrides = BTreeMap::from([("amount".to_string(), orla_graph::Value::Int(0))]);
    let second = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &overrides,
        )
        .unwrap();
    assert_eq!(second.executed, ids(&["gray"]));
    near(
        runtime.host().read_pixel(second.output, 1, 0).unwrap(),
        [1.0, 0.0, 0.0, 1.0],
    );
    let missing = BTreeMap::from([("nope".to_string(), orla_graph::Value::Int(0))]);
    let third = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &missing,
        )
        .unwrap();
    assert_eq!(
        third.executed,
        ids(&["gray"]),
        "dropping the override reverts to the default"
    );
    let fourth = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &missing,
        )
        .unwrap();
    assert!(
        fourth.executed.is_empty(),
        "unknown overrides change nothing"
    );
}
