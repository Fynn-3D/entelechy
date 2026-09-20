use std::collections::BTreeMap;

use orla_graph::{Context, EvaluateError, Evaluator, Graph, GraphError, NodeId};
use orla_runtime::{ExecutionError, Runtime};

fn at(time: f32) -> Context {
    Context {
        time,
        ..Context::default()
    }
}

const GRADIENT: &str = "kernel image.gradient@1 { requires { field.write; } output result: Image; compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.0, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { field.read; field.write; } input source: Image; param amount: Float = 1.0; output result: Image; compute { let c = sample(source); let g = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return mix(c, vec4(g, g, g, c.a), amount); } }";
const INVERT: &str = "kernel image.invert@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a); } }";
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
    evaluator.set_fusion(false);
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
    assert_eq!(runtime.host().live(), 3);
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
    evaluator.set_fusion(false);
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
    assert_eq!(runtime.host().live(), 3, "replaced outputs were released");
}

#[test]
fn parameter_and_structure_changes_are_tracked() {
    let mut graph = Graph::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    evaluator.set_fusion(false);
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
    assert_eq!(runtime.host().live(), 2, "removed node released its image");

    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
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
        .load("kernel image.invert@1 { requires { field.read; field.write; } input source: Image; param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }")
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
        .load("kernel image.pulse@1 { requires { field.read; field.write; time.read; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(c.r * fract(time()), c.g, c.b, c.a); } }")
        .unwrap();
    let graph = Graph::parse("node source: image.gradient@1 { domain = vec2(2, 2); }\nnode pulse: image.pulse@1 { source <- source.result; }\noutput r <- pulse.result;").unwrap();
    let mut evaluator = Evaluator::new();
    evaluator.set_fusion(false);
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

const ZERO: &str = "kernel test.zero@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }";
const INCREMENT: &str = "kernel test.increment@1 { requires { field.read; field.write; } input previous: Image; param by: Float = 1.0; output result: Image; compute { let c = sample(previous); return vec4(c.r + by, c.g, c.b, 1.0); } }";
const COUNTER: &str = "
node seed: test.zero@1 { domain = vec2(2, 2); }
node count: test.increment@1 { previous <~ count.result from seed.result; }
output result <- count.result;
";

fn red(runtime: &Runtime, handle: orla_runtime::Handle) -> f32 {
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
            runtime.host().live(),
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
    assert_eq!(runtime.host().live(), 2);

    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}

#[test]
fn a_failed_evaluation_commits_nothing() {
    let mut runtime = Runtime::new(1024);
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    runtime
        .load("kernel test.gate@1 { requires { field.read; field.write; } input source: Image; param d: Int = 1; output result: Image; compute { let q = 1 / d; return sample(source) * float(q); } }")
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
        runtime.host().live() <= 4,
        "no leaked frames: {}",
        runtime.host().live()
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
        .load("kernel image.paint@1 { requires { field.read; field.write; input.read; } input source: Image; output result: Image; compute { let c = sample(source); let near = length(coord() - pointer()) < 1.5; let hit = if near && pointer_down() { 1.0 } else { 0.0 }; return vec4(hit, float(key(65)), c.b, c.a); } }")
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

fn pixels(runtime: &Runtime, handle: orla_runtime::Handle) -> Vec<[f32; 4]> {
    runtime.host().image(handle).unwrap().pixels().to_vec()
}

#[test]
fn fusion_absorbs_sole_pointwise_producers_and_changes_no_pixel() {
    let mut runtime = runtime();
    runtime
        .load("kernel image.pixelate@1 { requires { field.read; field.write; } input source: Image; param size: Int = 2; output result: Image; compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }")
        .unwrap();
    // gradient -> gray -> inv -> blocks: gray and inv are pointwise; blocks
    // fetches, so it stays separate and absorbs nothing.
    let graph = Graph::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node gray: image.grayscale@1 { source <- source.result; amount = 0.5; }
         node inv: image.invert@1 { source <- gray.result; }
         node blocks: image.pixelate@1 { source <- inv.result; }
         output result <- blocks.result;",
    )
    .unwrap();
    let mut plain = Evaluator::new();
    plain.set_fusion(false);
    let reference = plain
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(reference.fused.is_empty());
    let expected = pixels(&runtime, reference.output);
    plain.clear(&mut runtime);

    let mut fused = Evaluator::new();
    let report = fused
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(pixels(&runtime, report.output), expected, "bit-identical");
    assert_eq!(report.executed, ids(&["inv", "blocks"]));
    assert_eq!(
        report.fused,
        vec![(NodeId("inv".into()), ids(&["gray", "source"]))]
    );
    assert_eq!(
        runtime.host().live(),
        2,
        "only the fused group and blocks are materialized"
    );
    assert!(fused.cached(&NodeId("gray".into()), "result").is_none());
    assert!(
        runtime.kernel(&"fused.inv@1".parse().unwrap()).is_none(),
        "fused versions are not registered"
    );

    // Republishing an absorbed producer rebuilds the group.
    runtime.load(&GRAYSCALE.replace("0.2126", "1.0")).unwrap();
    let report = fused
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, ids(&["inv", "blocks"]));
    assert_ne!(pixels(&runtime, report.output), expected);
    let again = fused
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(again.executed.is_empty(), "the rebuilt group is cached");

    // Pinning materializes the node again; unpinning absorbs it.
    fused.pin(NodeId("gray".into()));
    let report = fused
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(
        report.fused,
        vec![(NodeId("gray".into()), ids(&["source"]))]
    );
    assert!(fused.cached(&NodeId("gray".into()), "result").is_some());
    fused.unpin(&NodeId("gray".into()));
    let report = fused
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.fused[0].1, ids(&["gray", "source"]));
    fused.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}

#[test]
fn outputs_state_sources_and_shared_producers_stay_materialized() {
    let mut runtime = runtime();
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    let graph = Graph::parse(
        "node seed: test.zero@1 { domain = vec2(2, 2); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node a: image.invert@1 { source <- count.result; }
         node b: image.invert@1 { source <- count.result; }
         node c: image.invert@1 { source <- a.result; }
         output result <- c.result;
         output other <- b.result;",
    )
    .unwrap();
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
    // count is a state source, so it is materialized; seed is its initial
    // source and count's `previous` counts as a reader, so seed stays too.
    // a is c's sole reader and pointwise: absorbed. b is an output: kept.
    assert_eq!(report.fused, vec![(NodeId("c".into()), ids(&["a"]))]);
    assert_eq!(report.executed, ids(&["seed", "count", "b", "c"]));
    assert_eq!(
        report.committed,
        vec![(NodeId("count".into()), "result".into())]
    );
    let second = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(
        red(&runtime, second.outputs[1].1),
        1.0 - 2.0,
        "state advanced through the fused frame"
    );
}

#[test]
fn fusion_prepares_on_the_worker_and_never_changes_a_pixel() {
    let mut runtime = runtime();
    runtime.set_background(true);
    let graph = Graph::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node gray: image.grayscale@1 { source <- source.result; amount = 0.5; }
         node inv: image.invert@1 { source <- gray.result; }
         output result <- inv.result;",
    )
    .unwrap();
    let mut plain = Evaluator::new();
    plain.set_fusion(false);
    let reference = plain
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    let expected = pixels(&runtime, reference.output);
    plain.clear(&mut runtime);

    // Every frame is right whether or not the fused versions have arrived.
    // The chain grows as they do, in whichever order the worker finishes
    // them: nothing, then gray with source or inv with gray, then
    // everything under inv; it never comes apart.
    let whole = vec![(NodeId("inv".into()), ids(&["gray", "source"]))];
    let tail = vec![(NodeId("gray".into()), ids(&["source"]))];
    let head = vec![(NodeId("inv".into()), ids(&["gray"]))];
    let mut evaluator = Evaluator::new();
    let mut fused_since = None;
    for frame in 0..2000 {
        let report = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &Context::default(),
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(pixels(&runtime, report.output), expected, "frame {frame}");
        if report.fused == whole {
            fused_since.get_or_insert(frame);
            if frame > fused_since.unwrap() + 2 {
                break;
            }
        } else {
            assert!(fused_since.is_none(), "a fused group does not come apart");
            assert!(
                report.fused.is_empty() || report.fused == tail || report.fused == head,
                "frame {frame}: {:?}",
                report.fused
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    assert!(
        fused_since.is_some(),
        "the worker delivers the fused version"
    );
    evaluator.clear(&mut runtime);
}

#[test]
fn reductions_produce_one_element_and_stay_out_of_fusion() {
    let mut runtime = runtime();
    runtime
        .load("kernel test.total@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Vec4>; compute { return sample(source); } merge(a, b) { return a + b; } }")
        .unwrap();
    runtime
        .load("kernel test.show@1 { requires { field.read; field.write; } input total: Rows<Vec4>; output result: Image; compute { let t = at(total, 0); return t / 24.0; } }")
        .unwrap();
    let graph = Graph::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node gray: image.grayscale@1 { source <- source.result; amount = 1.0; }
         node total: test.total@1 { source <- gray.result; }
         node show: test.show@1 { total <- total.result; domain = vec2(2, 2); }
         output result <- show.result;",
    )
    .unwrap();
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
    // gray absorbs source, but the reduction absorbs nothing and is not
    // absorbed by `show`, which reads it by address.
    assert_eq!(
        report.fused,
        vec![(NodeId("gray".into()), ids(&["source"]))]
    );
    let total = evaluator.cached(&NodeId("total".into()), "result").unwrap();
    assert_eq!(
        runtime.host().domain(total).unwrap(),
        orla_runtime::Domain::Rows(1)
    );
    let expected: [f32; 4] = pixels(
        &runtime,
        evaluator.cached(&NodeId("gray".into()), "result").unwrap(),
    )
    .iter()
    .fold([0.0; 4], |acc, p| std::array::from_fn(|k| acc[k] + p[k]));
    let shown = pixels(&runtime, report.output)[0];
    for k in 0..4 {
        assert!(
            (shown[k] * 24.0 - expected[k]).abs() < 1e-3,
            "{shown:?} vs {expected:?}"
        );
    }

    // A reduction with a domain is refused with the node named.
    let graph = Graph::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node total: test.total@1 { source <- source.result; domain = 1; }
         output result <- total.result;",
    )
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
    assert!(error.to_string().contains("remove its `domain`"), "{error}");
    evaluator.clear(&mut runtime);
}

#[test]
fn scans_follow_their_input_and_stay_out_of_fusion() {
    let mut runtime = runtime();
    runtime
        .load("kernel test.ramp@1 { requires { field.write; } output result: Rows<Int>; compute { return element() + 1; } }")
        .unwrap();
    runtime
        .load("kernel test.prefix@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with total = 0 { return total + sample(rows); } }")
        .unwrap();
    let graph = Graph::parse(
        "node ramp: test.ramp@1 { domain = 5; }
         node prefix: test.prefix@1 { rows <- ramp.result; }
         output result <- prefix.result;",
    )
    .unwrap();
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
    assert!(
        report.fused.is_empty(),
        "a scan is not fused: {:?}",
        report.fused
    );
    assert_eq!(
        runtime.host().domain(report.output).unwrap(),
        orla_runtime::Domain::Rows(5)
    );
    let totals: Vec<i32> = (0..5)
        .map(|i| runtime.host().element(report.output, i).unwrap()[0] as i32)
        .collect();
    assert_eq!(totals, [1, 3, 6, 10, 15]);

    // A domain other than the input's is refused by the backend.
    let graph = Graph::parse(
        "node ramp: test.ramp@1 { domain = 5; }
         node prefix: test.prefix@1 { rows <- ramp.result; domain = 3; }
         output result <- prefix.result;",
    )
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
        error.to_string().contains("one element per element"),
        "{error}"
    );
    evaluator.clear(&mut runtime);
}

#[test]
fn a_write_replaces_committed_state_and_is_checked() {
    let mut runtime = Runtime::new(1024);
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    let graph = Graph::parse(COUNTER).unwrap();
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    let count = NodeId("count".into());
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        evaluator
            .evaluate(&graph, runtime, &BTreeMap::new(), &ctx, &BTreeMap::new())
            .unwrap()
    };
    // Before any evaluation the port's layout is unknown.
    assert!(matches!(
        evaluator.write(&graph, &mut runtime, &count, "result", vec![0; 16]),
        Err(EvaluateError::UnknownOutput { .. })
    ));
    let report = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(red(&runtime, report.output), 1.0);
    // Two by two pixels of four words: red 40 everywhere.
    let mut words = Vec::new();
    for _ in 0..4 {
        words.extend([40.0f32, 0.0, 0.0, 1.0].map(f32::to_bits));
    }
    evaluator
        .write(&graph, &mut runtime, &count, "result", words)
        .unwrap();
    let report = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(
        red(&runtime, report.output),
        41.0,
        "the next frame counts on from what was written"
    );
    assert_eq!(runtime.host().live(), 2, "the replaced state was released");
    // Only state can be written, and only with the right number of words.
    assert!(matches!(
        evaluator.write(
            &graph,
            &mut runtime,
            &NodeId("seed".into()),
            "result",
            vec![0; 16]
        ),
        Err(EvaluateError::NotState { .. })
    ));
    assert!(matches!(
        evaluator.write(&graph, &mut runtime, &count, "result", vec![0; 15]),
        Err(EvaluateError::WordCount {
            expected: 16,
            found: 15,
            ..
        })
    ));
    evaluator.clear(&mut runtime);
}

#[test]
fn a_demanded_evaluation_runs_the_state_and_skips_the_rest() {
    let mut runtime = Runtime::new(1024);
    runtime.load(ZERO).unwrap();
    runtime.load(INCREMENT).unwrap();
    let graph = Graph::parse(
        "node seed: test.zero@1 { domain = vec2(2, 2); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node view: test.increment@1 { previous <- count.result; by = 100.0; }
         output result <- view.result;",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    let count = NodeId("count".into());
    let view = NodeId("view".into());
    let demand: std::collections::BTreeSet<NodeId> = [count.clone()].into_iter().collect();
    // The first frame produces the output once, since it never was.
    let report = evaluator
        .evaluate_for(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
            &demand,
        )
        .unwrap();
    assert!(report.executed.contains(&view), "{:?}", report.executed);
    assert_eq!(red(&runtime, report.output), 101.0);
    // The next frames advance the state and leave the picture as it was.
    for expected in 2..=3 {
        let report = evaluator
            .evaluate_for(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &ctx,
                &BTreeMap::new(),
                &demand,
            )
            .unwrap();
        assert_eq!(report.executed, vec![count.clone()], "frame {expected}");
        assert_eq!(report.committed, vec![(count.clone(), "result".into())]);
        assert_eq!(
            red(&runtime, report.output),
            101.0,
            "the picture kept its last value"
        );
        assert_eq!(
            red(&runtime, evaluator.cached(&count, "result").unwrap()),
            expected as f32
        );
    }
    // A full evaluation catches the picture up.
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &ctx,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, vec![count.clone(), view.clone()]);
    assert_eq!(red(&runtime, report.output), 104.0);
    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}
