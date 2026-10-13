use std::collections::BTreeMap;

use entelechy_graph::{Context, EvaluateError, Evaluator, Graph, GraphError, NodeId};
use entelechy_kernel::{ExecutionError, Runtime};

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
    let mut runtime = cpu(1024);
    for source in [GRADIENT, GRAYSCALE, INVERT] {
        load(&mut runtime, source);
    }
    runtime
}

#[test]
fn graphs_bind_nodes_to_kernel_ids_and_reuse_unchanged_results() {
    let graph = entelechy_scene::parse(DOCUMENT).unwrap();
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
    let graph = entelechy_scene::parse(DOCUMENT).unwrap();
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

    load(
        &mut runtime,
        &GRAYSCALE
            .replace("0.2126", "1.0")
            .replace("0.7152", "0.0")
            .replace("0.0722", "0.0"),
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
    let mut graph = entelechy_scene::parse(DOCUMENT).unwrap();
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
        entelechy_graph::ParamValue::Literal(entelechy_graph::Value::Int(0)),
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
        entelechy_graph::Source::Node {
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
    let graph = entelechy_scene::parse(
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
    let graph = entelechy_scene::parse(DOCUMENT).unwrap();
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

    load(
        &mut runtime,
        "kernel image.invert@1 { requires { field.read; field.write; } input source: Image; param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }",
    );
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

    let missing = entelechy_scene::parse(
        "node a: image.missing@1 { domain = vec2(1, 1); }\noutput r <- a.result;",
    )
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
    let cyclic = entelechy_scene::parse("node a: image.invert@1 { source <- b.result; }\nnode b: image.invert@1 { source <- a.result; }\noutput r <- a.result;").unwrap();
    assert!(matches!(cyclic.order(), Err(GraphError::Cycle(_))));
    let dangling = entelechy_scene::parse(
        "node a: image.invert@1 { source <- nowhere; }\noutput r <- a.result;",
    )
    .unwrap();
    assert!(matches!(
        dangling.order(),
        Err(GraphError::UnknownSource { .. })
    ));
    // A name one letter out is suggested.
    let near = entelechy_scene::parse(
        "node source: image.gradient@1 { domain = vec2(1, 1); }\nnode a: image.invert@1 { source <- sorce.result; }\noutput r <- a.result;",
    )
    .unwrap();
    let message = near.order().unwrap_err().to_string();
    assert!(message.ends_with("did you mean `source`?"), "{message}");
    let typo_port = entelechy_scene::parse(
        "node a: image.gradient@1 { domain = vec2(1, 1); }\nnode b: image.grayscale@1 { sorce <- a.result; }\noutput r <- b.result;",
    )
    .unwrap();
    let message = Evaluator::new()
        .evaluate(
            &typo_port,
            &mut runtime(),
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err()
        .to_string();
    assert!(message.ends_with("did you mean `source`?"), "{message}");
    let typo_kernel = entelechy_scene::parse(
        "node a: image.gradiant@1 { domain = vec2(1, 1); }\noutput r <- a.result;",
    )
    .unwrap();
    let message = Evaluator::new()
        .evaluate(
            &typo_kernel,
            &mut runtime(),
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err()
        .to_string();
    assert!(
        message.ends_with("did you mean `image.gradient@1`?"),
        "{message}"
    );
    // The scene refuses a scene without output first; the graph still does.
    let mut no_output = entelechy_scene::parse(
        "node a: image.gradient@1 { domain = vec2(1, 1); }\noutput r <- a.result;",
    )
    .unwrap();
    no_output.outputs.clear();
    assert!(matches!(no_output.order(), Err(GraphError::NoOutput)));
    let no_domain =
        entelechy_scene::parse("node a: image.gradient@1 { }\noutput r <- a.result;").unwrap();
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
    let bad_port = entelechy_scene::parse(
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
    load(
        &mut runtime,
        "kernel image.pulse@1 { requires { field.read; field.write; time.read; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(c.r * fract(time()), c.g, c.b, c.a); } }",
    );
    let graph = entelechy_scene::parse("node source: image.gradient@1 { domain = vec2(2, 2); }\nnode pulse: image.pulse@1 { source <- source.result; }\noutput r <- pulse.result;").unwrap();
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

#[test]
fn a_node_whose_outputs_are_unchanged_spares_what_reads_them() {
    // A node that reads the clock runs again as time advances; within a
    // second its outputs are the same word for word, so what reads them
    // is not run again, and its fresh Fields are let go.
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel test.second@1 { requires { field.write; time.read; } output result: Image; compute { return vec4(floor(time()), 0.0, 0.0, 1.0); } }",
    );
    let graph = entelechy_scene::parse("node second: test.second@1 { domain = vec2(2, 2); }\nnode final: image.invert@1 { source <- second.result; }\noutput r <- final.result;").unwrap();
    let mut evaluator = Evaluator::new();
    evaluator.set_fusion(false);
    let mut frame = |time: f32| {
        evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &at(time),
                &BTreeMap::new(),
            )
            .unwrap()
    };
    let first = frame(0.25);
    assert_eq!(first.executed, ids(&["second", "final"]));
    let same = frame(0.5);
    assert_eq!(same.executed, ids(&["second"]));
    assert_eq!(same.reused, ids(&["final"]));
    assert_eq!(same.output, first.output);
    let later = frame(1.5);
    assert_eq!(later.executed, ids(&["second", "final"]));
    near(
        runtime.host().read_pixel(later.output, 0, 0).unwrap(),
        [0.0, 1.0, 1.0, 1.0],
    );
    assert_eq!(
        runtime.host().live(),
        2,
        "the fresh Fields that gave way were let go"
    );
}

#[test]
fn state_at_a_fixed_point_stops_advancing() {
    // A counter that adds nothing: its state is the same Field each
    // evaluation, so it does not advance and the counter is not run again.
    let mut runtime = runtime();
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse("node seed: test.zero@1 { domain = vec2(2, 2); }\nnode count: test.increment@1 { previous <~ count.result from seed.result; by = 0.0; }\noutput result <- count.result;").unwrap();
    let mut evaluator = Evaluator::new();
    evaluator.set_fusion(false);
    let mut frame = || {
        evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &Context::default(),
                &BTreeMap::new(),
            )
            .unwrap()
    };
    let first = frame();
    assert_eq!(first.executed, ids(&["seed", "count"]));
    let second = frame();
    assert_eq!(second.executed, ids(&["count"]));
    let third = frame();
    assert!(third.executed.is_empty(), "{third:?}");
    assert!(third.committed.is_empty(), "{third:?}");
}

const ZERO: &str = "kernel test.zero@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }";
const INCREMENT: &str = "kernel test.increment@1 { requires { field.read; field.write; } input previous: Image; param by: Float = 1.0; output result: Image; compute { let c = sample(previous); return vec4(c.r + by, c.g, c.b, 1.0); } }";
const COUNTER: &str = "
node seed: test.zero@1 { domain = vec2(2, 2); }
node count: test.increment@1 { previous <~ count.result from seed.result; }
output result <- count.result;
";

fn red(runtime: &Runtime, handle: entelechy_kernel::Handle) -> f32 {
    runtime.host().read_pixel(handle, 1, 1).unwrap()[0]
}

#[test]
fn feedback_bindings_read_committed_state_and_advance_per_evaluation() {
    let mut runtime = cpu(1024);
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse(COUNTER).unwrap();
    assert_eq!(
        entelechy_scene::parse(&entelechy_scene::print(&graph)).unwrap(),
        graph
    );
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
    load(
        &mut runtime,
        &INCREMENT.replace("param by: Float = 1.0", "param by: Float = 10.0"),
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
    let mut runtime = cpu(1024);
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    load(
        &mut runtime,
        "kernel test.gate@1 { requires { field.read; field.write; } input source: Image; param d: Int = 1; output result: Image; compute { let q = 1 / d; return sample(source) * float(q); } }",
    );
    let mut graph = entelechy_scene::parse(&format!(
        "{COUNTER}node gate: test.gate@1 {{ source <- count.result; d = 1; }}\n"
    ))
    .unwrap();
    graph.outputs = vec![(
        "result".into(),
        entelechy_graph::Source::Node {
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
        entelechy_graph::ParamValue::Literal(entelechy_graph::Value::Int(0)),
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
        entelechy_graph::ParamValue::Literal(entelechy_graph::Value::Int(1)),
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
    let ring = entelechy_scene::parse("node a: test.increment@1 { previous <~ b.result from seed.result; }\nnode b: test.increment@1 { previous <~ a.result from seed.result; }\nnode seed: test.zero@1 { domain = vec2(1, 1); }\noutput r <- a.result;").unwrap();
    assert!(ring.order().is_ok(), "mutual feedback is not a cycle");
    let error = entelechy_scene::parse("input photo: Image;\nnode a: test.increment@1 { previous <~ photo from photo; }\noutput r <- a.result;").unwrap_err();
    assert!(error.message.contains("carry no state"), "{error}");
    let error = entelechy_scene::parse(
        "node a: test.increment@1 { previous <~ a.result; }\noutput r <- a.result;",
    )
    .unwrap_err();
    assert!(error.message.contains("`from`"), "{error}");
}

#[test]
fn input_reading_nodes_re_execute_when_input_changes() {
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel image.paint@1 { requires { field.read; field.write; input.read; } input source: Image; output result: Image; compute { let c = sample(source); let near = length(coord() - pointer()) < 1.5; let hit = if near && pointer_down() { 1.0 } else { 0.0 }; return vec4(hit, float(key(65)), c.b, c.a); } }",
    );
    let graph = entelechy_scene::parse("node source: image.gradient@1 { domain = vec2(4, 4); }\nnode paint: image.paint@1 { source <- source.result; }\noutput r <- paint.result;").unwrap();
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
    let graph = entelechy_scene::parse(
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
    let overrides = BTreeMap::from([("amount".to_string(), entelechy_graph::Value::Int(0))]);
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
    let missing = BTreeMap::from([("nope".to_string(), entelechy_graph::Value::Int(0))]);
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

fn pixels(runtime: &Runtime, handle: entelechy_kernel::Handle) -> Vec<[f32; 4]> {
    runtime.host().image(handle).unwrap().pixels().to_vec()
}

#[test]
fn fusion_absorbs_sole_pointwise_producers_and_changes_no_pixel() {
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel image.pixelate@1 { requires { field.read; field.write; } input source: Image; param size: Int = 2; output result: Image; compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }",
    );
    // gradient -> gray -> inv -> blocks: gray and inv are pointwise; blocks
    // fetches, so it stays separate and absorbs nothing.
    let graph = entelechy_scene::parse(
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
    assert_eq!(
        fused.fused_into(&NodeId("gray".into())),
        Some(&NodeId("inv".into())),
        "gray runs folded into its reader"
    );
    assert_eq!(fused.fused_into(&NodeId("blocks".into())), None);
    assert!(
        runtime.kernel(&"fused.inv@1".parse().unwrap()).is_none(),
        "fused versions are not registered"
    );

    // Republishing an absorbed producer rebuilds the group.
    load(&mut runtime, &GRAYSCALE.replace("0.2126", "1.0"));
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

    // Pinning materializes the node; unpinning absorbs it.
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

/// How many instructions a module holds, nested blocks included.
fn instructions(module: &entelechy_ir::Module) -> usize {
    fn walk(body: &[entelechy_ir::Instruction]) -> usize {
        body.iter()
            .map(|instruction| {
                1 + instruction
                    .op
                    .blocks()
                    .iter()
                    .map(|block| walk(&block.body))
                    .sum::<usize>()
            })
            .sum()
    }
    module
        .functions
        .iter()
        .chain(Some(&module.entry))
        .map(|function| walk(&function.body))
        .sum()
}

/// A backend that refuses a module past a size, as a processor refuses a
/// module whose values pass its stack: it takes each Kernel alone, but
/// not a fused group of them.
struct Bounded {
    most: usize,
}

impl entelechy_kernel::Backend for Bounded {
    fn name(&self) -> &'static str {
        "bounded"
    }
    fn prepare(
        &self,
        module: &entelechy_ir::Validated,
    ) -> Result<Box<dyn entelechy_kernel::Prepared>, String> {
        let size = instructions(module);
        if size > self.most {
            return Err(format!("its {size} instructions pass {}", self.most));
        }
        entelechy_kernel::Backend::prepare(&entelechy_cpu::CpuBackend, module)
    }
    fn execute(
        &self,
        prepared: &dyn entelechy_kernel::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_abi::Invocation,
        scope: &mut entelechy_abi::Scope<'_>,
    ) -> Result<Vec<entelechy_abi::Handle>, ExecutionError> {
        entelechy_kernel::Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
}

#[test]
fn a_fused_group_the_backend_refuses_runs_unfused() {
    // The backend takes the largest of the three alone, and no more.
    let most = [GRADIENT, GRAYSCALE, INVERT]
        .iter()
        .map(|source| instructions(&entelechy_compiler::compile(source).unwrap()))
        .max()
        .unwrap();
    let mut runtime = entelechy_kernel::Runtime::with_backend(1024, Box::new(Bounded { most }));
    for source in [GRADIENT, GRAYSCALE, INVERT] {
        load(&mut runtime, source);
    }
    let graph = entelechy_scene::parse(DOCUMENT).unwrap();
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

    // Each member was taken alone, so the group the backend refuses runs
    // as its members, frame after frame, and gives the same picture; the
    // refusal of each group the planner tries (the whole, then smaller
    // ones) is reported once, naming the group and the backend's reason.
    let mut evaluator = Evaluator::new();
    let mut refused = Vec::new();
    for _ in 0..6 {
        let report = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &at(0.0),
                &BTreeMap::new(),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(report.fused.is_empty(), "{:?}", report.fused);
        assert_eq!(pixels(&runtime, report.output), expected);
        refused.extend(report.refused);
    }
    assert!(!refused.is_empty());
    for (k, group) in refused.iter().enumerate() {
        assert!(
            group.members.len() > 1 && group.members.contains(&group.node),
            "{group:?}"
        );
        assert!(group.reason.contains("instructions"), "{group}");
        assert!(group.to_string().contains("run apart"), "{group}");
        assert!(!refused[..k].contains(group), "said twice: {group}");
    }
    let later = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(later.refused.is_empty(), "{:?}", later.refused);
}

#[test]
fn outputs_state_sources_and_shared_producers_stay_materialized() {
    let mut runtime = runtime();
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse(
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
    // count is a state source, so it is materialized; seed stays as its
    // initial source. a is c's sole pointwise reader: absorbed. b is an
    // output: kept.
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
fn a_long_pointwise_chain_fuses_into_groups_of_bounded_size() {
    let mut runtime = runtime();
    // A gradient and 199 inversions: 200 pointwise nodes in a line.
    let mut scene = String::from("node n0: image.gradient@1 { domain = vec2(4, 2); }\n");
    for i in 1..200 {
        scene.push_str(&format!(
            "node n{i}: image.invert@1 {{ source <- n{}.result; }}\n",
            i - 1
        ));
    }
    scene.push_str("output result <- n199.result;\n");
    let graph = entelechy_scene::parse(&scene).unwrap();
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
    // 200 nodes in groups of at most 64: four groups, none larger.
    assert_eq!(report.executed.len(), 4, "{:?}", report.executed);
    assert!(
        report.fused.iter().all(|(_, absorbed)| absorbed.len() < 64),
        "{:?}",
        report
            .fused
            .iter()
            .map(|(_, a)| a.len())
            .collect::<Vec<_>>()
    );
}

#[test]
fn fusion_prepares_on_the_worker_and_never_changes_a_pixel() {
    let mut runtime = runtime();
    runtime.set_background(true);
    let graph = entelechy_scene::parse(
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

    // Every frame is right whatever fused versions have arrived. The chain
    // grows as the worker finishes them, in any order, and never splits.
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
    load(
        &mut runtime,
        "kernel test.total@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Vec4>; compute { return sample(source); } merge(a, b) { return a + b; } }",
    );
    load(
        &mut runtime,
        "kernel test.show@1 { requires { field.read; field.write; } input total: Rows<Vec4>; output result: Image; compute { let t = at(total, 0); return t / 24.0; } }",
    );
    let graph = entelechy_scene::parse(
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
    // gray absorbs source; the reduction absorbs nothing and is not
    // absorbed by `show`, which reads it by address.
    assert_eq!(
        report.fused,
        vec![(NodeId("gray".into()), ids(&["source"]))]
    );
    let total = evaluator.cached(&NodeId("total".into()), "result").unwrap();
    assert_eq!(
        runtime.host().domain(total).unwrap(),
        entelechy_kernel::Domain::Rows(1)
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

    // A reduction with a domain is refused, the node named.
    let graph = entelechy_scene::parse(
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
    load(
        &mut runtime,
        "kernel test.ramp@1 { requires { field.write; } output result: Rows<Int>; compute { return element() + 1; } }",
    );
    load(
        &mut runtime,
        "kernel test.prefix@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with total = 0 { return total + sample(rows); } }",
    );
    let graph = entelechy_scene::parse(
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
        entelechy_kernel::Domain::Rows(5)
    );
    let totals: Vec<i32> = (0..5)
        .map(|i| runtime.host().element(report.output, i).unwrap()[0] as i32)
        .collect();
    assert_eq!(totals, [1, 3, 6, 10, 15]);

    // A domain other than the input's is refused by the backend.
    let graph = entelechy_scene::parse(
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
    let mut runtime = cpu(1024);
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse(COUNTER).unwrap();
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
    // Only state is writable, and only with the right word count.
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
    let mut runtime = cpu(1024);
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse(
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
    // The first frame produces the output.
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
    // The next frames advance the state and keep the picture.
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

/// A remainder is a fraction that `ceil` and `floor` decide; a bare
/// fraction sizes nothing.
#[test]
fn a_side_from_a_size_divides_as_one_from_the_window() {
    let mut runtime = cpu(1 << 20);
    load(&mut runtime, GRADIENT);
    load(&mut runtime, GRAYSCALE);
    let context = Context {
        window: [101.0, 101.0],
        ..Context::default()
    };
    let side = |runtime: &mut entelechy_kernel::Runtime, domain: &str| {
        let graph = entelechy_scene::parse(&format!(
            "node source: image.gradient@1 {{ domain = window(); }}\nnode half: image.grayscale@1 {{ source <- source.result; amount = 1.0; domain = {domain}; }}\noutput r <- half.result;"
        ))
        .unwrap();
        let mut evaluator = Evaluator::new();
        let size = evaluator
            .evaluate(
                &graph,
                runtime,
                &BTreeMap::new(),
                &context,
                &BTreeMap::new(),
            )
            .map(|report| runtime.host().image_size(report.output).unwrap())
            .map_err(|e| e.to_string());
        evaluator.clear(runtime);
        size
    };
    assert_eq!(side(&mut runtime, "ceil(size(source) / 2)"), Ok((51, 51)));
    assert_eq!(side(&mut runtime, "ceil(window() / 2)"), Ok((51, 51)));
    assert_eq!(side(&mut runtime, "floor(size(source) / 2)"), Ok((50, 50)));
    assert_eq!(side(&mut runtime, "(size(source) + 1) / 2"), Ok((51, 51)));
    let bare = side(&mut runtime, "size(source) / 2").unwrap_err();
    assert!(bare.contains("floor"), "{bare}");
}

#[test]
fn a_domain_may_take_the_window_and_follows_it() {
    let mut runtime = runtime();
    let graph = entelechy_scene::parse(
        "node source: image.gradient@1 { domain = window() / scale(); }\noutput r <- source.result;",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    // No window: the node cannot be sized, and the error says what to pass.
    let error = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("no window"), "{error}");
    let mut context = Context {
        window: [8.0, 6.0],
        scale: 2.0,
        ..Context::default()
    };
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &context,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(
        runtime.host().image_size(report.output).unwrap(),
        (4, 3),
        "the window in logical pixels"
    );
    // The same window: nothing reruns. A resized one: the node runs at the
    // new extent, the graph unchanged.
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
    context.window = [16.0, 6.0];
    let resized = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &context,
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(resized.executed, ids(&["source"]));
    assert_eq!(runtime.host().image_size(resized.output).unwrap(), (8, 3));
}

#[test]
fn a_memory_fed_back_persists_across_evaluations() {
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel test.ticks@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }",
    );
    load(
        &mut runtime,
        "kernel test.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }",
    );
    load(
        &mut runtime,
        "kernel test.count@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { let v = at(ram, 0) + 1; put(ram, 0, v); return v; } }",
    );
    let graph = entelechy_scene::parse(
        "node ticks: test.ticks@1 { domain = 3; }
node zero: test.zero@1 { domain = 2; }
node machine: test.count@1 { ticks <- ticks.result; ram <~ machine.ram from zero.result; }
output r <- machine.result;",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    let first = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.0),
            &BTreeMap::new(),
        )
        .unwrap();
    let read = |runtime: &Runtime, handle: entelechy_kernel::Handle, i: u32| {
        runtime.host().element(handle, i).unwrap()[0]
    };
    // Three steps count to three within the frame...
    assert_eq!(read(&runtime, first.output, 2), 3);
    // ...and the memory carries the count into the next frame.
    let second = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(1.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(read(&runtime, second.output, 0), 4);
    assert_eq!(read(&runtime, second.output, 2), 6);
    let ram = evaluator.cached(&NodeId("machine".into()), "ram").unwrap();
    assert_eq!(read(&runtime, ram, 0), 6);
    assert_eq!(read(&runtime, ram, 1), 0);
    evaluator.reset_state(&mut runtime);
    let again = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(2.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(read(&runtime, again.output, 2), 3);
}

const TALLY_ZERO: &str = "kernel tally.zero@1 { requires { field.write; } struct Tally { count: Int; } output result: Rows<Tally>; compute { return Tally { count: 0 }; } }";
const TALLY_STEP_V1: &str = "kernel tally.step@1 { requires { field.read; field.write; } struct Tally { count: Int; } input previous: Rows<Tally>; output result: Rows<Tally>; compute { let t = sample(previous); return Tally { count: t.count + 1 }; } }";
const TALLY_STEP_V2: &str = "kernel tally.step@1 { requires { field.read; field.write; } struct Tally { count: Int; doubled: Int; } input previous: Rows<Tally>; output result: Rows<Tally>; compute { let t = sample(previous); return Tally { count: t.count + 1, doubled: (t.count + 1) * 2 }; } }";
const TALLY_UPGRADE: &str = "kernel tally.upgrade@1 { requires { field.read; field.write; } struct Old { count: Int; } struct Tally { count: Int; doubled: Int; } input old: Rows<Old>; output result: Rows<Tally>; compute { let o = sample(old); return Tally { count: o.count, doubled: o.count * 2 }; } }";
const TALLY_UPGRADE_WRONG: &str = "kernel tally.wrong@1 { requires { field.read; field.write; } struct Old { count: Int; } input old: Rows<Old>; output result: Rows<Old>; compute { return sample(old); } }";
const TALLY_SCENE: &str = "
node seed: tally.zero@1 { domain = 1; }
node count: tally.step@1 { previous <~ count.result from seed.result migrate tally.upgrade@1; }
output result <- count.result;
";

fn tally(runtime: &Runtime, handle: entelechy_kernel::Handle) -> Vec<i32> {
    runtime
        .host()
        .element(handle, 0)
        .unwrap()
        .iter()
        .map(|w| *w as i32)
        .collect()
}

#[test]
fn a_republished_kernels_new_state_layout_is_migrated_reset_or_refused() {
    let mut runtime = cpu(1024);
    for source in [
        TALLY_ZERO,
        TALLY_STEP_V1,
        TALLY_UPGRADE,
        TALLY_UPGRADE_WRONG,
    ] {
        load(&mut runtime, source);
    }
    let graph = entelechy_scene::parse(TALLY_SCENE).unwrap();
    assert_eq!(
        entelechy_scene::parse(&entelechy_scene::print(&graph)).unwrap(),
        graph
    );
    assert!(entelechy_scene::print(&graph).contains("migrate tally.upgrade@1"));
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime, graph: &Graph| {
        evaluator.evaluate(graph, runtime, &BTreeMap::new(), &ctx, &BTreeMap::new())
    };
    for _ in 0..3 {
        evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    }
    let report = evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    assert_eq!(tally(&runtime, report.output), vec![4]);

    // The Kernel grows a field: the migration carries the committed state
    // over, and the count goes on.
    load(&mut runtime, TALLY_STEP_V2);
    let report = evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    assert_eq!(tally(&runtime, report.output), vec![5, 10]);
    assert_eq!(
        report.migrated,
        vec![(
            NodeId("count".into()),
            "previous".into(),
            "tally.upgrade@1".parse().unwrap()
        )]
    );
    assert!(report.reset.is_empty());
    let report = evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    assert_eq!(tally(&runtime, report.output), vec![6, 12]);
    assert!(report.migrated.is_empty());

    // Back to the old layout with a wrong-layout migration: the evaluation
    // fails, the state stays, and the old Kernel reads it again.
    let wrong =
        entelechy_scene::parse(&TALLY_SCENE.replace("tally.upgrade@1", "tally.wrong@1")).unwrap();
    load(&mut runtime, TALLY_STEP_V1);
    let error = evaluate(&mut evaluator, &mut runtime, &wrong).unwrap_err();
    assert!(
        matches!(&error, EvaluateError::Migration { kernel, .. } if kernel.to_string() == "tally.wrong@1"),
        "{error}"
    );
    assert!(
        error.to_string().contains("the state stays as it was"),
        "{error}"
    );
    load(&mut runtime, TALLY_STEP_V2);
    let report = evaluate(&mut evaluator, &mut runtime, &wrong).unwrap();
    assert_eq!(
        tally(&runtime, report.output),
        vec![7, 14],
        "nothing was lost"
    );

    // Without a migration, a changed layout restarts the state from its
    // initial source, reported.
    let plain =
        entelechy_scene::parse(&TALLY_SCENE.replace(" migrate tally.upgrade@1", "")).unwrap();
    load(&mut runtime, TALLY_STEP_V1);
    let report = evaluate(&mut evaluator, &mut runtime, &plain).unwrap();
    assert_eq!(tally(&runtime, report.output), vec![1]);
    assert_eq!(report.reset.len(), 1);
    assert_eq!(report.reset[0].0, NodeId("count".into()));
    assert!(
        report.reset[0]
            .2
            .contains("expects an element of one integer (i)")
            && report.reset[0].2.contains("two integers"),
        "{}",
        report.reset[0].2
    );
    assert_eq!(
        runtime.host().live(),
        2,
        "seed and the committed state only"
    );
}

const TALLY_PEEK_V1: &str = "kernel tally.peek@1 { requires { field.read; field.write; } struct Tally { count: Int; } input previous: Rows<Tally>; output result: Rows<Int>; compute { return sample(previous).count; } }";
const TALLY_PEEK_V2: &str = "kernel tally.peek@1 { requires { field.read; field.write; } struct Tally { count: Int; doubled: Int; } input previous: Rows<Tally>; output result: Rows<Int>; compute { return sample(previous).count; } }";
/// A reader of the count's state before the count itself, naming no
/// migration.
const TALLY_PEEKED: &str = "
node seed: tally.zero@1 { domain = 1; }
node peek: tally.peek@1 { previous <~ count.result from seed.result; }
node count: tally.step@1 { previous <~ count.result from seed.result migrate tally.upgrade@1; }
output result <- count.result;
output seen <- peek.result;
";

/// A state's migration is the state's: one reader naming it carries the
/// state for every reader, whichever meets the changed layout first; two
/// readers naming different ones are refused.
#[test]
fn a_states_migration_is_named_by_any_of_its_readers() {
    let mut runtime = cpu(1024);
    for source in [
        TALLY_ZERO,
        TALLY_STEP_V1,
        TALLY_PEEK_V1,
        TALLY_UPGRADE,
        TALLY_UPGRADE_WRONG,
    ] {
        load(&mut runtime, source);
    }
    let graph = entelechy_scene::parse(TALLY_PEEKED).unwrap();
    let mut evaluator = Evaluator::new();
    let ctx = Context::default();
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime, graph: &Graph| {
        evaluator.evaluate(graph, runtime, &BTreeMap::new(), &ctx, &BTreeMap::new())
    };
    for _ in 0..3 {
        evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    }
    load(&mut runtime, TALLY_STEP_V2);
    load(&mut runtime, TALLY_PEEK_V2);
    let report = evaluate(&mut evaluator, &mut runtime, &graph).unwrap();
    assert!(report.reset.is_empty(), "{:?}", report.reset);
    assert_eq!(
        tally(&runtime, report.output),
        vec![4, 8],
        "carried, not restarted"
    );
    assert_eq!(report.migrated.len(), 1);

    let two = entelechy_scene::parse(&TALLY_PEEKED.replace(
        "previous <~ count.result from seed.result; }",
        "previous <~ count.result from seed.result migrate tally.wrong@1; }",
    ))
    .unwrap();
    let error = evaluate(&mut evaluator, &mut runtime, &two).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("is migrated by `tally.wrong@1` and by `tally.upgrade@1`")
            || error
                .to_string()
                .contains("is migrated by `tally.upgrade@1` and by `tally.wrong@1`"),
        "{error}"
    );
}

const TALLY_FAILS_AT_5: &str = "kernel tally.check@1 { requires { field.read; field.write; } struct Tally { count: Int; doubled: Int; } input tally: Rows<Tally>; output result: Rows<Int>; compute { let t = sample(tally); return 100 / (t.count - 5); } }";

/// The replaced state is put back for the Kernel that wrote it; when one
/// of several ports fails to migrate, the others are put back too.
#[test]
fn a_failed_evaluation_puts_back_the_state_a_migration_replaced() {
    let mut runtime = cpu(1024);
    for source in [
        TALLY_ZERO,
        TALLY_STEP_V1,
        TALLY_UPGRADE,
        TALLY_UPGRADE_WRONG,
        TALLY_FAILS_AT_5,
    ] {
        load(&mut runtime, source);
    }
    let ctx = Context::default();
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime, graph: &Graph| {
        evaluator.evaluate(graph, runtime, &BTreeMap::new(), &ctx, &BTreeMap::new())
    };
    let plain = entelechy_scene::parse(TALLY_SCENE).unwrap();
    let checked = entelechy_scene::parse(&format!(
        "{TALLY_SCENE}node check: tally.check@1 {{ tally <- count.result; }}\noutput checked <- check.result;\n"
    ))
    .unwrap();
    let mut evaluator = Evaluator::new();
    for _ in 0..4 {
        evaluate(&mut evaluator, &mut runtime, &plain).unwrap();
    }
    // The count is 4; migrated, stepped to 5, the check fails: nothing of
    // this evaluation stands.
    load(&mut runtime, TALLY_STEP_V2);
    let error = evaluate(&mut evaluator, &mut runtime, &checked).unwrap_err();
    assert!(error.to_string().contains("check"), "{error}");
    // The Kernel that wrote the state reads 4 and counts on.
    load(&mut runtime, TALLY_STEP_V1);
    let report = evaluate(&mut evaluator, &mut runtime, &plain).unwrap();
    assert!(report.reset.is_empty(), "{:?}", report.reset);
    assert_eq!(tally(&runtime, report.output), vec![5]);

    // Two ports: the second's migration gives the wrong layout. Both stay.
    let two = entelechy_scene::parse(
        "node seed: tally.zero@1 { domain = 1; }
node a: tally.step@1 { previous <~ a.result from seed.result migrate tally.upgrade@1; }
node b: tally.step@1 { previous <~ b.result from seed.result migrate tally.wrong@1; }
output a <- a.result;
output b <- b.result;
",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    for _ in 0..3 {
        evaluate(&mut evaluator, &mut runtime, &two).unwrap();
    }
    load(&mut runtime, TALLY_STEP_V2);
    let error = evaluate(&mut evaluator, &mut runtime, &two).unwrap_err();
    assert!(
        matches!(&error, EvaluateError::Migration { node, .. } if node.0 == "b"),
        "{error}"
    );
    load(&mut runtime, TALLY_STEP_V1);
    let report = evaluate(&mut evaluator, &mut runtime, &two).unwrap();
    assert!(report.reset.is_empty(), "{:?}", report.reset);
    let counts: Vec<Vec<i32>> = report
        .outputs
        .iter()
        .map(|(_, handle)| tally(&runtime, *handle))
        .collect();
    assert_eq!(counts, vec![vec![4], vec![4]]);
}

const COUNT: &str = "kernel test.count@1 { requires { field.read; field.write; } input previous: Image; output result: Image; compute { let c = sample(previous); return vec4(c.r + 1.0, c.g, c.b, c.a); } }";
const OFF_FRAME: &str = "
node source: image.gradient@1 { domain = vec2(2, 2); }
node count: test.count@1 { previous <~ count.result from source.result; async = true; }
node final: image.invert@1 { source <- count.result; }
output result <- final.result;
";

#[test]
fn an_async_node_runs_off_the_frame_and_the_frame_carries_its_last_output() {
    let graph = entelechy_scene::parse(OFF_FRAME).unwrap();
    assert!(entelechy_scene::print(&graph).contains("async = true;"));
    let mut runtime = runtime();
    load(&mut runtime, COUNT);
    let mut evaluator = Evaluator::new();
    let none = BTreeMap::new();
    let defaults = BTreeMap::new();
    let red = |runtime: &Runtime, handle| runtime.host().read_pixel(handle, 0, 0).unwrap()[0];
    // The first run is on the frame, so `final` has something to read.
    let first = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(0.0), &defaults)
        .unwrap();
    assert_eq!(first.executed, ids(&["source", "count", "final"]));
    assert!(first.pending.is_empty());
    assert_eq!(
        first.committed,
        vec![(NodeId("count".into()), "result".into())]
    );
    assert_eq!(red(&runtime, first.output), 1.0 - 1.0);
    // The state advanced, so the count reruns off the frame; its reader
    // keeps the last output, and the state stays.
    let second = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(1.0), &defaults)
        .unwrap();
    assert_eq!(second.pending, ids(&["count"]));
    assert!(second.executed.is_empty());
    assert_eq!(second.reused, ids(&["source", "final"]));
    assert!(second.committed.is_empty());
    assert!(evaluator.has_running());
    assert_eq!(red(&runtime, second.output), 0.0);
    // Settling waits for the run; the next evaluation reads it, reruns the
    // reader, and commits the state.
    assert_eq!(evaluator.settle(&mut runtime).unwrap(), ids(&["count"]));
    assert!(!evaluator.has_running());
    let third = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(2.0), &defaults)
        .unwrap();
    assert!(third.pending.is_empty());
    assert_eq!(third.executed, ids(&["final"]));
    assert_eq!(
        third.committed,
        vec![(NodeId("count".into()), "result".into())]
    );
    assert_eq!(red(&runtime, third.output), 1.0 - 2.0);
    let fourth = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(3.0), &defaults)
        .unwrap();
    assert_eq!(fourth.pending, ids(&["count"]));
    evaluator.settle(&mut runtime).unwrap();
    let fifth = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(4.0), &defaults)
        .unwrap();
    assert!(
        fifth.landed.is_empty(),
        "settled runs landed before the evaluation"
    );
    let sixth = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(5.0), &defaults)
        .unwrap();
    assert_eq!(sixth.pending, ids(&["count"]));
    while runtime.runs_in_flight() > 0 {
        std::thread::yield_now();
    }
    let seventh = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(6.0), &defaults)
        .unwrap();
    assert_eq!(seventh.landed, ids(&["count"]));
    assert_eq!(seventh.executed, ids(&["final"]));
    assert_eq!(red(&runtime, seventh.output), 1.0 - 4.0);
    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}

const ZEROS: &str = "kernel test.zeros@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }";
const SPREAD: &str = "kernel test.spread@1 { requires { field.read; field.write; } input target: Rows<Int>; output result: Rows<Int>; struct Put { at: Int; value: Int; } scatter into target with sum { return Put { at: element() % 4, value: 1 }; } }";

#[test]
fn a_scatter_node_may_set_its_own_domain_and_still_writes_the_targets_extent() {
    let graph = entelechy_scene::parse(
        "node zeros: test.zeros@1 { domain = 4; }\nnode spread: test.spread@1 { target <- zeros.result; domain = 10; }\nnode plain: test.spread@1 { target <- zeros.result; }\noutput result <- spread.result;\noutput plain <- plain.result;\n",
    )
    .unwrap();
    let mut runtime = cpu(1024);
    load(&mut runtime, ZEROS);
    load(&mut runtime, SPREAD);
    let mut evaluator = Evaluator::new();
    let none = BTreeMap::new();
    let defaults = BTreeMap::new();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &none, &at(0.0), &defaults)
        .unwrap();
    let words = |handle| runtime.host().data(handle).unwrap().storage().to_vec();
    // Ten elements over a target of four: 3, 3, 2, 2.
    assert_eq!(words(report.output), vec![3, 3, 2, 2]);
    // Without a domain: once per element of the first input, the target.
    let plain = report
        .outputs
        .iter()
        .find(|(name, _)| name == "plain")
        .unwrap()
        .1;
    assert_eq!(words(plain), vec![1, 1, 1, 1]);
}

#[test]
fn a_filled_field_says_what_it_cannot_hold_and_what_it_needs() {
    // A Field of a type a parameter cannot hold is a Kernel's work.
    let error = entelechy_scene::parse(
        "node cells: Grid<Cell> { domain = vec2(2, 2); }\noutput result <- cells.result;\n",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("a Kernel fills it"), "{error}");
    let error =
        entelechy_scene::parse("node zeros: Rows<Float> { }\noutput result <- zeros.result;\n")
            .unwrap_err()
            .to_string();
    assert!(error.contains("needs a `domain`"), "{error}");
    // A filled Field is a Kernel of the document, so it takes a session.
    let error = entelechy_scene::parse(
        "node zeros: Rows<Float> { domain = 3; }\noutput result <- zeros.result;\n",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("through a session"), "{error}");
}

/// The last keeps its name and output, nothing between is made, and the
/// words match the unfused run.
#[test]
fn a_node_placed_over_its_own_output_runs_as_one_repeated_execution() {
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel test.step@1 { requires { field.read; field.write; } input previous: Grid<Float>; param by: Float = 1.0; output result: Grid<Float>; compute { return at(previous, icoord()) + by; } }",
    );
    load(
        &mut runtime,
        "kernel test.seed@1 { requires { field.write; } output result: Grid<Float>; compute { return coord().x; } }",
    );
    let source = "node seed: test.seed@1 { domain = vec2(4, 3); }
         node step: test.step@1 { times = 5; carry previous <- result; previous <- seed.result; by = 2.0; }
         output result <- step.result;";
    let graph = entelechy_scene::parse(source).unwrap();

    // Unfused, every time is its own execution: the reference.
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
    assert_eq!(reference.executed.len(), 6, "a seed and five times");
    let expected: Vec<f32> = runtime
        .host()
        .data(reference.output)
        .unwrap()
        .storage()
        .iter()
        .map(|word| f32::from_bits(*word))
        .collect();
    plain.clear(&mut runtime);

    let mut joined = Evaluator::new();
    let report = joined
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    let words: Vec<f32> = runtime
        .host()
        .data(report.output)
        .unwrap()
        .storage()
        .iter()
        .map(|word| f32::from_bits(*word))
        .collect();
    assert_eq!(words, expected, "the same words, however they were run");
    assert_eq!(
        report.executed,
        ids(&["seed", "step/5"]),
        "the last time executes and keeps its name"
    );
    assert_eq!(
        report.fused,
        vec![(
            NodeId("step/5".into()),
            ids(&["step/1", "step/2", "step/3", "step/4"])
        )],
        "the times before it are what it absorbed"
    );
    // The first column started at zero and added two five times.
    assert_eq!(words[0], 10.0);
    joined.clear(&mut runtime);

    // The same run written by hand joins the same way; a step read
    // elsewhere is materialized, splitting the run there.
    let watched = entelechy_scene::parse(
        "node seed: test.seed@1 { domain = vec2(4, 3); }
         node one: test.step@1 { previous <- seed.result; by = 2.0; }
         node two: test.step@1 { previous <- one.result; by = 2.0; }
         node three: test.step@1 { previous <- two.result; by = 2.0; }
         output result <- three.result;
         output middle <- one.result;",
    )
    .unwrap();
    let mut split = Evaluator::new();
    let report = split
        .evaluate(
            &watched,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(report.executed, ids(&["seed", "one", "three"]));
    assert_eq!(
        report.fused,
        vec![(NodeId("three".into()), ids(&["two"]))],
        "the run that is left joins the two after the watched one"
    );
    split.clear(&mut runtime);
}

const TALLY: &str = "kernel test.tally@1 { requires { field.write; time.read; } output result: Rows<Int>; compute { return 2 + int(time()); } }";
const RUN: &str = "kernel test.run@1 { requires { field.read; field.write; } input live: Rows<Int>; output result: Rows<Float>; compute { return float(element()) + float(at(live, 0)) * 100.0; } }";
const DECIDED: &str = "
node live: test.tally@1 { domain = 1; }
node rows: test.run@1 { live <- live.result; domain = value(live); }
output result <- rows.result;
";

#[test]
fn a_kernel_decides_how_many_elements_the_next_node_runs() {
    // `rows` is as long as the number `live` wrote this frame: the one
    // domain term that reads data.
    let graph = entelechy_scene::parse(DECIDED).unwrap();
    let mut runtime = cpu(1 << 16);
    for source in [TALLY, RUN] {
        load(&mut runtime, source);
    }
    let mut evaluator = Evaluator::new();
    let count = |runtime: &Runtime, handle| runtime.host().domain(handle).unwrap().count();
    let first =
        |runtime: &Runtime, handle| f32::from_bits(runtime.host().element(handle, 0).unwrap()[0]);

    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(count(&runtime, report.output), 2);
    assert_eq!(first(&runtime, report.output), 200.0);

    // The count changes, so the Field's length follows.
    let later = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(5.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(later.executed, ids(&["live", "rows"]));
    assert_eq!(count(&runtime, later.output), 7);
    assert_eq!(first(&runtime, later.output), 700.0);

    // Nothing moved, so nothing runs: the cache key includes the resolved
    // domain.
    let again = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(5.0),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(again.executed.is_empty(), "{:?}", again.executed);
}

#[test]
fn a_domain_that_reads_a_field_still_fuses_with_its_producer() {
    let graph = entelechy_scene::parse(
        "
node live: test.tally@1 { domain = 1; }
node rows: test.run@1 { live <- live.result; domain = value(live) * 3; }
node twice: test.twice@1 { source <- rows.result; }
output result <- twice.result;
",
    )
    .unwrap();
    let mut runtime = cpu(1 << 16);
    for source in [
        TALLY,
        RUN,
        "kernel test.twice@1 { requires { field.read; field.write; } input source: Rows<Float>; output result: Rows<Float>; compute { return sample(source) * 2.0; } }",
    ] {
        load(&mut runtime, source);
    }
    let mut evaluator = Evaluator::new();
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &at(0.0),
            &BTreeMap::new(),
        )
        .unwrap();
    // `twice` has no domain of its own: it runs over what `rows` made, and
    // absorbs it.
    assert_eq!(report.executed, ids(&["live", "twice"]));
    assert_eq!(report.fused, vec![(NodeId("twice".into()), ids(&["rows"]))]);
    assert_eq!(runtime.host().domain(report.output).unwrap().count(), 6);
    assert_eq!(
        f32::from_bits(runtime.host().element(report.output, 5).unwrap()[0]),
        (5.0 + 200.0) * 2.0
    );
}

/// A runtime on the reference backend, as a session opens one.
fn cpu(budget_words: usize) -> entelechy_kernel::Runtime {
    entelechy_kernel::Runtime::with_backend(budget_words, Box::new(entelechy_cpu::CpuBackend))
}

/// Compiles and publishes a Kernel as a session does: compiling is the
/// caller's step, the runtime takes validated IR.
fn load(
    runtime: &mut entelechy_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<entelechy_kernel::KernelVersion> {
    let module = entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

const SPLIT: &str = "kernel shade.split@1 { requires { field.read; field.write; } input source: Image; output color: Image; output depth: Grid<Float>; output glow: Grid<Float>; compute { let c = sample(source); return { depth: c.r * 0.5, glow: c.g, color: vec4(c.b, c.g, c.r, 1.0) }; } }";
const COMBINE: &str = "kernel shade.combine@1 { requires { field.read; field.write; } input d: Grid<Float>; input c: Image; output result: Image; compute { let v = sample(d); return sample(c) + vec4(v, v, v, 0.0); } }";

/// Evaluates `document` fused and unfused; returns the fused report and
/// both pictures.
fn fused_and_plain(document: &str) -> (entelechy_graph::Report, Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut runtime = runtime();
    load(&mut runtime, SPLIT);
    load(&mut runtime, COMBINE);
    let graph = entelechy_scene::parse(document).unwrap_or_else(|error| panic!("{error}"));
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
    let actual = pixels(&runtime, report.output);
    (report, actual, expected)
}

#[test]
fn a_producer_of_several_outputs_fuses_whole_into_its_only_reader() {
    // `combine` reads two of `split`'s three outputs, and nothing else
    // reads them: the three run as one Kernel and `split` writes nothing.
    let (report, actual, expected) = fused_and_plain(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node split: shade.split@1 { source <- source.result; }
         node show: shade.combine@1 { c <- split.color; d <- split.depth; }
         output result <- show.result;",
    );
    assert_eq!(actual, expected, "bit-identical");
    assert_eq!(report.executed, ids(&["show"]));
    assert_eq!(
        report.fused,
        vec![(NodeId("show".into()), ids(&["split", "source"]))]
    );
}

#[test]
fn a_producer_of_several_outputs_with_another_reader_stays_whole() {
    // `glow` is also a graph output: `split` writes its Fields and runs
    // alone; the picture is the same.
    let (report, actual, expected) = fused_and_plain(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node split: shade.split@1 { source <- source.result; }
         node show: shade.combine@1 { c <- split.color; d <- split.depth; }
         output result <- show.result;
         output glow <- split.glow;",
    );
    assert_eq!(actual, expected, "bit-identical");
    assert_eq!(report.executed, ids(&["split", "show"]));
    assert_eq!(report.outputs.len(), 2);
}

#[test]
fn a_map_of_several_outputs_repeats_and_keeps_the_last_times_outputs() {
    let mut runtime = runtime();
    load(
        &mut runtime,
        "kernel test.walk@1 { requires { field.read; field.write; } input previous: Grid<Float>; output result: Grid<Float>; output trail: Grid<Int>; compute { let v = at(previous, icoord()); return { trail: int(v * 10.0), result: v + 1.5 }; } }",
    );
    load(
        &mut runtime,
        "kernel test.seed@1 { requires { field.write; } output result: Grid<Float>; compute { return coord().x; } }",
    );
    let graph = entelechy_scene::parse(
        "node seed: test.seed@1 { domain = vec2(4, 3); }
         node walk: test.walk@1 { times = 3; carry previous <- result; previous <- seed.result; }
         output result <- walk.result;
         output trail <- walk.trail;",
    )
    .unwrap();
    let words = |runtime: &Runtime, report: &entelechy_graph::Report| -> Vec<Vec<u32>> {
        report
            .outputs
            .iter()
            .map(|(_, handle)| runtime.host().data(*handle).unwrap().storage().to_vec())
            .collect()
    };
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
    let expected = words(&runtime, &reference);
    plain.clear(&mut runtime);
    let mut joined = Evaluator::new();
    let report = joined
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(words(&runtime, &report), expected);
    assert_eq!(report.executed, ids(&["seed", "walk/3"]));
    // Three steps of 1.5 from x, and the trail the last step read.
    let trail: Vec<i32> = expected[1].iter().map(|w| *w as i32).collect();
    assert_eq!(&trail[..4], &[30, 40, 50, 60]);
}

const PACK: &str = "kernel image.pack@1 { requires { field.read; field.write; } input source: Image; output result: Grid<Vec4 as unorm8>; compute { return sample(source); } }";
const SHOW: &str = "kernel image.show@1 { requires { field.read; field.write; } input packed: Grid<Vec4 as unorm8>; output result: Image; compute { return sample(packed) * 2.0; } }";

#[test]
fn a_narrow_field_between_two_nodes_is_written_and_read_not_fused_away() {
    // The gradient stored as bytes: every pixel is its byte widened, so
    // nothing fuses across the rounding.
    let mut runtime = runtime();
    load(&mut runtime, PACK);
    load(&mut runtime, SHOW);
    let graph = entelechy_scene::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node pack: image.pack@1 { source <- source.result; }
         node show: image.show@1 { packed <- pack.result; }
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
    assert_eq!(
        report.executed,
        ids(&["pack", "show"]),
        "the gradient fuses into pack; pack does not fuse into show"
    );
    // Pixel (3, 1) of a 6 by 4 gradient: u = 3/5, v = 1/3, as bytes.
    let byte = |x: f32| (x * 255.0).round_ties_even() * entelechy_kernel::encoding::UNORM8 * 2.0;
    let pixel = runtime.host().read_pixel(report.output, 3, 1).unwrap();
    assert_eq!(pixel, [byte(0.6), byte(1.0 / 3.0), 0.0, 2.0]);
    let packed = evaluator.cached(&NodeId("pack".into()), "result").unwrap();
    assert_eq!(
        runtime.host().data(packed).unwrap().storage().len(),
        24,
        "twenty-four pixels of four bytes"
    );

    // A picture of floats is not a picture of bytes: the binding says so.
    let wide = entelechy_scene::parse(
        "node source: image.gradient@1 { domain = vec2(6, 4); }
         node show: image.show@1 { packed <- source.result; }
         output result <- show.result;",
    )
    .unwrap();
    let error = Evaluator::new()
        .evaluate(
            &wide,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("kept as unorm8"), "{error}");
}

const INC: &str = "kernel t.inc@1 { requires { field.read; field.write; } input previous: Rows<Int>; param by: Int = 1; output result: Rows<Int>; compute { return at(previous, 0) + by; } }";
const INT_ZERO: &str =
    "kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }";
const FAILS_AT: &str = "kernel t.fails@1 { requires { field.read; field.write; } input x: Rows<Int>; param at: Int = 5; output result: Rows<Int>; compute { return 100 / (at(x, 0) - at); } }";
const COPY: &str = "kernel t.copy@1 { requires { field.read; field.write; } input x: Rows<Int>; output result: Rows<Int>; compute { return at(x, 0); } }";

fn int_at(runtime: &Runtime, handle: entelechy_kernel::Handle) -> i32 {
    runtime.host().element(handle, 0).unwrap()[0] as i32
}

#[test]
fn a_failed_restore_leaves_no_run_going_and_commits_are_whole() {
    let mut runtime = cpu(1 << 12);
    for source in [INC, INT_ZERO, FAILS_AT, COPY] {
        load(&mut runtime, source);
    }
    let (none, params) = (BTreeMap::new(), BTreeMap::new());
    let ctx = Context::default();
    // An async reader of the count, and a check that fails at 5.
    let graph = entelechy_scene::parse(
        "node seed: t.zero@1 { domain = 1; }
node count: t.inc@1 { previous <~ count.result from seed.result; domain = 1; }
node slow: t.copy@1 { x <- count.result; async = true; domain = 1; }
node check: t.fails@1 { x <- count.result; domain = 1; }
output result <- check.result;
output slow <- slow.result;
",
    )
    .unwrap();
    let mut evaluator = Evaluator::new();
    for _ in 0..3 {
        evaluator
            .evaluate(&graph, &mut runtime, &none, &ctx, &params)
            .unwrap();
        evaluator.settle(&mut runtime).unwrap();
    }
    let four = runtime
        .host_mut()
        .adopt(
            entelechy_kernel::FieldData::from_values(
                entelechy_kernel::Schema::new(vec![entelechy_kernel::Scalar::Int]),
                entelechy_kernel::Domain::Rows(1),
                &[4],
            )
            .unwrap(),
        )
        .unwrap();
    let live = runtime.host().live();
    // Restored to 4, the trial counts 5 and the check divides by zero.
    let refused = evaluator.restore(
        &graph,
        &mut runtime,
        &none,
        &ctx,
        &params,
        vec![(NodeId("count".into()), "result".into(), four)],
    );
    assert!(refused.is_err());
    assert!(
        !evaluator.has_running(),
        "no run of the refused world goes on"
    );
    assert!(
        runtime.host().live() <= live,
        "what the restore brought is let go"
    );
    let report = evaluator
        .evaluate(&graph, &mut runtime, &none, &ctx, &params)
        .unwrap();
    let count = evaluator.cached(&NodeId("count".into()), "result").unwrap();
    assert_eq!(int_at(&runtime, count), 4, "on from the world that stood");
    let _ = report;

    // Duplicate ports in one restore are refused, every handle released.
    let make = |runtime: &mut Runtime, v: u32| {
        runtime
            .host_mut()
            .adopt(
                entelechy_kernel::FieldData::from_values(
                    entelechy_kernel::Schema::new(vec![entelechy_kernel::Scalar::Int]),
                    entelechy_kernel::Domain::Rows(1),
                    &[v],
                )
                .unwrap(),
            )
            .unwrap()
    };
    let before = runtime.host().live();
    let (a, b) = (make(&mut runtime, 1), make(&mut runtime, 2));
    let twice = evaluator.restore(
        &graph,
        &mut runtime,
        &none,
        &ctx,
        &params,
        vec![
            (NodeId("count".into()), "result".into(), a),
            (NodeId("count".into()), "result".into(), b),
        ],
    );
    assert!(twice.is_err());
    assert_eq!(runtime.host().live(), before, "no handle leaks");
}

/// A commit is whole: a state port with no output fails the evaluation
/// before any port advances.
#[test]
fn a_commit_advances_every_port_or_none() {
    let mut runtime = cpu(1 << 12);
    for source in [INC, INT_ZERO] {
        load(&mut runtime, source);
    }
    let graph = entelechy_scene::parse(
        "node seed: t.zero@1 { domain = 1; }
node a: t.inc@1 { previous <~ a.result from seed.result; domain = 1; }
node b: t.inc@1 { previous <~ a.bogus from seed.result; domain = 1; }
output result <- b.result;
",
    )
    .unwrap();
    let (none, params) = (BTreeMap::new(), BTreeMap::new());
    let mut evaluator = Evaluator::new();
    for _ in 0..3 {
        assert!(
            evaluator
                .evaluate(&graph, &mut runtime, &none, &Context::default(), &params)
                .is_err()
        );
    }
    assert!(evaluator.committed_states().is_empty(), "nothing advanced");
}

/// Two ports' states swapped by a restore whose trial fails are each put
/// back, neither released while the other needs it.
#[test]
fn a_restore_that_swaps_two_ports_and_fails_puts_both_back() {
    let mut runtime = cpu(1 << 12);
    for source in [INC, INT_ZERO, FAILS_AT] {
        load(&mut runtime, source);
    }
    let graph = entelechy_scene::parse(
        "node seed: t.zero@1 { domain = 1; }
node a: t.inc@1 { previous <~ a.result from seed.result; domain = 1; }
node b: t.inc@1 { previous <~ b.result from seed.result; by = 10; domain = 1; }
node check: t.fails@1 { x <- a.result; at = 21; domain = 1; }
output result <- check.result;
",
    )
    .unwrap();
    let (none, params) = (BTreeMap::new(), BTreeMap::new());
    let ctx = Context::default();
    let mut evaluator = Evaluator::new();
    for _ in 0..2 {
        evaluator
            .evaluate(&graph, &mut runtime, &none, &ctx, &params)
            .unwrap();
    }
    let committed: BTreeMap<String, entelechy_kernel::Handle> = evaluator
        .committed_states()
        .into_iter()
        .map(|(node, _, handle)| (node.0, handle))
        .collect();
    let (ha, hb) = (committed["a"], committed["b"]);
    assert_eq!((int_at(&runtime, ha), int_at(&runtime, hb)), (2, 20));
    // a takes b's 20, and the trial's 21 fails the check.
    let refused = evaluator.restore(
        &graph,
        &mut runtime,
        &none,
        &ctx,
        &params,
        vec![
            (NodeId("a".into()), "result".into(), hb),
            (NodeId("b".into()), "result".into(), ha),
        ],
    );
    assert!(refused.is_err());
    let after: BTreeMap<String, entelechy_kernel::Handle> = evaluator
        .committed_states()
        .into_iter()
        .map(|(node, _, handle)| (node.0, handle))
        .collect();
    assert_eq!(
        (int_at(&runtime, after["a"]), int_at(&runtime, after["b"])),
        (2, 20)
    );
    evaluator
        .evaluate(&graph, &mut runtime, &none, &ctx, &params)
        .unwrap();
}

const TWO_COUNTERS: &str = "
node seed: test.zero@1 { domain = vec2(2, 2); }
node a: test.increment@1 { previous <~ a.result from seed.result; }
node b: test.increment@1 { previous <~ b.result from seed.result; }
output a <- a.result;
output b <- b.result;
";

#[test]
fn one_port_starts_again_and_the_others_keep_their_state() {
    let mut runtime = cpu(1024);
    load(&mut runtime, ZERO);
    load(&mut runtime, INCREMENT);
    let graph = entelechy_scene::parse(TWO_COUNTERS).unwrap();
    let mut evaluator = Evaluator::new();
    let none = BTreeMap::new();
    let defaults = BTreeMap::new();
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        let report = evaluator
            .evaluate(&graph, runtime, &none, &Context::default(), &defaults)
            .unwrap();
        let counts: Vec<f32> = report
            .outputs
            .iter()
            .map(|(_, handle)| red(runtime, *handle))
            .collect();
        (report, counts)
    };
    for _ in 0..3 {
        evaluate(&mut evaluator, &mut runtime);
    }
    let a = NodeId("a".into());
    evaluator
        .reset_port(&graph, &mut runtime, &a, "result")
        .unwrap();
    let (report, counts) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(counts, vec![1.0, 4.0], "a started again, b counted on");
    assert_eq!(report.executed, ids(&["a", "b"]));
    assert_eq!(runtime.host().live(), 3, "the seed and the two states");

    // Only state resets: an output no `<~` reads holds none, and a refused
    // reset moves nothing.
    let seed = NodeId("seed".into());
    let error = evaluator
        .reset_port(&graph, &mut runtime, &seed, "result")
        .unwrap_err();
    assert!(matches!(error, EvaluateError::NotState { .. }), "{error}");
    let error = evaluator
        .reset_port(&graph, &mut runtime, &a, "nothing")
        .unwrap_err();
    assert!(matches!(error, EvaluateError::NotState { .. }), "{error}");
    let (_, counts) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(counts, vec![2.0, 5.0]);

    // A port not committed yet is left as it is.
    evaluator.reset_state(&mut runtime);
    evaluator
        .reset_port(&graph, &mut runtime, &a, "result")
        .unwrap();
    let (_, counts) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(counts, vec![1.0, 1.0]);
    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}

const TWO_OFF_FRAME: &str = "
node source: image.gradient@1 { domain = vec2(2, 2); }
node count: test.count@1 { previous <~ count.result from source.result; async = true; }
node shown: image.invert@1 { source <- count.result; async = true; }
node other: test.count@1 { previous <~ other.result from source.result; async = true; }
output result <- shown.result;
output other <- other.result;
";

#[test]
fn a_reset_lets_go_of_the_runs_that_began_from_the_state_it_forgot() {
    let graph = entelechy_scene::parse(TWO_OFF_FRAME).unwrap();
    let mut runtime = runtime();
    load(&mut runtime, COUNT);
    let mut evaluator = Evaluator::new();
    let none = BTreeMap::new();
    let defaults = BTreeMap::new();
    let red = |runtime: &Runtime, handle| runtime.host().read_pixel(handle, 0, 0).unwrap()[0];
    let mut frame = 0.0;
    let mut evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        frame += 1.0;
        let report = evaluator
            .evaluate(&graph, runtime, &none, &at(frame), &defaults)
            .unwrap();
        let values: Vec<f32> = report
            .outputs
            .iter()
            .map(|(_, handle)| red(runtime, *handle))
            .collect();
        (report, values)
    };
    // On the frame first, then both counts off it, from state 1.
    let (_, values) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(values, vec![1.0 - 1.0, 1.0]);
    let (report, _) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(report.pending, ids(&["count", "other"]));

    // One port resets: the run reading it is dropped; the other lands.
    let count = NodeId("count".into());
    evaluator
        .reset_port(&graph, &mut runtime, &count, "result")
        .unwrap();
    assert_eq!(evaluator.settle(&mut runtime).unwrap(), ids(&["other"]));
    // The count's first run read its initial source, so the frame after
    // the reset commits that cached 1, not a stale 2.
    let (report, values) = evaluate(&mut evaluator, &mut runtime);
    assert!(report.reused.contains(&count), "{report:?}");
    assert!(report.committed.contains(&(count.clone(), "result".into())));
    assert_eq!(values, vec![1.0 - 1.0, 2.0]);
    // It counts on from there.
    let (report, _) = evaluate(&mut evaluator, &mut runtime);
    assert!(report.pending.contains(&count), "{report:?}");
    evaluator.settle(&mut runtime).unwrap();
    evaluate(&mut evaluator, &mut runtime);
    evaluator.settle(&mut runtime).unwrap();
    let (_, values) = evaluate(&mut evaluator, &mut runtime);
    assert_eq!(values[0], 1.0 - 2.0, "{values:?}");
    assert!(values[1] > 2.0, "the other counted on: {values:?}");

    // Resetting all state drops every run: none lands afterwards.
    let (report, _) = evaluate(&mut evaluator, &mut runtime);
    assert!(!report.pending.is_empty());
    evaluator.reset_state(&mut runtime);
    assert!(!evaluator.has_running());
    evaluator.clear(&mut runtime);
    assert_eq!(runtime.host().live(), 0);
}

/// A chain of pointwise nodes fuses into its last node and only that one
/// fused version is prepared and kept: every node answers that it runs
/// in the last, and the plan is reused from frame to frame.
#[test]
fn a_fused_chain_is_one_version_run_by_its_last_node() {
    let mut runtime = runtime();
    let mut text = String::from("node n00: image.gradient@1 { domain = vec2(6, 4); }\n");
    for i in 1..32 {
        text.push_str(&format!(
            "node n{i:02}: image.invert@1 {{ source <- n{:02}.result; }}\n",
            i - 1
        ));
    }
    text.push_str("output result <- n31.result;\n");
    let graph = entelechy_scene::parse(&text).unwrap();
    let mut evaluator = Evaluator::new();
    for _ in 0..3 {
        evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &BTreeMap::new(),
                &Context::default(),
                &BTreeMap::new(),
            )
            .unwrap();
        for i in 1..31 {
            assert_eq!(
                evaluator.fused_into(&NodeId(format!("n{i:02}"))),
                Some(&NodeId("n31".into())),
                "n{i:02}"
            );
        }
    }
}

#[test]
fn a_plan_follows_fusion_turned_off_and_on() {
    let graph = entelechy_scene::parse(DOCUMENT).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    let summary = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        evaluator
            .plan_summary(&graph, runtime, &BTreeMap::new(), &BTreeMap::new())
            .unwrap()
            .into_iter()
            .map(|entry| entry.node)
            .collect::<Vec<_>>()
    };
    for _ in 0..2 {
        evaluator.set_fusion(true);
        assert_eq!(summary(&mut evaluator, &mut runtime), ids(&["final"]));
        assert_eq!(summary(&mut evaluator, &mut runtime), ids(&["final"]));
        evaluator.set_fusion(false);
        assert_eq!(
            summary(&mut evaluator, &mut runtime),
            ids(&["source", "gray", "final"])
        );
    }
}

#[test]
fn a_changed_graph_is_planned_anew_and_the_first_again() {
    let first = entelechy_scene::parse(DOCUMENT).unwrap();
    // `gray` is also read as an output, so it is not fused away.
    let second =
        entelechy_scene::parse(&format!("{DOCUMENT}output shade <- gray.result;\n")).unwrap();
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    let mut summary = |graph: &entelechy_graph::Graph| {
        evaluator
            .plan_summary(graph, &mut runtime, &BTreeMap::new(), &BTreeMap::new())
            .unwrap()
            .into_iter()
            .map(|entry| entry.node)
            .collect::<Vec<_>>()
    };
    for _ in 0..2 {
        assert_eq!(summary(&first), ids(&["final"]));
        assert_eq!(summary(&second), ids(&["gray", "final"]));
    }
}

/// What each node reads across evaluations: a node a port reads `<~`
/// holds state; another reading it `<-` reads its new state, made earlier
/// in the evaluation, one reading it `<~` its state as committed.
#[test]
fn a_graph_says_what_each_node_reads_of_state() {
    let graph = entelechy_scene::parse(
        "node begin: t.begin@1 { domain = 1; }
node world: t.world@1 { previous <~ world.result from begin.result; fight <~ fight.result from begin.result; }
node fight: t.fight@1 { previous <~ fight.result from begin.result; world <- world.result; }
node field: t.field@1 { fight <- fight.result; world <- world.result; begin <- begin.result; }
output result <- field.result;",
    )
    .unwrap();
    let reads = |id: &str| graph.reads(&entelechy_graph::NodeId(id.into())).unwrap();
    let names = |nodes: &[entelechy_graph::NodeId]| {
        nodes.iter().map(ToString::to_string).collect::<Vec<_>>()
    };
    assert!(!reads("begin").holds_state);
    let world = reads("world");
    assert!(world.holds_state);
    assert_eq!(names(&world.committed), ["fight"]);
    assert!(world.new_state.is_empty());
    let fight = reads("fight");
    assert!(fight.holds_state);
    assert_eq!(names(&fight.new_state), ["world"]);
    // Reading what holds no state is no reading of state.
    let field = reads("field");
    assert!(!field.holds_state);
    assert_eq!(names(&field.new_state), ["fight", "world"]);
    assert!(field.committed.is_empty());
}
