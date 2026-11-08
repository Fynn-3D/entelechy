//! Composition as the evaluator holds it, on a graph and IR written here
//! and a backend that runs only that IR: nodes bind Kernels by id, a node
//! runs again only when what it reads changed, a failure names its node
//! and keeps the last good results, and a malformed graph is refused
//! before anything runs.

use std::any::Any;
use std::collections::BTreeMap;

use entelechy_graph::{
    Binding, Context, Domain, DomainSpec, EvaluateError, Evaluator, Graph, GraphError, Node,
    NodeId, Report, Source,
};
use entelechy_ir::{Function, Instruction, Mode, Module, Op, Port, PortType, Type, ValueId};
use entelechy_kernel::{
    Backend, Capability, ExecutionError, Handle, Invocation, Manifest, Prepared, Runtime, Scope,
    Validated, Value,
};

/// Kernel `id` reading one image `source` when `reads`, giving each
/// element `vec4(red, 0, 0, 1)`, plus the whole seconds of the clock when
/// it requires `time.read`.
fn kernel(id: &str, red: f32, reads: bool, requires: &[Capability]) -> Validated {
    let mut requires: Vec<Capability> = requires.to_vec();
    requires.push(Capability::FieldWrite);
    if reads {
        requires.push(Capability::FieldRead);
    }
    let image = |name: &str| Port {
        name: name.into(),
        ty: PortType::image(),
    };
    let module = Module {
        manifest: Manifest {
            id: id.parse().unwrap(),
            mode: Mode::Map,
            requires: requires.into_iter().collect(),
            inputs: if reads { vec![image("source")] } else { vec![] },
            params: vec![],
            outputs: vec![image("result")],
            memories: vec![],
            budget: entelechy_ir::DEFAULT_BUDGET,
        },
        structs: vec![],
        functions: vec![],
        entry: Function {
            name: "compute".into(),
            args: vec![],
            body: vec![Instruction {
                id: ValueId(0),
                ty: Type::Vec4,
                op: Op::Const(Value::Vec4([red, 0.0, 0.0, 1.0])),
            }],
            results: vec![ValueId(0)],
            bindings: vec![],
        },
        merge: None,
        init: None,
    };
    entelechy_ir::validate(module).unwrap()
}

/// What the stub prepares a module to.
struct Given {
    value: [f32; 4],
    clocked: bool,
    fails: bool,
}

impl Prepared for Given {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A backend for the Kernels written here: each element is the constant
/// the entry returns, its red raised by the clock's whole seconds for a
/// Kernel that reads it; a Kernel whose id holds `fails` fails.
struct Stub;

impl Backend for Stub {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let [
            Instruction {
                op: Op::Const(Value::Vec4(value)),
                ..
            },
        ] = &module.entry.body[..]
        else {
            return Err("the stub runs a constant alone".into());
        };
        Ok(Box::new(Given {
            value: *value,
            clocked: module.manifest.requires.contains(Capability::TimeRead),
            fails: module.manifest.id.to_string().contains("fails"),
        }))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let given = prepared.as_any().downcast_ref::<Given>().unwrap();
        if given.fails {
            return Err(ExecutionError::Device("it fails".into()));
        }
        let mut value = given.value;
        if given.clocked {
            value[0] += invocation.context.time.floor() as f32;
        }
        let words: Vec<u32> = value.iter().map(|v| v.to_bits()).collect();
        let port = &module.manifest.outputs[0];
        Ok(vec![scope.adopt_storage(
            port.ty.schema.clone(),
            invocation.domain,
            words.repeat(invocation.domain.count()),
        )?])
    }
}

/// A runtime holding `test.source@1` (a 2x1 image, red 0.25),
/// `test.middle@1` (red 0.5) and `test.last@1` (red 0.75), the last two
/// reading an image.
fn runtime() -> Runtime {
    let mut runtime = Runtime::with_backend(1 << 12, Box::new(Stub));
    for module in [
        kernel("test.source@1", 0.25, false, &[]),
        kernel("test.middle@1", 0.5, true, &[]),
        kernel("test.last@1", 0.75, true, &[]),
    ] {
        runtime.load(module, "").unwrap();
    }
    runtime
}

fn id(name: &str) -> NodeId {
    NodeId(name.into())
}

fn ids(names: &[&str]) -> Vec<NodeId> {
    names.iter().map(|name| id(name)).collect()
}

/// Node `name` running `kernel`, reading `from`'s result, or over a 2x1
/// grid when it reads nothing.
fn node(name: &str, kernel: &str, from: Option<&str>) -> Node {
    Node {
        id: id(name),
        kernel: kernel.parse().unwrap(),
        inputs: from
            .map(|from| {
                (
                    "source".to_string(),
                    Binding::Now(Source::Node {
                        node: id(from),
                        port: "result".into(),
                    }),
                )
            })
            .into_iter()
            .collect(),
        params: BTreeMap::new(),
        domain: from
            .is_none()
            .then_some(DomainSpec::Literal(Domain::Grid(2, 1))),
        is_async: false,
        awake: Vec::new(),
        at: None,
    }
}

/// `nodes`, giving `output`'s result.
fn graph(nodes: Vec<Node>, output: &str) -> Graph {
    Graph {
        name: "test".into(),
        externals: vec![],
        texts: vec![],
        params: vec![],
        requirements: vec![],
        times_params: vec![],
        nodes,
        outputs: vec![(
            "result".into(),
            Source::Node {
                node: id(output),
                port: "result".into(),
            },
        )],
        fallbacks: vec![],
        placed: vec![],
        repeated: vec![],
    }
}

/// source -> middle -> last.
fn chain() -> Graph {
    graph(
        vec![
            node("source", "test.source@1", None),
            node("middle", "test.middle@1", Some("source")),
            node("last", "test.last@1", Some("middle")),
        ],
        "last",
    )
}

fn evaluator() -> Evaluator {
    let mut evaluator = Evaluator::new();
    evaluator.set_fusion(false);
    evaluator
}

fn at(time: f64) -> Context {
    Context {
        time,
        ..Context::default()
    }
}

fn frame(
    evaluator: &mut Evaluator,
    graph: &Graph,
    runtime: &mut Runtime,
    context: &Context,
) -> Result<Report, EvaluateError> {
    evaluator.evaluate(graph, runtime, &BTreeMap::new(), context, &BTreeMap::new())
}

fn red(runtime: &Runtime, handle: Handle) -> f32 {
    runtime.host().read_pixel(handle, 1, 0).unwrap()[0]
}

#[test]
fn nodes_bind_kernels_by_id_and_an_unchanged_graph_reuses_every_result() {
    let graph = chain();
    let mut runtime = runtime();
    let mut evaluator = evaluator();
    let first = frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();
    assert_eq!(first.executed, ids(&["source", "middle", "last"]));
    assert_eq!(red(&runtime, first.output), 0.75);
    assert_eq!(runtime.host().live(), 3);

    let again = frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();
    assert!(again.executed.is_empty());
    assert_eq!(again.reused, ids(&["source", "middle", "last"]));
    assert_eq!(again.output, first.output);
}

#[test]
fn republishing_a_kernel_runs_again_only_its_nodes_and_what_reads_them() {
    let graph = chain();
    let mut runtime = runtime();
    let mut evaluator = evaluator();
    frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();

    runtime
        .load(kernel("test.middle@1", 0.1, true, &[]), "")
        .unwrap();
    let report = frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();
    assert_eq!(report.reused, ids(&["source"]));
    assert_eq!(report.executed, ids(&["middle", "last"]));
    assert_eq!(
        red(&runtime, evaluator.cached(&id("middle"), "result").unwrap()),
        0.1
    );
    assert_eq!(runtime.host().live(), 3, "replaced outputs were released");
}

/// A node that reads the clock runs again as time advances; within a
/// second it gives the same words, so what reads it is spared.
#[test]
fn only_nodes_reading_the_clock_run_again_as_time_advances() {
    let mut runtime = runtime();
    runtime
        .load(
            kernel("test.second@1", 0.0, true, &[Capability::TimeRead]),
            "",
        )
        .unwrap();
    let graph = graph(
        vec![
            node("source", "test.source@1", None),
            node("second", "test.second@1", Some("source")),
            node("last", "test.last@1", Some("second")),
        ],
        "last",
    );
    let mut evaluator = evaluator();
    let first = frame(&mut evaluator, &graph, &mut runtime, &at(0.25)).unwrap();
    assert_eq!(first.executed, ids(&["source", "second", "last"]));

    let same = frame(&mut evaluator, &graph, &mut runtime, &at(0.5)).unwrap();
    assert_eq!(same.executed, ids(&["second"]));
    assert_eq!(same.reused, ids(&["source", "last"]));
    assert_eq!(same.output, first.output);

    let later = frame(&mut evaluator, &graph, &mut runtime, &at(1.5)).unwrap();
    assert_eq!(later.executed, ids(&["second", "last"]));
    assert_eq!(
        red(&runtime, evaluator.cached(&id("second"), "result").unwrap()),
        1.0
    );
    assert_eq!(runtime.host().live(), 3, "the Fields that gave way went");

    let still = frame(&mut evaluator, &graph, &mut runtime, &at(1.5)).unwrap();
    assert!(still.executed.is_empty(), "same time, nothing runs again");
}

#[test]
fn a_failure_names_its_node_and_the_last_good_results_survive() {
    let graph = chain();
    let mut runtime = runtime();
    let mut evaluator = evaluator();
    let first = frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();

    let failing = graph_with_last(&graph, "test.fails@1");
    runtime
        .load(kernel("test.fails@1", 0.0, true, &[]), "")
        .unwrap();
    let error = frame(&mut evaluator, &failing, &mut runtime, &Context::default()).unwrap_err();
    assert!(
        matches!(&error, EvaluateError::Execution { node, error: ExecutionError::Device(_) } if node.0 == "last"),
        "{error}"
    );
    assert_eq!(evaluator.cached(&id("last"), "result"), Some(first.output));
    assert_eq!(red(&runtime, first.output), 0.75);

    // The graph that ran runs again from what it kept.
    let back = frame(&mut evaluator, &graph, &mut runtime, &Context::default()).unwrap();
    assert_eq!(back.output, first.output);
}

/// `graph` with its last node bound to `kernel`.
fn graph_with_last(graph: &Graph, kernel: &str) -> Graph {
    let mut changed = graph.clone();
    changed.nodes.last_mut().unwrap().kernel = kernel.parse().unwrap();
    changed
}

#[test]
fn a_malformed_graph_is_refused_before_anything_runs() {
    let refused = |graph: Graph| {
        let mut runtime = runtime();
        let error = frame(&mut evaluator(), &graph, &mut runtime, &Context::default()).unwrap_err();
        assert_eq!(runtime.host().live(), 0, "nothing ran before {error}");
        error
    };

    let cycle = graph(
        vec![
            node("a", "test.middle@1", Some("b")),
            node("b", "test.middle@1", Some("a")),
        ],
        "a",
    );
    assert!(matches!(cycle.order(), Err(GraphError::Cycle(_))));
    assert!(matches!(
        refused(cycle),
        EvaluateError::Graph(GraphError::Cycle(_))
    ));

    let dangling = graph(vec![node("a", "test.middle@1", Some("nowhere"))], "a");
    assert!(matches!(
        dangling.order(),
        Err(GraphError::UnknownSource { .. })
    ));

    // A node, a port and a Kernel id one letter out are suggested.
    let mut near = chain();
    near.nodes[1].inputs.insert(
        "source".into(),
        Binding::Now(Source::Node {
            node: id("sorce"),
            port: "result".into(),
        }),
    );
    let message = near.order().unwrap_err().to_string();
    assert!(message.ends_with("did you mean `source`?"), "{message}");

    let mut typo_port = chain();
    let binding = typo_port.nodes[1].inputs.remove("source").unwrap();
    typo_port.nodes[1].inputs.insert("sorce".into(), binding);
    let message = refused(typo_port).to_string();
    assert!(message.ends_with("did you mean `source`?"), "{message}");

    let mut unknown_param = chain();
    unknown_param.nodes[0].params.insert(
        "gain".into(),
        entelechy_graph::ParamValue::Literal(Value::Int(2)),
    );
    assert!(matches!(
        refused(unknown_param),
        EvaluateError::UnknownPort { .. }
    ));

    let typo = graph_with_last(&chain(), "test.lsat@1");
    let message = refused(typo).to_string();
    assert!(
        message.ends_with("did you mean `test.last@1`?"),
        "{message}"
    );

    let mut no_output = chain();
    no_output.outputs.clear();
    assert!(matches!(no_output.order(), Err(GraphError::NoOutput)));

    let mut no_domain = chain();
    no_domain.nodes[0].domain = None;
    assert_eq!(refused(no_domain), EvaluateError::NoDomain(id("source")));
}

/// A producer that only one node reads, run apart from it, is named in
/// that node's plan with why: here pinned, and read neither at the
/// reader's own element nor at an offset of a row.
#[test]
fn the_plan_says_why_a_sole_reader_does_not_absorb_its_producer() {
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    evaluator.pin(id("source"));
    let chain = graph(
        vec![
            node("source", "test.source@1", None),
            node("middle", "test.middle@1", Some("source")),
            node("last", "test.last@1", Some("middle")),
        ],
        "last",
    );
    let plan = evaluator
        .plan_summary(
            &chain,
            &mut runtime,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &Context::default(),
        )
        .unwrap();
    let apart = |name: &str| {
        plan.iter()
            .find(|entry| entry.node == id(name))
            .map(|entry| entry.apart.clone())
            .unwrap()
    };
    assert_eq!(
        apart("middle"),
        vec![(id("source"), "it is pinned".to_owned())]
    );
    let last = apart("last");
    assert_eq!(last.len(), 1, "{last:?}");
    assert_eq!(last[0].0, id("middle"));
    assert!(last[0].1.contains("sample"), "{}", last[0].1);
    assert!(apart("source").is_empty());
}

/// A plan is made under the context it is asked with: a node sized by
/// `window()` runs over the window given.
#[test]
fn a_plan_is_made_for_the_window_asked() {
    let mut runtime = runtime();
    let mut evaluator = Evaluator::new();
    let mut source = node("source", "test.source@1", None);
    source.domain = Some(DomainSpec::Expr(entelechy_graph::DomainExpr::Context(
        "window".into(),
    )));
    let windowed = graph(vec![source], "source");
    let context = Context {
        window: [6.0, 4.0],
        ..Context::default()
    };
    let plan = evaluator
        .plan_summary(
            &windowed,
            &mut runtime,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &context,
        )
        .unwrap();
    assert_eq!(plan[0].domain, Some(Domain::Grid(6, 4)));
}
