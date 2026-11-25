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
        times_rows: vec![],
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

/// State made on a device and read `<~` on the processor queues its readback
/// as soon as its producer runs, ahead of the device work that follows it.
#[test]
fn state_read_on_the_processor_queues_its_readback_when_made() {
    use std::sync::{Arc, Mutex};

    struct ResidentWords(Vec<u32>);
    struct Queued {
        words: Vec<u32>,
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    impl entelechy_kernel::Downloading for Queued {
        fn take(self: Box<Self>) -> Result<Vec<u32>, String> {
            self.events.lock().unwrap().push("take");
            Ok(self.words)
        }
    }
    struct LoggedDevice {
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Backend for LoggedDevice {
        fn name(&self) -> &'static str {
            "device"
        }
        fn has_device(&self) -> bool {
            true
        }
        fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
            Stub.prepare(module)
        }
        fn execute(
            &self,
            prepared: &dyn Prepared,
            module: &Validated,
            invocation: &Invocation,
            scope: &mut Scope<'_>,
        ) -> Result<Vec<Handle>, ExecutionError> {
            let tag = if module.manifest.id.to_string().contains("depths") {
                "depths"
            } else {
                "picture"
            };
            self.events.lock().unwrap().push(tag);
            let given = prepared.as_any().downcast_ref::<Given>().unwrap();
            let words: Vec<u32> = given
                .value
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                .repeat(invocation.domain.count());
            let port = &module.manifest.outputs[0];
            Ok(vec![scope.adopt_resident(
                port.ty.schema.clone(),
                invocation.domain,
                Box::new(ResidentWords(words)),
            )?])
        }
        fn download(
            &self,
            device: &(dyn Any + Send + Sync),
            _byte_len: usize,
        ) -> Result<Vec<u32>, String> {
            self.events.lock().unwrap().push("download");
            Ok(device.downcast_ref::<ResidentWords>().unwrap().0.clone())
        }
        fn download_later(
            &self,
            device: &(dyn Any + Send + Sync),
            _byte_len: usize,
        ) -> Result<Box<dyn entelechy_kernel::Downloading>, String> {
            self.events.lock().unwrap().push("download_later");
            Ok(Box::new(Queued {
                words: device.downcast_ref::<ResidentWords>().unwrap().0.clone(),
                events: Arc::clone(&self.events),
            }))
        }
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(LoggedDevice {
            events: Arc::clone(&events),
        }),
        Some(Box::new(Stub)),
    );
    runtime.set_small(10);
    runtime
        .load(kernel("test.init@1", 0.0, false, &[]), "")
        .unwrap();
    runtime
        .load(
            kernel("test.step@1", 1.0, true, &[Capability::TimeRead]),
            "",
        )
        .unwrap();
    runtime
        .load(kernel("test.depths@1", 2.0, true, &[]), "")
        .unwrap();
    runtime
        .load(kernel("test.picture@1", 3.0, true, &[]), "")
        .unwrap();

    let mut init = node("init", "test.init@1", None);
    init.domain = Some(DomainSpec::Literal(Domain::Grid(1, 1)));

    let mut step = node("step", "test.step@1", None);
    step.domain = Some(DomainSpec::Literal(Domain::Grid(1, 1)));
    step.inputs.insert(
        "source".into(),
        Binding::Previous {
            source: Source::Node {
                node: id("depths"),
                port: "result".into(),
            },
            initial: Source::Node {
                node: id("init"),
                port: "result".into(),
            },
            migration: None,
        },
    );

    let mut depths = node("depths", "test.depths@1", Some("step"));
    depths.domain = Some(DomainSpec::Literal(Domain::Grid(128, 1)));

    let mut picture = node("picture", "test.picture@1", Some("depths"));
    picture.domain = Some(DomainSpec::Literal(Domain::Grid(128, 1)));

    let scene = graph(vec![init, step, depths, picture], "picture");
    let mut evaluator = Evaluator::new();

    for t in [0.0, 1.0] {
        let context = Context {
            time: t,
            ..Context::default()
        };
        evaluator
            .evaluate(
                &scene,
                &mut runtime,
                &BTreeMap::new(),
                &context,
                &BTreeMap::new(),
            )
            .unwrap();
    }

    assert_eq!(
        &events.lock().unwrap()[..],
        &[
            "depths",
            "download_later",
            "picture",
            "take",
            "depths",
            "download_later",
            "picture",
        ]
    );
}

#[test]
fn fusion_stops_past_instruction_bound() {
    use entelechy_ir::BinaryOp;

    let heavy_kernel = |id: &str, reads: bool, ops: usize| -> Validated {
        let mut requires = vec![Capability::FieldWrite];
        if reads {
            requires.push(Capability::FieldRead);
        }
        let image = |name: &str| Port {
            name: name.into(),
            ty: PortType::image(),
        };
        let mut body = Vec::with_capacity(ops + 1);
        let mut cur = if reads {
            body.push(Instruction {
                id: ValueId(0),
                ty: Type::Vec4,
                op: Op::Sample(0),
            });
            ValueId(0)
        } else {
            body.push(Instruction {
                id: ValueId(0),
                ty: Type::Vec4,
                op: Op::Const(Value::Vec4([0.25, 0.5, 0.75, 1.0])),
            });
            ValueId(0)
        };
        let mut next = body.len() as u32;
        for _ in 0..ops {
            let id_now = ValueId(next);
            next += 1;
            body.push(Instruction {
                id: id_now,
                ty: Type::Vec4,
                op: Op::Binary(BinaryOp::Add, cur, cur),
            });
            cur = id_now;
        }
        entelechy_ir::validate(Module {
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
                body,
                results: vec![cur],
                bindings: vec![],
            },
            merge: None,
            init: None,
        })
        .unwrap()
    };

    struct CountOnly;
    impl Backend for CountOnly {
        fn name(&self) -> &'static str {
            "count_only"
        }
        fn prepare(&self, _module: &Validated) -> Result<Box<dyn Prepared>, String> {
            Ok(Box::new(Given {
                value: [1.0, 0.0, 0.0, 1.0],
                clocked: false,
                fails: false,
            }))
        }
        fn execute(
            &self,
            _prepared: &dyn Prepared,
            _module: &Validated,
            _invocation: &Invocation,
            _scope: &mut Scope<'_>,
        ) -> Result<Vec<Handle>, ExecutionError> {
            unreachable!("plan_summary executes nothing")
        }
    }

    let mut runtime = Runtime::with_backends(1 << 20, Box::new(CountOnly), None);
    runtime
        .load(heavy_kernel("test.prod@1", false, 9_000), "")
        .unwrap();
    runtime
        .load(heavy_kernel("test.cons@1", true, 9_000), "")
        .unwrap();

    let mut prod = node("prod", "test.prod@1", None);
    prod.domain = Some(DomainSpec::Literal(Domain::Grid(16, 16)));
    let mut cons = node("cons", "test.cons@1", Some("prod"));
    cons.domain = Some(DomainSpec::Literal(Domain::Grid(16, 16)));
    let scene = graph(vec![prod, cons], "cons");

    let mut evaluator = Evaluator::new();
    let entries = evaluator
        .plan_summary(
            &scene,
            &mut runtime,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &Context::default(),
        )
        .unwrap();
    assert_eq!(
        entries.len(),
        2,
        "producer and consumer stay apart: {entries:?}"
    );
    let cons_entry = entries.iter().find(|e| e.node == id("cons")).unwrap();
    assert!(cons_entry.absorbed.is_empty());
    assert!(
        cons_entry
            .apart
            .iter()
            .any(|(who, why)| *who == id("prod") && why.contains("past 16000")),
        "expected instruction-bound apart reason, got: {:?}",
        cons_entry.apart
    );
}

#[test]
fn order_hoists_value_count_producers_and_defers_consumers() {
    let mut indep_a = node("indep_a", "test.a@1", None);
    indep_a.domain = Some(DomainSpec::Literal(Domain::Rows(64)));
    let mut indep_b = node("indep_b", "test.b@1", Some("indep_a"));
    indep_b.domain = Some(DomainSpec::Literal(Domain::Rows(64)));

    let mut count_src = node("count_src", "test.c@1", None);
    count_src.domain = Some(DomainSpec::Literal(Domain::Rows(16)));
    let mut count_node = node("count_node", "test.d@1", Some("count_src"));
    count_node.domain = Some(DomainSpec::Literal(Domain::Rows(1)));

    let mut consumer = Node {
        id: id("consumer"),
        kernel: "test.e@1".parse().unwrap(),
        inputs: BTreeMap::from([
            (
                "n".into(),
                Binding::Now(Source::Node {
                    node: id("count_node"),
                    port: "result".into(),
                }),
            ),
            (
                "data".into(),
                Binding::Now(Source::Node {
                    node: id("indep_b"),
                    port: "result".into(),
                }),
            ),
        ]),
        params: BTreeMap::new(),
        domain: Some(DomainSpec::Expr(entelechy_graph::DomainExpr::Value(
            "n".into(),
            0,
        ))),
        is_async: false,
        awake: Vec::new(),
        at: None,
    };
    let mut out = node("out", "test.out@1", Some("consumer"));
    out.domain = Some(DomainSpec::Literal(Domain::Rows(64)));

    // Put independent heavy work first in declaration order; readback-aware
    // scheduling must hoist `count_src -> count_node` ahead of `indep_a -> indep_b`
    // and schedule `consumer` after independent work to hide prefetch latency.
    let g = graph(
        vec![
            indep_a,
            indep_b,
            count_src,
            count_node,
            consumer.clone(),
            out,
        ],
        "out",
    );
    let ordered: Vec<&str> = g
        .execution_order()
        .unwrap()
        .into_iter()
        .map(|i| g.nodes[i].id.0.as_ref())
        .collect();
    let pos = |name: &str| ordered.iter().position(|n| *n == name).unwrap();
    assert!(
        pos("count_src") < pos("count_node"),
        "dependency violated: {ordered:?}"
    );
    assert!(
        pos("count_node") < pos("indep_a"),
        "count producer should be hoisted ahead of independent work: {ordered:?}"
    );
    assert!(
        pos("indep_b") < pos("consumer"),
        "independent work should execute before value(port) consumer: {ordered:?}"
    );

    // Canonical `order()` preserves declaration DFS order for graph tools.
    let canonical: Vec<&str> = g
        .order()
        .unwrap()
        .into_iter()
        .map(|i| g.nodes[i].id.0.as_ref())
        .collect();
    assert_eq!(
        canonical,
        vec![
            "indep_a",
            "indep_b",
            "count_src",
            "count_node",
            "consumer",
            "out"
        ]
    );

    // Graphs without `value(...)` domains keep exact original order.
    consumer.domain = Some(DomainSpec::Literal(Domain::Rows(16)));
    let plain = graph(
        vec![
            node("indep_a", "test.a@1", None),
            node("indep_b", "test.b@1", Some("indep_a")),
            node("count_src", "test.c@1", None),
            node("count_node", "test.d@1", Some("count_src")),
            consumer,
            node("out", "test.out@1", Some("consumer")),
        ],
        "out",
    );
    let plain_order: Vec<&str> = plain
        .execution_order()
        .unwrap()
        .into_iter()
        .map(|i| plain.nodes[i].id.0.as_ref())
        .collect();
    assert_eq!(
        plain_order,
        vec![
            "indep_a",
            "indep_b",
            "count_src",
            "count_node",
            "consumer",
            "out"
        ]
    );
}
