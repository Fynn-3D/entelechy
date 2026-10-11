//! A producer read at an offset of its reader's element fuses into it as
//! a function of the row, and a reader that only counts it reads its
//! extent from params: on every backend, not a bit changes.

use std::collections::BTreeMap;

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_graph::{Context, Evaluator, NodeId, Report};
use entelechy_ir::Validated;
use entelechy_kernel::{Backend, Runtime};
use entelechy_rules::backend::Checked;
use entelechy_rules::{ExecutionError, NativeOp, Prepared};

const SEQ: &str = "kernel test.seq@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()) * 1.25 - 2.0; } }";
const LIFT: &str = "kernel test.lift@1 { requires { field.read; field.write; } struct P { x: Float; n: Int; } input source: Rows<Float>; param k: Float = 2.0; output result: Rows<P>; compute { let v = sample(source) * k + float(element()) * 0.5 + float(domain()); return P { x: v, n: element() * 3 + count(source) }; } }";
const JOIN: &str = "kernel test.join@1 { requires { field.read; field.write; } struct P { x: Float; n: Int; } input a: Rows<P>; input b: Rows<Float>; output result: Rows<Float>; compute { let k = element(); let n = count(a); if k < n { let p = at(a, k + 1); return p.x + float(p.n); } return at(b, k - n) + float(size(a)); } }";
const TALLY: &str = "kernel test.tally@1 { requires { field.read; field.write; } struct P { x: Float; n: Int; } input a: Rows<P>; input b: Rows<Float>; output result: Rows<Float>; compute { return at(b, element() + 2) * 0.5 + float(count(a)); } }";

/// `join` reads `lift` one row on and `t` past `lift`'s rows, two rows
/// past both; `tally` reads `join` two rows on and counts `lift`.
const CHAIN: &str = "
node s: test.seq@1 { domain = 5; }
node t: test.seq@1 { domain = 3; }
node lift: test.lift@1 { source <- s.result; k = 1.5; }
node join: test.join@1 { a <- lift.result; b <- t.result; domain = count(a) + count(b) + 2; }
node tally: test.tally@1 { a <- lift.result; b <- join.result; domain = 9; }
output result <- tally.result;
";

fn ids(names: &[&str]) -> Vec<NodeId> {
    names.iter().map(|n| NodeId((*n).into())).collect()
}

fn runtime(backend: Box<dyn Backend>) -> Runtime {
    let mut runtime = Runtime::with_backend(1 << 20, backend);
    for source in [SEQ, LIFT, JOIN, TALLY] {
        let module = entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
        runtime.load(module, source).unwrap();
    }
    runtime
}

fn words(runtime: &mut Runtime, report: &Report) -> Vec<Vec<u32>> {
    report
        .outputs
        .iter()
        .map(|(_, handle)| {
            runtime.materialize(*handle).unwrap();
            runtime.host().data(*handle).unwrap().storage().to_vec()
        })
        .collect()
}

/// `document` evaluated plain and fused: the fused report and both
/// outputs' words.
fn plain_and_fused(
    backend: Box<dyn Backend>,
    document: &str,
) -> (Report, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut runtime = runtime(backend);
    let graph = entelechy_scene::parse(document).unwrap_or_else(|error| panic!("{error}"));
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        evaluator
            .evaluate(
                &graph,
                runtime,
                &BTreeMap::new(),
                &Context::default(),
                &BTreeMap::new(),
            )
            .unwrap()
    };
    let mut plain = Evaluator::new();
    plain.set_fusion(false);
    let reference = evaluate(&mut plain, &mut runtime);
    assert!(reference.fused.is_empty());
    let expected = words(&mut runtime, &reference);
    plain.clear(&mut runtime);
    let mut fused = Evaluator::new();
    let report = evaluate(&mut fused, &mut runtime);
    let actual = words(&mut runtime, &report);
    (report, expected, actual)
}

fn backends() -> Vec<(&'static str, Box<dyn Backend>)> {
    let mut backends: Vec<(&'static str, Box<dyn Backend>)> = vec![
        ("cpu", Box::new(entelechy_cpu::CpuBackend)),
        ("jit", Box::new(entelechy_jit::JitBackend::new().unwrap())),
    ];
    match entelechy_gpu::GpuBackend::new() {
        Ok(gpu) => backends.push(("gpu", Box::new(gpu))),
        Err(error) => eprintln!("skipping the gpu: {error}"),
    }
    backends
}

#[test]
fn a_chain_read_at_offsets_runs_as_one_kernel_and_changes_no_bit() {
    let mut seen: Option<Vec<Vec<u32>>> = None;
    for (name, backend) in backends() {
        let (report, expected, actual) = plain_and_fused(backend, CHAIN);
        assert_eq!(actual, expected, "{name}: fusion changes no bit");
        assert_eq!(report.executed, ids(&["tally"]), "{name}");
        // Each backend computes the same words, as the reference does.
        match &seen {
            Some(words) => assert_eq!(&expected, words, "{name} against cpu"),
            None => seen = Some(expected),
        }
    }
}

#[test]
fn a_reader_that_only_counts_a_fused_producer_is_given_its_extent() {
    // `join` is an output too, so it stays stored and `tally` reads it on
    // its own; `lift` fuses into `join` all the same, and `tally` counts
    // it from params.
    let document = format!("{CHAIN}output join <- join.result;");
    for (name, backend) in backends() {
        let (report, expected, actual) = plain_and_fused(backend, &document);
        assert_eq!(actual, expected, "{name}: fusion changes no bit");
        assert_eq!(report.executed, ids(&["join", "tally"]), "{name}");
        let into_join = report
            .fused
            .iter()
            .find(|(node, _)| node.0 == "join")
            .map(|(_, absorbed)| absorbed.clone());
        assert_eq!(into_join, Some(ids(&["lift", "s", "t"])), "{name}");
    }
    // The plan says what `tally` counts without reading it.
    let mut runtime = runtime(Box::new(entelechy_cpu::CpuBackend));
    let graph = entelechy_scene::parse(&document).unwrap();
    let plan = Evaluator::new()
        .plan_summary(&graph, &mut runtime, &BTreeMap::new(), &BTreeMap::new())
        .unwrap();
    let tally = plan.iter().find(|entry| entry.node.0 == "tally").unwrap();
    assert_eq!(tally.counted, ids(&["lift"]));
    assert!(!tally.inputs.contains_key("a"), "{:?}", tally.inputs);
}

#[test]
fn a_producer_read_at_two_rows_stays_stored() {
    // Two rows of `lift` per element would compute it twice: it is
    // stored, and only what reads at one row fuses.
    let mut runtime = runtime(Box::new(entelechy_cpu::CpuBackend));
    let twice = JOIN.replace("test.join@1", "test.join@2").replace(
        "let p = at(a, k + 1); return p.x + float(p.n);",
        "return at(a, k + 1).x + at(a, k).x;",
    );
    let module = entelechy_compiler::compile(&twice).unwrap();
    runtime.load(module, &twice).unwrap();
    let graph = entelechy_scene::parse(&CHAIN.replace("test.join@1", "test.join@2")).unwrap();
    let report = Evaluator::new()
        .evaluate(
            &graph,
            &mut runtime,
            &BTreeMap::new(),
            &Context::default(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(
        report.executed.contains(&NodeId("lift".into())),
        "{:?}",
        report.executed
    );
}

/// The CPU, slow to prepare what counts a producer from params: the
/// worker delivers that last.
struct SlowToCount;

impl Backend for SlowToCount {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        Backend::declines(&entelechy_cpu::CpuBackend, module)
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let counts = module.manifest.params.iter().any(|p| p.name.contains('#'));
        if counts && module.manifest.id.to_string().contains("tally") {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        Backend::prepare(&entelechy_cpu::CpuBackend, module)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        Backend::execute_checked(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        Backend::execute_native(&entelechy_cpu::CpuBackend, op, invocation, scope)
    }
}

#[test]
fn a_counting_reader_and_the_group_it_counts_arrive_together_from_the_worker() {
    // `tally` runs its counted version only once `join` runs `lift`
    // inside: until both arrive, both run as they were, and every frame
    // is the plain one, though `join` arrives first.
    let document = format!("{CHAIN}output join <- join.result;");
    let graph = entelechy_scene::parse(&document).unwrap();
    let mut runtime = runtime(Box::new(SlowToCount));
    runtime.set_background(true);
    let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
        evaluator
            .evaluate(
                &graph,
                runtime,
                &BTreeMap::new(),
                &Context::default(),
                &BTreeMap::new(),
            )
            .unwrap()
    };
    let mut plain = Evaluator::new();
    plain.set_fusion(false);
    let reference = evaluate(&mut plain, &mut runtime);
    let expected = words(&mut runtime, &reference);
    plain.clear(&mut runtime);
    let mut evaluator = Evaluator::new();
    let mut frames = 0;
    loop {
        let report = evaluate(&mut evaluator, &mut runtime);
        assert_eq!(words(&mut runtime, &report), expected, "frame {frames}");
        if report.executed == ids(&["join", "tally"]) {
            break;
        }
        assert!(
            report.fused.is_empty(),
            "frame {frames}: {:?}",
            report.fused
        );
        frames += 1;
        assert!(frames < 2000, "the worker delivers both");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(frames > 5, "`join` waited for `tally`");
}

#[test]
fn a_domain_that_reads_a_value_still_fuses_what_it_reads_at_an_offset() {
    // Neither `join`'s extent nor `tally`'s is known before `m` runs; the
    // extent of `lift` is, and is all either counts of it.
    let four = "kernel test.four@1 { requires { field.write; } output result: Rows<Int>; compute { return 2; } }";
    let join = JOIN.replace("test.join@1", "test.joinm@1").replace(
        "input b: Rows<Float>;",
        "input b: Rows<Float>; input m: Rows<Int>;",
    );
    let tally = TALLY.replace("test.tally@1", "test.tallym@1").replace(
        "input b: Rows<Float>;",
        "input b: Rows<Float>; input m: Rows<Int>;",
    );
    let document = "
node m: test.four@1 { domain = 1; }
node s: test.seq@1 { domain = 5; }
node t: test.seq@1 { domain = 3; }
node lift: test.lift@1 { source <- s.result; k = 1.5; }
node join: test.joinm@1 { a <- lift.result; b <- t.result; m <- m.result; domain = count(a) + count(b) + value(m, 0); }
node tally: test.tallym@1 { a <- lift.result; b <- join.result; m <- m.result; domain = count(a) + value(m, 0) + 2; }
output result <- tally.result;
output join <- join.result;
";
    for (name, backend) in backends() {
        let mut runtime = runtime(backend);
        for source in [four, &join, &tally] {
            let module =
                entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
            runtime.load(module, source).unwrap();
        }
        let graph = entelechy_scene::parse(document).unwrap();
        let evaluate = |evaluator: &mut Evaluator, runtime: &mut Runtime| {
            evaluator
                .evaluate(
                    &graph,
                    runtime,
                    &BTreeMap::new(),
                    &Context::default(),
                    &BTreeMap::new(),
                )
                .unwrap()
        };
        let mut plain = Evaluator::new();
        plain.set_fusion(false);
        let reference = evaluate(&mut plain, &mut runtime);
        let expected = words(&mut runtime, &reference);
        plain.clear(&mut runtime);
        let report = evaluate(&mut Evaluator::new(), &mut runtime);
        assert_eq!(words(&mut runtime, &report), expected, "{name}");
        assert_eq!(report.executed, ids(&["m", "join", "tally"]), "{name}");
    }
}
