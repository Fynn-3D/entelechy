//! Hot reload as the runtime holds it, on IR written here and a backend
//! that runs only it: a candidate is prepared beside the running version,
//! checked, and swapped in whole; what is refused leaves the running
//! version as it was, and a version already handed out runs on.

use std::any::Any;
use std::sync::Arc;

use entelechy_ir::{Function, Instruction, Mode, Module, Op, Port, PortType, Type, ValueId};
use entelechy_kernel::{
    Backend, Capability, Context, Domain, ExecutionError, Expect, Generation, Handle, Invocation,
    KernelId, KernelVersion, Manifest, NativeOp, PrepareError, Prepared, Runtime, Scope, Validated,
    Value,
};

/// Kernel `id`, giving each element of each of its `outputs` images
/// `vec4(red, 0, 0, 1)`, requiring `requires`.
fn constant(id: &str, red: f32, outputs: usize, requires: &[Capability]) -> Validated {
    let module = Module {
        manifest: Manifest {
            id: id.parse().unwrap(),
            mode: Mode::Map,
            requires: requires.iter().copied().collect(),
            inputs: vec![],
            params: vec![],
            outputs: (0..outputs)
                .map(|k| Port {
                    name: format!("out{k}"),
                    ty: PortType::image(),
                })
                .collect(),
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
            results: vec![ValueId(0); outputs],
            bindings: vec![],
        },
        merge: None,
        init: None,
    };
    entelechy_ir::validate(module).unwrap()
}

/// `test.color@1`, writing `red`.
fn color(red: f32) -> Validated {
    constant("test.color@1", red, 1, &[Capability::FieldWrite])
}

fn id() -> KernelId {
    "test.color@1".parse().unwrap()
}

/// What the stub prepares a module to: the value its entry returns.
struct Given([f32; 4]);

impl Prepared for Given {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A backend for the Kernels written here: each element of each output
/// is the value the entry returns. It panics preparing a Kernel whose id
/// holds `panics`, refuses one whose id holds `refuses`, and, `losing`,
/// gives one output fewer than the manifest declares.
#[derive(Default)]
struct Stub {
    panics: Option<&'static str>,
    refuses: Option<&'static str>,
    losing: bool,
}

impl Backend for Stub {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let id = module.manifest.id.to_string();
        if self.panics.is_some_and(|word| id.contains(word)) {
            panic!("the backend could not handle this module");
        }
        if self.refuses.is_some_and(|word| id.contains(word)) {
            return Err("the backend refuses this module".into());
        }
        match &module.entry.body[..] {
            [
                Instruction {
                    op: Op::Const(Value::Vec4(value)),
                    ..
                },
            ] => Ok(Box::new(Given(*value))),
            _ => Err("the stub runs a constant alone".into()),
        }
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let Given(value) = prepared.as_any().downcast_ref::<Given>().unwrap();
        let words: Vec<u32> = value.iter().map(|v| v.to_bits()).collect();
        let mut outputs = Vec::new();
        for port in &module.manifest.outputs {
            let storage = words.repeat(invocation.domain.count());
            outputs.push(scope.adopt_storage(
                port.ty.schema.clone(),
                invocation.domain,
                storage,
            )?);
        }
        if self.losing
            && let Some(lost) = outputs.pop()
        {
            let _ = scope.release(lost);
        }
        Ok(outputs)
    }
}

fn runtime(stub: Stub) -> Runtime {
    Runtime::with_backend(1 << 12, Box::new(stub))
}

fn invocation() -> Invocation {
    Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    }
}

/// The red `version` writes, its output released after.
fn red(runtime: &mut Runtime, version: &KernelVersion) -> f32 {
    let outputs = runtime.execute(version, &invocation()).unwrap();
    let pixel = runtime.host().read_pixel(outputs[0], 0, 0).unwrap();
    runtime.host_mut().release(outputs[0]).unwrap();
    pixel[0]
}

fn load(runtime: &mut Runtime, module: Validated) -> Arc<KernelVersion> {
    runtime.load(module, "").unwrap()
}

/// Waits for the worker's answer to `ticket`, giving a turn each time as
/// a frame does where no thread can start (WebAssembly).
fn answer(
    runtime: &mut Runtime,
    ticket: entelechy_kernel::Ticket,
) -> Result<entelechy_kernel::Candidate, PrepareError> {
    loop {
        runtime.prepare_turn(1);
        if let Some(result) = runtime.poll(ticket) {
            break result;
        }
        std::thread::yield_now();
    }
}

#[test]
fn publish_is_an_atomic_swap_that_keeps_pinned_versions_valid() {
    let mut runtime = runtime(Stub::default());
    let first = load(&mut runtime, color(0.25));
    assert_eq!(first.generation(), Generation(1));

    let candidate = runtime.prepare_module(color(1.0), "second").unwrap();
    assert_eq!(candidate.id(), &id());
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(1));
    let second = runtime
        .publish(candidate, Expect::Generation(first.generation()))
        .unwrap();
    assert_eq!(second.generation(), Generation(2));
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(2));

    // The old version still runs for whoever holds it.
    assert_eq!(red(&mut runtime, &first), 0.25);
    assert_eq!(red(&mut runtime, &second), 1.0);
    assert_eq!(second.source(), "second");
}

#[test]
fn a_candidate_the_backend_refuses_leaves_the_running_kernel_alive() {
    let mut runtime = runtime(Stub {
        refuses: Some("color"),
        ..Stub::default()
    });
    let running = load(
        &mut runtime,
        constant("test.kept@1", 0.5, 1, &[Capability::FieldWrite]),
    );
    let refused = runtime.load(
        constant("test.kept_color@1", 1.0, 1, &[Capability::FieldWrite]),
        "",
    );
    assert!(
        refused
            .unwrap_err()
            .to_string()
            .contains("refuses this module")
    );
    assert!(
        runtime
            .kernel(&"test.kept_color@1".parse().unwrap())
            .is_none()
    );
    assert_eq!(red(&mut runtime, &running), 0.5);
}

#[test]
fn stale_and_foreign_candidates_are_rejected() {
    let mut runtime = runtime(Stub::default());
    let first = load(&mut runtime, color(0.25));

    let stale = runtime.prepare_module(color(0.0), "").unwrap();
    let error = runtime
        .publish(stale, Expect::Generation(Generation(7)))
        .unwrap_err();
    assert_eq!(error.actual, Some(first.generation()));
    assert!(error.to_string().contains("registry holds g1"), "{error}");

    let duplicate = runtime.prepare_module(color(0.0), "").unwrap();
    let error = runtime.publish(duplicate, Expect::Absent).unwrap_err();
    assert_eq!(error.expected, Expect::Absent);
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(1));
    let current = runtime.kernel(&id()).unwrap();
    assert_eq!(red(&mut runtime, &current), 0.25);

    runtime.retire(&id()).unwrap();
    assert!(runtime.kernel(&id()).is_none());
    let fresh = runtime.prepare_module(color(0.0), "").unwrap();
    assert_eq!(
        runtime.publish(fresh, Expect::Absent).unwrap().generation(),
        Generation(2)
    );
}

/// Candidates published as one change are all checked before any swaps:
/// one stale expectation, and none is published.
#[test]
fn publishing_several_is_all_or_nothing() {
    let mut runtime = runtime(Stub::default());
    let first = load(&mut runtime, color(0.25));
    let other = || constant("test.other@1", 0.1, 1, &[Capability::FieldWrite]);
    let stale = Expect::Generation(Generation(first.generation().0 + 7));

    let fresh = runtime.prepare_module(other(), "").unwrap();
    let replacement = runtime.prepare_module(color(0.5), "").unwrap();
    let refused = runtime.publish_all(vec![(fresh, Expect::Absent), (replacement, stale)]);
    assert!(refused.is_err());
    assert!(runtime.kernel(&"test.other@1".parse().unwrap()).is_none());
    assert_eq!(
        runtime.kernel(&id()).unwrap().generation(),
        first.generation()
    );

    let fresh = runtime.prepare_module(other(), "").unwrap();
    let replacement = runtime.prepare_module(color(0.5), "").unwrap();
    let published = runtime
        .publish_all(vec![
            (fresh, Expect::Absent),
            (replacement, Expect::Generation(first.generation())),
        ])
        .unwrap();
    assert_eq!(published.len(), 2);
    assert!(runtime.kernel(&"test.other@1".parse().unwrap()).is_some());
    let current = runtime.kernel(&id()).unwrap();
    assert_eq!(red(&mut runtime, &current), 0.5);
}

#[test]
fn background_preparation_resolves_through_tickets() {
    let mut runtime = runtime(Stub::default());
    // Inline: a ticket is ready when issued.
    let ticket = runtime.prepare_later(color(0.25), String::new());
    let candidate = runtime.poll(ticket).unwrap().unwrap();
    assert_eq!(candidate.id(), &id());
    assert!(runtime.poll(ticket).is_none(), "a ticket is redeemed once");

    // With a worker: the running version runs until the replacement is
    // collected and published.
    runtime.set_background(true);
    assert!(runtime.background());
    let running = load(&mut runtime, color(0.5));
    let ticket = runtime.prepare_later(color(0.75), String::new());
    let mut candidate = None;
    for _ in 0..2000 {
        assert_eq!(red(&mut runtime, &running), 0.5);
        runtime.prepare_turn(1);
        if let Some(result) = runtime.poll(ticket) {
            candidate = Some(result.unwrap());
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let candidate = candidate.expect("the worker finishes");
    let published = runtime
        .publish(candidate, Expect::Generation(running.generation()))
        .unwrap();
    assert_eq!(red(&mut runtime, &published), 0.75);

    // A discarded ticket never surfaces; stopping the worker keeps what it
    // finished.
    let dropped = runtime.prepare_later(color(0.1), String::new());
    runtime.discard(dropped);
    let kept = runtime.prepare_later(color(0.2), String::new());
    runtime.set_background(false);
    assert!(runtime.poll(dropped).is_none());
    let kept = runtime.poll(kept).unwrap().unwrap();
    let detached = runtime.detach(kept, "fused.k@1".parse().unwrap());
    assert_eq!(red(&mut runtime, &detached), 0.2);
    assert!(runtime.kernel(&"fused.k@1".parse().unwrap()).is_none());
}

/// A backend panic while preparing rejects the Kernel, inline or on the
/// worker: the worker answers the ticket and goes on, and the running
/// Kernel is untouched. Where a panic aborts (WebAssembly) there is
/// nothing to catch.
#[cfg(panic = "unwind")]
#[test]
fn a_backend_that_panics_preparing_rejects_the_kernel_and_the_worker_lives() {
    let mut runtime = runtime(Stub {
        panics: Some("panics"),
        ..Stub::default()
    });
    let first = load(&mut runtime, color(0.25));
    let panics = || constant("test.panics@1", 1.0, 1, &[Capability::FieldWrite]);
    let Err(inline) = runtime.prepare_module(panics(), "") else {
        panic!("taken inline");
    };
    assert!(
        inline.to_string().contains("could not handle this module"),
        "{inline}"
    );

    runtime.set_background(true);
    let refused = runtime.prepare_later(panics(), String::new());
    let taken = runtime.prepare_later(color(0.5), String::new());
    let Err(error) = answer(&mut runtime, refused) else {
        panic!("taken on the worker");
    };
    assert!(
        error.to_string().contains("could not handle this module"),
        "{error}"
    );
    let candidate = answer(&mut runtime, taken).expect("the worker went on");
    runtime
        .publish(candidate, Expect::Generation(first.generation()))
        .unwrap();
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(2));
}

/// A Kernel requiring an ungranted capability is refused at prepare,
/// inline or on the worker; the running version goes on.
#[test]
fn a_kernel_requiring_what_is_not_granted_is_refused_and_the_running_one_stays() {
    let mut runtime = runtime(Stub::default());
    let first = load(&mut runtime, color(0.25));
    runtime.set_grants([Capability::FieldWrite].into_iter().collect());
    let reads_keys = || {
        constant(
            "test.color@1",
            1.0,
            1,
            &[Capability::FieldWrite, Capability::InputRead],
        )
    };
    let Err(refused) = runtime.prepare_module(reads_keys(), "") else {
        panic!("taken inline");
    };
    assert!(
        refused
            .to_string()
            .contains("requires input.read, which this session does not grant"),
        "{refused}"
    );
    runtime.set_background(true);
    let ticket = runtime.prepare_later(reads_keys(), String::new());
    assert!(matches!(
        answer(&mut runtime, ticket),
        Err(PrepareError::NotGranted { .. })
    ));
    assert_eq!(
        runtime.kernel(&id()).unwrap().generation(),
        first.generation()
    );
    assert_eq!(red(&mut runtime, &first), 0.25);
    // What is granted is taken.
    let plain = runtime.prepare_module(color(0.5), "").unwrap();
    runtime
        .publish(plain, Expect::Generation(first.generation()))
        .unwrap();
}

#[test]
fn a_backend_that_answers_with_the_wrong_count_of_fields_is_refused() {
    let mut runtime = runtime(Stub {
        losing: true,
        ..Stub::default()
    });
    let version = load(
        &mut runtime,
        constant("test.two@1", 1.0, 2, &[Capability::FieldWrite]),
    );
    let before = runtime.host().live();
    let error = runtime.execute(&version, &invocation()).unwrap_err();
    assert!(
        error.to_string().contains("declares 2 output(s)"),
        "{error}"
    );
    assert_eq!(
        runtime.host().live(),
        before,
        "what it did give is released"
    );
}

#[test]
fn native_operations_are_published_by_the_runtime_and_cannot_be_replaced() {
    let mut runtime = runtime(Stub::default());
    let raster = runtime.kernel(&NativeOp::Rasterize.id()).unwrap();
    assert_eq!(raster.generation(), Generation(0));
    assert_eq!(raster.native(), Some(NativeOp::Rasterize));
    assert!(raster.module().is_none());
    assert!(raster.source().contains("native.rasterize@1"));

    // A Kernel under the native id is refused, whatever it expects.
    let candidate = runtime
        .prepare_module(
            constant("native.rasterize@1", 1.0, 1, &[Capability::FieldWrite]),
            "",
        )
        .unwrap();
    let error = runtime.publish(candidate, Expect::Any).unwrap_err();
    assert!(error.to_string().contains("native operation"), "{error}");
    assert_eq!(
        runtime
            .kernel(&NativeOp::Rasterize.id())
            .unwrap()
            .generation(),
        Generation(0)
    );
}
