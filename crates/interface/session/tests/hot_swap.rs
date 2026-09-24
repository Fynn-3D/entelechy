use orla_runtime::{Context, Domain, Expect, Generation, Invocation, KernelId, Runtime, Value};

fn source(red: &str) -> String {
    format!(
        "kernel test.color@1 {{ requires {{ field.read; field.write; }} input source: Image; output result: Image; compute {{ let c = sample(source); return vec4({red}, c.g, c.b, c.a); }} }}"
    )
}

fn id() -> KernelId {
    "test.color@1".parse().unwrap()
}

fn run(
    runtime: &mut Runtime,
    version: &orla_runtime::KernelVersion,
    input: orla_runtime::Handle,
) -> [f32; 4] {
    let invocation = Invocation {
        inputs: vec![input],
        params: vec![],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    };
    let outputs = runtime.execute(version, &invocation).unwrap();
    let pixel = runtime.host().read_pixel(outputs[0], 0, 0).unwrap();
    runtime.host_mut().release(outputs[0]).unwrap();
    pixel
}

#[test]
fn a_failed_candidate_leaves_the_running_kernel_alive() {
    let mut runtime = cpu(64);
    let first = load(&mut runtime, &source("c.r"));
    assert_eq!(first.generation(), Generation(1));
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.25, 0.5, 0.75, 1.0])
        .unwrap();
    assert_eq!(run(&mut runtime, &first, input), [0.25, 0.5, 0.75, 1.0]);

    let error = orla_compiler::compile(&source("nonsense")).unwrap_err();
    assert!(
        error.to_string().contains("`nonsense` is not defined"),
        "{error}"
    );
    let current = runtime.kernel(&id()).unwrap();
    assert_eq!(current.generation(), Generation(1));
    assert_eq!(run(&mut runtime, &current, input), [0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn publish_is_an_atomic_swap_that_keeps_pinned_versions_valid() {
    let mut runtime = cpu(64);
    let first = load(&mut runtime, &source("c.r"));
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.25, 0.5, 0.75, 1.0])
        .unwrap();

    let candidate = prepare(&runtime, &source("1.0")).unwrap();
    assert_eq!(candidate.id(), &id());
    let second = runtime
        .publish(candidate, Expect::Generation(first.generation()))
        .unwrap();
    assert_eq!(second.generation(), Generation(2));
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(2));

    // The old version is still executable by whoever pinned it.
    assert_eq!(run(&mut runtime, &first, input), [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(run(&mut runtime, &second, input), [1.0, 0.5, 0.75, 1.0]);
    assert_eq!(first.source(), source("c.r"));
}

#[test]
fn stale_and_foreign_candidates_are_rejected() {
    let mut runtime = cpu(64);
    let first = load(&mut runtime, &source("c.r"));

    let stale = prepare(&runtime, &source("0.0")).unwrap();
    let error = runtime
        .publish(stale, Expect::Generation(Generation(7)))
        .unwrap_err();
    assert_eq!(error.actual, Some(first.generation()));
    assert!(error.to_string().contains("registry holds g1"), "{error}");

    let duplicate = prepare(&runtime, &source("0.0")).unwrap();
    let error = runtime.publish(duplicate, Expect::Absent).unwrap_err();
    assert_eq!(error.expected, Expect::Absent);
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(1));

    runtime.retire(&id()).unwrap();
    assert!(runtime.kernel(&id()).is_none());
    let fresh = prepare(&runtime, &source("0.0")).unwrap();
    assert_eq!(
        runtime.publish(fresh, Expect::Absent).unwrap().generation(),
        Generation(2)
    );
}

#[test]
fn execution_is_scoped_to_the_manifest_capabilities() {
    let mut runtime = cpu(64);
    let version = load(
        &mut runtime,
        "kernel test.gen@1 { requires { field.write; } output result: Image; compute { return vec4(coord().x, 0.0, 0.0, 1.0); } }",
    );
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(2, 1),
        context: Context::default(),
    };
    let outputs = runtime.execute(&version, &invocation).unwrap();
    assert_eq!(
        runtime.host().read_pixel(outputs[0], 1, 0).unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        runtime.kernels().filter(|v| v.native().is_none()).count(),
        1
    );
    let _ = Value::Float(0.0);
}

#[test]
fn background_preparation_resolves_through_tickets() {
    let mut runtime = cpu(64);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    // Inline: a ticket is ready as soon as it is issued.
    let module = orla_compiler::compile(&source("0.25")).unwrap();
    let ticket = runtime.prepare_later(module, source("0.25"));
    let candidate = runtime.poll(ticket).unwrap().unwrap();
    assert_eq!(candidate.id(), &id());
    assert!(runtime.poll(ticket).is_none(), "a ticket is redeemed once");

    // With a worker: the running version keeps executing until the
    // replacement is collected and published.
    runtime.set_background(true);
    assert!(runtime.background());
    let running = load(&mut runtime, &source("0.5"));
    let module = orla_compiler::compile(&source("0.75")).unwrap();
    let ticket = runtime.prepare_later(module, source("0.75"));
    let mut candidate = None;
    for _ in 0..2000 {
        assert_eq!(run(&mut runtime, &running, input)[0], 0.5);
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
    assert_eq!(run(&mut runtime, &published, input)[0], 0.75);

    // A discarded ticket never surfaces, and stopping the worker keeps
    // what it finished.
    let module = orla_compiler::compile(&source("0.1")).unwrap();
    let dropped = runtime.prepare_later(module, source("0.1"));
    runtime.discard(dropped);
    let module = orla_compiler::compile(&source("0.2")).unwrap();
    let kept = runtime.prepare_later(module, source("0.2"));
    runtime.set_background(false);
    assert!(runtime.poll(dropped).is_none());
    let kept = runtime.poll(kept).unwrap().unwrap();
    let detached = runtime.detach(kept, "fused.k@1".parse().unwrap());
    assert_eq!(run(&mut runtime, &detached, input)[0], 0.2);
    assert!(runtime.kernel(&"fused.k@1".parse().unwrap()).is_none());
}

#[test]
fn native_operations_are_published_by_the_runtime_and_cannot_be_replaced() {
    use orla_runtime::NativeOp;
    let mut runtime = cpu(1 << 16);
    let raster = runtime.kernel(&NativeOp::Rasterize.id()).unwrap();
    assert_eq!(raster.generation(), Generation(0));
    assert_eq!(raster.native(), Some(NativeOp::Rasterize));
    assert!(raster.module().is_none());
    assert_eq!(raster.manifest().inputs.len(), 2);
    assert_eq!(raster.manifest().outputs[0].ty.to_string(), "Grid<S2>");
    assert!(raster.source().contains("native.rasterize@1"));

    // A Kernel under the native id is refused, whatever it expects.
    let candidate = prepare(
        &runtime,
        "kernel native.rasterize@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let error = runtime.publish(candidate, Expect::Any).unwrap_err();
    assert!(error.to_string().contains("native operation"), "{error}");

    // The CPU backend executes it: a triangle over an 8x8 grid.
    let vertex = |clip: [f32; 4]| -> Vec<u32> {
        clip.iter()
            .chain([0.0f32, 0.0, 1.0, 0.5, 0.5].iter())
            .map(|v| v.to_bits())
            .collect()
    };
    let mut words = Vec::new();
    words.extend(vertex([-1.0, -1.0, 0.5, 1.0]));
    words.extend(vertex([1.0, -1.0, 0.5, 1.0]));
    words.extend(vertex([1.0, 1.0, 0.5, 1.0]));
    let host = runtime.host_mut();
    let vertices = host
        .adopt(
            orla_runtime::FieldData::from_storage(
                orla_rules::raster::vertex_schema(),
                Domain::Rows(3),
                words,
            )
            .unwrap(),
        )
        .unwrap();
    let triangles = host
        .adopt(
            orla_runtime::FieldData::from_storage(
                orla_rules::raster::triangle_schema(),
                Domain::Rows(1),
                vec![0, 1, 2],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![vertices, triangles],
        params: vec![],
        domain: Domain::Grid(8, 8),
        context: Context::default(),
    };
    let surface = runtime.execute(&raster, &invocation).unwrap()[0];
    let words = runtime.host().data(surface).unwrap().storage();
    // Bottom-right pixel is inside the triangle, top-left is not.
    let id = |x: usize, y: usize| words[(y * 8 + x) * 7 + 6] as i32;
    assert_eq!(id(7, 7), 0);
    assert_eq!(id(0, 0), -1);
    assert_eq!(f32::from_bits(words[(7 * 8 + 7) * 7]), 0.5);

    // A grid of the wrong shape or layout is refused with the contract.
    let bad = Invocation {
        inputs: vec![triangles, vertices],
        params: vec![],
        domain: Domain::Grid(8, 8),
        context: Context::default(),
    };
    let error = runtime.execute(&raster, &bad).unwrap_err();
    assert!(
        error.to_string().contains("`vertices` must be rows of"),
        "{error}"
    );
}

/// A runtime on the reference backend, as a session opens one.
fn cpu(budget_words: usize) -> orla_runtime::Runtime {
    orla_runtime::Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend))
}

/// Compiles and publishes a Kernel as a session does: the runtime takes
/// validated IR, and compiling is the caller's step.
fn load(
    runtime: &mut orla_runtime::Runtime,
    source: &str,
) -> std::sync::Arc<orla_runtime::KernelVersion> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

/// Compiles and prepares a Kernel without publishing it.
fn prepare(
    runtime: &orla_runtime::Runtime,
    source: &str,
) -> Result<orla_runtime::Candidate, orla_runtime::PrepareError> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.prepare_module(module, source)
}

/// A backend that loses the last Field it makes: a broken contract the
/// runtime must refuse to pass on.
struct Lossy;

impl orla_runtime::Backend for Lossy {
    fn name(&self) -> &'static str {
        "lossy"
    }

    fn prepare(
        &self,
        module: &orla_runtime::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        orla_runtime::Backend::prepare(&orla_cpu::CpuBackend, module)
    }

    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_runtime::Validated,
        invocation: &Invocation,
        scope: &mut orla_runtime::Scope<'_>,
    ) -> Result<Vec<orla_runtime::Handle>, orla_runtime::ExecutionError> {
        let mut outputs = orla_runtime::Backend::execute(
            &orla_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )?;
        if let Some(lost) = outputs.pop() {
            let _ = scope.release(lost);
        }
        Ok(outputs)
    }
}

#[test]
fn a_backend_that_answers_with_the_wrong_count_of_fields_is_refused() {
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 12, Box::new(Lossy));
    let version = load(
        &mut runtime,
        "kernel test.two@1 { requires { field.write; } output a: Image; output b: Grid<Float>; compute { return { a: vec4(1.0, 0.0, 0.0, 1.0), b: 2.0 }; } }",
    );
    let before = runtime.host().live();
    let error = runtime
        .execute(
            &version,
            &Invocation {
                inputs: vec![],
                params: vec![],
                domain: Domain::Grid(2, 2),
                context: Context::default(),
            },
        )
        .unwrap_err();
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

/// A backend that panics on a module containing `PANIC`: a backend bug
/// a live session must survive.
struct Brittle;

impl orla_runtime::Backend for Brittle {
    fn name(&self) -> &'static str {
        "brittle"
    }

    fn prepare(
        &self,
        module: &orla_runtime::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        if module.manifest.id.to_string().contains("panics") {
            panic!("the backend could not handle this module");
        }
        orla_runtime::Backend::prepare(&orla_cpu::CpuBackend, module)
    }

    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_runtime::Validated,
        invocation: &Invocation,
        scope: &mut orla_runtime::Scope<'_>,
    ) -> Result<Vec<orla_runtime::Handle>, orla_runtime::ExecutionError> {
        orla_runtime::Backend::execute(&orla_cpu::CpuBackend, prepared, module, invocation, scope)
    }
}

/// A backend that panics preparing a Kernel rejects it, inline or on the
/// worker: the worker answers the ticket and goes on preparing, and the
/// Kernel running is untouched. (A panic on the worker used to end it
/// without an answer, and the publish waited for ever.)
#[test]
fn a_backend_that_panics_preparing_rejects_the_kernel_and_the_worker_lives() {
    let mut runtime = Runtime::with_backend(1 << 12, Box::new(Brittle));
    let first = load(&mut runtime, &source("c.r"));
    let panics = |id: &str| {
        orla_compiler::compile(&format!(
            "kernel test.{id}@1 {{ requires {{ field.write; }} output result: Image; compute {{ return vec4(1.0, 0.0, 0.0, 1.0); }} }}"
        ))
        .unwrap()
    };
    let Err(inline) = runtime.prepare_module(panics("panics"), "") else {
        panic!("taken inline");
    };
    assert!(
        inline.to_string().contains("could not handle this module"),
        "{inline}"
    );

    runtime.set_background(true);
    let refused = runtime.prepare_later(panics("panics"), String::new());
    let taken = runtime.prepare_later(compile_ok(&source("0.5")), source("0.5"));
    let wait = |runtime: &mut Runtime, ticket| loop {
        if let Some(result) = runtime.poll(ticket) {
            break result;
        }
        std::thread::yield_now();
    };
    let Err(error) = wait(&mut runtime, refused) else {
        panic!("taken on the worker");
    };
    assert!(
        error.to_string().contains("could not handle this module"),
        "{error}"
    );
    let candidate = wait(&mut runtime, taken).expect("the worker went on");
    runtime
        .publish(candidate, Expect::Generation(first.generation()))
        .unwrap();
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(2));
}

fn compile_ok(source: &str) -> orla_runtime::Validated {
    orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"))
}

/// Candidates published as one change are checked before any is swapped
/// in: one stale expectation, and none is published.
#[test]
fn publishing_several_is_all_or_nothing() {
    let mut runtime = cpu(64);
    let first = load(&mut runtime, &source("c.r"));
    let other = |red: &str| {
        let text = source(red).replace("test.color@1", "test.other@1");
        (compile_ok(&text), text)
    };
    let (module, text) = other("0.1");
    let fresh = runtime.prepare_module(module, &text).unwrap();
    let replacement = runtime
        .prepare_module(compile_ok(&source("0.5")), &source("0.5"))
        .unwrap();
    // The replacement expects a generation that is not current.
    let stale = Expect::Generation(Generation(first.generation().0 + 7));
    let refused = runtime.publish_all(vec![(fresh, Expect::Absent), (replacement, stale)]);
    assert!(refused.is_err());
    assert!(runtime.kernel(&"test.other@1".parse().unwrap()).is_none());
    assert_eq!(
        runtime.kernel(&id()).unwrap().generation(),
        first.generation()
    );

    let (module, text) = other("0.1");
    let fresh = runtime.prepare_module(module, &text).unwrap();
    let replacement = runtime
        .prepare_module(compile_ok(&source("0.5")), &source("0.5"))
        .unwrap();
    let published = runtime
        .publish_all(vec![
            (fresh, Expect::Absent),
            (replacement, Expect::Generation(first.generation())),
        ])
        .unwrap();
    assert_eq!(published.len(), 2);
    assert!(runtime.kernel(&"test.other@1".parse().unwrap()).is_some());
    assert_ne!(
        runtime.kernel(&id()).unwrap().generation(),
        first.generation()
    );
}

/// What a Kernel requires is granted by whoever runs the runtime, not
/// taken: one that requires a capability not granted is refused when it
/// is prepared, inline or on the worker, and the version running goes
/// on; one that requires only what is granted is taken.
#[test]
fn a_kernel_requiring_what_is_not_granted_is_refused_and_the_running_one_stays() {
    let mut runtime = cpu(64);
    let first = load(&mut runtime, &source("c.r"));
    runtime.set_grants(
        [
            orla_runtime::Capability::FieldRead,
            orla_runtime::Capability::FieldWrite,
        ]
        .into_iter()
        .collect(),
    );
    let reads_keys = source("float(key(65))").replace(
        "requires { field.read; field.write; }",
        "requires { field.read; field.write; input.read; }",
    );
    let Err(refused) = runtime.prepare_module(compile_ok(&reads_keys), &reads_keys) else {
        panic!("taken inline");
    };
    assert!(
        refused
            .to_string()
            .contains("requires input.read, which this session does not grant"),
        "{refused}"
    );
    runtime.set_background(true);
    let ticket = runtime.prepare_later(compile_ok(&reads_keys), reads_keys.clone());
    let refused = loop {
        if let Some(result) = runtime.poll(ticket) {
            break result;
        }
    };
    assert!(matches!(
        refused,
        Err(orla_runtime::PrepareError::NotGranted { .. })
    ));
    assert_eq!(
        runtime.kernel(&id()).unwrap().generation(),
        first.generation()
    );
    // What is granted is taken.
    let plain = runtime
        .prepare_module(compile_ok(&source("0.5")), &source("0.5"))
        .unwrap();
    runtime
        .publish(plain, Expect::Generation(first.generation()))
        .unwrap();
}
