use orla_runtime::{Context, Domain, Invocation};

const COLOR: &str = "kernel test.color@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(c.r * 2.0, c.g, c.b, c.a); } }";

fn invocation(input: orla_runtime::Handle) -> Invocation {
    Invocation {
        inputs: vec![input],
        params: vec![],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    }
}

#[test]
fn a_run_off_the_frame_lands_as_fields_of_the_host_and_is_redeemed_once() {
    let mut runtime = cpu(64);
    let version = load(&mut runtime, COLOR);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.25, 0.5, 0.75, 1.0])
        .unwrap();
    let run = runtime.execute_later(&version, &invocation(input)).unwrap();
    // The inputs were copied: overwriting or releasing them changes nothing.
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.0, 0.0, 0.0, 0.0])
        .unwrap();
    runtime.host_mut().release(input).unwrap();
    let outputs = runtime.wait_run(run).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        runtime.host().read_pixel(outputs[0], 0, 0).unwrap(),
        [0.5, 0.5, 0.75, 1.0]
    );
    assert!(runtime.poll_run(run).is_none(), "a run is redeemed once");
    assert_eq!(runtime.runs_in_flight(), 0);
}

#[test]
fn a_discarded_run_keeps_nothing_and_a_failed_run_says_why() {
    let mut runtime = cpu(64);
    let version = load(&mut runtime, COLOR);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    let live = runtime.host().live();
    let run = runtime.execute_later(&version, &invocation(input)).unwrap();
    runtime.discard_run(run);
    while runtime.runs_in_flight() > 0 {
        std::thread::yield_now();
    }
    assert!(runtime.poll_run(run).is_none());
    assert_eq!(
        runtime.host().live(),
        live,
        "a discarded run adopts nothing"
    );
    // A run over the wrong domain fails as a run on the frame would.
    let wrong = Invocation {
        domain: Domain::Rows(3),
        ..invocation(input)
    };
    let run = runtime.execute_later(&version, &wrong).unwrap();
    let error = runtime.wait_run(run).unwrap_err().to_string();
    assert!(
        error.contains("Grid") || error.contains("domain"),
        "{error}"
    );
    assert_eq!(runtime.host().live(), live);
}

/// A run off the frame spends the one budget the Host spends: its copy
/// of the inputs and its outputs count with what the Host already holds,
/// so runs on threads of their own do not each get the whole budget.
#[test]
fn a_run_off_the_frame_holds_what_it_holds_against_the_one_budget() {
    let mut runtime = cpu(64);
    let version = load(&mut runtime, COLOR);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    // 4 + 56 of 64 words: room for the run's copy of its input, not for
    // its output as well.
    let _held = runtime.host_mut().create_image(14, 1).unwrap();
    let run = runtime.execute_later(&version, &invocation(input)).unwrap();
    let error = runtime.wait_run(run).unwrap_err().to_string();
    assert!(error.contains("budget"), "{error}");
    // What the run held went back: the Host takes the last 4 words.
    assert!(runtime.host_mut().create_image(1, 1).is_ok());
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
