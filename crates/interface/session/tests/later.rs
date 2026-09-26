use orla_kernel::{Context, Domain, Invocation};

const COLOR: &str = "kernel test.color@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(c.r * 2.0, c.g, c.b, c.a); } }";

fn invocation(input: orla_kernel::Handle) -> Invocation {
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
    // A run over the wrong domain fails as on the frame.
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

/// An off-frame run reserves its room when it starts, in the caller's
/// order: its input copy and its output twice. The reservation counts
/// against the budget until redeemed, so Host and run together stay
/// within it.
#[test]
fn a_run_off_the_frame_is_given_its_room_when_it_starts() {
    let mut runtime = cpu(64);
    let version = load(&mut runtime, COLOR);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    // 4 + 48 of 64 words: 12 left, the room a 1 by 1 run takes (4 for the
    // copy, 8 for its output twice).
    let held = runtime.host_mut().create_image(12, 1).unwrap();
    let run = runtime.execute_later(&version, &invocation(input)).unwrap();
    // The room is the run's until redeemed: the Host has none left.
    assert!(runtime.host_mut().create_image(1, 1).is_err());
    let outputs = runtime.wait_run(run).unwrap();
    assert_eq!(outputs.len(), 1);
    // Redeemed, the room returns and the output is held as a Field.
    assert!(runtime.host_mut().create_image(1, 2).is_ok());

    // A run that does not fit is refused at start.
    runtime.host_mut().release(held).unwrap();
    // 4 + 4 + 8 + 40: 8 words left, fewer than the run takes.
    let _full = runtime.host_mut().create_image(10, 1).unwrap();
    let error = runtime
        .execute_later(&version, &invocation(input))
        .unwrap_err()
        .to_string();
    assert!(error.contains("budget"), "{error}");
    // Nothing of it stays reserved.
    assert!(runtime.host_mut().create_image(1, 1).is_ok());
}

/// A discarded run returns its room at once.
#[test]
fn a_discarded_run_gives_its_room_back() {
    let mut runtime = cpu(64);
    let version = load(&mut runtime, COLOR);
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    let _held = runtime.host_mut().create_image(12, 1).unwrap();
    let run = runtime.execute_later(&version, &invocation(input)).unwrap();
    assert!(runtime.host_mut().create_image(1, 1).is_err());
    runtime.discard_run(run);
    assert!(runtime.host_mut().create_image(3, 1).is_ok());
}

/// A runtime on the reference backend, as a session opens one.
fn cpu(budget_words: usize) -> orla_kernel::Runtime {
    orla_kernel::Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend))
}

/// Compiles and publishes a Kernel as a session does: compiling is the
/// caller's step, the runtime takes validated IR.
fn load(
    runtime: &mut orla_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<orla_kernel::KernelVersion> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}
