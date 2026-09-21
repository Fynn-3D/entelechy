use orla_runtime::{Context, Domain, Invocation, Runtime};

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
    let mut runtime = Runtime::new(64);
    let version = runtime.load(COLOR).unwrap();
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
    let mut runtime = Runtime::new(64);
    let version = runtime.load(COLOR).unwrap();
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
