use orla_runtime::{Context, Domain, Expect, Generation, Invocation, KernelId, Runtime, Value};

fn source(red: &str) -> String {
    format!(
        "kernel test.color@1 {{ requires {{ image.read; image.write; }} input source: Image; output result: Image; compute {{ let c = sample(source); return vec4({red}, c.g, c.b, c.a); }} }}"
    )
}

fn id() -> KernelId {
    "test.color@1".parse().unwrap()
}

fn run(
    runtime: &mut Runtime,
    version: &orla_runtime::KernelVersion,
    input: orla_runtime::ImageHandle,
) -> [f32; 4] {
    let invocation = Invocation {
        inputs: vec![input],
        params: vec![],
        domain: Domain {
            width: 1,
            height: 1,
        },
        context: Context::default(),
    };
    let outputs = runtime.execute(version, &invocation).unwrap();
    let pixel = runtime.host().read_pixel(outputs[0], 0, 0).unwrap();
    runtime.host_mut().release_image(outputs[0]).unwrap();
    pixel
}

#[test]
fn a_failed_candidate_leaves_the_running_kernel_alive() {
    let mut runtime = Runtime::new(64);
    let first = runtime.load(&source("c.r")).unwrap();
    assert_eq!(first.generation(), Generation(1));
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.25, 0.5, 0.75, 1.0])
        .unwrap();
    assert_eq!(run(&mut runtime, &first, input), [0.25, 0.5, 0.75, 1.0]);

    let error = runtime.prepare(&source("nonsense")).unwrap_err();
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
    let mut runtime = Runtime::new(64);
    let first = runtime.load(&source("c.r")).unwrap();
    let input = runtime.host_mut().create_image(1, 1).unwrap();
    runtime
        .host_mut()
        .write_pixel(input, 0, 0, [0.25, 0.5, 0.75, 1.0])
        .unwrap();

    let candidate = runtime.prepare(&source("1.0")).unwrap();
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
    let mut runtime = Runtime::new(64);
    let first = runtime.load(&source("c.r")).unwrap();

    let stale = runtime.prepare(&source("0.0")).unwrap();
    let error = runtime
        .publish(stale, Expect::Generation(Generation(7)))
        .unwrap_err();
    assert_eq!(error.actual, Some(first.generation()));
    assert!(error.to_string().contains("registry holds g1"), "{error}");

    let duplicate = runtime.prepare(&source("0.0")).unwrap();
    let error = runtime.publish(duplicate, Expect::Absent).unwrap_err();
    assert_eq!(error.expected, Expect::Absent);
    assert_eq!(runtime.kernel(&id()).unwrap().generation(), Generation(1));

    runtime.retire(&id()).unwrap();
    assert!(runtime.kernel(&id()).is_none());
    let fresh = runtime.prepare(&source("0.0")).unwrap();
    assert_eq!(
        runtime.publish(fresh, Expect::Absent).unwrap().generation(),
        Generation(2)
    );
}

#[test]
fn execution_is_scoped_to_the_manifest_capabilities() {
    let mut runtime = Runtime::new(64);
    let version = runtime
        .load("kernel test.gen@1 { requires { image.write; } output result: Image; compute { return vec4(coord().x, 0.0, 0.0, 1.0); } }")
        .unwrap();
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain {
            width: 2,
            height: 1,
        },
        context: Context::default(),
    };
    let outputs = runtime.execute(&version, &invocation).unwrap();
    assert_eq!(
        runtime.host().read_pixel(outputs[0], 1, 0).unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(runtime.kernels().count(), 1);
    let _ = Value::Float(0.0);
}

#[test]
fn background_preparation_resolves_through_tickets() {
    let mut runtime = Runtime::new(64);
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
    let running = runtime.load(&source("0.5")).unwrap();
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
