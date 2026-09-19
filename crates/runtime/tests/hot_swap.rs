use orla_runtime::{Domain, Expect, Generation, Invocation, KernelId, Runtime, Value};

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
    };
    let outputs = runtime.execute(&version, &invocation).unwrap();
    assert_eq!(
        runtime.host().read_pixel(outputs[0], 1, 0).unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(runtime.kernels().count(), 1);
    let _ = Value::Float(0.0);
}
