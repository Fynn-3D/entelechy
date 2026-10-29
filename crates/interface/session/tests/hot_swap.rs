//! Hot reload from source: a Kernel the compiler refuses leaves the
//! running one, and what the reference backend runs. The swap itself is
//! `entelechy-kernel`'s (`crates/core/kernel/tests/swap.rs`).

use entelechy_kernel::{Context, Domain, Generation, Invocation, KernelId, Runtime, Value};

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
    version: &entelechy_kernel::KernelVersion,
    input: entelechy_kernel::Handle,
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

    let error = entelechy_compiler::compile(&source("nonsense")).unwrap_err();
    assert!(
        error.to_string().contains("`nonsense` is not defined"),
        "{error}"
    );
    let current = runtime.kernel(&id()).unwrap();
    assert_eq!(current.generation(), Generation(1));
    assert_eq!(run(&mut runtime, &current, input), [0.25, 0.5, 0.75, 1.0]);
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

/// The CPU backend executes the runtime's native rasterizer; the
/// registry's half of native operations is `entelechy-kernel`'s.
#[test]
fn the_cpu_backend_runs_the_native_rasterizer() {
    use entelechy_kernel::NativeOp;
    let mut runtime = cpu(1 << 16);
    let raster = runtime.kernel(&NativeOp::Rasterize.id()).unwrap();
    assert_eq!(raster.manifest().inputs.len(), 2);
    assert_eq!(raster.manifest().outputs[0].ty.to_string(), "Grid<S2>");

    // A triangle over an 8x8 grid.
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
            entelechy_kernel::FieldData::from_storage(
                entelechy_rules::raster::vertex_schema(),
                Domain::Rows(3),
                words,
            )
            .unwrap(),
        )
        .unwrap();
    let triangles = host
        .adopt(
            entelechy_kernel::FieldData::from_storage(
                entelechy_rules::raster::triangle_schema(),
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
fn cpu(budget_words: usize) -> entelechy_kernel::Runtime {
    entelechy_kernel::Runtime::with_backend(budget_words, Box::new(entelechy_cpu::CpuBackend))
}

/// Compiles and publishes a Kernel as a session does: compiling is the
/// caller's step, the runtime takes validated IR.
fn load(
    runtime: &mut entelechy_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<entelechy_kernel::KernelVersion> {
    let module = entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}
