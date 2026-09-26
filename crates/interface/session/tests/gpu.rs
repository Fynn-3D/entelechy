//! The GPU held to the interpreter where a Kernel comes out of a scene:
//! the examples' own Kernels, and graphs that keep their state on the
//! device. They need the scene language and the runtime together, which
//! only the session's layer holds.

use std::path::Path;

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_rules::Backend;

/// Downloads a resident GPU output into Host memory and returns its pixels.
fn materialized(gpu: &GpuBackend, host: &mut Host, handle: orla_abi::Handle) -> Vec<[f32; 4]> {
    let byte_len = host
        .schema(handle)
        .unwrap()
        .storage_words(host.domain(handle).unwrap().count())
        * 4;
    let device = host
        .resident(handle)
        .unwrap()
        .expect("gpu outputs are resident");
    let bytes = Backend::download(gpu, device, byte_len).unwrap();
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    host.materialize(handle, words).unwrap();
    host.image(handle).unwrap().pixels().to_vec()
}

fn gpu() -> Option<GpuBackend> {
    match GpuBackend::new() {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

/// Runs a kernel on both backends and returns (cpu, gpu) pixels.
fn both(
    gpu: &GpuBackend,
    source: &str,
    inputs: &[Vec<[f32; 4]>],
    input_size: (u32, u32),
    params: Vec<Value>,
    domain: Domain,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    both_module(gpu, &module, inputs, input_size, params, domain)
}

fn both_module(
    gpu: &GpuBackend,
    module: &orla_compiler::Validated,
    inputs: &[Vec<[f32; 4]>],
    input_size: (u32, u32),
    params: Vec<Value>,
    domain: Domain,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut host = Host::new(1 << 24);
    let handles: Vec<_> = inputs
        .iter()
        .map(|pixels| {
            host.adopt_image(
                orla_abi::Image::from_pixels(input_size.0, input_size.1, pixels.clone()).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let invocation = Invocation {
        inputs: handles,
        params,
        domain,
        context: Context {
            time: 1.25,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = CpuBackend.execute(module, &invocation, &mut scope).unwrap()[0];
    let prepared = gpu
        .prepare(module)
        .unwrap_or_else(|error| panic!("prepare failed: {error}"));
    let mut scope = Scope::new(&mut host, all());
    let gpu_out = Backend::execute(gpu, prepared.as_ref(), module, &invocation, &mut scope)
        .unwrap_or_else(|error| panic!("gpu execution failed: {error}"))[0];
    (
        host.image(cpu_out).unwrap().pixels().to_vec(),
        materialized(gpu, &mut host, gpu_out),
    )
}

/// The inline Kernel a `.osn` document carries in the named node.
fn inline_kernel(path: &Path, node: &str) -> orla_compiler::Validated {
    let text = std::fs::read_to_string(path).unwrap();
    let document = orla_language::parse_document(&text).unwrap_or_else(|error| panic!("{error}"));
    let decl = document
        .nodes
        .iter()
        .find(|decl| decl.name.value == node)
        .unwrap_or_else(|| panic!("node `{node}` in {}", path.display()));
    let orla_language::NodeKind::Inline(kernel) = &decl.kind else {
        panic!("node `{node}` is not inline");
    };
    orla_compiler::compile_kernel(kernel, None).unwrap_or_else(|error| panic!("{error}"))
}

/// Fraction of pixels whose components differ by more than `tolerance`.
fn mismatch(cpu: &[[f32; 4]], gpu: &[[f32; 4]], tolerance: f32) -> f32 {
    assert_eq!(cpu.len(), gpu.len());
    let bad = cpu
        .iter()
        .zip(gpu)
        .filter(|(a, b)| {
            a.iter()
                .zip(*b)
                .any(|(x, y)| (x - y).abs() > tolerance || x.is_nan() != y.is_nan())
        })
        .count();
    bad as f32 / cpu.len() as f32
}

/// Compiles and publishes a Kernel as a session does: the runtime takes
/// validated IR, and compiling is the caller's step.
fn load(
    runtime: &mut orla_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<orla_kernel::KernelVersion> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

#[test]
fn the_examples_match_within_tolerance() {
    let Some(gpu) = gpu() else { return };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let ball = inline_kernel(&root.join("material-ball/scene.osn"), "ball");
    let (cpu, gpu_pixels) = both_module(
        &gpu,
        &ball,
        &[],
        (1, 1),
        vec![
            Value::Vec3([0.86, 0.32, 0.18]),
            Value::Float(0.35),
            Value::Float(0.0),
            Value::Vec3([1.0, 0.96, 0.9]),
            Value::Float(0.7),
        ],
        Domain::Grid(120, 100),
    );
    let fraction = mismatch(&cpu, &gpu_pixels, 2e-3);
    assert!(
        fraction < 0.01,
        "material ball differs on {fraction} of pixels"
    );

    let march = std::fs::read_to_string(root.join("raymarch/kernels/march.orla")).unwrap();
    let (cpu, gpu_pixels) = both(
        &gpu,
        &march,
        &[],
        (1, 1),
        vec![Value::Float(0.35), Value::Int(96)],
        Domain::Grid(100, 65),
    );
    let fraction = mismatch(&cpu, &gpu_pixels, 2e-2);
    assert!(fraction < 0.03, "ray march differs on {fraction} of pixels");
}

#[test]
fn chains_and_state_stay_resident_until_observed() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.zero@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    load(
        &mut runtime,
        "kernel test.increment@1 { requires { field.read; field.write; } input previous: Image; output result: Image; compute { let c = sample(previous); return vec4(c.r + 1.0, c.g, c.b, 1.0); } }",
    );
    load(
        &mut runtime,
        "kernel test.blur@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = coord(); return (fetch(source, c) + fetch(source, c + vec2(1.0, 0.0))) * 0.5; } }",
    );
    let graph = orla_scene::parse(
        "node seed: test.zero@1 { domain = vec2(8, 8); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node soft: test.blur@1 { source <- count.result; }
         output result <- soft.result;",
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, orla_abi::Handle> =
        std::collections::BTreeMap::new();
    let no_params: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for frame in 1..=3 {
        let report = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &empty,
                &Context::default(),
                &no_params,
            )
            .unwrap();
        assert_eq!(
            runtime.host().resident_count(),
            runtime.host().live(),
            "nothing came back to the Host during evaluation"
        );
        if frame == 3 {
            let image = runtime.image(report.output).unwrap();
            assert_eq!(image.pixels()[0][0], 3.0, "state advanced on the device");
            assert_eq!(runtime.host().resident_count(), runtime.host().live() - 1);
        }
    }
    // The observed image is usable as an input again: it is re-uploaded once.
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &empty,
            &Context::default(),
            &no_params,
        )
        .unwrap();
    assert_eq!(runtime.image(report.output).unwrap().pixels()[0][0], 4.0);
}

#[test]
fn fields_flow_through_the_gpu_exactly() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.seed@1 { requires { field.write; } output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let i = element(); return Particle { position: vec2(float(i), float(i) * 0.5), life: i * 3, alive: i / 2 * 2 == i }; } }",
    );
    load(
        &mut runtime,
        "kernel test.step@1 { requires { field.read; field.write; } input previous: Rows<Particle>; output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = sample(previous); let n = at(previous, element() + 1); return Particle { position: p.position + n.position, life: p.life - 1 + count(previous), alive: !p.alive }; } }",
    );
    load(
        &mut runtime,
        "kernel test.draw@1 { requires { field.read; field.write; } input particles: Rows<Particle>; output result: Image; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = at(particles, int(coord().x)); return vec4(p.position.x, p.position.y, float(p.life), float(p.alive)); } }",
    );
    let graph = orla_scene::parse(
        "node seed: test.seed@1 { domain = 8; }
         node step: test.step@1 { previous <~ step.result from seed.result; }
         node draw: test.draw@1 { source <- step.result; domain = vec2(8, 1); }
         output result <- draw.result;"
            .replace("source <-", "particles <-")
            .as_str(),
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let none: std::collections::BTreeMap<String, orla_abi::Handle> = Default::default();
    let params: std::collections::BTreeMap<String, Value> = Default::default();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &none, &Context::default(), &params)
        .unwrap();
    assert_eq!(runtime.host().resident_count(), runtime.host().live());
    let image = runtime.image(report.output).unwrap();
    // Element 2 after one step: position (2,1)+(3,1.5), life 6-1+8, alive flipped.
    assert_eq!(image.pixels()[2], [5.0, 2.5, 13.0, 0.0]);
    // Element 7 reads its clamped neighbour (itself).
    assert_eq!(image.pixels()[7], [14.0, 7.0, 28.0, 1.0]);
}
