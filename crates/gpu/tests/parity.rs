//! GPU results must match the CPU interpreter within floating-point
//! tolerance. Tests skip when no adapter is available.

use std::path::Path;

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_runtime::Backend;

/// Downloads a resident GPU output into Host memory and returns its pixels.
fn materialized(gpu: &GpuBackend, host: &mut Host, handle: orla_abi::ImageHandle) -> Vec<[f32; 4]> {
    let (width, height) = host.image_size(handle).unwrap();
    let device = host
        .resident(handle)
        .unwrap()
        .expect("gpu outputs are resident");
    let bytes = Backend::download(gpu, device, width as usize * height as usize * 16).unwrap();
    let pixels: Vec<[f32; 4]> = bytes
        .chunks_exact(16)
        .map(|c| {
            std::array::from_fn(|i| {
                f32::from_le_bytes([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]])
            })
        })
        .collect();
    host.materialize(handle, pixels).unwrap();
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

#[test]
fn pointwise_kernels_match_exactly_enough() {
    let Some(gpu) = gpu() else { return };
    let gradient: Vec<[f32; 4]> = (0..64 * 32)
        .map(|i| [(i % 64) as f32 / 63.0, (i / 64) as f32 / 31.0, 0.5, 1.0])
        .collect();
    let source = "kernel test.pointwise@1 { requires { image.read; image.write; time.read; } input source: Image; param amount: Float = 1.0; param tint: Vec3 = vec3(1.0, 0.5, 0.25); param flip: Bool = true; param n: Int = 3; output result: Image;
        fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; }
        compute {
            let c = sample(source);
            let g = luma(c);
            let s = size(source);
            let far = fetch(source, coord() + vec2(100.0, -100.0));
            let k = if flip { -1.0 } else { 1.0 };
            let v = mix(vec3(c.r, c.g, c.b), tint * g, amount) * k + vec3(float(n), s.x / 64.0, far.r);
            return vec4(v.x + sin(time()), v.y, v.z, float(n < 4 && !flip == false));
        } }";
    let (cpu, gpu_pixels) = both(
        &gpu,
        source,
        &[gradient],
        (64, 32),
        vec![
            Value::Float(0.6),
            Value::Vec3([1.0, 0.5, 0.25]),
            Value::Bool(true),
            Value::Int(3),
        ],
        Domain {
            width: 64,
            height: 32,
        },
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-4), 0.0);
}

#[test]
fn loops_and_branches_match() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.flow@1 { requires { image.write; } param steps: Int = 5; output result: Image;
        fn sum_to(n: Int) -> Int { return loop 100 with acc = 0 until index() >= n { acc + index() }; }
        compute {
            let c = coord();
            let side = if c.x < 8.0 { if c.y < 4.0 { 0.25 } else { 0.75 } } else { let z = 0.5; z };
            let total = sum_to(steps + int(c.x));
            let march = loop 64 with s = vec2(0.0, 1.0) until s.y < 0.01 { vec2(s.x + 1.0, s.y * 0.5) };
            let n = normalize(vec3(0.0, 0.0, 0.0));
            return vec4(side, float(total), march.x + n.x, pow(2.0, c.y) + step(4.0, c.y));
        } }";
    let (cpu, gpu_pixels) = both(
        &gpu,
        source,
        &[],
        (1, 1),
        vec![Value::Int(5)],
        Domain {
            width: 16,
            height: 8,
        },
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-4), 0.0);
}

#[test]
fn the_examples_match_within_tolerance() {
    let Some(gpu) = gpu() else { return };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
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
        Domain {
            width: 120,
            height: 100,
        },
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
        Domain {
            width: 100,
            height: 65,
        },
    );
    let fraction = mismatch(&cpu, &gpu_pixels, 2e-2);
    assert!(fraction < 0.03, "ray march differs on {fraction} of pixels");
}

#[test]
fn rejected_shaders_report_the_backend_not_a_panic() {
    let Some(gpu) = gpu() else { return };
    let module = compile("kernel test.ok@1 { requires { image.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }").unwrap();
    assert!(gpu.prepare(&module).is_ok());
    eprintln!("adapter: {}", gpu.adapter_name());
}

#[test]
fn structs_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.structs@1 { requires { image.write; } output result: Image;
        struct Ray { origin: Vec3; direction: Vec3; }
        struct March { ray: Ray; t: Float; steps: Int; hit: Bool; }
        fn advance(m: March, d: Float) -> March { return March { ..m, t: m.t + d, steps: m.steps + 1 }; }
        compute {
            let c = coord();
            let init = March { ray: Ray { origin: vec3(0.0, 0.0, 0.0), direction: vec3(0.0, 0.0, -1.0) }, t: 0.0, steps: 0, hit: false };
            let m = loop 32 with s = init until s.hit {
                let next = advance(s, 0.5);
                if next.t >= c.x { March { ..next, hit: true } } else { next }
            };
            let flipped = if c.y > 2.5 { Ray { ..m.ray, origin: vec3(9.0, 9.0, 9.0) } } else { m.ray };
            return vec4(m.t, float(m.steps), flipped.origin.x + m.ray.direction.z, float(m.hit));
        } }";
    let (cpu, gpu_pixels) = both(
        &gpu,
        source,
        &[],
        (1, 1),
        vec![],
        Domain {
            width: 8,
            height: 6,
        },
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-5), 0.0);
}

#[test]
fn input_context_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.input@1 { requires { image.write; input.read; time.read; } output result: Image; compute {
        let d = length(coord() - pointer());
        let k = float(key(65)) + float(key(37)) * 2.0 + float(key(255)) * 4.0 + float(key(-1)) * 8.0 + float(key(300)) * 16.0;
        return vec4(d, float(pointer_down()), k, time());
    } }";
    let module = compile(source).unwrap();
    let mut host = Host::new(1 << 16);
    let mut input = orla_abi::InputState::default();
    input.pointer = [3.0, 1.0];
    input.pointer_down = true;
    input.set_key(65, true);
    input.set_key(255, true);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain {
            width: 6,
            height: 3,
        },
        context: Context { time: 0.5, input },
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let prepared = gpu.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let cpu = host.image(cpu_out).unwrap().pixels().to_vec();
    let gpu_pixels = materialized(&gpu, &mut host, gpu_out);
    assert_eq!(cpu[0], [f32::sqrt(10.0), 1.0, 5.0, 0.5]);
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-5), 0.0);
}

#[test]
fn chains_and_state_stay_resident_until_observed() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 24, backend);
    runtime
        .load("kernel test.zero@1 { requires { image.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }")
        .unwrap();
    runtime
        .load("kernel test.increment@1 { requires { image.read; image.write; } input previous: Image; output result: Image; compute { let c = sample(previous); return vec4(c.r + 1.0, c.g, c.b, 1.0); } }")
        .unwrap();
    runtime
        .load("kernel test.blur@1 { requires { image.read; image.write; } input source: Image; output result: Image; compute { let c = coord(); return (fetch(source, c) + fetch(source, c + vec2(1.0, 0.0))) * 0.5; } }")
        .unwrap();
    let graph = orla_graph::Graph::parse(
        "node seed: test.zero@1 { domain = vec2(8, 8); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node soft: test.blur@1 { source <- count.result; }
         output result <- soft.result;",
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, orla_abi::ImageHandle> =
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
            runtime.host().resident_images(),
            runtime.host().live_images(),
            "nothing came back to the Host during evaluation"
        );
        if frame == 3 {
            let image = runtime.image(report.output).unwrap();
            assert_eq!(image.pixels()[0][0], 3.0, "state advanced on the device");
            assert_eq!(
                runtime.host().resident_images(),
                runtime.host().live_images() - 1
            );
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
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 24, backend);
    runtime
        .load("kernel test.seed@1 { requires { field.write; } output result: Field<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let i = element(); return Particle { position: vec2(float(i), float(i) * 0.5), life: i * 3, alive: i / 2 * 2 == i }; } }")
        .unwrap();
    runtime
        .load("kernel test.step@1 { requires { field.read; field.write; } input previous: Field<Particle>; output result: Field<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = sample(previous); let n = at(previous, element() + 1); return Particle { position: p.position + n.position, life: p.life - 1 + count(previous), alive: !p.alive }; } }")
        .unwrap();
    runtime
        .load("kernel test.draw@1 { requires { field.read; image.write; } input particles: Field<Particle>; output result: Image; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = at(particles, int(coord().x)); return vec4(p.position.x, p.position.y, float(p.life), float(p.alive)); } }")
        .unwrap();
    let graph = orla_graph::Graph::parse(
        "node seed: test.seed@1 { domain = 8; }
         node step: test.step@1 { previous <~ step.result from seed.result; }
         node draw: test.draw@1 { source <- step.result; domain = vec2(8, 1); }
         output result <- draw.result;"
            .replace("source <-", "particles <-")
            .as_str(),
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let none: std::collections::BTreeMap<String, orla_abi::ImageHandle> = Default::default();
    let params: std::collections::BTreeMap<String, Value> = Default::default();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &none, &Context::default(), &params)
        .unwrap();
    assert_eq!(
        runtime.host().resident_images(),
        runtime.host().live_images()
    );
    let image = runtime.image(report.output).unwrap();
    // Element 2 after one step: position (2,1)+(3,1.5), life 6-1+8, alive flipped.
    assert_eq!(image.pixels()[2], [5.0, 2.5, 13.0, 0.0]);
    // Element 7 reads its clamped neighbour (itself).
    assert_eq!(image.pixels()[7], [14.0, 7.0, 28.0, 1.0]);
}
