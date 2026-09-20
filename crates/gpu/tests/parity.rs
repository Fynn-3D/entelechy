//! GPU results must match the CPU interpreter within floating-point
//! tolerance. Tests skip when no adapter is available.

use std::path::Path;

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_runtime::Backend;

/// Downloads a resident GPU output into Host memory and returns its pixels.
fn materialized(gpu: &GpuBackend, host: &mut Host, handle: orla_abi::Handle) -> Vec<[f32; 4]> {
    let byte_len = host.domain(handle).unwrap().count() * host.schema(handle).unwrap().words() * 4;
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
    let source = "kernel test.pointwise@1 { requires { field.read; field.write; time.read; } input source: Image; param amount: Float = 1.0; param tint: Vec3 = vec3(1.0, 0.5, 0.25); param flip: Bool = true; param n: Int = 3; output result: Image;
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
        Domain::Grid(64, 32),
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-4), 0.0);
}

#[test]
fn math_builtins_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let gradient: Vec<[f32; 4]> = (0..64 * 32)
        .map(|i| {
            [
                (i % 64) as f32 / 63.0 * 4.0 - 2.0,
                (i / 64) as f32 / 31.0 * 2.0 - 1.0,
                0.5,
                1.0,
            ]
        })
        .collect();
    let source = "kernel test.math@1 { requires { field.read; field.write; } input source: Image; output result: Image;
        compute {
            let c = sample(source);
            let v = vec3(c.r, c.g, c.b);
            let a = clamp(v, -0.5, 0.5) + min(v, 0.25) + max(v, -0.25) + mix(v, v * 2.0, vec3(0.25, 0.5, 0.75)) + pow(abs(v) + 0.1, 0.5);
            let b = ceil(v) + round(v * 3.0) + sign(v) + log(abs(v) + 1.0) + tan(v * 0.5) + asin(clamp(v, -1.0, 1.0)) + acos(clamp(v, -1.0, 1.0)) + atan(v);
            return vec4(a.x + b.x, a.y + b.y, a.z + b.z, float(sign(int(c.r * 3.0))));
        } }";
    let (cpu, gpu_pixels) = both(
        &gpu,
        source,
        &[gradient],
        (64, 32),
        vec![],
        Domain::Grid(64, 32),
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-4), 0.0);
}

#[test]
fn loops_and_branches_match() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.flow@1 { requires { field.write; } param steps: Int = 5; output result: Image;
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
        Domain::Grid(16, 8),
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
fn rejected_shaders_report_the_backend_not_a_panic() {
    let Some(gpu) = gpu() else { return };
    let module = compile("kernel test.ok@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }").unwrap();
    assert!(gpu.prepare(&module).is_ok());
    eprintln!("adapter: {}", gpu.adapter_name());
}

#[test]
fn structs_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.structs@1 { requires { field.write; } output result: Image;
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
    let (cpu, gpu_pixels) = both(&gpu, source, &[], (1, 1), vec![], Domain::Grid(8, 6));
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-5), 0.0);
}

#[test]
fn input_context_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.input@1 { requires { field.write; input.read; time.read; } output result: Image; compute {
        let d = length(coord() - pointer()) + window().y - scale();
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
        domain: Domain::Grid(6, 3),
        context: Context {
            time: 0.5,
            rate: 0.0,
            input,
            window: [6.0, 3.0],
            scale: 2.0,
        },
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
    assert_eq!(cpu[0], [f32::sqrt(10.0) + 1.0, 1.0, 5.0, 0.5]);
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-5), 0.0);
}

#[test]
fn chains_and_state_stay_resident_until_observed() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 24, backend);
    runtime
        .load("kernel test.zero@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }")
        .unwrap();
    runtime
        .load("kernel test.increment@1 { requires { field.read; field.write; } input previous: Image; output result: Image; compute { let c = sample(previous); return vec4(c.r + 1.0, c.g, c.b, 1.0); } }")
        .unwrap();
    runtime
        .load("kernel test.blur@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = coord(); return (fetch(source, c) + fetch(source, c + vec2(1.0, 0.0))) * 0.5; } }")
        .unwrap();
    let graph = orla_graph::Graph::parse(
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
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 24, backend);
    runtime
        .load("kernel test.seed@1 { requires { field.write; } output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let i = element(); return Particle { position: vec2(float(i), float(i) * 0.5), life: i * 3, alive: i / 2 * 2 == i }; } }")
        .unwrap();
    runtime
        .load("kernel test.step@1 { requires { field.read; field.write; } input previous: Rows<Particle>; output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = sample(previous); let n = at(previous, element() + 1); return Particle { position: p.position + n.position, life: p.life - 1 + count(previous), alive: !p.alive }; } }")
        .unwrap();
    runtime
        .load("kernel test.draw@1 { requires { field.read; field.write; } input particles: Rows<Particle>; output result: Image; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = at(particles, int(coord().x)); return vec4(p.position.x, p.position.y, float(p.life), float(p.alive)); } }")
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

#[test]
fn preparing_on_the_worker_never_disturbs_the_executing_thread() {
    let Some(gpu) = gpu() else { return };
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 22, Box::new(gpu));
    runtime.set_background(true);
    let device = runtime
        .backend()
        .as_any()
        .downcast_ref::<GpuBackend>()
        .unwrap()
        .device()
        .clone();
    let source = |value: &str| {
        format!(
            "kernel test.flat@1 {{ requires {{ field.write; }} output result: Image; compute {{ return vec4({value}, 0.5, 0.25, 1.0); }} }}"
        )
    };
    let running = runtime.load(&source("0.125")).unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(64, 64),
        context: orla_abi::Context::default(),
    };
    // A source the device has not compiled before, long enough that the
    // worker's job overlaps everything below.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let value = format!("0.{:03}", nanos % 1000);
    let mut body = String::from("let v0 = 0.0;");
    for step in 1..400 {
        body.push_str(&format!(
            " let v{step} = sin(v{} * {value} + {step}.0);",
            step - 1
        ));
    }
    let long = format!(
        "kernel test.flat@1 {{ requires {{ field.write; }} output result: Image; compute {{ {body} return vec4({value}, 0.5, 0.25, 1.0 + 0.0 * v399); }} }}"
    );
    let module = compile(&long).unwrap();
    // This thread opens an error scope, frames push and pop their own
    // while the worker's scope is open, and the outer scope closes before
    // the worker's does. wgpu keeps scope stacks per thread only with its
    // `std` feature; on one shared stack this pop would be out of order.
    let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let ticket = runtime.prepare_later(module, String::new());
    std::thread::sleep(std::time::Duration::from_millis(20));
    for _ in 0..3 {
        let output = runtime.execute(&running, &invocation).unwrap()[0];
        let pixel = runtime.image(output).unwrap().pixels()[0];
        assert_eq!(pixel, [0.125, 0.5, 0.25, 1.0]);
        runtime.host_mut().release(output).unwrap();
    }
    assert!(pollster::block_on(outer.pop()).is_none());
    let mut waited = 0;
    let candidate = loop {
        if let Some(result) = runtime.poll(ticket) {
            break result.unwrap();
        }
        let output = runtime.execute(&running, &invocation).unwrap()[0];
        runtime.host_mut().release(output).unwrap();
        waited += 1;
        assert!(waited < 5000, "the worker never delivered");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    let version = runtime.detach(candidate, "test.later@1".parse().unwrap());
    let output = runtime.execute(&version, &invocation).unwrap()[0];
    let pixel = runtime.image(output).unwrap().pixels()[0];
    assert_eq!(pixel[0], value.parse::<f32>().unwrap());
    runtime.host_mut().release(output).unwrap();
}

#[test]
fn the_rasterizer_matches_the_reference_on_interior_pixels() {
    use orla_cpu::raster::{SURFACE_WORDS, rasterize_words, triangle_schema, vertex_schema};
    let Some(gpu) = gpu() else { return };
    let mut runtime = orla_runtime::Runtime::with_backend(1 << 24, Box::new(gpu));
    // Two triangles: a large one at depth 0.6 and a nearer one at 0.3
    // covering the lower-left, with distinct normals and uvs, at w = 2 so
    // perspective correction is exercised.
    let vertex = |clip: [f32; 4], normal: [f32; 3], uv: [f32; 2]| -> Vec<u32> {
        clip.iter()
            .chain(normal.iter())
            .chain(uv.iter())
            .map(|v| v.to_bits())
            .collect()
    };
    let mut vertices = Vec::new();
    vertices.extend(vertex([-1.8, -1.8, 1.2, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0]));
    vertices.extend(vertex([1.8, -1.8, 1.2, 2.0], [0.0, 0.0, 1.0], [1.0, 0.0]));
    vertices.extend(vertex([0.0, 1.8, 1.2, 2.0], [0.0, 0.0, 1.0], [0.5, 1.0]));
    vertices.extend(vertex([-1.0, -1.0, 0.3, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0]));
    vertices.extend(vertex([0.2, -1.0, 0.3, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0]));
    vertices.extend(vertex([-1.0, 0.2, 0.3, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0]));
    let triangles: Vec<u32> = [0u32, 1, 2, 3, 4, 5].into_iter().collect();
    let (width, height) = (64, 48);
    let host = runtime.host_mut();
    let vertex_rows = host
        .adopt(
            orla_abi::FieldData::from_words(vertex_schema(), Domain::Rows(6), vertices.clone())
                .unwrap(),
        )
        .unwrap();
    let triangle_rows = host
        .adopt(
            orla_abi::FieldData::from_words(triangle_schema(), Domain::Rows(2), triangles.clone())
                .unwrap(),
        )
        .unwrap();
    let raster = runtime
        .kernel(&"native.rasterize@1".parse().unwrap())
        .expect("the rasterizer is published from the start");
    assert!(raster.native().is_some() && raster.module().is_none());
    let invocation = orla_abi::Invocation {
        inputs: vec![vertex_rows, triangle_rows],
        params: vec![],
        domain: Domain::Grid(width, height),
        context: orla_abi::Context::default(),
    };
    let surface = runtime.execute(&raster, &invocation).unwrap()[0];
    runtime.materialize(surface).unwrap();
    let device = runtime.host().data(surface).unwrap().words().to_vec();
    assert_eq!(
        runtime.host().schema(surface).unwrap().to_string(),
        "f,f,f,f,f,f,i"
    );
    let reference = rasterize_words(&vertices, &triangles, width, height);
    // Compare every pixel whose eight neighbours carry the same triangle
    // on the CPU: away from edges, where coverage rules may differ.
    let id_at =
        |words: &[u32], x: u32, y: u32| words[(y * width + x) as usize * SURFACE_WORDS + 6] as i32;
    let mut compared = 0;
    let mut covered = 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let id = id_at(&reference, x, y);
            let interior =
                (0..3).all(|dy| (0..3).all(|dx| id_at(&reference, x + dx - 1, y + dy - 1) == id));
            if !interior {
                continue;
            }
            compared += 1;
            covered += (id >= 0) as usize;
            let base = (y * width + x) as usize * SURFACE_WORDS;
            assert_eq!(id_at(&device, x, y), id, "id at ({x}, {y})");
            for k in 0..6 {
                let (r, d) = (
                    f32::from_bits(reference[base + k]),
                    f32::from_bits(device[base + k]),
                );
                assert!(
                    (r - d).abs() <= 2e-3,
                    "word {k} at ({x}, {y}): cpu {r}, gpu {d}"
                );
            }
        }
    }
    assert!(
        compared > 2000 && covered > 800,
        "{compared} compared, {covered} covered"
    );
}

#[test]
fn reductions_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // Ints exactly, floats within the tree's rounding, over an odd count
    // so the tail path runs, and over an image so the map is a grid.
    let source = "kernel test.stats@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Stats>; struct Stats { bright: Int; sum: Vec4; max: Float; } compute { let c = sample(source); let l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return Stats { bright: if l > 0.5 { 1 } else { 0 }, sum: c, max: l }; } merge(a, b) { return Stats { bright: a.bright + b.bright, sum: a.sum + b.sum, max: max(a.max, b.max) }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (37, 23);
    let mut host = Host::new(1 << 20);
    let image = host
        .adopt_image(
            orla_abi::Image::from_pixels(
                width,
                height,
                (0..width * height)
                    .map(|i| {
                        let f = i as f32 / (width * height) as f32;
                        [f, 1.0 - f, (i % 7) as f32 / 7.0, 1.0]
                    })
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![image],
        params: vec![],
        domain: Domain::Rows(1),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_stats = host.element(cpu_out, 0).unwrap().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let byte_len =
        host.domain(gpu_out).unwrap().count() * host.schema(gpu_out).unwrap().words() * 4;
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, byte_len).unwrap();
    let gpu_stats: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(host.domain(gpu_out).unwrap(), Domain::Rows(1));
    assert_eq!(cpu_stats[0], gpu_stats[0], "bright counts are exact");
    for k in 1..6 {
        let (c, g) = (f32::from_bits(cpu_stats[k]), f32::from_bits(gpu_stats[k]));
        assert!(
            (c - g).abs() <= 1e-3 * c.abs().max(1.0),
            "word {k}: cpu {c}, gpu {g}"
        );
    }
    assert_eq!(
        cpu_stats[0],
        (0..width * height)
            .filter(|i| {
                let f = *i as f32 / (width * height) as f32;
                f * 0.2126 + (1.0 - f) * 0.7152 + ((i % 7) as f32 / 7.0) * 0.0722 > 0.5
            })
            .count() as u32
    );
}

#[test]
fn scans_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // A prefix sum of ints (exact) and a one-pole filter (rounding only),
    // over a block long enough to matter.
    let source = "kernel test.scan@1 { requires { field.read; field.write; } input rows: Rows<Item>; param k: Float = 0.1; output result: Rows<Run>; struct Item { value: Int; x: Float; } struct Run { total: Int; y: Float; } scan with run = Run { total: 0, y: 0.0 } { let it = sample(rows); return Run { total: run.total + it.value, y: run.y + (it.x - run.y) * k }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let count = 801u32;
    let words: Vec<u32> = (0..count)
        .flat_map(|i| {
            [
                ((i * 7) % 13) as i32 as u32,
                ((i as f32) * 0.05).sin().to_bits(),
            ]
        })
        .collect();
    let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
    let mut host = Host::new(1 << 20);
    let rows = host
        .adopt(orla_abi::FieldData::from_words(schema, Domain::Rows(count), words).unwrap())
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![rows],
        params: vec![Value::Float(0.1)],
        domain: Domain::Rows(count),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let byte_len =
        host.domain(gpu_out).unwrap().count() * host.schema(gpu_out).unwrap().words() * 4;
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, byte_len).unwrap();
    let gpu_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(host.domain(gpu_out).unwrap(), Domain::Rows(count));
    for i in 0..count as usize {
        assert_eq!(cpu_words[i * 2], gpu_words[i * 2], "total at {i}");
        let (c, g) = (
            f32::from_bits(cpu_words[i * 2 + 1]),
            f32::from_bits(gpu_words[i * 2 + 1]),
        );
        assert!((c - g).abs() <= 1e-5, "y at {i}: cpu {c}, gpu {g}");
    }
    assert_eq!(
        cpu_words[(count as usize - 1) * 2],
        (0..count).map(|i| (i * 7) % 13).sum::<u32>()
    );
}

#[test]
fn linear_interpolation_matches_between_backends() {
    let Some(gpu) = gpu() else { return };
    // Every pixel of a larger grid reads the small image at a fractional
    // position, so both bilinear paths run at every weight; a struct of
    // floats over rows exercises the fieldwise mixer.
    let source = "kernel test.zoom@1 { requires { field.read; field.write; } input small: Image; output result: Image; compute { let p = (coord() + vec2(0.5, 0.5)) / domain() * size(small); return at_linear(small, p); } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let small: Vec<[f32; 4]> = (0..12)
        .map(|i| {
            [
                (i % 4) as f32 / 3.0,
                (i / 4) as f32 / 2.0,
                ((i * 5) % 7) as f32 / 6.0,
                1.0,
            ]
        })
        .collect();
    let (cpu, device) = both_module(
        &gpu,
        &module,
        &[small],
        (4, 3),
        vec![],
        Domain::Grid(40, 30),
    );
    assert!(mismatch(&cpu, &device, 1e-5) <= 1e-5);
    assert!(
        (cpu[0][0] - 0.0).abs() < 1e-6 && cpu[0][0] < cpu[39][0],
        "left to right ramps"
    );

    let rows = "kernel test.lerp_struct@1 { requires { field.read; field.write; } input points: Rows<P>; output result: Rows<P>; struct P { a: Vec2; b: Float; } compute { return at_linear(points, float(element()) * 0.37 + 0.5); } }";
    let module = compile(rows).unwrap_or_else(|error| panic!("{error}"));
    let words: Vec<u32> = (0..9)
        .flat_map(|i| {
            [
                (i as f32).to_bits(),
                (i as f32 * 2.0).to_bits(),
                (10.0 - i as f32).to_bits(),
            ]
        })
        .collect();
    let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Float; 3]);
    let mut host = Host::new(1 << 16);
    let points = host
        .adopt(orla_abi::FieldData::from_words(schema, Domain::Rows(9), words).unwrap())
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![points],
        params: vec![],
        domain: Domain::Rows(9),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let bytes =
        Backend::download(&gpu, host.resident(gpu_out).unwrap().unwrap(), 9 * 3 * 4).unwrap();
    for (k, chunk) in bytes.chunks_exact(4).enumerate() {
        let g = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let c = f32::from_bits(cpu_words[k]);
        assert!((c - g).abs() <= 1e-5, "word {k}: cpu {c}, gpu {g}");
    }
    assert!(
        (f32::from_bits(cpu_words[3]) - 0.37).abs() < 1e-5,
        "element 1 sits 0.37 past element 0"
    );
}

#[test]
fn remainder_widening_and_vector_equality_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.rem@1 { requires { field.write; } output result: Image; compute { let c = coord(); let x = int(c.x); let y = int(c.y); let r = x % 7; let s = (c.x - 8.0) % 3.0; let cell = vec2(x / 4, y / 4); let picked = cell == vec2(2.0, 1.0); return vec4(float(r) / 7.0, s / 3.0, float(picked), float(vec3(x, y, 1) != vec3(3.0, 3.0, 1.0))); } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let (cpu, device) = both_module(&gpu, &module, &[], (1, 1), vec![], Domain::Grid(16, 8));
    assert!(mismatch(&cpu, &device, 1e-6) <= 1e-6);
    // The bit operators, the shift count modulo 32, and an arithmetic shift
    // of a negative number, exact on both.
    let bits = compile("kernel test.bits@1 { requires { field.write; } output result: Image; compute { let c = coord(); let x = int(c.x); let y = int(c.y) - 4; return vec4(float((x & 5) | (y ^ 3)), float((x << 33) + (y << (x % 4))), float(y >> 1), float((x * 37) ^ (y * 11) & 255)); } }").unwrap_or_else(|error| panic!("{error}"));
    let (bits_cpu, bits_gpu) = both_module(&gpu, &bits, &[], (1, 1), vec![], Domain::Grid(16, 8));
    assert_eq!(mismatch(&bits_cpu, &bits_gpu, 0.0), 0.0);
    assert_eq!(bits_cpu[3][0], (3 & 5 | (-4 ^ 3)) as f32);
    assert_eq!(bits_cpu[3][1], ((3 << 1) + (-4 << 3)) as f32);
    assert_eq!(bits_cpu[3][2], -2.0);
    let at = |x: usize, y: usize| cpu[y * 16 + x];
    assert_eq!(at(9, 0)[0], 2.0 / 7.0);
    assert!(
        (at(3, 0)[1] - (-2.0 / 3.0)).abs() < 1e-6,
        "float remainder keeps the sign of the left side"
    );
    assert_eq!(at(9, 5)[2], 1.0, "cell (2, 1) is picked");
    assert_eq!(at(3, 3)[3], 0.0);
}

#[test]
fn scatters_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // Many drops into a small grid, several per cell, so the order of the
    // float sums matters and both backends must follow the same one; a
    // struct target combined by `max` with ints; `last` over rows.
    let source = "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input canvas: Grid<Cell>; output result: Grid<Cell>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Cell { mass: Float; top: Int; } struct Splat { cell: Vec2; value: Cell; } scatter into canvas with sum { let d = sample(drops); return Splat { cell: d.at, value: Cell { mass: d.weight, top: d.id } }; } }";
    let count = 2000u32;
    let drop_words: Vec<u32> = (0..count)
        .flat_map(|i| {
            let x = (i as f32 * 0.37).sin() * 5.0 + 3.0;
            let y = (i as f32 * 0.11).cos() * 3.0 + 1.5;
            [
                x.to_bits(),
                y.to_bits(),
                (0.1 + (i % 7) as f32 * 0.013).to_bits(),
                i % 11,
            ]
        })
        .collect();
    let drop_schema = orla_abi::Schema::new(vec![
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Int,
    ]);
    let cell_schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Float, orla_abi::Scalar::Int]);
    let canvas_words: Vec<u32> = (0..24).flat_map(|i| [0.0f32.to_bits(), i % 3]).collect();
    for combine in ["sum", "max", "min", "last"] {
        let source = source.replace("with sum", &format!("with {combine}"));
        let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                orla_abi::FieldData::from_words(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    drop_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let canvas = host
            .adopt(
                orla_abi::FieldData::from_words(
                    cell_schema.clone(),
                    Domain::Grid(6, 4),
                    canvas_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = orla_abi::Invocation {
            inputs: vec![drops, canvas],
            params: vec![],
            domain: Domain::Grid(6, 4),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        assert_eq!(host.domain(gpu_out).unwrap(), Domain::Grid(6, 4));
        let byte_len =
            host.domain(gpu_out).unwrap().count() * host.schema(gpu_out).unwrap().words() * 4;
        let device = host.resident(gpu_out).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, byte_len).unwrap();
        let gpu_words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(cpu_words, gpu_words, "scatter with {combine} differs");
        if combine == "sum" {
            let total: f32 = cpu_words
                .chunks_exact(2)
                .map(|c| f32::from_bits(c[0]))
                .sum();
            assert!(total > 10.0, "drops landed: {total}");
        }
    }
    // Ints alone sum with atomics, in parallel; and floats with NaN and
    // both zeros meet `min` and `max` in the one total order both
    // backends use, whichever order the pairs arrive in.
    let source = "kernel test.count@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Tally>; output result: Grid<Tally>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Tally { hits: Int; ids: Int; } struct Splat { cell: Vec2; value: Tally; } scatter into grid with sum { let d = sample(drops); return Splat { cell: d.at, value: Tally { hits: 1, ids: d.id } }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let tally_schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Int]);
    let mut odd_words = drop_words.clone();
    for i in 0..count as usize {
        if i % 97 == 0 {
            odd_words[i * 4 + 2] = f32::NAN.to_bits();
        } else if i % 89 == 0 {
            odd_words[i * 4 + 2] = (-0.0f32).to_bits();
        } else if i % 83 == 0 {
            odd_words[i * 4 + 2] = 0.0f32.to_bits();
        }
    }
    for (label, module, words, schema, grid_words) in [
        (
            "int sum",
            module,
            drop_words.clone(),
            tally_schema.clone(),
            (0..48).map(|_| 0u32).collect::<Vec<_>>(),
        ),
        (
            "min with NaN",
            compile(&source.replace("with sum", "with min").replace(
                "Tally { hits: 1, ids: d.id }",
                "Tally { hits: d.id, ids: 0 }",
            ))
            .unwrap(),
            odd_words.clone(),
            tally_schema.clone(),
            (0..48).map(|_| 5u32).collect(),
        ),
    ] {
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                orla_abi::FieldData::from_words(drop_schema.clone(), Domain::Rows(count), words)
                    .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(orla_abi::FieldData::from_words(schema, Domain::Grid(6, 4), grid_words).unwrap())
            .unwrap();
        let invocation = orla_abi::Invocation {
            inputs: vec![drops, grid],
            params: vec![],
            domain: Domain::Grid(6, 4),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let byte_len =
            host.domain(gpu_out).unwrap().count() * host.schema(gpu_out).unwrap().words() * 4;
        let device = host.resident(gpu_out).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, byte_len).unwrap();
        let gpu_words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(cpu_words, gpu_words, "{label} differs");
        if label == "int sum" {
            let hits: u32 = cpu_words.chunks_exact(2).map(|c| c[0]).sum();
            assert!(hits > 100, "{label}: {hits} drops landed");
        }
    }
    // A hundred thousand drops summed into a small grid: long runs per
    // cell, folded in source order on both sides, bit for bit.
    let many = 100_000u32;
    let many_words: Vec<u32> = (0..many)
        .flat_map(|i| {
            let x = ((i as f32 * 0.37).sin() * 3.0 + 3.0).clamp(0.0, 5.9);
            let y = ((i as f32 * 0.11).cos() * 2.0 + 2.0).clamp(0.0, 3.9);
            [
                x.to_bits(),
                y.to_bits(),
                (0.001 + (i % 13) as f32 * 0.0007).to_bits(),
                i % 11,
            ]
        })
        .collect();
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 24);
    let drops = host
        .adopt(
            orla_abi::FieldData::from_words(drop_schema.clone(), Domain::Rows(many), many_words)
                .unwrap(),
        )
        .unwrap();
    let canvas = host
        .adopt(
            orla_abi::FieldData::from_words(
                cell_schema.clone(),
                Domain::Grid(6, 4),
                canvas_words.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![drops, canvas],
        params: vec![],
        domain: Domain::Grid(6, 4),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, 24 * 2 * 4).unwrap();
    let gpu_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        cpu_words, gpu_words,
        "a hundred thousand drops summed in order differ"
    );
    let landed: u32 = cpu_words.chunks_exact(2).map(|c| c[1]).sum();
    assert!(landed > 100_000, "ids summed: {landed}");

    // Floats with NaN and both zeros under `min` and `max`, on a Float target.
    let source = "kernel test.low@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Float>; output result: Grid<Float>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Splat { cell: Vec2; value: Float; } scatter into grid with min { let d = sample(drops); return Splat { cell: d.at, value: d.weight }; } }";
    for combine in ["min", "max"] {
        let module = compile(&source.replace("with min", &format!("with {combine}"))).unwrap();
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                orla_abi::FieldData::from_words(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    odd_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(
                orla_abi::FieldData::from_words(
                    orla_abi::Schema::new(vec![orla_abi::Scalar::Float]),
                    Domain::Grid(6, 4),
                    (0..24).map(|_| 0.5f32.to_bits()).collect(),
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = orla_abi::Invocation {
            inputs: vec![drops, grid],
            params: vec![],
            domain: Domain::Grid(6, 4),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().words().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let device = host.resident(gpu_out).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, 24 * 4).unwrap();
        let gpu_words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(cpu_words, gpu_words, "float {combine} with NaN differs");
        let nans = cpu_words
            .iter()
            .filter(|w| f32::from_bits(**w).is_nan())
            .count();
        if combine == "max" {
            assert!(nans > 0, "NaN is beyond infinity under max");
        }
    }
}

#[test]
fn memories_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // A machine that walks a grid memory: each step toggles the cell it is
    // on and steps by what it read there, and tallies ints in a rows
    // memory; both memories come out the same on both backends.
    let source = "kernel test.walk@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory screen: Grid<Int>; memory tally: Rows<Cell>; output result: Rows<Vec2>; struct Cell { hits: Int; last: Float; } fn flip(s: Grid<Int>, p: Vec2, seen: Int) -> Int { return put(s, p, (1 - seen) >>> 0); } scan with at_ = vec2(3.0, 2.0) { let seen = at(screen, at_); flip(screen, at_, seen); let t = at(tally, seen); put(tally, seen, Cell { hits: t.hits + 1, last: float(index()) }); put(screen, vec2(-1.0, 0.0), 9); let next = vec2(float((int(at_.x) + 3 + seen * 5) % 16), float((int(at_.y) + 1 + seen) % 8)); return next; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let steps = 300u32;
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let cells = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
    let mut host = Host::new(1 << 20);
    let ticks = host
        .adopt(
            orla_abi::FieldData::from_words(
                ints.clone(),
                Domain::Rows(steps),
                vec![0; steps as usize],
            )
            .unwrap(),
        )
        .unwrap();
    let screen = host
        .adopt(orla_abi::FieldData::from_words(ints, Domain::Grid(16, 8), vec![0; 128]).unwrap())
        .unwrap();
    let tally = host
        .adopt(
            orla_abi::FieldData::from_words(
                cells,
                Domain::Rows(2),
                vec![0, 0.0f32.to_bits(), 0, 0.0f32.to_bits()],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![ticks, screen, tally],
        params: vec![],
        domain: Domain::Rows(steps),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_outputs = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap();
    assert_eq!(cpu_outputs.len(), 3);
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let gpu_outputs =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap();
    assert_eq!(gpu_outputs.len(), 3);
    for (which, (cpu_out, gpu_out)) in cpu_outputs.iter().zip(&gpu_outputs).enumerate() {
        let cpu_words = host.data(*cpu_out).unwrap().words().to_vec();
        assert_eq!(
            host.domain(*gpu_out).unwrap(),
            host.domain(*cpu_out).unwrap()
        );
        let byte_len = cpu_words.len() * 4;
        let device = host.resident(*gpu_out).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, byte_len).unwrap();
        let gpu_words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(cpu_words, gpu_words, "output {which} differs");
    }
    let screen_words = host.data(cpu_outputs[1]).unwrap().words().to_vec();
    assert!(screen_words.contains(&1), "the walk left marks");
    let tally_words = host.data(cpu_outputs[2]).unwrap().words().to_vec();
    assert_eq!(
        tally_words[0] + tally_words[2],
        steps,
        "every step tallied once"
    );
}

#[test]
fn integer_arithmetic_matches_between_backends() {
    // Bits, shifts and the wrapping arithmetic over the whole range of
    // signs: both backends give the same words.
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.ints@1 { requires { field.write; } output result: Rows<Bits>; struct Bits { and: Int; or: Int; xor: Int; shl: Int; shr: Int; ushr: Int; wadd: Int; wsub: Int; wmul: Int; }
        compute {
            let i = element();
            let a = mul_wrap(i, 0x9E3779B1) ^ 0x5555;
            let b = sub_wrap(i * 7, 0x4000);
            return Bits { and: a & b, or: a | b, xor: a ^ b, shl: a << i, shr: a >> i, ushr: a >>> i, wadd: add_wrap(a, b), wsub: sub_wrap(a, b), wmul: mul_wrap(a, b) };
        } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 16);
    let invocation = orla_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(200),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().words().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, cpu_words.len() * 4).unwrap();
    let gpu_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(cpu_words, gpu_words);
    assert!(
        cpu_words[6..].iter().any(|w| *w as i32 > 0)
            && cpu_words[6..].iter().any(|w| (*w as i32) < 0)
    );
}
