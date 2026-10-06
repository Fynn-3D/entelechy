//! GPU results match the CPU interpreter within floating-point tolerance.
//! Tests skip when no adapter is available.

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_rules::Backend;

/// Downloads a resident GPU output into Host memory as pixels.
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

/// Runs a Kernel on both backends and returns (cpu, gpu) pixels.
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
fn preparing_on_the_worker_never_disturbs_the_executing_thread() {
    let Some(gpu) = gpu() else { return };
    // The device is taken before the backend goes to the runtime.
    let device = gpu.device().clone();
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 22, Box::new(gpu));
    runtime.set_background(true);
    let source = |value: &str| {
        format!(
            "kernel test.flat@1 {{ requires {{ field.write; }} output result: Image; compute {{ return vec4({value}, 0.5, 0.25, 1.0); }} }}"
        )
    };
    let running = load(&mut runtime, &source("0.125"));
    let invocation = orla_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(64, 64),
        context: orla_abi::Context::default(),
    };
    // A source new to the device, long enough that the worker's compile
    // overlaps everything below.
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
    // This thread's error scopes nest around the worker's, the outer one
    // closing first: wgpu keeps scope stacks per thread.
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
    use orla_rules::raster::{SURFACE_WORDS, rasterize_words, triangle_schema, vertex_schema};
    let Some(gpu) = gpu() else { return };
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, Box::new(gpu));
    // Two triangles at w = 2, for perspective correction: a large one at
    // depth 0.6 and a nearer one at 0.3 over the lower-left, with distinct
    // normals and uvs.
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
            orla_abi::FieldData::from_storage(vertex_schema(), Domain::Rows(6), vertices.clone())
                .unwrap(),
        )
        .unwrap();
    let triangle_rows = host
        .adopt(
            orla_abi::FieldData::from_storage(
                triangle_schema(),
                Domain::Rows(2),
                triangles.clone(),
            )
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
    let device = runtime.host().data(surface).unwrap().storage().to_vec();
    assert_eq!(
        runtime.host().schema(surface).unwrap().to_string(),
        "f,f,f,f,f,f,i"
    );
    let reference = rasterize_words(&vertices, &triangles, width, height);
    // Only pixels whose eight neighbours carry the same triangle on the
    // CPU are compared: coverage rules at edges may differ.
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
    // (the tail path) and an image (a grid map).
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
    let byte_len = host
        .schema(gpu_out)
        .unwrap()
        .storage_words(host.domain(gpu_out).unwrap().count())
        * 4;
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

/// Integer sums, and `max` over a struct carrying a float, a NaN and both
/// zeros, at counts straddling a block's edge: the reference's words.
#[test]
fn prefixes_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    let sum = compile(
        "kernel test.prefix_sum@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; prefix with sum { return sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let worst = compile(
        "kernel test.prefix_max@1 { requires { field.read; field.write; } input rows: Rows<Run>; output result: Rows<Run>; struct Run { i: Int; f: Float; } prefix with max { return sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    // 1 and 10 are one block; 5000 is seventy-one with a ragged last;
    // 70000 exceeds one invocation's walk in a frame.
    for count in [1u32, 10, 5000, 70000] {
        let ints: Vec<u32> = (0..count).map(|i| ((i as i32 % 11) - 5) as u32).collect();
        let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
        let mut host = Host::new(1 << 24);
        let rows = host
            .adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(count), ints).unwrap())
            .unwrap();
        let (cpu_words, gpu_words) = on_both(&gpu, &mut host, &sum, rows, count);
        let mut running = 0i32;
        for (index, (a, b)) in cpu_words.iter().zip(&gpu_words).enumerate() {
            running += (index as i32 % 11) - 5;
            assert_eq!(*a as i32, running, "the reference's own sum at {index}");
            assert_eq!(a, b, "sum at {index} of {count}");
        }

        // A NaN, both zeros and an infinity: `max` orders them by their
        // bits on both backends.
        let odd = [
            f32::NAN,
            -0.0,
            0.0,
            f32::INFINITY,
            -f32::INFINITY,
            3.5,
            -2.0,
        ];
        let words: Vec<u32> = (0..count)
            .flat_map(|i| {
                [
                    ((i as i32 % 11) - 5) as u32,
                    odd[i as usize % odd.len()].to_bits(),
                ]
            })
            .collect();
        let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
        let mut host = Host::new(1 << 24);
        let rows = host
            .adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(count), words).unwrap())
            .unwrap();
        let (cpu_words, gpu_words) = on_both(&gpu, &mut host, &worst, rows, count);
        assert_eq!(cpu_words, gpu_words, "max over {count}");
    }
}

/// Runs a module over one input on both backends and returns both
/// outputs' words.
fn on_both(
    gpu: &GpuBackend,
    host: &mut Host,
    module: &orla_compiler::Validated,
    input: orla_abi::Handle,
    count: u32,
) -> (Vec<u32>, Vec<u32>) {
    let invocation = orla_abi::Invocation {
        inputs: vec![input],
        params: Vec::new(),
        domain: Domain::Rows(count),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
    let prepared = Backend::prepare(gpu, module).unwrap();
    let mut scope = Scope::new(host, all());
    let gpu_out =
        Backend::execute(gpu, prepared.as_ref(), module, &invocation, &mut scope).unwrap()[0];
    let byte_len = host
        .schema(gpu_out)
        .unwrap()
        .storage_words(host.domain(gpu_out).unwrap().count())
        * 4;
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let bytes = Backend::download(gpu, device, byte_len).unwrap();
    let gpu_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (cpu_words, gpu_words)
}

#[test]
fn scans_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // A prefix sum of ints (exact) and a one-pole filter (rounding only).
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
        .adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(count), words).unwrap())
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
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let byte_len = host
        .schema(gpu_out)
        .unwrap()
        .storage_words(host.domain(gpu_out).unwrap().count())
        * 4;
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
    // floats over rows runs the fieldwise mixer.
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
        .adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(9), words).unwrap())
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
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
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
    // Bit operators, shift counts modulo 32, and an arithmetic shift of a
    // negative number, exact on both.
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
    // Several drops per cell of a small grid, so both backends follow one
    // float summation order; a struct target under `max` with ints; `last`
    // over rows.
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
                orla_abi::FieldData::from_storage(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    drop_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let canvas = host
            .adopt(
                orla_abi::FieldData::from_storage(
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
            domain: Domain::Rows(count),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        assert_eq!(host.domain(gpu_out).unwrap(), Domain::Grid(6, 4));
        let byte_len = host
            .schema(gpu_out)
            .unwrap()
            .storage_words(host.domain(gpu_out).unwrap().count())
            * 4;
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
    // Ints sum with atomics in parallel; floats with NaN and both zeros
    // meet `min` and `max` in one total order, in any arrival order.
    let counting = "kernel test.count@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Tally>; output result: Grid<Tally>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Tally { hits: Int; ids: Int; } struct Splat { cell: Vec2; value: Tally; } scatter into grid with sum { let d = sample(drops); return Splat { cell: d.at, value: Tally { hits: 1, ids: d.id } }; } }";
    let module = compile(counting).unwrap_or_else(|error| panic!("{error}"));
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
            compile(&counting.replace("with sum", "with min").replace(
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
                orla_abi::FieldData::from_storage(drop_schema.clone(), Domain::Rows(count), words)
                    .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(
                orla_abi::FieldData::from_storage(schema, Domain::Grid(6, 4), grid_words).unwrap(),
            )
            .unwrap();
        let invocation = orla_abi::Invocation {
            inputs: vec![drops, grid],
            params: vec![],
            domain: Domain::Rows(count),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let byte_len = host
            .schema(gpu_out)
            .unwrap()
            .storage_words(host.domain(gpu_out).unwrap().count())
            * 4;
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
    // A hundred thousand drops into a small grid, folded in source order
    // on both sides, bit for bit.
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
            orla_abi::FieldData::from_storage(drop_schema.clone(), Domain::Rows(many), many_words)
                .unwrap(),
        )
        .unwrap();
    let canvas = host
        .adopt(
            orla_abi::FieldData::from_storage(
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
        domain: Domain::Rows(many),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
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

    // Floats with NaN and both zeros under `min` and `max`, Float target.
    let source = "kernel test.low@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Float>; output result: Grid<Float>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Splat { cell: Vec2; value: Float; } scatter into grid with min { let d = sample(drops); return Splat { cell: d.at, value: d.weight }; } }";
    for combine in ["min", "max"] {
        let module = compile(&source.replace("with min", &format!("with {combine}"))).unwrap();
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                orla_abi::FieldData::from_storage(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    odd_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(
                orla_abi::FieldData::from_storage(
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
            domain: Domain::Rows(count),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = orla_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
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
    // A machine toggles grid-memory cells and steps by what it reads, and
    // tallies ints in a rows memory; both memories match on both backends.
    let source = "kernel test.walk@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory screen: Grid<Int>; memory tally: Rows<Cell>; output result: Rows<Vec2>; struct Cell { hits: Int; last: Float; } fn flip(s: Grid<Int>, p: Vec2, seen: Int) -> Int { return put(s, p, (1 - seen) >>> 0); } scan with at_ = vec2(3.0, 2.0) { let seen = at(screen, at_); flip(screen, at_, seen); let t = at(tally, seen); put(tally, seen, Cell { hits: t.hits + 1, last: float(index()) }); put(screen, vec2(-1.0, 0.0), 9); let next = vec2(float((int(at_.x) + 3 + seen * 5) % 16), float((int(at_.y) + 1 + seen) % 8)); return next; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let steps = 300u32;
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let cells = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
    let mut host = Host::new(1 << 20);
    let ticks = host
        .adopt(
            orla_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(steps),
                vec![0; steps as usize],
            )
            .unwrap(),
        )
        .unwrap();
    let screen = host
        .adopt(orla_abi::FieldData::from_storage(ints, Domain::Grid(16, 8), vec![0; 128]).unwrap())
        .unwrap();
    let tally = host
        .adopt(
            orla_abi::FieldData::from_storage(
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
        let cpu_words = host.data(*cpu_out).unwrap().storage().to_vec();
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
    let screen_words = host.data(cpu_outputs[1]).unwrap().storage().to_vec();
    assert!(screen_words.contains(&1), "the walk left marks");
    let tally_words = host.data(cpu_outputs[2]).unwrap().storage().to_vec();
    assert_eq!(
        tally_words[0] + tally_words[2],
        steps,
        "every step tallied once"
    );
}

#[test]
fn integer_arithmetic_matches_between_backends() {
    // Bits, shifts and wrapping arithmetic over all sign combinations.
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
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
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

#[test]
fn a_step_reads_what_the_step_before_wrote_even_when_it_reads_first() {
    // Each step first reads the memory element the previous step wrote;
    // a barrier between steps keeps that load in the walk.
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.pc@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory regs: Rows<Int>; output result: Rows<Int>; scan with n = 0 { let pc = at(regs, 15); put(regs, 15, pc + 2); return pc; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 16);
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let ticks = host
        .adopt(
            orla_abi::FieldData::from_storage(ints.clone(), Domain::Rows(5), vec![0; 5]).unwrap(),
        )
        .unwrap();
    let regs = host
        .adopt(orla_abi::FieldData::from_storage(ints, Domain::Rows(17), vec![0; 17]).unwrap())
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![ticks, regs],
        params: vec![],
        domain: Domain::Rows(5),
        context: orla_abi::Context::default(),
    };
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let outputs =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap();
    let words = |handle: orla_abi::Handle, n: usize| -> Vec<u32> {
        let device = host.resident(handle).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, n * 4).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    assert_eq!(words(outputs[0], 5), [0, 2, 4, 6, 8]);
    assert_eq!(words(outputs[1], 17)[15], 10);
}

#[test]
fn rows_of_every_length_are_every_element() {
    // Rows fill whole workgroups in strips: each count about a workgroup's
    // and a strip's edge computes each element once.
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.each@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()) * 2.0 + 1.0; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    for count in [1u32, 7, 8, 63, 64, 65, 511, 512, 513, 4095, 4097, 100_003] {
        let mut host = Host::new(1 << 24);
        let invocation = orla_abi::Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(count),
            context: orla_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let device = host.resident(out).unwrap().expect("resident");
        let bytes = Backend::download(&gpu, device, count as usize * 4).unwrap();
        let wrong = bytes
            .chunks_exact(4)
            .enumerate()
            .filter(|(i, c)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != *i as f32 * 2.0 + 1.0)
            .count();
        assert_eq!(wrong, 0, "{count} rows");
    }
}

#[test]
fn long_rows_are_dispatched_as_a_grid() {
    // More elements than 65,535 workgroups of eight: the dispatch folds
    // them and every element is computed.
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.long@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 3 + 1; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let count = 1_100_000u32;
    let mut host = Host::new(1 << 24);
    let invocation = orla_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(count),
        context: orla_abi::Context::default(),
    };
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, count as usize * 4).unwrap();
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for probe in [
        0u32,
        7,
        524_279,
        524_280,
        524_281,
        1_048_559,
        1_048_560,
        count - 1,
    ] {
        assert_eq!(words[probe as usize], probe * 3 + 1, "element {probe}");
    }
    assert!(
        words
            .iter()
            .enumerate()
            .all(|(i, w)| *w == i as u32 * 3 + 1)
    );
}

#[test]
fn a_match_is_a_switch_on_the_gpu() {
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.pick@1 { requires { field.write; } output result: Rows<Int>; compute { let i = element() - 3; let picked = match i { 0 => 100, 1 | 2 => { let doubled = i * 2; doubled }, -1 | -3 => -7, 5 => 5, _ => 42 }; let summed = loop 4 with acc = 0 { acc + match index() { 0 | 3 => 1, _ => 10 } }; return picked * 1000 + summed; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 16);
    let invocation = orla_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(10),
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
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
    assert_eq!(cpu_words[3] as i32, 100_022);
}

#[test]
fn a_merge_may_read_inputs_params_and_the_context() {
    // The fold pass binds what the map pass binds: `merge` reads a Field
    // directly and through a function, a param, and the time.
    let Some(gpu) = gpu() else { return };
    let source = "kernel test.fold@1 { requires { field.read; field.write; time.read; } input xs: Rows<Float>; input weights: Rows<Float>; param k: Float = 0.5; output result: Rows<Float>;
        fn first(w: Rows<Float>) -> Float { return at(w, 0); }
        compute { return sample(xs) * at(weights, element()); }
        merge(a, b) { return a + b + at(weights, 1) * k + first(weights) + time(); } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let floats = orla_abi::Schema::new(vec![orla_abi::Scalar::Float]);
    let mut host = Host::new(1 << 16);
    let count = 37u32;
    let xs = host
        .adopt(
            orla_abi::FieldData::from_storage(
                floats.clone(),
                Domain::Rows(count),
                (0..count).map(|i| (i as f32 * 0.5).to_bits()).collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let weights = host
        .adopt(
            orla_abi::FieldData::from_storage(
                floats,
                Domain::Rows(count),
                (0..count)
                    .map(|i| (1.0 + (i % 3) as f32).to_bits())
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = orla_abi::Invocation {
        inputs: vec![xs, weights],
        params: vec![orla_abi::Value::Float(0.25)],
        domain: Domain::Rows(1),
        context: orla_abi::Context {
            time: 2.0,
            ..orla_abi::Context::default()
        },
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = orla_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let bytes = Backend::download(&gpu, device, 4).unwrap();
    let gpu_value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let cpu_value = f32::from_bits(cpu_words[0]);
    assert!(
        (cpu_value - gpu_value).abs() <= cpu_value.abs() * 1e-5,
        "{cpu_value} vs {gpu_value}"
    );
    // 36 merges each add at(weights, 1) * k + first(weights) + time
    // = 2 * 0.25 + 1 + 2.
    let plain: f32 = (0..count)
        .map(|i| i as f32 * 0.5 * (1.0 + (i % 3) as f32))
        .sum();
    assert!(
        (cpu_value - (plain + 36.0 * 3.5)).abs() < 1e-2,
        "{cpu_value}"
    );
}

/// A loop counter `i` numbered 32 would read `i32`; the Kernel is written
/// at every length so one loop lands there.
#[test]
fn nothing_the_shader_declares_spells_a_type() {
    for length in 0..48 {
        let padding: String = (0..length)
            .map(|k| format!("let p{k} = element() * {k} + {k};"))
            .collect();
        let sum: String = (0..length)
            .map(|k| format!(" + p{k}"))
            .collect::<Vec<_>>()
            .join("");
        let source = format!(
            "kernel test.names@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ {padding} let counted = loop 16 with a = 0 {{ a + int(float(index()) * 0.5) }}; return counted{sum}; }} }}"
        );
        let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
        let shader = orla_gpu::wgsl::emit(&module);
        for declared in ["i32", "u32", "f32", "bool", "vec2", "mat2x2", "array"] {
            for form in [
                format!("var {declared}:"),
                format!("let {declared}:"),
                format!("var {declared} ="),
            ] {
                assert!(
                    !shader.contains(&form),
                    "with {length} bindings the shader declares `{form}`:\n{shader}"
                );
            }
        }
    }
}

/// An uncovered fragment draws nothing and settles no depth, so the far
/// triangle shows through the near one's holes, as on the reference.
#[test]
fn a_cutout_drops_the_fragments_its_mask_does_not_cover() {
    use orla_rules::raster::{
        Cutout, SURFACE_WORDS, mask_schema, rasterize_masked, triangle_schema, vertex_schema,
    };
    let Some(gpu) = gpu() else { return };
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, Box::new(gpu));
    let vertex = |clip: [f32; 4], normal: [f32; 3], uv: [f32; 2]| -> Vec<u32> {
        clip.iter()
            .chain(normal.iter())
            .chain(uv.iter())
            .map(|v| v.to_bits())
            .collect()
    };
    // A far quad and a near one over it, uv spanning the mask.
    let mut vertices = Vec::new();
    // The far quad sits on one covered texel, so it is drawn whole; the
    // near one spans the checkerboard.
    let solid = [0.0625f32, 0.0625];
    vertices.extend(vertex([-1.0, -1.0, 0.8, 1.0], [0.0, 0.0, 1.0], solid));
    vertices.extend(vertex([1.0, -1.0, 0.8, 1.0], [0.0, 0.0, 1.0], solid));
    vertices.extend(vertex([1.0, 1.0, 0.8, 1.0], [0.0, 0.0, 1.0], solid));
    vertices.extend(vertex([-1.0, 1.0, 0.8, 1.0], [0.0, 0.0, 1.0], solid));
    vertices.extend(vertex([-1.0, -1.0, 0.3, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0]));
    vertices.extend(vertex([1.0, -1.0, 0.3, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0]));
    vertices.extend(vertex([1.0, 1.0, 0.3, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0]));
    vertices.extend(vertex([-1.0, 1.0, 0.3, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0]));
    let triangles: Vec<u32> = [0u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        .into_iter()
        .collect();
    // The mask covers the near quad in eight-by-eight squares.
    let (mask_w, mask_h) = (8u32, 8u32);
    let mut mask = Vec::new();
    for y in 0..mask_h {
        for x in 0..mask_w {
            let on = (x + y) % 2 == 0;
            for k in 0..4 {
                let value: f32 = if k == 3 && !on { 0.0 } else { 1.0 };
                mask.push(value.to_bits());
            }
        }
    }
    let (width, height) = (64, 64);
    let host = runtime.host_mut();
    let vertex_rows = host
        .adopt(
            orla_abi::FieldData::from_storage(vertex_schema(), Domain::Rows(8), vertices.clone())
                .unwrap(),
        )
        .unwrap();
    let triangle_rows = host
        .adopt(
            orla_abi::FieldData::from_storage(
                triangle_schema(),
                Domain::Rows(4),
                triangles.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let mask_grid = host
        .adopt(
            orla_abi::FieldData::from_storage(
                mask_schema(),
                Domain::Grid(mask_w, mask_h),
                mask.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let raster = runtime
        .kernel(&"native.rasterize@2".parse().unwrap())
        .expect("the cutout rasterizer is published from the start");
    let invocation = orla_abi::Invocation {
        inputs: vec![vertex_rows, triangle_rows, mask_grid],
        params: vec![orla_abi::Value::Float(0.5)],
        domain: Domain::Grid(width, height),
        context: orla_abi::Context::default(),
    };
    let surface = runtime.execute(&raster, &invocation).unwrap()[0];
    runtime.materialize(surface).unwrap();
    let device = runtime.host().data(surface).unwrap().storage().to_vec();
    let reference = rasterize_masked(
        &vertices,
        &triangles,
        width,
        height,
        Some(Cutout {
            words: &mask,
            width: mask_w,
            height: mask_h,
            alpha: 0.5,
        }),
    );
    let id_at =
        |words: &[u32], x: u32, y: u32| words[(y * width + x) as usize * SURFACE_WORDS + 6] as i32;
    let depth_at = |words: &[u32], x: u32, y: u32| {
        f32::from_bits(words[(y * width + x) as usize * SURFACE_WORDS])
    };
    // The reference drew the near quad where covered, the far one through
    // its holes.
    let ids: std::collections::BTreeSet<i32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| id_at(&reference, x, y))
        .collect();
    assert!(
        ids.contains(&0) || ids.contains(&1),
        "the far quad shows through: {ids:?}"
    );
    assert!(
        ids.contains(&2) || ids.contains(&3),
        "and the near quad is drawn: {ids:?}"
    );
    // Away from the squares' edges, the device and the reference agree.
    let mut compared = 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let id = id_at(&reference, x, y);
            let interior =
                (0..3).all(|dy| (0..3).all(|dx| id_at(&reference, x + dx - 1, y + dy - 1) == id));
            if !interior {
                continue;
            }
            assert_eq!(id_at(&device, x, y), id, "triangle at ({x}, {y})");
            assert!(
                (depth_at(&device, x, y) - depth_at(&reference, x, y)).abs() < 0.002,
                "depth at ({x}, {y})"
            );
            compared += 1;
        }
    }
    assert!(compared > 1500, "only {compared} pixels compared");
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

#[test]
fn a_node_off_the_device_says_what_its_trip_carried() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut runtime =
        orla_kernel::Runtime::with_backends(1 << 20, Box::new(gpu), Some(Box::new(CpuBackend)));
    // Only what the device declines leaves it, however small: its trips are
    // what this test asks.
    runtime.set_small(0);
    let mut publish = |source: &str| {
        let module = compile(source).unwrap();
        let candidate = runtime.prepare_module(module, source).unwrap();
        runtime
            .publish(candidate, orla_kernel::Expect::Absent)
            .unwrap()
    };
    let map = publish(
        "kernel t.map@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 2; } }",
    );
    let scan = publish(
        "kernel t.scan@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { return acc + sample(rows); } }",
    );
    let back = publish(
        "kernel t.back@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return sample(rows) + 1; } }",
    );
    let run = |runtime: &mut orla_kernel::Runtime, version, inputs| {
        let outputs = runtime
            .execute(
                version,
                &Invocation {
                    inputs,
                    params: vec![],
                    domain: Domain::Rows(5),
                    context: Context::default(),
                },
            )
            .unwrap();
        (outputs[0], runtime.take_moved())
    };
    let (rows, moved) = run(&mut runtime, &map, vec![]);
    assert_eq!(moved, orla_kernel::Moved::default());
    // The scan reads in Host memory: the map's rows come home.
    assert!(runtime.placement(&scan).routed.is_some());
    let (sums, moved) = run(&mut runtime, &scan, vec![rows]);
    assert_eq!(moved.downloaded, 20);
    // The map after it uploads the scan's rows again.
    let (_, moved) = run(&mut runtime, &back, vec![sums]);
    assert_eq!(moved.uploaded, 20);
    assert_eq!(moved.downloaded, 0);
}

#[test]
fn a_failure_on_the_device_is_the_reference_failure() {
    let Some(gpu) = gpu() else {
        return;
    };
    let run = |backend: &dyn Backend, source: &str, domain: Domain, inputs: &[Vec<u32>]| {
        let module = compile(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
        let mut host = Host::new(1 << 20);
        let handles = inputs
            .iter()
            .map(|words| {
                host.adopt(
                    orla_abi::FieldData::from_values(
                        orla_abi::Schema::new(vec![orla_abi::Scalar::Int]),
                        Domain::Rows(words.len() as u32),
                        words,
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let invocation = Invocation {
            inputs: handles,
            params: vec![],
            domain,
            context: Context::default(),
        };
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, Capability::ALL.into_iter().collect());
        backend
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .map(|_| ())
            .map_err(|error| error.to_string())
    };
    let map = |body: &str, ty: &str| {
        format!(
            "kernel t.f@1 {{ requires {{ field.write; }} output result: {ty}; compute {{ {body} }} }}"
        )
    };
    let cases = [
        (map("return 2147483600 + element() * 10;", "Rows<Int>"), Domain::Rows(16)),
        (map("let c = icoord(); return 100 / (c.x - 3) + c.y;", "Grid<Int>"), Domain::Grid(8, 4)),
        (map("let c = icoord(); return 7 % (c.y - 2) + c.x;", "Grid<Int>"), Domain::Grid(8, 4)),
        (map("return (element() + 1) * 1000000000;", "Rows<Int>"), Domain::Rows(8)),
        (map("return -(0x80000000 + element() * 0);", "Rows<Int>"), Domain::Rows(4)),
        (map("return udiv(10, element() - 5);", "Rows<Int>"), Domain::Rows(8)),
        (map("return ivec2(element(), 1) * 1073741824;", "Rows<IVec2>"), Domain::Rows(8)),
        (map("return 3000000000000000000l * long(element());", "Rows<Long>"), Domain::Rows(8)),
        (map("return 9223372036854775807l / (long(element()) - 6) + 1;", "Rows<Long>"), Domain::Rows(8)),
        // A guarded division fails nowhere.
        (map("let i = element(); return if i != 0 { 100 / i } else { 0 };", "Rows<Int>"), Domain::Rows(8)),
        (
            "kernel t.r@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>;
                compute { return sample(rows); } merge(a, b) { return a + b; } }"
                .to_owned(),
            Domain::Rows(1),
        ),
    ];
    let rows: Vec<u32> = (0..64).map(|_| 1 << 28).collect();
    for (source, domain) in &cases {
        let inputs = if source.contains("merge") {
            vec![rows.clone()]
        } else {
            Vec::new()
        };
        let reference = run(&CpuBackend, source, *domain, &inputs);
        let device = run(&gpu, source, *domain, &inputs);
        assert_eq!(device, reference, "{source}");
    }
    // Every case but the guarded one fails on the reference.
    let failing = cases
        .iter()
        .filter(|(source, domain)| {
            let inputs = if source.contains("merge") {
                vec![rows.clone()]
            } else {
                Vec::new()
            };
            run(&CpuBackend, source, *domain, &inputs).is_err()
        })
        .count();
    assert_eq!(failing, cases.len() - 1);
}

/// Invented names never spell a WGSL type (`f16`, `f32`, `f64`).
#[test]
fn a_kernel_of_many_helpers_and_fields_runs_on_the_device() {
    let Some(gpu) = gpu() else { return };
    const MANY: usize = 70;
    let helpers: String = (0..MANY)
        .map(|k| match k {
            0 => "fn h0(x: Float) -> Float { return loop 2 with a = x { a + 0.5 }; }\n".to_owned(),
            k => format!(
                "fn h{k}(x: Float) -> Float {{ return loop 2 with a = h{}(x) {{ a * 0.5 + {k}.0 }}; }}\n",
                k - 1
            ),
        })
        .collect();
    let fields: String = (0..MANY).map(|k| format!("m{k}: Float; ")).collect();
    let values: String = (0..MANY).map(|k| format!("m{k}: c.x + {k}.0, ")).collect();
    let source = format!(
        "kernel test.many@1 {{ requires {{ field.write; }} output result: Image;
        struct Wide {{ {fields}}}
        {helpers}
        compute {{
            let c = coord();
            let w = Wide {{ {values}}};
            return vec4(h{last}(c.x), w.m16 + w.m32, w.m64, w.m{last});
        }} }}",
        last = MANY - 1
    );
    let (cpu, gpu_pixels) = both(&gpu, &source, &[], (1, 1), vec![], Domain::Grid(4, 2));
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-4), 0.0);
}

/// Runs `module` over `inputs` on both backends and returns each output's
/// words.
fn on_both_inputs(
    gpu: &GpuBackend,
    host: &mut Host,
    module: &orla_compiler::Validated,
    inputs: Vec<orla_abi::Handle>,
    domain: Domain,
) -> (Vec<u32>, Vec<u32>) {
    let invocation = orla_abi::Invocation {
        inputs,
        params: Vec::new(),
        domain,
        context: orla_abi::Context::default(),
    };
    let mut scope = Scope::new(host, all());
    let cpu_out = orla_cpu::CpuBackend
        .execute(module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
    let prepared = Backend::prepare(gpu, module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(host, all());
    let gpu_out = Backend::execute(gpu, prepared.as_ref(), module, &invocation, &mut scope)
        .unwrap_or_else(|error| panic!("{error}"))[0];
    let byte_len = host
        .schema(gpu_out)
        .unwrap()
        .storage_words(host.domain(gpu_out).unwrap().count())
        * 4;
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let bytes = Backend::download(gpu, device, byte_len).unwrap();
    let gpu_words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (cpu_words, gpu_words)
}

/// A map run several times over its own output on the device, in one
/// command buffer, is the interpreter's runs one after another: whole
/// words and words its elements share, an odd and an even count.
#[test]
fn repeated_runs_are_the_interpreters_runs_in_turn() {
    let Some(gpu) = gpu() else { return };
    for kept in ["Float", "Float as f16"] {
        let module = compile(&format!(
            "kernel test.step@1 {{ requires {{ field.read; field.write; }}
             input prev: Rows<{kept}>; output result: Rows<{kept}>;
             compute {{ return sample(prev) * 0.5 + at_linear(prev, 3.7) + 1.0; }} }}"
        ))
        .unwrap_or_else(|error| panic!("{error}"));
        let schema = module.manifest.inputs[0].ty.schema.clone();
        let count = 1000u32;
        let start: Vec<f32> = (0..count).map(|i| (i % 17) as f32 * 0.25).collect();
        for repeat in 1..=4u32 {
            let mut host = Host::new(1 << 24);
            let mut seed = || {
                let data = orla_abi::FieldData::new(schema.clone(), Domain::Rows(count)).unwrap();
                let handle = host.adopt(data).unwrap();
                host.write_values(
                    handle,
                    &start.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                )
                .unwrap();
                handle
            };
            let first = seed();
            let second = seed();
            let invocation = |input| orla_abi::Invocation {
                inputs: vec![input],
                params: vec![],
                domain: Domain::Rows(count),
                context: orla_abi::Context::default(),
            };
            let mut last = first;
            for _ in 0..repeat {
                let mut scope = Scope::new(&mut host, all());
                last = CpuBackend
                    .execute(&module, &invocation(last), &mut scope)
                    .unwrap()[0];
            }
            let cpu = host.data(last).unwrap().values().into_owned();
            let prepared =
                Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
            let mut scope = Scope::new(&mut host, all());
            let answered = Backend::execute_repeated(
                &gpu,
                prepared.as_ref(),
                &module,
                &invocation(second),
                &mut scope,
                repeat,
                (0, 0),
            );
            let out = match answered {
                Some(outputs) => outputs.unwrap_or_else(|error| panic!("{error}"))[0],
                None => {
                    assert_eq!(repeat, 1, "{kept}: the device declined {repeat} times");
                    continue;
                }
            };
            let byte_len = schema.storage_words(count as usize) * 4;
            let device = host.resident(out).unwrap().expect("resident");
            let bytes = Backend::download(&gpu, device, byte_len).unwrap();
            let words = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            host.materialize(out, words).unwrap();
            let gpu_values = host.data(out).unwrap().values().into_owned();
            let far = cpu
                .iter()
                .zip(&gpu_values)
                .filter(|(a, b)| (f32::from_bits(**a) - f32::from_bits(**b)).abs() > 1e-3)
                .count();
            assert_eq!(far, 0, "{kept}, {repeat} times");
        }
    }
}

/// An extent past 2^24, which a float cannot hold exactly, is every
/// element and the whole count, size and domain on the device.
#[test]
fn extents_past_a_floats_integers_match() {
    let Some(gpu) = gpu() else { return };
    let count = (1u32 << 24) + 1;
    let mut host = Host::new(1 << 27);
    let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let flags = host
        .adopt(
            orla_abi::FieldData::from_storage(schema, Domain::Rows(count), vec![0; count as usize])
                .unwrap(),
        )
        .unwrap();
    let module = compile(
        "kernel test.far@1 { requires { field.read; field.write; } input flags: Rows<Int>;
         output result: Rows<Int>;
         compute { return count(flags) + size(flags) - domain() - element() + sample(flags); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let (cpu, gpu_words) =
        on_both_inputs(&gpu, &mut host, &module, vec![flags], Domain::Rows(count));
    assert_eq!(cpu[count as usize - 1], 1);
    assert!(cpu == gpu_words, "the device differs from the interpreter");
}

/// Past 65,535 workgroups of 64 in one dimension, a scatter, a prefix and
/// a reduction fold their dispatch into rows.
#[test]
fn passes_over_more_rows_than_one_dimension_holds_match() {
    let Some(gpu) = gpu() else { return };
    let count = 5_000_000u32;
    let mut host = Host::new(1 << 26);
    let ints = |host: &mut Host, words: Vec<u32>| {
        let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
        let n = words.len() as u32;
        host.adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(n), words).unwrap())
            .unwrap()
    };
    let floats = |host: &mut Host, words: Vec<u32>| {
        let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Float]);
        let n = words.len() as u32;
        host.adopt(orla_abi::FieldData::from_storage(schema, Domain::Rows(n), words).unwrap())
            .unwrap()
    };
    // Each run takes fresh inputs: one the device ran on is resident there.
    let source = |host: &mut Host| ints(host, (0..count).map(|i| i % 7).collect());

    let prefix = compile(
        "kernel test.rank@1 { requires { field.read; field.write; } input flags: Rows<Int>;
         output result: Rows<Int>; prefix with sum { return sample(flags) % 3; } }",
    )
    .unwrap();
    let inputs = vec![source(&mut host)];
    let (cpu, gpu_words) = on_both_inputs(&gpu, &mut host, &prefix, inputs, Domain::Rows(count));
    assert_eq!(cpu, gpu_words, "prefix");

    let reduce = compile(
        "kernel test.most@1 { requires { field.read; field.write; } input flags: Rows<Int>;
         output result: Rows<Int>; compute { return sample(flags) * 3 + element() % 5; }
         merge(a, b) { return max(a, b); } }",
    )
    .unwrap();
    let inputs = vec![source(&mut host)];
    let (cpu, gpu_words) = on_both_inputs(&gpu, &mut host, &reduce, inputs, Domain::Rows(1));
    assert_eq!(cpu, gpu_words, "reduce");

    for rule in ["max", "last"] {
        let scatter = compile(&format!(
            "kernel test.spread@1 {{ requires {{ field.read; field.write; }}
             input flags: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
             struct Put {{ slot: Int; value: Int; }}
             scatter into slots with {rule} {{
                 return Put {{ slot: element() % 1000, value: element() + sample(flags) }};
             }} }}"
        ))
        .unwrap();
        let inputs = vec![source(&mut host), ints(&mut host, vec![0; 1000])];
        let (cpu, gpu_words) =
            on_both_inputs(&gpu, &mut host, &scatter, inputs, Domain::Rows(count));
        assert_eq!(cpu, gpu_words, "scatter with {rule}");
    }

    // A float sum, sorted on the device by target and source.
    let summed = compile(
        "kernel test.pile@1 { requires { field.read; field.write; }
         input flags: Rows<Int>; input sums: Rows<Float>; output result: Rows<Float>;
         struct Add { slot: Int; value: Float; }
         scatter into sums with sum {
             return Add { slot: element() % 1000, value: float(sample(flags)) * 0.5 };
         } }",
    )
    .unwrap();
    let inputs = vec![
        source(&mut host),
        floats(&mut host, vec![0.0f32.to_bits(); 1000]),
    ];
    let (cpu, gpu_words) = on_both_inputs(&gpu, &mut host, &summed, inputs, Domain::Rows(count));
    assert_eq!(cpu, gpu_words, "scatter with a float sum");
}

/// By 64-bit atomics on signed order keys: negatives, values past 32 bits,
/// and a pair of longs as one element.
#[test]
fn long_extremes_land_as_the_references_do() {
    let Some(gpu) = gpu() else { return };
    let count = 50_000u32;
    let mut host = Host::new(1 << 24);
    for rule in ["min", "max"] {
        let module = compile(&format!(
            "kernel test.deep@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input slots: Rows<Deep>; output result: Rows<Deep>;
             struct Deep {{ near: Long; id: Long; }}
             struct Pair {{ at: Int; value: Deep; }}
             scatter into slots with {rule} {{
                 let k = sample(keys);
                 let wide = long(k % 4001 - 2000) * 3000000001l;
                 let _fan = loop k % 4 with j = 0 {{
                     emit(Pair {{ at: (k * 13 + j) % 61, value: Deep {{ near: wide + long(j), id: -long(k) }} }});
                     j + 1
                 }};
                 return Pair {{ at: k % 67 - 3, value: Deep {{ near: -wide, id: long(k) << 33 }} }};
             }} }}"
        ))
        .unwrap();
        let keys = host
            .adopt(
                orla_abi::FieldData::from_storage(
                    orla_abi::Schema::new(vec![orla_abi::Scalar::Int]),
                    Domain::Rows(count),
                    (0..count)
                        .map(|i| i.wrapping_mul(2_654_435_761) >> 5)
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap();
        let slots = host
            .adopt(
                orla_abi::FieldData::from_storage(
                    orla_abi::Schema::new(vec![orla_abi::Scalar::Long; 4]),
                    Domain::Rows(61),
                    (0..61u64)
                        .flat_map(|i| {
                            let v = (i as i64 - 30) * 1_000_000_007;
                            [v as u32, (v >> 32) as u32, i as u32, 0]
                        })
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap();
        let (cpu, gpu_words) = on_both_inputs(
            &gpu,
            &mut host,
            &module,
            vec![keys, slots],
            Domain::Rows(count),
        );
        assert_eq!(cpu, gpu_words, "longs under {rule}");
    }
}

/// Ints under every order-free rule; floats, as ordered keys, under `min`
/// and `max` on a grid.
#[test]
fn emitted_pairs_land_as_the_references_do() {
    let Some(gpu) = gpu() else { return };
    let count = 50_000u32;
    let mut host = Host::new(1 << 24);
    let adopt = |host: &mut Host, scalar, domain: Domain, words: Vec<u32>| {
        let schema = orla_abi::Schema::new(vec![scalar]);
        host.adopt(orla_abi::FieldData::from_storage(schema, domain, words).unwrap())
            .unwrap()
    };
    for rule in ["sum", "max", "min"] {
        let module = compile(&format!(
            "kernel test.fan@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
             struct Pair {{ at: Int; value: Int; }}
             scatter into slots with {rule} {{
                 let k = sample(keys);
                 let _fan = loop k % 5 with j = 0 {{
                     emit(Pair {{ at: (k * 31 + j * 7) % 97, value: k % 1000 - j * 3 }});
                     j + 1
                 }};
                 return Pair {{ at: k % 101 - 3, value: k % 17 }};
             }} }}"
        ))
        .unwrap();
        let inputs = vec![
            adopt(
                &mut host,
                orla_abi::Scalar::Int,
                Domain::Rows(count),
                (0..count).collect(),
            ),
            adopt(
                &mut host,
                orla_abi::Scalar::Int,
                Domain::Rows(97),
                vec![5; 97],
            ),
        ];
        let (cpu, gpu_words) =
            on_both_inputs(&gpu, &mut host, &module, inputs, Domain::Rows(count));
        assert_eq!(cpu, gpu_words, "emitted ints under {rule}");
    }
    for rule in ["min", "max"] {
        let module = compile(&format!(
            "kernel test.spray@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input canvas: Grid<Float>; output result: Grid<Float>;
             struct Dot {{ at: Vec2; value: Float; }}
             scatter into canvas with {rule} {{
                 let k = sample(keys);
                 let v = float(k % 2001 - 1000) * 0.25;
                 emit(Dot {{ at: vec2(float(k % 23), float(k % 11)), value: v }});
                 emit(Dot {{ at: vec2(float(k % 19) - 1.0, float(k % 13)), value: -v }});
                 return Dot {{ at: vec2(float(k % 7), float(k % 5)), value: v * 2.0 }};
             }} }}"
        ))
        .unwrap();
        let inputs = vec![
            adopt(
                &mut host,
                orla_abi::Scalar::Int,
                Domain::Rows(count),
                (0..count)
                    .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
                    .collect(),
            ),
            adopt(
                &mut host,
                orla_abi::Scalar::Float,
                Domain::Grid(20, 12),
                vec![0.5f32.to_bits(); 240],
            ),
        ];
        let (cpu, gpu_words) =
            on_both_inputs(&gpu, &mut host, &module, inputs, Domain::Rows(count));
        assert_eq!(cpu, gpu_words, "emitted floats under {rule}");
    }
}

/// A record of a float and an int, its base every sign and both zeros,
/// lands every word as the reference keeps it, unreached cells included.
#[test]
fn a_keyed_target_starts_from_every_word_of_its_base() {
    let Some(gpu) = gpu() else { return };
    let count = 20_000u32;
    let mut host = Host::new(1 << 24);
    let schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Float, orla_abi::Scalar::Int]);
    for rule in ["min", "max"] {
        let module = compile(&format!(
            "kernel test.mixed@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input cells: Rows<Cell>; output result: Rows<Cell>;
             struct Cell {{ f: Float; i: Int; }}
             struct Pair {{ at: Int; value: Cell; }}
             scatter into cells with {rule} {{
                 let k = sample(keys);
                 let v = float(k % 2001 - 1000) * 0.125;
                 emit(Pair {{ at: k % 89, value: Cell {{ f: v, i: k % 77 - 38 }} }});
                 return Pair {{ at: k % 131, value: Cell {{ f: -v, i: 38 - k % 77 }} }};
             }} }}"
        ))
        .unwrap();
        let keys = host
            .adopt(
                orla_abi::FieldData::from_storage(
                    orla_abi::Schema::new(vec![orla_abi::Scalar::Int]),
                    Domain::Rows(count),
                    (0..count)
                        .map(|i| i.wrapping_mul(2_654_435_761) >> 9)
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap();
        // More cells than any pair reaches, so some keep their base.
        let cells = host
            .adopt(
                orla_abi::FieldData::from_storage(
                    schema.clone(),
                    Domain::Rows(160),
                    (0..160i64)
                        .flat_map(|i| {
                            let f = match i % 4 {
                                0 => -0.0f32,
                                1 => 0.0,
                                2 => -(i as f32) * 3.5,
                                _ => i as f32 * 2.25,
                            };
                            [f.to_bits(), (i - 80) as u32]
                        })
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap();
        let (cpu, gpu_words) = on_both_inputs(
            &gpu,
            &mut host,
            &module,
            vec![keys, cells],
            Domain::Rows(count),
        );
        assert_eq!(cpu, gpu_words, "a mixed record under {rule}");
    }
}

/// Twenty-five, forty and eighty inputs (plain, narrow, records) past the
/// device's bindings are packed into one buffer, with a division so the
/// failure record binds, and a fold of forty: the interpreter's words. A
/// scan's memories are never packed: past the limit it declines.
#[test]
fn inputs_past_the_devices_bindings_are_packed_and_read_where_they_are() {
    let Some(gpu) = gpu() else { return };
    // Terms summed as a balanced tree, within a debug test thread's stack.
    fn tree(terms: &[String]) -> String {
        match terms {
            [one] => one.clone(),
            _ => {
                let (left, right) = terms.split_at(terms.len() / 2);
                format!("({} + {})", tree(left), tree(right))
            }
        }
    }
    // Every fifteenth input is plain.
    let plain = |k: usize| k.is_multiple_of(15);
    let narrow = |k: usize| !plain(k) && k % 3 == 1;
    let record = |k: usize| !narrow(k) && k % 5 == 2;
    let source = |inputs: usize, reduce: bool| {
        let ports: String = (0..inputs)
            .map(|k| {
                if narrow(k) {
                    format!("input f{k}: Rows<Int as u8>; ")
                } else if record(k) {
                    format!("input f{k}: Rows<Pair>; ")
                } else {
                    format!("input f{k}: Rows<Int>; ")
                }
            })
            .collect();
        let terms: Vec<String> = (0..inputs)
            .map(|k| {
                if record(k) {
                    format!("at(f{k}, element()).b")
                } else {
                    format!("sample(f{k})")
                }
            })
            .collect();
        let body = format!("return {} / (element() % 7 + 1);", tree(&terms));
        let mode = if reduce {
            format!("compute {{ {body} }} merge(a, b) {{ return a + b; }}")
        } else {
            format!("compute {{ {body} }}")
        };
        format!(
            "kernel test.packed@1 {{ requires {{ field.read; field.write; }}
             struct Pair {{ a: Int; b: Int; }} {ports} output result: Rows<Int>; {mode} }}"
        )
    };
    let rows = 64u32;
    for (inputs, reduce) in [(25usize, false), (40, false), (80, false), (40, true)] {
        let module = compile(&source(inputs, reduce)).unwrap_or_else(|error| panic!("{error}"));
        assert!(
            gpu.declines(&module).is_none(),
            "{inputs} inputs: {:?}",
            gpu.declines(&module)
        );
        let mut host = Host::new(1 << 22);
        let handles: Vec<_> = (0..inputs)
            .map(|k| {
                let (schema, words): (orla_abi::Schema, Vec<u32>) = if narrow(k) {
                    (
                        orla_abi::Schema::encoded(
                            vec![orla_abi::Scalar::Int],
                            vec![orla_abi::Encoding::U8],
                        )
                        .unwrap(),
                        (0..rows).map(|i| (i * 7 + k as u32) % 256).collect(),
                    )
                } else if record(k) {
                    (
                        orla_abi::Schema::new(vec![orla_abi::Scalar::Int; 2]),
                        (0..rows)
                            .flat_map(|i| [i + 1000, i * 5 + k as u32])
                            .collect(),
                    )
                } else {
                    (
                        orla_abi::Schema::new(vec![orla_abi::Scalar::Int]),
                        (0..rows).map(|i| i * 3 + k as u32 * 11).collect(),
                    )
                };
                host.adopt(
                    orla_abi::FieldData::from_values(schema, Domain::Rows(rows), &words).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let domain = if reduce {
            Domain::Rows(1)
        } else {
            Domain::Rows(rows)
        };
        let (cpu, device) = on_both_inputs(&gpu, &mut host, &module, handles, domain);
        assert_eq!(cpu, device, "{inputs} inputs, reduce {reduce}");
    }

    // A scan's memories are never packed: past the device's storage
    // buffers, the scan declines and says why.
    let memories = 40;
    let ports: String = (0..memories)
        .map(|k| format!("memory m{k}: Rows<Int>; "))
        .collect();
    let module = compile(&format!(
        "kernel test.remember@1 {{ requires {{ field.read; field.write; }}
         input ticks: Rows<Int>; {ports} output result: Rows<Int>;
         scan with acc = 0 {{ put(m0, 0, acc); put(m{last}, 0, acc); return acc + 1; }} }}",
        last = memories - 1
    ))
    .unwrap_or_else(|error| panic!("{error}"));
    let why = gpu
        .declines(&module)
        .expect("forty memories are more than the device binds");
    assert!(why.contains("binds") && why.contains("at most"), "{why}");
    let error = Backend::prepare(&gpu, &module)
        .err()
        .expect("declined, so not prepared");
    assert!(error.contains(&why), "{error}");
}

/// The backend predicts a module's brace depth before emitting it (never
/// less than the shader's, at most one more); past WGSL's limit it
/// declines, and a session runs the module on the ordered backend.
#[test]
fn branches_nested_past_what_a_shader_holds_run_off_the_device() {
    let Some(gpu) = gpu() else { return };
    fn source(shape: &str, n: usize) -> String {
        let body = match shape {
            "nested" => format!(
                "let x = element(); return {}x{};",
                (0..n)
                    .map(|k| format!("if x > {k} {{ "))
                    .collect::<String>(),
                " } else { 0 }".repeat(n)
            ),
            "ladder" => format!(
                "let x = element(); return {} else {{ -1 }};",
                (0..n)
                    .map(|k| format!("if x < {k} {{ {k} }}"))
                    .collect::<Vec<_>>()
                    .join(" else ")
            ),
            "loops" => format!(
                "let x = element(); return {}x{};",
                (0..n)
                    .map(|k| format!("loop 2 with a{k} = 0 {{ a{k} + "))
                    .collect::<String>(),
                " }".repeat(n)
            ),
            _ => format!(
                "let x = element(); return {}x{};",
                "match x { 0 => 1, _ => ".repeat(n),
                " }".repeat(n)
            ),
        };
        format!(
            "kernel t.deep@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ {body} }} }}"
        )
    }
    let braces = |shader: &str| {
        let (mut depth, mut deepest) = (0usize, 0usize);
        for c in shader.chars() {
            if c == '{' {
                depth += 1;
                deepest = deepest.max(depth);
            } else if c == '}' {
                depth = depth.saturating_sub(1);
            }
        }
        deepest
    };
    for shape in ["nested", "ladder", "loops", "match"] {
        for n in [1usize, 10, 40, 60] {
            let module = compile(&source(shape, n)).unwrap_or_else(|error| panic!("{error}"));
            let estimate = orla_gpu::wgsl::brace_depth(&module);
            let emitted = braces(&orla_gpu::wgsl::emit(&module));
            assert!(
                (emitted..=emitted + 1).contains(&estimate),
                "{shape} {n}: estimated {estimate}, emitted {emitted}"
            );
        }
    }
    for shape in ["nested", "ladder"] {
        let source = source(shape, 150);
        let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
        let why = gpu.declines(&module).expect("past the brace limit");
        assert!(why.contains("nest"), "{why}");
        let mut runtime = orla_kernel::Runtime::with_backends(
            1 << 20,
            Box::new(GpuBackend::new().unwrap()),
            Some(Box::new(CpuBackend)),
        );
        let candidate = runtime.prepare_module(module.clone(), &source).unwrap();
        let version = runtime
            .publish(candidate, orla_kernel::Expect::Absent)
            .unwrap();
        assert!(runtime.placement(&version).routed.is_some(), "{shape}");
        let outputs = runtime
            .execute(
                &version,
                &Invocation {
                    inputs: vec![],
                    params: vec![],
                    domain: Domain::Rows(200),
                    context: Context::default(),
                },
            )
            .unwrap();
        let routed = runtime.host().data(outputs[0]).unwrap().storage().to_vec();
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(200),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let reference = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        assert_eq!(routed, host.data(reference).unwrap().storage(), "{shape}");
    }
}

/// `smoothstep` follows the interpreter's rule on the device: at
/// `low == high` a step (0 below, 1 at and above), at `low > high` a
/// falling ramp, scalar and vector alike.
#[test]
fn smoothstep_is_the_interpreters_at_its_edges() {
    let Some(gpu) = gpu() else { return };
    let ramp: Vec<[f32; 4]> = (0..32 * 4)
        .map(|i| [(i % 32) as f32 / 31.0 * 2.0 - 0.5, 0.5, 0.25, 1.0])
        .collect();
    let source = "kernel test.edges@1 { requires { field.read; field.write; } input source: Image; output result: Image;
        compute {
            let x = sample(source).r;
            let flat = smoothstep(0.5, 0.5, x);
            let falling = smoothstep(1.0, 0.0, x);
            let v = smoothstep(vec3(0.0, 0.5, 1.0), vec3(1.0, 0.5, 0.0), vec3(x, x, x));
            return vec4(flat, falling, v.y, v.z);
        } }";
    let (cpu, gpu_pixels) = both(&gpu, source, &[ramp], (32, 4), vec![], Domain::Grid(32, 4));
    assert!(cpu.iter().any(|p| p[0] == 0.0) && cpu.iter().any(|p| p[0] == 1.0));
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-5), 0.0);
}

/// A Field made from another is brought up to the device on the pages it
/// changed alone: from a copy of the other's buffer while that lives,
/// from the buffer its upload left behind once it is released. Its words
/// on the device are its own either way.
#[test]
fn a_field_made_from_another_is_uploaded_on_its_changed_pages() {
    let Some(gpu) = gpu() else { return };
    let module = compile(
        "kernel test.twice@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return sample(rows) * 2; } }",
    )
    .unwrap();
    let prepared = gpu.prepare(&module).unwrap();
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let page = orla_abi::PAGE_WORDS;
    let len = page * 5 + 3;
    let domain = Domain::Rows(len as u32);
    let mut host = Host::new(len * 64);
    // Runs the Kernel over `rows`: its words doubled, and the bytes uploaded.
    let run = |host: &mut Host, rows: orla_abi::Handle| -> (Vec<u32>, u64) {
        let before = gpu.uploaded();
        let invocation = Invocation {
            inputs: vec![rows],
            params: vec![],
            domain,
            context: Context::default(),
        };
        let out = {
            let mut scope = Scope::new(host, all());
            gpu.execute(prepared.as_ref(), &module, &invocation, &mut scope)
                .unwrap()
        };
        let bytes =
            Backend::download(&gpu, host.resident(out[0]).unwrap().unwrap(), len * 4).unwrap();
        host.release(out[0]).unwrap();
        let words = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) / 2)
            .collect();
        (words, gpu.uploaded() - before)
    };
    let field = |words: &Vec<u32>| {
        orla_abi::FieldData::from_storage(ints.clone(), domain, words.clone()).unwrap()
    };
    let first: Vec<u32> = (0..len as u32).collect();
    let base = host.adopt(field(&first)).unwrap();
    assert_eq!(run(&mut host, base), (first.clone(), len as u64 * 4));

    // Made from the base, which lives on the device: a page and the last.
    let mut second = first.clone();
    second[page + 1] = 7;
    second[len - 1] = 9;
    let mut changed = orla_abi::Pages::new();
    changed.mark(page + 1);
    changed.mark(len - 1);
    let made = host.adopt_derived(field(&second), base, changed).unwrap();
    assert_eq!(
        run(&mut host, made),
        (second.clone(), (page + 3) as u64 * 4)
    );

    // Made from that one, released before it is bound: the first page.
    let mut third = second.clone();
    third[3] = 11;
    let mut changed = orla_abi::Pages::new();
    changed.mark(3);
    let again = host.adopt_derived(field(&third), made, changed).unwrap();
    host.release(base).unwrap();
    host.release(made).unwrap();
    assert_eq!(run(&mut host, again), (third.clone(), page as u64 * 4));
    host.release(again).unwrap();

    // Two made from words never on the device, the first released before
    // the second is bound: the second is brought up from the first's
    // buffer on the pages either changed.
    let plain = host.adopt(field(&first)).unwrap();
    let sibling = |host: &mut Host, at: usize| {
        let mut words = first.clone();
        words[at] = 13;
        let mut changed = orla_abi::Pages::new();
        changed.mark(at);
        (
            host.adopt_derived(field(&words), plain, changed).unwrap(),
            words,
        )
    };
    let (one, words) = sibling(&mut host, page + 2);
    assert_eq!(run(&mut host, one), (words, len as u64 * 4));
    host.release(one).unwrap();
    let (other, words) = sibling(&mut host, page * 4 + 1);
    assert_eq!(run(&mut host, other), (words, page as u64 * 2 * 4));
    host.release(other).unwrap();

    // Written in place after it was uploaded, a Field's old buffer holds
    // other words: what is made from it after is uploaded whole.
    let written = host.adopt(field(&first)).unwrap();
    run(&mut host, written);
    host.write_element(written, 9, &[17]).unwrap();
    let mut words = first.clone();
    words[9] = 17;
    let after = host
        .adopt_derived(field(&words), written, orla_abi::Pages::new())
        .unwrap();
    assert_eq!(run(&mut host, after), (words, len as u64 * 4));
}
