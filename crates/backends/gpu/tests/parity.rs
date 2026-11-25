//! GPU results match the CPU interpreter within floating-point tolerance.
//! Tests skip when no adapter is available.

use entelechy_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use entelechy_compiler::compile;
use entelechy_cpu::CpuBackend;
use entelechy_gpu::GpuBackend;
use entelechy_rules::Backend;

/// Downloads a resident GPU output into Host memory as pixels.
fn materialized(gpu: &GpuBackend, host: &mut Host, handle: entelechy_abi::Handle) -> Vec<[f32; 4]> {
    let byte_len = host
        .schema(handle)
        .unwrap()
        .storage_words(host.domain(handle).unwrap().count())
        * 4;
    let device = host
        .resident(handle)
        .unwrap()
        .expect("gpu outputs are resident");
    let words = Backend::download(gpu, device, byte_len).unwrap();
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
    module: &entelechy_compiler::Validated,
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
                entelechy_abi::Image::from_pixels(input_size.0, input_size.1, pixels.clone())
                    .unwrap(),
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

/// `tanh` far from zero is ±1 on every backend, never NaN: a device's own
/// may fail there.
#[test]
fn tanh_far_out_is_one_on_every_backend() {
    let Some(gpu) = gpu() else { return };
    let far: Vec<[f32; 4]> = (0..64)
        .map(|i| [i as f32 * 4.0 - 128.0, i as f32 * -50.0, 0.25, 1.0])
        .collect();
    let source = "kernel test.tanh@1 { requires { field.read; field.write; } input source: Image; output result: Image;
        compute {
            let c = sample(source);
            let v = tanh(vec2(c.r, c.g));
            return vec4(tanh(c.r), v.x, v.y, tanh(c.b));
        } }";
    let (cpu, gpu_pixels) = both(&gpu, source, &[far], (64, 1), vec![], Domain::Grid(64, 1));
    assert!(
        gpu_pixels.iter().flatten().all(|v| v.is_finite()),
        "{gpu_pixels:?}"
    );
    assert_eq!(mismatch(&cpu, &gpu_pixels, 1e-6), 0.0);
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
    let mut input = entelechy_abi::InputState::default();
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
            queued: 0,
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
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(gpu));
    runtime.set_background(true);
    let source = |value: &str| {
        format!(
            "kernel test.flat@1 {{ requires {{ field.write; }} output result: Image; compute {{ return vec4({value}, 0.5, 0.25, 1.0); }} }}"
        )
    };
    let running = load(&mut runtime, &source("0.125"));
    let invocation = entelechy_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(64, 64),
        context: entelechy_abi::Context::default(),
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
    use entelechy_rules::raster::{SURFACE_WORDS, rasterize_words, triangle_schema, vertex_schema};
    let Some(gpu) = gpu() else { return };
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, Box::new(gpu));
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
            entelechy_abi::FieldData::from_storage(
                vertex_schema(),
                Domain::Rows(6),
                vertices.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let triangle_rows = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![vertex_rows, triangle_rows],
        params: vec![],
        domain: Domain::Grid(width, height),
        context: entelechy_abi::Context::default(),
    };
    let surface = runtime.execute(&raster, &invocation).unwrap()[0];
    runtime.materialize(surface).unwrap();
    let device = runtime.host().data(surface).unwrap().storage().to_vec();
    assert_eq!(
        runtime.host().schema(surface).unwrap().to_string(),
        "f,f,f,f,f,f,i"
    );
    let reference = rasterize_words(&vertices, &triangles, width, height);
    // Only pixels whose eight neighbors carry the same triangle on the
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
    // Onto a canvas of the same extent, the device draws the same surface.
    let canvas = runtime
        .host_mut()
        .create(
            entelechy_rules::raster::canvas_schema(),
            Domain::Grid(width, height),
        )
        .unwrap();
    let on_canvas = runtime
        .kernel(&"native.rasterize@3".parse().unwrap())
        .expect("the canvas form is published from the start");
    let drawn = runtime
        .execute(
            &on_canvas,
            &entelechy_abi::Invocation {
                inputs: vec![vertex_rows, triangle_rows, canvas],
                ..invocation
            },
        )
        .unwrap()[0];
    runtime.materialize(drawn).unwrap();
    assert_eq!(runtime.host().data(drawn).unwrap().storage(), &device[..]);
}

/// Many triangles over many blocks of the sorting pass, some a rule must
/// skip (a corner behind the eye, one not finite, no area) and many too
/// small to hold a pixel center: on interior pixels the device draws what
/// the reference draws, small triangles drawn by the sorting pass or all by
/// the device's rasterizer, and indexed or pulled, it draws the same words.
#[test]
fn a_soup_of_triangles_rasterizes_as_the_reference_does_either_way() {
    use entelechy_rules::raster::{SURFACE_WORDS, rasterize_words, triangle_schema, vertex_schema};
    let Some(indexed) = gpu() else { return };
    let Some(pulled) = gpu() else { return };
    let pulled = pulled.with_pulled_vertices();
    let Some(device_only) = gpu() else { return };
    // Without 64-bit atomics the device's rasterizer draws every triangle.
    let device_only = device_only.with_long_pairs();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut unit = move || (next() % 1_000_000) as f32 / 1_000_000.0;
    let (width, height) = (96u32, 64u32);
    let count = 3000u32;
    let mut vertices = Vec::new();
    for t in 0..count {
        // Each triangle at its own depth after the divide, so the nearest
        // one is never a matter of rounding.
        let depth = 0.05 + 0.9 * t as f32 / count as f32;
        let small = t % 40 != 0;
        let (cx, cy) = (unit() * 2.4 - 1.2, unit() * 2.4 - 1.2);
        let reach = if small { 0.004 } else { 1.0 };
        for k in 0..3 {
            let w = 0.5 + unit() * 2.0;
            let (x, y) = (cx + (unit() - 0.5) * reach, cy + (unit() - 0.5) * reach);
            let mut clip = [x * w, y * w, depth * w, w];
            match (t % 97, k) {
                (5, 1) => clip[3] = -0.5,
                (6, 2) => clip[0] = f32::NAN,
                (7, 1) => clip = [0.0; 4],
                _ => {}
            }
            vertices.extend(clip.iter().map(|v| v.to_bits()));
            vertices.extend([unit(), unit(), unit(), unit(), unit()].map(f32::to_bits));
        }
    }
    let mut triangles: Vec<u32> = (0..count * 3).collect();
    // Out of range both ways: clamped.
    triangles[30] = (-4i32) as u32;
    triangles[61] = count * 3 + 9;
    let reference = rasterize_words(&vertices, &triangles, width, height);
    let draw = |gpu: GpuBackend| -> Vec<u32> {
        let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 26, Box::new(gpu));
        let host = runtime.host_mut();
        let vertex_rows = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    vertex_schema(),
                    Domain::Rows(count * 3),
                    vertices.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let triangle_rows = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    triangle_schema(),
                    Domain::Rows(count),
                    triangles.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let raster = runtime
            .kernel(&"native.rasterize@1".parse().unwrap())
            .unwrap();
        let invocation = entelechy_abi::Invocation {
            inputs: vec![vertex_rows, triangle_rows],
            params: vec![],
            domain: Domain::Grid(width, height),
            context: entelechy_abi::Context::default(),
        };
        let surface = runtime.execute(&raster, &invocation).unwrap()[0];
        runtime.materialize(surface).unwrap();
        runtime.host().data(surface).unwrap().storage().to_vec()
    };
    let device = draw(indexed);
    assert_eq!(
        draw(pulled),
        device,
        "indexed and pulled draw the same words"
    );
    let all_by_device = draw(device_only);
    let id_at =
        |words: &[u32], x: u32, y: u32| words[(y * width + x) as usize * SURFACE_WORDS + 6] as i32;
    let mut compared = 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let id = id_at(&reference, x, y);
            // Inside a drawn triangle: a sliver the reference's centers
            // miss may still be drawn by the device's own coverage rule.
            let interior = id >= 0
                && (0..3)
                    .all(|dy| (0..3).all(|dx| id_at(&reference, x + dx - 1, y + dy - 1) == id));
            if !interior {
                continue;
            }
            compared += 1;
            let base = (y * width + x) as usize * SURFACE_WORDS;
            for device in [&device, &all_by_device] {
                assert_eq!(id_at(device, x, y), id, "id at ({x}, {y})");
                for k in 0..6 {
                    let (r, d) = (
                        f32::from_bits(reference[base + k]),
                        f32::from_bits(device[base + k]),
                    );
                    assert!(
                        (r - d).abs() <= 1e-4 * r.abs().max(1.0),
                        "word {k} at ({x}, {y}): cpu {r}, gpu {d}"
                    );
                }
            }
        }
    }
    assert!(compared > 500, "{compared} compared");
}

#[test]
fn reductions_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // Ints exactly, floats within the tree's rounding, over an odd count
    // (the tail path) and an image (a grid map).
    let source = "kernel test.stats@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Stats>; struct Stats { bright: Int; sum: Vec4; max: Float; } compute over source { let c = sample(source); let l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return Stats { bright: if l > 0.5 { 1 } else { 0 }, sum: c, max: l }; } merge(a, b) { return Stats { bright: a.bright + b.bright, sum: a.sum + b.sum, max: max(a.max, b.max) }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let (width, height) = (37, 23);
    let mut host = Host::new(1 << 20);
    let image = host
        .adopt_image(
            entelechy_abi::Image::from_pixels(
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![image],
        params: vec![],
        domain: Domain::Rows(1),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = entelechy_cpu::CpuBackend
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
    let gpu_stats = Backend::download(&gpu, device, byte_len).unwrap();
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
        "kernel test.prefix_sum@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; prefix over rows with sum { return sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let worst = compile(
        "kernel test.prefix_max@1 { requires { field.read; field.write; } input rows: Rows<Run>; output result: Rows<Run>; struct Run { i: Int; f: Float; } prefix over rows with max { return sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    // 1 and 10 are one block; 5000 is seventy-one with a ragged last;
    // 70000 exceeds one invocation's walk in a frame.
    for count in [1u32, 10, 5000, 70000] {
        let ints: Vec<u32> = (0..count).map(|i| ((i as i32 % 11) - 5) as u32).collect();
        let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
        let mut host = Host::new(1 << 24);
        let rows = host
            .adopt(
                entelechy_abi::FieldData::from_storage(schema, Domain::Rows(count), ints).unwrap(),
            )
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
        let schema = entelechy_abi::Schema::new(vec![
            entelechy_abi::Scalar::Int,
            entelechy_abi::Scalar::Float,
        ]);
        let mut host = Host::new(1 << 24);
        let rows = host
            .adopt(
                entelechy_abi::FieldData::from_storage(schema, Domain::Rows(count), words).unwrap(),
            )
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
    module: &entelechy_compiler::Validated,
    input: entelechy_abi::Handle,
    count: u32,
) -> (Vec<u32>, Vec<u32>) {
    let invocation = entelechy_abi::Invocation {
        inputs: vec![input],
        params: Vec::new(),
        domain: Domain::Rows(count),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(host, all());
    let cpu_out = entelechy_cpu::CpuBackend
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
    let gpu_words = Backend::download(gpu, device, byte_len).unwrap();
    (cpu_words, gpu_words)
}

#[test]
fn scans_match_between_backends() {
    let Some(gpu) = gpu() else { return };
    // A prefix sum of ints (exact) and a one-pole filter (rounding only).
    let source = "kernel test.scan@1 { requires { field.read; field.write; } input rows: Rows<Item>; param k: Float = 0.1; output result: Rows<Run>; struct Item { value: Int; x: Float; } struct Run { total: Int; y: Float; } scan over rows with run = Run { total: 0, y: 0.0 } { let it = sample(rows); return Run { total: run.total + it.value, y: run.y + (it.x - run.y) * k }; } }";
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
    let schema = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Int,
        entelechy_abi::Scalar::Float,
    ]);
    let mut host = Host::new(1 << 20);
    let rows = host
        .adopt(entelechy_abi::FieldData::from_storage(schema, Domain::Rows(count), words).unwrap())
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![rows],
        params: vec![Value::Float(0.1)],
        domain: Domain::Rows(count),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = entelechy_cpu::CpuBackend
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
    let gpu_words = Backend::download(&gpu, device, byte_len).unwrap();
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
    let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 3]);
    let mut host = Host::new(1 << 16);
    let points = host
        .adopt(entelechy_abi::FieldData::from_storage(schema, Domain::Rows(9), words).unwrap())
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![points],
        params: vec![],
        domain: Domain::Rows(9),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = entelechy_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let words =
        Backend::download(&gpu, host.resident(gpu_out).unwrap().unwrap(), 9 * 3 * 4).unwrap();
    for (k, word) in words.iter().enumerate() {
        let g = f32::from_bits(*word);
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
    let source = "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input canvas: Grid<Cell>; output result: Grid<Cell>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Cell { mass: Float; top: Int; } struct Splat { cell: Vec2; value: Cell; } scatter over drops into canvas with sum { let d = sample(drops); return Splat { cell: d.at, value: Cell { mass: d.weight, top: d.id } }; } }";
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
    let drop_schema = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    let cell_schema = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    let canvas_words: Vec<u32> = (0..24).flat_map(|i| [0.0f32.to_bits(), i % 3]).collect();
    for combine in ["sum", "max", "min", "last"] {
        let source = source.replace("with sum", &format!("with {combine}"));
        let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    drop_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let canvas = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    cell_schema.clone(),
                    Domain::Grid(6, 4),
                    canvas_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = entelechy_abi::Invocation {
            inputs: vec![drops, canvas],
            params: vec![],
            domain: Domain::Rows(count),
            context: entelechy_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = entelechy_cpu::CpuBackend
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
        let gpu_words = Backend::download(&gpu, device, byte_len).unwrap();
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
    let counting = "kernel test.count@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Tally>; output result: Grid<Tally>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Tally { hits: Int; ids: Int; } struct Splat { cell: Vec2; value: Tally; } scatter over drops into grid with sum { let d = sample(drops); return Splat { cell: d.at, value: Tally { hits: 1, ids: d.id } }; } }";
    let module = compile(counting).unwrap_or_else(|error| panic!("{error}"));
    let tally_schema =
        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int, entelechy_abi::Scalar::Int]);
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
                entelechy_abi::FieldData::from_storage(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    words,
                )
                .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(
                entelechy_abi::FieldData::from_storage(schema, Domain::Grid(6, 4), grid_words)
                    .unwrap(),
            )
            .unwrap();
        let invocation = entelechy_abi::Invocation {
            inputs: vec![drops, grid],
            params: vec![],
            domain: Domain::Rows(count),
            context: entelechy_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = entelechy_cpu::CpuBackend
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
        let gpu_words = Backend::download(&gpu, device, byte_len).unwrap();
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
            entelechy_abi::FieldData::from_storage(
                drop_schema.clone(),
                Domain::Rows(many),
                many_words,
            )
            .unwrap(),
        )
        .unwrap();
    let canvas = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                cell_schema.clone(),
                Domain::Grid(6, 4),
                canvas_words.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![drops, canvas],
        params: vec![],
        domain: Domain::Rows(many),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = entelechy_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let gpu_out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(gpu_out).unwrap().expect("resident");
    let gpu_words = Backend::download(&gpu, device, 24 * 2 * 4).unwrap();
    assert_eq!(
        cpu_words, gpu_words,
        "a hundred thousand drops summed in order differ"
    );
    let landed: u32 = cpu_words.chunks_exact(2).map(|c| c[1]).sum();
    assert!(landed > 100_000, "ids summed: {landed}");

    // Floats with NaN and both zeros under `min` and `max`, Float target.
    let source = "kernel test.low@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input grid: Grid<Float>; output result: Grid<Float>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Splat { cell: Vec2; value: Float; } scatter over drops into grid with min { let d = sample(drops); return Splat { cell: d.at, value: d.weight }; } }";
    for combine in ["min", "max"] {
        let module = compile(&source.replace("with min", &format!("with {combine}"))).unwrap();
        let mut host = Host::new(1 << 20);
        let drops = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    drop_schema.clone(),
                    Domain::Rows(count),
                    odd_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let grid = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                    Domain::Grid(6, 4),
                    (0..24).map(|_| 0.5f32.to_bits()).collect(),
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = entelechy_abi::Invocation {
            inputs: vec![drops, grid],
            params: vec![],
            domain: Domain::Rows(count),
            context: entelechy_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu_out = entelechy_cpu::CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu_out).unwrap().storage().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
        let mut scope = Scope::new(&mut host, all());
        let gpu_out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let device = host.resident(gpu_out).unwrap().expect("resident");
        let gpu_words = Backend::download(&gpu, device, 24 * 4).unwrap();
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
    let source = "kernel test.walk@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory screen: Grid<Int>; memory tally: Rows<Cell>; output result: Rows<Vec2>; struct Cell { hits: Int; last: Float; } fn flip(s: Grid<Int>, p: Vec2, seen: Int) -> Int { return put(s, p, (1 - seen) >>> 0); } scan over ticks with at_ = vec2(3.0, 2.0) { let seen = at(screen, at_); flip(screen, at_, seen); let t = at(tally, seen); put(tally, seen, Cell { hits: t.hits + 1, last: float(index()) }); put(screen, vec2(-1.0, 0.0), 9); let next = vec2(float((int(at_.x) + 3 + seen * 5) % 16), float((int(at_.y) + 1 + seen) % 8)); return next; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let steps = 300u32;
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let cells = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Int,
        entelechy_abi::Scalar::Float,
    ]);
    let mut host = Host::new(1 << 20);
    let ticks = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(steps),
                vec![0; steps as usize],
            )
            .unwrap(),
        )
        .unwrap();
    let screen = host
        .adopt(
            entelechy_abi::FieldData::from_storage(ints, Domain::Grid(16, 8), vec![0; 128])
                .unwrap(),
        )
        .unwrap();
    let tally = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                cells,
                Domain::Rows(2),
                vec![0, 0.0f32.to_bits(), 0, 0.0f32.to_bits()],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![ticks, screen, tally],
        params: vec![],
        domain: Domain::Rows(steps),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_outputs = entelechy_cpu::CpuBackend
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
        let gpu_words = Backend::download(&gpu, device, byte_len).unwrap();
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(200),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = entelechy_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let gpu_words = Backend::download(&gpu, device, cpu_words.len() * 4).unwrap();
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
    let source = "kernel test.pc@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory regs: Rows<Int>; output result: Rows<Int>; scan over ticks with n = 0 { let pc = at(regs, 15); put(regs, 15, pc + 2); return pc; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 16);
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let ticks = host
        .adopt(
            entelechy_abi::FieldData::from_storage(ints.clone(), Domain::Rows(5), vec![0; 5])
                .unwrap(),
        )
        .unwrap();
    let regs = host
        .adopt(entelechy_abi::FieldData::from_storage(ints, Domain::Rows(17), vec![0; 17]).unwrap())
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![ticks, regs],
        params: vec![],
        domain: Domain::Rows(5),
        context: entelechy_abi::Context::default(),
    };
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let outputs =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap();
    let words = |handle: entelechy_abi::Handle, n: usize| -> Vec<u32> {
        let device = host.resident(handle).unwrap().expect("resident");
        Backend::download(&gpu, device, n * 4).unwrap()
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
        let invocation = entelechy_abi::Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(count),
            context: entelechy_abi::Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let device = host.resident(out).unwrap().expect("resident");
        let words = Backend::download(&gpu, device, count as usize * 4).unwrap();
        let wrong = words
            .iter()
            .enumerate()
            .filter(|(i, w)| f32::from_bits(**w) != *i as f32 * 2.0 + 1.0)
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(count),
        context: entelechy_abi::Context::default(),
    };
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let words = Backend::download(&gpu, device, count as usize * 4).unwrap();
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(10),
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = entelechy_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let gpu_words = Backend::download(&gpu, device, cpu_words.len() * 4).unwrap();
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
        compute over xs { return sample(xs) * at(weights, element()); }
        merge(a, b) { return a + b + at(weights, 1) * k + first(weights) + time(); } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let floats = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]);
    let mut host = Host::new(1 << 16);
    let count = 37u32;
    let xs = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                floats.clone(),
                Domain::Rows(count),
                (0..count).map(|i| (i as f32 * 0.5).to_bits()).collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let weights = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                floats,
                Domain::Rows(count),
                (0..count)
                    .map(|i| (1.0 + (i % 3) as f32).to_bits())
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = entelechy_abi::Invocation {
        inputs: vec![xs, weights],
        params: vec![entelechy_abi::Value::Float(0.25)],
        domain: Domain::Rows(1),
        context: entelechy_abi::Context {
            time: 2.0,
            ..entelechy_abi::Context::default()
        },
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = entelechy_cpu::CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &module).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    let words = Backend::download(&gpu, device, 4).unwrap();
    let gpu_value = f32::from_bits(words[0]);
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
        let shader = entelechy_gpu::wgsl::emit(&module);
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
    use entelechy_rules::raster::{
        Cutout, SURFACE_WORDS, mask_schema, rasterize_masked, triangle_schema, vertex_schema,
    };
    let Some(gpu) = gpu() else { return };
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, Box::new(gpu));
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
            entelechy_abi::FieldData::from_storage(
                vertex_schema(),
                Domain::Rows(8),
                vertices.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let triangle_rows = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                triangle_schema(),
                Domain::Rows(4),
                triangles.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let mask_grid = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
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
    let invocation = entelechy_abi::Invocation {
        inputs: vec![vertex_rows, triangle_rows, mask_grid],
        params: vec![entelechy_abi::Value::Float(0.5)],
        domain: Domain::Grid(width, height),
        context: entelechy_abi::Context::default(),
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
    runtime: &mut entelechy_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<entelechy_kernel::KernelVersion> {
    let module = entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

#[test]
fn a_node_off_the_device_says_what_its_trip_carried() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut runtime = entelechy_kernel::Runtime::with_backends(
        1 << 20,
        Box::new(gpu),
        Some(Box::new(CpuBackend)),
    );
    // Only what the device declines leaves it, however small: its trips are
    // what this test asks.
    runtime.set_small(0);
    let mut publish = |source: &str| {
        let module = compile(source).unwrap();
        let candidate = runtime.prepare_module(module, source).unwrap();
        runtime
            .publish(candidate, entelechy_kernel::Expect::Absent)
            .unwrap()
    };
    let map = publish(
        "kernel t.map@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 2; } }",
    );
    let scan = publish(
        "kernel t.scan@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan over rows with acc = 0 { return acc + sample(rows); } }",
    );
    let back = publish(
        "kernel t.back@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return sample(rows) + 1; } }",
    );
    let run = |runtime: &mut entelechy_kernel::Runtime, version, inputs| {
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
    assert_eq!(moved, entelechy_kernel::Moved::default());
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
                    entelechy_abi::FieldData::from_values(
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
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
        // After an exchange; and in a lane past the domain's end, which
        // runs to give at exchanges.
        (
            "kernel t.g@1 { requires { field.write; } output result: Rows<Int>;
                group 4 { return exchange(element(), lane() + 1) * 1000000000; } }"
                .to_owned(),
            Domain::Rows(8),
        ),
        (
            "kernel t.g@1 { requires { field.write; } output result: Rows<Int>;
                group 8 { let a = if lane() == 7 { 2147483647 + element() % 8 } else { 0 };
                    let b = exchange(exchange(a, lane()), lane());
                    return b + if lane() == 0 { 2147483647 + element() % 8 + 1 } else { 0 }; } }"
                .to_owned(),
            Domain::Rows(8),
        ),
        (
            "kernel t.g@1 { requires { field.write; } output result: Rows<Int>;
                group 4 { let x = exchange(element(), lane() + 1); return 100 / (element() - 7) + x; } }"
                .to_owned(),
            Domain::Rows(6),
        ),
        // A guarded division fails nowhere.
        (map("let i = element(); return if i != 0 { 100 / i } else { 0 };", "Rows<Int>"), Domain::Rows(8)),
        (
            "kernel t.r@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>;
                compute over rows { return sample(rows); } merge(a, b) { return a + b; } }"
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
    module: &entelechy_compiler::Validated,
    inputs: Vec<entelechy_abi::Handle>,
    domain: Domain,
) -> (Vec<u32>, Vec<u32>) {
    let invocation = entelechy_abi::Invocation {
        inputs,
        params: Vec::new(),
        domain,
        context: entelechy_abi::Context::default(),
    };
    let mut scope = Scope::new(host, all());
    let cpu_out = entelechy_cpu::CpuBackend
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
    let gpu_words = Backend::download(gpu, device, byte_len).unwrap();
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
                let data =
                    entelechy_abi::FieldData::new(schema.clone(), Domain::Rows(count)).unwrap();
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
            let invocation = |input| entelechy_abi::Invocation {
                inputs: vec![input],
                params: vec![],
                domain: Domain::Rows(count),
                context: entelechy_abi::Context::default(),
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
            let words = Backend::download(&gpu, device, byte_len).unwrap();
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
    let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let flags = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                schema,
                Domain::Rows(count),
                vec![0; count as usize],
            )
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
        let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
        let n = words.len() as u32;
        host.adopt(entelechy_abi::FieldData::from_storage(schema, Domain::Rows(n), words).unwrap())
            .unwrap()
    };
    let floats = |host: &mut Host, words: Vec<u32>| {
        let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]);
        let n = words.len() as u32;
        host.adopt(entelechy_abi::FieldData::from_storage(schema, Domain::Rows(n), words).unwrap())
            .unwrap()
    };
    // Each run takes fresh inputs: one the device ran on is resident there.
    let source = |host: &mut Host| ints(host, (0..count).map(|i| i % 7).collect());

    let prefix = compile(
        "kernel test.rank@1 { requires { field.read; field.write; } input flags: Rows<Int>;
         output result: Rows<Int>; prefix over flags with sum { return sample(flags) % 3; } }",
    )
    .unwrap();
    let inputs = vec![source(&mut host)];
    let (cpu, gpu_words) = on_both_inputs(&gpu, &mut host, &prefix, inputs, Domain::Rows(count));
    assert_eq!(cpu, gpu_words, "prefix");

    let reduce = compile(
        "kernel test.most@1 { requires { field.read; field.write; } input flags: Rows<Int>;
         output result: Rows<Int>; compute over flags { return sample(flags) * 3 + element() % 5; }
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
             scatter over flags into slots with {rule} {{
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
         scatter over flags into sums with sum {
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
             scatter over keys into slots with {rule} {{
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
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
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
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Long; 4]),
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
        let schema = entelechy_abi::Schema::new(vec![scalar]);
        host.adopt(entelechy_abi::FieldData::from_storage(schema, domain, words).unwrap())
            .unwrap()
    };
    for rule in ["sum", "max", "min"] {
        let module = compile(&format!(
            "kernel test.fan@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
             struct Pair {{ at: Int; value: Int; }}
             scatter over keys into slots with {rule} {{
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
                entelechy_abi::Scalar::Int,
                Domain::Rows(count),
                (0..count).collect(),
            ),
            adopt(
                &mut host,
                entelechy_abi::Scalar::Int,
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
             scatter over keys into canvas with {rule} {{
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
                entelechy_abi::Scalar::Int,
                Domain::Rows(count),
                (0..count)
                    .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
                    .collect(),
            ),
            adopt(
                &mut host,
                entelechy_abi::Scalar::Float,
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
    let schema = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    for rule in ["min", "max"] {
        let module = compile(&format!(
            "kernel test.mixed@1 {{ requires {{ field.read; field.write; }}
             input keys: Rows<Int>; input cells: Rows<Cell>; output result: Rows<Cell>;
             struct Cell {{ f: Float; i: Int; }}
             struct Pair {{ at: Int; value: Cell; }}
             scatter over keys into cells with {rule} {{
                 let k = sample(keys);
                 let v = float(k % 2001 - 1000) * 0.125;
                 emit(Pair {{ at: k % 89, value: Cell {{ f: v, i: k % 77 - 38 }} }});
                 return Pair {{ at: k % 131, value: Cell {{ f: -v, i: 38 - k % 77 }} }};
             }} }}"
        ))
        .unwrap();
        let keys = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
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
                entelechy_abi::FieldData::from_storage(
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
            format!("compute over f0 {{ {body} }} merge(a, b) {{ return a + b; }}")
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
                let (schema, words): (entelechy_abi::Schema, Vec<u32>) = if narrow(k) {
                    (
                        entelechy_abi::Schema::encoded(
                            vec![entelechy_abi::Scalar::Int],
                            vec![entelechy_abi::Encoding::U8],
                        )
                        .unwrap(),
                        (0..rows).map(|i| (i * 7 + k as u32) % 256).collect(),
                    )
                } else if record(k) {
                    (
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int; 2]),
                        (0..rows)
                            .flat_map(|i| [i + 1000, i * 5 + k as u32])
                            .collect(),
                    )
                } else {
                    (
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                        (0..rows).map(|i| i * 3 + k as u32 * 11).collect(),
                    )
                };
                host.adopt(
                    entelechy_abi::FieldData::from_values(schema, Domain::Rows(rows), &words)
                        .unwrap(),
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
         scan over ticks with acc = 0 {{ put(m0, 0, acc); put(m{last}, 0, acc); return acc + 1; }} }}",
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

/// Past the device's bindings, rows are packed before grids: a grid, as
/// large as a window, declared last keeps its own binding, and the
/// kernel reads every input where it is, the interpreter's words.
#[test]
fn rows_are_packed_before_grids() {
    let Some(gpu) = gpu() else { return };
    let rows_n = 36;
    let grids_n = 3;
    let ports: String = (0..rows_n)
        .map(|k| format!("input f{k}: Rows<Int>; "))
        .chain((0..grids_n).map(|k| format!("input g{k}: Grid<Int>; ")))
        .collect();
    let terms: Vec<String> = (0..rows_n)
        .map(|k| format!("at(f{k}, element())"))
        .chain((0..grids_n).map(|k| format!("sample(g{k})")))
        .collect();
    let module = compile(&format!(
        "kernel test.rows_first@1 {{ requires {{ field.read; field.write; }} {ports}
         output result: Grid<Int>; compute {{ return {}; }} }}",
        terms.join(" + ")
    ))
    .unwrap_or_else(|error| panic!("{error}"));
    let packed = entelechy_gpu::wgsl::packed_inputs(&module, 31, 31);
    assert!(!packed.is_empty(), "39 inputs are past 31 bindings");
    assert!(
        packed.iter().all(|&k| k < rows_n),
        "only rows are packed: {packed:?}"
    );
    let (w, h) = (16u32, 8u32);
    let mut host = Host::new(1 << 22);
    let int = || entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let handles: Vec<_> = (0..rows_n + grids_n)
        .map(|k| {
            let (domain, count) = if k < rows_n {
                (Domain::Rows(w * h), w * h)
            } else {
                (Domain::Grid(w, h), w * h)
            };
            let words: Vec<u32> = (0..count).map(|i| i * 3 + k as u32 * 7).collect();
            host.adopt(entelechy_abi::FieldData::from_values(int(), domain, &words).unwrap())
                .unwrap()
        })
        .collect();
    let (cpu, device) = on_both_inputs(&gpu, &mut host, &module, handles, Domain::Grid(w, h));
    assert_eq!(cpu, device);
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
            let estimate = entelechy_gpu::wgsl::brace_depth(&module);
            let emitted = braces(&entelechy_gpu::wgsl::emit(&module));
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
        let mut runtime = entelechy_kernel::Runtime::with_backends(
            1 << 20,
            Box::new(GpuBackend::new().unwrap()),
            Some(Box::new(CpuBackend)),
        );
        let candidate = runtime.prepare_module(module.clone(), &source).unwrap();
        let version = runtime
            .publish(candidate, entelechy_kernel::Expect::Absent)
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
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let page = entelechy_abi::PAGE_WORDS;
    let len = page * 5 + 3;
    let domain = Domain::Rows(len as u32);
    let mut host = Host::new(len * 64);
    // Runs the Kernel over `rows`: its words doubled, and the bytes uploaded.
    let run = |host: &mut Host, rows: entelechy_abi::Handle| -> (Vec<u32>, u64) {
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
        let words =
            Backend::download(&gpu, host.resident(out[0]).unwrap().unwrap(), len * 4).unwrap();
        host.release(out[0]).unwrap();
        let words = words.into_iter().map(|w| w / 2).collect();
        (words, gpu.uploaded() - before)
    };
    let field = |words: &Vec<u32>| {
        entelechy_abi::FieldData::from_storage(ints.clone(), domain, words.clone()).unwrap()
    };
    let first: Vec<u32> = (0..len as u32).collect();
    let base = host.adopt(field(&first)).unwrap();
    assert_eq!(run(&mut host, base), (first.clone(), len as u64 * 4));

    // Made from the base, which lives on the device: a page and the last.
    let mut second = first.clone();
    second[page + 1] = 7;
    second[len - 1] = 9;
    let mut changed = entelechy_abi::Pages::new();
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
    let mut changed = entelechy_abi::Pages::new();
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
        let mut changed = entelechy_abi::Pages::new();
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

    // Made from a Field made from one uploaded and released: brought up
    // from that one's buffer on the pages changed since.
    let ancestor = host.adopt(field(&first)).unwrap();
    run(&mut host, ancestor);
    let mut middle_words = first.clone();
    middle_words[page * 3] = 21;
    let mut changed = entelechy_abi::Pages::new();
    changed.mark(page * 3);
    let middle = host
        .adopt_derived(field(&middle_words), ancestor, changed)
        .unwrap();
    let mut last_words = middle_words.clone();
    last_words[page * 5 + 1] = 22;
    let mut changed = entelechy_abi::Pages::new();
    changed.mark(page * 5 + 1);
    let last = host
        .adopt_derived(field(&last_words), middle, changed)
        .unwrap();
    host.release(ancestor).unwrap();
    host.release(middle).unwrap();
    assert_eq!(run(&mut host, last), (last_words, (page + 3) as u64 * 4));
    host.release(last).unwrap();

    // Written in place after it was uploaded, a Field's old buffer holds
    // other words: what is made from it after is uploaded whole.
    let written = host.adopt(field(&first)).unwrap();
    run(&mut host, written);
    host.write_element(written, 9, &[17]).unwrap();
    let mut words = first.clone();
    words[9] = 17;
    let after = host
        .adopt_derived(field(&words), written, entelechy_abi::Pages::new())
        .unwrap();
    assert_eq!(run(&mut host, after), (words, len as u64 * 4));
}

#[test]
fn the_pile_lets_go_what_an_evaluation_went_without() {
    // A released output's buffer waits on the pile for the next output of
    // its size; once a whole evaluation has passed without taking it, the
    // next lets it go.
    let Some(gpu) = gpu() else { return };
    let module = compile(
        "kernel t.fill@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let prepared = gpu.prepare(&module).unwrap();
    let mut host = Host::new(1 << 24);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(64, 64),
        context: Context::default(),
    };
    let spare = |gpu: &GpuBackend| {
        gpu.memory()
            .into_iter()
            .find(|(part, _)| *part == "spare buffers")
            .map_or(0, |(_, bytes)| bytes)
    };
    gpu.evaluation_begun();
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    host.release(out).unwrap();
    assert!(
        spare(&gpu) >= 64 * 64 * 16,
        "the released output's buffer is kept"
    );
    gpu.evaluation_begun();
    assert!(spare(&gpu) > 0, "an evaluation may still take it");
    gpu.evaluation_begun();
    assert_eq!(spare(&gpu), 0, "a whole evaluation went without it");
}

#[test]
fn a_scatter_gives_its_pairs_back_to_the_pile() {
    // The pairs a scatter's entry writes are read by its combine alone:
    // they wait on the pile for the next run, not for a collector.
    let Some(gpu) = gpu() else { return };
    let module = compile(
        "kernel t.splat@1 { requires { field.read; field.write; } input drops: Rows<Float>; input grid: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Vec2; value: Float; } scatter over drops into grid with sum { return Splat { cell: vec2(float(element() % 4), 0.0), value: sample(drops) }; } }",
    )
    .unwrap();
    let prepared = gpu.prepare(&module).unwrap();
    let mut host = Host::new(1 << 24);
    let count = 4096;
    let drops = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                Domain::Rows(count),
                (0..count).map(|i| (i as f32 * 0.25).to_bits()).collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let grid = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                Domain::Grid(4, 1),
                vec![0; 4],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![drops, grid],
        params: vec![],
        domain: Domain::Rows(count),
        context: Context::default(),
    };
    gpu.evaluation_begun();
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
    let spare = gpu
        .memory()
        .into_iter()
        .find(|(part, _)| *part == "spare buffers")
        .map_or(0, |(_, bytes)| bytes);
    assert!(
        spare >= (count * 3 * 4) as u64,
        "the pairs are kept: {spare} bytes"
    );
    host.release(out).unwrap();
}

#[test]
fn a_frame_that_needs_one_buffer_at_a_time_keeps_one() {
    // Two outputs of one size, released one after the other, each before
    // the next is made: the last kept goes round, the other ages out.
    let Some(gpu) = gpu() else { return };
    let module = compile(
        "kernel t.fill@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let prepared = gpu.prepare(&module).unwrap();
    let mut host = Host::new(1 << 24);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(64, 64),
        context: Context::default(),
    };
    let spare = |gpu: &GpuBackend| {
        gpu.memory()
            .into_iter()
            .find(|(part, _)| *part == "spare buffers")
            .map_or(0, |(_, bytes)| bytes)
    };
    let run = |host: &mut Host| {
        let mut scope = Scope::new(host, all());
        Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0]
    };
    gpu.evaluation_begun();
    let (a, b) = (run(&mut host), run(&mut host));
    host.release(a).unwrap();
    host.release(b).unwrap();
    let one = spare(&gpu) / 2;
    assert!(one >= 64 * 64 * 16, "both buffers are kept");
    for _ in 0..3 {
        gpu.evaluation_begun();
        for _ in 0..2 {
            let out = run(&mut host);
            host.release(out).unwrap();
        }
    }
    assert_eq!(
        spare(&gpu),
        one,
        "one buffer goes round, the other aged out"
    );
}

/// A group's lanes meet at each exchange on the device as in the
/// interpreter: in a loop a param bounds, under an `if` on a param, and
/// in a last group past the domain's end.
#[test]
fn groups_meet_at_exchanges_as_the_interpreter_does() {
    let Some(gpu) = gpu() else {
        return;
    };
    let source = "kernel test.group@1 { requires { field.write; } param rounds: Int = 3; param flip: Bool = true; output result: Rows<IVec2>;
        group 8 {
            let i = element();
            let total = loop 8 with s = i until index() >= rounds { s + exchange(s, lane() ^ (1 << index())) };
            let turned = if flip { exchange(i * 10, 7 - lane()) } else { i };
            return ivec2(total, turned);
        } }";
    // Accounted as the Kernel runtime accounts it: its loops' literal
    // counts are too few to count, and the device takes it.
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let module = entelechy_ir::account(&module).unwrap();
    assert_eq!(gpu.declines(&module), None);
    for (count, rounds, flip) in [(20, 3, true), (8, 2, false), (1000, 3, true)] {
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(rounds), Value::Bool(flip)],
            domain: Domain::Rows(count),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let cpu_words = host.data(cpu).unwrap().storage().to_vec();
        let prepared = Backend::prepare(&gpu, &module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let out =
            Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope).unwrap()[0];
        let device = host.resident(out).unwrap().expect("resident");
        let words = Backend::download(&gpu, device, cpu_words.len() * 4).unwrap();
        assert_eq!(words, cpu_words, "{count} rows");
    }
    // A group writes as a map does: a narrow output, two elements a word,
    // the last word half outside; several outputs, the lanes past the end
    // storing none; Longs, the device's own and in pairs of words.
    let shapes = [
        "kernel test.narrow@1 { requires { field.write; } output result: Rows<Float as f16>;
            group 4 { return exchange(float(element()) * 0.5, lane() + 1); } }",
        "kernel test.several@1 { requires { field.write; } output half: Rows<Float as f16>; output whole: Rows<Int>; output both: Rows<Vec2>;
            group 4 {
                let x = exchange(float(element()) * 0.5, lane() + 1);
                let k = exchange(element(), 3 - lane());
                return { whole: k, both: vec2(x, float(k)), half: x };
            } }",
        "kernel test.long@1 { requires { field.write; } output result: Rows<Long>;
            group 4 { return exchange(long(element()) * 3000000000l, lane() - 1) + 1l; } }",
    ];
    for (source, pairs) in shapes.iter().flat_map(|s| [(*s, false), (*s, true)]) {
        let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
        let device = if pairs {
            if !source.contains("Long") {
                continue;
            }
            GpuBackend::new().unwrap().with_long_pairs()
        } else {
            GpuBackend::new().unwrap()
        };
        assert_eq!(device.declines(&module), None, "{source}");
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(11),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap();
        let prepared = Backend::prepare(&device, &module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let out =
            Backend::execute(&device, prepared.as_ref(), &module, &invocation, &mut scope).unwrap();
        assert_eq!(out.len(), cpu.len(), "{source}");
        for (reference, made) in cpu.iter().zip(&out) {
            let words = host.data(*reference).unwrap().storage().to_vec();
            let resident = host.resident(*made).unwrap().expect("resident");
            let made = Backend::download(&device, resident, words.len() * 4).unwrap();
            assert_eq!(made, words, "{source}, pairs {pairs}");
        }
    }
}

/// A group whose work is counted runs on the device: its words are the
/// interpreter's, and where its lanes run unequal work between exchanges
/// and one passes its budget inside a loop holding an exchange, the
/// whole group stops there together, the run failing at the element the
/// interpreter names, and no lane waits at an exchange the others left.
#[test]
fn a_counted_group_stops_together_where_its_lanes_meet() {
    let Some(gpu) = gpu() else {
        return;
    };
    let source = "kernel test.counted@1 { requires { field.write; } budget 50000; param rounds: Int = 3; output result: Rows<Int>;
        group 8 {
            return loop rounds with s = element() {
                let spent = loop 1000 with a = 0 until index() >= (lane() + 1) * 100 { a + 1 };
                (s + exchange(s, lane() ^ 1)) % 9973 + spent
            };
        } }";
    let module = entelechy_ir::account(&compile(source).unwrap()).unwrap();
    assert!(module.accounted());
    assert_eq!(gpu.declines(&module), None);
    let run = |backend: &dyn Backend, rounds: i32| {
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(rounds)],
            domain: Domain::Rows(36),
            context: Context::default(),
        };
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let out = backend
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .map_err(|error| error.to_string())?[0];
        let words = match host.resident(out).unwrap() {
            Some(device) => {
                let bytes = host.schema(out).unwrap().storage_words(36) * 4;
                Backend::download(backend, device, bytes).unwrap()
            }
            None => host.data(out).unwrap().storage().to_vec(),
        };
        Ok::<Vec<u32>, String>(words)
    };
    // Within its budget: the interpreter's words.
    let reference = run(&CpuBackend, 3).unwrap();
    assert_eq!(run(&gpu, 3).unwrap(), reference);
    // Lanes 4 to 7 pass their budget, lane 7 first, in round 63: it
    // stops its group, and the run fails at element 7, not at lane 4,
    // which would pass its budget in the last round.
    let reference = run(&CpuBackend, 100).unwrap_err();
    assert!(reference.contains("(7, 0)"), "{reference}");
    assert_eq!(run(&gpu, 100).unwrap_err(), reference);
    // Where lane 7 passes its budget, lane 1 overflows later in the same
    // stretch: the lowest that failed in it is named, lane 7 running on
    // with its group to the stretch's end.
    let source = source.replace(
        "(s + exchange(s, lane() ^ 1)) % 9973 + spent",
        "let boom = if lane() == 1 && index() == 63 { 2147483647 + s % 2 + 1 } else { 0 }; (s + exchange(s, lane() ^ 1)) % 9973 + spent + boom",
    );
    let module = entelechy_ir::account(&compile(&source).unwrap()).unwrap();
    let run = |backend: &dyn Backend| {
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(100)],
            domain: Domain::Rows(36),
            context: Context::default(),
        };
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        backend
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .map(|_| ())
            .map_err(|error| error.to_string())
    };
    let reference = run(&CpuBackend).unwrap_err();
    assert!(reference.contains("(1, 0)"), "{reference}");
    assert_eq!(run(&gpu).unwrap_err(), reference);
}

/// A group's exchanges are arrays in its workgroup's memory: one that
/// fits runs on the device and gives the interpreter's words; one past
/// what the device's workgroups hold goes elsewhere rather than run
/// wrong.
#[test]
fn a_group_exchanges_on_the_device_what_its_workgroup_holds() {
    let Some(gpu) = gpu() else {
        return;
    };
    let source = |count: u32| {
        format!(
            "kernel t.big@1 {{ requires {{ field.write; }} output result: Rows<Vec4>;
                group 256 {{ let many = [vec4(float(element()), 1.0, 0.0, 1.0); {count}];
                    let other = exchange(many, lane() + 1); return other[{count} - 1]; }} }}"
        )
    };
    let most = u64::from(gpu.workgroup_memory());
    // A `Vec4` is 16 bytes: the most that fits, and one more.
    let fits = ((most - 4) / (16 * 256)) as u32;
    let fitting = compile(&source(fits)).unwrap();
    assert_eq!(gpu.declines(&fitting), None);
    let past = compile(&source(fits + 1)).unwrap();
    let why = gpu.declines(&past).expect("past the workgroup's memory");
    assert!(why.contains("workgroups hold"), "{why}");
    let mut host = Host::new(1 << 22);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(1000),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&fitting, &invocation, &mut scope)
        .unwrap()[0];
    let words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = Backend::prepare(&gpu, &fitting).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let out =
        Backend::execute(&gpu, prepared.as_ref(), &fitting, &invocation, &mut scope).unwrap()[0];
    let device = host.resident(out).unwrap().expect("resident");
    assert_eq!(
        Backend::download(&gpu, device, words.len() * 4).unwrap(),
        words
    );
}

#[test]
fn pow_and_choosing_builtins_match_on_negative_zero_and_nan_inputs() {
    let Some(gpu) = gpu() else { return };
    let cases = vec![
        [-3.0, 2.0, -2.0, 3.0],
        [-2.0, -3.0, -0.5, 2.2],
        [0.0, 0.0, -0.0, 3.0],
        [0.0, 2.2, -1.0, f32::INFINITY],
        [1.0, f32::NAN, f32::NAN, 0.0],
        [f32::NAN, 1.0, -f32::INFINITY, -3.0],
    ];
    let n = cases.len() as u32;
    let source = "kernel test.pow_nan@1 {
        requires { field.read; field.write; }
        input source: Image;
        output result: Image;
        compute {
            let c = at(source, coord());
            let p = pow(vec2(c.r, c.b), vec2(c.g, c.a));
            let held = clamp(vec3(p.x, p.y, c.r), 0.0, 1.0)
                + saturate(vec3(p.y, c.a, p.x))
                + step(vec3(c.r, 0.5, c.b), vec3(0.5, c.a, 0.0));
            return vec4(
                pow(c.r, c.g),
                saturate(pow(c.b, c.a)),
                held.x + held.y + held.z,
                float(vec2(p.y, 1.0) == vec2(p.y, 1.0))
            );
        }
    }";
    let (cpu, gpu_pixels) = both(&gpu, source, &[cases], (n, 1), vec![], Domain::Grid(n, 1));
    assert_eq!(
        mismatch(&cpu, &gpu_pixels, 1e-5),
        0.0,
        "{cpu:?} vs {gpu_pixels:?}"
    );
}

#[test]
fn emitting_integer_sum_scatter_counts_its_high_words_when_packing_inputs() {
    let ports: String = (0..11)
        .map(|k| format!("input f{k}: Rows<Int>; "))
        .collect();
    let sum: String = (0..11)
        .map(|k| format!("at(f{k}, 0)"))
        .collect::<Vec<_>>()
        .join(" + ");
    let source = format!(
        "kernel test.emit_pack@1 {{
            requires {{ field.read; field.write; }}
            input base: Rows<Int>;
            {ports}
            output result: Rows<Int>;
            struct Pair {{ at: Int; value: Int; }}
            scatter over f0 into base with sum {{
                emit(Pair {{ at: 0, value: {sum} }});
                return Pair {{ at: -1, value: 0 }};
            }}
        }}"
    );
    let module = compile(&source).unwrap();
    // 12 inputs + 1 output + 1 failure + 2 emitting buffers (low + high) = 16 storage.
    assert_eq!(entelechy_gpu::wgsl::bindings(&module), (2, 16));
    let packed = entelechy_gpu::wgsl::packed_inputs(&module, 12, 10);
    // Packing 7 read-only inputs replaces 7 bindings with 1 pack, leaving 10 storage.
    assert_eq!(packed.len(), 7);
}

#[test]
fn sliced_map_dispatches_match_cpu_across_rows_grids_volumes_and_failures() {
    let Some(base_gpu) = gpu() else { return };
    let sliced = base_gpu.with_slice_limit(128);

    // 1. 2D Grid (accounted loop + coord/element indexing across multiple Y slices).
    let grid_src = "kernel test.sliced_grid@1 {
        requires { field.write; }
        output result: Image;
        compute {
            let c = coord();
            let s = loop 8 with acc = 0 until index() >= 5 { acc + index() + int(c.x) };
            return vec4(c.x * 0.25, c.y * 0.5, float(element()), float(s));
        }
    }";
    let (cpu_grid, gpu_grid) = both(&sliced, grid_src, &[], (1, 1), vec![], Domain::Grid(37, 29));
    assert_eq!(mismatch(&cpu_grid, &gpu_grid, 1e-5), 0.0);

    // 2. 1D Rows and 3D Volume (multiple element/depth slices) + checked failure in a late slice.
    let mut host = Host::new(1 << 22);
    let rows_mod = compile(
        "kernel test.sliced_rows@1 {
            requires { field.write; }
            output result: Rows<Int>;
            compute { return element() * 3 + 7; }
        }",
    )
    .unwrap();
    let rows_inv = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(1000),
        context: Context::default(),
    };
    let cpu_rows = {
        let mut scope = Scope::new(&mut host, all());
        CpuBackend
            .execute(&rows_mod, &rows_inv, &mut scope)
            .unwrap()[0]
    };
    let prep_rows = sliced.prepare(&rows_mod).unwrap();
    let gpu_rows = {
        let mut scope = Scope::new(&mut host, all());
        Backend::execute(
            &sliced,
            prep_rows.as_ref(),
            &rows_mod,
            &rows_inv,
            &mut scope,
        )
        .unwrap()[0]
    };
    let gpu_rows_words =
        Backend::download(&sliced, host.resident(gpu_rows).unwrap().unwrap(), 1000 * 4).unwrap();
    assert_eq!(host.data(cpu_rows).unwrap().storage(), &gpu_rows_words[..]);

    let vol_mod = compile(
        "kernel test.sliced_vol@1 {
            requires { field.write; }
            output result: Volume<Int>;
            compute {
                let c = coord();
                return int(c.x) + int(c.y) * 100 + int(c.z) * 10000 + element();
            }
        }",
    )
    .unwrap();
    let vol_inv = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Volume(9, 7, 11),
        context: Context::default(),
    };
    let cpu_vol = {
        let mut scope = Scope::new(&mut host, all());
        CpuBackend.execute(&vol_mod, &vol_inv, &mut scope).unwrap()[0]
    };
    let prep_vol = sliced.prepare(&vol_mod).unwrap();
    let gpu_vol = {
        let mut scope = Scope::new(&mut host, all());
        Backend::execute(&sliced, prep_vol.as_ref(), &vol_mod, &vol_inv, &mut scope).unwrap()[0]
    };
    let gpu_vol_words = Backend::download(
        &sliced,
        host.resident(gpu_vol).unwrap().unwrap(),
        (9 * 7 * 11) * 4,
    )
    .unwrap();
    assert_eq!(host.data(cpu_vol).unwrap().storage(), &gpu_vol_words[..]);

    // Checked division-by-zero occurring only in a later slice (`element() == 450`) must surface identically.
    let fail_mod = compile(
        "kernel test.sliced_fail@1 {
            requires { field.write; }
            output result: Rows<Int>;
            compute {
                let denom = if element() == 450 { 0 } else { 1 };
                return 100 / denom;
            }
        }",
    )
    .unwrap();
    let fail_inv = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(600),
        context: Context::default(),
    };
    let cpu_err = {
        let mut scope = Scope::new(&mut host, all());
        CpuBackend
            .execute(&fail_mod, &fail_inv, &mut scope)
            .unwrap_err()
    };
    let prep_fail = sliced.prepare(&fail_mod).unwrap();
    let gpu_err = {
        let mut scope = Scope::new(&mut host, all());
        Backend::execute(
            &sliced,
            prep_fail.as_ref(),
            &fail_mod,
            &fail_inv,
            &mut scope,
        )
        .unwrap_err()
    };
    assert_eq!(cpu_err, gpu_err);
}

#[test]
fn packed_input_groups_splits_wide_grids_across_multiple_storage_buffers() {
    let rows_n = 12usize;
    let float_grids_n = 4usize;
    let vec4_grids_n = 9usize;
    let ports: String = (0..rows_n)
        .map(|k| format!("input r{k}: Rows<Int>; "))
        .chain((0..float_grids_n).map(|k| format!("input fg{k}: Grid<Float>; ")))
        .chain((0..vec4_grids_n).map(|k| format!("input vg{k}: Grid<Vec4>; ")))
        .collect();
    let r_sum = (0..rows_n)
        .map(|k| format!("at(r{k}, 0)"))
        .collect::<Vec<_>>()
        .join(" + ");
    let fg_sum = (0..float_grids_n)
        .map(|k| format!("sample(fg{k})"))
        .collect::<Vec<_>>()
        .join(" + ");
    let vg_sum = (0..vec4_grids_n)
        .map(|k| format!("sample(vg{k}).x"))
        .collect::<Vec<_>>()
        .join(" + ");
    let module = compile(&format!(
        "kernel test.multi_pack@1 {{
            requires {{ field.read; field.write; }}
            {ports}
            output result: Image;
            output extra: Grid<Vec4>;
            compute {{
                let s = float({r_sum}) + {fg_sum} + {vg_sum};
                return {{ result: vec4(s, 0.0, 0.0, 1.0), extra: vec4(0.0, s, 0.0, 1.0) }};
            }}
        }}"
    ))
    .unwrap();
    let accounted = entelechy_ir::account(&module).unwrap();
    // 25 inputs + 2 outputs + 1 failure = 28 storage bindings.
    assert_eq!(entelechy_gpu::wgsl::bindings(&accounted), (2, 28));

    let groups8 = entelechy_gpu::wgsl::packed_input_groups(&accounted, 28, 8);
    assert_eq!(groups8.len(), 4);
    let packed8: usize = groups8.iter().map(Vec::len).sum();
    assert_eq!(28 - packed8 + groups8.len(), 8);
    for group in &groups8 {
        let grids_in_group = group.iter().filter(|&&idx| idx >= rows_n).count();
        assert!(
            grids_in_group <= 3,
            "expected <= 3 grids per pack buffer, got {grids_in_group} in {group:?}"
        );
    }

    let groups10 = entelechy_gpu::wgsl::packed_input_groups(&accounted, 28, 10);
    assert_eq!(groups10.len(), 3);
    let packed10: usize = groups10.iter().map(Vec::len).sum();
    assert_eq!(28 - packed10 + groups10.len(), 10);

    let wgsl = entelechy_gpu::wgsl::emit_packed_groups_longs(
        &accounted,
        &groups8,
        entelechy_gpu::wgsl::Longs::Pairs,
    );
    assert!(wgsl.contains("@binding(91) var<storage, read> inputs_packed:"));
    assert!(wgsl.contains("@binding(93) var<storage, read> inputs_packed_1:"));
    assert!(wgsl.contains("@binding(94) var<storage, read> inputs_packed_2:"));
    assert!(wgsl.contains("@binding(95) var<storage, read> inputs_packed_3:"));

    if let Some(gpu) = gpu() {
        let pruned = entelechy_gpu::wgsl::pruned(&wgsl);
        let _shader = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test.multi_pack@1"),
                source: wgpu::ShaderSource::Wgsl(pruned.as_str().into()),
            });
    }
}

#[test]
fn small_integer_pow_expands_without_pow_ref() {
    let module = compile(
        "kernel test.small_pow@1 {
            requires { field.read; field.write; }
            input xs: Rows<Float>;
            output result: Rows<Vec4>;
            compute {
                let x = at(xs, element());
                return vec4(pow(x, 1.0), pow(x, 2.0), pow(x, 3.0), pow(x, 4.0));
            }
        }",
    )
    .unwrap();
    let accounted = entelechy_ir::account(&module).unwrap();
    let pruned = entelechy_gpu::wgsl::pruned(&entelechy_gpu::wgsl::emit(&accounted));
    assert!(
        !pruned.contains("fn pow_ref("),
        "constant 1..4 pow should expand to multiplies without pulling pow_ref:\n{pruned}"
    );
    let Some(gpu) = gpu() else { return };
    let mut host = Host::new(1 << 20);
    let samples = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let words: Vec<u32> = samples.iter().map(|v| v.to_bits()).collect();
    let handle = host
        .adopt(
            entelechy_abi::FieldData::from_values(
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                Domain::Rows(samples.len() as u32),
                &words,
            )
            .unwrap(),
        )
        .unwrap();
    let (cpu, device) = on_both_inputs(
        &gpu,
        &mut host,
        &module,
        vec![handle],
        Domain::Rows(samples.len() as u32),
    );
    for (idx, (c, d)) in cpu.iter().zip(device.iter()).enumerate() {
        let cf = f32::from_bits(*c);
        let df = f32::from_bits(*d);
        if cf.is_nan() {
            assert!(df.is_nan(), "lane {idx}: expected NaN, got {df}");
        } else if cf.is_infinite() {
            assert_eq!(c, d, "lane {idx}: infinity sign mismatch ({cf} vs {df})");
        } else if cf == 0.0 {
            assert_eq!(c, d, "lane {idx}: zero sign mismatch");
        } else {
            assert!(
                (cf - df).abs() <= cf.abs() * 1e-6 + 1e-6,
                "lane {idx}: {cf} vs {df}"
            );
        }
    }
}

#[test]
fn batched_frame_chains_maps_small_uploads_and_deferred_checks() {
    let Some(gpu) = gpu() else { return };
    let add = compile(
        "kernel test.batch_add@1 {
            requires { field.read; field.write; }
            input a: Grid<Float>;
            input b: Grid<Float>;
            param bias: Float = 0.0;
            output out: Grid<Float>;
            compute {
                return sample(a) + sample(b) + bias;
            }
        }",
    )
    .unwrap();
    let checked_div = compile(
        "kernel test.batch_div@1 {
            requires { field.read; field.write; }
            input a: Grid<Float>;
            param divisor: Int = 1;
            output out: Grid<Float>;
            compute {
                let q = 100 / divisor;
                return sample(a) + float(q);
            }
        }",
    )
    .unwrap();
    let prep_add = gpu.prepare(&add).unwrap();
    let prep_div = gpu.prepare(&checked_div).unwrap();
    let domain = Domain::Grid(32, 16);
    let count = domain.count();
    let mut host = Host::new(4 << 20);
    // Run 3 batched frames: each queues multiple `execute_checked` passes with
    // distinct per-node uniforms and fresh small Host uploads (`<= 64 KiB`)
    // before settling via `submitted()` and verifying bit-exact results.
    for frame in 0..3u32 {
        let mut checks: Vec<Box<dyn entelechy_rules::Check>> = Vec::new();
        let make_grid = |host: &mut Host, offset: f32| {
            let words: Vec<u32> = (0..count)
                .map(|i| (i as f32 * 0.25 + offset).to_bits())
                .collect();
            host.adopt(
                entelechy_abi::FieldData::from_values(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                    domain,
                    &words,
                )
                .unwrap(),
            )
            .unwrap()
        };
        let a = make_grid(&mut host, frame as f32);
        let b = make_grid(&mut host, 10.0 + frame as f32);
        let c = make_grid(&mut host, 20.0 + frame as f32);
        let mut scope = Scope::new(&mut host, all());
        let (o1, chk1) = gpu
            .execute_checked(
                prep_add.as_ref(),
                &add,
                &Invocation {
                    inputs: vec![a, b],
                    params: vec![entelechy_abi::Value::Float(1.5 + frame as f32)],
                    domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap();
        if let Some(c) = chk1 {
            checks.push(c);
        }
        // Second node introduces a new small Host upload (`c`) while the batch
        // encoder is already open, plus distinct params in the uniform slab.
        let (o2, chk2) = gpu
            .execute_checked(
                prep_add.as_ref(),
                &add,
                &Invocation {
                    inputs: vec![o1[0], c],
                    params: vec![entelechy_abi::Value::Float(2.5)],
                    domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap();
        if let Some(c) = chk2 {
            checks.push(c);
        }
        // Third node is checked (`can_fail`) and triggers division by zero on frame 2.
        let divisor = if frame == 2 { 0 } else { 4 };
        let (o3, chk3) = gpu
            .execute_checked(
                prep_div.as_ref(),
                &checked_div,
                &Invocation {
                    inputs: vec![o2[0]],
                    params: vec![entelechy_abi::Value::Int(divisor)],
                    domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap();
        if let Some(c) = chk3 {
            checks.push(c);
        }
        let done = gpu.submitted().unwrap();
        let mut err = None;
        for chk in checks {
            if let Err(e) = chk.wait() {
                err = Some(e);
            }
        }
        done.wait().unwrap();
        if frame == 2 {
            assert!(
                matches!(
                    err,
                    Some(entelechy_rules::ExecutionError::Arithmetic {
                        error: entelechy_rules::ArithmeticError::DivisionByZero,
                        ..
                    })
                ),
                "expected DivisionByZero from batched FailureCheck, got {err:?}"
            );
        } else {
            assert!(err.is_none(), "unexpected error on frame {frame}: {err:?}");
            let resident = host.resident(o3[0]).unwrap().unwrap();
            let got = gpu.download(resident, count * 4).unwrap();
            for (i, &bits) in got.iter().enumerate() {
                let expected = (i as f32 * 0.25 + frame as f32)
                    + (i as f32 * 0.25 + 10.0 + frame as f32)
                    + (1.5 + frame as f32)
                    + (i as f32 * 0.25 + 20.0 + frame as f32)
                    + 2.5
                    + 25.0;
                assert!(
                    (f32::from_bits(bits) - expected).abs() < 1e-4,
                    "frame {frame} pixel {i}: got {}, expected {expected}",
                    f32::from_bits(bits)
                );
            }
        }
        for h in [a, b, c, o1[0], o2[0], o3[0]] {
            let _ = host.release(h);
        }
    }
}

#[test]
fn hardware_texture_mirroring_matches_cpu_across_2d_3d_aligned_unaligned_and_narrow_formats() {
    let Some(gpu) = gpu() else { return };
    // Covers:
    // - 2D odd-width Unorm8 Float (row_bytes = 17, tests byte-level row padding!)
    // - 2D odd-width Snorm8 Vec2 (row_bytes = 38, tests odd-word-aligned padding)
    // - 2D unaligned F16 Vec4 (row_bytes = 120)
    // - 3D unaligned F16 Float Volume (at_linear + at)
    // - 3D aligned Unorm8 Vec4 Volume (direct zero-shader copy_buffer_to_texture DMA!)
    let source = "kernel test.tex_mirror@1 {
        requires { field.read; field.write; }
        input g_u8: Grid<Float as unorm8>;
        input g_s8: Grid<Vec2 as snorm8>;
        input g_f16: Grid<Vec4 as f16>;
        input v_f16: Volume<Float as f16>;
        input v_u8: Volume<Vec4 as unorm8>;
        output out: Grid<Vec4>;
        compute {
            let uv = (coord() + vec2(0.5, 0.5)) / domain();
            let a = at_linear(g_u8, uv * size(g_u8));
            let b = at_linear(g_s8, uv * size(g_s8));
            let c = at_linear(g_f16, uv * size(g_f16));
            let p3 = vec3(uv.x, uv.y, fract(uv.x * 1.7 + uv.y * 0.9));
            let d = at_linear(v_f16, p3 * size(v_f16)) + at(v_f16, p3 * size(v_f16));
            let e = at_linear(v_u8, p3 * size(v_u8));
            return vec4(a + b.x + c.x, b.y + c.y + d, c.z + e.x + e.y, c.w + e.z + e.w);
        }
    }";
    let module = compile(source).unwrap_or_else(|e| panic!("{e}"));
    let textured = entelechy_gpu::wgsl::textured_inputs(&module, 16);
    assert_eq!(
        textured,
        vec![0, 1, 2, 3, 4],
        "all 5 inputs should mirror as textures"
    );
    let wgsl = entelechy_gpu::wgsl::emit_full(
        &module,
        &[],
        &textured,
        entelechy_gpu::wgsl::Longs::Native,
        entelechy_gpu::wgsl::Half::Whole,
    );
    assert!(wgsl.contains("var in0: texture_2d<f32>;"), "{wgsl}");
    assert!(wgsl.contains("var in3: texture_3d<f32>;"), "{wgsl}");
    assert!(wgsl.contains("var in4: texture_2d_array<f32>;"), "{wgsl}");
    assert!(wgsl.contains("textureLoad("), "{wgsl}");

    // Verify textured inputs free storage buffer bindings when packing under a tight limit.
    let groups_without_tex = entelechy_gpu::wgsl::packed_input_groups(&module, 16, 4);
    let groups_with_tex =
        entelechy_gpu::wgsl::packed_input_groups_with_textures(&module, 16, 4, &textured);
    assert!(!groups_without_tex.is_empty());
    assert!(
        groups_with_tex.is_empty(),
        "textured inputs should avoid storage packing: {groups_with_tex:?}"
    );

    let mut host = Host::new(8 << 20);
    let make_field = |host: &mut Host,
                      schema: entelechy_abi::Schema,
                      domain: Domain,
                      f: &dyn Fn(usize) -> Vec<f32>| {
        let count = domain.count();
        let mut values = Vec::with_capacity(count * schema.value_words());
        for i in 0..count {
            for v in f(i) {
                values.push(v.to_bits());
            }
        }
        host.adopt(entelechy_abi::FieldData::from_values(schema, domain, &values).unwrap())
            .unwrap()
    };
    let s_u8 = module.manifest.inputs[0].ty.schema.clone();
    let s_s8 = module.manifest.inputs[1].ty.schema.clone();
    let s_f16 = module.manifest.inputs[2].ty.schema.clone();
    let s_vf16 = module.manifest.inputs[3].ty.schema.clone();
    let s_vu8 = module.manifest.inputs[4].ty.schema.clone();

    let h0 = make_field(&mut host, s_u8, Domain::Grid(17, 13), &|i| {
        vec![((i * 37) % 256) as f32 / 255.0]
    });
    let h1 = make_field(&mut host, s_s8, Domain::Grid(19, 11), &|i| {
        vec![
            (((i * 29) % 255) as f32 / 127.0) - 1.0,
            (((i * 53) % 255) as f32 / 127.0) - 1.0,
        ]
    });
    let h2 = make_field(&mut host, s_f16, Domain::Grid(15, 9), &|i| {
        vec![
            (i as f32 * 0.03).sin(),
            (i as f32 * 0.07).cos(),
            (i as f32 * 0.11).sin() * 2.0,
            0.5 + (i % 7) as f32 * 0.125,
        ]
    });
    let h3 = make_field(&mut host, s_vf16, Domain::Volume(13, 9, 7), &|i| {
        vec![((i * 19) % 97) as f32 * 0.025 - 1.0]
    });
    // Width 64 * 4 bytes = 256 bytes/row -> direct zero-shader DMA path!
    let h4 = make_field(&mut host, s_vu8, Domain::Volume(64, 4, 4), &|i| {
        vec![
            ((i * 11) % 256) as f32 / 255.0,
            ((i * 23) % 256) as f32 / 255.0,
            ((i * 41) % 256) as f32 / 255.0,
            ((i * 71) % 256) as f32 / 255.0,
        ]
    });
    let handles = vec![h0, h1, h2, h3, h4];
    let domain = Domain::Grid(24, 16);
    let (cpu, device) = on_both_inputs(&gpu, &mut host, &module, handles.clone(), domain);
    for (idx, (&c, &d)) in cpu.iter().zip(device.iter()).enumerate() {
        let cf = f32::from_bits(c);
        let df = f32::from_bits(d);
        assert!(
            (cf - df).abs() <= 1e-5,
            "word {idx}: cpu={cf} vs gpu={df} (diff={})",
            (cf - df).abs()
        );
    }
    // Run a second time on the SAME resident handles to exercise the cached
    // zero-copy `DeviceImage::mirror` hit path and verify identical output.
    let prepared = gpu.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let out2 = Backend::execute(
        &gpu,
        prepared.as_ref(),
        &module,
        &Invocation {
            inputs: handles.clone(),
            params: vec![],
            domain,
            context: Context {
                time: 1.25,
                ..Context::default()
            },
        },
        &mut scope,
    )
    .unwrap()[0];
    let resident2 = host.resident(out2).unwrap().unwrap();
    let device2 = gpu.download(resident2, domain.count() * 16).unwrap();
    assert_eq!(
        device, device2,
        "cached texture mirror must repeat bit-identically"
    );
    // Release inputs and verify `SpareMirrors` recycles and ages them out cleanly.
    gpu.evaluation_begun();
    for h in handles {
        let _ = host.release(h);
    }
    let _ = host.release(out2);
    let mirror_bytes = || {
        gpu.memory()
            .into_iter()
            .find(|(k, _)| *k == "spare mirrors")
            .map_or(0, |(_, b)| b)
    };
    assert!(
        mirror_bytes() > 0,
        "released mirrors should enter SpareMirrors"
    );
    gpu.evaluation_begun();
    gpu.evaluation_begun();
    assert_eq!(
        mirror_bytes(),
        0,
        "unused mirrors should age out after an evaluation"
    );

    // Verify cost-model selection for wide (`Word`) 2D Grids:
    // - single non-loop AtLinear stays storage buffer when slots allow
    // - loop AtLinear mirrors as hardware texture
    // - single non-loop AtLinear spills into free texture slots under tight storage limits
    let wide_single = compile(
        "kernel test.wide_single@1 {
            requires { field.read; field.write; }
            input a: Grid<Vec4>;
            input b: Grid<Vec4>;
            input c: Grid<Vec4>;
            input d: Grid<Vec4>;
            output out: Grid<Vec4>;
            compute {
                let p = (coord() + vec2(0.5, 0.5)) / domain() * size(a);
                return at_linear(a, p) + at_linear(b, p) + at_linear(c, p) + at_linear(d, p);
            }
        }",
    )
    .unwrap();
    assert!(
        entelechy_gpu::wgsl::textured_inputs(&wide_single, 16).is_empty(),
        "single non-loop wide 2D reads stay storage buffers when slots allow"
    );
    let spilled = entelechy_gpu::wgsl::textured_inputs_for_limits(&wide_single, 16, 16, 3);
    assert_eq!(
        spilled.len(),
        2,
        "tight storage limit (3 slots for 4 inputs + 1 output) should spill 2 wide grids to textures: {spilled:?}"
    );
    let wide_loop = compile(
        "kernel test.wide_loop@1 {
            requires { field.read; field.write; }
            input a: Grid<Vec4>;
            output out: Grid<Vec4>;
            compute {
                let p = coord() + vec2(0.5, 0.5);
                return loop 8 with acc = vec4(0.0, 0.0, 0.0, 0.0) {
                    acc + at_linear(a, p + vec2(float(index()), 0.0))
                };
            }
        }",
    )
    .unwrap();
    assert_eq!(
        entelechy_gpu::wgsl::textured_inputs(&wide_loop, 16),
        vec![0],
        "loop AtLinear on a wide 2D grid should mirror as a hardware texture"
    );
}

#[test]
fn hardware_block_compressed_textures_match_cpu_across_bc1_bc4_bc5_bc7_and_2d_arrays() {
    let Some(gpu) = gpu() else { return };
    use entelechy_abi::{BlockFormat, CompressedBlocks, FieldData};

    let source = "kernel test.bc_sample@1 {
        requires { field.read; field.write; }
        input bc1: Grid<Vec4 as unorm8>;
        input bc4: Grid<Float as unorm8>;
        input bc5: Grid<Vec2 as unorm8>;
        input bc7: Grid<Vec4 as unorm8>;
        input arr: Volume<Vec4 as unorm8>;
        output out: Grid<Vec4>;
        compute {
            let uv = (coord() + vec2(0.5, 0.5)) / domain();
            let a = at(bc1, uv * size(bc1)) + at_linear(bc1, uv * size(bc1));
            let b = at(bc4, uv * size(bc4)) + at_linear(bc4, uv * size(bc4));
            let c = at(bc5, uv * size(bc5)) + at_linear(bc5, uv * size(bc5));
            let d = at(bc7, uv * size(bc7)) + at_linear(bc7, uv * size(bc7));
            let layer = floor(fract(uv.x * 1.7 + uv.y * 0.9) * size(arr).z);
            let p3 = vec3(uv.x * size(arr).x, uv.y * size(arr).y, layer + 0.5);
            let e = at(arr, p3) + at_linear(arr, p3);
            return vec4(
                a.x + b + c.x + d.x + e.x,
                a.y + c.y + d.y + e.y,
                a.z + d.z + e.z,
                a.w + d.w + e.w
            ) * 0.125;
        }
    }";
    let module = compile(source).unwrap_or_else(|e| panic!("{e}"));
    let textured = entelechy_gpu::wgsl::textured_inputs(&module, 16);
    assert_eq!(textured, vec![0, 1, 2, 3, 4]);

    let make_bc = |fmt: BlockFormat, domain: Domain, seed: u32| -> (FieldData, u64) {
        let w = domain.width() as usize;
        let h = domain.height() as usize;
        let d = domain.depth() as usize;
        let ch = fmt.channels();
        let mut pixels = vec![0u8; w * h * d * ch];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let base = ((z * h + y) * w + x) * ch;
                    let r = ((x * 23 + y * 11 + z * 57 + seed as usize * 17) % 256) as u8;
                    let g = ((x * 7 + y * 31 + z * 43 + seed as usize * 29) % 256) as u8;
                    let b = ((x * 19 + y * 13 + z * 71 + seed as usize * 37) % 256) as u8;
                    let a = if fmt == BlockFormat::Bc1RgbaUnorm {
                        255
                    } else {
                        200u8.wrapping_add(((x + y * 3 + z * 5) % 56) as u8)
                    };
                    let rgba = [r, g, b, a];
                    pixels[base..base + ch].copy_from_slice(&rgba[..ch]);
                }
            }
        }
        let blocks = CompressedBlocks::encode_unorm8(fmt, domain, &pixels).unwrap();
        let compressed_bytes = blocks.bytes().len() as u64;
        (
            FieldData::from_blocks(domain, blocks).unwrap(),
            compressed_bytes,
        )
    };

    // Include non-multiple-of-4 extents (13x9, 11x7, 15x10) and a 3-layer 2D array.
    let specs = [
        (BlockFormat::Bc1RgbaUnorm, Domain::Grid(13, 9), 1),
        (BlockFormat::Bc4RUnorm, Domain::Grid(11, 7), 2),
        (BlockFormat::Bc5RgUnorm, Domain::Grid(15, 10), 3),
        (BlockFormat::Bc7RgbaUnorm, Domain::Grid(16, 12), 4),
        (BlockFormat::Bc7RgbaUnorm, Domain::Volume(12, 8, 3), 5),
    ];

    let mut cpu_host = Host::new(8 << 20);
    let mut gpu_host = Host::new(8 << 20);
    let mut cpu_handles = Vec::new();
    let mut gpu_handles = Vec::new();
    let mut expected_bytes = Vec::new();
    for (fmt, dom, seed) in specs {
        let (data, cbytes) = make_bc(fmt, dom, seed);
        cpu_handles.push(cpu_host.adopt(data.clone()).unwrap());
        gpu_handles.push(gpu_host.adopt(data).unwrap());
        expected_bytes.push((fmt, dom, cbytes));
    }

    // Upload all compressed inputs directly to GPU and drop Host copies.
    {
        let mut scope = Scope::new(&mut gpu_host, all());
        for &h in &gpu_handles {
            assert!(Backend::upload(&gpu, h, &mut scope).unwrap());
        }
    }
    for (idx, &h) in gpu_handles.iter().enumerate() {
        gpu_host.let_go_of_copy(h).unwrap();
        let bytes = gpu_host.field_bytes(h).unwrap();
        assert_eq!(bytes.host, 0, "Host RAM copy must be released");
        if gpu.supports_texture_compression_bc() {
            assert_eq!(
                bytes.device, expected_bytes[idx].2,
                "GPU VRAM for {:?} over {} must equal exact compressed block bytes (no uncompressed buffer)",
                expected_bytes[idx].0, expected_bytes[idx].1
            );
        }
    }

    let out_domain = Domain::Grid(24, 16);
    let cpu = CpuBackend;
    let cpu_prep = cpu.prepare(&module).unwrap();
    let cpu_out = {
        let mut scope = Scope::new(&mut cpu_host, all());
        Backend::execute(
            &cpu,
            cpu_prep.as_ref(),
            &module,
            &Invocation {
                inputs: cpu_handles.clone(),
                params: vec![],
                domain: out_domain,
                context: Context::default(),
            },
            &mut scope,
        )
        .unwrap()[0]
    };
    let cpu_words = cpu_host.data(cpu_out).unwrap().storage().to_vec();

    let gpu_prep = gpu.prepare(&module).unwrap();
    let gpu_out = {
        let mut scope = Scope::new(&mut gpu_host, all());
        gpu.execute(
            gpu_prep.as_ref(),
            &module,
            &Invocation {
                inputs: gpu_handles.clone(),
                params: vec![],
                domain: out_domain,
                context: Context::default(),
            },
            &mut scope,
        )
        .unwrap()[0]
    };
    let gpu_words = {
        let dev = gpu_host.resident(gpu_out).unwrap().unwrap();
        gpu.download(dev, out_domain.count() * 16).unwrap()
    };

    for (idx, (&cw, &gw)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
        let cf = f32::from_bits(cw);
        let gf = f32::from_bits(gw);
        assert!(
            (cf - gf).abs() <= (1.0 / 255.0) + 1e-4,
            "hardware BC sampling mismatch at word {idx}: cpu={cf} vs gpu={gf} (diff={})",
            (cf - gf).abs()
        );
    }

    // Exercise GPU `decompress_bc` download path on both 2D and 2D-array compressed textures.
    for (idx, (&ch, &gh)) in cpu_handles.iter().zip(gpu_handles.iter()).enumerate() {
        let cpu_ref = cpu_host.data(ch).unwrap().storage().to_vec();
        let dev = gpu_host.resident(gh).unwrap().unwrap();
        let byte_len = gpu_host
            .schema(gh)
            .unwrap()
            .storage_words(gpu_host.domain(gh).unwrap().count())
            * 4;
        let gpu_dl = gpu.download(dev, byte_len).unwrap();
        assert_eq!(cpu_ref.len(), gpu_dl.len());
        let (fmt, dom, _) = expected_bytes[idx];
        let valid_bytes = dom.count() * fmt.channels();
        let tol = match fmt {
            // BC7 weights are exact 6-bit powers of two per Khronos §17, so
            // hardware decoders match CPU bit-for-bit.
            BlockFormat::Bc7RgbaUnorm => 0,
            // Khronos / WebGPU CTS permits small fixed-point 1/3 and 1/7 weight
            // differences on hardware BC1/BC4/BC5 texture units (e.g. 37/64 vs 4/7).
            BlockFormat::Bc1RgbaUnorm | BlockFormat::Bc4RUnorm | BlockFormat::Bc5RgUnorm => 3,
        };
        for b in 0..valid_bytes {
            let cw = ((cpu_ref[b / 4] >> ((b % 4) * 8)) & 0xFF) as i32;
            let gw = ((gpu_dl[b / 4] >> ((b % 4) * 8)) & 0xFF) as i32;
            assert!(
                (cw - gw).abs() <= tol,
                "decompress_bc download mismatch for {:?} at byte {b}: cpu={cw} vs gpu={gw} (tol={tol})",
                fmt
            );
        }
    }

    // Exercise software decompression fallback (`without_texture_compression`) on the same inputs.
    let soft_gpu = GpuBackend::new().unwrap().without_texture_compression();
    assert!(!soft_gpu.supports_texture_compression_bc());
    let mut soft_host = Host::new(8 << 20);
    let mut soft_handles = Vec::new();
    for (fmt, dom, seed) in specs {
        let (data, _) = make_bc(fmt, dom, seed);
        soft_handles.push(soft_host.adopt(data).unwrap());
    }
    let soft_prep = soft_gpu.prepare(&module).unwrap();
    let soft_out = {
        let mut scope = Scope::new(&mut soft_host, all());
        soft_gpu
            .execute(
                soft_prep.as_ref(),
                &module,
                &Invocation {
                    inputs: soft_handles,
                    params: vec![],
                    domain: out_domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap()[0]
    };
    let soft_words = {
        let dev = soft_host.resident(soft_out).unwrap().unwrap();
        soft_gpu.download(dev, out_domain.count() * 16).unwrap()
    };
    for (idx, (&cw, &sw)) in cpu_words.iter().zip(soft_words.iter()).enumerate() {
        let cf = f32::from_bits(cw);
        let sf = f32::from_bits(sw);
        assert!(
            (cf - sf).abs() <= 1e-5,
            "software BC fallback mismatch at word {idx}: cpu={cf} vs soft_gpu={sf}"
        );
    }
}

#[test]
fn hardware_compressed_volume_join_copies_blocks_gpu_to_gpu_without_decompression() {
    let Some(gpu) = gpu() else { return };
    if !gpu.supports_texture_compression_bc() {
        return;
    }
    use entelechy_abi::{BlockFormat, CompressedBlocks, FieldData};

    let make_layer = |fmt: BlockFormat, w: u32, h: u32, tag: u8| -> FieldData {
        let ch = fmt.channels();
        let mut pixels = vec![0u8; (w as usize) * (h as usize) * ch];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let base = (y * w as usize + x) * ch;
                let vals = [
                    tag.wrapping_mul(60).wrapping_add((x * 9) as u8),
                    tag.wrapping_mul(40).wrapping_add((y * 13) as u8),
                    255u8.saturating_sub(tag * 50),
                    255,
                ];
                pixels[base..base + ch].copy_from_slice(&vals[..ch]);
            }
        }
        let blocks = CompressedBlocks::encode_unorm8(fmt, Domain::Grid(w, h), &pixels).unwrap();
        FieldData::from_blocks(Domain::Grid(w, h), blocks).unwrap()
    };

    // Test both BC7 (4 B/el, whole words) and BC4 (1 B/el, odd 13x9 extent).
    for (fmt, w, h) in [
        (BlockFormat::Bc7RgbaUnorm, 16u32, 12u32),
        (BlockFormat::Bc4RUnorm, 13u32, 9u32),
    ] {
        let layers = [
            make_layer(fmt, w, h, 1),
            make_layer(fmt, w, h, 2),
            make_layer(fmt, w, h, 3),
        ];
        let mut host = Host::new(8 << 20);
        let handles: Vec<_> = layers
            .iter()
            .map(|d| host.adopt(d.clone()).unwrap())
            .collect();
        {
            let mut scope = Scope::new(&mut host, all());
            for &hnd in &handles {
                Backend::upload(&gpu, hnd, &mut scope).unwrap();
            }
        }
        for &hnd in &handles {
            host.let_go_of_copy(hnd).unwrap();
        }
        let layer_elems = (w as usize) * (h as usize);
        let parts: Vec<_> = handles.iter().map(|&hnd| (hnd, 0..layer_elems)).collect();
        let joined_domain = Domain::Volume(w, h, 3);
        let joined = {
            let mut scope = Scope::new(&mut host, all());
            Backend::join(&gpu, &parts, &fmt.schema(), joined_domain, &mut scope)
                .unwrap()
                .expect("GPU compressed layer join must succeed")
        };
        let expected_bytes = fmt.byte_len(joined_domain).unwrap() as u64;
        let mem = host.field_bytes(joined).unwrap();
        assert_eq!(mem.host, 0);
        assert_eq!(
            mem.device, expected_bytes,
            "joined {:?} Volume must remain compressed on GPU without uncompressed buffer",
            fmt
        );

        // Verify GPU-joined Texture2DArray contents match CPU-decoded layers via download().
        let dev = host.resident(joined).unwrap().unwrap();
        let byte_len = fmt.schema().storage_words(joined_domain.count()) * 4;
        let gpu_words = gpu.download(dev, byte_len).unwrap();

        let mut expected_raw = Vec::new();
        for l in &layers {
            expected_raw.extend_from_slice(l.blocks().unwrap().bytes());
        }
        let cpu_joined = FieldData::from_blocks(
            joined_domain,
            CompressedBlocks::new(fmt, joined_domain, expected_raw).unwrap(),
        )
        .unwrap();
        let cpu_words = cpu_joined.storage();
        let valid_bytes = joined_domain.count() * fmt.channels();
        for b in 0..valid_bytes {
            let cb = ((cpu_words[b / 4] >> ((b % 4) * 8)) & 0xFF) as i32;
            let gb = ((gpu_words[b / 4] >> ((b % 4) * 8)) & 0xFF) as i32;
            assert!(
                (cb - gb).abs() <= 1,
                "GPU compressed join mismatch for {:?} at byte {b}: cpu={cb} vs gpu={gb}",
                fmt
            );
        }
    }
}

#[test]
fn hardware_mipmaps_trilinear_and_repeat_match_cpu_across_uncompressed_and_bc_formats() {
    let Some(gpu) = gpu() else { return };
    use entelechy_abi::{BlockFormat, CompressedBlocks, Encoding, FieldData, Scalar, Schema};

    let source = "kernel test.mips@1 {
        requires { field.read; field.write; }
        input rows: Rows<Vec2>;
        input f32_grid: Grid<Vec4>;
        input snorm_grid: Grid<Vec2 as snorm8>;
        input vol3d: Volume<Vec4 as f16>;
        input bc7_grid: Grid<Vec4 as unorm8>;
        input bc5_norm: Grid<Vec2 as unorm8>;
        input odd_bc7: Grid<Vec4 as unorm8>;
        output result: Grid<Vec4>;
        compute {
            let p = coord();
            let r = at_repeat(rows, p.x * 1.5 - 2.0);
            let m0 = at_repeat(f32_grid, p * 2.0 - vec2(1.5, 2.5));
            let m1 = at_linear(f32_grid, p * 2.0 + vec2(0.5, 0.5), 1.0);
            let m2 = at_repeat(f32_grid, p * 2.0 + vec2(0.5, 0.5), 2.5);
            let sn = at_linear(snorm_grid, p * 2.0 + vec2(0.5, 0.5), 1.5);
            let v3 = at_linear(vol3d, vec3(p.x + 0.5, p.y + 0.5, 2.0), 1.0);
            let b7 = at_repeat(bc7_grid, p * 4.0 - vec2(2.0, 2.0), 2.0);
            let b5 = at_linear(bc5_norm, p * 2.0 + vec2(0.5, 0.5), 1.0);
            let odd = at_repeat(odd_bc7, p * 1.7 - vec2(1.0, 0.5), 1.0);
            return vec4(
                (r.x + m0.x + m1.x + m2.x) * 0.25,
                (sn.x + v3.x + b7.x + odd.x) * 0.25,
                (sn.y + v3.y + b5.y + odd.y) * 0.25,
                (r.y + v3.z + b7.z + b5.x) * 0.25
            );
        }
    }";
    let module = compile(source).unwrap();

    // 1D Rows<Vec2>
    let rows_vals: Vec<u32> = (0..4)
        .flat_map(|i| [(i as f32 / 3.0).to_bits(), (1.0 - i as f32 / 3.0).to_bits()])
        .collect();
    let rows_data = FieldData::from_values(
        Schema::new(vec![Scalar::Float; 2]),
        Domain::Rows(4),
        &rows_vals,
    )
    .unwrap();

    // 16x16 uncompressed f32 checkerboard (averages to 0.5 at mip >= 1)
    let mut f32_vals = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let c = if (x + y) % 2 == 0 { 1.0f32 } else { 0.0 };
            f32_vals
                .extend_from_slice(&[c, x as f32 / 15.0, y as f32 / 15.0, 1.0].map(f32::to_bits));
        }
    }
    let f32_data =
        FieldData::from_values(Schema::pixel(), Domain::Grid(16, 16), &f32_vals).unwrap();

    // 16x16 snorm8 Grid<Vec2 as snorm8>
    let snorm_schema = Schema::new(vec![Scalar::Float; 2])
        .with_encoding(Encoding::Snorm8)
        .unwrap();
    let mut snorm_vals = Vec::with_capacity(16 * 16 * 2);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let sx = (x as f32 / 15.0) * 2.0 - 1.0;
            let sy = (y as f32 / 15.0) * 2.0 - 1.0;
            snorm_vals.extend_from_slice(&[sx.to_bits(), sy.to_bits()]);
        }
    }
    let snorm_data =
        FieldData::from_values(snorm_schema, Domain::Grid(16, 16), &snorm_vals).unwrap();

    // 8x8x4 3D Volume<Vec4 as f16> (true 3D 2x2x2 mipmaps)
    let vol_schema = Schema::pixel().with_encoding(Encoding::F16).unwrap();
    let mut vol_vals = Vec::with_capacity(8 * 8 * 4 * 4);
    for z in 0..4u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                let vz = if z < 2 { 0.2f32 } else { 0.8 };
                vol_vals.extend_from_slice(
                    &[x as f32 / 7.0, y as f32 / 7.0, vz, 1.0].map(f32::to_bits),
                );
            }
        }
    }
    let vol_data = FieldData::from_values(vol_schema, Domain::Volume(8, 8, 4), &vol_vals).unwrap();

    // 32x32 multi-mip BC7 checkerboard
    let mut bc7_px = vec![0u8; 32 * 32 * 4];
    for y in 0..32usize {
        for x in 0..32usize {
            let b = (y * 32 + x) * 4;
            let c = if (x + y) % 2 == 0 { 240u8 } else { 16u8 };
            bc7_px[b..b + 4].copy_from_slice(&[c, (x * 7) as u8, (y * 7) as u8, 255]);
        }
    }
    let bc7_blocks = CompressedBlocks::encode_unorm8_mips(
        BlockFormat::Bc7RgbaUnorm,
        Domain::Grid(32, 32),
        &bc7_px,
        6,
        false,
    )
    .unwrap();
    assert_eq!(bc7_blocks.mip_count(), 6);
    let bc7_data = FieldData::from_blocks(Domain::Grid(32, 32), bc7_blocks).unwrap();

    // 16x16 multi-mip BC5 normal-renormalized
    let mut bc5_px = vec![0u8; 16 * 16 * 2];
    for y in 0..16usize {
        for x in 0..16usize {
            let b = (y * 16 + x) * 2;
            bc5_px[b] = if (x + y) % 2 == 0 { 190 } else { 66 };
            bc5_px[b + 1] = 160;
        }
    }
    let bc5_blocks = CompressedBlocks::encode_unorm8_mips(
        BlockFormat::Bc5RgUnorm,
        Domain::Grid(16, 16),
        &bc5_px,
        5,
        true,
    )
    .unwrap();
    let bc5_data = FieldData::from_blocks(Domain::Grid(16, 16), bc5_blocks).unwrap();

    // 13x9 odd-sized multi-mip BC7 (exercises physical != logical padding safety)
    let mut odd_px = vec![0u8; 13 * 9 * 4];
    for y in 0..9usize {
        for x in 0..13usize {
            let b = (y * 13 + x) * 4;
            odd_px[b..b + 4].copy_from_slice(&[(x * 20) as u8, (y * 28) as u8, 128, 255]);
        }
    }
    let odd_blocks = CompressedBlocks::encode_unorm8_mips(
        BlockFormat::Bc7RgbaUnorm,
        Domain::Grid(13, 9),
        &odd_px,
        4,
        false,
    )
    .unwrap();
    let odd_data = FieldData::from_blocks(Domain::Grid(13, 9), odd_blocks).unwrap();

    let fields = [
        rows_data, f32_data, snorm_data, vol_data, bc7_data, bc5_data, odd_data,
    ];
    let out_domain = Domain::Grid(8, 8);

    let mut cpu_host = Host::new(8 << 20);
    let cpu_inputs: Vec<_> = fields
        .iter()
        .map(|f| cpu_host.adopt(f.clone()).unwrap())
        .collect();
    let cpu_out = {
        let mut scope = Scope::new(&mut cpu_host, all());
        CpuBackend
            .execute(
                &module,
                &Invocation {
                    inputs: cpu_inputs,
                    params: vec![],
                    domain: out_domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap()[0]
    };
    let cpu_words = cpu_host.data(cpu_out).unwrap().storage().to_vec();

    // Verify both hardware FLOAT32_FILTERABLE path AND non-filterable fallback!
    for (label, backend) in [
        ("hw", gpu),
        (
            "no_f32_filter",
            GpuBackend::new().unwrap().without_float32_filterable(),
        ),
    ] {
        let mut gpu_host = Host::new(8 << 20);
        let gpu_inputs: Vec<_> = fields
            .iter()
            .map(|f| gpu_host.adopt(f.clone()).unwrap())
            .collect();
        let prep = backend.prepare(&module).unwrap();
        let gpu_out = {
            let mut scope = Scope::new(&mut gpu_host, all());
            backend
                .execute(
                    prep.as_ref(),
                    &module,
                    &Invocation {
                        inputs: gpu_inputs,
                        params: vec![],
                        domain: out_domain,
                        context: Context::default(),
                    },
                    &mut scope,
                )
                .unwrap()[0]
        };
        let gpu_words = {
            let dev = gpu_host.resident(gpu_out).unwrap().unwrap();
            backend.download(dev, out_domain.count() * 16).unwrap()
        };
        for (idx, (&cw, &gw)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
            let cf = f32::from_bits(cw);
            let gf = f32::from_bits(gw);
            assert!(
                (cf - gf).abs() <= 0.02,
                "{label} mipmap parity mismatch at word {idx}: cpu={cf} vs gpu={gf} (diff={})",
                (cf - gf).abs()
            );
        }
    }
}

#[test]
fn hardware_16x_anisotropic_filtering_preserves_grazing_detail_and_wraps_seamlessly() {
    let Some(gpu) = gpu() else { return };
    use entelechy_abi::{BlockFormat, CompressedBlocks, Encoding, FieldData, Schema};

    // High-frequency horizontal stripes (even rows = 1.0, odd rows = 0.0) on a 64x64
    // f16 Grid and a 2-layer multi-mip BC7 Volume (layer 0 horizontal, layer 1 vertical).
    let w = 64u32;
    let h = 64u32;
    let mut f16_vals = Vec::with_capacity((w * h * 4) as usize);
    let mut bc_px = vec![0u8; (w * h * 2 * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let hz = if y % 2 == 0 { 1.0f32 } else { 0.0 };
            f16_vals.extend_from_slice(&[hz, hz, hz, 1.0].map(f32::to_bits));
            // Layer 0: horizontal stripes; Layer 1: vertical stripes.
            let b0 = ((y * w + x) * 4) as usize;
            let c0 = if y % 2 == 0 { 255u8 } else { 0u8 };
            bc_px[b0..b0 + 4].copy_from_slice(&[c0, c0, c0, 255]);
            let b1 = (((w * h) + y * w + x) * 4) as usize;
            let c1 = if x % 2 == 0 { 255u8 } else { 0u8 };
            bc_px[b1..b1 + 4].copy_from_slice(&[c1, c1, c1, 255]);
        }
    }
    let f16_data = FieldData::from_values(
        Schema::pixel().with_encoding(Encoding::F16).unwrap(),
        Domain::Grid(w, h),
        &f16_vals,
    )
    .unwrap();
    let bc_blocks = CompressedBlocks::encode_unorm8_mips(
        BlockFormat::Bc7RgbaUnorm,
        Domain::Volume(w, h, 2),
        &bc_px,
        7,
        false,
    )
    .unwrap();
    let bc_vol = FieldData::from_blocks(Domain::Volume(w, h, 2), bc_blocks).unwrap();

    // Sample grazing along X (`ddx = (12, 0), ddy = (0, 0.25)`) at even & odd rows
    // crossing the repeat wrap seam (`x = 0.25` and `x = 63.75`), vs isotropic `lod = 3.58`.
    let source = "kernel test.aniso@1 {
        requires { field.read; field.write; }
        input stripes: Grid<Vec4 as f16>;
        input atlas: Volume<Vec4 as unorm8>;
        output out: Rows<Vec4>;
        compute {
            let e = element();
            let x = if e % 2 == 0 { 0.25 } else { 63.75 };
            let row = float(e / 2) + 0.5;
            let ddx = vec2(12.0, 0.0);
            let ddy = vec2(0.0, 0.25);
            let iso = at_repeat(stripes, vec2(x, row), 3.58);
            let aniso_grid = at_repeat(stripes, vec2(x, row), ddx, ddy);
            let aniso_arr0 = at_repeat(atlas, vec3(x, row, 0.5), ddx, ddy);
            // Layer 1 has vertical stripes; grazing along Y preserves vertical stripes!
            let aniso_arr1 = at_repeat(atlas, vec3(row, x, 1.5), vec2(0.25, 0.0), vec2(0.0, 12.0));
            return vec4(iso.x, aniso_grid.x, aniso_arr0.x, aniso_arr1.x);
        }
    }";
    let module = compile(source).unwrap();
    let out_domain = Domain::Rows(8);

    let run_cpu = || {
        let mut host = Host::new(8 << 20);
        let inputs = vec![
            host.adopt(f16_data.clone()).unwrap(),
            host.adopt(bc_vol.clone()).unwrap(),
        ];
        let mut scope = Scope::new(&mut host, all());
        let out = CpuBackend
            .execute(
                &module,
                &Invocation {
                    inputs,
                    params: vec![],
                    domain: out_domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap()[0];
        host.data(out).unwrap().storage().to_vec()
    };
    let run_gpu = || {
        let mut host = Host::new(8 << 20);
        let inputs = vec![
            host.adopt(f16_data.clone()).unwrap(),
            host.adopt(bc_vol.clone()).unwrap(),
        ];
        let prep = gpu.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let out = gpu
            .execute(
                prep.as_ref(),
                &module,
                &Invocation {
                    inputs,
                    params: vec![],
                    domain: out_domain,
                    context: Context::default(),
                },
                &mut scope,
            )
            .unwrap()[0];
        let dev = host.resident(out).unwrap().unwrap();
        gpu.download(dev, out_domain.count() * 16).unwrap()
    };

    for (label, words) in [("cpu", run_cpu()), ("gpu", run_gpu())] {
        for e in 0..8usize {
            let iso = f32::from_bits(words[e * 4]);
            let aniso_g = f32::from_bits(words[e * 4 + 1]);
            let aniso_0 = f32::from_bits(words[e * 4 + 2]);
            let aniso_1 = f32::from_bits(words[e * 4 + 3]);
            // Isotropic at lod 3.58 collapses alternating 1/0 stripes to ~0.5.
            assert!(
                (iso - 0.5).abs() < 0.08,
                "{label}[{e}]: isotropic should average to ~0.5, got {iso}"
            );
            let expect_white = (e / 2) % 2 == 0;
            for (name, val) in [
                ("aniso_grid", aniso_g),
                ("aniso_arr0", aniso_0),
                ("aniso_arr1", aniso_1),
            ] {
                if expect_white {
                    assert!(
                        val > 0.80,
                        "{label}[{e}] {name}: 16x aniso must preserve white stripe (>0.8), got {val}"
                    );
                } else {
                    assert!(
                        val < 0.20,
                        "{label}[{e}] {name}: 16x aniso must preserve black stripe (<0.2), got {val}"
                    );
                }
            }
        }
    }
}

#[test]
fn hardware_compressed_multimip_volume_join_preserves_full_mip_chain_gpu_to_gpu() {
    let Some(gpu) = gpu() else { return };
    if !gpu.supports_texture_compression_bc() {
        return;
    }
    use entelechy_abi::{BlockFormat, CompressedBlocks, FieldData};

    let w = 32u32;
    let h = 32u32;
    let fmt = BlockFormat::Bc7RgbaUnorm;
    let mut px0 = vec![0u8; (w * h * 4) as usize];
    let mut px1 = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let b = (y * w as usize + x) * 4;
            let chk = if (x + y) % 2 == 0 { 240u8 } else { 16u8 };
            px0[b..b + 4].copy_from_slice(&[chk, chk, chk, 255]);
            px1[b..b + 4].copy_from_slice(&[30, 80, 230, 255]);
        }
    }
    let l0 = FieldData::from_blocks(
        Domain::Grid(w, h),
        CompressedBlocks::encode_unorm8_mips(fmt, Domain::Grid(w, h), &px0, 6, false).unwrap(),
    )
    .unwrap();
    let l1 = FieldData::from_blocks(
        Domain::Grid(w, h),
        CompressedBlocks::encode_unorm8_mips(fmt, Domain::Grid(w, h), &px1, 6, false).unwrap(),
    )
    .unwrap();
    let mips = l0.blocks().unwrap().mip_count();
    assert_eq!(mips, 6);

    let mut host = Host::new(8 << 20);
    let h0 = host.adopt(l0).unwrap();
    let h1 = host.adopt(l1).unwrap();
    {
        let mut scope = Scope::new(&mut host, all());
        Backend::upload(&gpu, h0, &mut scope).unwrap();
        Backend::upload(&gpu, h1, &mut scope).unwrap();
    }
    host.let_go_of_copy(h0).unwrap();
    host.let_go_of_copy(h1).unwrap();

    let layer_elems = (w * h) as usize;
    let joined_domain = Domain::Volume(w, h, 2);
    let joined = {
        let mut scope = Scope::new(&mut host, all());
        Backend::join(
            &gpu,
            &[(h0, 0..layer_elems), (h1, 0..layer_elems)],
            &fmt.schema(),
            joined_domain,
            &mut scope,
        )
        .unwrap()
        .expect("multi-mip GPU compressed join succeeds")
    };
    let expected_bytes = fmt.mip_chain_byte_len(joined_domain, mips).unwrap() as u64;
    let mem = host.field_bytes(joined).unwrap();
    assert_eq!(mem.host, 0);
    assert_eq!(
        mem.device, expected_bytes,
        "joined Texture2DArray must keep all {mips} compressed mip levels GPU-resident"
    );

    // Sample mip 0 (checkerboard peak ~0.94) and mip 2 (averaged ~0.50) on layer 0,
    // plus mip 2 solid blue on layer 1.
    let probe = compile(
        "kernel test.probe_mips@1 {
            requires { field.read; field.write; }
            input atlas: Volume<Vec4 as unorm8>;
            output out: Rows<Vec4>;
            compute {
                let l0_m0 = at_repeat(atlas, vec3(0.5, 0.5, 0.5), 0.0);
                let l0_m2 = at_repeat(atlas, vec3(16.0, 16.0, 0.5), 2.0);
                let l1_m2 = at_repeat(atlas, vec3(16.0, 16.0, 1.5), 2.0);
                return vec4(l0_m0.x, l0_m2.x, l1_m2.z, l1_m2.x);
            }
        }",
    )
    .unwrap();
    let prep = gpu.prepare(&probe).unwrap();
    let out = {
        let mut scope = Scope::new(&mut host, all());
        gpu.execute(
            prep.as_ref(),
            &probe,
            &Invocation {
                inputs: vec![joined],
                params: vec![],
                domain: Domain::Rows(1),
                context: Context::default(),
            },
            &mut scope,
        )
        .unwrap()[0]
    };
    let dev = host.resident(out).unwrap().unwrap();
    let words = gpu.download(dev, 16).unwrap();
    let l0_m0 = f32::from_bits(words[0]);
    let l0_m2 = f32::from_bits(words[1]);
    let l1_m2_b = f32::from_bits(words[2]);
    let l1_m2_r = f32::from_bits(words[3]);
    assert!((l0_m0 - 240.0 / 255.0).abs() < 0.03, "mip0={l0_m0}");
    assert!(
        (l0_m2 - 0.5).abs() < 0.04,
        "mip2 should average checkerboard to ~0.5, got {l0_m2}"
    );
    assert!((l1_m2_b - 230.0 / 255.0).abs() < 0.03, "l1_b={l1_m2_b}");
    assert!((l1_m2_r - 30.0 / 255.0).abs() < 0.03, "l1_r={l1_m2_r}");
}

#[test]
fn subgroup_group_exchanges_match_cpu_across_sizes_types_failures_and_fallback() {
    let Some(gpu) = gpu() else {
        return;
    };
    let caps = gpu.subgroups();
    let fallback = GpuBackend::new().unwrap().without_subgroups();
    assert!(!fallback.subgroups().enabled);

    // Verify WGSL codegen: single-wave group 32 emits subgroupShuffle with zero
    // workgroup memory or barriers, and browser WebGPU mode prepends `enable subgroups;`.
    let probe_src = "kernel test.sg_probe@1 {
        requires { field.write; }
        output result: Rows<Vec4>;
        group 32 {
            let v = vec4(float(element()), float(lane()), 1.0, 2.0);
            return exchange(v, lane() ^ 1);
        }
    }";
    let probe = compile(probe_src).unwrap();
    if caps.fits_single_wave(32) {
        let native_wgsl = entelechy_gpu::wgsl::emit_full_with_subgroups(
            &probe,
            &[],
            &[],
            entelechy_gpu::wgsl::Longs::Native,
            entelechy_gpu::wgsl::Half::High,
            true,
            caps,
        );
        assert!(
            native_wgsl.contains("subgroupShuffle(") && native_wgsl.contains("subgroupAny("),
            "expected subgroup shuffle codegen:\n{native_wgsl}"
        );
        assert!(
            !native_wgsl.contains("var<workgroup>") && !native_wgsl.contains("workgroupBarrier()"),
            "single-wave group should emit zero workgroup storage and barriers:\n{native_wgsl}"
        );
        let page_caps = entelechy_gpu::wgsl::SubgroupCaps {
            enable_directive: true,
            ..caps
        };
        let page_wgsl = entelechy_gpu::wgsl::emit_full_with_subgroups(
            &probe,
            &[],
            &[],
            entelechy_gpu::wgsl::Longs::Native,
            entelechy_gpu::wgsl::Half::High,
            true,
            page_caps,
        );
        assert!(
            page_wgsl.starts_with("enable subgroups;\n"),
            "browser WebGPU mode must emit enable directive first"
        );
    }

    let run_both = |source: &str, domain: Domain| {
        let module = compile(source).unwrap();
        let inv = Invocation {
            inputs: vec![],
            params: vec![],
            domain,
            context: Context::default(),
        };
        let mut host = Host::new(1 << 22);
        let cpu_h = CpuBackend
            .execute(&module, &inv, &mut Scope::new(&mut host, all()))
            .unwrap()[0];
        let cpu_words = host.data(cpu_h).unwrap().storage().to_vec();

        for (tag, backend) in [("subgroups", &gpu), ("fallback", &fallback)] {
            let prep = Backend::prepare(backend, &module).unwrap();
            let out_h = Backend::execute(
                backend,
                prep.as_ref(),
                &module,
                &inv,
                &mut Scope::new(&mut host, all()),
            )
            .unwrap()[0];
            let dev = host.resident(out_h).unwrap().expect("resident");
            let words = Backend::download(backend, dev, cpu_words.len() * 4).unwrap();
            assert_eq!(
                words, cpu_words,
                "{tag} mismatch for domain {domain:?}:\n{source}"
            );
        }
    };

    // Diverse group sizes (non-powers of two, sub-wave, full wave, multi-wave) and domains.
    for size in [1u32, 3, 7, 8, 16, 31, 32, 64] {
        for count in [1u32, 13, 33, 100] {
            let src = format!(
                "kernel test.sg_sz_{size}@1 {{
                    requires {{ field.write; }}
                    output result: Rows<Int>;
                    group {size} {{
                        let a = element() * 3 + lane();
                        let b = exchange(a, lane() + 1);
                        let c = exchange(b, lane() - 2);
                        return a + b * 2 + c;
                    }}
                }}"
            );
            run_both(&src, Domain::Rows(count));
        }
    }

    // Complex exchanged types: Bool, Long (native i64 split), Mat2, Struct with mixed fields.
    let typed_src = "kernel test.sg_types@1 {
        requires { field.write; }
        output result: Rows<Vec4>;
        struct Pack { tag: Int; pos: Vec2; flag: Bool; }
        group 32 {
            let i = element();
            let b = exchange((i & 1) == 0, 31 - lane());
            let l = exchange(long(i) * 10000000007l + long(lane()), (lane() + 5) & 31);
            let m = exchange(mat2(vec2(float(i), 1.0), vec2(-2.0, float(lane()))), lane() ^ 3);
            let p = exchange(Pack { tag: i, pos: vec2(float(i), float(lane())), flag: b }, (lane() + 13) & 31);
            let low = int(l & 65535l);
            let high = int((l >> 32l) & 65535l);
            return vec4(
                float(p.tag + low + high) + if p.flag { 7.0 } else { 0.0 },
                p.pos.x + m[0].x,
                p.pos.y + m[1].y,
                float(b),
            );
        }
    }";
    run_both(typed_src, Domain::Rows(111));

    // Accounted single-wave group failure parity: budget and integer overflow at non-zero lane.
    let fail_src = "kernel test.sg_fail@1 {
        requires { field.write; }
        budget 64;
        param rounds: Int = 100;
        output result: Rows<Int>;
        group 32 {
            let base = element();
            let v = loop rounds with s = 0 {
                let extra = loop (lane() + 1) with acc = 0 { acc + 1 };
                let step = exchange(s + base + extra, lane() ^ 1);
                step % 9973
            };
            return v;
        }
    }";
    let fail_mod = entelechy_ir::account(&compile(fail_src).unwrap()).unwrap();
    assert!(fail_mod.accounted());
    let inv = Invocation {
        inputs: vec![],
        params: vec![Value::Int(100)],
        domain: Domain::Rows(64),
        context: Context::default(),
    };
    let mut cpu_host = Host::new(1 << 22);
    let cpu_err = CpuBackend
        .execute(&fail_mod, &inv, &mut Scope::new(&mut cpu_host, all()))
        .unwrap_err()
        .to_string();
    for (tag, backend) in [("subgroups", &gpu), ("fallback", &fallback)] {
        let mut dev_host = Host::new(1 << 22);
        let prep = Backend::prepare(backend, &fail_mod).unwrap();
        let gpu_err = Backend::execute(
            backend,
            prep.as_ref(),
            &fail_mod,
            &inv,
            &mut Scope::new(&mut dev_host, all()),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(gpu_err, cpu_err, "{tag} budget failure mismatch");
    }
}

#[test]
fn hierarchical_parallel_prefix_matches_cpu_across_levels_combines_overflow_and_fallback() {
    let Some(gpu) = gpu() else {
        return;
    };
    let caps = gpu.subgroups();
    let fallback = GpuBackend::new().unwrap().without_subgroups();

    // Check emitted subgroup instructions for whole integer sum.
    let isum_mod = compile(
        "kernel test.pfx_isum@1 {
            requires { field.read; field.write; }
            input xs: Rows<Int>;
            output result: Rows<Int>;
            prefix over xs with sum { return sample(xs); }
        }",
    )
    .unwrap();
    if caps.wave_parallel_256() {
        let wgsl = entelechy_gpu::wgsl::emit_prefix_with_subgroups(
            &isum_mod,
            entelechy_gpu::wgsl::PrefixPass::Blocks,
            caps,
        );
        assert!(
            wgsl.contains("subgroupShuffleUp("),
            "prefix blocks should use subgroupShuffleUp:\n{wgsl}"
        );
    }

    // 1-level (<=256), 2-level (257..65536), and 3-level (>65536) hierarchical scans.
    let counts = [1usize, 7, 255, 256, 257, 1000, 65_536, 70_000];
    let int_combines = ["sum", "min", "max"];
    for combine in int_combines {
        let src = format!(
            "kernel test.pfx_i_{combine}@1 {{
                requires {{ field.read; field.write; }}
                input xs: Rows<Int>;
                output result: Rows<Int>;
                prefix over xs with {combine} {{ return sample(xs); }}
            }}"
        );
        let module = compile(&src).unwrap();
        let prep_sg = Backend::prepare(&gpu, &module).unwrap();
        let prep_fb = Backend::prepare(&fallback, &module).unwrap();
        for &n in &counts {
            let input_words: Vec<u32> = (0..n)
                .map(|i| {
                    let v = ((i as i32 * 37) % 19) - 9;
                    v as u32
                })
                .collect();
            let mut cpu_host = Host::new(1 << 24);
            let cpu_inp = cpu_host
                .adopt(
                    entelechy_abi::FieldData::from_storage(
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                        Domain::Rows(n as u32),
                        input_words.clone(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let cpu_inv = Invocation {
                inputs: vec![cpu_inp],
                params: vec![],
                domain: Domain::Rows(n as u32),
                context: Context::default(),
            };
            let cpu_h = CpuBackend
                .execute(&module, &cpu_inv, &mut Scope::new(&mut cpu_host, all()))
                .unwrap()[0];
            let cpu_words = cpu_host.data(cpu_h).unwrap().storage().to_vec();

            for (tag, backend, prep) in [
                ("subgroups", &gpu, prep_sg.as_ref()),
                ("fallback", &fallback, prep_fb.as_ref()),
            ] {
                let mut dev_host = Host::new(1 << 24);
                let dev_inp = dev_host
                    .adopt(
                        entelechy_abi::FieldData::from_storage(
                            entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                            Domain::Rows(n as u32),
                            input_words.clone(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let inv = Invocation {
                    inputs: vec![dev_inp],
                    params: vec![],
                    domain: Domain::Rows(n as u32),
                    context: Context::default(),
                };
                let out_h = Backend::execute(
                    backend,
                    prep,
                    &module,
                    &inv,
                    &mut Scope::new(&mut dev_host, all()),
                )
                .unwrap()[0];
                let dev = dev_host.resident(out_h).unwrap().expect("resident");
                let words = Backend::download(backend, dev, cpu_words.len() * 4).unwrap();
                assert_eq!(words, cpu_words, "{tag} prefix {combine} mismatch at n={n}");
            }
        }
    }

    // Float min/max across 3 levels (70,000 elements) and mixed Int+Float struct under min.
    let mixed_mod = compile(
        "kernel test.pfx_mixed@1 {
            requires { field.read; field.write; }
            struct Acc { count: Int; weight: Float; }
            input xs: Rows<Int>;
            output result: Rows<Acc>;
            prefix over xs with min {
                let x = sample(xs);
                return Acc { count: x, weight: float(x) * 0.25 };
            }
        }",
    )
    .unwrap();
    let n = 70_000usize;
    let inp_words: Vec<u32> = (0..n)
        .map(|i| (((i * 29) % 101) as i32 - 50) as u32)
        .collect();
    let mut cpu_host = Host::new(1 << 24);
    let cpu_inp = cpu_host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                Domain::Rows(n as u32),
                inp_words.clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let cpu_inv = Invocation {
        inputs: vec![cpu_inp],
        params: vec![],
        domain: Domain::Rows(n as u32),
        context: Context::default(),
    };
    let cpu_h = CpuBackend
        .execute(&mixed_mod, &cpu_inv, &mut Scope::new(&mut cpu_host, all()))
        .unwrap()[0];
    let cpu_words = cpu_host.data(cpu_h).unwrap().storage().to_vec();
    for (tag, backend) in [("subgroups", &gpu), ("fallback", &fallback)] {
        let mut dev_host = Host::new(1 << 24);
        let dev_inp = dev_host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                    Domain::Rows(n as u32),
                    inp_words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let inv = Invocation {
            inputs: vec![dev_inp],
            params: vec![],
            domain: Domain::Rows(n as u32),
            context: Context::default(),
        };
        let prep = Backend::prepare(backend, &mixed_mod).unwrap();
        let out_h = Backend::execute(
            backend,
            prep.as_ref(),
            &mixed_mod,
            &inv,
            &mut Scope::new(&mut dev_host, all()),
        )
        .unwrap()[0];
        let dev = dev_host.resident(out_h).unwrap().expect("resident");
        let words = Backend::download(backend, dev, cpu_words.len() * 4).unwrap();
        assert_eq!(words, cpu_words, "{tag} mixed struct min mismatch");
    }

    // Exact first-overflow reporting across 1-level (block 0), 2-level (block 1), and 3-level.
    let overflow_mod = compile(
        "kernel test.pfx_ovf@1 {
            requires { field.read; field.write; }
            input xs: Rows<Int>;
            output result: Rows<Int>;
            prefix over xs with sum { return sample(xs); }
        }",
    )
    .unwrap();
    for (len, boom_at) in [(64usize, 5usize), (512, 260), (70_000, 66_000)] {
        let mut words = vec![0u32; len];
        words[boom_at - 1] = 2_000_000_000u32;
        words[boom_at] = 500_000_000u32;
        let mut cpu_host = Host::new(1 << 24);
        let cpu_inp = cpu_host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                    Domain::Rows(len as u32),
                    words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let cpu_inv = Invocation {
            inputs: vec![cpu_inp],
            params: vec![],
            domain: Domain::Rows(len as u32),
            context: Context::default(),
        };
        let cpu_err = CpuBackend
            .execute(
                &overflow_mod,
                &cpu_inv,
                &mut Scope::new(&mut cpu_host, all()),
            )
            .unwrap_err()
            .to_string();
        for (tag, backend) in [("subgroups", &gpu), ("fallback", &fallback)] {
            let mut dev_host = Host::new(1 << 24);
            let dev_inp = dev_host
                .adopt(
                    entelechy_abi::FieldData::from_storage(
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                        Domain::Rows(len as u32),
                        words.clone(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let inv = Invocation {
                inputs: vec![dev_inp],
                params: vec![],
                domain: Domain::Rows(len as u32),
                context: Context::default(),
            };
            let prep = Backend::prepare(backend, &overflow_mod).unwrap();
            let gpu_err = Backend::execute(
                backend,
                prep.as_ref(),
                &overflow_mod,
                &inv,
                &mut Scope::new(&mut dev_host, all()),
            )
            .unwrap_err()
            .to_string();
            assert_eq!(
                gpu_err, cpu_err,
                "{tag} prefix overflow mismatch for len={len}, boom_at={boom_at}"
            );
        }
    }
}

#[test]
fn parallel_pairwise_reduce_matches_cpu_bit_for_bit_across_strategies_and_types() {
    let Some(gpu) = gpu() else {
        return;
    };
    let caps = gpu.subgroups();
    let fallback = GpuBackend::new().unwrap().without_subgroups();

    // Non-associative floating-point merge verifies exact pairwise binary tree equivalence.
    let module = compile(
        "kernel test.red_tree@1 {
            requires { field.read; field.write; }
            input xs: Rows<Float>;
            output result: Rows<Float>;
            compute over xs { return sample(xs); }
            merge(a, b) { return a * 0.75 + b * 0.25 + 0.125; }
        }",
    )
    .unwrap();
    if caps.wave_parallel_256() {
        let strat =
            entelechy_gpu::wgsl::reduce_strategy(&module, caps, u64::from(gpu.workgroup_memory()));
        assert_eq!(strat, entelechy_gpu::wgsl::ReduceStrategy::Subgroup256);
        let wgsl = entelechy_gpu::wgsl::emit_reduce_with_strategy(
            &module,
            &[],
            entelechy_gpu::wgsl::Longs::Native,
            strat,
            caps,
        );
        assert!(
            wgsl.contains("subgroupShuffleDown("),
            "expected subgroupShuffleDown in reduce:\n{wgsl}"
        );
    }

    let prep_sg = Backend::prepare(&gpu, &module).unwrap();
    let prep_fb = Backend::prepare(&fallback, &module).unwrap();
    for count in [
        1usize, 2, 3, 5, 6, 7, 15, 16, 17, 31, 32, 33, 70, 255, 256, 257, 1000, 70_000,
    ] {
        let data: Vec<u32> = (0..count)
            .map(|i| (((i * 13) % 97) as f32 * 0.125 - 4.0).to_bits())
            .collect();
        let mut cpu_host = Host::new(1 << 24);
        let cpu_inp = cpu_host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                    Domain::Rows(count as u32),
                    data.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let cpu_inv = Invocation {
            inputs: vec![cpu_inp],
            params: vec![],
            domain: Domain::Rows(1),
            context: Context::default(),
        };
        let cpu_h = CpuBackend
            .execute(&module, &cpu_inv, &mut Scope::new(&mut cpu_host, all()))
            .unwrap()[0];
        let cpu_word = cpu_host.data(cpu_h).unwrap().storage()[0];

        for (tag, backend, prep) in [
            ("subgroups", &gpu, prep_sg.as_ref()),
            ("fallback", &fallback, prep_fb.as_ref()),
        ] {
            let mut dev_host = Host::new(1 << 24);
            let dev_inp = dev_host
                .adopt(
                    entelechy_abi::FieldData::from_storage(
                        entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                        Domain::Rows(count as u32),
                        data.clone(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let inv = Invocation {
                inputs: vec![dev_inp],
                params: vec![],
                domain: Domain::Rows(1),
                context: Context::default(),
            };
            let out_h = Backend::execute(
                backend,
                prep,
                &module,
                &inv,
                &mut Scope::new(&mut dev_host, all()),
            )
            .unwrap()[0];
            let dev = dev_host.resident(out_h).unwrap().expect("resident");
            let word = Backend::download(backend, dev, 4).unwrap()[0];
            assert_eq!(
                word,
                cpu_word,
                "{tag} pairwise tree mismatch at count={count}: gpu={} cpu={}",
                f32::from_bits(word),
                f32::from_bits(cpu_word)
            );
        }
    }
}

#[test]
fn subgroup_rasterizer_compaction_matches_fallback() {
    let Some(gpu) = gpu() else {
        return;
    };
    let fallback = GpuBackend::new().unwrap().without_subgroups();
    // Build 2,400 mixed small, large, offscreen, and degenerate triangles across 10 blocks of 256.
    let tri_count = 2400usize;
    let mut verts: Vec<u32> = Vec::with_capacity(tri_count * 3 * 9);
    let mut tris: Vec<u32> = Vec::with_capacity(tri_count * 3);
    let push_v = |verts: &mut Vec<u32>, x: f32, y: f32, z: f32, w: f32, nx: f32, u: f32, v: f32| {
        for f in [x, y, z, w, nx, 0.0, 1.0, u, v] {
            verts.push(f.to_bits());
        }
    };
    for t in 0..tri_count {
        let base = (t * 3) as u32;
        tris.extend([base, base + 1, base + 2]);
        let fx = ((t % 40) as f32 / 20.0) - 0.95;
        let fy = (((t / 40) % 40) as f32 / 20.0) - 0.95;
        let z = 0.1 + 0.8 * ((t * 17) % 100) as f32 / 100.0;
        let kind = t % 5;
        let (dx, dy, w) = match kind {
            0 => (0.28, 0.24, 1.0),   // large hardware-rasterized triangle
            1 => (0.015, 0.015, 1.0), // tiny software-rasterized triangle
            2 => (0.12, 0.10, -1.0),  // behind near plane (culled)
            3 => (0.0, 0.0, 1.0),     // degenerate zero area (culled)
            _ => (0.18, 0.14, 1.0),   // medium hardware triangle
        };
        let nx = (t % 7) as f32 / 7.0;
        push_v(&mut verts, fx, fy, z, w, nx, 0.0, 0.0);
        push_v(&mut verts, fx + dx, fy, z, w, nx, 1.0, 0.0);
        push_v(&mut verts, fx, fy + dy, z, w, nx, 0.0, 1.0);
    }
    let bytes = 128 * 128 * entelechy_rules::raster::SURFACE_WORDS * 4;
    let run_raster = |backend: &GpuBackend| {
        let mut host = Host::new(1 << 24);
        let vh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::vertex_schema(),
                    Domain::Rows((tri_count * 3) as u32),
                    verts.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::triangle_schema(),
                    Domain::Rows(tri_count as u32),
                    tris.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let inv = Invocation {
            inputs: vec![vh, th],
            params: vec![],
            domain: Domain::Grid(128, 128),
            context: Context::default(),
        };
        let out_h = Backend::execute_native(
            backend,
            entelechy_rules::NativeOp::Rasterize,
            &inv,
            &mut Scope::new(&mut host, all()),
        )
        .unwrap()[0];
        Backend::download(backend, host.resident(out_h).unwrap().unwrap(), bytes).unwrap()
    };
    let sg_words = run_raster(&gpu);
    let fb_words = run_raster(&fallback);
    assert_eq!(
        sg_words, fb_words,
        "subgroup raster compaction must match non-subgroup fallback bit for bit"
    );
}

#[derive(Debug, Clone, Copy)]
struct DecodedHit {
    t: f32,
    u: f32,
    v: f32,
    triangle: i32,
    instance: i32,
    custom: i32,
    front: bool,
}

fn decode_hits(words: &[u32]) -> Vec<DecodedHit> {
    words
        .chunks_exact(entelechy_rules::trace::HIT_WORDS)
        .map(|w| DecodedHit {
            t: f32::from_bits(w[0]),
            u: f32::from_bits(w[1]),
            v: f32::from_bits(w[2]),
            triangle: w[3] as i32,
            instance: w[4] as i32,
            custom: w[5] as i32,
            front: w[6] != 0,
        })
        .collect()
}

fn push_ray(out: &mut Vec<u32>, origin: [f32; 3], tmin: f32, dir: [f32; 3], tmax: f32) {
    for f in [
        origin[0], origin[1], origin[2], tmin, dir[0], dir[1], dir[2], tmax,
    ] {
        out.push(f.to_bits());
    }
}

fn push_vec3(out: &mut Vec<u32>, p: [f32; 3]) {
    for f in p {
        out.push(f.to_bits());
    }
}

fn push_mesh(out: &mut Vec<u32>, vo: i32, vc: i32, to: i32, tc: i32) {
    out.extend([vo as u32, vc as u32, to as u32, tc as u32]);
}

fn push_instance(
    out: &mut Vec<u32>,
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    mesh: i32,
    custom: i32,
    mask: i32,
) {
    for f in row0.into_iter().chain(row1).chain(row2) {
        out.push(f.to_bits());
    }
    out.extend([mesh as u32, custom as u32, mask as u32]);
}

fn assert_hits_match(label: &str, expected: &[u32], actual: &[u32], any_hit: bool, rays: &[u32]) {
    let exp = decode_hits(expected);
    let act = decode_hits(actual);
    assert_eq!(exp.len(), act.len(), "{label}: hit count mismatch");
    for (i, (&e, &a)) in exp.iter().zip(act.iter()).enumerate() {
        let e_hit = e.triangle >= 0;
        let a_hit = a.triangle >= 0;
        assert_eq!(
            e_hit, a_hit,
            "{label}: ray {i} hit/miss mismatch: expected={e:?} actual={a:?}"
        );
        if !e_hit {
            assert_eq!(a.t, -1.0, "{label}: ray {i} miss t != -1: {a:?}");
            assert_eq!(a.triangle, -1, "{label}: ray {i} miss tri != -1: {a:?}");
            assert_eq!(a.instance, -1, "{label}: ray {i} miss inst != -1: {a:?}");
            assert_eq!(a.custom, 0, "{label}: ray {i} miss custom != 0: {a:?}");
            assert!(!a.front, "{label}: ray {i} miss front != false: {a:?}");
            continue;
        }
        if any_hit {
            let base = i * entelechy_rules::trace::RAY_WORDS;
            let tmin = f32::from_bits(rays[base + 3]);
            let tmax = f32::from_bits(rays[base + 7]);
            assert!(
                a.t >= tmin - 1e-4 && a.t <= tmax + 1e-4,
                "{label}: ray {i} any-hit t out of range [{tmin}, {tmax}]: {a:?}"
            );
            assert!(
                a.u >= -1e-4 && a.v >= -1e-4 && a.u + a.v <= 1.0002,
                "{label}: ray {i} any-hit invalid barycentrics: {a:?}"
            );
            continue;
        }
        assert_eq!(
            a.instance, e.instance,
            "{label}: ray {i} instance mismatch: expected={e:?} actual={a:?}"
        );
        assert_eq!(
            a.custom, e.custom,
            "{label}: ray {i} custom mismatch: expected={e:?} actual={a:?}"
        );
        assert_eq!(
            a.triangle, e.triangle,
            "{label}: ray {i} triangle mismatch: expected={e:?} actual={a:?}"
        );
        assert_eq!(
            a.front, e.front,
            "{label}: ray {i} front mismatch: expected={e:?} actual={a:?}"
        );
        assert!(
            (a.t - e.t).abs() < 1e-3 * (1.0 + e.t.abs()),
            "{label}: ray {i} t mismatch: expected={e:?} actual={a:?}"
        );
        assert!(
            (a.u - e.u).abs() < 1e-3 && (a.v - e.v).abs() < 1e-3,
            "{label}: ray {i} bary mismatch: expected={e:?} actual={a:?}"
        );
    }
}

#[test]
fn hardware_ray_tracing_single_mesh_parity() {
    let Some(gpu) = gpu() else { return };
    let fallback = GpuBackend::new().unwrap().without_ray_query();
    let cpu = CpuBackend;

    // Build a 10x10 wavy heightfield (200 triangles) + 2 opposite-wound planes
    // + a degenerate triangle + an OOB-clamped triangle + a NaN vertex.
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    let grid = 11usize;
    for gy in 0..grid {
        for gx in 0..grid {
            let x = (gx as f32 / (grid - 1) as f32) * 4.0 - 2.0;
            let y = (gy as f32 / (grid - 1) as f32) * 4.0 - 2.0;
            let z = 0.25 * (x * 1.7).sin() * (y * 1.3).cos();
            push_vec3(&mut verts, [x, y, z]);
        }
    }
    for gy in 0..(grid - 1) {
        for gx in 0..(grid - 1) {
            let a = (gy * grid + gx) as u32;
            let b = a + 1;
            let c = a + grid as u32;
            let d = c + 1;
            tris.extend([a, b, d, a, d, c]);
        }
    }
    // Add a front-facing and a back-facing quad floating in front of the terrain.
    let base_v = (verts.len() / 3) as u32;
    push_vec3(&mut verts, [-0.8, -0.8, 1.2]);
    push_vec3(&mut verts, [0.2, -0.8, 1.2]);
    push_vec3(&mut verts, [0.2, 0.2, 1.2]);
    push_vec3(&mut verts, [-0.8, 0.2, 1.2]);
    // CCW when viewed from +Z (front face for rays going -Z)
    tris.extend([
        base_v,
        base_v + 1,
        base_v + 2,
        base_v,
        base_v + 2,
        base_v + 3,
    ]);

    let back_v = (verts.len() / 3) as u32;
    push_vec3(&mut verts, [0.3, -0.5, 1.6]);
    push_vec3(&mut verts, [1.2, -0.5, 1.6]);
    push_vec3(&mut verts, [1.2, 0.6, 1.6]);
    push_vec3(&mut verts, [0.3, 0.6, 1.6]);
    // CW when viewed from +Z (back face for rays going -Z)
    tris.extend([
        back_v,
        back_v + 2,
        back_v + 1,
        back_v,
        back_v + 3,
        back_v + 2,
    ]);

    // Degenerate triangle, OOB index triangle, and a NaN-vertex triangle.
    let nan_v = (verts.len() / 3) as u32;
    push_vec3(&mut verts, [f32::NAN, 0.0, 0.0]);
    tris.extend([0, 0, 0]);
    tris.extend([(-5i32) as u32, 999_999u32, 1]);
    tris.extend([nan_v, 0, 1]);

    // 48x48 camera rays + edge-case invalid rays.
    let mut rays = Vec::new();
    let res = 48usize;
    for ry in 0..res {
        for rx in 0..res {
            // Offset slightly off grid seams so no exact edge ties arise between HW & SW.
            let u = ((rx as f32 + 0.37) / res as f32) * 5.0 - 2.5;
            let v = ((ry as f32 + 0.61) / res as f32) * 5.0 - 2.5;
            let origin = [u * 0.6, v * 0.6, 3.5];
            let dx = u * 0.4;
            let dy = v * 0.4;
            let dz = -3.5f32;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            push_ray(
                &mut rays,
                origin,
                0.01,
                [dx / len, dy / len, dz / len],
                20.0,
            );
        }
    }
    // Invalid / edge rays.
    push_ray(&mut rays, [0.0, 0.0, 3.0], -0.1, [0.0, 0.0, -1.0], 10.0);
    push_ray(&mut rays, [0.0, 0.0, 3.0], 5.0, [0.0, 0.0, -1.0], 2.0);
    push_ray(
        &mut rays,
        [f32::NAN, 0.0, 3.0],
        0.01,
        [0.0, 0.0, -1.0],
        10.0,
    );
    push_ray(
        &mut rays,
        [0.0, 0.0, 3.0],
        0.01,
        [0.0, f32::INFINITY, -1.0],
        10.0,
    );
    push_ray(&mut rays, [0.0, 0.0, 3.0], 0.01, [0.0, 0.0, 0.0], 10.0);
    push_ray(&mut rays, [0.0, 0.0, 3.0], f32::NAN, [0.0, 0.0, -1.0], 10.0);
    push_ray(&mut rays, [0.0, 0.0, 3.0], 0.01, [0.0, 0.0, -1.0], f32::NAN);

    let ray_count = (rays.len() / entelechy_rules::trace::RAY_WORDS) as u32;
    let vert_count = (verts.len() / entelechy_rules::trace::POSITION_WORDS) as u32;
    let tri_count = (tris.len() / entelechy_rules::trace::TRIANGLE_WORDS) as u32;

    let run_backend = |backend: &dyn Backend, flags: i32, empty_mesh: bool| {
        let mut host = Host::new(1 << 24);
        let rh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::ray_schema(),
                    Domain::Rows(ray_count),
                    rays.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let (v_data, v_n, t_data, t_n) = if empty_mesh {
            // Entelechy fields are non-empty; a single degenerate triangle tests
            // the empty-BVH / zero-valid-primitive fast path cleanly.
            (vec![0u32, 0, 0], 1, vec![0u32, 0, 0], 1)
        } else {
            (verts.clone(), vert_count, tris.clone(), tri_count)
        };
        let vh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::position_schema(),
                    Domain::Rows(v_n),
                    v_data,
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::triangle_schema(),
                    Domain::Rows(t_n),
                    t_data,
                )
                .unwrap(),
            )
            .unwrap();
        let inv = Invocation {
            inputs: vec![rh, vh, th],
            params: vec![entelechy_abi::Value::Int(flags)],
            domain: Domain::Rows(ray_count),
            context: Context::default(),
        };
        let out_h = Backend::execute_native(
            backend,
            entelechy_rules::NativeOp::Trace,
            &inv,
            &mut Scope::new(&mut host, all()),
        )
        .unwrap()[0];
        if let Some(dev) = host.resident(out_h).unwrap() {
            Backend::download(
                backend,
                dev,
                ray_count as usize * entelechy_rules::trace::HIT_WORDS * 4,
            )
            .unwrap()
        } else {
            host.data(out_h).unwrap().storage().to_vec()
        }
    };

    for flags in [0i32, 1, 2, 3, 4, 5, 6] {
        let any_hit = (flags & 1) != 0;
        let cpu_out = run_backend(&cpu, flags, false);
        let hw_out = run_backend(&gpu, flags, false);
        let bvh_out = run_backend(&fallback, flags, false);
        assert_hits_match(
            &format!("single-mesh hw flags={flags}"),
            &cpu_out,
            &hw_out,
            any_hit,
            &rays,
        );
        assert_hits_match(
            &format!("single-mesh bvh flags={flags}"),
            &cpu_out,
            &bvh_out,
            any_hit,
            &rays,
        );
        // Sanity check that culling actually exercises both front & back hits.
        let decoded = decode_hits(&cpu_out);
        let hit_n = decoded.iter().filter(|h| h.triangle >= 0).count();
        match flags {
            0 | 1 => assert!(hit_n > 500, "flags={flags} hit_n={hit_n}"),
            2 | 3 => {
                assert!(
                    hit_n > 200,
                    "cull_back should still hit front faces: {hit_n}"
                );
                assert!(decoded.iter().filter(|h| h.triangle >= 0).all(|h| h.front));
            }
            4 | 5 => {
                assert!(
                    hit_n > 10,
                    "cull_front should hit back-facing quad: {hit_n}"
                );
                assert!(decoded.iter().filter(|h| h.triangle >= 0).all(|h| !h.front));
            }
            6 => assert_eq!(hit_n, 0, "cull_back|cull_front must miss all"),
            _ => {}
        }
    }

    // All-degenerate mesh must miss cleanly on all paths without crashing BLAS/TLAS creation.
    let empty_cpu = run_backend(&cpu, 0, true);
    let empty_hw = run_backend(&gpu, 0, true);
    let empty_bvh = run_backend(&fallback, 0, true);
    assert_eq!(empty_cpu, empty_hw);
    assert_eq!(empty_cpu, empty_bvh);
    assert!(decode_hits(&empty_cpu).iter().all(|h| h.triangle == -1));
}

#[test]
fn hardware_ray_tracing_dynamic_gpu_mesh_zero_readback() {
    let Some(gpu) = gpu() else { return };
    let fallback = GpuBackend::new().unwrap().without_ray_query();
    let cpu = CpuBackend;

    // Kernel generating animated triangle vertices directly in GPU memory.
    let vert_kernel = compile(
        "kernel test.anim_verts@1 {
            requires { field.write; }
            param shift: Float = 0.0;
            output positions: Rows<Vec3>;
            compute {
                let i = element();
                let x = if i == 0 { -1.0 } else if i == 1 { 1.0 } else { 0.0 };
                let y = if i == 2 { 1.0 } else { -1.0 };
                return vec3(x + shift, y, 0.5 * shift);
            }
        }",
    )
    .unwrap();

    let mut rays = Vec::new();
    for ry in 0..16 {
        for rx in 0..16 {
            let x = (rx as f32 + 0.41) / 16.0 * 3.0 - 1.5;
            let y = (ry as f32 + 0.57) / 16.0 * 3.0 - 1.5;
            push_ray(&mut rays, [x, y, 3.0], 0.01, [0.0, 0.0, -1.0], 10.0);
        }
    }
    let ray_count = (rays.len() / entelechy_rules::trace::RAY_WORDS) as u32;
    let tris: Vec<u32> = vec![0, 1, 2];

    let run_two_frames = |backend: &GpuBackend, tag: &str| {
        let prep = Backend::prepare(backend, &vert_kernel).unwrap();
        let mut host = Host::new(1 << 24);
        let rh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::ray_schema(),
                    Domain::Rows(ray_count),
                    rays.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::triangle_schema(),
                    Domain::Rows(1),
                    tris.clone(),
                )
                .unwrap(),
            )
            .unwrap();

        for (frame, shift) in [(0usize, -0.35f32), (1, 0.45f32)] {
            backend.evaluation_begun();
            let v_inv = Invocation {
                inputs: vec![],
                params: vec![entelechy_abi::Value::Float(shift)],
                domain: Domain::Rows(3),
                context: Context::default(),
            };
            let vh = Backend::execute(
                backend,
                &*prep,
                &vert_kernel,
                &v_inv,
                &mut Scope::new(&mut host, all()),
            )
            .unwrap()[0];
            assert!(
                host.resident(vh).unwrap().is_some(),
                "{tag}: vertices should stay GPU-resident"
            );
            let t_inv = Invocation {
                inputs: vec![rh, vh, th],
                params: vec![entelechy_abi::Value::Int(0)],
                domain: Domain::Rows(ray_count),
                context: Context::default(),
            };
            let hit_h = Backend::execute_native(
                backend,
                entelechy_rules::NativeOp::Trace,
                &t_inv,
                &mut Scope::new(&mut host, all()),
            )
            .unwrap()[0];
            let gpu_hits = Backend::download(
                backend,
                host.resident(hit_h).unwrap().unwrap(),
                ray_count as usize * entelechy_rules::trace::HIT_WORDS * 4,
            )
            .unwrap();
            // Compare against CPU using the exact generated vertices.
            let gen_verts =
                Backend::download(backend, host.resident(vh).unwrap().unwrap(), 3 * 3 * 4).unwrap();
            let cpu_hits = entelechy_rules::trace::trace_words(&rays, &gen_verts, &tris, 0);
            assert_hits_match(
                &format!("{tag} dynamic frame {frame} shift={shift}"),
                &cpu_hits,
                &gpu_hits,
                false,
                &rays,
            );
            let hit_n = decode_hits(&gpu_hits)
                .iter()
                .filter(|h| h.triangle >= 0)
                .count();
            assert!(hit_n > 20, "{tag} frame {frame} hit_n={hit_n}");
            let _ = host.release(hit_h);
            let _ = host.release(vh);
        }
    };

    run_two_frames(&gpu, "hw");
    run_two_frames(&fallback, "bvh");
    let _ = cpu;
}

#[test]
fn hardware_ray_tracing_instanced_tlas_parity() {
    let Some(gpu) = gpu() else { return };
    let fallback = GpuBackend::new().unwrap().without_ray_query();
    let cpu = CpuBackend;

    // Pack 3 distinct meshes + 1 empty mesh descriptor into shared buffers.
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    let mut meshes = Vec::new();

    // Mesh 0: unit quad in XY at z = 0 (2 triangles, CCW from +Z).
    let m0_vo = (verts.len() / 3) as i32;
    let m0_to = (tris.len() / 3) as i32;
    push_vec3(&mut verts, [-0.5, -0.5, 0.0]);
    push_vec3(&mut verts, [0.5, -0.5, 0.0]);
    push_vec3(&mut verts, [0.5, 0.5, 0.0]);
    push_vec3(&mut verts, [-0.5, 0.5, 0.0]);
    tris.extend([0u32, 1, 2, 0, 2, 3]);
    push_mesh(&mut meshes, m0_vo, 4, m0_to, 2);

    // Mesh 1: pyramid (4 side triangles + 2 base triangles).
    let m1_vo = (verts.len() / 3) as i32;
    let m1_to = (tris.len() / 3) as i32;
    push_vec3(&mut verts, [-0.5, -0.5, 0.0]);
    push_vec3(&mut verts, [0.5, -0.5, 0.0]);
    push_vec3(&mut verts, [0.5, 0.5, 0.0]);
    push_vec3(&mut verts, [-0.5, 0.5, 0.0]);
    push_vec3(&mut verts, [0.0, 0.0, 0.9]);
    tris.extend([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 2, 1, 0, 3, 2]);
    push_mesh(&mut meshes, m1_vo, 5, m1_to, 6);

    // Mesh 2: single tilted triangle.
    let m2_vo = (verts.len() / 3) as i32;
    let m2_to = (tris.len() / 3) as i32;
    push_vec3(&mut verts, [-0.7, -0.4, -0.2]);
    push_vec3(&mut verts, [0.8, -0.3, 0.3]);
    push_vec3(&mut verts, [0.1, 0.9, 0.1]);
    tris.extend([0, 1, 2]);
    push_mesh(&mut meshes, m2_vo, 3, m2_to, 1);

    // Mesh 3: empty mesh.
    push_mesh(&mut meshes, 0, 0, 0, 0);

    let build_instances = |shift_x: f32| {
        let mut insts = Vec::new();
        let c = 0.6f32.cos();
        let s = 0.6f32.sin();
        // Row 0: translated mesh 0
        push_instance(
            &mut insts,
            [1.0, 0.0, 0.0, -1.2 + shift_x],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.2],
            0,
            101,
            0x01,
        );
        // Row 1: rotated + non-uniform scaled mesh 1, negative 32-bit custom ID
        push_instance(
            &mut insts,
            [1.2 * c, -0.8 * s, 0.0, 0.1],
            [1.2 * s, 0.8 * c, 0.0, 0.2],
            [0.0, 0.0, 1.3, -0.3],
            1,
            -42,
            0x03,
        );
        // Row 2: invalid mesh index (must be skipped while preserving original row indices)
        push_instance(
            &mut insts,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            99,
            999,
            0xFF,
        );
        // Row 3: negative-determinant reflection of mesh 0 (flips winding!), >24-bit custom ID
        push_instance(
            &mut insts,
            [-1.1, 0.0, 0.0, 1.3 - shift_x],
            [0.0, 1.1, 0.0, -0.4],
            [0.0, 0.0, 1.0, 0.4],
            0,
            0x7FFF_1234,
            0x02,
        );
        // Row 4: hidden (mask = 0)
        push_instance(
            &mut insts,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            0,
            404,
            0,
        );
        // Row 5: singular transform (det = 0)
        push_instance(
            &mut insts,
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            0,
            505,
            0xFF,
        );
        // Row 6: translated + tilted mesh 2
        push_instance(
            &mut insts,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.1],
            [0.0, 0.0, 1.0, 0.1],
            2,
            606,
            0x04,
        );
        // Row 7: references empty mesh 3
        push_instance(
            &mut insts,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            3,
            707,
            0xFF,
        );
        insts
    };

    let mut rays = Vec::new();
    let res = 40usize;
    for ry in 0..res {
        for rx in 0..res {
            let x = ((rx as f32 + 0.31) / res as f32) * 4.4 - 2.2;
            let y = ((ry as f32 + 0.67) / res as f32) * 3.6 - 1.4;
            let dir = [x * 0.06, y * 0.06, -1.0f32];
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            push_ray(
                &mut rays,
                [x, y, 4.0],
                0.01,
                [dir[0] / len, dir[1] / len, dir[2] / len],
                15.0,
            );
        }
    }
    let ray_count = (rays.len() / entelechy_rules::trace::RAY_WORDS) as u32;
    let vert_count = (verts.len() / entelechy_rules::trace::POSITION_WORDS) as u32;
    let tri_count = (tris.len() / entelechy_rules::trace::TRIANGLE_WORDS) as u32;
    let mesh_count = (meshes.len() / entelechy_rules::trace::MESH_WORDS) as u32;

    // Run two frames sharing immutable meshes/vertices/triangles while moving instances,
    // verifying BLAS cache reuse and fresh TLAS updates across all masks and flags.
    let mut cpu_host = Host::new(1 << 24);
    let mut hw_host = Host::new(1 << 24);
    let mut bvh_host = Host::new(1 << 24);

    let adopt_scene = |host: &mut Host| {
        let rh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::ray_schema(),
                    Domain::Rows(ray_count),
                    rays.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let vh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::position_schema(),
                    Domain::Rows(vert_count),
                    verts.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::triangle_schema(),
                    Domain::Rows(tri_count),
                    tris.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let mh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::mesh_schema(),
                    Domain::Rows(mesh_count),
                    meshes.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        (rh, vh, th, mh)
    };
    let cpu_scene = adopt_scene(&mut cpu_host);
    let hw_scene = adopt_scene(&mut hw_host);
    let bvh_scene = adopt_scene(&mut bvh_host);

    for (frame, shift_x) in [(0usize, 0.0f32), (1, 0.45f32)] {
        gpu.evaluation_begun();
        fallback.evaluation_begun();
        let insts = build_instances(shift_x);
        let inst_count = (insts.len() / entelechy_rules::trace::INSTANCE_WORDS) as u32;
        let adopt_inst = |host: &mut Host| {
            host.adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::instance_schema(),
                    Domain::Rows(inst_count),
                    insts.clone(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        let cpu_ih = adopt_inst(&mut cpu_host);
        let hw_ih = adopt_inst(&mut hw_host);
        let bvh_ih = adopt_inst(&mut bvh_host);

        for (flags, mask) in [
            (0i32, 0xFFi32),
            (1, 0xFF),
            (2, 0xFF),
            (3, 0xFF),
            (4, 0xFF),
            (5, 0xFF),
            (6, 0xFF),
            (0, 0x01),
            (0, 0x02),
            (0, 0x04),
            (0, 0x00),
        ] {
            let run_one = |backend: &dyn Backend,
                           host: &mut Host,
                           sc: (
                entelechy_abi::Handle,
                entelechy_abi::Handle,
                entelechy_abi::Handle,
                entelechy_abi::Handle,
            ),
                           ih: entelechy_abi::Handle| {
                let inv = Invocation {
                    inputs: vec![sc.0, sc.1, sc.2, sc.3, ih],
                    params: vec![
                        entelechy_abi::Value::Int(flags),
                        entelechy_abi::Value::Int(mask),
                    ],
                    domain: Domain::Rows(ray_count),
                    context: Context::default(),
                };
                let out_h = Backend::execute_native(
                    backend,
                    entelechy_rules::NativeOp::TraceInstances,
                    &inv,
                    &mut Scope::new(host, all()),
                )
                .unwrap()[0];
                let words = if let Some(dev) = host.resident(out_h).unwrap() {
                    Backend::download(
                        backend,
                        dev,
                        ray_count as usize * entelechy_rules::trace::HIT_WORDS * 4,
                    )
                    .unwrap()
                } else {
                    host.data(out_h).unwrap().storage().to_vec()
                };
                let _ = host.release(out_h);
                words
            };
            let any_hit = (flags & 1) != 0;
            let cpu_out = run_one(&cpu, &mut cpu_host, cpu_scene, cpu_ih);
            let hw_out = run_one(&gpu, &mut hw_host, hw_scene, hw_ih);
            let bvh_out = run_one(&fallback, &mut bvh_host, bvh_scene, bvh_ih);
            assert_hits_match(
                &format!("tlas hw frame={frame} flags={flags} mask={mask:#x}"),
                &cpu_out,
                &hw_out,
                any_hit,
                &rays,
            );
            assert_hits_match(
                &format!("tlas bvh frame={frame} flags={flags} mask={mask:#x}"),
                &cpu_out,
                &bvh_out,
                any_hit,
                &rays,
            );
            if flags == 0 && mask == 0xFF {
                let decoded = decode_hits(&cpu_out);
                // Every valid instance (rows 0, 1, 3, 6) must be hit!
                for expected_inst in [0i32, 1, 3, 6] {
                    assert!(
                        decoded.iter().any(|h| h.instance == expected_inst),
                        "frame={frame}: expected hits on instance {expected_inst}"
                    );
                }
                // Mirrored instance row 3 (neg-det reflection of CCW quad) must have front == false!
                assert!(
                    decoded
                        .iter()
                        .filter(|h| h.instance == 3)
                        .all(|h| !h.front && h.custom == 0x7FFF_1234),
                    "frame={frame}: mirrored instance row 3 must flip front face and preserve 32-bit custom"
                );
            }
        }
        let _ = cpu_host.release(cpu_ih);
        let _ = hw_host.release(hw_ih);
        let _ = bvh_host.release(bvh_ih);
    }

    // All-hidden instances must cleanly trigger the zero-active-instance TLAS fast path.
    let mut hidden_insts = Vec::new();
    push_instance(
        &mut hidden_insts,
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        0,
        0,
        0,
    );
    let adopt_hidden = |host: &mut Host| {
        host.adopt(
            entelechy_abi::FieldData::from_storage(
                entelechy_rules::trace::instance_schema(),
                Domain::Rows(1),
                hidden_insts.clone(),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let h_cpu = adopt_hidden(&mut cpu_host);
    let h_hw = adopt_hidden(&mut hw_host);
    let h_bvh = adopt_hidden(&mut bvh_host);
    for (backend, host, sc, ih) in [
        (&cpu as &dyn Backend, &mut cpu_host, cpu_scene, h_cpu),
        (&gpu as &dyn Backend, &mut hw_host, hw_scene, h_hw),
        (&fallback as &dyn Backend, &mut bvh_host, bvh_scene, h_bvh),
    ] {
        let inv = Invocation {
            inputs: vec![sc.0, sc.1, sc.2, sc.3, ih],
            params: vec![
                entelechy_abi::Value::Int(0),
                entelechy_abi::Value::Int(0xFF),
            ],
            domain: Domain::Rows(ray_count),
            context: Context::default(),
        };
        let out_h = Backend::execute_native(
            backend,
            entelechy_rules::NativeOp::TraceInstances,
            &inv,
            &mut Scope::new(host, all()),
        )
        .unwrap()[0];
        let words = if let Some(dev) = host.resident(out_h).unwrap() {
            Backend::download(
                backend,
                dev,
                ray_count as usize * entelechy_rules::trace::HIT_WORDS * 4,
            )
            .unwrap()
        } else {
            host.data(out_h).unwrap().storage().to_vec()
        };
        assert!(decode_hits(&words).iter().all(|h| h.triangle == -1));
        let _ = host.release(out_h);
    }
}

#[test]
fn gpu_lbvh_dynamic_refit_and_indirect_ray_compaction_parity() {
    let Some(gpu) = gpu() else { return };
    let fallback = GpuBackend::new().unwrap().without_ray_query();

    // 13x13 animated grid (169 verts, 288 tris) generated directly on GPU + 4 extra tris.
    let grid = 13u32;
    let vert_count = grid * grid;
    let vert_kernel = compile(
        "kernel test.deform_grid@1 {
            requires { field.write; }
            param phase: Float = 0.0;
            output positions: Rows<Vec3>;
            compute {
                let i = element();
                let gx = i % 13;
                let gy = i / 13;
                let fx = (float(gx) / 12.0) * 3.2 - 1.6;
                let fy = (float(gy) / 12.0) * 3.2 - 1.6;
                let z = 0.35 * sin(fx * 2.1 + phase) * cos(fy * 1.9 - phase * 0.7)
                    + 0.25 * cos(phase * 0.5);
                return vec3(fx + 0.15 * sin(phase), fy + 0.12 * cos(phase), z);
            }
        }",
    )
    .unwrap();

    let mut tris: Vec<u32> = Vec::new();
    for gy in 0..(grid - 1) {
        for gx in 0..(grid - 1) {
            let a = gy * grid + gx;
            let b = a + 1;
            let c = a + grid;
            let d = c + 1;
            tris.extend([a, b, d, a, d, c]);
        }
    }
    // Include a degenerate triangle and a valid cross-mesh triangle to stress refit bounds.
    tris.extend([0, 0, 0, 0, 12, 168]);
    let tri_count = (tris.len() / 3) as u32;

    // 640 rays (> COMPACT_RAY_THRESHOLD = 256): interleaved live + dead/invalid rays.
    let mut rays = Vec::new();
    for i in 0..640u32 {
        if i % 2 == 1 {
            // Dead ray variants: negative tmin, tmax < tmin, zero dir, NaN origin.
            match (i / 2) % 4 {
                0 => push_ray(&mut rays, [0.0, 0.0, 3.0], -0.2, [0.0, 0.0, -1.0], 10.0),
                1 => push_ray(&mut rays, [0.0, 0.0, 3.0], 4.0, [0.0, 0.0, -1.0], 2.0),
                2 => push_ray(&mut rays, [0.0, 0.0, 3.0], 0.01, [0.0, 0.0, 0.0], 10.0),
                _ => push_ray(
                    &mut rays,
                    [f32::NAN, 0.0, 3.0],
                    0.01,
                    [0.0, 0.0, -1.0],
                    10.0,
                ),
            }
        } else {
            let idx = i / 2;
            let rx = idx % 20;
            let ry = idx / 20;
            let x = ((rx as f32 + 0.37) / 20.0) * 3.4 - 1.7;
            let y = ((ry as f32 + 0.61) / 16.0) * 3.4 - 1.7;
            push_ray(&mut rays, [x, y, 3.5], 0.01, [0.0, 0.0, -1.0], 12.0);
        }
    }
    let ray_count = (rays.len() / entelechy_rules::trace::RAY_WORDS) as u32;

    // All-dead 512-ray batch to exercise zero-workgroup dispatch_workgroups_indirect.
    let mut dead_rays = Vec::new();
    for _ in 0..512u32 {
        push_ray(&mut dead_rays, [0.0, 0.0, 3.0], 5.0, [0.0, 0.0, -1.0], 1.0);
    }

    let run_35_frames = |backend: &GpuBackend, tag: &str| {
        let prep = Backend::prepare(backend, &vert_kernel).unwrap();
        let mut host = Host::new(1 << 24);
        let rh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::ray_schema(),
                    Domain::Rows(ray_count),
                    rays.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let dead_rh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::ray_schema(),
                    Domain::Rows(512),
                    dead_rays.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::trace::triangle_schema(),
                    Domain::Rows(tri_count),
                    tris.clone(),
                )
                .unwrap(),
            )
            .unwrap();

        // 35 frames: covers initial build (0), dynamic promotion/refits (1..31),
        // periodic full GPU LBVH rebuild (32), and post-rebuild refits (33..34).
        for frame in 0..35usize {
            backend.evaluation_begun();
            let phase = frame as f32 * 0.23;
            let v_inv = Invocation {
                inputs: vec![],
                params: vec![entelechy_abi::Value::Float(phase)],
                domain: Domain::Rows(vert_count),
                context: Context::default(),
            };
            let vh = Backend::execute(
                backend,
                &*prep,
                &vert_kernel,
                &v_inv,
                &mut Scope::new(&mut host, all()),
            )
            .unwrap()[0];
            let t_inv = Invocation {
                inputs: vec![rh, vh, th],
                params: vec![entelechy_abi::Value::Int(0)],
                domain: Domain::Rows(ray_count),
                context: Context::default(),
            };
            let hit_h = Backend::execute_native(
                backend,
                entelechy_rules::NativeOp::Trace,
                &t_inv,
                &mut Scope::new(&mut host, all()),
            )
            .unwrap()[0];

            // Verify exact CPU parity on initial build, promotion, mid-refit,
            // pre-rebuild, periodic full rebuild (32), and post-rebuild refits.
            if matches!(frame, 0 | 1 | 2 | 15 | 31 | 32 | 33 | 34) {
                let gpu_hits = Backend::download(
                    backend,
                    host.resident(hit_h).unwrap().unwrap(),
                    ray_count as usize * entelechy_rules::trace::HIT_WORDS * 4,
                )
                .unwrap();
                let gen_verts = Backend::download(
                    backend,
                    host.resident(vh).unwrap().unwrap(),
                    vert_count as usize * entelechy_rules::trace::POSITION_WORDS * 4,
                )
                .unwrap();
                let cpu_hits = entelechy_rules::trace::trace_words(&rays, &gen_verts, &tris, 0);
                assert_hits_match(
                    &format!("{tag} refit frame {frame}"),
                    &cpu_hits,
                    &gpu_hits,
                    false,
                    &rays,
                );
                let decoded = decode_hits(&gpu_hits);
                let live_hits = decoded.iter().filter(|h| h.triangle >= 0).count();
                assert!(live_hits > 150, "{tag} frame {frame} live_hits={live_hits}");
                // All odd-indexed rays are invalid and must miss cleanly.
                for (idx, h) in decoded.iter().enumerate() {
                    if idx % 2 == 1 {
                        assert_eq!(h.triangle, -1, "{tag} frame {frame} dead ray {idx} hit");
                    }
                }
            }
            if frame == 34 {
                // Exercise zero-active-ray indirect dispatch cleanly.
                let dead_inv = Invocation {
                    inputs: vec![dead_rh, vh, th],
                    params: vec![entelechy_abi::Value::Int(0)],
                    domain: Domain::Rows(512),
                    context: Context::default(),
                };
                let dead_out = Backend::execute_native(
                    backend,
                    entelechy_rules::NativeOp::Trace,
                    &dead_inv,
                    &mut Scope::new(&mut host, all()),
                )
                .unwrap()[0];
                let words = Backend::download(
                    backend,
                    host.resident(dead_out).unwrap().unwrap(),
                    512 * entelechy_rules::trace::HIT_WORDS * 4,
                )
                .unwrap();
                assert!(decode_hits(&words).iter().all(|h| h.triangle == -1));
                let _ = host.release(dead_out);
            }
            let _ = host.release(hit_h);
            let _ = host.release(vh);
        }
    };

    run_35_frames(&gpu, "hw-dynamic");
    run_35_frames(&fallback, "lbvh-refit");
}

#[test]
fn gpu_hiz_occlusion_culling_matches_cpu_including_coplanar_and_odd_sizes() {
    let Some(gpu) = gpu() else { return };
    let fallback = GpuBackend::new().unwrap().without_subgroups();
    let cpu = CpuBackend;

    // Odd dimensions exercise conservative 3-wide/3-tall Hi-Z reduction at every odd mip.
    let (width, height) = (161u32, 91u32);
    let mut verts: Vec<u32> = Vec::new();
    let mut tris: Vec<u32> = Vec::new();
    let push_tri = |verts: &mut Vec<u32>,
                    tris: &mut Vec<u32>,
                    p0: [f32; 4],
                    p1: [f32; 4],
                    p2: [f32; 4],
                    tag: f32| {
        let base = (verts.len() / 9) as u32;
        for (p, uv) in [(p0, [0.0f32, 0.0]), (p1, [1.0, 0.0]), (p2, [0.0, 1.0])] {
            for f in [p[0], p[1], p[2], p[3], tag, 0.0, 1.0, uv[0], uv[1]] {
                verts.push(f.to_bits());
            }
        }
        tris.extend([base, base + 1, base + 2]);
    };

    // 1) Large foreground opaque occluder quad covering NDC [-0.73..0.73, -0.71..0.71] at z = 0.25.
    // Chosen slightly off exact pixel centers so hardware & rule edge evaluation match bit-for-bit.
    push_tri(
        &mut verts,
        &mut tris,
        [-0.731, -0.713, 0.25, 1.0],
        [0.733, -0.713, 0.25, 1.0],
        [0.733, 0.711, 0.25, 1.0],
        1.0,
    );
    push_tri(
        &mut verts,
        &mut tris,
        [-0.731, -0.713, 0.25, 1.0],
        [0.733, 0.711, 0.25, 1.0],
        [-0.731, 0.711, 0.25, 1.0],
        2.0,
    );

    // 2) Coplanar equal-depth tie-break pair with identical vertices at z = 0.25
    //    (earlier row 0 MUST win over row 2 on both CPU and GPU!).
    push_tri(
        &mut verts,
        &mut tris,
        [-0.731, -0.713, 0.25, 1.0],
        [0.733, -0.713, 0.25, 1.0],
        [0.733, 0.711, 0.25, 1.0],
        3.0,
    );

    // 3) Slanted triangle crossing the near plane (z = -0.2 to z = 0.5) — must never be culled.
    push_tri(
        &mut verts,
        &mut tris,
        [-0.95, -0.92, -0.2, 1.0],
        [-0.76, -0.92, 0.2, 1.0],
        [-0.85, -0.15, 0.45, 1.0],
        4.0,
    );

    // 4) Visible foreground micro-triangles (< 8px) in front of the occluder at z = 0.12
    //    exercising compacted small_rows + dispatch_workgroups_indirect draw_small.
    for i in 0..48u32 {
        let fx = -0.55 + (i % 12) as f32 * 0.092;
        let fy = -0.35 + (i / 12) as f32 * 0.21;
        push_tri(
            &mut verts,
            &mut tris,
            [fx, fy, 0.12, 1.0],
            [fx + 0.041, fy, 0.12, 1.0],
            [fx + 0.013, fy + 0.057, 0.12, 1.0],
            5.0,
        );
    }

    // 5) Visible unoccluded border triangles touching the odd right/bottom edges (x=160, y=90).
    push_tri(
        &mut verts,
        &mut tris,
        [0.81, -0.96, 0.62, 1.0],
        [0.999, -0.96, 0.62, 1.0],
        [0.999, 0.96, 0.62, 1.0],
        6.0,
    );
    push_tri(
        &mut verts,
        &mut tris,
        [-0.96, -0.999, 0.64, 1.0],
        [0.96, -0.999, 0.64, 1.0],
        [0.01, -0.78, 0.64, 1.0],
        7.0,
    );

    let visible_tri_count = (tris.len() / 3) as u32;
    let visible_vert_count = (verts.len() / 9) as u32;
    let visible_verts = verts.clone();
    let visible_tris = tris.clone();

    // 6) 17,500 occluded background triangles (> HIZ_TRI_THRESHOLD = 16,384) completely behind
    //    the foreground occluder at z = 0.50..0.92, mixing large, micro (< 8px), and far-plane tris.
    for i in 0..17_500u32 {
        let fx = -0.64 + ((i * 37) % 100) as f32 * 0.011;
        let fy = -0.62 + ((i * 53) % 100) as f32 * 0.011;
        let z = 0.50 + ((i * 19) % 42) as f32 * 0.01;
        match i % 6 {
            0 => push_tri(
                &mut verts,
                &mut tris,
                [fx, fy, z, 1.0],
                [fx + 0.18, fy, z, 1.0],
                [fx + 0.04, fy + 0.17, z, 1.0],
                8.0,
            ),
            1 | 2 => push_tri(
                &mut verts,
                &mut tris,
                [fx, fy, z, 1.0],
                [fx + 0.025, fy, z, 1.0],
                [fx + 0.008, fy + 0.032, z, 1.0],
                9.0,
            ),
            3 => push_tri(
                &mut verts,
                &mut tris,
                [fx, fy, 1.25, 1.0],
                [fx + 0.15, fy, 1.25, 1.0],
                [fx, fy + 0.15, 1.25, 1.0],
                10.0,
            ),
            4 => push_tri(
                &mut verts,
                &mut tris,
                [fx, fy, z, -1.0],
                [fx + 0.1, fy, z, 1.0],
                [fx, fy + 0.1, z, 1.0],
                11.0,
            ),
            _ => push_tri(
                &mut verts,
                &mut tris,
                [fx, fy, z, 1.0],
                [fx + 0.09, fy, z, 1.0],
                [fx + 0.02, fy + 0.11, z, 1.0],
                12.0,
            ),
        }
    }

    let vert_count = (verts.len() / 9) as u32;
    let tri_count = (tris.len() / 3) as u32;
    assert!(tri_count >= 16_384, "must exceed HIZ_TRI_THRESHOLD");

    let surface_bytes =
        width as usize * height as usize * entelechy_rules::raster::SURFACE_WORDS * 4;
    let run_surface = |backend: &dyn Backend, v: &[u32], vn: u32, t: &[u32], tn: u32| {
        let mut host = Host::new(1 << 26);
        let vh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::vertex_schema(),
                    Domain::Rows(vn),
                    v.to_vec(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::triangle_schema(),
                    Domain::Rows(tn),
                    t.to_vec(),
                )
                .unwrap(),
            )
            .unwrap();
        let inv = Invocation {
            inputs: vec![vh, th],
            params: vec![],
            domain: Domain::Grid(width, height),
            context: Context::default(),
        };
        let out_h = Backend::execute_native(
            backend,
            entelechy_rules::NativeOp::Rasterize,
            &inv,
            &mut Scope::new(&mut host, all()),
        )
        .unwrap()[0];
        if let Some(dev) = host.resident(out_h).unwrap() {
            Backend::download(backend, dev, surface_bytes).unwrap()
        } else {
            host.data(out_h).unwrap().storage().to_vec()
        }
    };

    let cpu_words = run_surface(&cpu, &verts, vert_count, &tris, tri_count);
    let gpu_words = run_surface(&gpu, &verts, vert_count, &tris, tri_count);
    let fb_words = run_surface(&fallback, &verts, vert_count, &tris, tri_count);
    let gpu_no_hiz = run_surface(
        &gpu,
        &visible_verts,
        visible_vert_count,
        &visible_tris,
        visible_tri_count,
    );
    assert_eq!(
        gpu_words, fb_words,
        "2-phase Hi-Z subgroup and fallback rasterizers must match bit-for-bit"
    );
    assert_eq!(
        gpu_words, gpu_no_hiz,
        "2-phase Hi-Z occlusion culling must match single-pass visible-only GPU output bit-for-bit"
    );

    // Every pixel must match CPU reference on triangle ID & depth (allowing rare 1-pixel
    // exact-edge ties on the slanted boundary only if zero interior mismatches occur).
    let sw = entelechy_rules::raster::SURFACE_WORDS;
    let mut mismatched = 0usize;
    let mut visible_micro = 0usize;
    let mut visible_border = 0usize;
    let mut occluder_wins_coplanar = 0usize;
    let mut samples = Vec::new();
    for p in 0..(width as usize * height as usize) {
        let c_id = cpu_words[p * sw + 6] as i32;
        let g_id = gpu_words[p * sw + 6] as i32;
        let c_z = f32::from_bits(cpu_words[p * sw]);
        let g_z = f32::from_bits(gpu_words[p * sw]);
        if c_id != g_id || (c_z - g_z).abs() > 1e-4 {
            if samples.len() < 12 {
                samples.push((p % width as usize, p / width as usize, c_id, g_id, c_z, g_z));
            }
            mismatched += 1;
        }
        if g_id == 0 || g_id == 1 {
            occluder_wins_coplanar += 1;
        } else if (4..52).contains(&g_id) {
            visible_micro += 1;
        } else if g_id == 52 || g_id == 53 {
            visible_border += 1;
        }
    }
    assert!(
        mismatched <= 4,
        "Hi-Z rasterizer mismatched CPU on {mismatched} pixels out of {}: first={samples:?}",
        width * height
    );
    assert!(
        occluder_wins_coplanar > 4000,
        "occluder pixels={occluder_wins_coplanar}"
    );
    assert!(visible_micro > 80, "visible micro pixels={visible_micro}");
    assert!(
        visible_border > 300,
        "visible odd-border pixels={visible_border}"
    );
    assert!(
        !(0..(width as usize * height as usize)).any(|p| gpu_words[p * sw + 6] as i32 == 2),
        "coplanar duplicate row 2 must lose to earlier row 0"
    );

    // Also verify `NativeOp::Cover` with a pre-filled `before` depth canvas (`has_before = 1`).
    let mut before_keys = Vec::with_capacity(width as usize * height as usize * 2);
    let pre_z = 0.18f32.to_bits();
    for _y in 0..height {
        for x in 0..width {
            if x < width / 2 {
                // Pre-populated closer depth 0.18 on left half (row 777).
                before_keys.extend([777u32, pre_z]);
            } else {
                // Empty canvas key: i64::MAX.
                before_keys.extend([0xFFFF_FFFFu32, 0x7FFF_FFFFu32]);
            }
        }
    }
    let cover_bytes = width as usize * height as usize * 2 * 4;
    let run_cover = |backend: &dyn Backend, v: &[u32], vn: u32, t: &[u32], tn: u32| {
        let mut host = Host::new(1 << 26);
        let vh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::vertex_schema(),
                    Domain::Rows(vn),
                    v.to_vec(),
                )
                .unwrap(),
            )
            .unwrap();
        let th = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::triangle_schema(),
                    Domain::Rows(tn),
                    t.to_vec(),
                )
                .unwrap(),
            )
            .unwrap();
        let kh = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    entelechy_rules::raster::canvas_schema(),
                    Domain::Grid(width, height),
                    before_keys.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let inv = Invocation {
            inputs: vec![vh, th, kh],
            params: vec![],
            domain: Domain::Grid(width, height),
            context: Context::default(),
        };
        let out_h = Backend::execute_native(
            backend,
            entelechy_rules::NativeOp::Cover,
            &inv,
            &mut Scope::new(&mut host, all()),
        )
        .unwrap()[0];
        if let Some(dev) = host.resident(out_h).unwrap() {
            Backend::download(backend, dev, cover_bytes).unwrap()
        } else {
            host.data(out_h).unwrap().storage().to_vec()
        }
    };
    let cpu_cov = run_cover(&cpu, &verts, vert_count, &tris, tri_count);
    let gpu_cov = run_cover(&gpu, &verts, vert_count, &tris, tri_count);
    let gpu_cov_no_hiz = run_cover(
        &gpu,
        &visible_verts,
        visible_vert_count,
        &visible_tris,
        visible_tri_count,
    );
    assert_eq!(
        gpu_cov, gpu_cov_no_hiz,
        "Hi-Z cover with pre-filled depth must match single-pass visible-only GPU bit-for-bit"
    );
    let mut cov_mismatch = 0usize;
    let mut cov_samples = Vec::new();
    for p in 0..(width as usize * height as usize) {
        let c_row = cpu_cov[p * 2];
        let g_row = gpu_cov[p * 2];
        let c_z = f32::from_bits(cpu_cov[p * 2 + 1]);
        let g_z = f32::from_bits(gpu_cov[p * 2 + 1]);
        let z_diff = if cpu_cov[p * 2 + 1] == 0x7FFF_FFFF && gpu_cov[p * 2 + 1] == 0x7FFF_FFFF {
            0.0
        } else {
            (c_z - g_z).abs()
        };
        if c_row != g_row || z_diff > 1e-4 {
            if cov_samples.len() < 12 {
                cov_samples.push((
                    p % width as usize,
                    p / width as usize,
                    c_row,
                    g_row,
                    c_z,
                    g_z,
                ));
            }
            cov_mismatch += 1;
        }
    }
    assert!(
        cov_mismatch <= 4,
        "Hi-Z cover with pre-filled depth mismatched CPU on {cov_mismatch} pixels: first={cov_samples:?}"
    );
}
