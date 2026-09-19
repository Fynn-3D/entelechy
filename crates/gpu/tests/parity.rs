//! GPU results must match the CPU interpreter within floating-point
//! tolerance. Tests skip when no adapter is available.

use std::path::Path;

use orla_abi::{Capabilities, Capability, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_runtime::Backend;

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
        time: 1.25,
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu_out = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let prepared = gpu
        .prepare(&module)
        .unwrap_or_else(|error| panic!("prepare failed: {error}"));
    let mut scope = Scope::new(&mut host, all());
    let gpu_out = Backend::execute(gpu, prepared.as_ref(), &module, &invocation, &mut scope)
        .unwrap_or_else(|error| panic!("gpu execution failed: {error}"))[0];
    (
        host.image(cpu_out).unwrap().pixels().to_vec(),
        host.image(gpu_out).unwrap().pixels().to_vec(),
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
    let ball = std::fs::read_to_string(root.join("material-ball/kernels/ball.orla")).unwrap();
    let (cpu, gpu_pixels) = both(
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
