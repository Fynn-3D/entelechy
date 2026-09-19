use orla_abi::{Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_ir::{fuse, samples_only};

fn granted() -> orla_abi::Capabilities {
    Capability::ALL.into_iter().collect()
}

const GRADIENT: &str = "kernel image.gradient@1 { requires { field.write; time.read; } param depth: Float = 0.5; output result: Image; struct Uv { u: Float; v: Float; } fn make(c: Vec2, d: Vec2) -> Uv { return Uv { u: c.x / d.x, v: c.y / d.y }; } compute { let uv = make(coord(), domain()); let t = loop 3 with s = 0.0 { s + time() }; return vec4(uv.u, uv.v, depth + t, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { field.read; field.write; } input source: Image; param amount: Float = 1.0; output result: Image; fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; } compute { let c = sample(source); let g = luma(c); let k = if g > 0.5 { 1.0 } else { amount }; return mix(c, vec4(g, g, g, c.a), k); } }";
const PIXELATE: &str = "kernel image.pixelate@1 { requires { field.read; field.write; } input source: Image; param size: Int = 4; output result: Image; compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }";

fn run(
    module: &orla_ir::Validated,
    inputs: Vec<orla_abi::ImageHandle>,
    params: Vec<Value>,
    host: &mut Host,
) -> Vec<[f32; 4]> {
    let invocation = Invocation {
        inputs,
        params,
        domain: Domain::Grid(5, 4),
        context: Context {
            time: 0.25,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(host, granted());
    let output = CpuBackend.execute(module, &invocation, &mut scope).unwrap()[0];
    host.image(output).unwrap().pixels().to_vec()
}

#[test]
fn fused_pointwise_chain_is_bit_identical_to_the_materialized_chain() {
    let gradient = compile(GRADIENT).unwrap();
    let grayscale = compile(GRAYSCALE).unwrap();
    let fused = fuse(&grayscale, 0, &gradient, "source").unwrap();
    let manifest = &fused.manifest;
    assert_eq!(manifest.inputs.len(), 0);
    assert_eq!(
        manifest
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["amount", "source/depth"]
    );
    assert!(manifest.requires.contains(Capability::TimeRead));
    assert_eq!(fused.structs[0].name, "source/Uv");
    assert_eq!(fused.functions.len(), 2);

    let mut host = Host::new(4096);
    let intermediate = run(&gradient, vec![], vec![Value::Float(0.5)], &mut host);
    let handle = host
        .adopt_image(orla_abi::Image::from_pixels(5, 4, intermediate).unwrap())
        .unwrap();
    let separate = run(&grayscale, vec![handle], vec![Value::Float(0.3)], &mut host);
    let together = run(
        &fused,
        vec![],
        vec![Value::Float(0.3), Value::Float(0.5)],
        &mut host,
    );
    assert_eq!(separate, together, "fusion must not change a single bit");
}

#[test]
fn addressed_inputs_do_not_fuse() {
    let gradient = compile(GRADIENT).unwrap();
    let pixelate = compile(PIXELATE).unwrap();
    assert!(!samples_only(&pixelate, 0));
    let error = fuse(&pixelate, 0, &gradient, "source").unwrap_err();
    assert!(error.to_string().contains("address or size"), "{error}");
    assert!(samples_only(&compile(GRAYSCALE).unwrap(), 0));
}
