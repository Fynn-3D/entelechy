use orla_abi::{Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_ir::{fuse, fuse_outputs, samples_only};

fn granted() -> orla_abi::Capabilities {
    Capability::ALL.into_iter().collect()
}

const GRADIENT: &str = "kernel image.gradient@1 { requires { field.write; time.read; } param depth: Float = 0.5; output result: Image; struct Uv { u: Float; v: Float; } fn make(c: Vec2, d: Vec2) -> Uv { return Uv { u: c.x / d.x, v: c.y / d.y }; } compute { let uv = make(coord(), domain()); let t = loop 3 with s = 0.0 { s + time() }; return vec4(uv.u, uv.v, depth + t, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { field.read; field.write; } input source: Image; param amount: Float = 1.0; output result: Image; fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; } compute { let c = sample(source); let g = luma(c); let k = if g > 0.5 { 1.0 } else { amount }; return mix(c, vec4(g, g, g, c.a), k); } }";
const PIXELATE: &str = "kernel image.pixelate@1 { requires { field.read; field.write; } input source: Image; param size: Int = 4; output result: Image; compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }";

fn run(
    module: &orla_ir::Validated,
    inputs: Vec<orla_abi::Handle>,
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

#[test]
fn producer_helpers_with_struct_arguments_survive_fusion() {
    // The consumer has structs of its own, so a producer struct the
    // consumer does not have is appended; a helper taking it as an
    // argument must shift exactly once. The element type the two share
    // is the same struct in the fused table, not a copy of it.
    let producer = compile(
        "kernel test.points@1 { requires { field.write; } output result: Rows<Point>; struct Point { p: Vec2; w: Float; } struct Wide { a: Float; b: Float; c: Float; } fn nudge(a: Wide) -> Point { return Point { p: vec2(a.a + 1.0, a.b), w: a.c }; } compute { return nudge(Wide { a: float(element()), b: 0.0, c: 2.0 }); } }",
    )
    .unwrap();
    let consumer = compile(
        "kernel test.weigh@1 { requires { field.read; field.write; } input points: Rows<Point>; output result: Rows<Float>; struct Pad { a: Float; } struct Extra { b: Int; } struct Point { p: Vec2; w: Float; } compute { let q = sample(points); return q.p.x * q.w; } }",
    )
    .unwrap();
    let fused = fuse(&consumer, 0, &producer, "points").unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(fused.structs.len(), 4, "Point is shared, Wide is added");
    assert_eq!(fused.structs[3].name, "points/Wide");
    let helper = fused
        .functions
        .iter()
        .find(|f| f.name == "points/nudge")
        .expect("the helper is carried over");
    assert_eq!(
        helper.args,
        [orla_ir::Type::Struct(3)],
        "shifted past the consumer's three structs"
    );
    let mut host = Host::new(1 << 16);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(3),
        context: Context::default(),
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldRead, Capability::FieldWrite]
            .into_iter()
            .collect(),
    );
    let output = orla_cpu::CpuBackend
        .execute(&fused, &invocation, &mut scope)
        .unwrap()[0];
    let values: Vec<f32> = (0..3)
        .map(|i| f32::from_bits(host.element(output, i).unwrap()[0]))
        .collect();
    assert_eq!(values, [2.0, 4.0, 6.0]);
}

#[test]
fn a_whole_element_the_consumer_passes_on_keeps_its_own_struct() {
    // The consumer's sample of the fused input becomes the producer's
    // result, so a whole element it hands around — one arm of an `if`,
    // the other arm a struct it builds itself — has to be the same
    // struct as its own. If the producer's copy were added to the table
    // beside it, the two arms would yield different types and the fused
    // module would not validate.
    let producer = compile(
        "kernel test.faces@1 { requires { field.write; } output result: Rows<Tri>; struct Tri { a: Int; b: Int; c: Int; } compute { let i = element(); return Tri { a: i, b: i + 1, c: i + 2 }; } }",
    )
    .unwrap();
    let consumer = compile(
        "kernel test.cull@1 { requires { field.read; field.write; } input faces: Rows<Tri>; output result: Rows<Tri>; struct Tri { a: Int; b: Int; c: Int; } compute { let t = sample(faces); return if t.a % 2 == 0 { Tri { a: 0, b: 0, c: 0 } } else { t }; } }",
    )
    .unwrap();
    let fused = fuse(&consumer, 0, &producer, "faces").unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(fused.structs.len(), 1, "one Tri, not two");

    let mut host = Host::new(1 << 16);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldRead, Capability::FieldWrite]
            .into_iter()
            .collect(),
    );
    let output = orla_cpu::CpuBackend
        .execute(&fused, &invocation, &mut scope)
        .unwrap()[0];
    let rows: Vec<[u32; 3]> = (0..4)
        .map(|i| {
            let words = host.element(output, i).unwrap();
            [words[0], words[1], words[2]]
        })
        .collect();
    assert_eq!(rows, [[0, 0, 0], [1, 2, 3], [0, 0, 0], [3, 4, 5]]);
}

const SPLIT: &str = "kernel shade.split@1 { requires { field.write; } param bias: Float = 0.25; output color: Image; output hit: Grid<Hit>; output glow: Grid<Float>; struct Hit { t: Float; id: Int; } compute { let uv = coord() / domain(); let hit = Hit { t: uv.x * 3.0 + bias, id: int(coord().y) }; if uv.x < 0.5 { return { color: vec4(uv.x, uv.y, bias, 1.0), hit: hit, glow: 1.0 }; } { glow: 0.0, hit: Hit { t: hit.t, id: -hit.id }, color: vec4(0.0, uv.y, 1.0, 1.0) } } }";
const SHOW: &str = "kernel shade.show@1 { requires { field.read; field.write; } input other: Image; input h: Grid<Hit>; input c: Image; output result: Image; struct Hit { t: Float; id: Int; } compute { let t = sample(h).t; let id = float(sample(h).id); return sample(c) * t + sample(other) + vec4(id, 0.0, 0.0, 0.0); } }";

#[test]
fn a_producer_of_several_outputs_fuses_whole_and_changes_no_bit() {
    let split = compile(SPLIT).unwrap_or_else(|error| panic!("{error}"));
    let show = compile(SHOW).unwrap_or_else(|error| panic!("{error}"));
    // `h` reads `hit` and `c` reads `color`; `glow` has no reader.
    let fused = fuse_outputs(&show, &[(1, 1), (2, 0)], &split, "split")
        .unwrap_or_else(|error| panic!("{error}"));
    let inputs: Vec<&str> = fused
        .manifest
        .inputs
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(inputs, ["other"]);
    let params: Vec<&str> = fused
        .manifest
        .params
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(params, ["split/bias"]);

    let mut host = Host::new(1 << 16);
    let other = host
        .adopt_image(
            orla_abi::Image::from_pixels(
                5,
                4,
                (0..20).map(|i| [i as f32 * 0.1, 0.5, 0.25, 0.0]).collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let produce = Invocation {
        inputs: vec![],
        params: vec![Value::Float(0.5)],
        domain: Domain::Grid(5, 4),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let made = CpuBackend.execute(&split, &produce, &mut scope).unwrap();
    assert_eq!(made.len(), 3, "one Field per output");
    let separate = run(&show, vec![other, made[1], made[0]], vec![], &mut host);
    let together = run(&fused, vec![other], vec![Value::Float(0.5)], &mut host);
    assert_eq!(separate, together, "fusion must not change a single bit");

    // Two links to one input, or a link to an output that is not there,
    // are refused rather than guessed at.
    assert!(fuse_outputs(&show, &[(1, 1), (1, 0)], &split, "split").is_err());
    assert!(fuse_outputs(&show, &[(1, 3)], &split, "split").is_err());
    // A link whose layouts differ is the scene's mismatch, not fusion's.
    let wrong = fuse_outputs(&show, &[(2, 1)], &split, "split").unwrap_err();
    assert!(wrong.mismatch, "{wrong}");
}
