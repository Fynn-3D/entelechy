use entelechy_abi::{Capability, Context, Domain, Host, Invocation, Scope, Value};
use entelechy_compiler::{compile, compile_unoptimized};
use entelechy_cpu::CpuBackend;
use entelechy_ir::{
    counted, counts_only, fuse, fuse_indexed, fuse_outputs, offset_read, row_reads, samples_only,
};

fn granted() -> entelechy_abi::Capabilities {
    Capability::ALL.into_iter().collect()
}

const GRADIENT: &str = "kernel image.gradient@1 { requires { field.write; time.read; } param depth: Float = 0.5; output result: Image; struct Uv { u: Float; v: Float; } fn make(c: Vec2, d: Vec2) -> Uv { return Uv { u: c.x / d.x, v: c.y / d.y }; } compute { let uv = make(coord(), domain()); let t = loop 3 with s = 0.0 { s + time() }; return vec4(uv.u, uv.v, depth + t, 1.0); } }";
const GRAYSCALE: &str = "kernel image.grayscale@1 { requires { field.read; field.write; } input source: Image; param amount: Float = 1.0; output result: Image; fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; } compute { let c = sample(source); let g = luma(c); let k = if g > 0.5 { 1.0 } else { amount }; return mix(c, vec4(g, g, g, c.a), k); } }";
const PIXELATE: &str = "kernel image.pixelate@1 { requires { field.read; field.write; } input source: Image; param size: Int = 4; output result: Image; compute { let block = float(max(size, 1)); return fetch(source, floor(coord() / block) * block); } }";

fn run(
    module: &entelechy_ir::Validated,
    inputs: Vec<entelechy_abi::Handle>,
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
    // Unoptimized, so the helpers stay for fusion to handle.
    let gradient = compile_unoptimized(GRADIENT).unwrap();
    let grayscale = compile_unoptimized(GRAYSCALE).unwrap();
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
        .adopt_image(entelechy_abi::Image::from_pixels(5, 4, intermediate).unwrap())
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
    assert!(error.to_string().contains("by address"), "{error}");
    assert!(samples_only(&compile(GRAYSCALE).unwrap(), 0));
}

#[test]
fn producer_helpers_with_struct_arguments_survive_fusion() {
    // The consumer has its own structs, so a producer struct it lacks is
    // appended and a helper taking it shifts exactly once; the shared
    // element type is one struct in the fused table.
    let producer = compile_unoptimized(
        "kernel test.points@1 { requires { field.write; } output result: Rows<Point>; struct Point { p: Vec2; w: Float; } struct Wide { a: Float; b: Float; c: Float; } fn nudge(a: Wide) -> Point { return Point { p: vec2(a.a + 1.0, a.b), w: a.c }; } compute { return nudge(Wide { a: float(element()), b: 0.0, c: 2.0 }); } }",
    )
    .unwrap();
    let consumer = compile_unoptimized(
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
        [entelechy_ir::Type::Struct(3)],
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
    let output = entelechy_cpu::CpuBackend
        .execute(&fused, &invocation, &mut scope)
        .unwrap()[0];
    let values: Vec<f32> = (0..3)
        .map(|i| f32::from_bits(host.element(output, i).unwrap()[0]))
        .collect();
    assert_eq!(values, [2.0, 4.0, 6.0]);
}

#[test]
fn a_whole_element_the_consumer_passes_on_keeps_its_own_struct() {
    // The consumer's sample becomes the producer's result, so a whole
    // element it passes (one `if` arm, beside a struct it builds) is the
    // same struct as its own, and the fused module validates.
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
    let output = entelechy_cpu::CpuBackend
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

#[test]
fn a_struct_of_another_name_with_the_same_fields_stays_apart() {
    // The consumer declares a `Face` laid out as `Tri` is, before it: the
    // producer's `Tri` is the consumer's `Tri`, not the first struct of
    // its fields, so the whole element and the one built match.
    let producer = compile(
        "kernel test.faces@1 { requires { field.write; } output result: Rows<Tri>; struct Tri { a: Int; b: Int; c: Int; } compute { let i = element(); return Tri { a: i, b: i + 1, c: i + 2 }; } }",
    )
    .unwrap();
    let consumer = compile(
        "kernel test.shift@1 { requires { field.read; field.write; } input faces: Rows<Tri>; output result: Rows<Tri>; struct Face { a: Int; b: Int; c: Int; } struct Tri { a: Int; b: Int; c: Int; } fn first(f: Face) -> Int { return f.a; } compute { let o = first(Face { a: 3, b: 0, c: 0 }); if element() < 2 { return sample(faces); } let t = sample(faces); return Tri { a: t.a + o, b: t.b + o, c: t.c + o }; } }",
    )
    .unwrap();
    let fused = fuse(&consumer, 0, &producer, "faces").unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(fused.structs.len(), 2, "Face and one Tri");

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
    let output = entelechy_cpu::CpuBackend
        .execute(&fused, &invocation, &mut scope)
        .unwrap()[0];
    let rows: Vec<[u32; 3]> = (0..4)
        .map(|i| {
            let words = host.element(output, i).unwrap();
            [words[0], words[1], words[2]]
        })
        .collect();
    assert_eq!(rows, [[0, 1, 2], [1, 2, 3], [5, 6, 7], [6, 7, 8]]);
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
            entelechy_abi::Image::from_pixels(
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

    // Two links to one input, or a link to a missing output, are refused.
    assert!(fuse_outputs(&show, &[(1, 1), (1, 0)], &split, "split").is_err());
    assert!(fuse_outputs(&show, &[(1, 3)], &split, "split").is_err());
    // A link whose layouts differ is the scene's mismatch, not fusion's.
    let wrong = fuse_outputs(&show, &[(2, 1)], &split, "split").unwrap_err();
    assert!(wrong.declines, "{wrong}");
}

const LIFT: &str = "kernel test.lift@1 { requires { field.read; field.write; time.read; } struct P { x: Float; n: Int; } input source: Rows<Float>; param k: Float = 2.0; output result: Rows<P>; compute { let v = sample(source) * k + float(element()) * 0.5 + float(domain()) + time(); return P { x: v, n: element() * 3 + count(source) }; } }";
const JOIN: &str = "kernel test.join@1 { requires { field.read; field.write; } struct P { x: Float; n: Int; } input a: Rows<P>; input b: Rows<Float>; output result: Rows<Float>; compute { let k = element(); let n = count(a); if k < n { let p = at(a, k + 1); return p.x + float(p.n); } return at(b, k - n) + float(size(a)); } }";
const TALLY: &str = "kernel test.tally@1 { requires { field.read; field.write; } input a: Rows<Float>; input b: Rows<Float>; output result: Rows<Float>; compute { return at(b, element()) + float(count(a)) * 0.5 + float(size(a)); } }";

fn rows(host: &mut Host, values: &[f32]) -> entelechy_abi::Handle {
    let schema = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]);
    let words: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    host.adopt(
        entelechy_abi::FieldData::from_values(schema, Domain::Rows(values.len() as u32), &words)
            .unwrap(),
    )
    .unwrap()
}

fn run_rows(
    module: &entelechy_ir::Validated,
    inputs: Vec<entelechy_abi::Handle>,
    params: Vec<Value>,
    count: u32,
    host: &mut Host,
) -> (entelechy_abi::Handle, Vec<u32>) {
    let invocation = Invocation {
        inputs,
        params,
        domain: Domain::Rows(count),
        context: Context {
            time: 0.25,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(host, granted());
    let output = CpuBackend.execute(module, &invocation, &mut scope).unwrap()[0];
    (output, host.data(output).unwrap().storage().to_vec())
}

#[test]
fn a_producer_read_at_an_offset_fuses_as_a_function_of_the_row_and_changes_no_bit() {
    for compiled in [compile, compile_unoptimized] {
        let lift = compiled(LIFT).unwrap();
        let join = compiled(JOIN).unwrap();
        assert!(offset_read(&join, 0), "`a` is read at element() + 1");
        assert!(offset_read(&join, 1), "`b` at element() - count(a)");
        assert!(!samples_only(&join, 0));
        let fused = fuse_indexed(&join, &[(0, 0)], &lift, "lift").unwrap();
        assert_eq!(
            fused
                .manifest
                .params
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["lift/k", "lift/#count", "lift/#size"]
        );
        let mut host = Host::new(1 << 20);
        let source = rows(&mut host, &[1.0, 2.5, -3.0, 4.0, 0.125]);
        let b = rows(&mut host, &[10.0, 20.0, 30.0]);
        // Two rows past `b`, so its last is read clamped; `a`'s last row
        // reads one past its end, clamped too.
        let (a, _) = run_rows(&lift, vec![source], vec![Value::Float(1.5)], 5, &mut host);
        let (_, separate) = run_rows(&join, vec![a, b], vec![], 10, &mut host);
        let (_, together) = run_rows(
            &fused,
            vec![b, source],
            vec![Value::Float(1.5), Value::Int(5), Value::Int(5)],
            10,
            &mut host,
        );
        assert_eq!(separate, together, "fusion must not change a single bit");
    }
}

#[test]
fn a_kernel_that_only_counts_an_input_reads_its_extent_from_params() {
    let tally = compile(TALLY).unwrap();
    assert!(counts_only(&tally, 0));
    assert!(!counts_only(&tally, 1));
    let alone = counted(&tally, &[(0, "lift".into())]).unwrap();
    assert_eq!(alone.manifest.inputs.len(), 1);
    let mut host = Host::new(1 << 20);
    let a = rows(&mut host, &[0.0; 7]);
    let b = rows(&mut host, &[1.0, 2.0, 3.0]);
    let (_, stored) = run_rows(&tally, vec![a, b], vec![], 3, &mut host);
    let (_, told) = run_rows(
        &alone,
        vec![b],
        vec![Value::Int(7), Value::Int(7)],
        3,
        &mut host,
    );
    assert_eq!(stored, told);
    assert!(counted(&tally, &[(1, "b".into())]).is_err(), "`b` is read");
}

#[test]
fn only_one_offset_of_the_element_outside_loops_is_an_offset_read() {
    let reads = |body: &str| {
        let source = format!(
            "kernel test.read@1 {{ requires {{ field.read; field.write; }} input a: Rows<Float>; output result: Rows<Float>; compute {{ {body} }} }}"
        );
        offset_read(&compile_unoptimized(&source).unwrap(), 0)
    };
    assert!(reads("return at(a, element());"));
    assert!(reads("let n = count(a) / 2; return at(a, n + element());"));
    assert!(!reads("return at(a, element() * 2);"), "not an offset");
    assert!(
        !reads("return at(a, element()) + at(a, element() + 1);"),
        "two rows"
    );
    assert!(!reads("return sample(a);"), "a sample");
    assert!(
        !reads("return loop 2 with s = 0.0 { s + at(a, element()) };"),
        "read in a loop"
    );
    assert!(
        !reads("return at(a, int(at(a, 0)));"),
        "an address read from the input"
    );
}

const PICK: &str = "kernel test.pick@1 { requires { field.read; field.write; } struct P { x: Float; n: Int; } input a: Rows<P>; input b: Rows<Float>; output result: Rows<Float>; compute { let k = element(); let far = (k * 7 + 3) % count(a); let p = at(a, far); let led = at(a, p.n - 20); return p.x - led.x * 0.5 + at(b, k) + float(at(a, k * k).n); } }";

#[test]
fn a_producer_read_wherever_its_reader_is_led_fuses_as_a_function_of_the_row_and_changes_no_bit() {
    for compiled in [compile, compile_unoptimized] {
        let lift = compiled(LIFT).unwrap();
        let pick = compiled(PICK).unwrap();
        // Three places: one worked out from the element, one from what
        // the first read held, one past the rows, clamped.
        assert_eq!(row_reads(&pick, 0), Some(3));
        assert!(!offset_read(&pick, 0));
        let fused = fuse_indexed(&pick, &[(0, 0)], &lift, "lift").unwrap();
        let mut host = Host::new(1 << 20);
        let source = rows(&mut host, &[1.0, 2.5, -3.0, 4.0, 0.125, 7.0, -0.5]);
        let b = rows(&mut host, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let (a, _) = run_rows(&lift, vec![source], vec![Value::Float(1.5)], 7, &mut host);
        let (_, separate) = run_rows(&pick, vec![a, b], vec![], 6, &mut host);
        let (_, together) = run_rows(
            &fused,
            vec![b, source],
            vec![Value::Float(1.5), Value::Int(7), Value::Int(7)],
            6,
            &mut host,
        );
        assert_eq!(separate, together, "fusion must not change a single bit");
    }
}

#[test]
fn rows_read_in_the_entry_outside_loops_are_counted_by_the_places_that_read_them() {
    let reads = |body: &str| {
        let source = format!(
            "kernel test.read@1 {{ requires {{ field.read; field.write; }} input a: Rows<Float>; output result: Rows<Float>; fn third(a: Rows<Float>) -> Float {{ return at(a, 3); }} compute {{ {body} }} }}"
        );
        row_reads(&compile_unoptimized(&source).unwrap(), 0)
    };
    assert_eq!(reads("return at(a, element());"), Some(1));
    assert_eq!(reads("return at(a, element() * 2);"), Some(1));
    assert_eq!(
        reads("return at(a, element()) + at(a, int(at(a, 0)));"),
        Some(3)
    );
    assert_eq!(reads("return sample(a);"), None, "a sample");
    assert_eq!(reads("return float(count(a));"), None, "nothing read");
    assert_eq!(
        reads("return loop 2 with s = 0.0 { s + at(a, element()) };"),
        None,
        "read in a loop"
    );
    assert_eq!(reads("return third(a);"), None, "read in a function");
}
