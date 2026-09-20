use orla_abi::{
    AbiError, Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value,
};
use orla_compiler::compile;
use orla_cpu::{CpuBackend, ExecutionError};

fn granted() -> Capabilities {
    [Capability::FieldRead, Capability::FieldWrite]
        .into_iter()
        .collect()
}

fn near(actual: [f32; 4], expected: [f32; 4]) {
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} != {expected:?}");
    }
}

const GRAYSCALE: &str = r#"
kernel image.grayscale@1 {
    requires { field.read; field.write; }
    input source: Image;
    param amount: Float = 1.0;
    output result: Image;
    fn luma(c: Vec4) -> Float {
        return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
    }
    compute {
        let c = sample(source);
        let g = luma(c);
        return mix(c, vec4(g, g, g, c.a), amount);
    }
}
"#;

#[test]
fn grayscale_reads_and_writes_through_the_abi() {
    let module = compile(GRAYSCALE).unwrap();
    let mut host = Host::new(64);
    let source = host.create_image(2, 1).unwrap();
    host.write_pixel(source, 0, 0, [1.0, 0.0, 0.0, 1.0])
        .unwrap();
    host.write_pixel(source, 1, 0, [0.0, 1.0, 0.0, 0.5])
        .unwrap();
    let invocation = Invocation {
        inputs: vec![source],
        params: vec![Value::Float(1.0)],
        domain: Domain::Grid(2, 1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let outputs = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap();
    assert_eq!(outputs.len(), 1);
    near(
        host.read_pixel(outputs[0], 0, 0).unwrap(),
        [0.2126, 0.2126, 0.2126, 1.0],
    );
    near(
        host.read_pixel(outputs[0], 1, 0).unwrap(),
        [0.7152, 0.7152, 0.7152, 0.5],
    );

    let half = Invocation {
        params: vec![Value::Float(0.5)],
        ..invocation
    };
    let mut scope = Scope::new(&mut host, granted());
    let outputs = CpuBackend.execute(&module, &half, &mut scope).unwrap();
    near(
        host.read_pixel(outputs[0], 0, 0).unwrap(),
        [0.6063, 0.1063, 0.1063, 1.0],
    );
}

#[test]
fn execution_without_grant_is_denied_and_leaks_nothing() {
    let module = compile(GRAYSCALE).unwrap();
    let mut host = Host::new(64);
    let source = host.create_image(1, 1).unwrap();
    let invocation = Invocation {
        inputs: vec![source],
        params: vec![Value::Float(1.0)],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, [Capability::FieldWrite].into_iter().collect());
    let error = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(
        error,
        ExecutionError::Abi(AbiError::Denied(Capability::FieldRead))
    );
    assert_eq!(host.live(), 1, "failed execution released its outputs");
}

#[test]
fn invocations_are_checked_against_the_manifest() {
    let module = compile(GRAYSCALE).unwrap();
    let mut host = Host::new(64);
    let source = host.create_image(1, 1).unwrap();
    let mut scope = Scope::new(&mut host, granted());
    let bad_params = Invocation {
        inputs: vec![source],
        params: vec![Value::Int(1)],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    };
    assert!(matches!(
        CpuBackend.execute(&module, &bad_params, &mut scope),
        Err(ExecutionError::ParamType { .. })
    ));
    let no_inputs = Invocation {
        inputs: vec![],
        ..bad_params
    };
    assert_eq!(
        CpuBackend.execute(&module, &no_inputs, &mut scope),
        Err(ExecutionError::InputCount {
            expected: 1,
            actual: 0
        })
    );
}

#[test]
fn generators_use_coord_and_domain_and_fetch_clamps() {
    let gradient = compile(
        "kernel test.gradient@1 { requires { field.write; } output result: Image; compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.0, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(256);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(3, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let gradient_out = CpuBackend
        .execute(&gradient, &invocation, &mut scope)
        .unwrap()[0];
    near(
        host.read_pixel(gradient_out, 2, 1).unwrap(),
        [1.0, 1.0, 0.0, 1.0],
    );
    near(
        host.read_pixel(gradient_out, 1, 0).unwrap(),
        [0.5, 0.0, 0.0, 1.0],
    );

    let shift = compile(
        "kernel test.shift@1 { requires { field.read; field.write; } input source: Image; param offset: Int = 10; output result: Image; compute { return fetch(source, coord() + vec2(float(offset), 0.0)); } }",
    )
    .unwrap();
    let invocation = Invocation {
        inputs: vec![gradient_out],
        params: vec![Value::Int(10)],
        domain: Domain::Grid(3, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let shifted = CpuBackend.execute(&shift, &invocation, &mut scope).unwrap()[0];
    near(
        host.read_pixel(shifted, 0, 0).unwrap(),
        [1.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn integer_failures_name_the_pixel() {
    let module = compile(
        "kernel test.div@1 { requires { field.write; } param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Int(0)],
        domain: Domain::Grid(2, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "at element (0, 0): integer division by zero"
    );
    assert_eq!(host.live(), 0);
}

#[test]
fn vector_intrinsics_and_time_follow_their_contracts() {
    let module = compile(
        "kernel test.vectors@1 { requires { field.write; time.read; } param light: Vec3 = vec3(0.0, 3.0, 4.0); output result: Image; compute { let n = normalize(light); let d = dot(n, vec3(0.0, 0.0, 1.0)); let r = reflect(vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0)); let s = step(0.5, vec3(0.2, 0.5, 0.9)); let c = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); return vec4(d, r.z * s.y + s.x, c.z * time(), length(light) + pow(2.0, 3.0) + smoothstep(0.0, 1.0, 0.5)); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Vec3([0.0, 3.0, 4.0])],
        domain: Domain::Grid(1, 1),
        context: Context {
            time: 2.5,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldWrite, Capability::TimeRead]
            .into_iter()
            .collect(),
    );
    let outputs = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap();
    near(
        host.read_pixel(outputs[0], 0, 0).unwrap(),
        [0.8, 1.0, 2.5, 13.5],
    );
}

#[test]
fn parallel_rows_match_the_per_pixel_formula_and_report_the_first_failure() {
    let gradient = compile(
        "kernel test.big@1 { requires { field.write; } output result: Image; compute { let c = coord(); return vec4(c.x, c.y, c.x + c.y, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(1 << 20);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(301, 257),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&gradient, &invocation, &mut scope)
        .unwrap()[0];
    for (x, y) in [(0, 0), (300, 0), (0, 256), (300, 256), (17, 200), (299, 1)] {
        near(
            host.read_pixel(output, x, y).unwrap(),
            [x as f32, y as f32, (x + y) as f32, 1.0],
        );
    }

    let failing = compile(
        "kernel test.fail@1 { requires { field.write; } param d: Int = 0; output result: Image; compute { let c = coord(); let q = 7 / d; return vec4(float(q), c.y, 0.0, 1.0); } }",
    )
    .unwrap();
    let invocation = Invocation {
        params: vec![Value::Int(0)],
        ..invocation
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend
        .execute(&failing, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "at element (0, 0): integer division by zero"
    );
    assert_eq!(host.live(), 1);
}

#[test]
fn branches_loops_and_booleans_follow_their_contracts() {
    let module = compile(
        "kernel test.flow@1 { requires { field.write; } param invert: Bool = false; param steps: Int = 5; output result: Image;
            fn sum_to(n: Int) -> Int { return loop 100 with acc = 0 until index() >= n { acc + index() }; }
            compute {
                let c = coord();
                let side = if c.x < 1.0 { if invert { 0.25 } else { 0.75 } } else { let z = 0.5; z };
                let total = sum_to(steps);
                let nan = 0.0 / 0.0;
                let cmp = if nan < 1.0 || nan >= 1.0 { 1.0 } else { 0.0 };
                let march = loop 64 with s = vec2(0.0, 1.0) until s.y < 0.01 { vec2(s.x + 1.0, s.y * 0.5) };
                return vec4(side, float(total), cmp + float(!(c.x == 0.0)) * 8.0, march.x);
            } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let run = |host: &mut Host, invert: bool, steps: i32| {
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Bool(invert), Value::Int(steps)],
            domain: Domain::Grid(2, 1),
            context: Context::default(),
        };
        let mut scope = Scope::new(host, granted());
        CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0]
    };
    let output = run(&mut host, false, 5);
    // sum 0..5 = 10; the march halves 1.0 until < 0.01: 7 steps.
    near(
        host.read_pixel(output, 0, 0).unwrap(),
        [0.75, 10.0, 0.0, 7.0],
    );
    near(
        host.read_pixel(output, 1, 0).unwrap(),
        [0.5, 10.0, 8.0, 7.0],
    );
    let output = run(&mut host, true, 0);
    near(
        host.read_pixel(output, 0, 0).unwrap(),
        [0.25, 0.0, 0.0, 7.0],
    );
}

#[test]
fn untaken_branches_do_not_execute() {
    let module = compile(
        "kernel test.guard@1 { requires { field.write; } param d: Int = 0; output result: Image; compute { let q = if d == 0 { 0 } else { 1 / d }; return vec4(float(q), 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Int(0)],
        domain: Domain::Grid(1, 1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    near(host.read_pixel(output, 0, 0).unwrap(), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn structs_carry_several_values_through_loops_calls_and_branches() {
    let module = compile(
        "kernel test.structs@1 { requires { field.write; } output result: Image;
            struct Ray { origin: Vec3; direction: Vec3; }
            struct March { ray: Ray; t: Float; steps: Int; hit: Bool; }
            fn advance(m: March, d: Float) -> March {
                return March { ..m, t: m.t + d, steps: m.steps + 1 };
            }
            fn start(dir: Vec3) -> March {
                return March { ray: Ray { origin: vec3(0.0, 0.0, 0.0), direction: dir }, t: 0.0, steps: 0, hit: false };
            }
            compute {
                let c = coord();
                let m = loop 32 with s = start(vec3(0.0, 0.0, -1.0)) until s.hit {
                    let next = advance(s, 0.5);
                    if next.t >= c.x { March { ..next, hit: true } } else { next }
                };
                let flipped = if c.y > 0.5 { Ray { ..m.ray, origin: vec3(9.0, 9.0, 9.0) } } else { m.ray };
                return vec4(m.t, float(m.steps), flipped.origin.x + m.ray.direction.z, float(m.hit));
            } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Grid(4, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    // x = 2: advance until t >= 2.0 takes 4 steps of 0.5.
    near(
        host.read_pixel(output, 2, 0).unwrap(),
        [2.0, 4.0, -1.0, 1.0],
    );
    near(host.read_pixel(output, 2, 1).unwrap(), [2.0, 4.0, 8.0, 1.0]);
    // x = 0: the first step already reaches t >= 0.
    near(
        host.read_pixel(output, 0, 0).unwrap(),
        [0.5, 1.0, -1.0, 1.0],
    );
}

#[test]
fn fields_carry_struct_elements_with_exact_integers() {
    let seed = compile(
        "kernel test.seed@1 { requires { field.write; } output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let i = element(); return Particle { position: vec2(float(i), float(i) * 0.5), life: i * 3 + 2000000000, alive: i / 2 * 2 == i }; } }",
    )
    .unwrap();
    let step = compile(
        "kernel test.step@1 { requires { field.read; field.write; } input previous: Rows<Particle>; output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = sample(previous); let n = at(previous, element() + 1); return Particle { position: p.position + n.position, life: p.life - 2000000000 + count(previous), alive: !p.alive }; } }",
    )
    .unwrap();
    let mut host = Host::new(4096);
    let domain = Domain::Rows(8);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain,
        context: Context::default(),
    };
    let fields: Capabilities = [Capability::FieldRead, Capability::FieldWrite]
        .into_iter()
        .collect();
    let mut scope = Scope::new(&mut host, fields.clone());
    let seeded = CpuBackend.execute(&seed, &invocation, &mut scope).unwrap()[0];
    assert_eq!(host.schema(seeded).unwrap().to_string(), "f,f,i,b");
    assert_eq!(
        host.element(seeded, 3).unwrap(),
        &[3.0f32.to_bits(), 1.5f32.to_bits(), 2000000009u32, 0]
    );
    let invocation = Invocation {
        inputs: vec![seeded],
        params: vec![],
        domain,
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, fields);
    let stepped = CpuBackend.execute(&step, &invocation, &mut scope).unwrap()[0];
    let element = host.element(stepped, 2).unwrap().to_vec();
    assert_eq!(element, vec![5.0f32.to_bits(), 2.5f32.to_bits(), 14, 0]);
    let last = host.element(stepped, 7).unwrap().to_vec();
    assert_eq!(
        last,
        vec![14.0f32.to_bits(), 7.0f32.to_bits(), 29, 1],
        "at clamps to the last element"
    );

    let error = compile(
        "kernel test.bad@1 { requires { field.read; field.write; } input p: Rows<Vec2>; output result: Image; compute { return sample(p); } }",
    )
    .unwrap_err();
    assert!(error.to_string().contains("use `at`"), "{error}");
}

#[test]
fn reductions_fold_exactly_in_a_fixed_tree() {
    // Sum, maximum, and count over 1000 rows of ints and floats; the ints
    // are exact whatever the tree, the floats follow the pairwise tree.
    let source = "kernel test.stats@1 { requires { field.read; field.write; } input rows: Rows<Item>; output result: Rows<Stats>; struct Item { value: Int; weight: Float; } struct Stats { sum: Int; max: Int; weight: Float; count: Int; } compute { let it = sample(rows); return Stats { sum: it.value, max: it.value, weight: it.weight, count: 1 }; } merge(a, b) { return Stats { sum: a.sum + b.sum, max: max(a.max, b.max), weight: a.weight + b.weight, count: a.count + b.count }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(module.manifest.mode, orla_ir::Mode::Reduce);
    assert!(module.merge.is_some());
    let mut host = Host::new(1 << 16);
    let count = 1000u32;
    let words: Vec<u32> = (0..count)
        .flat_map(|i| [(i as i32 * 7 - 3000) as u32, (i as f32 * 0.25).to_bits()])
        .collect();
    let rows = host
        .adopt(
            orla_abi::FieldData::from_words(
                orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]),
                Domain::Rows(count),
                words,
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![rows],
        params: vec![],
        domain: Domain::Rows(1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.domain(output).unwrap(), Domain::Rows(1));
    let stats = host.element(output, 0).unwrap().to_vec();
    let expected_sum: i32 = (0..count as i32).map(|i| i * 7 - 3000).sum();
    assert_eq!(stats[0] as i32, expected_sum);
    assert_eq!(stats[1] as i32, (count as i32 - 1) * 7 - 3000);
    let weight = f32::from_bits(stats[2]);
    assert!((weight - 124_875.0).abs() < 8.0, "{weight}");
    assert_eq!(stats[3], count);

    // The output domain of a reduction is fixed.
    let bad = Invocation {
        inputs: vec![rows],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend.execute(&module, &bad, &mut scope).unwrap_err();
    assert!(error.to_string().contains("one element"), "{error}");

    // A single-element input folds nothing and still yields its value.
    let one = host
        .adopt(
            orla_abi::FieldData::from_words(
                orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]),
                Domain::Rows(1),
                vec![5, 1.5f32.to_bits()],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![one],
        params: vec![],
        domain: Domain::Rows(1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(
        host.element(output, 0).unwrap(),
        &[5, 5, 1.5f32.to_bits(), 1]
    );
}

#[test]
fn scans_thread_the_carried_value_in_order() {
    // A prefix sum over ints (exact) and a one-pole filter over floats,
    // both in one struct, with the element index available.
    let source = "kernel test.scan@1 { requires { field.read; field.write; } input rows: Rows<Item>; param k: Float = 0.5; output result: Rows<Run>; struct Item { value: Int; x: Float; } struct Run { total: Int; y: Float; last: Int; } scan with run = Run { total: 0, y: 0.0, last: -1 } { let it = sample(rows); return Run { total: run.total + it.value, y: run.y + (it.x - run.y) * k, last: index() }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(module.manifest.mode, orla_ir::Mode::Scan);
    assert!(module.init.is_some());
    let mut host = Host::new(1 << 16);
    let count = 7u32;
    let words: Vec<u32> = (0..count)
        .flat_map(|i| [(i as i32 + 1) as u32, 1.0f32.to_bits()])
        .collect();
    let rows = host
        .adopt(
            orla_abi::FieldData::from_words(
                orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]),
                Domain::Rows(count),
                words,
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![rows],
        params: vec![Value::Float(0.5)],
        domain: Domain::Rows(count),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.domain(output).unwrap(), Domain::Rows(count));
    let totals: Vec<i32> = (0..count)
        .map(|i| host.element(output, i).unwrap()[0] as i32)
        .collect();
    assert_eq!(totals, [1, 3, 6, 10, 15, 21, 28]);
    let ys: Vec<f32> = (0..count)
        .map(|i| f32::from_bits(host.element(output, i).unwrap()[1]))
        .collect();
    assert_eq!(ys, [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]);
    let lasts: Vec<i32> = (0..count)
        .map(|i| host.element(output, i).unwrap()[2] as i32)
        .collect();
    assert_eq!(lasts, [0, 1, 2, 3, 4, 5, 6]);

    // The domain must be the input's.
    let bad = Invocation {
        inputs: vec![rows],
        params: vec![Value::Float(0.5)],
        domain: Domain::Rows(3),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend.execute(&module, &bad, &mut scope).unwrap_err();
    assert!(
        error.to_string().contains("one element per element"),
        "{error}"
    );
}

#[test]
fn the_device_rate_reaches_kernels_as_context() {
    let module = compile(
        "kernel test.hz@1 { requires { field.write; time.read; } output result: Rows<Float>; compute { return rate(); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(1),
        context: Context {
            rate: 48000.0,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldWrite, Capability::TimeRead]
            .into_iter()
            .collect(),
    );
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(f32::from_bits(host.element(output, 0).unwrap()[0]), 48000.0);
}

#[test]
fn linear_interpolation_blends_neighbours_and_clamps() {
    // A 2x2 image: reading at pixel centres is exact, halfway blends, and
    // beyond the edge clamps.
    let module = compile(
        "kernel test.lerp@1 { requires { field.read; field.write; } input source: Image; param at: Vec2 = vec2(0.5, 0.5); output result: Rows<Vec4>; compute { return at_linear(source, at); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(1 << 12);
    let image = host.create_image(2, 2).unwrap();
    host.write_pixel(image, 0, 0, [0.0, 0.0, 0.0, 1.0]).unwrap();
    host.write_pixel(image, 1, 0, [1.0, 0.0, 0.0, 1.0]).unwrap();
    host.write_pixel(image, 0, 1, [0.0, 1.0, 0.0, 1.0]).unwrap();
    host.write_pixel(image, 1, 1, [1.0, 1.0, 0.0, 1.0]).unwrap();
    let read = |host: &mut Host, at: [f32; 2]| -> [f32; 4] {
        let invocation = Invocation {
            inputs: vec![image],
            params: vec![Value::Vec2(at)],
            domain: Domain::Rows(1),
            context: Context::default(),
        };
        let mut scope = Scope::new(host, granted());
        let output = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        let words = host.element(output, 0).unwrap();
        std::array::from_fn(|k| f32::from_bits(words[k]))
    };
    assert_eq!(read(&mut host, [0.5, 0.5]), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(read(&mut host, [1.5, 0.5]), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(read(&mut host, [1.0, 1.0]), [0.5, 0.5, 0.0, 1.0]);
    assert_eq!(read(&mut host, [1.25, 0.5]), [0.75, 0.0, 0.0, 1.0]);
    assert_eq!(
        read(&mut host, [-3.0, 9.0]),
        [0.0, 1.0, 0.0, 1.0],
        "clamped to a corner"
    );

    // Rows take a fractional position; an integer layout cannot blend.
    let rows = compile(
        "kernel test.lerp_rows@1 { requires { field.read; field.write; } input samples: Rows<Float>; output result: Rows<Float>; compute { return at_linear(samples, float(element()) + 1.0); } }",
    )
    .unwrap();
    let samples = host
        .adopt(
            orla_abi::FieldData::from_words(
                orla_abi::Schema::new(vec![orla_abi::Scalar::Float]),
                Domain::Rows(3),
                vec![0.0f32.to_bits(), 2.0f32.to_bits(), 4.0f32.to_bits()],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![samples],
        params: vec![],
        domain: Domain::Rows(3),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend.execute(&rows, &invocation, &mut scope).unwrap()[0];
    let values: Vec<f32> = (0..3)
        .map(|i| f32::from_bits(host.element(output, i).unwrap()[0]))
        .collect();
    assert_eq!(
        values,
        [1.0, 3.0, 4.0],
        "halfway between neighbours, clamped at the end"
    );
    let error = compile(
        "kernel test.bad@1 { requires { field.read; field.write; } input ids: Rows<Int>; output result: Rows<Int>; compute { return at_linear(ids, 0.5); } }",
    )
    .unwrap_err();
    assert!(error.to_string().contains("lays out as i"), "{error}");
}

#[test]
fn remainder_truncates_like_division_and_vectors_compare_whole() {
    let source = "kernel test.rem@1 { requires { field.write; } param a: Int = 7; param b: Int = 3; param x: Float = 7.5; output result: Rows<Out>; struct Out { q: Int; r: Int; f: Float; same: Bool; other: Bool; } compute { let v = vec2(a, b); return Out { q: a / b, r: a % b, f: x % 2.0, same: v == vec2(7.0, 3.0), other: vec3(1, 2, 3) != vec3(1.0, 2.0, 3.0) }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(64);
    let run = |host: &mut Host, a: i32, b: i32| -> Result<Vec<u32>, orla_cpu::ExecutionError> {
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(a), Value::Int(b), Value::Float(7.5)],
            domain: Domain::Rows(1),
            context: Context::default(),
        };
        let mut scope = Scope::new(host, granted());
        let output = CpuBackend.execute(&module, &invocation, &mut scope)?[0];
        Ok(host.element(output, 0).unwrap().to_vec())
    };
    let words = run(&mut host, 7, 3).unwrap();
    assert_eq!(words[0] as i32, 2);
    assert_eq!(words[1] as i32, 1);
    assert_eq!(f32::from_bits(words[2]), 1.5);
    assert_eq!(words[3], 1, "vec2(7, 3) widens and equals vec2(7.0, 3.0)");
    assert_eq!(words[4], 0);
    let words = run(&mut host, -7, 3).unwrap();
    assert_eq!(words[0] as i32, -2);
    assert_eq!(
        words[1] as i32, -1,
        "the remainder takes the sign of the left side"
    );
    let error = run(&mut host, 7, 0).unwrap_err();
    assert!(error.to_string().contains("division by zero"), "{error}");
}

#[test]
fn scatters_combine_into_the_target_in_element_order() {
    // Particles splat weights into a grid of floats by `sum`; an id grid
    // takes the last particle's id by `last`; a struct element is
    // combined fieldwise by `max`; addresses outside the grid are dropped.
    let source = "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input canvas: Grid<Float>; output result: Grid<Float>; struct Drop { at: Vec2; weight: Float; } struct Splat { cell: Vec2; mass: Float; } scatter into canvas with sum { let d = sample(drops); return Splat { cell: d.at, mass: d.weight }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(
        module.manifest.mode,
        orla_ir::Mode::Scatter {
            base: 1,
            combine: orla_ir::Combine::Sum
        }
    ));
    let mut host = Host::new(1 << 16);
    let drops: Vec<(f32, f32, f32)> = vec![
        (0.2, 0.7, 1.0),      // cell (0, 0)
        (2.9, 1.1, 0.5),      // cell (2, 1)
        (0.5, 0.5, 0.25),     // cell (0, 0) again
        (-0.5, 0.0, 9.0),     // outside: dropped, not clamped
        (3.0, 0.0, 9.0),      // outside on the far edge
        (1.0, f32::NAN, 9.0), // NaN: dropped
    ];
    let words: Vec<u32> = drops
        .iter()
        .flat_map(|(x, y, w)| [x.to_bits(), y.to_bits(), w.to_bits()])
        .collect();
    let schema = orla_abi::Schema::new(vec![
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
    ]);
    let rows = host
        .adopt(
            orla_abi::FieldData::from_words(schema, Domain::Rows(drops.len() as u32), words)
                .unwrap(),
        )
        .unwrap();
    let canvas_words: Vec<u32> = (0..6).map(|i| (i as f32 * 0.1).to_bits()).collect();
    let canvas = host
        .adopt(
            orla_abi::FieldData::from_words(
                orla_abi::Schema::new(vec![orla_abi::Scalar::Float]),
                Domain::Grid(3, 2),
                canvas_words,
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![rows, canvas],
        params: vec![],
        domain: Domain::Grid(3, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.domain(output).unwrap(), Domain::Grid(3, 2));
    let got: Vec<f32> = (0..6)
        .map(|i| f32::from_bits(host.element(output, i).unwrap()[0]))
        .collect();
    let expected = [0.0 + 1.0 + 0.25, 0.1, 0.2, 0.3, 0.4, 0.5 + 0.5];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{got:?} != {expected:?}");
    }
    // The target is untouched.
    assert_eq!(
        f32::from_bits(host.element(canvas, 0).unwrap()[0]),
        0.0,
        "the base input is copied, not written"
    );

    // The domain must be the target's.
    let bad = Invocation {
        inputs: vec![rows, canvas],
        params: vec![],
        domain: Domain::Grid(2, 2),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend.execute(&module, &bad, &mut scope).unwrap_err();
    assert!(error.to_string().contains("scatters into"), "{error}");

    // `last` by element order, `max` fieldwise over a struct with ints,
    // into rows addressed by Int.
    let source = "kernel test.mark@1 { requires { field.read; field.write; } input hits: Rows<Hit>; input slots: Rows<Slot>; output result: Rows<Slot>; struct Hit { slot: Int; id: Int; score: Float; } struct Slot { id: Int; score: Float; } struct Put { at: Int; value: Slot; } scatter into slots with max { let h = sample(hits); return Put { at: h.slot, value: Slot { id: h.id, score: h.score } }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let hits: Vec<(i32, i32, f32)> = vec![(1, 5, 0.5), (1, 3, 0.9), (0, 7, -1.0), (4, 1, 1.0)];
    let words: Vec<u32> = hits
        .iter()
        .flat_map(|(s, i, f)| [*s as u32, *i as u32, f.to_bits()])
        .collect();
    let hit_schema = orla_abi::Schema::new(vec![
        orla_abi::Scalar::Int,
        orla_abi::Scalar::Int,
        orla_abi::Scalar::Float,
    ]);
    let hits = host
        .adopt(orla_abi::FieldData::from_words(hit_schema, Domain::Rows(4), words).unwrap())
        .unwrap();
    let slot_schema = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
    let slots = host
        .adopt(
            orla_abi::FieldData::from_words(
                slot_schema,
                Domain::Rows(3),
                vec![
                    0,
                    0.0f32.to_bits(),
                    4,
                    0.7f32.to_bits(),
                    9,
                    0.0f32.to_bits(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![hits, slots],
        params: vec![],
        domain: Domain::Rows(3),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let element = |i: u32| {
        let w = host.element(output, i).unwrap();
        (w[0] as i32, f32::from_bits(w[1]))
    };
    assert_eq!(
        element(0),
        (7, 0.0),
        "max keeps the higher score already there"
    );
    assert_eq!(
        element(1),
        (5, 0.9),
        "max is fieldwise: id from one hit, score from another"
    );
    assert_eq!(
        element(2),
        (9, 0.0),
        "slot 4 is outside three rows and dropped"
    );

    let source = source.replace("with max", "with last");
    let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, granted());
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let w = host.element(output, 1).unwrap();
    assert_eq!(
        (w[0] as i32, f32::from_bits(w[1])),
        (3, 0.9),
        "the later element wins"
    );
    let w = host.element(output, 0).unwrap();
    assert_eq!((w[0] as i32, f32::from_bits(w[1])), (7, -1.0));
}

#[test]
fn the_window_reaches_kernels_as_context() {
    let module = compile(
        "kernel test.fit@1 { requires { field.write; input.read; } output result: Rows<Vec3>; compute { let w = window(); return vec3(w.x, w.y, scale()); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(1),
        context: Context {
            window: [1920.0, 1080.0],
            scale: 2.0,
            ..Context::default()
        },
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldWrite, Capability::InputRead]
            .into_iter()
            .collect(),
    );
    let output = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let words = host.element(output, 0).unwrap();
    assert_eq!(
        words.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
        [1920.0, 1080.0, 2.0]
    );
    // Without a window the size is zero and the scale one.
    let plain = Invocation {
        context: Context::default(),
        ..invocation
    };
    let mut scope = Scope::new(
        &mut host,
        [Capability::FieldWrite, Capability::InputRead]
            .into_iter()
            .collect(),
    );
    let output = CpuBackend.execute(&module, &plain, &mut scope).unwrap()[0];
    let words = host.element(output, 0).unwrap();
    assert_eq!(
        words.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
        [0.0, 0.0, 1.0]
    );
}

#[test]
fn bit_operators_work_on_ints_with_the_count_modulo_32() {
    let source = "kernel test.bits@1 { requires { field.write; } param a: Int = 0; param b: Int = 0; output result: Rows<Bits>; struct Bits { and: Int; or: Int; xor: Int; shl: Int; shr: Int; wide: Int; nibble: Int; } compute { return Bits { and: a & b, or: a | b, xor: a ^ b, shl: a << b, shr: a >> b, wide: 1 << 33, nibble: (a >> 4) & 15 }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let mut host = Host::new(64);
    let run = |host: &mut Host, a: i32, b: i32| -> Vec<i32> {
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(a), Value::Int(b)],
            domain: Domain::Rows(1),
            context: Context::default(),
        };
        let mut scope = Scope::new(host, granted());
        let output = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap()[0];
        host.element(output, 0)
            .unwrap()
            .iter()
            .map(|w| *w as i32)
            .collect()
    };
    assert_eq!(
        run(&mut host, 0xF0, 0x3C),
        [
            0x30,
            0xFC,
            0xCC,
            0xF0 << (0x3C % 32),
            0xF0 >> (0x3C % 32),
            2,
            0xF
        ]
    );
    // A negative number shifts right arithmetically, and its bits are two's complement.
    assert_eq!(
        run(&mut host, -8, 1),
        [-8 & 1, -8 | 1, -8 ^ 1, -16, -4, 2, 15]
    );
}
