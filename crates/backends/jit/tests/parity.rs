//! The JIT against the CPU interpreter: the same words and the same
//! integer failures over a map Kernel's whole vocabulary.

use entelechy_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use entelechy_compiler::compile;
use entelechy_cpu::{ArithmeticError, CpuBackend, ExecutionError};
use entelechy_jit::JitBackend;
use entelechy_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

/// Runs a module on both backends and returns each one's output words.
fn both(
    source: &str,
    inputs: &[(entelechy_abi::Schema, Domain, Vec<u32>)],
    params: Vec<Value>,
    domain: Domain,
    context: Context,
) -> (Vec<u32>, Vec<u32>) {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 22);
    let handles: Vec<_> = inputs
        .iter()
        .map(|(schema, domain, words)| {
            host.adopt(
                entelechy_abi::FieldData::from_storage(schema.clone(), *domain, words.clone())
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    let invocation = Invocation {
        inputs: handles,
        params,
        domain,
        context,
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let cpu_words = host.data(cpu).unwrap().storage().to_vec();
    let prepared = jit
        .prepare(&module)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out = jit
        .execute(prepared.as_ref(), &module, &invocation, &mut scope)
        .unwrap()[0];
    let jit_words = host.data(out).unwrap().storage().to_vec();
    (cpu_words, jit_words)
}

#[test]
fn the_vocabulary_matches_the_interpreter_bit_for_bit() {
    let source = "kernel test.all@1 { requires { field.read; field.write; input.read; time.read; } input grid: Grid<Cell>; input rows: Rows<Float>; param k: Float = 0.3; param n: Int = 7; param v: Vec3 = vec3(1.0, 2.0, 3.0); output result: Rows<Out>;
        struct Cell { a: Float; b: Int; on: Bool; }
        struct Out { f: Float; i: Int; v: Vec3; b: Bool; }
        struct Acc { sum: Float; count: Int; }
        fn twist(p: Vec3, t: Float) -> Vec3 { return vec3(p.x * cos(t) - p.y * sin(t), p.x * sin(t) + p.y * cos(t), p.z); }
        fn score(c: Cell) -> Acc { return Acc { sum: c.a * 2.0, count: c.b + 1 }; }
        compute {
            let i = element();
            let x = float(i);
            let c = at(grid, vec2(x * 0.7, x * 0.3));
            let s = score(c);
            let lin = at_linear(rows, x * 0.37 + 0.2);
            let w = twist(v + vec3(x, 0.0, 0.0), k * x);
            let walked = loop 12 with acc = Acc { sum: 0.0, count: 0 } until acc.count > n {
                Acc { sum: acc.sum + sin(acc.sum + x) * 0.5 + float(index()), count: acc.count + 1 }
            };
            let m = if c.on { mix(w, v, k) } else { normalize(w) * length(v) };
            let bits = (i << 3) ^ (i >> 1) | (n & 5) ^ ((0 - i) >>> 3);
            let dense = match i % 7 { 0 => 1, 1 | 2 => { let two = i * 2; two }, 3 => 30, 5 => 50, _ => -1 };
            let sparse = match i * 1000 { 0 => 7, 3000 => 8, 9000 => 9, -1 => 10, _ => 11 };
            let trig = tan(x * 0.1) + asin(fract(x * 0.13)) + acos(fract(x * 0.17)) + atan(x) + atan2(x, 3.0) + exp(k * x) + log(x + 1.0) + pow(x + 1.0, 1.5) + sqrt(x);
            let shaped = smoothstep(2.0, 9.0, x) + step(4.0, x) + floor(x * 0.9) + ceil(x * 0.2) + round(x * 0.5) + sign(x - 5.0) + abs(x - 6.0) + clamp(x, 2.0, 8.0) + max(x, 3.0) + min(x, 4.0);
            let ints = clamp(i * 3, 2, 20) + max(i, 4) + min(i, 3) + abs(i - 4) + sign(i - 2) + (i % 3) + (i / 2) + int(x * 1.5) + int(c.on) + (add_wrap(0x7FFFFFF0, i) >>> 28) + (sub_wrap(0x80000000, i) >>> 28) + (mul_wrap(0x40000001, i) >>> 29);
            let r = reflect(normalize(w), normalize(vec3(0.0, 1.0, 0.0))) + cross(w, v);
            let v4 = vec4(x, 1.0, 2.0, 3.0) * 0.5 + vec4(1, 2, 3, 4);
            let same = vec2(x, 1.0) == vec2(float(i), 1.0) && v != vec3(0.0, 0.0, 0.0) && !(x > 100.0);
            return Out { f: s.sum + lin + walked.sum + trig + shaped + v4.w + dot(m, r) + time() + pointer().x + float(key(65)) + float(pointer_down()) + rate() + window().y + scale(), i: s.count + walked.count + bits + ints + int(same) + dense * 3 + sparse, v: m + r, b: same || c.on };
        } }";
    let cell = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
        entelechy_abi::Scalar::Bool,
    ]);
    let grid: Vec<u32> = (0..48)
        .flat_map(|i| {
            [
                ((i as f32) * 0.21).sin().to_bits(),
                (i % 5) as u32,
                (i % 3 == 0) as u32,
            ]
        })
        .collect();
    let rows: Vec<u32> = (0..16)
        .map(|i| ((i as f32) * 0.5).cos().to_bits())
        .collect();
    let mut context = Context {
        time: 1.5,
        rate: 48000.0,
        window: [640.0, 480.0],
        scale: 2.0,
        ..Context::default()
    };
    context.input.pointer = [3.0, 4.0];
    context.input.pointer_down = true;
    context.input.set_key(65, true);
    let (cpu, jit) = both(
        source,
        &[
            (cell, Domain::Grid(8, 6), grid),
            (
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
                Domain::Rows(16),
                rows,
            ),
        ],
        vec![
            Value::Float(0.3),
            Value::Int(7),
            Value::Vec3([1.0, 2.0, 3.0]),
        ],
        Domain::Rows(40),
        context,
    );
    assert_eq!(cpu.len(), 40 * 6);
    for (k, (a, b)) in cpu.iter().zip(&jit).enumerate() {
        assert_eq!(
            a,
            b,
            "word {k} (element {}, field word {}): cpu {} jit {}",
            k / 6,
            k % 6,
            f32::from_bits(*a),
            f32::from_bits(*b)
        );
    }
}

/// A float address floors and clamps each axis into the Field, NaN as
/// zero: an element, a part, a grid and a volume, at the edges.
#[test]
fn float_addresses_clamp_as_the_interpreters_do() {
    let source = "kernel test.edges@1 { requires { field.read; field.write; } input cells: Grid<Cell>; input cube: Volume<Int>; output result: Rows<Int>;
        struct Cell { a: Float; b: Int; }
        fn edge(k: Int, zero: Float) -> Float {
            return match k % 12 { 0 => zero / zero, 1 => -1.0 / zero, 2 => 1.0 / zero, 3 => -zero, 4 => -0.5, 5 => 0.0, 6 => 0.99, 7 => 4.0, 8 => 4.5, 9 => 1000000000000000000000000000000.0, 10 => -1000000000000000000000000000000.0, _ => 2.5 };
        }
        compute {
            let i = element();
            let zero = float(i) - float(i);
            let x = edge(i, zero);
            let y = edge(i / 12, zero);
            let z = edge(i / 144, zero);
            let whole = at(cells, vec2(x, y));
            return whole.b + at(cells, vec2(y, x)).b * 100 + int(whole.a) * 10000 + at(cube, vec3(x, y, z)) * 1000000;
        } }";
    let cell = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    let cells: Vec<u32> = (0..20)
        .flat_map(|i| [((i % 7) as f32).to_bits(), i as u32])
        .collect();
    let cube: Vec<u32> = (0..18).map(|i| i as u32 + 1).collect();
    let (cpu, jit) = both(
        source,
        &[
            (cell, Domain::Grid(5, 4), cells),
            (
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]),
                Domain::Volume(3, 3, 2),
                cube,
            ),
        ],
        vec![],
        Domain::Rows(12 * 12 * 12),
        Context::default(),
    );
    assert_eq!(cpu, jit);
}

#[test]
fn integer_failures_are_reported_at_their_element() {
    let source = "kernel test.div@1 { requires { field.write; } param d: Int = 1; output result: Rows<Int>; compute { let i = element(); return (100 / (d - i)) + (if i == 5 { 2147483647 + i } else { 0 }); } }";
    let module = compile(source).unwrap();
    let jit = JitBackend::new().unwrap();
    let run = |d: i32| {
        let mut host = Host::new(1 << 16);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(d)],
            domain: Domain::Rows(8),
            context: Context::default(),
        };
        let prepared = jit.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let outcome = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
        (outcome.map(|_| ()), host.live())
    };
    // d = 3: element 3 divides by zero before element 5 overflows.
    let (error, live) = run(3);
    assert_eq!(
        error.unwrap_err(),
        ExecutionError::Arithmetic {
            x: 3,
            y: 0,
            error: ArithmeticError::DivisionByZero
        }
    );
    assert_eq!(live, 0, "a failed run leaks nothing");
    // d = 100: no zero divisor; element 5 overflows.
    let (error, _) = run(100);
    assert_eq!(
        error.unwrap_err(),
        ExecutionError::Arithmetic {
            x: 5,
            y: 0,
            error: ArithmeticError::Overflow
        }
    );
}

/// Runs a module on both backends and returns every output's words.
fn both_all(
    source: &str,
    inputs: &[(entelechy_abi::Schema, Domain, Vec<u32>)],
    domain: Domain,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 22);
    let handles: Vec<_> = inputs
        .iter()
        .map(|(schema, domain, words)| {
            host.adopt(
                entelechy_abi::FieldData::from_storage(schema.clone(), *domain, words.clone())
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
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap();
    let cpu_words: Vec<Vec<u32>> = cpu
        .iter()
        .map(|handle| host.data(*handle).unwrap().storage().to_vec())
        .collect();
    let prepared = jit
        .prepare(&module)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out = jit
        .execute(prepared.as_ref(), &module, &invocation, &mut scope)
        .unwrap();
    let jit_words = out
        .iter()
        .map(|handle| host.data(*handle).unwrap().storage().to_vec())
        .collect();
    (cpu_words, jit_words)
}

#[test]
fn scans_and_their_memories_match_the_interpreter() {
    // A machine walks a grid memory and tallies into a rows memory, reading
    // its carried value and step index, with a loop inside a step; writes
    // outside both memories are dropped.
    let source = "kernel test.walk@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory screen: Grid<Int>; memory tally: Rows<Cell>; output result: Rows<Vec2>; struct Cell { hits: Int; last: Float; }
        fn flip(s: Grid<Int>, p: Vec2, seen: Int) -> Int { return put(s, p, (1 - seen) >>> 0); }
        scan over ticks with at_ = vec2(3.0, 2.0) {
            let seen = at(screen, at_);
            flip(screen, at_, seen);
            let t = at(tally, seen);
            put(tally, seen, Cell { hits: t.hits + 1, last: float(index()) });
            put(screen, vec2(-1.0, 0.0), 9);
            put(tally, 7, Cell { hits: 0, last: 0.0 });
            let spun = loop 3 with acc = seen { acc + index() + at(ticks, index()) };
            let next = vec2(float((int(at_.x) + 3 + spun * 5) % 16), float((int(at_.y) + 1 + seen) % 8));
            return next;
        } }";
    let steps = 300u32;
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let cells = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Int,
        entelechy_abi::Scalar::Float,
    ]);
    let (cpu, jit) = both_all(
        source,
        &[
            (
                ints.clone(),
                Domain::Rows(steps),
                (0..steps).map(|i| i % 3).collect(),
            ),
            (ints, Domain::Grid(16, 8), vec![0; 128]),
            (
                cells,
                Domain::Rows(2),
                vec![0, 0.0f32.to_bits(), 0, 0.0f32.to_bits()],
            ),
        ],
        Domain::Rows(steps),
    );
    assert_eq!(cpu.len(), 3);
    assert_eq!(cpu, jit);
    assert!(cpu[1].contains(&1), "the walk left marks");
    assert_eq!(cpu[2][0] + cpu[2][2], steps, "every step tallied once");
}

#[test]
fn a_scan_step_that_fails_names_its_element() {
    let source = "kernel test.count@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan over rows with acc = 1000 { return acc / sample(rows); } }";
    let module = compile(source).unwrap();
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 16);
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let rows = host
        .adopt(
            entelechy_abi::FieldData::from_storage(ints, Domain::Rows(6), vec![2, 5, 1, 0, 3, 3])
                .unwrap(),
        )
        .unwrap();
    let invocation = Invocation {
        inputs: vec![rows],
        params: vec![],
        domain: Domain::Rows(6),
        context: Context::default(),
    };
    let prepared = jit.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let outcome = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
    assert_eq!(
        outcome.unwrap_err(),
        ExecutionError::Arithmetic {
            x: 3,
            y: 0,
            error: ArithmeticError::DivisionByZero
        }
    );
    assert_eq!(host.live(), 1, "the input alone survives a failed run");
}

#[test]
fn reductions_fold_in_the_interpreters_tree() {
    // Ints exactly and floats bit for bit: both fold the same pairs in the
    // same order; an odd count runs the tail path.
    let source = "kernel test.stats@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Stats>; struct Stats { bright: Int; sum: Vec4; max: Float; } compute over source { let c = sample(source); let l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return Stats { bright: if l > 0.5 { 1 } else { 0 }, sum: c, max: l }; } merge(a, b) { return Stats { bright: a.bright + b.bright, sum: a.sum + b.sum, max: max(a.max, b.max) }; } }";
    let (width, height) = (37u32, 23u32);
    let pixels: Vec<u32> = (0..width * height)
        .flat_map(|i| {
            let f = i as f32 / (width * height) as f32;
            [f, (f * 7.0).fract(), 1.0 - f, 1.0].map(f32::to_bits)
        })
        .collect();
    let (cpu, jit) = both_all(
        source,
        &[(
            entelechy_abi::Schema::pixel(),
            Domain::Grid(width, height),
            pixels,
        )],
        Domain::Rows(1),
    );
    assert_eq!(cpu, jit);
    assert_eq!(cpu[0].len(), 6);
    assert!(cpu[0][0] > 0 && f32::from_bits(cpu[0][5]) > 0.5);
}

#[test]
fn scatters_apply_in_element_order_with_the_same_rule() {
    // Several drops per cell of a small grid, so float order matters; an
    // address outside is dropped.
    let source = "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input canvas: Grid<Cell>; output result: Grid<Cell>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Cell { mass: Float; top: Int; } struct Splat { cell: Vec2; value: Cell; } scatter over drops into canvas with sum { let d = sample(drops); return Splat { cell: d.at, value: Cell { mass: d.weight, top: d.id } }; } }";
    let count = 2000u32;
    let drops: Vec<u32> = (0..count)
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
    let cell = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    let (cpu, jit) = both_all(
        source,
        &[
            (drop_schema, Domain::Rows(count), drops),
            (cell, Domain::Grid(8, 4), vec![0; 64]),
        ],
        Domain::Rows(count),
    );
    assert_eq!(cpu, jit);
    let mass: f32 = cpu[0].chunks_exact(2).map(|c| f32::from_bits(c[0])).sum();
    assert!(mass > 50.0, "the drops inside landed: {mass}");
    // The other rules and an integer target: min, max and last over rows.
    for rule in ["min", "max", "last"] {
        let source = format!(
            "kernel test.rows@1 {{ requires {{ field.read; field.write; }} input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>; struct Pair {{ at: Int; value: Int; }} scatter over keys into slots with {rule} {{ let k = sample(keys); return Pair {{ at: k % 9 - 1, value: k * 7 % 13 - 6 }}; }} }}"
        );
        let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
        let (cpu, jit) = both_all(
            &source,
            &[
                (ints.clone(), Domain::Rows(100), (0..100).collect()),
                (ints, Domain::Rows(8), vec![0; 8]),
            ],
            Domain::Rows(100),
        );
        assert_eq!(cpu, jit, "{rule}");
    }
}

#[test]
fn a_merge_or_a_combine_that_overflows_is_reported() {
    let jit = JitBackend::new().unwrap();
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    // The fold overflows at the second level.
    let source = "kernel test.total@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute over rows { return sample(rows); } merge(a, b) { return a + b; } }";
    let module = compile(source).unwrap();
    let mut host = Host::new(1 << 16);
    let rows = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(4),
                vec![0x4000_0000, 0x4000_0000, 1, 1],
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
    let prepared = jit.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let outcome = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
    assert_eq!(
        outcome.unwrap_err(),
        ExecutionError::Arithmetic {
            x: 0,
            y: 0,
            error: ArithmeticError::Overflow
        }
    );
    // An integer sum into one slot whose whole total does not fit names
    // the slot.
    let source = "kernel test.pile@1 { requires { field.read; field.write; } input rows: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>; struct Pair { at: Int; value: Int; } scatter over rows into slots with sum { return Pair { at: 0, value: sample(rows) }; } }";
    let module = compile(source).unwrap();
    let mut host = Host::new(1 << 16);
    let rows = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(4),
                vec![0x4000_0000, 0x3000_0000, 0x4000_0000, 1],
            )
            .unwrap(),
        )
        .unwrap();
    let slots = host
        .adopt(entelechy_abi::FieldData::from_storage(ints, Domain::Rows(2), vec![0; 2]).unwrap())
        .unwrap();
    let invocation = Invocation {
        inputs: vec![rows, slots],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let prepared = jit.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let outcome = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
    assert_eq!(
        outcome.unwrap_err(),
        ExecutionError::Arithmetic {
            x: 0,
            y: 0,
            error: ArithmeticError::Overflow
        }
    );
    assert_eq!(host.live(), 2, "a failed run leaks nothing");
}

#[test]
fn a_merge_may_read_inputs_params_and_the_context() {
    let source = "kernel test.fold@1 { requires { field.read; field.write; time.read; } input xs: Rows<Float>; input weights: Rows<Float>; param k: Float = 0.5; output result: Rows<Float>;
        fn first(w: Rows<Float>) -> Float { return at(w, 0); }
        compute over xs { return sample(xs) * at(weights, element()); }
        merge(a, b) { return a + b + at(weights, 1) * k + first(weights) + time(); } }";
    let floats = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]);
    let count = 37u32;
    let (cpu, jit) = both(
        source,
        &[
            (
                floats.clone(),
                Domain::Rows(count),
                (0..count).map(|i| (i as f32 * 0.5).to_bits()).collect(),
            ),
            (
                floats,
                Domain::Rows(count),
                (0..count)
                    .map(|i| (1.0 + (i % 3) as f32).to_bits())
                    .collect(),
            ),
        ],
        vec![Value::Float(0.25)],
        Domain::Rows(1),
        Context {
            time: 2.0,
            ..Context::default()
        },
    );
    assert_eq!(cpu, jit);
    let plain: f32 = (0..count)
        .map(|i| i as f32 * 0.5 * (1.0 + (i % 3) as f32))
        .sum();
    assert!((f32::from_bits(cpu[0]) - (plain + 36.0 * 3.5)).abs() < 1e-2);
}

/// Inlining and hoisting keep the failing element on both backends; a
/// loop that never runs fails at nothing, since failing work never hoists.
#[test]
fn a_failure_names_its_element_through_a_call_and_a_loop() {
    let source = "kernel test.deep@1 { requires { field.write; } param d: Int = 1; output result: Rows<Int>; fn risky(a: Int, b: Int) -> Int { return a / b; } compute { let i = element(); let times = if i % 2 == 0 { 3 } else { 0 }; return loop times with acc = 0 { acc + risky(100, d - i) }; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let run = |backend: &dyn Backend, d: i32| {
        let mut host = Host::new(1 << 16);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![Value::Int(d)],
            domain: Domain::Rows(8),
            context: Context::default(),
        };
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let outcome = backend.execute(prepared.as_ref(), &module, &invocation, &mut scope);
        outcome.map(|outputs| scope.storage(outputs[0]).unwrap().to_vec())
    };
    // Element 4 divides by zero in an inlined function inside a loop: both
    // backends blame element 4.
    let expected = ExecutionError::Arithmetic {
        x: 4,
        y: 0,
        error: ArithmeticError::DivisionByZero,
    };
    assert_eq!(run(&CpuBackend, 4).unwrap_err(), expected);
    assert_eq!(run(&jit, 4).unwrap_err(), expected);
    // Element 5's division sits in a loop that runs no times: nothing
    // fails.
    let cpu = run(&CpuBackend, 5).unwrap();
    let jitted = run(&jit, 5).unwrap();
    assert_eq!(cpu, jitted);
    assert_eq!(cpu[5], 0, "the loop that never ran carries its initial");
    assert_eq!(cpu[4], 300, "three times a hundred over one");
}

#[test]
fn emitted_pairs_land_as_the_interpreters_do() {
    // Each element emits as many pairs as its key says, into slots its key
    // picks, across worker threads on both backends; the returned pair
    // lands too.
    let count = 50_000u32;
    for rule in ["sum", "max", "min"] {
        let source = format!(
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
        );
        let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
        let (cpu, jit) = both_all(
            &source,
            &[
                (ints.clone(), Domain::Rows(count), (0..count).collect()),
                (ints, Domain::Rows(97), vec![0; 97]),
            ],
            Domain::Rows(count),
        );
        assert_eq!(cpu, jit, "{rule}");
    }
}

/// A pair wider than every input's element goes through a scratch slot
/// sized for it: long keys from nested loops under a branch all land.
#[test]
fn keys_emitted_from_nested_loops_land_as_the_interpreters_do() {
    let count = 20_000u32;
    let source = "kernel test.cover@1 { requires { field.read; field.write; }
        input keys: Rows<Int>; input canvas: Grid<Long>; output result: Grid<Long>;
        struct Hit { pixel: Vec2; key: Long; }
        scatter over keys into canvas with min {
            let t = element();
            let k = sample(keys);
            let none = Hit { pixel: vec2(-1.0, -1.0), key: 0l };
            let _rows = loop k % 4 + 1 with dy = 0 {
                let _cols = loop k % 3 + 1 with dx = 0 {
                    let z = float((k + dx * 7 + dy * 3) % 1000) / 1000.0;
                    let _hit = if (dx + dy + k) % 3 != 0 {
                        emit(Hit { pixel: vec2(float((k + dx) % 64), float((k / 64 + dy) % 48)), key: (long(bits(z)) << 32) | long(t) })
                    } else { none };
                    dx + 1
                };
                dy + 1
            };
            return none;
        } }";
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let longs = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Long; 2]);
    let (cpu, jit) = both_all(
        source,
        &[
            (
                ints,
                Domain::Rows(count),
                (0..count)
                    .map(|i| i.wrapping_mul(2_654_435_761) >> 12)
                    .collect(),
            ),
            (
                longs,
                Domain::Grid(64, 48),
                (0..64 * 48)
                    .flat_map(|_| [u32::MAX, i32::MAX as u32])
                    .collect(),
            ),
        ],
        Domain::Rows(count),
    );
    let differ = cpu[0]
        .chunks(2)
        .zip(jit[0].chunks(2))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differ, 0, "{differ} pixels differ");
}

/// Parts applied by separate workers land as the interpreter's single
/// pass, for order-dependent rules (float sum, `last`) too; an overflow
/// names the element the sequential order fails at.
#[test]
fn a_target_cut_into_parts_sees_its_pairs_in_order() {
    let count = 200_000u32;
    let keys: Vec<u32> = (0..count)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 8)
        .collect();
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let cells = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Float,
        entelechy_abi::Scalar::Int,
    ]);
    for rule in ["sum", "last", "min", "max"] {
        let source = format!(
            "kernel test.parts@1 {{ requires {{ field.read; field.write; }}
            input keys: Rows<Int>; input cells: Rows<Cell>; output result: Rows<Cell>;
            struct Cell {{ mass: Float; id: Int; }}
            struct Pair {{ at: Int; value: Cell; }}
            scatter over keys into cells with {rule} {{
                let k = sample(keys);
                return Pair {{ at: k % 5003 - 2, value: Cell {{ mass: float(k % 977) * 0.013 + 0.1, id: element() % 1000 }} }};
            }} }}"
        );
        let (cpu, jit) = both_all(
            &source,
            &[
                (ints.clone(), Domain::Rows(count), keys.clone()),
                (cells.clone(), Domain::Rows(5000), vec![0; 10_000]),
            ],
            Domain::Rows(count),
        );
        assert_eq!(cpu, jit, "{rule}");
    }
    // An integer sum overflowing at several places names the earliest
    // element.
    let module = compile(
        "kernel test.over@1 { requires { field.read; field.write; }
         input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
         struct Pair { at: Int; value: Int; }
         scatter over keys into slots with sum {
             let k = sample(keys);
             return Pair { at: k % 4999, value: 1000000000 };
         } }",
    )
    .unwrap();
    let mut host = Host::new(1 << 22);
    let inputs = vec![
        host.adopt(
            entelechy_abi::FieldData::from_storage(ints.clone(), Domain::Rows(count), keys)
                .unwrap(),
        )
        .unwrap(),
        host.adopt(
            entelechy_abi::FieldData::from_storage(ints, Domain::Rows(4999), vec![0; 4999])
                .unwrap(),
        )
        .unwrap(),
    ];
    let invocation = Invocation {
        inputs,
        params: vec![],
        domain: Domain::Rows(count),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap_err();
    let jit = JitBackend::new().unwrap();
    let prepared = jit.prepare(&module).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let got = jit
        .execute(prepared.as_ref(), &module, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(cpu.to_string(), got.to_string());
}

/// A reduction's and a prefix's values held in Host memory count against
/// the budget, the same on both.
#[test]
fn values_a_reduction_holds_past_the_hosts_room_fail_the_run_on_both() {
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    for (source, runs) in [
        (
            "kernel test.total@1 { requires { field.read; field.write; } input keys: Rows<Int>; output result: Rows<Int>; compute over keys { return sample(keys); } merge(a, b) { return a + b; } }",
            1,
        ),
        (
            "kernel test.ranks@1 { requires { field.read; field.write; } input keys: Rows<Int>; output result: Rows<Int>; prefix over keys with sum { return sample(keys); } }",
            1000,
        ),
    ] {
        let module = compile(source).unwrap();
        let run = |budget: usize, backend: &dyn Backend| {
            let mut host = Host::new(budget);
            let keys = host
                .adopt(
                    entelechy_abi::FieldData::from_storage(
                        ints.clone(),
                        Domain::Rows(1000),
                        vec![1; 1000],
                    )
                    .unwrap(),
                )
                .unwrap();
            let invocation = Invocation {
                inputs: vec![keys],
                params: vec![],
                domain: Domain::Rows(runs),
                context: Context::default(),
            };
            let prepared = backend.prepare(&module).unwrap();
            let mut scope = Scope::new(&mut host, all());
            backend
                .execute(prepared.as_ref(), &module, &invocation, &mut scope)
                .map(|out| host.data(out[0]).unwrap().storage().to_vec())
        };
        let jit = JitBackend::new().unwrap();
        // A thousand keys, and room for a thousand more words: not the
        // values a reduction holds, nor a prefix's with its output.
        for backend in [&CpuBackend as &dyn Backend, &jit] {
            let error = run(2000, backend).unwrap_err();
            assert!(
                matches!(error, entelechy_rules::ExecutionError::OverRoom { .. }),
                "{source}: {error}"
            );
        }
        assert_eq!(run(8000, &CpuBackend).unwrap(), run(8000, &jit).unwrap());
    }
}

/// A scatter's pairs held in Host memory count against the budget.
#[test]
fn pairs_past_the_hosts_room_fail_the_run_on_both() {
    let module = compile(
        "kernel test.flood@1 { requires { field.read; field.write; }
         input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
         struct Pair { at: Int; value: Int; }
         scatter over keys into slots with sum {
             let k = sample(keys);
             let _fan = loop 1000 with j = 0 {
                 emit(Pair { at: (k + j) % 64, value: 1 });
                 j + 1
             };
             return Pair { at: -1, value: 0 };
         } }",
    )
    .unwrap();
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let run = |budget: usize, backend: &dyn Backend| {
        let mut host = Host::new(budget);
        let inputs = vec![
            host.adopt(
                entelechy_abi::FieldData::from_storage(
                    ints.clone(),
                    Domain::Rows(200),
                    (0..200).collect(),
                )
                .unwrap(),
            )
            .unwrap(),
            host.adopt(
                entelechy_abi::FieldData::from_storage(ints.clone(), Domain::Rows(64), vec![0; 64])
                    .unwrap(),
            )
            .unwrap(),
        ];
        let invocation = Invocation {
            inputs,
            params: vec![],
            domain: Domain::Rows(200),
            context: Context::default(),
        };
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        backend
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .map(|out| host.data(out[0]).unwrap().storage().to_vec())
    };
    let jit = JitBackend::new().unwrap();
    // Two hundred elements of a thousand pairs, three words each.
    for backend in [&CpuBackend as &dyn Backend, &jit] {
        let error = run(100_000, backend).unwrap_err();
        assert!(
            matches!(error, entelechy_rules::ExecutionError::OverRoom { .. }),
            "{error}"
        );
    }
    assert_eq!(
        run(100_000, &CpuBackend).unwrap_err().to_string(),
        run(100_000, &jit).unwrap_err().to_string()
    );
    let cpu = run(4_000_000, &CpuBackend).unwrap();
    assert_eq!(cpu, run(4_000_000, &jit).unwrap());
    assert_eq!(cpu.iter().sum::<u32>(), 200_000);
}

/// A module on both backends over one rows input: each one's words, or
/// its failure.
fn outcomes(
    source: &str,
    schema: entelechy_abi::Schema,
    words: Vec<u32>,
    domain: Domain,
) -> [Result<Vec<u32>, ExecutionError>; 2] {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let count = words.len() / schema.value_words();
    let jit = JitBackend::new().unwrap();
    let prepared = jit.prepare(&module).unwrap();
    let run = |on_jit: bool| {
        let mut host = Host::new(1 << 24);
        let rows = host
            .adopt(
                entelechy_abi::FieldData::from_storage(
                    schema.clone(),
                    Domain::Rows(count as u32),
                    words.clone(),
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = Invocation {
            inputs: vec![rows],
            params: vec![],
            domain,
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out = if on_jit {
            jit.execute(prepared.as_ref(), &module, &invocation, &mut scope)
        } else {
            CpuBackend.execute(&module, &invocation, &mut scope)
        }?[0];
        Ok(host.data(out).unwrap().storage().to_vec())
    };
    [run(false), run(true)]
}

/// A prefix of one `Int` and of a record, one whose partial totals
/// overflow where the walk does not, a late overflow, and a reduction.
#[test]
fn folds_split_across_workers_are_the_walks() {
    let n = 400_000usize;
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let mixed = (0..n)
        .map(|i| ((i as i64 * 2_654_435_761 % 20_001) - 10_000) as i32 as u32)
        .collect::<Vec<_>>();
    for rule in ["sum", "min", "max"] {
        let source = format!(
            "kernel test.running@1 {{ requires {{ field.read; field.write; }} input rows: Rows<Int>; output result: Rows<Int>; prefix over rows with {rule} {{ return sample(rows); }} }}"
        );
        let [cpu, jit] = outcomes(&source, ints.clone(), mixed.clone(), Domain::Rows(n as u32));
        assert!(cpu.is_ok(), "{rule}: {cpu:?}");
        assert_eq!(cpu, jit, "an Int prefix with {rule}");
    }
    let record = entelechy_abi::Schema::new(vec![
        entelechy_abi::Scalar::Int,
        entelechy_abi::Scalar::Long,
        entelechy_abi::Scalar::Long,
    ]);
    let words = (0..n)
        .flat_map(|i| {
            let long = (i as i64 - 200_000) * 1_000_003;
            [mixed[i], long as u32, (long >> 32) as u32]
        })
        .collect::<Vec<_>>();
    let [cpu, jit] = outcomes(
        "kernel test.pair@1 { requires { field.read; field.write; } input rows: Rows<P>; output result: Rows<P>; struct P { i: Int; l: Long; } prefix over rows with sum { return sample(rows); } }",
        record,
        words,
        Domain::Rows(n as u32),
    );
    assert!(cpu.is_ok(), "{cpu:?}");
    assert_eq!(cpu, jit, "a record's prefix");
    let sum = "kernel test.running@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; prefix over rows with sum { return sample(rows); } }";
    // The first ten rows owe two billion, the last two repay three: a
    // worker's partial total overflows, the walk never does.
    let mut paid = vec![0u32; n];
    for row in paid.iter_mut().take(10) {
        *row = (-200_000_000i32) as u32;
    }
    paid[n - 2] = 1_500_000_000;
    paid[n - 1] = 1_500_000_000;
    let [cpu, jit] = outcomes(sum, ints.clone(), paid, Domain::Rows(n as u32));
    assert_eq!(cpu.as_ref().unwrap()[n - 1], 1_000_000_000);
    assert_eq!(
        cpu, jit,
        "a run total that overflows where the walk does not"
    );
    // A late overflow is named at the row the walk meets it.
    let mut late = vec![1u32; n];
    late[n - 1000] = i32::MAX as u32;
    let [cpu, jit] = outcomes(sum, ints.clone(), late.clone(), Domain::Rows(n as u32));
    assert!(
        matches!(
            cpu,
            Err(ExecutionError::Arithmetic {
                error: ArithmeticError::Overflow,
                ..
            })
        ),
        "{cpu:?}"
    );
    assert_eq!(cpu, jit, "a late overflow");
    let total = "kernel test.total@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute over rows { return sample(rows); } merge(a, b) { return a + b; } }";
    let [cpu, jit] = outcomes(total, ints.clone(), mixed, Domain::Rows(1));
    assert!(cpu.is_ok(), "{cpu:?}");
    assert_eq!(cpu, jit, "a reduction");
    let [cpu, jit] = outcomes(total, ints, late, Domain::Rows(1));
    assert!(cpu.is_err(), "{cpu:?}");
    assert_eq!(cpu, jit, "a reduction that overflows");
}

#[test]
fn at_linear_blends_an_element_wider_than_64_words() {
    let source = "kernel test.wide@1 { requires { field.read; field.write; } input rows: Rows<Wide>; output result: Rows<Float>;
        struct Wide { a0: Vec4; a1: Vec4; a2: Vec4; a3: Vec4; a4: Vec4; a5: Vec4; a6: Vec4; a7: Vec4; a8: Vec4; a9: Vec4; a10: Vec4; a11: Vec4; a12: Vec4; a13: Vec4; a14: Vec4; a15: Vec4; a16: Vec4; }
        compute {
            let w = at_linear(rows, float(element()) * 0.61 + 0.3);
            return dot(w.a0, vec4(1.0, 2.0, 3.0, 1.0)) + dot(w.a1, vec4(1.0, 2.0, 3.0, 2.0)) + dot(w.a2, vec4(1.0, 2.0, 3.0, 3.0)) + dot(w.a3, vec4(1.0, 2.0, 3.0, 4.0)) + dot(w.a4, vec4(1.0, 2.0, 3.0, 5.0)) + dot(w.a5, vec4(1.0, 2.0, 3.0, 6.0)) + dot(w.a6, vec4(1.0, 2.0, 3.0, 7.0)) + dot(w.a7, vec4(1.0, 2.0, 3.0, 8.0)) + dot(w.a8, vec4(1.0, 2.0, 3.0, 9.0)) + dot(w.a9, vec4(1.0, 2.0, 3.0, 10.0)) + dot(w.a10, vec4(1.0, 2.0, 3.0, 11.0)) + dot(w.a11, vec4(1.0, 2.0, 3.0, 12.0)) + dot(w.a12, vec4(1.0, 2.0, 3.0, 13.0)) + dot(w.a13, vec4(1.0, 2.0, 3.0, 14.0)) + dot(w.a14, vec4(1.0, 2.0, 3.0, 15.0)) + dot(w.a15, vec4(1.0, 2.0, 3.0, 16.0)) + dot(w.a16, vec4(1.0, 2.0, 3.0, 17.0));
        } }";
    let wide = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 68]);
    let rows: Vec<u32> = (0..6 * 68)
        .map(|i| ((i as f32) * 0.37).sin().to_bits())
        .collect();
    let (cpu, jit) = both(
        source,
        &[(wide, Domain::Rows(6), rows)],
        vec![],
        Domain::Rows(9),
        Context::default(),
    );
    assert_eq!(cpu.len(), 9);
    assert_eq!(cpu, jit);
}

/// The check runs before a word is read.
#[test]
fn an_input_of_the_wrong_layout_is_refused_by_every_backend() {
    let module = compile(
        "kernel t.wide@1 { requires { field.read; field.write; } struct W { a: Vec4; b: Vec4; } input rows: Rows<W>; output result: Rows<Float>; compute { return sample(rows).b.w; } }",
    )
    .unwrap();
    let mut host = Host::new(1 << 16);
    let narrow = host
        .create(
            entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float]),
            Domain::Rows(4),
        )
        .unwrap();
    let grid = host
        .create(
            entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 8]),
            Domain::Grid(2, 2),
        )
        .unwrap();
    let jit = JitBackend::new().unwrap();
    let prepared = jit.prepare(&module).unwrap();
    for (input, said) in [
        (narrow, "expects an element of"),
        (grid, "but is bound to a Grid"),
    ] {
        let invocation = Invocation {
            inputs: vec![input],
            params: vec![],
            domain: Domain::Rows(4),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let error = jit
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .unwrap_err();
        assert!(error.to_string().contains(said), "{error}");
        let mut scope = Scope::new(&mut host, all());
        let error = CpuBackend
            .execute(&module, &invocation, &mut scope)
            .unwrap_err();
        assert!(error.to_string().contains(said), "{error}");
    }
}

#[test]
fn values_past_the_stack_are_left_to_the_interpreter() {
    // Two values of 240 KiB subscripted, which the compiled frame holds
    // in slots of their own: past the stack the compiled code may take,
    // so the JIT refuses the module and the interpreter, holding values
    // on the heap, runs it.
    let source = "kernel test.deep@1 { requires { field.write; } output result: Rows<Int>;
        compute {
            let a = [[element(); 4096]; 15];
            let b = [[element() * 2; 4096]; 15];
            return a[element() % 15][3] + b[(element() + 1) % 15][5] - element() * 2;
        } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 22);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.data(cpu).unwrap().storage(), [0, 1, 2, 3]);
    match jit.prepare(&module) {
        // The refusal names where the stack goes, as compiled: the
        // function, and the type whose values are the most of it.
        Err(reason) => assert!(
            reason.contains("bytes of stack") && reason.contains("of `[[Int; 4096]; 15]`"),
            "{reason}"
        ),
        Ok(prepared) => {
            let mut scope = Scope::new(&mut host, all());
            let out = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
            panic!("the JIT ran the module: {:?}", out.map(|_| ()));
        }
    }
    assert!(jit.declines(&module).is_some());
}

#[test]
fn a_narrow_output_is_charged_the_words_it_is_computed_in() {
    // 4096 bytes kept, 4096 words computed: a budget of 2048 words holds
    // the one and not the other, so the run is refused before it starts.
    let source = "kernel test.bytes@1 { requires { field.write; } output result: Rows<Int as u8>;
        compute { return element() % 256; } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(4096),
        context: Context::default(),
    };
    let jit = JitBackend::new().unwrap();
    let prepared = jit.prepare(&module).unwrap();
    for backend in ["cpu", "jit"] {
        let mut host = Host::new(2048);
        let mut scope = Scope::new(&mut host, all());
        let ran = match backend {
            "cpu" => CpuBackend.execute(&module, &invocation, &mut scope),
            _ => jit.execute(prepared.as_ref(), &module, &invocation, &mut scope),
        };
        let error = ran.expect_err(backend).to_string();
        assert!(error.contains("budget"), "{backend}: {error}");
        assert_eq!(host.used_words(), 0, "{backend}");
    }
}

/// Runs `source` on the interpreter and, when the JIT takes it, on the
/// JIT; answers the interpreter's rows and the JIT's (`None`: refused,
/// with its reason).
fn on_both(source: &str) -> (Vec<u32>, Result<Vec<u32>, String>) {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 22);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let cpu = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap()[0];
    let expected = host.data(cpu).unwrap().storage().to_vec();
    let compiled = jit.prepare(&module).map(|prepared| {
        let mut scope = Scope::new(&mut host, all());
        let out = jit
            .execute(prepared.as_ref(), &module, &invocation, &mut scope)
            .unwrap()[0];
        host.data(out).unwrap().storage().to_vec()
    });
    (expected, compiled)
}

/// The rows of 4096 words a long function's array holds: a frame of
/// about 130 KiB as compiled; for the WebAssembly code, bounded by the
/// estimate, about 160 KiB estimated.
const ROWS: usize = if cfg!(target_arch = "wasm32") { 2 } else { 8 };

/// A function `name` too long to be written out where it is called: a
/// frame of [`ROWS`] x 4096 words subscripted, after `first` (another
/// function's call, or the argument) and a chain of arithmetic.
fn long_function(name: &str, first: &str) -> String {
    let chain: String = (1..120)
        .map(|i| format!("let v{i} = (v{} * 3 + {i}) % 1000; ", i - 1))
        .collect();
    format!(
        "fn {name}(k: Int) -> Int {{ let v0 = {first}; {chain}let a = [[v119; 4096]; {ROWS}]; return a[k % {ROWS}][1]; }}\n"
    )
}

#[test]
fn functions_called_one_after_another_share_the_stack() {
    // Three large frames, called in turn: one is on the stack
    // at a time, so the JIT takes the module and agrees with the
    // interpreter.
    let source = format!(
        "{}{}{}kernel test.turns@1 {{ requires {{ field.write; }} output result: Rows<Int>;
        compute {{ return one(element()) + two(element()) + three(element()); }} }}",
        long_function("one", "k"),
        long_function("two", "k + 1"),
        long_function("three", "k + 2"),
    );
    let (expected, jit) = on_both(&source);
    assert_eq!(jit, Ok(expected));
}

#[test]
fn functions_nested_past_the_stack_are_refused() {
    // The same frames, each calling the next: all on the stack at once,
    // past what compiled code may take; the refusal names the chain.
    let source = format!(
        "{}{}{}kernel test.nested@1 {{ requires {{ field.write; }} output result: Rows<Int>;
        compute {{ return three(element()); }} }}",
        long_function("one", "k"),
        long_function("two", "one(k)"),
        long_function("three", "two(k)"),
    );
    let (_, jit) = on_both(&source);
    match jit {
        Err(reason) => assert!(
            reason.contains("`three`")
                && reason.contains("calling `two`")
                && reason.contains("calling `one`"),
            "{reason}"
        ),
        Ok(rows) => panic!("the JIT took the module: {rows:?}"),
    }
}

// The WebAssembly code is bounded by the estimate, which counts each
// value's words.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn values_built_from_others_share_their_words() {
    // Forty values of a struct of 1025 words, each built from the one
    // before: counted word for word they would pass the stack, but each
    // shares the words it keeps of the one before, so the compiled frame
    // is small and the JIT takes the module, unasked and asked.
    let updates: String = (1..=40)
        .map(|i| format!("let b{i} = Big {{ ..b{}, n: b{}.n + {i} }}; ", i - 1, i - 1))
        .collect();
    let source = format!(
        "kernel test.updates@1 {{ requires {{ field.write; }} output result: Rows<Int>;
        struct Big {{ cells: [Int; 1024]; n: Int; }}
        compute {{ let b0 = Big {{ cells: [element(); 1024], n: 0 }}; {updates}return b40.n + b40.cells[element() % 4]; }} }}"
    );
    let (expected, jit) = on_both(&source);
    assert_eq!(jit, Ok(expected));
    let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(JitBackend::new().unwrap().declines(&module), None);
}

/// A memory carried from run to run, as an evaluation carries a scan's
/// state: each run's memory is the last one's, which is released after.
/// The JIT's memory is made from the bound one on the pages its writes
/// changed; it holds the interpreter's words every run, and the Host
/// finds it the same as the last exactly when its words are.
#[test]
fn a_memory_carried_from_run_to_run_matches_the_interpreter_and_compares_by_its_pages() {
    let module = compile(
        "kernel test.carry@1 { requires { field.read; field.write; }
         input frame: Rows<Int>; memory cells: Rows<Int>; output result: Rows<Int>;
         scan over frame with y = 0 {
             let k = at(frame, 0);
             let n = count(cells);
             // Frames of three kinds: two cells changed, a cell written with
             // what it holds, nothing written.
             let w = if k % 3 == 0 { put(cells, (k * 1237) % n, k) + put(cells, (k * 7919 + 4095) % n, k + 1) }
                 else if k % 3 == 1 { put(cells, 5, at(cells, 5)) } else { 0 };
             return w;
         } }",
    )
    .unwrap();
    let ints = entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Int]);
    let len = entelechy_abi::PAGE_WORDS as u32 * 5 + 3;
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(len as usize * 64);
    let mut cells = host
        .adopt(
            entelechy_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(len),
                vec![0; len as usize],
            )
            .unwrap(),
        )
        .unwrap();
    let mut expected = vec![0u32; len as usize];
    for k in 0..12u32 {
        let frame = host
            .adopt(
                entelechy_abi::FieldData::from_storage(ints.clone(), Domain::Rows(1), vec![k])
                    .unwrap(),
            )
            .unwrap();
        let invocation = Invocation {
            inputs: vec![frame, cells],
            params: vec![],
            domain: Domain::Rows(1),
            context: Context::default(),
        };
        let out = {
            let prepared = jit.prepare(&module).unwrap();
            let mut scope = Scope::new(&mut host, all());
            jit.execute(prepared.as_ref(), &module, &invocation, &mut scope)
                .unwrap()
        };
        // The interpreter on the same memory, as the reference.
        let reference = {
            let mut scope = Scope::new(&mut host, all());
            CpuBackend
                .execute(&module, &invocation, &mut scope)
                .unwrap()
        };
        let carried = host.data(out[1]).unwrap().storage().to_vec();
        assert_eq!(
            carried,
            host.data(reference[1]).unwrap().storage(),
            "run {k}"
        );
        let before = expected.clone();
        if k % 3 == 0 {
            expected[((k * 1237) % len) as usize] = k;
            expected[((k * 7919 + 4095) % len) as usize] = k + 1;
        }
        assert_eq!(carried, expected, "run {k}");
        assert_eq!(
            host.same(cells, out[1]),
            Some(before == expected),
            "run {k}"
        );
        for handle in [frame, cells, out[0], reference[0], reference[1]] {
            host.release(handle).unwrap();
        }
        cells = out[1];
    }
}

#[test]
fn two_functions_of_one_name_each_keep_their_tables() {
    // A module may hold two functions of one name (one made for each call
    // that reads a wide input through); each defines its constant arrays
    // under its own place in the module. A function long enough to stay
    // a function, called twice, its twin added beside it.
    let steps: String = (0..60)
        .map(|i| format!("let v{} = sin(x * {}.0 + v{}) * 0.5; ", i + 1, i + 2, i))
        .collect();
    let source = format!(
        "kernel test.tables@1 {{ requires {{ field.write; }} output result: Rows<Float>;
            compute {{ let k = element(); return if k == 0 {{ picked(k) }} else {{ picked(k + 1) * 2.0 }}; }} }}
        fn picked(k: Int) -> Float {{ let table = [3.0, 5.0, 7.0, 11.0]; let x = table[k - 4 * (k / 4)]; let v0 = x; {steps} return v60; }}"
    );
    let mut plain = compile(&source)
        .unwrap_or_else(|error| panic!("{error}"))
        .into_module();
    assert_eq!(
        plain.functions.len(),
        1,
        "the function stays a function of its own"
    );
    let twin = plain.functions[0].clone();
    plain.functions.push(twin);
    let module = entelechy_ir::validate(plain).unwrap_or_else(|error| panic!("{error}"));
    let alone = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    let run = |module: &entelechy_ir::Validated| {
        let jit = JitBackend::new().unwrap();
        let prepared = jit
            .prepare(module)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(4),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out = jit
            .execute(prepared.as_ref(), module, &invocation, &mut scope)
            .unwrap()[0];
        host.data(out).unwrap().storage().to_vec()
    };
    assert_eq!(run(&module), run(&alone));
}

/// A group's lanes meet at each exchange, in a loop a param bounds and
/// under an `if` on a param, and the last group, past the domain's end,
/// runs whole: its lanes past the end give at exchanges and write
/// nothing.
#[test]
fn groups_meet_at_exchanges_as_the_interpreter_does() {
    let source = "kernel test.group@1 { requires { field.write; } param rounds: Int = 3; param flip: Bool = true; output result: Rows<IVec2>;
        group 8 {
            let i = element();
            let total = loop rounds with s = i { s + exchange(s, lane() ^ (1 << index())) };
            let turned = if flip { exchange(i * 10, 7 - lane()) } else { i };
            return ivec2(total, turned);
        } }";
    let (cpu, jit) = both(
        source,
        &[],
        vec![Value::Int(3), Value::Bool(true)],
        Domain::Rows(20),
        Context::default(),
    );
    assert_eq!(cpu, jit);
    let words: Vec<i32> = cpu.iter().map(|w| *w as i32).collect();
    let at = |e: usize| (words[2 * e], words[2 * e + 1]);
    // Each group's sum, past-end lanes 20..24 counted; lane l takes lane
    // 7 - l's element.
    assert_eq!(at(0), (28, 70));
    assert_eq!(at(9), (92, 140));
    assert_eq!(at(16), (156, 230));
    assert_eq!(at(19), (156, 200));
    assert_eq!(words.len(), 40);
    // Accounted as the Kernel runtime accounts it, a param's count among
    // its loops: a stopped run's work is given back, the finished one's
    // counted, and the words are the same.
    let counted = entelechy_ir::account(&compile(source).unwrap()).unwrap();
    assert!(counted.accounted());
    let jit_backend = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 20);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Int(3), Value::Bool(true)],
        domain: Domain::Rows(20),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let out = CpuBackend
        .execute(&counted, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.data(out).unwrap().storage(), &cpu[..]);
    let prepared = jit_backend.prepare(&counted).unwrap();
    let mut scope = Scope::new(&mut host, all());
    let out = jit_backend
        .execute(prepared.as_ref(), &counted, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.data(out).unwrap().storage(), &cpu[..]);
}

/// What fails after an exchange fails at its element, as in the
/// interpreter; a group's failure is named among the lanes that failed
/// before the first exchange the group met with a failure, the lowest; a
/// lane past the domain's end runs to give at exchanges, and what fails
/// there fails the run.
#[test]
fn a_failure_after_an_exchange_names_its_element() {
    let cases = [
        // Lane 2 takes element 3: three billion overflows.
        (
            "group 4 { return exchange(element(), lane() + 1) * 1000000000; }",
            8,
            ExecutionError::Arithmetic {
                x: 2,
                y: 0,
                error: ArithmeticError::Overflow,
            },
        ),
        // Lane 7 fails before the first exchange, lane 0 after the last:
        // the group's failure is lane 7's, the first it met with.
        (
            "group 8 { let a = if lane() == 7 { 2147483647 + element() % 8 } else { 0 };
                let b = exchange(exchange(a, lane()), lane());
                return b + if lane() == 0 { 2147483647 + element() % 8 + 1 } else { 0 }; }",
            8,
            ExecutionError::Arithmetic {
                x: 7,
                y: 0,
                error: ArithmeticError::Overflow,
            },
        ),
        // Element 7, past the end of six, is placed as the domain lays
        // its elements out: (1, 1).
        (
            "group 4 { let x = exchange(element(), lane() + 1); return 100 / (element() - 7) + x; }",
            6,
            ExecutionError::Arithmetic {
                x: 1,
                y: 1,
                error: ArithmeticError::DivisionByZero,
            },
        ),
    ];
    let jit = JitBackend::new().unwrap();
    for (body, count, expected) in cases {
        let source = format!(
            "kernel test.group_fail@1 {{ requires {{ field.write; }} output result: Rows<Int>; {body} }}"
        );
        let module = compile(&source).unwrap();
        let invocation = Invocation {
            inputs: vec![],
            params: vec![],
            domain: Domain::Rows(count),
            context: Context::default(),
        };
        let mut host = Host::new(1 << 16);
        let mut scope = Scope::new(&mut host, all());
        let cpu = CpuBackend.execute(&module, &invocation, &mut scope);
        let prepared = jit.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let out = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
        assert_eq!(cpu.unwrap_err(), expected, "{body}");
        assert_eq!(out.unwrap_err(), expected, "{body}");
        assert_eq!(host.live(), 0, "a failed run leaks nothing");
    }
}

/// A counted group whose lanes run unequal work between exchanges: the
/// lane that passes its budget first, in the stretch the group runs
/// before an exchange, names the failure, as the interpreter checks every
/// lane's budget in a run the group stops too.
#[test]
fn a_counted_group_fails_where_a_lane_first_passes_its_budget() {
    let source = "kernel test.counted@1 { requires { field.write; } budget 50000; param rounds: Int = 3; output result: Rows<Int>;
        group 8 {
            return loop rounds with s = element() {
                let spent = loop 1000 with a = 0 until index() >= (lane() + 1) * 100 { a + 1 };
                (s + exchange(s, lane() ^ 1)) % 9973 + spent
            };
        } }";
    let module = entelechy_ir::account(&compile(source).unwrap()).unwrap();
    let jit = JitBackend::new().unwrap();
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
        let out = backend.execute(prepared.as_ref(), &module, &invocation, &mut scope)?[0];
        Ok::<Vec<u32>, ExecutionError>(host.data(out).unwrap().storage().to_vec())
    };
    assert_eq!(run(&jit, 3).unwrap(), run(&CpuBackend, 3).unwrap());
    // Lane 7 passes its budget in round 63, lane 4 would in round 100.
    let expected = ExecutionError::Exhausted {
        x: 7,
        y: 0,
        budget: 50000,
    };
    assert_eq!(run(&CpuBackend, 100).unwrap_err(), expected);
    assert_eq!(run(&jit, 100).unwrap_err(), expected);
    // Lane 1 overflows later in the stretch lane 7 passed its budget in:
    // the lowest that failed in it is named.
    let source = source.replace(
        "(s + exchange(s, lane() ^ 1)) % 9973 + spent",
        "let boom = if lane() == 1 && index() == 63 { 2147483647 + s % 2 + 1 } else { 0 }; (s + exchange(s, lane() ^ 1)) % 9973 + spent + boom",
    );
    let module = entelechy_ir::account(&compile(&source).unwrap()).unwrap();
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Int(100)],
        domain: Domain::Rows(36),
        context: Context::default(),
    };
    let expected = ExecutionError::Arithmetic {
        x: 1,
        y: 0,
        error: ArithmeticError::Overflow,
    };
    for backend in [&CpuBackend as &dyn Backend, &jit] {
        let mut host = Host::new(1 << 20);
        let prepared = backend.prepare(&module).unwrap();
        let mut scope = Scope::new(&mut host, all());
        let failed = backend.execute(prepared.as_ref(), &module, &invocation, &mut scope);
        assert_eq!(failed.unwrap_err(), expected, "{}", backend.name());
    }
}

#[test]
fn inline_linear_sampling_matches_the_interpreter_at_every_edge() {
    let source = "kernel test.linear_edges@1 {
        requires { field.read; field.write; }
        input rows: Rows<Vec2>;
        input grid: Grid<Vec4 as unorm8>;
        input vol: Volume<Vec3>;
        input coords: Rows<Vec3>;
        output out: Rows<IVec4>;
        fn probe(r: Vec2, g: Vec4, v: Vec3) -> IVec4 {
            return ivec4(bits(r.x), bits(r.y), bits(g.x + g.w), bits(v.x + v.y + v.z));
        }
        compute {
            let p = sample(coords);
            let r = at_linear(rows, p.x);
            let g = at_linear(grid, p.xy);
            let v = at_linear(vol, p);
            return probe(r, g, v);
        }
    }";
    // Place NaN and infinity in upper cells so zero-weight corners at 0.5
    // prove untouched neighbors never poison the blend.
    let rows: Vec<u32> = [
        (1.25f32, -0.5f32),
        (f32::NAN, f32::INFINITY),
        (3.75, 2.0),
        (-4.0, 0.25),
    ]
    .into_iter()
    .flat_map(|(a, b)| [a.to_bits(), b.to_bits()])
    .collect();
    let narrow_schema = entelechy_abi::Schema::pixel()
        .with_encoding(entelechy_abi::Encoding::Unorm8)
        .unwrap();
    let grid_values: Vec<u32> = (0..16 * 4)
        .map(|i| ((i as f32 * 0.17).sin() * 0.5 + 0.5).to_bits())
        .collect();
    let grid_storage = entelechy_abi::FieldData::from_values(
        narrow_schema.clone(),
        Domain::Grid(4, 4),
        &grid_values,
    )
    .unwrap()
    .storage()
    .to_vec();
    let mut vol: Vec<u32> = (0..27 * 3)
        .map(|i| ((i as f32 * 0.31).cos() * 2.5).to_bits())
        .collect();
    // Cell (1, 0, 0) holds NaN; at vec3(0.5, 0.5, 0.5) its weight is 0.0.
    vol[3] = f32::NAN.to_bits();
    let edge_coords: &[[f32; 3]] = &[
        [0.5, 0.5, 0.5],
        [-0.0, 0.0, -2.5],
        [0.85, 1.35, 2.15],
        [1.5, 2.5, 0.5],
        [99.0, 99.0, 99.0],
        [f32::NAN, 1.25, 0.75],
        [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::INFINITY, 1.0, 1.0],
    ];
    let coords: Vec<u32> = edge_coords
        .iter()
        .flat_map(|c| c.map(f32::to_bits))
        .collect();
    let (cpu, jit) = both(
        source,
        &[
            (
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 2]),
                Domain::Rows(4),
                rows,
            ),
            (narrow_schema, Domain::Grid(4, 4), grid_storage),
            (
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 3]),
                Domain::Volume(3, 3, 3),
                vol,
            ),
            (
                entelechy_abi::Schema::new(vec![entelechy_abi::Scalar::Float; 3]),
                Domain::Rows(edge_coords.len() as u32),
                coords,
            ),
        ],
        vec![],
        Domain::Rows(edge_coords.len() as u32),
        Context::default(),
    );
    assert_eq!(cpu, jit);
}

#[test]
fn multi_page_frames_probe_and_match_the_interpreter() {
    // 2,048 Vec4s = 32 KiB stack slot (> 4 KiB guard page), exercising inline stack probes.
    let source = "kernel test.probestack@1 {
        requires { field.write; }
        output out: Rows<Float>;
        compute {
            let e = element();
            let table = [each 2048 {
                let k = index() + e;
                vec4(float(k), float(k * 2), float(k * 3), float(k * 4))
            }];
            return table[(e * 131) % 2048].z + table[2047].x;
        }
    }";
    let (cpu, jit) = both(source, &[], vec![], Domain::Rows(8), Context::default());
    assert_eq!(cpu, jit);
}

#[test]
fn a_prepared_module_compiles_on_finish_or_first_run() {
    let source = "kernel test.lazy@1 {
        requires { field.write; }
        output out: Rows<Int>;
        compute { return element() * 3 + 1; }
    }";
    let module = compile(source).unwrap();
    let jit = JitBackend::new().unwrap();

    let by_finish = jit.prepare(&module).unwrap();
    assert!(!jit.finished(by_finish.as_ref()));
    jit.finish(by_finish.as_ref());
    assert!(jit.finished(by_finish.as_ref()));

    let by_run = jit.prepare(&module).unwrap();
    assert!(!jit.finished(by_run.as_ref()));
    let mut host = Host::new(1 << 16);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(4),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let out = jit
        .execute(by_run.as_ref(), &module, &invocation, &mut scope)
        .unwrap()[0];
    assert!(jit.finished(by_run.as_ref()));
    assert_eq!(host.data(out).unwrap().storage(), [1, 4, 7, 10]);
}

#[test]
fn signed_zero_constants_stay_distinct_across_shared_code_cache() {
    let pos = compile(
        "kernel test.pos_zero@1 {
            requires { field.write; }
            output out: Rows<Float>;
            compute { return 0.0; }
        }",
    )
    .unwrap();
    let neg = compile(
        "kernel test.neg_zero@1 {
            requires { field.write; }
            output out: Rows<Float>;
            compute { return -0.0; }
        }",
    )
    .unwrap();
    let jit = JitBackend::new().unwrap();
    let prep_pos = jit.prepare(&pos).unwrap();
    let prep_neg = jit.prepare(&neg).unwrap();
    jit.finish(prep_pos.as_ref());
    jit.finish(prep_neg.as_ref());

    let mut host = Host::new(1 << 16);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(1),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, all());
    let pos_out = jit
        .execute(prep_pos.as_ref(), &pos, &invocation, &mut scope)
        .unwrap()[0];
    let neg_out = jit
        .execute(prep_neg.as_ref(), &neg, &invocation, &mut scope)
        .unwrap()[0];
    assert_eq!(host.data(pos_out).unwrap().storage(), [0.0f32.to_bits()]);
    assert_eq!(host.data(neg_out).unwrap().storage(), [(-0.0f32).to_bits()]);
}

#[test]
fn at_repeat_trilinear_and_anisotropic_match_cpu_bit_for_bit() {
    use entelechy_abi::{BlockFormat, CompressedBlocks, Encoding, FieldData, Scalar, Schema};

    let source = "kernel test.filtered_jit@1 {
        requires { field.read; field.write; }
        input rows: Rows<Float>;
        input grid: Grid<Vec4>;
        input bc7: Grid<Vec4 as unorm8>;
        input vol: Volume<Vec4 as f16>;
        output out: Grid<Vec4>;
        compute {
            let p = coord();
            let r = at_repeat(rows, p.x * 1.7 - 2.3);
            let g0 = at_repeat(grid, p * vec2(1.5, 1.25) - vec2(1.0, 0.75));
            let g_lod = at_linear(grid, p * 2.0 + vec2(0.5, 0.5), 1.35);
            let b_lod = at_repeat(bc7, p * 2.5 - vec2(3.0, 2.0), 1.5);
            let b_aniso = at_repeat(bc7, p * 2.0 + vec2(0.5, 0.5), vec2(6.0, 0.0), vec2(0.0, 0.5));
            let v_lod = at_linear(vol, vec3(p.x + 0.5, p.y + 0.5, 1.25), 0.75);
            let v_aniso = at_linear(vol, vec3(p.x + 0.5, p.y + 0.5, 0.5), vec2(4.0, 0.0), vec2(0.0, 0.5));
            return vec4(r + g0.x + g_lod.x, b_lod.y + b_aniso.y, v_lod.z + v_aniso.z, g0.w + b_aniso.w);
        }
    }";
    let module = compile(source).unwrap();
    let jit = JitBackend::new().unwrap();
    let prep = jit.prepare(&module).unwrap();

    let rows_data = FieldData::from_values(
        Schema::new(vec![Scalar::Float]),
        Domain::Rows(4),
        &[0.1f32, 0.4, 0.7, 1.0].map(f32::to_bits),
    )
    .unwrap();

    let mut grid_vals = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let chk = if (x + y) % 2 == 0 { 1.0f32 } else { 0.0 };
            grid_vals
                .extend_from_slice(&[chk, x as f32 / 15.0, y as f32 / 15.0, 1.0].map(f32::to_bits));
        }
    }
    let grid_data =
        FieldData::from_values(Schema::pixel(), Domain::Grid(16, 16), &grid_vals).unwrap();

    let mut bc_pixels = vec![0u8; 16 * 16 * 4];
    for y in 0..16usize {
        for x in 0..16usize {
            let b = (y * 16 + x) * 4;
            let stripe = if y % 2 == 0 { 240u8 } else { 20u8 };
            bc_pixels[b..b + 4].copy_from_slice(&[stripe, (x * 15) as u8, (y * 15) as u8, 255]);
        }
    }
    let bc_blocks = CompressedBlocks::encode_unorm8_mips(
        BlockFormat::Bc7RgbaUnorm,
        Domain::Grid(16, 16),
        &bc_pixels,
        5,
        false,
    )
    .unwrap();
    let bc_data = FieldData::from_blocks(Domain::Grid(16, 16), bc_blocks).unwrap();

    let vol_schema = Schema::pixel().with_encoding(Encoding::F16).unwrap();
    let mut vol_vals = Vec::with_capacity(8 * 8 * 4 * 4);
    for z in 0..4u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                vol_vals.extend_from_slice(
                    &[x as f32 / 7.0, y as f32 / 7.0, z as f32 / 3.0, 1.0].map(f32::to_bits),
                );
            }
        }
    }
    let vol_data = FieldData::from_values(vol_schema, Domain::Volume(8, 8, 4), &vol_vals).unwrap();

    let run = |on_jit: bool| {
        let mut host = Host::new(8 << 20);
        let inputs = vec![
            host.adopt(rows_data.clone()).unwrap(),
            host.adopt(grid_data.clone()).unwrap(),
            host.adopt(bc_data.clone()).unwrap(),
            host.adopt(vol_data.clone()).unwrap(),
        ];
        let inv = Invocation {
            inputs,
            params: vec![],
            domain: Domain::Grid(8, 8),
            context: Context::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out = if on_jit {
            jit.execute(prep.as_ref(), &module, &inv, &mut scope)
                .unwrap()[0]
        } else {
            CpuBackend.execute(&module, &inv, &mut scope).unwrap()[0]
        };
        host.data(out).unwrap().storage().to_vec()
    };
    assert_eq!(run(false), run(true));
}
