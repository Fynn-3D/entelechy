//! The JIT against the CPU interpreter: the same words out, the same
//! integer failures, on the whole vocabulary a map Kernel has.

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::{ArithmeticError, CpuBackend, ExecutionError};
use orla_jit::JitBackend;
use orla_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

/// Runs a module on both backends and returns each one's output words.
fn both(
    source: &str,
    inputs: &[(orla_abi::Schema, Domain, Vec<u32>)],
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
                orla_abi::FieldData::from_storage(schema.clone(), *domain, words.clone()).unwrap(),
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
    let cell = orla_abi::Schema::new(vec![
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Int,
        orla_abi::Scalar::Bool,
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
                orla_abi::Schema::new(vec![orla_abi::Scalar::Float]),
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
    // d = 3: element 3 divides by zero, before element 5 overflows.
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
    inputs: &[(orla_abi::Schema, Domain, Vec<u32>)],
    domain: Domain,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 22);
    let handles: Vec<_> = inputs
        .iter()
        .map(|(schema, domain, words)| {
            host.adopt(
                orla_abi::FieldData::from_storage(schema.clone(), *domain, words.clone()).unwrap(),
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
    // A machine that walks a grid memory and tallies into a rows memory,
    // reading its carried value and the step index outside any loop, with
    // a loop inside a step and writes outside both memories dropped.
    let source = "kernel test.walk@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory screen: Grid<Int>; memory tally: Rows<Cell>; output result: Rows<Vec2>; struct Cell { hits: Int; last: Float; }
        fn flip(s: Grid<Int>, p: Vec2, seen: Int) -> Int { return put(s, p, (1 - seen) >>> 0); }
        scan with at_ = vec2(3.0, 2.0) {
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
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let cells = orla_abi::Schema::new(vec![orla_abi::Scalar::Int, orla_abi::Scalar::Float]);
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
    let source = "kernel test.count@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with acc = 1000 { return acc / sample(rows); } }";
    let module = compile(source).unwrap();
    let jit = JitBackend::new().unwrap();
    let mut host = Host::new(1 << 16);
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    let rows = host
        .adopt(
            orla_abi::FieldData::from_storage(ints, Domain::Rows(6), vec![2, 5, 1, 0, 3, 3])
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
    // Ints exactly and floats bit for bit, since both backends fold the
    // same pairs in the same order; an odd count runs the tail path.
    let source = "kernel test.stats@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Stats>; struct Stats { bright: Int; sum: Vec4; max: Float; } compute { let c = sample(source); let l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; return Stats { bright: if l > 0.5 { 1 } else { 0 }, sum: c, max: l }; } merge(a, b) { return Stats { bright: a.bright + b.bright, sum: a.sum + b.sum, max: max(a.max, b.max) }; } }";
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
            orla_abi::Schema::pixel(),
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
    // Many drops into a small grid, several per cell, so the order of the
    // float sums matters; an address outside is dropped.
    let source = "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Drop>; input canvas: Grid<Cell>; output result: Grid<Cell>; struct Drop { at: Vec2; weight: Float; id: Int; } struct Cell { mass: Float; top: Int; } struct Splat { cell: Vec2; value: Cell; } scatter into canvas with sum { let d = sample(drops); return Splat { cell: d.at, value: Cell { mass: d.weight, top: d.id } }; } }";
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
    let drop_schema = orla_abi::Schema::new(vec![
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Float,
        orla_abi::Scalar::Int,
    ]);
    let cell = orla_abi::Schema::new(vec![orla_abi::Scalar::Float, orla_abi::Scalar::Int]);
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
            "kernel test.rows@1 {{ requires {{ field.read; field.write; }} input keys: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>; struct Pair {{ at: Int; value: Int; }} scatter into slots with {rule} {{ let k = sample(keys); return Pair {{ at: k % 9 - 1, value: k * 7 % 13 - 6 }}; }} }}"
        );
        let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
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
    let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
    // The fold overflows at the second level.
    let source = "kernel test.total@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return sample(rows); } merge(a, b) { return a + b; } }";
    let module = compile(source).unwrap();
    let mut host = Host::new(1 << 16);
    let rows = host
        .adopt(
            orla_abi::FieldData::from_storage(
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
    // An integer sum into one slot overflows at the third source.
    let source = "kernel test.pile@1 { requires { field.read; field.write; } input rows: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>; struct Pair { at: Int; value: Int; } scatter into slots with sum { return Pair { at: 0, value: sample(rows) }; } }";
    let module = compile(source).unwrap();
    let mut host = Host::new(1 << 16);
    let rows = host
        .adopt(
            orla_abi::FieldData::from_storage(
                ints.clone(),
                Domain::Rows(4),
                vec![0x4000_0000, 0x3000_0000, 0x4000_0000, 1],
            )
            .unwrap(),
        )
        .unwrap();
    let slots = host
        .adopt(orla_abi::FieldData::from_storage(ints, Domain::Rows(2), vec![0; 2]).unwrap())
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
            x: 2,
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
        compute { return sample(xs) * at(weights, element()); }
        merge(a, b) { return a + b + at(weights, 1) * k + first(weights) + time(); } }";
    let floats = orla_abi::Schema::new(vec![orla_abi::Scalar::Float]);
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

/// Where a failure is blamed does not move when the compiler writes a
/// short function out where it was called or lifts a loop's unchanging
/// work out of it: the same element, on both backends. And what a loop
/// that never runs would have failed at does not fail, since work that
/// can fail is never lifted out of one.
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
    // Element 4 divides by zero, three times over, inside a function
    // that is written out inside a loop: both backends blame element 4.
    let expected = ExecutionError::Arithmetic {
        x: 4,
        y: 0,
        error: ArithmeticError::DivisionByZero,
    };
    assert_eq!(run(&CpuBackend, 4).unwrap_err(), expected);
    assert_eq!(run(&jit, 4).unwrap_err(), expected);
    // Element 5 would divide by zero too, but its loop runs no times:
    // the division cannot be lifted out of it, so nothing fails.
    let cpu = run(&CpuBackend, 5).unwrap();
    let jitted = run(&jit, 5).unwrap();
    assert_eq!(cpu, jitted);
    assert_eq!(cpu[5], 0, "the loop that never ran carries its initial");
    assert_eq!(cpu[4], 300, "three times a hundred over one");
}

#[test]
fn emitted_pairs_land_as_the_interpreters_do() {
    // Each element emits as many pairs as its key says, into slots its key
    // picks, over enough elements that both backends split them among
    // threads; the pair each returns lands too.
    let count = 50_000u32;
    for rule in ["sum", "max", "min"] {
        let source = format!(
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
        );
        let ints = orla_abi::Schema::new(vec![orla_abi::Scalar::Int]);
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
