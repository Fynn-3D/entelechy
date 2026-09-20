//! The JIT against the CPU interpreter: the same words out, the same
//! integer failures, on the whole vocabulary a map Kernel has.

use orla_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::{ArithmeticError, CpuBackend, ExecutionError};
use orla_jit::JitBackend;
use orla_runtime::Backend;

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
                orla_abi::FieldData::from_words(schema.clone(), *domain, words.clone()).unwrap(),
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
    let cpu_words = host.data(cpu).unwrap().words().to_vec();
    let prepared = jit
        .prepare(&module)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut scope = Scope::new(&mut host, all());
    let out = jit
        .execute(prepared.as_ref(), &module, &invocation, &mut scope)
        .unwrap()[0];
    let jit_words = host.data(out).unwrap().words().to_vec();
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
            let bits = (i << 3) ^ (i >> 1) | (n & 5);
            let trig = tan(x * 0.1) + asin(fract(x * 0.13)) + acos(fract(x * 0.17)) + atan(x) + atan2(x, 3.0) + exp(k * x) + log(x + 1.0) + pow(x + 1.0, 1.5) + sqrt(x);
            let shaped = smoothstep(2.0, 9.0, x) + step(4.0, x) + floor(x * 0.9) + ceil(x * 0.2) + round(x * 0.5) + sign(x - 5.0) + abs(x - 6.0) + clamp(x, 2.0, 8.0) + max(x, 3.0) + min(x, 4.0);
            let ints = clamp(i * 3, 2, 20) + max(i, 4) + min(i, 3) + abs(i - 4) + sign(i - 2) + (i % 3) + (i / 2) + int(x * 1.5) + int(c.on);
            let r = reflect(normalize(w), normalize(vec3(0.0, 1.0, 0.0))) + cross(w, v);
            let v4 = vec4(x, 1.0, 2.0, 3.0) * 0.5 + vec4(1, 2, 3, 4);
            let same = vec2(x, 1.0) == vec2(float(i), 1.0) && v != vec3(0.0, 0.0, 0.0) && !(x > 100.0);
            return Out { f: s.sum + lin + walked.sum + trig + shaped + v4.w + dot(m, r) + time() + pointer().x + float(key(65)) + float(pointer_down()) + rate() + window().y + scale(), i: s.count + walked.count + bits + ints + int(same), v: m + r, b: same || c.on };
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
