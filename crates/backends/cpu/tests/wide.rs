//! Doubles and longs on the interpreter, held to Rust's `f64` and `i64`.

use entelechy_abi::{
    Capabilities, Capability, Context, Domain, Host, Invocation, Scalar, Scope, Value,
};
use entelechy_compiler::compile;
use entelechy_cpu::{CpuBackend, ExecutionError};

fn granted() -> Capabilities {
    [Capability::FieldRead, Capability::FieldWrite]
        .into_iter()
        .collect()
}

/// Runs a Kernel over `count` rows with its params, returning its one
/// output's words.
fn run(source: &str, params: Vec<Value>, count: u32) -> Result<Vec<u32>, ExecutionError> {
    let module = compile(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let mut host = Host::new(1 << 20);
    let invocation = Invocation {
        inputs: Vec::new(),
        params,
        domain: Domain::Rows(count),
        context: Context::default(),
    };
    let mut scope = Scope::new(&mut host, granted());
    let outputs = CpuBackend.execute(&module, &invocation, &mut scope)?;
    Ok(host.data(outputs[0]).unwrap().values().into_owned())
}

fn doubles(words: &[u32]) -> Vec<f64> {
    words
        .chunks(2)
        .map(|pair| f64::from_bits(u64::from(pair[0]) | u64::from(pair[1]) << 32))
        .collect()
}

fn longs(words: &[u32]) -> Vec<i64> {
    words
        .chunks(2)
        .map(|pair| (u64::from(pair[0]) | u64::from(pair[1]) << 32) as i64)
        .collect()
}

#[test]
fn a_double_is_the_double_rust_computes() {
    // Survey coordinates: a tenth of a millimeter half a million meters
    // out, visible to a double and not to a float.
    let source = "kernel wide.d@1 {
        requires { field.write; }
        param origin: Double = 500000.123456789;
        output result: Rows<Double>;
        compute {
            let i = element();
            let far = origin + double(i) * 0.0001;
            let back = far - origin;
            return match i {
                0 => 0.1d + 0.2,
                1 => back,
                2 => double(float(far) - float(origin)),
                3 => sqrt(2.0d) * sin(0.5d) + pow(1.5d, 2.5) - atan2(1.0d, 3.0),
                4 => determinant(dmat3(dvec3(2.0, 1.0, 0.0), dvec3(0.5, 3.0, 1.0), dvec3(0.25, 0.0, 4.0))),
                5 => length(cross(dvec3(1.0, 2.0, 3.0), dvec3(-1.0, 0.5, 2.0))),
                _ => -0.1,
            };
        }
    }";
    let out = doubles(&run(source, vec![Value::Double(500000.123456789)], 7).unwrap());
    let origin = 500000.123456789f64;
    assert_eq!(out[0], 0.1 + 0.2);
    assert_eq!(out[1], (origin + 0.0001) - origin);
    assert!((out[1] - 0.0001).abs() < 1e-10, "{}", out[1]);
    // The same subtraction in floats loses it.
    let floats = f64::from((origin + 0.0002) as f32 - origin as f32);
    assert_eq!(out[2], floats);
    assert!(
        out[2] == 0.0 || (out[2] - 0.0002).abs() > 1e-3,
        "{}",
        out[2]
    );
    assert_eq!(
        out[3],
        2f64.sqrt() * 0.5f64.sin() + 1.5f64.powf(2.5) - 1f64.atan2(3.0)
    );
    let det = 2.0 * (3.0 * 4.0 - 1.0 * 0.0) - 0.5 * (1.0 * 4.0 - 0.0 * 0.0)
        + 0.25 * (1.0 * 1.0 - 0.0 * 3.0);
    assert!((out[4] - det).abs() < 1e-12, "{} against {det}", out[4]);
    let (a, b) = ([1.0f64, 2.0, 3.0], [-1.0f64, 0.5, 2.0]);
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    assert_eq!(out[5], (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
    // A written number is exact where it meets a double.
    assert_eq!(out[6], -0.1f64);
}

#[test]
fn a_long_is_the_long_rust_computes_and_fails_where_it_would() {
    let source = "kernel wide.l@1 {
        requires { field.write; }
        param big: Long = 5000000000l;
        param by: Long = 0l;
        output result: Rows<Long>;
        compute {
            let i = element();
            return match i {
                0 => big * 3,
                1 => -9223372036854775807l - 1,
                2 => (big << 30) >>> 62,
                3 => (-1l >> 70) + (1l << 64) + (1l << 63),
                4 => long(-2.75) + long(3.99d) + long(7),
                5 => long(int(big)),
                6 => add_wrap(9223372036854775807l, 1l),
                7 => 0x7FFFFFFFFFFFFFFFl / big % 1000l,
                _ => big / by,
            };
        }
    }";
    let params = vec![Value::Long(5_000_000_000), Value::Long(1)];
    let out = longs(&run(source, params, 9).unwrap());
    let big = 5_000_000_000i64;
    assert_eq!(out[0], big * 3);
    assert_eq!(out[1], i64::MIN);
    assert_eq!(out[2], ((big << 30) as u64 >> 62) as i64);
    // A shift's count is modulo 64: `-1 >> 6`, `1 << 0`, `1 << 63`.
    let (minus, one) = (-1i64, 1i64);
    assert_eq!(out[3], (minus >> 6) + one + (one << 63));
    assert_eq!(out[4], -2 + 3 + 7);
    // An int of a long keeps its low 32 bits.
    assert_eq!(out[5], i64::from(big as i32));
    assert_eq!(out[6], i64::MIN);
    assert_eq!(out[7], i64::MAX / big % 1000);
    assert_eq!(out[8], big);
    // Division by zero and a sum overflow fail, as for `Int`.
    let zero = run(source, vec![Value::Long(big), Value::Long(0)], 9).unwrap_err();
    assert!(zero.to_string().contains("division by zero"), "{zero}");
    let overflow = run(
        "kernel wide.o@1 { requires { field.write; } param top: Long = 9223372036854775807l;
            output result: Rows<Long>; compute { return top + long(element()); } }",
        vec![Value::Long(i64::MAX)],
        2,
    )
    .unwrap_err();
    assert!(overflow.to_string().contains("overflow"), "{overflow}");
}

#[test]
fn a_written_number_takes_the_width_it_meets() {
    // A written float beside a double is a double, a written integer
    // beside a long is a long; a named one keeps its named width.
    let source = "kernel wide.w@1 {
        requires { field.write; }
        struct Out { a: Double; b: Double; c: Long; d: Long; e: Float; f: DVec2; }
        output result: Rows<Out>;
        compute {
            let x = double(element()) + 0.1;
            let named = 0.1;
            return Out {
                a: x * 0.1,
                b: max(x, 0.3),
                c: long(element()) + 5000000000,
                d: 7,
                e: named,
                f: dvec2(0.1, 0.2) * 3.0,
            };
        }
    }";
    let module = compile(source).unwrap();
    let schema = &module.manifest.outputs[0].ty.schema;
    let mut want = vec![Scalar::Double; 4];
    want.extend([Scalar::Long; 4]);
    want.push(Scalar::Float);
    want.extend([Scalar::Double; 4]);
    assert_eq!(schema.scalars(), &want[..]);
    assert_eq!(schema.to_string(), "d,d,l,l,f,d,d");
    let words = run(source, Vec::new(), 1).unwrap();
    let a = doubles(&words[0..4]);
    assert_eq!(a, vec![0.1 * 0.1, 0.3]);
    assert_eq!(longs(&words[4..8]), vec![5_000_000_000, 7]);
    assert_eq!(f32::from_bits(words[8]), 0.1f32);
    assert_eq!(doubles(&words[9..13]), vec![0.1 * 3.0, 0.2 * 3.0]);
    // A float and a double meet only through a conversion.
    let error = compile(
        "kernel wide.m@1 { requires { field.write; } output result: Rows<Double>;
            compute { let f = 0.5; return double(element()) + f; } }",
    )
    .unwrap_err();
    assert!(error.to_string().contains("Double and Float"), "{error}");
}

#[test]
fn a_value_given_is_taken_in_the_width_it_is_given_to() {
    // A written value has no width: a float given to a double is the
    // decimal it reads as.
    let taken = Value::Float(0.001).shaped_like(&Value::Double(0.0));
    assert_eq!(taken, Some(Value::Double(0.001)));
    let taken = Value::Double(0.1).shaped_like(&Value::Float(0.0));
    assert_eq!(taken, Some(Value::Float(0.1)));
    let taken = Value::Int(-3).shaped_like(&Value::Long(0));
    assert_eq!(taken, Some(Value::Long(-3)));
    assert_eq!(Value::Long(1 << 40).shaped_like(&Value::Int(0)), None);
    let record = Value::Record(vec![Value::Double(1.5), Value::Int(2)]);
    assert_eq!(
        record.shaped_like(&Value::DVec2([0.0; 2])),
        Some(Value::DVec2([1.5, 2.0]))
    );
    // A value's text reads back as it.
    assert_eq!(Value::Double(0.1).to_string(), "0.1d");
    assert_eq!(Value::Long(-5).to_string(), "-5l");
    assert_eq!(
        Value::DMat2(Box::new([1.0, 0.0, 0.0, 1.0])).to_string(),
        "dmat2(dvec2(1.0, 0.0), dvec2(0.0, 1.0))"
    );
}

#[test]
fn written_numbers_are_exact_until_they_meet_a_width_whole() {
    // Arithmetic on written numbers is exact; a vector, array or branch of
    // them takes the width it meets as a whole; a written integer is a
    // float where one is wanted.
    let source = "kernel wide.x@1 {
        requires { field.write; }
        struct Poly { pts: [Double; 2]; w: DVec2; }
        struct Out { sum: Double; third: Double; picked: Double; poly: Poly; moved: DVec3; scaled: Float; }
        fn shift(v: DVec3) -> DVec3 { return v + dvec3(1.0, 1.0, 1.0); }
        output result: Rows<Out>;
        compute {
            let x = double(element()) + 0.5;
            let row = [x, 0.1, 1.0 / 3.0];
            let picked = if element() == 0 { x } else { -0.25 };
            let f = float(element()) + 1.0;
            return Out {
                sum: x * (0.1 + 0.2),
                third: row[2],
                picked: picked,
                poly: Poly { pts: [0.1, -0.2], w: vec2(0.3, 0.7) },
                moved: shift(vec3(0.1, 0.2, 0.3)),
                scaled: f * 2,
            };
        }
    }";
    let words = run(source, Vec::new(), 1).unwrap();
    let d = doubles(&words[..20]);
    assert_eq!(d[0], 0.5 * (0.1 + 0.2));
    assert_eq!(d[1], 1.0 / 3.0);
    assert_eq!(d[2], 0.5);
    assert_eq!(&d[3..7], &[0.1, -0.2, 0.3, 0.7]);
    assert_eq!(&d[7..10], &[0.1 + 1.0, 0.2 + 1.0, 0.3 + 1.0]);
    assert_eq!(f32::from_bits(words[20]), 2.0);
}
