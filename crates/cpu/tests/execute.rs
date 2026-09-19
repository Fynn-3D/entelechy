use orla_abi::{AbiError, Capabilities, Capability, Domain, Host, Invocation, Scope, Value};
use orla_compiler::compile;
use orla_cpu::{CpuBackend, ExecutionError};

fn granted() -> Capabilities {
    [Capability::ImageRead, Capability::ImageWrite]
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
    requires { image.read; image.write; }
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
        domain: Domain {
            width: 2,
            height: 1,
        },
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
        domain: Domain {
            width: 1,
            height: 1,
        },
    };
    let mut scope = Scope::new(&mut host, [Capability::ImageWrite].into_iter().collect());
    let error = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(
        error,
        ExecutionError::Abi(AbiError::Denied(Capability::ImageRead))
    );
    assert_eq!(
        host.live_images(),
        1,
        "failed execution released its outputs"
    );
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
        domain: Domain {
            width: 1,
            height: 1,
        },
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
        "kernel test.gradient@1 { requires { image.write; } output result: Image; compute { let uv = coord() / (domain() - vec2(1.0, 1.0)); return vec4(uv.x, uv.y, 0.0, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(256);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain {
            width: 3,
            height: 2,
        },
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
        "kernel test.shift@1 { requires { image.read; image.write; } input source: Image; param offset: Int = 10; output result: Image; compute { return fetch(source, coord() + vec2(float(offset), 0.0)); } }",
    )
    .unwrap();
    let invocation = Invocation {
        inputs: vec![gradient_out],
        params: vec![Value::Int(10)],
        domain: Domain {
            width: 3,
            height: 2,
        },
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
        "kernel test.div@1 { requires { image.write; } param d: Int = 0; output result: Image; compute { let q = 1 / d; return vec4(float(q), 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let mut host = Host::new(64);
    let invocation = Invocation {
        inputs: vec![],
        params: vec![Value::Int(0)],
        domain: Domain {
            width: 2,
            height: 2,
        },
    };
    let mut scope = Scope::new(&mut host, granted());
    let error = CpuBackend
        .execute(&module, &invocation, &mut scope)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "at pixel (0, 0): integer division by zero"
    );
    assert_eq!(host.live_images(), 0);
}
