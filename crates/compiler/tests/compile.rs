use orla_compiler::{CompileError, compile};
use orla_ir::{Capability, Op, Type};
use orla_language::Span;

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

fn semantic(source: &str) -> (Span, String) {
    match compile(source).unwrap_err() {
        CompileError::Semantic { span, message } => (span, message),
        other => panic!("expected a semantic error, got {other}"),
    }
}

#[test]
fn grayscale_compiles_to_validated_ir() {
    let module = compile(GRAYSCALE).unwrap();
    let manifest = &module.manifest;
    assert_eq!(manifest.id.to_string(), "image.grayscale@1");
    assert!(manifest.requires.contains(Capability::ImageRead));
    assert_eq!(manifest.inputs[0].name, "source");
    assert_eq!(manifest.params[0].name, "amount");
    assert_eq!(manifest.outputs[0].ty, Type::Image);
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].args, vec![Type::Vec4]);
    assert!(module.entry.body.iter().any(|i| i.op == Op::ImageSample(0)));
    assert!(
        module
            .entry
            .body
            .iter()
            .any(|i| matches!(i.op, Op::Call(0, _)))
    );
    let text = module.to_string();
    assert!(text.contains("host image.sample input0"), "{text}");
    assert!(text.contains("mix("), "{text}");
}

#[test]
fn diagnostics_name_the_problem_and_the_position() {
    let (span, message) = semantic(
        "kernel t.k@1 { requires { image.read; image.write; } input a: Image; output r: Image; compute { return sample(b); } }",
    );
    assert_eq!(message, "`b` is not an input image");
    assert_eq!(span.column, 111);

    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; compute { return 1.0; } }",
    );
    assert!(message.contains("needs a Vec4 per pixel"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; compute { return vec4(1, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("must be Float, found Int"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; fn f(x: Float) -> Float { return coord().x; } compute { return vec4(f(1.0), 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("only available inside `compute`"),
        "{message}"
    );

    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; fn f(x: Float) -> Float { return g(x); } fn g(x: Float) -> Float { return x; } compute { return vec4(f(1.0), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("defined above"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { file.write; } output r: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert_eq!(message, "unknown capability `file.write`");
}

#[test]
fn undeclared_capabilities_are_rejected_before_execution() {
    let source = "kernel t.k@1 { requires { image.write; } input a: Image; output r: Image; compute { return sample(a); } }";
    let (span, message) = semantic(source);
    assert_eq!(message, "`sample` needs `image.read`; add it to `requires`");
    assert_eq!(span.column, 92);
    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.read; } output r: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("`image.write`"), "{message}");
}

#[test]
fn parameter_defaults_are_constants_of_the_declared_type() {
    let module = compile(
        "kernel t.k@1 { requires { image.write; } param scale: Vec2 = vec2(1.0, -2.0); param n: Int = -3; param f: Float = 2; output r: Image; compute { return vec4(scale.x, float(n), f, 1.0); } }",
    )
    .unwrap();
    assert_eq!(
        module.manifest.params[0].default,
        orla_ir::Value::Vec2([1.0, -2.0])
    );
    assert_eq!(module.manifest.params[1].default, orla_ir::Value::Int(-3));
    assert_eq!(
        module.manifest.params[2].default,
        orla_ir::Value::Float(2.0)
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } param f: Float = 1.0 + 1.0; output r: Image; compute { return vec4(f, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("must be a literal"), "{message}");
}

#[test]
fn time_and_vec3_are_typed_and_gated() {
    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; compute { return vec4(time(), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("`time.read`"), "{message}");
    let module = compile(
        "kernel t.k@1 { requires { image.write; time.read; } param tint: Vec3 = vec3(1.0, 0.5, 0.25); output r: Image; compute { let c = tint * time(); return vec4(c.x, c.y, c.z, 1.0); } }",
    )
    .unwrap();
    assert_eq!(
        module.manifest.params[0].default,
        orla_ir::Value::Vec3([1.0, 0.5, 0.25])
    );
    assert!(module.to_string().contains("context time"));
    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; compute { let v = vec3(1.0, 0.0, 0.0); return vec4(v.w, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("no component `.w`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { image.write; } output r: Image; compute { let d = dot(vec3(1.0, 0.0, 0.0), vec2(0.0, 1.0)); return vec4(d, 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("`dot` cannot be applied to (Vec3, Vec2)"),
        "{message}"
    );
}
