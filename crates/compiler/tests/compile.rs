use orla_compiler::{CompileError, compile};
use orla_ir::{Capability, Op, Type};
use orla_language::Span;

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
    assert!(manifest.requires.contains(Capability::FieldRead));
    assert_eq!(manifest.inputs[0].name, "source");
    assert_eq!(manifest.params[0].name, "amount");
    assert_eq!(manifest.outputs[0].ty, orla_ir::PortType::image());
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].args, vec![Type::Vec4]);
    assert!(module.entry.body.iter().any(|i| i.op == Op::Sample(0)));
    assert!(
        module
            .entry
            .body
            .iter()
            .any(|i| matches!(i.op, Op::Call(0, _)))
    );
    let text = module.to_string();
    assert!(text.contains("host sample input0"), "{text}");
    assert!(text.contains("mix("), "{text}");
}

#[test]
fn diagnostics_name_the_problem_and_the_position() {
    let (span, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input a: Image; output r: Image; compute { return sample(b); } }",
    );
    assert_eq!(message, "`b` is not an input");
    assert_eq!(span.column, 111);

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { return 1.0; } }",
    );
    assert!(message.contains("needs a Vec4 per element"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { return vec4(true, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("must be Float or Int, found Bool"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; fn f(x: Float) -> Float { return coord().x; } compute { return vec4(f(1.0), 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("only available inside `compute`"),
        "{message}"
    );

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; fn f(x: Float) -> Float { return g(x); } fn g(x: Float) -> Float { return x; } compute { return vec4(f(1.0), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("defined above"), "{message}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { file.write; } output r: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert_eq!(message, "unknown capability `file.write`");
}

#[test]
fn undeclared_capabilities_are_rejected_before_execution() {
    let source = "kernel t.k@1 { requires { field.write; } input a: Image; output r: Image; compute { return sample(a); } }";
    let (span, message) = semantic(source);
    assert_eq!(message, "`sample` needs `field.read`; add it to `requires`");
    assert_eq!(span.column, 92);
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; } output r: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("`field.write`"), "{message}");
}

#[test]
fn parameter_defaults_are_constants_of_the_declared_type() {
    let module = compile(
        "kernel t.k@1 { requires { field.write; } param scale: Vec2 = vec2(1.0, -2.0); param n: Int = -3; param f: Float = 2; output r: Image; compute { return vec4(scale.x, float(n), f, 1.0); } }",
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
        "kernel t.k@1 { requires { field.write; } param f: Float = 1.0 + 1.0; output r: Image; compute { return vec4(f, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("must be a literal"), "{message}");
}

#[test]
fn time_and_vec3_are_typed_and_gated() {
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { return vec4(time(), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("`time.read`"), "{message}");
    let module = compile(
        "kernel t.k@1 { requires { field.write; time.read; } param tint: Vec3 = vec3(1.0, 0.5, 0.25); output r: Image; compute { let c = tint * time(); return vec4(c.x, c.y, c.z, 1.0); } }",
    )
    .unwrap();
    assert_eq!(
        module.manifest.params[0].default,
        orla_ir::Value::Vec3([1.0, 0.5, 0.25])
    );
    assert!(module.to_string().contains("context time"));
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let v = vec3(1.0, 0.0, 0.0); return vec4(v.w, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("no component `.w`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let d = dot(vec3(1.0, 0.0, 0.0), vec2(0.0, 1.0)); return vec4(d, 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("`dot` cannot be applied to (Vec3, Vec2)"),
        "{message}"
    );
}

#[test]
fn control_flow_is_scoped_and_typed() {
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = if 1.0 { 1.0 } else { 2.0 }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("needs a Bool condition"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = if true { 1.0 } else { 2 }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("branches give Float and Int"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = if true { let z = 1.0; z } else { 2.0 }; return vec4(z, 0.0, 0.0, 1.0); } }",
    );
    assert_eq!(message, "`z` is not defined");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = loop 4 with x = 1.0 { 2 }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("carries Float but the body gives Int"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = loop 4 with x = 1.0 { loop 2 with y = 0.0 { x } }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("carried by an outer loop"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let i = index(); return vec4(float(i), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("inside a loop"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = loop 4 with x = 1.0 until x { x }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("`until` needs a Bool"), "{message}");
    let module = compile(
        "kernel t.k@1 { requires { field.write; } param on: Bool = true; output r: Image; compute { let a = if on && !(1 > 2) { 1.0 } else { 0.0 }; return vec4(a, a, a, 1.0); } }",
    )
    .unwrap();
    assert_eq!(
        module.manifest.params[0].default,
        orla_ir::Value::Bool(true)
    );
    let text = module.to_string();
    assert!(text.contains("= if %"), "{text}");
}

#[test]
fn structs_are_checked_by_name_and_field() {
    let prefix = "kernel t.k@1 { requires { field.write; } output r: Image; struct Hit { t: Float; id: Int; }";
    let ok = compile(&format!("{prefix} compute {{ let h = Hit {{ t: 1.0, id: 2 }}; let g = Hit {{ ..h, t: 3.0 }}; return vec4(g.t, float(g.id), 0.0, 1.0); }} }}")).unwrap();
    assert!(
        ok.to_string()
            .contains("struct S0 Hit { t: Float, id: Int }"),
        "{ok}"
    );

    let (_, message) = semantic(&format!(
        "{prefix} compute {{ let h = Hit {{ t: 1.0 }}; return vec4(h.t, 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(message.contains("needs field `id`"), "{message}");
    let (_, message) = semantic(&format!(
        "{prefix} compute {{ let h = Hit {{ t: 1.0, id: 2, z: 3 }}; return vec4(h.t, 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(message.contains("no field `z`"), "{message}");
    let (_, message) = semantic(&format!(
        "{prefix} compute {{ let h = Hit {{ t: 1, id: 2 }}; return vec4(h.t, 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(
        message.contains("`Hit.t` expects Float, found Int"),
        "{message}"
    );
    let (_, message) = semantic(&format!(
        "{prefix} compute {{ let h = Hit {{ t: 1.0, id: 2 }}; return vec4(h.w, 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(message.contains("`Hit` has no field `w`"), "{message}");
    let (_, message) = semantic(&format!(
        "{prefix} compute {{ let h = Hit {{ ..1.0, t: 1.0 }}; return vec4(h.t, 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(message.contains("`..base` must be a `Hit`"), "{message}");
    let (_, message) = semantic(&format!(
        "{prefix} fn f(h: Hit) -> Float {{ return h.t; }} compute {{ return vec4(f(1.0), 0.0, 0.0, 1.0); }} }}"
    ));
    assert!(
        message.contains("expects Hit here, found Float"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; struct Hit { t: Float; } param h: Hit = 1.0; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("cannot be structs"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; struct Node { next: Node; } compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("cannot contain itself"), "{message}");
    let error = compile("kernel t.k@1 { requires { field.write; } output r: Image; struct hit { t: Float; } compute { return vec4(0.0, 0.0, 0.0, 1.0); } }").unwrap_err();
    assert!(error.to_string().contains("uppercase"), "{error}");
}

#[test]
fn merge_makes_a_reduction_and_is_checked() {
    let module = compile(
        "kernel test.mean@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Vec2>; compute { let c = sample(source); return vec2(c.r, 1.0); } merge(a, b) { return a + b; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(module.manifest.mode, orla_ir::Mode::Reduce);
    let text = module.to_string();
    assert!(text.contains("mode reduce"), "{text}");
    assert!(text.contains("fn merge(arg0: Vec2, arg1: Vec2)"), "{text}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { return sample(source); } merge(a, b) { return a + b; } }",
    );
    assert!(message.contains("must be `Rows<T>`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } merge(a, b) { return a + b; } }",
    );
    assert!(message.contains("maps over its first input"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input source: Image; output result: Rows<Float>; compute { return 1.0; } merge(a, b) { return vec2(a, b); } }",
    );
    assert!(message.contains("`merge` must give a Float"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input source: Rows<Float>; output result: Rows<Float>; compute { return coord().x; } merge(a, b) { return a + b; } }",
    );
    assert!(message.contains("use `element()`"), "{message}");
}

#[test]
fn scan_carries_a_value_and_is_checked() {
    let module = compile(
        "kernel test.prefix@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with total = 0 { return total + sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(module.manifest.mode, orla_ir::Mode::Scan);
    let text = module.to_string();
    assert!(text.contains("mode scan"), "{text}");
    assert!(text.contains("fn init()"), "{text}");
    assert!(text.contains("carried"), "{text}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Image; scan with total = 0 { return total; } }",
    );
    assert!(message.contains("produces a `Rows` output"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with total = 0.5 { return total; } }",
    );
    assert!(message.contains("initial value is Float"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with total = 0 { return total; } merge(a, b) { return a + b; } }",
    );
    assert!(message.contains("not both"), "{message}");
    let error = compile(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return 1; } scan with total = 0 { return total; } }",
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("either `compute` or `scan`"),
        "{error}"
    );
}

#[test]
fn rate_is_context_under_time_read() {
    let module = compile(
        "kernel test.hz@1 { requires { field.write; time.read; } output result: Rows<Float>; compute { return rate() + time(); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(module.to_string().contains("context rate"), "{module}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return rate(); } }",
    );
    assert!(message.contains("needs `time.read`"), "{message}");
}

#[test]
fn remainder_widening_and_vector_equality_type_check() {
    let module = compile(
        "kernel test.ops@1 { requires { field.write; } output result: Rows<Bool>; compute { let n = element() % 4; let p = vec2(n, 1); return p == vec2(1.0, 1.0) && n % 2 == 0 && fract(2.5 % 1.0) == 0.5; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let text = module.to_string();
    assert!(text.contains(" % "), "{text}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Bool>; compute { return vec2(1.0, 2.0) == vec3(1.0, 2.0, 3.0); } }",
    );
    assert!(message.contains("cannot apply `==`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Bool>; compute { return vec2(true, 1.0) == vec2(1.0, 1.0); } }",
    );
    assert!(message.contains("must be Float or Int"), "{message}");
}
