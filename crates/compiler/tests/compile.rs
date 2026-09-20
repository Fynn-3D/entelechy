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
    assert!(
        message.contains("must be Float or Int, found Bool"),
        "{message}"
    );

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
        error
            .to_string()
            .contains("one of `compute`, `scan`, or `scatter`"),
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
fn functions_with_field_parameters_are_instantiated_per_port() {
    let module = compile(
        "kernel test.peek@1 { requires { field.read; field.write; } input a: Grid<Float>; input b: Grid<Float>; output result: Grid<Float>;
            fn half(x: Float) -> Float { return x * 0.5; }
            fn peek(f: Grid<Float>, c: Vec2) -> Float { return half(at(f, c)) + at(f, c + vec2(1.0, 0.0)); }
            fn both(f: Grid<Float>, g: Grid<Float>, c: Vec2) -> Float { return peek(f, c) + peek(g, c) + size(g).x; }
            compute { let c = coord(); return peek(a, c) + peek(b, c) + peek(a, c * 2.0) + both(a, b, c); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let text = module.to_string();
    assert!(
        text.contains("peek<a>") && text.contains("peek<b>") && text.contains("both<a, b>"),
        "{text}"
    );
    assert_eq!(
        module.functions.len(),
        4,
        "half, peek<a>, peek<b>, both<a, b>: {text}"
    );
    assert!(text.contains("host at input1"), "{text}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input a: Grid<Float>; input r: Rows<Float>; output result: Grid<Float>; fn peek(f: Grid<Float>, c: Vec2) -> Float { return at(f, c); } compute { return peek(r, coord()); } }",
    );
    assert!(message.contains("expects Grid<Float> for `f`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input a: Grid<Float>; output result: Grid<Float>; fn peek(f: Grid<Float>, c: Vec2) -> Float { return peek(f, c); } compute { return peek(a, coord()); } }",
    );
    assert!(message.contains("calls itself"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input a: Grid<Float>; output result: Grid<Float>; fn peek(f: Grid<Float>) -> Float { return sample(f); } compute { return peek(a); } }",
    );
    assert!(
        message.contains("only available inside `compute`"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input a: Grid<Float>; output result: Grid<Float>; fn peek(f: Nope, c: Vec2) -> Float { return 1.0; } compute { return peek(a, coord()); } }",
    );
    assert!(message.contains("is not a type"), "{message}");
}

#[test]
fn math_builtins_take_vectors_and_broadcast_floats() {
    let module = compile(
        "kernel test.math@1 { requires { field.write; } output result: Rows<Vec3>; compute {
            let v = vec3(-1.5, 0.25, 2.0);
            let a = clamp(v, 0.0, 1.0) + min(v, 0.5) + max(v, 0.5) + mix(v, v * 2.0, v) + pow(v, 2.0);
            let b = ceil(v) + round(v) + sign(v) + log(abs(v)) + tan(v) + asin(clamp(v, -1.0, 1.0)) + acos(clamp(v, -1.0, 1.0)) + atan(v);
            let n = sign(-3) + abs(-2);
            return a + b + vec3(float(n), 0.0, 0.0);
        } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(module.to_string().contains("round"), "{module}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Vec3>; compute { return clamp(vec3(1.0, 1.0, 1.0), vec2(0.0, 0.0), 1.0); } }",
    );
    assert!(message.contains("cannot be applied"), "{message}");
}

#[test]
fn buttons_and_the_wheel_are_context_under_input_read() {
    let module = compile(
        "kernel test.stir@1 { requires { field.write; input.read; } output result: Rows<Float>; compute { return if button(1) { wheel().y } else { 0.0 }; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let text = module.to_string();
    assert!(
        text.contains("context button") && text.contains("context wheel"),
        "{text}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return wheel().x; } }",
    );
    assert!(message.contains("needs `input.read`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; input.read; } output result: Rows<Bool>; compute { return button(1.0); } }",
    );
    assert!(message.contains("Int"), "{message}");
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

#[test]
fn scatter_names_its_target_and_rule_and_is_checked() {
    let module = compile(
        "kernel test.splat@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Vec2; mass: Float; } scatter into canvas with sum { return Splat { cell: sample(drops), mass: 1.0 }; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        module.manifest.mode,
        orla_ir::Mode::Scatter {
            base: 1,
            combine: orla_ir::Combine::Sum
        }
    );
    let text = module.to_string();
    assert!(
        text.contains("mode scatter into input 1 with sum"),
        "{text}"
    );

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Vec2; mass: Float; } scatter into nowhere with sum { return Splat { cell: sample(drops), mass: 1.0 }; } }",
    );
    assert!(message.contains("not an input"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Vec2; mass: Float; } scatter into canvas with average { return Splat { cell: sample(drops), mass: 1.0 }; } }",
    );
    assert!(
        message.contains("`sum`, `min`, `max`, or `last`"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Image; struct Splat { cell: Vec2; mass: Float; } scatter into canvas with sum { return Splat { cell: sample(drops), mass: 1.0 }; } }",
    );
    assert!(
        message.contains("shaped and laid out like `canvas`"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Int; mass: Float; } scatter into canvas with sum { return Splat { cell: 1, mass: 1.0 }; } }",
    );
    assert!(
        message.contains("an address (Vec2) and a value (Float)"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input flags: Grid<Bool>; output result: Grid<Bool>; struct Splat { cell: Vec2; on: Bool; } scatter into flags with sum { return Splat { cell: sample(drops), on: true }; } }",
    );
    assert!(message.contains("scatter with `last`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input drops: Rows<Vec2>; input canvas: Grid<Float>; output result: Grid<Float>; struct Splat { cell: Vec2; mass: Float; } scatter into canvas with sum { return Splat { cell: sample(drops), mass: 1.0 }; } merge(a, b) { return a + b; } }",
    );
    assert!(message.contains("not both"), "{message}");
}

#[test]
fn window_and_scale_are_context_under_input_read() {
    let module = compile(
        "kernel test.fit@1 { requires { field.write; input.read; } output result: Rows<Vec2>; compute { return window() / scale(); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let text = module.to_string();
    assert!(text.contains("context window"), "{text}");
    assert!(text.contains("context scale"), "{text}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Vec2>; compute { return window(); } }",
    );
    assert!(message.contains("needs `input.read`"), "{message}");
}

#[test]
fn bit_operators_take_ints_only() {
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.5 & 2.0; } }",
    );
    assert!(message.contains("cannot apply `&`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { return 1 << 2.0; } }",
    );
    assert!(message.contains("cannot apply `<<`"), "{message}");
}

#[test]
fn memories_are_ordered_work_written_with_put() {
    let module = compile(
        "kernel test.tally@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { let cell = index() % 4; let v = at(ram, cell) + 1; put(ram, cell, v); return acc + v; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(module.manifest.memories, ["ram"]);
    assert_eq!(
        module.manifest.inputs.len(),
        2,
        "a memory is bound like an input"
    );
    assert_eq!(module.manifest.outputs.len(), 2, "and read like an output");
    let text = module.to_string();
    assert!(text.contains("memory ram: Rows<Int>"), "{text}");
    assert!(text.contains("host put input1"), "{text}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; compute { return 1; } }",
    );
    assert!(message.contains("needs ordered work"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; input ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { put(ram, 0, 1); return acc; } }",
    );
    assert!(message.contains("is not one"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { put(ram, 0.5, 1); return acc; } }",
    );
    assert!(message.contains("it takes Int"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { put(ram, 0, 1.5); return acc; } }",
    );
    assert!(message.contains("whose element is Int"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } memory ram: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { return acc; } }",
    );
    assert!(message.contains("cannot be a memory"), "{message}");
}

#[test]
fn a_function_writes_a_memory_for_the_steps_but_init_does_not() {
    let prefix = "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; fn store(m: Rows<Int>, a: Int, v: Int) -> Int { return put(m, a, v); }";
    compile(&format!(
        "{prefix} scan with acc = 0 {{ store(ram, index(), acc); return acc + 1; }} }}"
    ))
    .unwrap_or_else(|error| panic!("{error}"));
    let message = compile(&format!(
        "{prefix} scan with acc = store(ram, 0, 1) {{ return acc + 1; }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(message.contains("before the first step"), "{message}");
    let message = compile(
        "kernel t.k@1 { requires { field.read; field.write; } input ticks: Rows<Int>; memory ram: Rows<Int>; output result: Rows<Int>; fn store(m: Rows<Int>, a: Int, v: Int) -> Int { return put(m, a, v); } compute { return store(ram, 0, 1); } }",
    )
    .unwrap_err()
    .to_string();
    assert!(message.contains("use `scan`"), "{message}");
}
