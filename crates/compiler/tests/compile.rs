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
    // The optimizer writes a short function out where it was called, so
    // the call is in the lowering and the arithmetic is in the entry.
    let written = orla_compiler::compile_unoptimized(GRAYSCALE).unwrap();
    assert!(
        written
            .entry
            .body
            .iter()
            .any(|i| matches!(i.op, Op::Call(0, _)))
    );
    assert!(
        !module
            .entry
            .body
            .iter()
            .any(|i| matches!(i.op, Op::Call(..)))
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
        message.contains("must be Float, Int or a shorter vector, found Bool"),
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
    assert!(message.contains("expected a literal"), "{message}");
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
    // A name one letter out is named rather than searched for, in every
    // diagnostic that says a name is not known.
    for (source, expected) in [
        (
            "kernel t.k@1 { requires { field.write; } output r: Image; compute { let value = 1.0; return vec4(valeu, 0.0, 0.0, 1.0); } }",
            "`valeu` is not defined; did you mean `value`?",
        ),
        (
            "kernel t.k@1 { requires { field.read; field.write; } input source: Image; output r: Image; compute { return sample(sorce); } }",
            "`sorce` is not an input; did you mean `source`?",
        ),
        (
            "kernel t.k@1 { requires { field.write; } output r: Image; struct Point { x: Float; } compute { let p = Ponit { x: 1.0 }; return vec4(p.x, 0.0, 0.0, 1.0); } }",
            "`Ponit` is not a struct; did you mean `Point`?",
        ),
        (
            "kernel t.k@1 { requires { field.write; } output r: Image; param k: Flaot = 1.0; compute { return vec4(k, 0.0, 0.0, 1.0); } }",
            "unknown type `Flaot`; did you mean `Float`?",
        ),
    ] {
        let (_, message) = semantic(source);
        assert_eq!(message, expected);
    }
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { return vec4(lenght(vec2(1.0, 2.0)), 0.0, 0.0, 1.0); } }",
    );
    assert!(message.ends_with("did you mean `length`?"), "{message}");
    // And a name that is not near anything is left alone.
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { return vec4(nope, 0.0, 0.0, 1.0); } }",
    );
    assert_eq!(message, "`nope` is not defined");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = loop 4 with x = 1.0 { 2 }; return vec4(a, 0.0, 0.0, 1.0); } }",
    );
    assert!(
        message.contains("carries Float but the body gives Int"),
        "{message}"
    );
    // An inner loop reads the outer loop's carried value by name.
    compile(
        "kernel t.k@1 { requires { field.write; } output r: Image; compute { let a = loop 4 with x = 1.0 { loop 2 with y = 0.0 { x } }; return vec4(a, 0.0, 0.0, 1.0); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
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
        "kernel t.k@1 { requires { field.write; } output r: Image; struct Hit { t: Float; } param h: Hit = vec2(1.0, 2.0); compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    assert!(message.contains("a Hit default must be"), "{message}");
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
            .contains("one of `compute`, `scan`, `scatter`, or `prefix`"),
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
    assert!(
        message.contains("must be Float, Int or a shorter vector"),
        "{message}"
    );
}

#[test]
fn a_prefix_names_a_rule_that_folds_in_any_order() {
    let module = compile(
        "kernel test.rank@1 { requires { field.read; field.write; } input flags: Rows<Int>; output result: Rows<Int>; prefix with sum { return sample(flags); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        module.manifest.mode,
        orla_ir::Mode::Prefix {
            combine: orla_ir::Combine::Sum
        }
    );
    let text = module.to_string();
    assert!(text.contains("mode prefix with sum"), "{text}");

    // The refusals are the point of declaring the rule: each says what
    // to write instead.
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Float>; output result: Rows<Float>; prefix with sum { return sample(rows); } }",
    );
    assert!(message.contains("walk it with `scan`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; prefix with last { return sample(rows); } }",
    );
    assert!(message.contains("write a `compute` block"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; prefix with average { return sample(rows); } }",
    );
    assert!(message.contains("`sum`, `min`, or `max`"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input cells: Grid<Int>; output result: Grid<Int>; prefix with sum { return sample(cells); } }",
    );
    assert!(message.contains("`Rows<T>` input"), "{message}");
    // A float that only `min` or `max` touches is fine: those order
    // floats the same way everywhere.
    compile(
        "kernel t.k@1 { requires { field.read; field.write; } input rows: Rows<Float>; output result: Rows<Float>; prefix with max { return sample(rows); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
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

#[test]
fn a_match_is_checked_for_its_values_types_and_last_arm() {
    let prefix = "kernel t.k@1 { requires { field.write; } output r: Rows<Int>; compute { let i = element();";
    compile(&format!(
        "{prefix} return match i {{ 1 => 10, 2 | 3 => {{ let x = i; x }}, _ => 0 }}; }} }}"
    ))
    .unwrap_or_else(|error| panic!("{error}"));
    let message = compile(&format!(
        "{prefix} return match i {{ 1 => 10, 2 | 1 => 20, _ => 0 }}; }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(message.contains("already has an arm for 1"), "{message}");
    let message = compile(&format!(
        "{prefix} return match i {{ 1 => 10, _ => 0.5 }}; }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(message.contains("must agree"), "{message}");
    let message = compile(&format!(
        "{prefix} return match float(i) {{ 1 => 10, _ => 0 }}; }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(message.contains("takes an Int"), "{message}");
    let message = compile(&format!(
        "{prefix} return match i {{ 1 => 10, 2 => 20 }}; }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(message.contains("`_ => ...`"), "{message}");
}

#[test]
fn a_nested_loop_reads_the_enclosing_loops_carried_value() {
    // The inner loop reads `outer`, carried by the loop around it, and a
    // scan's step reads its own carried value inside a loop.
    compile("kernel t.k@1 { requires { field.write; } output r: Rows<Int>; compute { return loop 3 with outer = 1 { loop 2 with inner = outer { inner + outer } }; } }")
        .unwrap_or_else(|error| panic!("{error}"));
    compile("kernel t.s@1 { requires { field.read; field.write; } input rows: Rows<Int>; output r: Rows<Int>; scan with acc = 0 { return loop 2 with k = acc { k + acc + sample(rows) }; } }")
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn arrays_are_typed_built_indexed_and_checked() {
    let module = compile(
        "kernel t.k@1 { requires { field.write; }
            struct Table { rows: [Int; 4]; scale: Float; }
            output r: Rows<Int>;
            fn roll(t: [Int; 4], by: Int) -> [Int; 4] { return [t[by % 4], t[1], t[2], t[3]]; }
            compute {
                let i = element();
                let lut = [3, 1, 4, 1];
                let same = [i; 4];
                let grid = [[1, 2], [3, 4]];
                let t = Table { rows: roll(lut, i), scale: 0.5 };
                return lut[i % 4] + same[2] + grid[i % 2][1] + t.rows[-0] + int(t.scale);
            } }",
    )
    .unwrap();
    let names: Vec<&str> = module.structs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["[Int; 4]", "Table", "[Int; 2]", "[[Int; 2]; 2]"]
    );
    assert_eq!(module.structs[0].element, Some(Type::Int));
    assert_eq!(module.structs[3].element, Some(Type::Struct(2)));
    assert!(module.structs[1].element.is_none());
    let text = module.to_string();
    assert!(text.contains("subscript"), "{text}");
    assert!(text.contains("extract %"), "{text}");

    let head = "kernel t.k@1 { requires { field.write; } output r: Rows<Int>; compute { let i = element(); ";
    let (_, message) = semantic(&format!("{head}let a = [1, 2.0]; return a[0]; }} }}"));
    assert_eq!(
        message,
        "an array's elements are all one type: this one is Float, the first Int"
    );
    let (_, message) = semantic(&format!("{head}let a = [1, 2]; return a[2]; }} }}"));
    assert_eq!(
        message,
        "index 2 is outside [Int; 2], whose indices are 0 to 1"
    );
    let (_, message) = semantic(&format!("{head}let a = [1, 2]; return a[1.5]; }} }}"));
    assert_eq!(message, "an array takes an Int index, found Float");
    let (_, message) = semantic(&format!("{head}let a = [1, 2]; return a.x; }} }}"));
    assert_eq!(
        message,
        "`[Int; 2]` is an array; take an element with `[index]`"
    );
    let (_, message) = semantic(&format!("{head}return i[0]; }} }}"));
    assert_eq!(
        message,
        "`[index]` takes an element of an array, a column of a matrix or a component of a vector, found Int"
    );
    let (_, message) = semantic(&format!("{head}let a = [1; 5000]; return a[0]; }} }}"));
    assert_eq!(message, "an array has 1 to 4096 elements, not 5000");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } param p: [Int; 2] = 0; output r: Rows<Int>; compute { return 0; } }",
    );
    assert_eq!(
        message,
        "a [Int; 2] default must be a literal of that type; `0` is not"
    );
}

#[test]
fn matrices_are_built_from_columns_multiplied_and_checked() {
    let module = compile(
        "kernel t.k@1 { requires { field.write; }
            struct Frame { basis: Mat3; origin: Vec3; }
            output r: Rows<Vec3>;
            fn spin(a: Float) -> Mat2 { return mat2(vec2(cos(a), sin(a)), vec2(-sin(a), cos(a))); }
            compute {
                let i = element();
                let m = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 3.0));
                let f = Frame { basis: m * transpose(m) + mat3(1.0) * 0.5, origin: vec3(1.0, 2.0, 3.0) };
                let w = inverse(f.basis) * f.origin + f.origin * f.basis;
                let s = spin(0.5) * vec2(1.0, 0.0);
                let d = determinant(mat4(2.0)) + m[i % 3].x + m[2][2] + w[i % 3] + s.y;
                return -m * w * d + (f.basis == m) * w;
            } }",
    );
    let (_, message) = match module {
        Ok(_) => panic!("a Bool times a Vec3 must be refused"),
        Err(CompileError::Semantic { span, message }) => (span, message),
        Err(other) => panic!("{other}"),
    };
    assert_eq!(message, "cannot apply `*` to Bool and Vec3");

    let head = "kernel t.k@1 { requires { field.write; } output r: Rows<Float>; compute { let i = element(); ";
    let ok = compile(&format!(
        "{head}let m = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0)); let n = mat3(1.0); let p = mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)); return determinant(m) + determinant(n) + determinant(p) + (transpose(m) * 2.0 - m)[1].y + (2.0 * n)[i % 3][1] + (m * m)[0][0] + float(m != m); }} }}"
    ))
    .unwrap();
    assert_eq!(ok.manifest.outputs[0].element(), Type::Float);
    // A matrix builtin is its formula in the IR, which has no such
    // operation.
    assert!(!ok.to_string().contains("transpose("), "{ok}");
    assert!(ok.to_string().contains("subscript"), "{ok}");

    let (_, message) = semantic(&format!(
        "{head}let m = mat2(vec3(1.0, 2.0, 3.0), vec2(0.0, 1.0)); return m[0].x; }} }}"
    ));
    assert_eq!(message, "`mat2` takes Vec2 columns, found Vec3");
    let (_, message) = semantic(&format!("{head}let m = mat3(1); return m[0].x; }} }}"));
    assert_eq!(
        message,
        "`mat3` takes 3 Vec3 columns, or one Float for the diagonal; found Int"
    );
    let (_, message) = semantic(&format!("{head}let m = mat2(1.0); return m[2].x; }} }}"));
    assert_eq!(message, "index 2 is outside Mat2, whose indices are 0 to 1");
    let (_, message) = semantic(&format!("{head}let m = mat2(1.0); return m.x; }} }}"));
    assert_eq!(message, "Mat2 is a matrix; take a column with `[index]`");
    let (_, message) = semantic(&format!(
        "{head}let m = mat2(1.0); return (m * mat3(1.0))[0].x; }} }}"
    ));
    assert_eq!(message, "cannot apply `*` to Mat2 and Mat3");
    let (_, message) = semantic(&format!(
        "{head}let m = mat2(1.0); return (m + 1.0)[0].x; }} }}"
    ));
    assert_eq!(message, "cannot apply `+` to Mat2 and Float");
    let (_, message) = semantic(&format!(
        "{head}let m = mat2(1.0); return (m * vec3(1.0, 0.0, 0.0)).x; }} }}"
    ));
    assert_eq!(message, "cannot apply `*` to Mat2 and Vec3");
    let (_, message) = semantic(&format!("{head}return determinant(vec2(1.0, 0.0)); }} }}"));
    assert_eq!(message, "`determinant` cannot be applied to (Vec2)");
    let (_, message) = semantic(&format!(
        "{head}let v = vec3(1.0, 2.0, 3.0); return v[3]; }} }}"
    ));
    assert_eq!(message, "index 3 is outside Vec3, whose indices are 0 to 2");
}

#[test]
fn integer_vectors_address_fields_without_conversions_in_the_source() {
    let module = compile(
        "kernel t.k@1 { requires { field.read; field.write; }
            struct Site { cell: IVec2; weight: Int; }
            input cells: Grid<Int>;
            output r: Grid<Int>;
            fn walk(c: IVec2, d: IVec2) -> IVec2 { return (c + d) % isize(cells); }
            compute {
                let c = icoord();
                let site = Site { cell: walk(c, ivec2(1, -1)), weight: c.x };
                let far = c + idomain() * 2 - ivec2(3, 4) / 2 + abs(-c) + min(c, 5) + clamp(c, ivec2(0, 0), ivec2(7, 7));
                let f = float(far) * 0.5;
                let back = int(f);
                return at(cells, site.cell) + at(cells, back) + far[c.y % 2] + int(c == back);
            } }",
    )
    .unwrap();
    let text = module.to_string();
    // Every `at` addresses the grid with a Vec2: the integer one converted.
    assert!(text.contains("float("), "{text}");
    assert!(!text.contains("icoord"), "{text}");
    assert_eq!(module.functions[0].args, vec![Type::IVec2, Type::IVec2]);

    // A scatter's address may be an integer vector: the pair is converted
    // into a synthesized struct whose address is the float one.
    let scatter = compile(
        "kernel t.s@1 { requires { field.read; field.write; }
            input seeds: Rows<Int>; input canvas: Grid<Int>;
            output r: Grid<Int>;
            struct Drop { at: IVec2; value: Int; }
            scatter into canvas with sum {
                let i = element();
                return Drop { at: ivec2(i % 8, i / 8), value: 1 };
            } }",
    )
    .unwrap();
    let pair = scatter.entry.body.last().unwrap();
    let Type::Struct(index) = pair.ty else {
        panic!("{scatter}")
    };
    let def = &scatter.structs[index as usize];
    assert_eq!(def.name, "(Vec2, Int)");
    assert_eq!(def.fields[0].ty, Type::Vec2);

    let head = "kernel t.k@1 { requires { field.read; field.write; } input g: Grid<Float>; output r: Rows<Float>; compute { let i = element(); ";
    let (_, message) = semantic(&format!("{head}let c = ivec2(1.5, 2); return 0.0; }} }}"));
    assert_eq!(
        message,
        "`ivec2` components must be Int or a shorter integer vector, found Float; a Float truncates with int()"
    );
    let (_, message) = semantic(&format!(
        "{head}let c = ivec2(1, 2) + vec2(1.0, 2.0); return 0.0; }} }}"
    ));
    assert_eq!(message, "cannot apply `+` to IVec2 and Vec2");
    let (_, message) = semantic(&format!(
        "{head}let c = ivec2(1, 2) + ivec3(1, 2, 3); return 0.0; }} }}"
    ));
    assert_eq!(message, "cannot apply `+` to IVec2 and IVec3");
    let (_, message) = semantic(&format!("{head}return at(g, ivec3(1, 2, 3)); }} }}"));
    assert_eq!(
        message,
        "`at` addresses `g`, a Grid, with Vec2 or its integer vector, found IVec3"
    );
    let (_, message) = semantic(&format!("{head}let c = ivec2(1, 2); return c.z; }} }}"));
    assert_eq!(message, "IVec2 has no component `.z`");
    let (_, message) = semantic(&format!("{head}return at_linear(g, ivec2(1, 2)); }} }}"));
    assert_eq!(
        message,
        "`at_linear` addresses `g`, a Grid, with Vec2, found IVec2"
    );
}

#[test]
fn vectors_build_from_vectors_swizzle_and_functions_may_take_a_math_builtins_name() {
    const SHADOWING: &str = "kernel t.k@1 { requires { field.write; }
            output r: Rows<Vec4>;
            fn step(a: Vec2, b: Vec2) -> Vec2 { return a + b * 2.0; }
            fn mix(a: Int) -> Int { return a * 3; }
            compute {
                let i = element();
                let v = vec3(1.0, 2.0, 3.0);
                let p = vec4(v, 1.0);
                let q = vec4(vec2(1.0, 2.0), 3, 4.0);
                let s = vec3(i, vec2(0.5, 0.25));
                let w = p.wzyx + q.xyzw + vec4(s.zz, s.xy) + vec4(p.rgb, p.a);
                let iv = ivec3(ivec2(i, 2), 3).zx;
                let m = step(v.xy, s.yx) + vec2(float(mix(i)), float(iv.y));
                return w + vec4(m, m.yx);
            } }";
    let module = compile(SHADOWING).unwrap();
    let text = module.to_string();
    // The document's `step` and `mix` are what the calls named; the
    // builtins are not. The optimizer writes both functions out where
    // they were called, so the lowering is where the calls are.
    let written = orla_compiler::compile_unoptimized(SHADOWING)
        .unwrap()
        .to_string();
    assert!(written.contains("call fn0("), "{written}");
    assert!(written.contains("call fn1("), "{written}");
    let body = &text[text.find("fn compute()").unwrap_or(0)..];
    assert!(!body.contains("step("), "{body}");
    assert!(!body.contains("mix("), "{body}");

    let head = "kernel t.k@1 { requires { field.write; } output r: Rows<Float>; compute { let i = element(); let v = vec3(1.0, 2.0, 3.0); ";
    let (_, message) = semantic(&format!("{head}return vec4(v, v).x; }} }}"));
    assert_eq!(
        message,
        "`vec4` takes 4 components, found 6 in (Vec3, Vec3)"
    );
    let (_, message) = semantic(&format!("{head}return vec2(v).x; }} }}"));
    assert_eq!(message, "`vec2` takes 2 components, found 3 in (Vec3)");
    let (_, message) = semantic(&format!("{head}return v.xq; }} }}"));
    assert_eq!(
        message,
        "`.xq` is not a component or a swizzle; use x, y, z, w or r, g, b, a, one to four of one alphabet"
    );
    let (_, message) = semantic(&format!("{head}return v.xr; }} }}"));
    assert!(
        message.starts_with("`.xr` is not a component or a swizzle"),
        "{message}"
    );
    let (_, message) = semantic(&format!("{head}return v.xyw.x; }} }}"));
    assert_eq!(message, "Vec3 has no component `.w`");
    let (_, message) = semantic(&format!("{head}return float(ivec3(1, 2, 3).xyzz.x); }} }}"));
    assert_eq!(
        message,
        "`.xyzz` would be an integer vector of four, which there is none"
    );
    let (_, message) = semantic(&format!("{head}return ivec2(1.5, v).x; }} }}"));
    assert_eq!(
        message,
        "`ivec2` components must be Int or a shorter integer vector, found Float; a Float truncates with int()"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.read; field.write; } input g: Rows<Float>; output r: Rows<Float>; fn at(a: Float) -> Float { return a; } compute { return 0.0; } }",
    );
    assert_eq!(
        message,
        "`at` is a builtin of the language; a function cannot take its name (a math builtin's, like `step`, it can, and stands for it here)"
    );
}

#[test]
fn a_manifest_says_what_its_ports_hold_by_name() {
    let module = compile(
        "kernel t.k@1 { requires { field.read; field.write; } struct Note { at: Float; pitch: Int; } input song: Rows<Note>; memory desk: Rows<Note>; output result: Rows<Note>; scan with n = Note { at: 0.0, pitch: 0 } { let here = at(song, index()); put(desk, index(), here); return here; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let named = module.manifest.named(&module.structs).to_string();
    assert!(named.contains("input song: Rows<Note>"), "{named}");
    assert!(named.contains("output result: Rows<Note>"), "{named}");
    assert!(named.contains("memory desk: Rows<Note>"), "{named}");
    // Without the table there are no names to print: a manifest that
    // crossed the ABI on its own is structural, and says so.
    let bare = module.manifest.to_string();
    assert!(bare.contains("input song: Rows<S0>"), "{bare}");
    // A module prints its own manifest with its own names.
    assert!(
        module.to_string().contains("input song: Rows<Note>"),
        "{module}"
    );
}

#[test]
fn a_function_calls_only_what_is_defined_above_it() {
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; fn fact(n: Int) -> Int { return if n <= 1 { 1 } else { n * fact(n - 1) }; } compute { return fact(element()); } }",
    );
    assert!(
        message.contains("calls itself") && message.contains("no stack"),
        "{message}"
    );
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; fn a(n: Int) -> Int { return b(n); } fn b(n: Int) -> Int { return n; } compute { return a(element()); } }",
    );
    assert!(message.contains("defined below this one"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { return nowhere(element()); } }",
    );
    assert!(message.contains("is not a builtin"), "{message}");
}

#[test]
fn a_type_parameter_is_bound_by_the_call() {
    let module = compile(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>;
            fn twice<T>(v: T) -> T { return v + v; }
            compute { let i = element(); return twice(float(i)) + twice(vec2(1.0, 2.0)).x + float(twice(i)); } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let text = module.to_string();
    assert!(
        text.contains("twice<Float>")
            && text.contains("twice<Vec2>")
            && text.contains("twice<Int>"),
        "{text}"
    );
    assert_eq!(module.functions.len(), 3, "one instance a type: {text}");

    // The same types again are the same instance.
    let module = compile(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; fn twice<T>(v: T) -> T { return v + v; } compute { return twice(1.0) + twice(2.0); } }",
    )
    .unwrap();
    assert_eq!(module.functions.len(), 1);

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; fn pick<T>(a: T, b: T) -> T { return a; } compute { return pick(1.0, element()); } }",
    );
    assert!(message.contains("as Int here, but as Float"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; fn make<T>(n: Int) -> T { return n; } compute { return make(1); } }",
    );
    assert!(message.contains("which no argument binds"), "{message}");
    // A body that does not hold for the types it was given says so, and
    // says which instance it was lowering.
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; fn twice<T>(v: T) -> T { return v + v; } compute { return float(twice(element() > 2)); } }",
    );
    assert!(
        message.contains("cannot apply `+`") && message.contains("in `twice` with `T` as Bool"),
        "{message}"
    );
}

#[test]
fn a_break_stops_its_own_loop_and_gives_its_value() {
    let module = compile(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>;
            compute { let i = element(); return loop 20 with best = -1 { let c = i + index(); if c % 7 == 0 { break c; } best }; } }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    // The loop carries the value and whether it stopped; the test reads
    // the second, so no backend learns a new operation.
    let text = module.to_string();
    assert!(text.contains("(Int, stopped)"), "{text}");

    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { let i = element(); if i > 2 { break i; } return i; } }",
    );
    assert!(message.contains("gives a `loop` its value"), "{message}");
    let (_, message) = semantic(
        "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { return loop 4 with a = 0 { if a > 2 { break 1.5; } a + 1 }; } }",
    );
    assert!(
        message.contains("loop carries Int but `break` gives Float"),
        "{message}"
    );
}

#[test]
fn a_map_writes_several_outputs_given_by_name() {
    // Given out of order, through an `if` whose arms are records, one
    // of them a guard's `return`: the entry gives one value per output,
    // in declaration order.
    let module = compile(
        "kernel shade.split@1 {
            requires { field.write; }
            output color: Image;
            output depth: Grid<Float>;
            output hit: Grid<Bool>;
            compute {
                let uv = coord() / domain();
                if uv.x < 0.5 {
                    return { depth: uv.x, hit: true, color: vec4(uv, 0.0, 1.0) };
                }
                let r = { color: vec4(0.0, 0.0, 0.0, 1.0), depth: 1.0, hit: false };
                if uv.y < 0.5 { { hit: r.hit, depth: r.depth, color: r.color } } else { r }
            }
        }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let names: Vec<&str> = module
        .manifest
        .outputs
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(names, ["color", "depth", "hit"]);
    assert_eq!(module.entry.results.len(), 3);
}

#[test]
fn several_outputs_are_a_maps_and_are_checked() {
    let reduce = semantic(
        "kernel s.r@1 { requires { field.read; field.write; } input a: Rows<Float>;
            output sum: Rows<Float>; output most: Rows<Float>;
            compute { return sample(a); } merge(x, y) { return x + y; } }",
    );
    assert!(
        reduce.1.contains("a reduction folds one element"),
        "{}",
        reduce.1
    );
    assert!(reduce.1.contains("one struct"), "{}", reduce.1);
    let scan = semantic(
        "kernel s.s@1 { requires { field.read; field.write; } input a: Rows<Float>;
            output y: Rows<Float>; output z: Rows<Float>;
            scan with acc = 0.0 { return acc + sample(a); } }",
    );
    assert!(scan.1.contains("a scan carries one value"), "{}", scan.1);
    let shapes = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image; output b: Rows<Float>;
            compute { return { a: vec4(1.0, 1.0, 1.0, 1.0), b: 1.0 }; } }",
    );
    assert!(shapes.1.contains("share a shape"), "{}", shapes.1);
    let bare = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image; output b: Grid<Float>;
            compute { return vec4(1.0, 1.0, 1.0, 1.0); } }",
    );
    assert!(bare.1.contains("`{ a: .., b: .. }`"), "{}", bare.1);
    let missing = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image; output b: Grid<Float>;
            compute { return { a: vec4(1.0, 1.0, 1.0, 1.0) }; } }",
    );
    assert!(missing.1.contains("`b` is missing"), "{}", missing.1);
    let twice = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image; output b: Grid<Float>;
            compute { return { a: vec4(1.0, 1.0, 1.0, 1.0), b: 1.0, a: vec4(0.0, 0.0, 0.0, 1.0) }; } }",
    );
    assert!(twice.1.contains("given twice"), "{}", twice.1);
    let unknown = semantic(
        "kernel s.m@1 { requires { field.write; } output color: Image; output b: Grid<Float>;
            compute { return { colour: vec4(1.0, 1.0, 1.0, 1.0), b: 1.0 }; } }",
    );
    assert!(unknown.1.contains("did you mean `color`"), "{}", unknown.1);
    let typed = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image; output b: Grid<Float>;
            compute { return { a: vec4(1.0, 1.0, 1.0, 1.0), b: 1 }; } }",
    );
    assert!(typed.1.contains("output `b` needs a Float"), "{}", typed.1);
    let single = semantic(
        "kernel s.m@1 { requires { field.write; } output a: Image;
            compute { return { a: vec4(1.0, 1.0, 1.0, 1.0) }; } }",
    );
    assert!(single.1.contains("one output"), "{}", single.1);
}

#[test]
fn a_port_keeps_its_scalars_as_the_type_says() {
    let module = compile(
        "kernel image.pack@1 {
            requires { field.read; field.write; }
            input source: Image;
            input bytes: Rows<Int as u8>;
            output result: Grid<Vec4 as unorm8>;
            compute { return sample(source) * float(at(bytes, 0)); }
        }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let manifest = &module.manifest;
    assert_eq!(manifest.inputs[1].ty.to_string(), "Rows<Int as u8>");
    assert_eq!(manifest.outputs[0].ty.to_string(), "Grid<Vec4 as unorm8>");
    assert_eq!(manifest.outputs[0].ty.schema.element_bytes(), 4);
    // The Kernel sees a Vec4 and an Int however they are kept.
    assert_eq!(manifest.outputs[0].element(), Type::Vec4);

    let unknown = semantic(
        "kernel k.a@1 { requires { field.write; } output r: Grid<Float as u9>; compute { return 0.0; } }",
    );
    assert!(
        unknown.1.contains("`u9` is not an encoding"),
        "{}",
        unknown.1
    );
    assert!(unknown.1.contains("unorm8"), "{}", unknown.1);
    let unfit = semantic(
        "kernel k.a@1 { requires { field.write; } output r: Grid<Float as u8>; compute { return 0.0; } }",
    );
    assert!(unfit.1.contains("cannot be kept as `u8`"), "{}", unfit.1);
    let folded = semantic(
        "kernel k.r@1 { requires { field.read; field.write; } input a: Rows<Float>;
            output r: Rows<Float as f16>;
            compute { return sample(a); } merge(x, y) { return x + y; } }",
    );
    assert!(
        folded.1.contains("folds into its output in place"),
        "{}",
        folded.1
    );
}

#[test]
fn a_parameter_is_any_value_a_kernel_can_hold() {
    let module = compile(
        "kernel scene.light@1 {
            requires { field.write; }
            struct Light { dir: Vec3; color: Vec3; power: Float; }
            param view: Mat4 = mat4(1.0);
            param turn: Mat2 = mat2(vec2(0.0, 1.0), vec2(-1.0, 0.0));
            param cell: IVec2 = -ivec2(3, 4);
            param sun: Light = Light { color: vec3(1, 1, 1), dir: vec3(0.0, 1.0, 0.0), power: 2 };
            param ramp: [Float; 3] = [0, 0.5, 1];
            param same: [Int; 4] = [7; 4];
            output result: Image;
            compute {
                let p = view * vec4(1.0, 2.0, 3.0, 1.0);
                let t = turn * vec2(1.0, 0.0);
                return vec4(p.x + t.y + float(cell.x + same[2]), sun.color.y * sun.power, ramp[1], 1.0);
            }
        }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let params = &module.manifest.params;
    let names: Vec<String> = params
        .iter()
        .map(|p| orla_ir::type_name(p.ty, &module.structs))
        .collect();
    assert_eq!(
        names,
        ["Mat4", "Mat2", "IVec2", "Light", "[Float; 3]", "[Int; 4]"]
    );
    use orla_ir::Value;
    let mut identity = [0.0f32; 16];
    for k in 0..4 {
        identity[k * 5] = 1.0;
    }
    assert_eq!(params[0].default, Value::Mat4(identity));
    assert_eq!(params[1].default, Value::Mat2([0.0, 1.0, -1.0, 0.0]));
    assert_eq!(params[2].default, Value::IVec2([-3, -4]));
    // Fields in declaration order, integers widened where floats are.
    assert_eq!(
        params[3].default,
        Value::Record(vec![
            Value::Vec3([0.0, 1.0, 0.0]),
            Value::Vec3([1.0, 1.0, 1.0]),
            Value::Float(2.0),
        ])
    );
    assert_eq!(
        params[4].default,
        Value::Record(vec![
            Value::Float(0.0),
            Value::Float(0.5),
            Value::Float(1.0)
        ])
    );
    assert_eq!(params[5].default, Value::Record(vec![Value::Int(7); 4]));

    let (_, missing) = semantic(
        "kernel k.a@1 { requires { field.write; } struct L { a: Float; b: Float; } param l: L = L { a: 1.0 }; output r: Image; compute { return vec4(l.a); } }",
    );
    assert!(missing.contains("`L` needs field `b`"), "{missing}");
    let (_, short) = semantic(
        "kernel k.a@1 { requires { field.write; } param m: Mat3 = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); output r: Image; compute { return vec4(m[0], 1.0); } }",
    );
    assert!(short.contains("is not a literal of numbers"), "{short}");
    let (_, wrong) = semantic(
        "kernel k.a@1 { requires { field.write; } param c: IVec2 = vec2(1.5, 2.0); output r: Image; compute { return vec4(float(c.x)); } }",
    );
    assert!(wrong.contains("an IVec2 default must be"), "{wrong}");
}
