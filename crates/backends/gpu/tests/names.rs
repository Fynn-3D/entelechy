//! Shader names, without a device: a Kernel with more helpers and wider
//! structs than WGSL has numbered type names, and a scatter with its
//! synthesized pair struct, declare no name WGSL already means. The same
//! Kernel runs on a device in `parity.rs`.

use entelechy_compiler::compile;
use entelechy_gpu::wgsl::{emit, emit_scatter_gather};

/// Names WGSL gives meaning to that a numbered name could spell.
const WGSL: &[&str] = &[
    "bool", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "vec2",
    "vec3", "vec4", "mat2", "mat3", "mat4", "fn", "let", "var", "struct",
];

/// Every name the shader declares: functions, structs and their members,
/// `let`s and `var`s, and parameters.
fn declared(shader: &str) -> Vec<String> {
    let word = |text: &str| -> String {
        text.trim()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect()
    };
    let mut names = Vec::new();
    for line in shader.lines() {
        let line = line.trim();
        for keyword in ["fn ", "struct ", "let ", "var "] {
            if let Some(rest) = line.strip_prefix(keyword) {
                names.push(word(rest));
            }
        }
        // A function's parameters and a struct's members: `name: type`.
        let inside = line
            .split_once('(')
            .filter(|_| line.starts_with("fn "))
            .map(|(_, rest)| rest)
            .or_else(|| {
                line.split_once('{')
                    .filter(|_| line.starts_with("struct "))
                    .map(|(_, rest)| rest)
            });
        if let Some(inside) = inside {
            for part in inside.split(',') {
                if let Some((name, _)) = part.split_once(':') {
                    names.push(word(name));
                }
            }
        }
    }
    names.retain(|name| !name.is_empty());
    names
}

fn clean(shader: &str) {
    let clashes: Vec<String> = declared(shader)
        .into_iter()
        .filter(|name| WGSL.contains(&name.as_str()))
        .collect();
    assert!(
        clashes.is_empty(),
        "the shader declares names WGSL already has: {clashes:?}"
    );
}

#[test]
fn many_helpers_and_fields_are_named_apart_from_wgsl() {
    const MANY: usize = 70;
    let helpers: String = (0..MANY)
        .map(|k| match k {
            0 => "fn h0(x: Float) -> Float { return loop 2 with a = x { a + 0.5 }; }\n".to_owned(),
            k => format!(
                "fn h{k}(x: Float) -> Float {{ return loop 2 with a = h{}(x) {{ a * 0.5 }}; }}\n",
                k - 1
            ),
        })
        .collect();
    let fields: String = (0..MANY).map(|k| format!("m{k}: Float; ")).collect();
    let values: String = (0..MANY).map(|k| format!("m{k}: c.x, ")).collect();
    let source = format!(
        "kernel test.many@1 {{ requires {{ field.write; }} output result: Image;
        struct Wide {{ {fields}}}
        {helpers}
        compute {{
            let c = coord();
            let w = Wide {{ {values}}};
            return vec4(h{last}(c.x), w.m16, w.m32, w.m64);
        }} }}",
        last = MANY - 1
    );
    // Unoptimized, so the shader keeps its many helpers.
    let module =
        entelechy_compiler::compile_unoptimized(&source).unwrap_or_else(|error| panic!("{error}"));
    let shader = emit(&module);
    clean(&shader);
    assert!(
        shader.contains("fn_64("),
        "the helpers reach the shader as functions"
    );
}

#[test]
fn a_scatter_pair_is_named_apart_from_wgsl() {
    let module = compile(
        "kernel test.splat@1 {
            requires { field.read; field.write; }
            input points: Rows<Vec2>;
            input canvas: Grid<Float>;
            output result: Grid<Float>;
            struct Splat { cell: Vec2; heat: Float; }
            scatter into canvas with sum {
                return Splat { cell: sample(points), heat: 1.0 };
            }
        }",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    clean(&emit_scatter_gather(&module));
}
