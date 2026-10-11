//! Every signature help shows is one the compiler takes: a builtin called
//! with a value of each parameter's type gives its result's type, in a
//! Kernel that declares only what help says it needs, and one that leaves
//! that out is refused naming it. A signature over Fields or a generic
//! element, or one that holds only inside a form, is held by the
//! compiler's own tests.

use entelechy_compiler::{compile, help};

/// A value of `ty` as a Kernel writes it, for the types a signature names.
fn value_of(ty: &str) -> Option<&'static str> {
    Some(match ty {
        "Float" => "1.5",
        "Int" => "2",
        "Bool" => "true",
        "Double" => "double(1.5)",
        "Long" => "long(2)",
        "Vec2" => "vec2(1.0, 2.0)",
        "Vec3" => "vec3(1.0, 2.0, 3.0)",
        "Vec4" => "vec4(1.0, 2.0, 3.0, 4.0)",
        "IVec2" => "ivec2(1, 2)",
        "IVec3" => "ivec3(1, 2, 3)",
        "LVec2" => "lvec2(long(1), long(2))",
        "LVec3" => "lvec3(long(1), long(2), long(3))",
        "DVec2" => "dvec2(1.0, 2.0)",
        "DVec3" => "dvec3(1.0, 2.0, 3.0)",
        "DVec4" => "dvec4(1.0, 2.0, 3.0, 4.0)",
        "Mat2" => "mat2(1.0)",
        "Mat3" => "mat3(1.0)",
        "Mat4" => "mat4(1.0)",
        "DMat2" => "dmat2(double(1.0))",
        "DMat3" => "dmat3(double(1.0))",
        "DMat4" => "dmat4(double(1.0))",
        _ => return None,
    })
}

/// `name(a: A, b: B) -> R` as the name, the parameter types and the result
/// type.
fn parts(signature: &str) -> (&str, Vec<&str>, &str) {
    let (call, result) = signature
        .split_once(" -> ")
        .expect("a signature has a result");
    let (name, params) = call.split_once('(').expect("a signature has parameters");
    let params = params.trim_end_matches(')');
    let types = params
        .split(", ")
        .filter(|p| !p.is_empty())
        .map(|p| p.split_once(": ").map_or(p, |(_, ty)| ty))
        .collect();
    (name, types, result)
}

/// Builtins whose signatures hold only inside a form: `index` inside a
/// loop, an `each` or a `scan`.
const ELSEWHERE: [&str; 1] = ["index"];

#[test]
fn every_signature_of_values_compiles_to_its_result() {
    let mut wrong = Vec::new();
    let mut proved = 0;
    for builtin in help::builtins() {
        for signature in &builtin.signatures {
            let (name, params, result) = parts(signature);
            let Some(args) = params
                .iter()
                .map(|ty| value_of(ty))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if value_of(result).is_none() {
                continue;
            }
            let requires = builtin
                .requires
                .map(|c| format!(" {};", c.name()))
                .unwrap_or_default();
            // A context builtin answers for the Kernel's shape: the
            // signature holds over one of them.
            let source = |shape: &str, requires: &str| {
                format!(
                    "kernel t.probe@1 {{\n    requires {{ field.write;{requires} }}\n    output result: {shape}<Float>;\n    fn probe(x: {result}) -> Float {{ return 0.0; }}\n    compute {{ return probe({name}({})); }}\n}}",
                    args.join(", ")
                )
            };
            let shape = ["Rows", "Grid", "Volume"]
                .into_iter()
                .find(|shape| compile(&source(shape, &requires)).is_ok());
            let Some(shape) = shape else {
                if !ELSEWHERE.contains(&name) {
                    wrong.push(format!("`{signature}`"));
                }
                continue;
            };
            proved += 1;
            // What it needs is what the compiler asks for: without it, the
            // Kernel is refused naming it.
            if let Some(capability) = builtin.requires {
                match compile(&source(shape, "")) {
                    Err(error) if error.to_string().contains(capability.name()) => {}
                    other => wrong.push(format!(
                        "`{signature}` is said to need `{capability}`, but without it: {other:?}"
                    )),
                }
            }
        }
    }
    assert!(proved > 150, "only {proved} signatures proved");
    assert!(
        wrong.is_empty(),
        "help shows signatures the compiler does not take:\n{}",
        wrong.join("\n")
    );
}
