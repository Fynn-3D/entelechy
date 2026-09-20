//! What a Kernel may call, as `help` tells it: every builtin with its
//! signatures, one sentence, and the capability it needs. The math
//! intrinsics come from the IR's own table and typing rules, so what is
//! listed is what compiles; the Field, context and construction builtins
//! are the ones the lowering dispatches, described beside it.

use orla_ir::{Intrinsic, Type};

/// One builtin as `help` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Builtin {
    /// The name a Kernel calls.
    pub name: &'static str,
    /// What kind of builtin: "math", "field", "memory", "context", "construction".
    pub kind: &'static str,
    /// Every accepted signature, `name(arg: Type, ...) -> Type`; a generic
    /// element type is written `T`.
    pub signatures: Vec<String>,
    /// One sentence.
    pub doc: &'static str,
    /// The capability a Kernel must declare to call it, if any.
    pub requires: Option<&'static str>,
    /// Whether only the entry (`compute`, `scan`, `scatter`) may call it.
    pub entry_only: bool,
}

const PLAIN: [Type; 9] = [
    Type::Float,
    Type::Int,
    Type::Bool,
    Type::Vec2,
    Type::Vec3,
    Type::Vec4,
    Type::Mat2,
    Type::Mat3,
    Type::Mat4,
];

/// The signatures of an intrinsic, found by asking its typing rule about
/// every combination of the plain types.
fn intrinsic_signatures(intrinsic: Intrinsic, name: &str) -> Vec<String> {
    let args = intrinsic.argument_names();
    let mut out = Vec::new();
    let mut combos: Vec<Vec<Type>> = vec![Vec::new()];
    for _ in 0..args.len() {
        combos = combos
            .iter()
            .flat_map(|prefix| {
                PLAIN.iter().map(move |ty| {
                    let mut next = prefix.clone();
                    next.push(*ty);
                    next
                })
            })
            .collect();
    }
    for combo in combos {
        if let Some(result) = intrinsic.result_type(&combo) {
            let params = args
                .iter()
                .zip(&combo)
                .map(|(arg, ty)| format!("{arg}: {ty}"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!("{name}({params}) -> {result}"));
        }
    }
    out
}

fn fixed(
    name: &'static str,
    kind: &'static str,
    signatures: &[&str],
    doc: &'static str,
    requires: Option<&'static str>,
    entry_only: bool,
) -> Builtin {
    Builtin {
        name,
        kind,
        signatures: signatures.iter().map(|s| (*s).to_owned()).collect(),
        doc,
        requires,
        entry_only,
    }
}

/// Every builtin, the math ones first in the IR's order, then the Field,
/// memory, context and construction builtins.
pub fn builtins() -> Vec<Builtin> {
    let mut out: Vec<Builtin> = Intrinsic::ALL
        .iter()
        .map(|(intrinsic, name)| Builtin {
            name,
            kind: if matches!(intrinsic, Intrinsic::ToFloat | Intrinsic::ToInt) {
                "conversion"
            } else {
                "math"
            },
            signatures: intrinsic_signatures(*intrinsic, name),
            doc: intrinsic.describe(),
            requires: None,
            entry_only: false,
        })
        .collect();
    out.extend([
        fixed("saturate", "math", &["saturate(x: Float) -> Float", "saturate(x: Vec2) -> Vec2", "saturate(x: Vec3) -> Vec3", "saturate(x: Vec4) -> Vec4"],
            "x held between 0 and 1, componentwise: clamp(x, 0.0, 1.0).", None, false),
        fixed("distance", "math", &["distance(a: Vec2, b: Vec2) -> Float", "distance(a: Vec3, b: Vec3) -> Float", "distance(a: Vec4, b: Vec4) -> Float"],
            "The length of a - b.", None, false),
        fixed("radians", "math", &["radians(x: Float) -> Float", "radians(x: Vec2) -> Vec2", "radians(x: Vec3) -> Vec3", "radians(x: Vec4) -> Vec4"],
            "Degrees to radians, componentwise.", None, false),
        fixed("degrees", "math", &["degrees(x: Float) -> Float", "degrees(x: Vec2) -> Vec2", "degrees(x: Vec3) -> Vec3", "degrees(x: Vec4) -> Vec4"],
            "Radians to degrees, componentwise.", None, false),
        fixed("inversesqrt", "math", &["inversesqrt(x: Float) -> Float", "inversesqrt(x: Vec2) -> Vec2", "inversesqrt(x: Vec3) -> Vec3", "inversesqrt(x: Vec4) -> Vec4"],
            "1 / sqrt(x), componentwise.", None, false),
        fixed("mod", "math", &["mod(x: Float, y: Float) -> Float", "mod(x: Vec2, y: Vec2) -> Vec2", "mod(x: Vec2, y: Float) -> Vec2", "mod(x: Vec3, y: Vec3) -> Vec3", "mod(x: Vec3, y: Float) -> Vec3", "mod(x: Vec4, y: Vec4) -> Vec4", "mod(x: Vec4, y: Float) -> Vec4"],
            "x - y * floor(x / y): the remainder with y's sign, for wrapping coordinates, where % truncates toward zero.", None, false),
        fixed("isnan", "math", &["isnan(x: Float) -> Bool"], "Whether x is not a number.", None, false),
        fixed("isinf", "math", &["isinf(x: Float) -> Bool"], "Whether x is infinite.", None, false),
        fixed("refract", "math", &["refract(incident: Vec2, normal: Vec2, eta: Float) -> Vec2", "refract(incident: Vec3, normal: Vec3, eta: Float) -> Vec3", "refract(incident: Vec4, normal: Vec4, eta: Float) -> Vec4"],
            "incident bent through a surface with unit normal normal by the ratio of indices eta; the zero vector past total internal reflection.", None, false),
        fixed("at", "field", &["at(field: Rows<T>, index: Int) -> T", "at(field: Grid<T>, at: Vec2) -> T", "at(field: Volume<T>, at: Vec3) -> T"],
            "The element at an address: the index, or the coordinates floored, clamped into the Field (NaN as zero). A memory reads as the steps have left it.", Some("field.read"), false),
        fixed("fetch", "field", &["fetch(field: Rows<T>, index: Int) -> T", "fetch(field: Grid<T>, at: Vec2) -> T", "fetch(field: Volume<T>, at: Vec3) -> T"],
            "Another name for at.", Some("field.read"), false),
        fixed("at_linear", "field", &["at_linear(field: Rows<T>, position: Float) -> T", "at_linear(field: Grid<T>, position: Vec2) -> T", "at_linear(field: Volume<T>, position: Vec3) -> T"],
            "The neighbouring elements blended by the fractional position (element centres at .5), T of floats only; clamped at the edges.", Some("field.read"), false),
        fixed("sample", "field", &["sample(field: Rows<T>) -> T", "sample(field: Grid<T>) -> T", "sample(field: Volume<T>) -> T"],
            "The input's element at this element's own coordinates, clamped into the input.", Some("field.read"), true),
        fixed("count", "field", &["count(field: Rows<T>) -> Int", "count(field: Grid<T>) -> Int", "count(field: Volume<T>) -> Int"],
            "How many elements the Field has.", Some("field.read"), false),
        fixed("size", "field", &["size(field: Grid<T>) -> Vec2", "size(field: Volume<T>) -> Vec3"],
            "The Field's extent along each axis.", Some("field.read"), false),
        fixed("put", "memory", &["put(memory: Rows<T>, index: Int, value: T) -> T", "put(memory: Grid<T>, at: Vec2, value: T) -> T", "put(memory: Volume<T>, at: Vec3, value: T) -> T"],
            "Writes an element of a memory port in step order (scan only; from the entry or a function with the memory as a parameter); an address outside is dropped, never clamped. Yields the value written.", Some("field.write"), false),
        fixed("element", "context", &["element() -> Int"], "This element's index over rows; a scan's step.", None, true),
        fixed("coord", "context", &["coord() -> Vec2", "coord() -> Vec3"], "This element's coordinates over a grid or a volume; rows have element().", None, true),
        fixed("domain", "context", &["domain() -> Int", "domain() -> Vec2", "domain() -> Vec3"], "The output's extent in its shape.", None, true),
        fixed("index", "context", &["index() -> Int"], "The iteration of the innermost loop, or a scan's step outside every loop.", None, false),
        fixed("time", "context", &["time() -> Float"], "The session's time in seconds.", Some("time.read"), false),
        fixed("rate", "context", &["rate() -> Float"], "The sound device's samples per second, zero without one.", Some("time.read"), false),
        fixed("window", "context", &["window() -> Vec2"], "The window's size in physical pixels, zero by zero without one; a node's domain may take it.", Some("input.read"), false),
        fixed("scale", "context", &["scale() -> Float"], "The window's physical pixels per logical pixel, one without a window.", Some("input.read"), false),
        fixed("pointer", "context", &["pointer() -> Vec2"], "The pointer in output pixels.", Some("input.read"), false),
        fixed("pointer_down", "context", &["pointer_down() -> Bool"], "Whether the left button is held.", Some("input.read"), false),
        fixed("button", "context", &["button(n: Int) -> Bool"], "Whether pointer button n is held: 0 left, 1 right, 2 middle.", Some("input.read"), false),
        fixed("wheel", "context", &["wheel() -> Vec2"], "What the wheel scrolled since the last evaluation, in lines: x sideways, y positive away.", Some("input.read"), false),
        fixed("key", "context", &["key(code: Int) -> Bool"], "Whether the key with this code is held: letters and digits by uppercase ASCII, arrows 37 to 40, space 32, enter 13, shift 16, escape 27.", Some("input.read"), false),
        fixed("vec2", "construction", &["vec2(x: Float, y: Float) -> Vec2", "vec2(x: Int, y: Int) -> Vec2"], "A vector from components; Int components widen to Float.", None, false),
        fixed("vec3", "construction", &["vec3(x: Float, y: Float, z: Float) -> Vec3", "vec3(x: Int, y: Int, z: Int) -> Vec3"], "A vector from components; Int components widen to Float.", None, false),
        fixed("vec4", "construction", &["vec4(x: Float, y: Float, z: Float, w: Float) -> Vec4", "vec4(x: Int, y: Int, z: Int, w: Int) -> Vec4"], "A vector from components; Int components widen to Float.", None, false),
        fixed("mat2", "construction", &["mat2(c0: Vec2, c1: Vec2) -> Mat2", "mat2(diagonal: Float) -> Mat2"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
        fixed("mat3", "construction", &["mat3(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3", "mat3(diagonal: Float) -> Mat3"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
        fixed("mat4", "construction", &["mat4(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Mat4", "mat4(diagonal: Float) -> Mat4"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
    ]);
    out
}

/// The builtin named `name`, if any.
pub fn builtin(name: &str) -> Option<Builtin> {
    builtins().into_iter().find(|b| b.name == name)
}

/// Builtins whose names are close to `name`, for a miss.
pub fn similar(name: &str) -> Vec<&'static str> {
    builtins()
        .into_iter()
        .filter(|b| b.name.contains(name) || name.contains(b.name) || distance(b.name, name) <= 2)
        .map(|b| b.name)
        .collect()
}

/// The edit distance between two short names.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            current.push(
                (previous[j] + cost)
                    .min(previous[j + 1] + 1)
                    .min(current[j] + 1),
            );
        }
        previous = current;
    }
    previous[b.len()]
}
