//! What `help` lists: every builtin with its signatures, sentence and
//! capability, and every language and document form. Math intrinsics
//! come from the IR's table and typing rules; the other builtins are the
//! ones the lowering dispatches.

use orla_ir::{Capability, Intrinsic, Type};

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
    pub requires: Option<Capability>,
    /// Whether only the entry (`compute`, `scan`, `scatter`) may call it.
    pub entry_only: bool,
}

/// Every type a name spells.
fn plain() -> impl Iterator<Item = Type> + Clone {
    Type::NAMED.iter().map(|(ty, _)| *ty)
}

/// An intrinsic's signatures: its typing rule asked about every
/// combination of the plain types.
fn intrinsic_signatures(intrinsic: Intrinsic, name: &str) -> Vec<String> {
    let args = intrinsic.argument_names();
    let mut out = Vec::new();
    let mut combos: Vec<Vec<Type>> = vec![Vec::new()];
    for _ in 0..args.len() {
        combos = combos
            .iter()
            .flat_map(|prefix| {
                plain().map(move |ty| {
                    let mut next = prefix.clone();
                    next.push(ty);
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
    requires: Option<Capability>,
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
            kind: if matches!(
                intrinsic,
                Intrinsic::ToFloat | Intrinsic::ToInt | Intrinsic::ToDouble | Intrinsic::ToLong
            ) {
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
        fixed("dot", "math", &["dot(a: Vec2, b: Vec2) -> Float", "dot(a: Vec3, b: Vec3) -> Float", "dot(a: Vec4, b: Vec4) -> Float", "dot(a: DVec3, b: DVec3) -> Double"],
            "The dot product of two vectors of one size: the componentwise product, summed first to last.", None, false),
        fixed("cross", "math", &["cross(a: Vec3, b: Vec3) -> Vec3", "cross(a: DVec3, b: DVec3) -> DVec3"],
            "The cross product of two three-component vectors.", None, false),
        fixed("length", "math", &["length(v: Vec2) -> Float", "length(v: Vec3) -> Float", "length(v: Vec4) -> Float", "length(v: DVec3) -> Double"],
            "The Euclidean length of a vector: sqrt(dot(v, v)).", None, false),
        fixed("normalize", "math", &["normalize(v: Vec2) -> Vec2", "normalize(v: Vec3) -> Vec3", "normalize(v: Vec4) -> Vec4", "normalize(v: DVec3) -> DVec3"],
            "The unit vector in v's direction; the zero vector stays zero.", None, false),
        fixed("reflect", "math", &["reflect(incident: Vec3, normal: Vec3) -> Vec3", "reflect(incident: Vec2, normal: Vec2) -> Vec2", "reflect(incident: DVec3, normal: DVec3) -> DVec3"],
            "incident reflected off the surface with unit normal normal: incident - 2 * dot(normal, incident) * normal.", None, false),
        fixed("transpose", "math", &["transpose(m: Mat2) -> Mat2", "transpose(m: Mat3) -> Mat3", "transpose(m: Mat4) -> Mat4", "transpose(m: DMat4) -> DMat4"],
            "The matrix with its rows and columns exchanged.", None, false),
        fixed("determinant", "math", &["determinant(m: Mat2) -> Float", "determinant(m: Mat3) -> Float", "determinant(m: Mat4) -> Float", "determinant(m: DMat4) -> Double"],
            "The determinant of a square matrix, expanded along the first row.", None, false),
        fixed("inverse", "math", &["inverse(m: Mat2) -> Mat2", "inverse(m: Mat3) -> Mat3", "inverse(m: Mat4) -> Mat4", "inverse(m: DMat4) -> DMat4"],
            "The matrix that multiplies m to the identity: its adjugate over its determinant; a singular m gives no finite numbers.", None, false),
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
        fixed("at", "field", &["at(field: Rows<T>, index: Int) -> T", "at(field: Grid<T>, at: Vec2) -> T", "at(field: Grid<T>, at: IVec2) -> T", "at(field: Volume<T>, at: Vec3) -> T", "at(field: Volume<T>, at: IVec3) -> T"],
            "The element at an address: the index, or the coordinates floored, clamped into the Field (NaN as zero). A memory reads as the steps have left it, and as it was bound in a scan's initial value.", Some(Capability::FieldRead), false),
        fixed("fetch", "field", &["fetch(field: Rows<T>, index: Int) -> T", "fetch(field: Grid<T>, at: Vec2) -> T", "fetch(field: Grid<T>, at: IVec2) -> T", "fetch(field: Volume<T>, at: Vec3) -> T", "fetch(field: Volume<T>, at: IVec3) -> T"],
            "Another name for at.", Some(Capability::FieldRead), false),
        fixed("at_linear", "field", &["at_linear(field: Rows<T>, position: Float) -> T", "at_linear(field: Grid<T>, position: Vec2) -> T", "at_linear(field: Volume<T>, position: Vec3) -> T"],
            "The neighbouring elements blended by the fractional position (element centres at .5), T of floats only; clamped at the edges.", Some(Capability::FieldRead), false),
        fixed("sample", "field", &["sample(field: Rows<T>) -> T", "sample(field: Grid<T>) -> T", "sample(field: Volume<T>) -> T"],
            "The input's element at this element's own coordinates, clamped into the input.", Some(Capability::FieldRead), true),
        fixed("count", "field", &["count(field: Rows<T>) -> Int", "count(field: Grid<T>) -> Int", "count(field: Volume<T>) -> Int"],
            "How many elements the Field has.", Some(Capability::FieldRead), false),
        fixed("size", "field", &["size(field: Grid<T>) -> Vec2", "size(field: Volume<T>) -> Vec3"],
            "The Field's extent along each axis.", Some(Capability::FieldRead), false),
        fixed("isize", "field", &["isize(field: Rows<T>) -> Int", "isize(field: Grid<T>) -> IVec2", "isize(field: Volume<T>) -> IVec3"],
            "The Field's extent along each axis, as integers.", Some(Capability::FieldRead), false),
        fixed("put", "memory", &["put(memory: Rows<T>, index: Int, value: T) -> T", "put(memory: Grid<T>, at: Vec2, value: T) -> T", "put(memory: Grid<T>, at: IVec2, value: T) -> T", "put(memory: Volume<T>, at: Vec3, value: T) -> T", "put(memory: Volume<T>, at: IVec3, value: T) -> T"],
            "Writes an element of a memory port in step order (scan only; from the entry or a function with the memory as a parameter); an address outside is dropped, never clamped. Yields the value written.", Some(Capability::FieldWrite), false),
        fixed("emit", "scatter", &["emit(pair: P) -> P"],
            "Gives a scatter one more pair, a struct of an address into its target and a value of its element, as it returns one: an element's pairs are the ones it emits, in order, then the one it returns, and an address outside the target is dropped. Scatter only, from its body, under `min`, `max`, or a `sum` of integers, whose answer is the same in any order.", Some(Capability::FieldWrite), true),
        fixed("element", "context", &["element() -> Int"], "This element's index over rows; a scan's step.", None, true),
        fixed("coord", "context", &["coord() -> Vec2", "coord() -> Vec3"], "This element's column and row (and layer) over a grid or a volume, whole numbers from 0 with row 0 at the top; the element's centre is coord() + 0.5. Rows have element().", None, true),
        fixed("icoord", "context", &["icoord() -> IVec2", "icoord() -> IVec3"], "This element's coordinates as integers, for arithmetic on cells.", None, true),
        fixed("domain", "context", &["domain() -> Int", "domain() -> Vec2", "domain() -> Vec3"], "The output's extent in its shape.", None, true),
        fixed("idomain", "context", &["idomain() -> Int", "idomain() -> IVec2", "idomain() -> IVec3"], "The output's extent as integers.", None, true),
        fixed("index", "context", &["index() -> Int"], "The iteration of the innermost loop, or a scan's step outside every loop.", None, false),
        fixed("time", "context", &["time() -> Float"], "The session's time in seconds.", Some(Capability::TimeRead), false),
        fixed("rate", "context", &["rate() -> Float"], "The sound device's samples per second, zero without one.", Some(Capability::TimeRead), false),
        fixed("window", "context", &["window() -> Vec2"], "The window's size in physical pixels, zero by zero without one; a node's domain may take it.", Some(Capability::InputRead), false),
        fixed("scale", "context", &["scale() -> Float"], "The window's physical pixels per logical pixel, one without a window.", Some(Capability::InputRead), false),
        fixed("pointer", "context", &["pointer() -> Vec2"], "The pointer in the output's pixels from its top left; a player maps its window's into them.", Some(Capability::InputRead), false),
        fixed("pointer_down", "context", &["pointer_down() -> Bool"], "Whether the left button is held.", Some(Capability::InputRead), false),
        fixed("button", "context", &["button(n: Int) -> Bool"], "Whether button n is held: 0 left, 1 right, 2 middle of a pointer; a pad's from 8 (south, east, west, north, shoulders, thumbs, start, select).", Some(Capability::InputRead), false),
        fixed("stick", "context", &["stick(n: Int) -> Vec2"], "A pad's axis pair: 0 the left stick, 1 the right, 2 the two triggers (x left, y right). Sticks run -1 to 1 with y away from the person, triggers 0 to 1; all zero without a pad.", Some(Capability::InputRead), false),
        fixed("wheel", "context", &["wheel() -> Vec2"], "What the wheel scrolled since the last evaluation, in lines: x sideways, y positive away.", Some(Capability::InputRead), false),
        fixed("key", "context", &["key(code: Int) -> Bool"], "Whether the key with this code is held: letters and digits by uppercase ASCII, arrows 37 to 40, space 32, enter 13, shift 16, escape 27.", Some(Capability::InputRead), false),
        fixed("vec2", "construction", &["vec2(x: Float, y: Float) -> Vec2", "vec2(x: Int, y: Int) -> Vec2"], "A vector from components, Float or Int (widened), in order; v.xy, v.yx and the like are swizzles.", None, false),
        fixed("vec3", "construction", &["vec3(x: Float, y: Float, z: Float) -> Vec3", "vec3(x: Int, y: Int, z: Int) -> Vec3", "vec3(xy: Vec2, z: Float) -> Vec3", "vec3(x: Float, yz: Vec2) -> Vec3"], "A vector from components, Float, Int (widened) or shorter vectors, in order; v.xyz, v.zyx and the like are swizzles.", None, false),
        fixed("vec4", "construction", &["vec4(x: Float, y: Float, z: Float, w: Float) -> Vec4", "vec4(x: Int, y: Int, z: Int, w: Int) -> Vec4", "vec4(xyz: Vec3, w: Float) -> Vec4", "vec4(xy: Vec2, zw: Vec2) -> Vec4", "vec4(xy: Vec2, z: Float, w: Float) -> Vec4", "vec4(x: Float, yzw: Vec3) -> Vec4"], "A vector from components, Float, Int (widened) or shorter vectors, in order; v.xyz, v.wzyx, c.rgb and the like are swizzles.", None, false),
        fixed("ivec2", "construction", &["ivec2(x: Int, y: Int) -> IVec2"], "An integer vector from components: a cell, an offset; at and put take one as an address; v.yx and the like are swizzles.", None, false),
        fixed("ivec3", "construction", &["ivec3(x: Int, y: Int, z: Int) -> IVec3", "ivec3(xy: IVec2, z: Int) -> IVec3", "ivec3(x: Int, yz: IVec2) -> IVec3"], "An integer vector from components or a shorter integer vector: a cell, an offset; at and put take one as an address.", None, false),
        fixed("mat2", "construction", &["mat2(c0: Vec2, c1: Vec2) -> Mat2", "mat2(diagonal: Float) -> Mat2"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
        fixed("mat3", "construction", &["mat3(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3", "mat3(diagonal: Float) -> Mat3"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
        fixed("mat4", "construction", &["mat4(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Mat4", "mat4(diagonal: Float) -> Mat4"], "A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.", None, false),
        fixed("dvec2", "construction", &["dvec2(x: Double, y: Double) -> DVec2"], "A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact, so dvec2(0.1, 0.2) holds the doubles nearest a tenth and a fifth.", None, false),
        fixed("dvec3", "construction", &["dvec3(x: Double, y: Double, z: Double) -> DVec3", "dvec3(xy: DVec2, z: Double) -> DVec3", "dvec3(v: Vec3) -> DVec3"], "A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact.", None, false),
        fixed("dvec4", "construction", &["dvec4(x: Double, y: Double, z: Double, w: Double) -> DVec4", "dvec4(xyz: DVec3, w: Double) -> DVec4", "dvec4(v: Vec4) -> DVec4"], "A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact.", None, false),
        fixed("lvec2", "construction", &["lvec2(x: Long, y: Long) -> LVec2", "lvec2(v: IVec2) -> LVec2"], "A vector of longs from components or shorter integer vectors, an Int widened.", None, false),
        fixed("lvec3", "construction", &["lvec3(x: Long, y: Long, z: Long) -> LVec3", "lvec3(xy: LVec2, z: Long) -> LVec3", "lvec3(v: IVec3) -> LVec3"], "A vector of longs from components or shorter integer vectors, an Int widened.", None, false),
        fixed("dmat2", "construction", &["dmat2(c0: DVec2, c1: DVec2) -> DMat2", "dmat2(diagonal: Double) -> DMat2"], "A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.", None, false),
        fixed("dmat3", "construction", &["dmat3(c0: DVec3, c1: DVec3, c2: DVec3) -> DMat3", "dmat3(diagonal: Double) -> DMat3"], "A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.", None, false),
        fixed("dmat4", "construction", &["dmat4(c0: DVec4, c1: DVec4, c2: DVec4, c3: DVec4) -> DMat4", "dmat4(diagonal: Double) -> DMat4"], "A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.", None, false),
    ]);
    out
}

/// One form of the language, as `help` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    /// The word to ask for: `loop`, `each`, `scatter`, `node`.
    pub name: &'static str,
    /// Where it is written: "type", "expression", "kernel", "graph" or
    /// "document".
    pub kind: &'static str,
    /// The shape, as one writes it.
    pub shape: &'static str,
    /// One sentence.
    pub doc: &'static str,
}

const fn form(
    name: &'static str,
    kind: &'static str,
    shape: &'static str,
    doc: &'static str,
) -> Form {
    Form {
        name,
        kind,
        shape,
        doc,
    }
}

/// Every form, grouped by where it is written.
pub fn forms() -> Vec<Form> {
    vec![
        form(
            "Rows",
            "type",
            "Rows<T>",
            "A row of elements, element() the index of one; count(f) says how many, at(f, i) reads one.",
        ),
        form(
            "Grid",
            "type",
            "Grid<T>",
            "A grid of elements, coord() the column and row of one; size(f) and at(f, vec2) read it.",
        ),
        form(
            "Volume",
            "type",
            "Volume<T>",
            "A volume of elements, coord() the column, row and layer of one.",
        ),
        form(
            "Image",
            "type",
            "Image",
            "Grid<Vec4>: a picture, each channel a Float from 0 to 1, the first output that is one is what a scene shows.",
        ),
        form(
            "Float",
            "type",
            "Float",
            "A 32-bit float; written 1.0, never 1.",
        ),
        form(
            "Int",
            "type",
            "Int",
            "A 32-bit integer; an overflow fails the element, add_wrap and its kin wrap.",
        ),
        form("Bool", "type", "Bool", "true or false."),
        form(
            "Double",
            "type",
            "Double",
            "A 64-bit float; a literal is taken exact.",
        ),
        form("Long", "type", "Long", "A 64-bit integer."),
        form(
            "Vec2",
            "type",
            "Vec2   Vec3   Vec4",
            "Floats in two, three or four components: v.x, v.y, v.z, v.w, and v.rgb, v.xy and the like as swizzles.",
        ),
        form("Vec3", "type", "Vec3", "Three Floats; see Vec2."),
        form(
            "Vec4",
            "type",
            "Vec4",
            "Four Floats, a pixel's r g b a; see Vec2.",
        ),
        form(
            "IVec2",
            "type",
            "IVec2   IVec3",
            "Integers in two or three components: a cell, an offset, an address for at and put.",
        ),
        form("IVec3", "type", "IVec3", "Three integers; see IVec2."),
        form(
            "Mat2",
            "type",
            "Mat2   Mat3   Mat4",
            "Float matrices by columns: m[i] a column, m * v applies it.",
        ),
        form("Mat3", "type", "Mat3", "A 3 by 3 Float matrix; see Mat2."),
        form("Mat4", "type", "Mat4", "A 4 by 4 Float matrix; see Mat2."),
        form(
            "DVec2",
            "type",
            "DVec2   DVec3   DVec4",
            "Doubles in two, three or four components.",
        ),
        form("DVec3", "type", "DVec3", "Three Doubles."),
        form("DVec4", "type", "DVec4", "Four Doubles."),
        form(
            "LVec2",
            "type",
            "LVec2   LVec3",
            "Longs in two or three components.",
        ),
        form("LVec3", "type", "LVec3", "Three Longs."),
        form(
            "DMat2",
            "type",
            "DMat2   DMat3   DMat4",
            "Double matrices by columns.",
        ),
        form("DMat3", "type", "DMat3", "A 3 by 3 Double matrix."),
        form("DMat4", "type", "DMat4", "A 4 by 4 Double matrix."),
        form(
            "as",
            "type",
            "input albedo: Grid<Vec4 as unorm8>;",
            "How a port's Field is stored: f16, unorm8, snorm8, unorm16, u8, i8, u16, i16, or word; a Kernel reads and writes the type, the storage is narrower.",
        ),
        form(
            "requires",
            "kernel",
            "requires { field.read; field.write; time.read; input.read; }",
            "The capabilities a Kernel uses, and no others: reading inputs, writing outputs, the clock, the pointer, keys and window.",
        ),
        form(
            "input",
            "kernel",
            "input x: Image;   = \"atlas.png\";   = text \"ORLA\";   = frames \"clip/\" at 30;",
            "In a Kernel, a Field it reads, bound by the node that runs it (`source <- other.result;`); in a scene, a Field the caller supplies, or one the Host reads from a file, from text, or frame by frame from a directory.",
        ),
        form(
            "param",
            "kernel",
            "param gain: Float = 1.0;",
            "In a Kernel, a value the node that runs it may set (`gain = 0.5;`), with its default; in a scene, a value the caller sets (`--param`, the protocol's `param`) and its nodes read by name.",
        ),
        form(
            "output",
            "kernel",
            "output result: Grid<Vec4>;   output shown <- node.port;   output kept <- node.port to \"saves/n.field\";",
            "In a Kernel, a Field it writes, an element per element of its domain (a map may write several); in a scene, a named result: the first picture is what it shows, the first row of one or two Floats (mono, stereo) what it plays, a block each frame joined to the last (`rate()` over the frame rate samples keeps it in time), and one written `to` a file is saved after each evaluation that changed it, where allowed.",
        ),
        form(
            "type",
            "kernel",
            "type Cells = Grid<Int>;   struct Cell { alive: Int; age: Float; }",
            "A name for a port type, or a struct of named fields, shared by the Kernels of a document.",
        ),
        form(
            "let",
            "expression",
            "let name = expr;   let { x, y: top } = value;",
            "A binding, immutable and scoped to its block; `let _ = put(...)` is a statement for its effect. A `{ ... }` names fields of one value instead, each under its own name or the one written after `:`.",
        ),
        form(
            "if",
            "expression",
            "if condition { a } else { b }   if condition { return a; }",
            "Both arms give one type and only the arm taken runs, which a trace shows; with no `else`, among a block's statements, it guards, and the rest of the block is its other arm.",
        ),
        form(
            "operators",
            "expression",
            "a + b   a - b   a * b   a / b   a % b   a << b   a >> b   a >>> b   a & b   a ^ b   a | b   a == b   a != b   a < b   a <= b   a > b   a >= b",
            "Arithmetic on numbers and, component by component, vectors, `/` and `%` on Ints truncating toward zero; on Ints only, `<<`, `>>` keeping the sign, `>>>` filling with zeros, and `&`, `^`, `|` bit by bit; a comparison gives a Bool. From the tightest: unary `-` and `!`; `*` `/` `%`; `+` `-`; the shifts; `&`; `^`; `|`; the comparisons; `&&`; `||`.",
        ),
        form(
            "and",
            "expression",
            "a && b   a || b   !a",
            "Truth: `&&` and `||` ask the other side only where the first settles nothing, so a guard guards and what it guards does not run.",
        ),
        form(
            "match",
            "expression",
            "match i { 0 => a, 1 | 2 => b, 8..16 => c, 16..=31 => d, _ => e }",
            "On an Int: the arm whose values hold it, else the `_` arm, which comes last and is required; the GPU runs it as a switch. A range is the values it holds, `..` stopping before its end and `..=` taking it, at most 256 to an arm.",
        ),
        form(
            "loop",
            "expression",
            "loop n with x = init until done { next }",
            "A bounded fold: `n` is a literal, which is its bound, or any Int expression read once and clamped to 0..=65536; `index()` is the iteration, and `until` stops it early, as `break` does from inside.",
        ),
        form(
            "break",
            "expression",
            "loop n with x = init { if done { break x; } next }",
            "Among a loop body's statements: the loop stops after this iteration and gives that value, which is of the type it carries; the iterations after it do not run, and a `break` belongs to the innermost loop around it.",
        ),
        form(
            "each",
            "expression",
            "[each n { body }]",
            "An array built element by element, unrolled, with `index()` its position; a loop inside the body has its own `index()`.",
        ),
        form(
            "array",
            "expression",
            "[a, b, c]   [value; n]   a[i]",
            "A fixed-size array of one element type, or `n` copies of a value; `a[i]` takes an element, clamped, and a literal index is checked when compiled.",
        ),
        form(
            "struct",
            "expression",
            "Name { a: 1.0, b: 2 }   Name { ..base, a: 3.0 }",
            "A value of a declared struct, or a copy of one with some fields replaced.",
        ),
        form(
            "fn",
            "kernel",
            "fn name(a: Float) -> Float { ... }   fn name<T>(a: T) -> T { ... }",
            "A function of the document or the Kernel; it calls only what is declared above it, and so never itself, and may take a Field parameter. A `<T>` names a type parameter, bound by the type of the argument written for it, and one instance is compiled per set of types a call gives it.",
        ),
        form(
            "kernel",
            "kernel",
            "kernel id@version { requires { ... } input ... output ... compute { ... } }",
            "A Kernel: what it needs, its ports and parameters, and one of `compute`, `scan`, `scatter` or `prefix`.",
        ),
        form(
            "budget",
            "kernel",
            "budget 100000;",
            "The loop iterations one element may run (a step of a scan, a merge of a reduction), 16777216 unless declared, a loop's added as it finishes; an element past it fails the evaluation on every backend, as an overflow does, and the running system keeps its last state. A run of all the elements together is bounded by the Host (`--work`).",
        ),
        form(
            "compute",
            "kernel",
            "compute { return value; }",
            "The map mode: the body runs once per element of the node's domain and gives that element.",
        ),
        form(
            "merge",
            "kernel",
            "compute { ... } merge(a, b) { ... }",
            "The reduce mode: `compute` runs per element of the first input and `merge` folds the results pairwise into one element; the output may be written as its element type.",
        ),
        form(
            "scan",
            "kernel",
            "scan with y = init { return next; }",
            "The ordered mode: one element per element of the first input, in order, carrying `y` from the step before; `memory` ports are read and written step by step.",
        ),
        form(
            "prefix",
            "kernel",
            "prefix with sum { return value; }",
            "The prefix mode: the body runs once per element of the first input, a `Rows`, and that element of the output is its value combined with every value before it by `sum`, `min` or `max`. The rule must give one answer whatever order it is folded in, so a float `sum` is refused (a `scan` walks that in element order) and a device folds the row in blocks rather than walking it.",
        ),
        form(
            "scatter",
            "kernel",
            "scatter into target with sum { return Pair { at: address, value: v }; }",
            "The scatter mode: the body runs once per element of the node's domain and gives a struct the document declares, of two fields, an address into the target and a value of its element (`struct Pair { at: Int; value: Float; }`); the value is combined into the target at the address with `sum`, `min`, `max` or `last`, in element order.",
        ),
        form(
            "memory",
            "kernel",
            "memory ram: Rows<Int>;   put(ram, address, value)",
            "A port an ordered Kernel both reads and writes, step by step; it comes out as an output.",
        ),
        form(
            "node",
            "graph",
            "node name: fx.glow@1 { port <- source.result; param = 2; domain = vec2(w, h); }",
            "A node runs a Kernel: its inputs wired at the consumer, its parameters set, its extent explicit or the first input's.",
        ),
        form(
            "inline",
            "graph",
            "node name: kernel { ... input x: Image <- src; ... }",
            "A Kernel written in place; its inputs carry their wiring, `<-` or `<~` with its initial source (`input s: Rows<Float> <~ name.result from start.result;`), a param whose default names a scene param follows that param (`param gain: Float = amount;`), and the session publishes it under the document's name.",
        ),
        form(
            "fill",
            "graph",
            "node zeros: Rows<Float> { domain = 65; value = 0.5; }",
            "A node that is a Field rather than a Kernel: its element's zero, or the `value` given, over the domain. Its element is a Float, Int, Bool, Vec2, Vec3 or Vec4 (a pixel for `Image`), or an alias of one; a struct is filled by a Kernel.",
        ),
        form(
            "graph",
            "graph",
            "node name: graph \"other.osn\" { source <- src; param = 2; }",
            "Another document placed as a node: its nodes join under `name.`, its inputs take the wiring, its outputs are read as `name.output`.",
        ),
        form(
            "previous",
            "graph",
            "port <~ node.output from initial [migrate id@v];",
            "State: the port reads what the last evaluation committed, the initial source before the first, and a migration carries it when a republished Kernel changes its layout; the Kernels reading the port are republished for the new layout with it.",
        ),
        form(
            "times",
            "graph",
            "times = 5;   carry port <- output;",
            "A node written once and placed N times, each wired as written, `carry` feeding a port from the previous time's output; at N = 0, a carried output is what its port is first bound to.",
        ),
        form(
            "async",
            "graph",
            "async = true;",
            "The node runs off the frame: the evaluation starts it on a thread and goes on, its readers taking the last output until the run lands.",
        ),
        form(
            "use",
            "document",
            "use \"kernels/\";   use \"name.orlapkg\";",
            "The documents whose Kernels and declarations this one brings in: a file, a directory, or a package.",
        ),
        form(
            "package",
            "document",
            "package name@1 { language = 1; ir = 1; }",
            "The manifest of a package directory: what it is called and the least language and IR it needs.",
        ),
    ]
}

/// The form named `name`, if any.
pub fn form_named(name: &str) -> Option<Form> {
    forms().into_iter().find(|f| f.name == name)
}

/// The builtin named `name`, if any.
pub fn builtin(name: &str) -> Option<Builtin> {
    builtins().into_iter().find(|b| b.name == name)
}

/// Builtins and forms whose names are close to `name`, for a miss.
pub fn similar(name: &str) -> Vec<&'static str> {
    let close = |candidate: &str| {
        candidate.contains(name)
            || name.contains(candidate)
            || orla_language::edit_distance(candidate, name) <= 2
    };
    builtins()
        .into_iter()
        .filter(|b| close(b.name))
        .map(|b| b.name)
        .chain(
            forms()
                .into_iter()
                .filter(|f| close(f.name))
                .map(|f| f.name),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name the call lowering dispatches on is one of the tables
    /// [`crate::lower::callable`] is built from, so a builtin added as a
    /// dispatch arm is one help must list.
    #[test]
    fn every_dispatched_name_is_in_a_table() {
        let source = include_str!("lower/builtins.rs");
        let callable = crate::lower::callable();
        let mut wrong = Vec::new();
        let mut seen = 0;
        for line in source.lines() {
            let arm = line.trim_start().trim_start_matches('|').trim_start();
            if !arm.starts_with('"') || !arm.contains("=>") {
                continue;
            }
            let names = arm.split("=>").next().unwrap_or_default();
            for name in names.split('|').map(|n| n.trim().trim_matches('"')) {
                seen += 1;
                if !name.is_empty() && !callable.contains(&name) {
                    wrong.push(name.to_owned());
                }
            }
        }
        assert!(seen > 40, "the dispatch arms were not found ({seen})");
        assert!(wrong.is_empty(), "dispatched but in no table: {wrong:?}");
    }

    /// Help lists what a call resolves to, no more and no less: a builtin
    /// the compiler takes and help leaves out, or one help lists and the
    /// compiler does not take, is named.
    #[test]
    fn help_lists_every_builtin_a_call_resolves_to() {
        let listed: std::collections::BTreeSet<&str> = builtins().iter().map(|b| b.name).collect();
        let callable: std::collections::BTreeSet<&str> =
            crate::lower::callable().into_iter().collect();
        let unlisted: Vec<_> = callable.difference(&listed).collect();
        let unknown: Vec<_> = listed.difference(&callable).collect();
        assert!(
            unlisted.is_empty() && unknown.is_empty(),
            "the compiler takes {unlisted:?} that help leaves out; help lists {unknown:?} it does not take"
        );
    }

    #[test]
    fn every_builtin_and_form_has_one_name_and_one_sentence() {
        let mut names: Vec<&str> = builtins().iter().map(|b| b.name).collect();
        names.extend(forms().iter().map(|f| f.name));
        let mut seen = std::collections::BTreeSet::new();
        for name in &names {
            assert!(seen.insert(*name), "`{name}` is listed twice");
        }
        for form in forms() {
            assert!(!form.shape.is_empty(), "{} has no shape", form.name);
            assert!(form.doc.ends_with('.'), "{} reads oddly", form.name);
            assert!(
                ["type", "expression", "kernel", "graph", "document"].contains(&form.kind),
                "{} is written where?",
                form.name
            );
            assert_eq!(form_named(form.name), Some(form));
        }
        assert_eq!(form_named("loop").unwrap().kind, "expression");
        assert!(builtin("mix").is_some());
        assert!(similar("lopo").contains(&"loop"));
    }
}
