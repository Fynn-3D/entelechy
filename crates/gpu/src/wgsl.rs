//! WGSL emission from validated Orla IR.
//!
//! Every value slot becomes a WGSL `let` (straight-line values) or `var`
//! (results of `if` and `loop`). Blocks map to WGSL blocks, so IR scoping
//! and WGSL scoping coincide. Resources are storage buffers of `u32` words,
//! one per scalar, plus two small uniform arrays for context and parameters.

use std::fmt::Write;

use orla_abi::{Encoding, Scalar, encoding};
use orla_ir::{
    BinaryOp, Block, Combine, Function, Instruction, Intrinsic, Mode, Module, Op, Shape, Type,
    UnaryOp, Value, ValueId,
};

/// Binding index of the context uniform.
pub const CONTEXT_BINDING: u32 = 0;
/// Binding index of the parameter uniform.
pub const PARAMS_BINDING: u32 = 1;
/// Binding index of the output storage buffer.
pub const OUTPUT_BINDING: u32 = 2;
/// Binding index of the first input storage buffer; input `i` is at
/// `INPUT_BINDING + i`.
pub const INPUT_BINDING: u32 = 3;

/// How many outputs the main pass writes: a map's every output, each to a
/// buffer of its own; one for the other modes, whose single output is
/// what their later passes work on.
pub fn written_outputs(module: &Module) -> usize {
    match module.manifest.mode {
        Mode::Map => module.manifest.outputs.len().max(1),
        _ => 1,
    }
}

/// Binding index of output `k`: the first at [`OUTPUT_BINDING`], the
/// others after the inputs, so no input moves when a map has several.
pub fn output_binding(module: &Module, k: usize) -> u32 {
    if k == 0 {
        OUTPUT_BINDING
    } else {
        INPUT_BINDING + (module.manifest.inputs.len() + k - 1) as u32
    }
}

/// Workgroup edge length in the first two dimensions.
pub const WORKGROUP: u32 = 8;
/// Context slots before the per-input extents: `[width, height, time,
/// button bits]`, `[pointer.x, pointer.y, depth, rate]`, two slots of key
/// bits, `[wheel.x, wheel.y, window.x, window.y]`, and `[scale, fold, 0,
/// 0]`, where `fold` is the elements each unit of `gid.y` stands for
/// when a long `Rows` domain is dispatched as a grid of workgroups (zero
/// otherwise). Each input then has `[width, height, depth, 0]`.
pub const CONTEXT_HEAD: usize = 8;

/// The context word holding 1, as an expression: a value the shader
/// compiler cannot fold, multiplied into an integer dividend so that a
/// value it narrowed (the top sixteen bits of a word, say) is a whole
/// `i32` again when it divides. Metal divided such a narrowed value by
/// a constant wrongly (`(x >>> 16) % 7` came out 8192 for a zero).
pub const OPAQUE_ONE: &str = "ctx[5].z";

/// The most workgroups one dispatch dimension takes.
pub const MAX_GROUPS: u32 = 65535;

/// How a map over `domain` is dispatched: the workgroups in x, y and z,
/// and the fold (elements per unit of `gid.y`) when rows are too many
/// for one dimension.
pub fn dispatch(domain: orla_abi::Domain) -> (u32, u32, u32, u32) {
    let groups_x = domain.width().div_ceil(WORKGROUP);
    if domain.shape() == orla_abi::Shape::Rows && groups_x > MAX_GROUPS {
        let fold = MAX_GROUPS * WORKGROUP;
        let strips = domain.width().div_ceil(fold);
        (MAX_GROUPS, strips.div_ceil(WORKGROUP), 1, fold)
    } else {
        (
            groups_x,
            domain.height().div_ceil(WORKGROUP),
            domain.depth(),
            0,
        )
    }
}

/// The path to field `k` of a struct value: a named field, or an
/// array's element.
fn field_path(module: &Module, index: u32, k: usize) -> String {
    if module.structs[index as usize].is_array() {
        format!("[{k}]")
    } else {
        format!(".f{k}")
    }
}

/// The declaration of struct `index`: a WGSL struct, or an alias of an
/// array, which builds with the same `S{index}(...)` constructor.
fn struct_decl(module: &Module, index: usize) -> String {
    let def = &module.structs[index];
    match def.element {
        Some(element) => format!(
            "alias S{index} = array<{}, {}>;",
            wgsl_type(element),
            def.fields.len()
        ),
        None => {
            let fields = def
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| format!("f{k}: {}", wgsl_type(field.ty)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("struct S{index} {{ {fields} }}")
        }
    }
}

fn wgsl_type(ty: Type) -> String {
    match ty {
        Type::Scalar(kind) => scalar_type(kind).into(),
        Type::Vector(kind, n) => format!("vec{n}<{}>", scalar_type(kind)),
        Type::Matrix(kind, n) => format!("mat{n}x{n}<{}>", scalar_type(kind)),
        Type::Struct(index) => format!("S{index}"),
    }
}

/// The WGSL scalar a kind is; a `Double` never reaches a shader.
fn scalar_type(kind: Scalar) -> &'static str {
    match kind {
        Scalar::Float => "f32",
        Scalar::Int => "i32",
        Scalar::Bool => "bool",
        Scalar::Long => "i64",
        Scalar::Double => unreachable!("the GPU refuses a module holding a Double"),
    }
}

/// `f(m[0]), f(m[1]), ...` over a matrix's columns, for the builtins
/// WGSL has for vectors but not for matrices.
fn per_column(ty: Type, f: impl Fn(usize) -> String) -> String {
    let n = usize::from(ty.order().unwrap_or(0));
    let columns = (0..n).map(f).collect::<Vec<_>>().join(", ");
    format!("{}({columns})", wgsl_type(ty))
}

fn literal(value: &Value) -> String {
    let f = |v: f32| {
        if v.is_nan() {
            "(0.0 / 0.0)".to_owned()
        } else if v.is_infinite() {
            if v > 0.0 {
                "(1.0 / 0.0)".to_owned()
            } else {
                "(-1.0 / 0.0)".to_owned()
            }
        } else {
            format!("{v:?}")
        }
    };
    let ints = |v: &[i32]| {
        v.iter()
            .map(|i| literal(&Value::Int(*i)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let floats = |v: &[f32]| v.iter().map(|x| f(*x)).collect::<Vec<_>>().join(", ");
    let longs = |v: &[i64]| {
        v.iter()
            .map(|l| long_literal(*l))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match value {
        Value::Float(v) => f(*v),
        // WGSL has no literal for the most negative int; its bits do.
        Value::Int(i32::MIN) => "bitcast<i32>(0x80000000u)".to_owned(),
        Value::Int(v) => format!("{v}i"),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2<f32>({}, {})", f(*x), f(*y)),
        Value::Vec3([x, y, z]) => format!("vec3<f32>({}, {}, {})", f(*x), f(*y), f(*z)),
        Value::Vec4([x, y, z, w]) => {
            format!("vec4<f32>({}, {}, {}, {})", f(*x), f(*y), f(*z), f(*w))
        }
        Value::IVec2(v) => format!("vec2<i32>({})", ints(v)),
        Value::IVec3(v) => format!("vec3<i32>({})", ints(v)),
        Value::Mat2(v) => format!("mat2x2<f32>({})", floats(v)),
        Value::Mat3(v) => format!("mat3x3<f32>({})", floats(v)),
        Value::Mat4(v) => format!("mat4x4<f32>({})", floats(v)),
        Value::Long(v) => long_literal(*v),
        Value::LVec2(v) => format!("vec2<i64>({})", longs(v)),
        Value::LVec3(v) => format!("vec3<i64>({})", longs(v)),
        // The validator refuses a constant record: a struct or an array
        // is built by `Construct`.
        Value::Record(_) => unreachable!("a constant of the IR is never a record"),
        other => unreachable!("the GPU refuses a module holding {other}"),
    }
}

/// A `Long` as WGSL writes one; the most negative has no literal, and
/// is one below the next.
fn long_literal(v: i64) -> String {
    if v == i64::MIN {
        "(-9223372036854775807li - 1li)".to_owned()
    } else {
        format!("{v}li")
    }
}

/// What a module holding a `Long` needs besides WGSL's own: a division
/// and a remainder (the device compiler refuses the ones naga writes for
/// 64 bits), a sign, and a float's conversion, each the reference's
/// rule, and none of them through `select`, which the device compiler
/// cannot pick for a 64-bit integer.
const LONG_HELPERS: &str = "
// The quotient and remainder of magnitudes, a bit at a time.
fn udivmod_u64(n: u64, d: u64) -> vec2<u64> {
    var q: u64 = 0lu;
    var r: u64 = 0lu;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        r = (r << 1u) | ((n >> u32(i)) & 1lu);
        if (r >= d) {
            r = r - d;
            q = q | (1lu << u32(i));
        }
    }
    return vec2<u64>(q, r);
}
// Truncated toward zero, the remainder taking the dividend's sign; a
// zero divisor gives the dividend and a zero remainder, and the most
// negative over minus one gives itself, as WGSL does for an Int.
fn div_i64(a: i64, b: i64) -> i64 {
    if (b == 0li) { return a; }
    if (b == -1li) { return -a; }
    var ua = bitcast<u64>(a);
    if (a < 0li) { ua = bitcast<u64>(-a); }
    var ub = bitcast<u64>(b);
    if (b < 0li) { ub = bitcast<u64>(-b); }
    let q = bitcast<i64>(udivmod_u64(ua, ub).x);
    if ((a < 0li) != (b < 0li)) { return -q; }
    return q;
}
fn rem_i64(a: i64, b: i64) -> i64 {
    if (b == 0li || b == -1li) { return 0li; }
    var ua = bitcast<u64>(a);
    if (a < 0li) { ua = bitcast<u64>(-a); }
    var ub = bitcast<u64>(b);
    if (b < 0li) { ub = bitcast<u64>(-b); }
    let r = bitcast<i64>(udivmod_u64(ua, ub).y);
    if (a < 0li) { return -r; }
    return r;
}
fn div_i64x2(a: vec2<i64>, b: vec2<i64>) -> vec2<i64> {
    return vec2<i64>(div_i64(a.x, b.x), div_i64(a.y, b.y));
}
fn div_i64x3(a: vec3<i64>, b: vec3<i64>) -> vec3<i64> {
    return vec3<i64>(div_i64(a.x, b.x), div_i64(a.y, b.y), div_i64(a.z, b.z));
}
fn rem_i64x2(a: vec2<i64>, b: vec2<i64>) -> vec2<i64> {
    return vec2<i64>(rem_i64(a.x, b.x), rem_i64(a.y, b.y));
}
fn rem_i64x3(a: vec3<i64>, b: vec3<i64>) -> vec3<i64> {
    return vec3<i64>(rem_i64(a.x, b.x), rem_i64(a.y, b.y), rem_i64(a.z, b.z));
}
fn sign_i64(x: i64) -> i64 {
    if (x > 0li) { return 1li; }
    if (x < 0li) { return -1li; }
    return 0li;
}
fn sign_i64x2(v: vec2<i64>) -> vec2<i64> {
    return vec2<i64>(sign_i64(v.x), sign_i64(v.y));
}
fn sign_i64x3(v: vec3<i64>) -> vec3<i64> {
    return vec3<i64>(sign_i64(v.x), sign_i64(v.y), sign_i64(v.z));
}
// A Float to a Long as the reference converts it: truncated, saturating
// at the ends, and zero for what is not a number.
fn to_long(x: f32) -> i64 {
    if (is_nan(x)) { return 0li; }
    if (x >= 9223372036854775808.0) { return 9223372036854775807li; }
    if (x <= -9223372036854775808.0) { return -9223372036854775807li - 1li; }
    return i64(x);
}
fn to_long2(v: vec2<f32>) -> vec2<i64> {
    return vec2<i64>(to_long(v.x), to_long(v.y));
}
fn to_long3(v: vec3<f32>) -> vec3<i64> {
    return vec3<i64>(to_long(v.x), to_long(v.y), to_long(v.z));
}
// The words of a Long, the low one first.
fn long_low(x: i64) -> u32 {
    return u32(bitcast<u64>(x) & 0xFFFFFFFFlu);
}
fn long_high(x: i64) -> u32 {
    return u32(bitcast<u64>(x) >> 32u);
}
fn long_of(low: u32, high: u32) -> i64 {
    return bitcast<i64>((u64(high) << 32u) | u64(low));
}
";

/// The preamble both shaders share: the context and params uniforms, the
/// output buffer when `with_output`, every input's buffer, the structs,
/// the key and button readers, every input's loader and address, the
/// normalizers, mixers, linear readers and memory writers. A `merge` may
/// read inputs, params and the context like any function, so the fold
/// shader carries the same.
fn emit_preamble(out: &mut String, module: &Module, with_output: bool) {
    let manifest = &module.manifest;
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    let params_len = param_slots(module).1.max(1);
    let _ = writeln!(
        out,
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({PARAMS_BINDING}) var<uniform> params: array<vec4<u32>, {params_len}>;"
    );
    if with_output {
        for k in 0..written_outputs(module) {
            // Elements that share words are stored by atomic or.
            let element = if output_shares_words(module, k) {
                "atomic<u32>"
            } else {
                "u32"
            };
            let _ = writeln!(
                out,
                "@group(0) @binding({}) var<storage, read_write> out{k}: array<{element}>;",
                output_binding(module, k)
            );
        }
    }
    for (index, port) in manifest.inputs.iter().enumerate() {
        // A memory is written by the steps, so its binding is read-write.
        let access = if manifest.memories.contains(&port.name) {
            "read_write"
        } else {
            "read"
        };
        let _ = writeln!(
            out,
            "@group(0) @binding({}) var<storage, {access}> in{index}: array<u32>;",
            INPUT_BINDING + index as u32,
        );
    }
    out.push('\n');
    for index in 0..module.structs.len() {
        let _ = writeln!(out, "{}", struct_decl(module, index));
    }
    let _ = writeln!(
        out,
        "fn key_down(code: i32) -> bool {{
    if (code < 0 || code > 255) {{ return false; }}
    let c = u32(code);
    let word = bitcast<u32>(ctx[2u + (c >> 7u)][(c >> 5u) & 3u]);
    return ((word >> (c & 31u)) & 1u) == 1u;
}}
fn stick_at(which: i32) -> vec2<f32> {{
    if (which == 0) {{ return ctx[6].xy; }}
    if (which == 1) {{ return ctx[6].zw; }}
    if (which == 2) {{ return ctx[7].xy; }}
    return vec2<f32>(0.0, 0.0);
}}
fn button_down(code: i32) -> bool {{
    if (code < 0 || code > 23) {{ return false; }}
    return ((u32(ctx[0].w) >> u32(code)) & 1u) == 1u;
}}
// A Float to an Int the way the reference converts it: truncated
// toward zero, saturating at the ends of the range, and zero for what
// is not a number. WGSL's own `i32(x)` clamps the float and not the
// integer, so two to the thirty-first comes out as 2147483520, and it
// leaves not-a-number to the driver — which answered the other end of
// the range.
fn to_int(x: f32) -> i32 {{
    // Not a number is the ones exponent with something in the
    // mantissa, tested on the word: `x != x` is what a compiler told
    // there are no NaNs folds to false, and this one is told that.
    if (is_nan(x)) {{ return 0; }}
    if (x >= 2147483648.0) {{ return 2147483647; }}
    if (x <= -2147483648.0) {{ return -2147483648; }}
    return i32(x);
}}
// The angle of (x, y) from the x axis, as the reference gives it where
// y is a zero: a device's own answers not-a-number when both are zero —
// which is the middle pixel of anything drawn in polar coordinates —
// and loses the sign of a zero y beside a negative x. Everything else
// is the device's.
// Whether x is not a number, on the words.
fn is_nan(x: f32) -> bool {{
    let b = bitcast<u32>(x);
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}}
fn atan2_ref(y: f32, x: f32) -> f32 {{
    let yb = bitcast<u32>(y);
    let xb = bitcast<u32>(x);
    if ((yb & 0x7FFFFFFFu) == 0u && !is_nan(x)) {{
        // Zero where x is positive, pi where it is negative or a
        // negative zero, and the sign of y carried onto it.
        let turn = select(0u, 0x40490FDBu, (xb >> 31u) == 1u);
        return bitcast<f32>((yb & 0x80000000u) | turn);
    }}
    return atan2(y, x);
}}
fn to_int2(v: vec2<f32>) -> vec2<i32> {{
    return vec2<i32>(to_int(v.x), to_int(v.y));
}}
fn to_int3(v: vec3<f32>) -> vec3<i32> {{
    return vec3<i32>(to_int(v.x), to_int(v.y), to_int(v.z));
}}"
    );
    if module.holds(Type::Long) {
        out.push_str(LONG_HELPERS);
    }
    for (index, port) in manifest.inputs.iter().enumerate() {
        let element = port.element();
        // Words an element takes, or bytes when the input keeps it narrow.
        let encoding = input_encoding(module, index);
        let stride = if encoding == Encoding::Word {
            port.ty.schema.value_words()
        } else {
            port.ty.schema.element_bytes()
        };
        let (expr, _) = load_expr(module, element, &format!("in{index}"), "b", 0, encoding);
        let _ = writeln!(
            out,
            "fn load_in{index}(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
            wgsl_type(element)
        );
        // The linear index of an address, each axis floored and clamped
        // into the input's extent (NaN counts as zero).
        let slot = CONTEXT_HEAD + index;
        let _ = match port.shape() {
            Shape::Rows => writeln!(
                out,
                "fn index_in{index}(i: i32) -> u32 {{
    return u32(clamp(i, 0i, i32(ctx[{slot}].x) - 1i));
}}"
            ),
            Shape::Grid => writeln!(
                out,
                "fn index_in{index}(c: vec2<f32>) -> u32 {{
    let w = ctx[{slot}].x;
    let h = ctx[{slot}].y;
    let x = clamp(select(floor(c.x), 0.0, c.x != c.x), 0.0, w - 1.0);
    let y = clamp(select(floor(c.y), 0.0, c.y != c.y), 0.0, h - 1.0);
    return u32(y) * u32(w) + u32(x);
}}"
            ),
            Shape::Volume => writeln!(
                out,
                "fn index_in{index}(c: vec3<f32>) -> u32 {{
    let w = ctx[{slot}].x;
    let h = ctx[{slot}].y;
    let d = ctx[{slot}].z;
    let x = clamp(select(floor(c.x), 0.0, c.x != c.x), 0.0, w - 1.0);
    let y = clamp(select(floor(c.y), 0.0, c.y != c.y), 0.0, h - 1.0);
    let z = clamp(select(floor(c.z), 0.0, c.z != c.z), 0.0, d - 1.0);
    return (u32(z) * u32(h) + u32(y)) * u32(w) + u32(x);
}}"
            ),
        };
    }
    // The truncated remainder exactly: WGSL's `%` rounds the quotient
    // before truncating it, so 7.5 % 0.3 comes out 0 where the CPU has
    // 0.3 - epsilon; the fused multiply-add keeps the remainder exact
    // and a remainder of the wrong sign is one divisor off.
    for ty in [Type::Float, Type::Vec2, Type::Vec3, Type::Vec4] {
        let name = wgsl_type(ty);
        let _ = writeln!(
            out,
            "fn fmod_{}(x: {name}, y: {name}) -> {name} {{
    let q = trunc(x / y);
    let r = fma(-q, y, x);
    let wrong = (r != {name}(0.0)) & ((r < {name}(0.0)) != (x < {name}(0.0)));
    return select(r, r + sign(x) * abs(y), wrong);
}}",
            ty.components().unwrap_or(1)
        );
    }
    // Integer division and remainder through a dividend the compiler
    // cannot narrow (see `OPAQUE_ONE`), scalar and vector.
    let _ = writeln!(
        out,
        "fn whole_i32(x: i32) -> i32 {{ return x * i32({OPAQUE_ONE}); }}
fn whole_u32(x: u32) -> u32 {{ return x * u32({OPAQUE_ONE}); }}
fn whole_v2(x: vec2<i32>) -> vec2<i32> {{ return x * i32({OPAQUE_ONE}); }}
fn whole_v3(x: vec3<i32>) -> vec3<i32> {{ return x * i32({OPAQUE_ONE}); }}"
    );
    // Rotations by a runtime count (WGSL refuses a shift by the width),
    // the high half of a 64-bit product from 16-bit halves, and unsigned
    // division through the signed one, which this GPU gets right where
    // its native `u32 / u32` is wrong for dividends with the top bit set.
    let _ = writeln!(
        out,
        "fn rotl_i32(x: i32, n: i32) -> i32 {{
    let k = u32(n) & 31u;
    return select(x, i32((u32(x) << k) | (u32(x) >> (32u - k))), k != 0u);
}}
fn rotr_i32(x: i32, n: i32) -> i32 {{
    let k = u32(n) & 31u;
    return select(x, i32((u32(x) >> k) | (u32(x) << (32u - k))), k != 0u);
}}
fn umulhi_u32(a: u32, b: u32) -> u32 {{
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}}
fn mulhi_i32(a: i32, b: i32) -> i32 {{
    let u = umulhi_u32(u32(a), u32(b));
    return i32(u - select(0u, u32(b), a < 0i) - select(0u, u32(a), b < 0i));
}}
fn udiv_u32(narrow: u32, b: u32) -> u32 {{
    let a = whole_u32(narrow);
    if (b >= 0x80000000u) {{ return select(0u, 1u, a >= b); }}
    if (a < 0x80000000u) {{ return u32(i32(a) / i32(b)); }}
    var q = u32(i32(a >> 1u) / i32(b)) << 1u;
    if (a - q * b >= b) {{ q = q + 1u; }}
    return q;
}}
fn urem_u32(a: u32, b: u32) -> u32 {{
    return a - udiv_u32(a, b) * b;
}}"
    );
    if any_narrow(module) {
        emit_encodings(out);
    }
    emit_mixers(out, module);
    for (index, port) in manifest.inputs.iter().enumerate() {
        emit_linear(out, module, index, port);
        if manifest.memories.contains(&port.name) {
            emit_memory_writer(out, module, index, port);
        }
    }
    out.push('\n');
}

/// Emits a complete compute shader for the module.
pub fn emit(module: &Module) -> String {
    let manifest = &module.manifest;
    let mut out = String::new();
    emit_preamble(&mut out, module, true);
    // One store per output the pass writes: a map's every output, into
    // its own buffer; a scatter's map pass stores the entry's pairs.
    let scattering = matches!(manifest.mode, Mode::Scatter { .. });
    for (k, port) in manifest
        .outputs
        .iter()
        .enumerate()
        .take(written_outputs(module))
    {
        let element = if scattering {
            result_type(&module.entry)
        } else {
            port.element()
        };
        let body = if !scattering && !port.ty.schema.is_wide() {
            narrow_store_body(module, k, element)
        } else {
            let stride = type_words(module, element);
            let mut stores = Vec::new();
            store_lines(module, element, "v", 0, &mut stores);
            format!(
                "    let b = index * {stride}u;\n{}",
                stores.join("\n").replace("out0[", &format!("out{k}["))
            )
        };
        let _ = writeln!(
            out,
            "fn store_out{k}(index: u32, v: {}) {{\n{body}\n}}",
            wgsl_type(element)
        );
    }
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(&mut out, module, function, &format!("f{index}"), false);
        out.push('\n');
    }
    emit_function(&mut out, module, &module.entry, "entry", true);
    if let Some(merge) = &module.merge {
        emit_function(&mut out, module, merge, "merge", false);
    }
    if let Some(init) = &module.init {
        emit_function(&mut out, module, init, "init", false);
        // Ordered work: one invocation walks the rows, carrying the value.
        let _ = writeln!(
            out,
            "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{
        return;
    }}
    let n = u32(ctx[0].x);
    var carried: {} = init();
    for (var i: u32 = 0u; i < n; i = i + 1u) {{
        storageBarrier();
        carried = entry(vec3<f32>(f32(i), 0.0, 0.0), i, carried);
        store_out0(i, carried);
    }}
}}",
            wgsl_type(
                init.results
                    .first()
                    .map_or(Type::Float, |r| result_type_of(init, *r))
            )
        );
        return out;
    }
    // A map of several outputs stores them from the entry itself.
    let store = if written_outputs(module) > 1 {
        "entry(coord, element);"
    } else {
        "store_out0(element, entry(coord, element));"
    };
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({WORKGROUP}, {WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let w = u32(ctx[0].x);
    let h = u32(ctx[0].y);
    let d = u32(ctx[1].z);
    let fold = u32(ctx[5].y);
    var x = gid.x;
    var y = gid.y;
    if (fold != 0u) {{
        x = gid.x + gid.y * fold;
        y = 0u;
    }}
    if (x >= w || y >= h || gid.z >= d) {{
        return;
    }}
    let element = (gid.z * h + y) * w + x;
    let coord = vec3<f32>(f32(x), f32(y), f32(gid.z));
    {store}
}}"
    );
    out
}

/// Words a value of a type takes in a buffer.
pub fn type_words(module: &Module, ty: Type) -> usize {
    match ty {
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .map(|field| type_words(module, field.ty))
            .sum(),
        other => other.words().unwrap_or(1),
    }
}

/// Binding index of the scatter pass's pairs, the map pass's output.
pub const SCATTER_PAIRS_BINDING: u32 = 1;
/// Binding index of the scatter pass's target, a copy of the base input.
pub const SCATTER_OUT_BINDING: u32 = 2;

/// The combine of two values of a type, as a WGSL expression: fieldwise
/// through structs, WGSL's own operators and builtins below.
fn combine_expr(module: &Module, ty: Type, combine: Combine, a: &str, b: &str) -> String {
    match ty {
        Type::Struct(index) => {
            let fields = module.structs[index as usize]
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| {
                    combine_expr(
                        module,
                        field.ty,
                        combine,
                        &format!("{a}{}", field_path(module, index, k)),
                        &format!("{b}{}", field_path(module, index, k)),
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("S{index}({fields})")
        }
        _ if ty.order().is_some() => match combine {
            Combine::Sum => format!("({a} + {b})"),
            Combine::Min => per_column(ty, |c| format!("min({a}[{c}], {b}[{c}])")),
            Combine::Max => per_column(ty, |c| format!("max({a}[{c}], {b}[{c}])")),
            Combine::Last => b.to_owned(),
        },
        _ => match combine {
            Combine::Sum => format!("({a} + {b})"),
            Combine::Min => format!("min({a}, {b})"),
            Combine::Max => format!("max({a}, {b})"),
            Combine::Last => b.to_owned(),
        },
    }
}

/// Binding index of the scatter pass's winners, for `last`: one word per
/// target element, the greatest source index plus one, zero for none.
pub const SCATTER_WINNER_BINDING: u32 = 3;
/// Workgroup length of the parallel scatter passes.
pub const SCATTER_WORKGROUP: u32 = 64;

/// How a scatter's combine runs on the GPU: order-free rules apply the
/// pairs in parallel with atomics; a float sum, whose rounding is the
/// element order, sorts the pairs by target and source and folds each
/// target's run in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterPlan {
    /// The pairs sorted by address then source index, then one invocation
    /// per target element folding its run in element order.
    Sorted,
    /// One invocation per pair, atomics on the target; `last` records the
    /// winning source and gathers afterwards; `keys` says float words are
    /// turned into ordered keys around the pass.
    Parallel {
        /// The rule is `last`.
        last: bool,
        /// Float words need keys for `min` and `max`.
        keys: bool,
    },
}

/// The plan for a scatter module.
pub fn scatter_plan(module: &Module) -> ScatterPlan {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return ScatterPlan::Sorted;
    };
    let scalars = module.manifest.inputs[base as usize].ty.schema.scalars();
    let floats = scalars.iter().any(|s| matches!(s, orla_abi::Scalar::Float));
    match combine {
        Combine::Sum if floats => ScatterPlan::Sorted,
        Combine::Sum => ScatterPlan::Parallel {
            last: false,
            keys: false,
        },
        Combine::Min | Combine::Max => ScatterPlan::Parallel {
            last: false,
            keys: floats,
        },
        Combine::Last => ScatterPlan::Parallel {
            last: true,
            keys: false,
        },
    }
}

/// The words of a value as WGSL `u32` expressions, one per word in layout
/// order, each with its scalar type.
fn value_words(module: &Module, ty: Type, path: &str, out: &mut Vec<(String, Type)>) {
    match ty {
        Type::Struct(index) => {
            for (k, field) in module.structs[index as usize].fields.iter().enumerate() {
                let path = format!("{path}{}", field_path(module, index, k));
                value_words(module, field.ty, &path, out);
            }
        }
        other => {
            for (expr, kind) in scalar_parts(other, path) {
                for word in stored_words(kind, &expr) {
                    out.push((word, Type::Scalar(kind)));
                }
            }
        }
    }
}

/// The header the scatter passes share: the context and the pairs, the
/// target when `target` says so (as atomics when `Some(true)`), and any
/// further bindings in `extra`; then the structs, the float key, and the
/// pair loader.
fn scatter_header(module: &Module, target: Option<bool>, extra: &str) -> String {
    let manifest = &module.manifest;
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({SCATTER_PAIRS_BINDING}) var<storage, read> pairs: array<u32>;"
    );
    if let Some(atomic) = target {
        let ty = if atomic {
            "array<atomic<i32>>"
        } else {
            "array<u32>"
        };
        let _ = writeln!(
            out,
            "@group(0) @binding({SCATTER_OUT_BINDING}) var<storage, read_write> out0: {ty};"
        );
    }
    out.push_str(extra);
    for index in 0..module.structs.len() {
        let _ = writeln!(out, "{}", struct_decl(module, index));
    }
    // The total order over floats as keys, an involution on the bits:
    // every negative below every positive, NaN beyond the infinities.
    let _ = writeln!(
        out,
        "fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}"
    );
    let pair = result_type(&module.entry);
    let pair_stride = type_words(module, pair);
    let (expr, _) = load_expr(module, pair, "pairs", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_pair(index: u32) -> {} {{
    let b = index * {pair_stride}u;
    return {expr};
}}",
        wgsl_type(pair)
    );
    out
}

/// The winners of `last`, one word per target element.
fn winners_binding(read_only: bool) -> String {
    let (access, ty) = if read_only {
        ("read", "array<u32>")
    } else {
        ("read_write", "array<atomic<u32>>")
    };
    format!("@group(0) @binding({SCATTER_WINNER_BINDING}) var<storage, {access}> winner: {ty};\n")
}

/// Binding index of the sort's keys: one target index per source, the
/// greatest value for a source that lands outside or pads the sort.
pub const SCATTER_SORT_KEYS_BINDING: u32 = 3;
/// Binding index of the sort's order: the source index each key came from.
pub const SCATTER_SORT_ORDER_BINDING: u32 = 4;
/// Binding index of the sort's parameters, a `vec4<u32>` of the padded
/// count and the step's `k` and `j`, one record per dispatch at a dynamic
/// offset.
pub const SCATTER_SORT_PARAMS_BINDING: u32 = 5;

/// The sort's bindings.
fn sort_bindings(read_only: bool) -> String {
    let access = if read_only { "read" } else { "read_write" };
    format!(
        "@group(0) @binding({SCATTER_SORT_KEYS_BINDING}) var<storage, {access}> keys: array<u32>;
@group(0) @binding({SCATTER_SORT_ORDER_BINDING}) var<storage, {access}> order: array<u32>;
@group(0) @binding({SCATTER_SORT_PARAMS_BINDING}) var<uniform> sort: vec4<u32>;
"
    )
}

/// The lines that turn a pair's address into `at`, leaving the invocation
/// with `exit` when it falls outside the target.
fn scatter_address(module: &Module, exit: &str) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let slot = CONTEXT_HEAD + base as usize;
    match target.shape() {
        // Named `row`, not `i`: the pass's own `i` is the source index.
        Shape::Rows => format!(
            "let row = p.f0;
    if (row < 0i || row >= i32(ctx[{slot}].x)) {{ {exit} }}
    let at = u32(row);"
        ),
        Shape::Grid => format!(
            "let x = floor(p.f0.x);
    let y = floor(p.f0.y);
    if (!(x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y)) {{ {exit} }}
    let at = u32(y) * u32(ctx[{slot}].x) + u32(x);"
        ),
        Shape::Volume => format!(
            "let x = floor(p.f0.x);
    let y = floor(p.f0.y);
    let z = floor(p.f0.z);
    if (!(x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y && z >= 0.0 && z < ctx[{slot}].z)) {{ {exit} }}
    let at = (u32(z) * u32(ctx[{slot}].y) + u32(y)) * u32(ctx[{slot}].x) + u32(x);"
        ),
    }
}

/// The prefix passes' bindings: the plan, the output being folded in
/// place, and one element a block.
pub const PREFIX_PLAN_BINDING: u32 = 0;
/// The output buffer the fold walks in place.
pub const PREFIX_OUT_BINDING: u32 = 1;
/// One element a block: each block's total, then those scanned.
pub const PREFIX_SUMS_BINDING: u32 = 2;
/// Invocations a workgroup in the prefix passes.
pub const PREFIX_WORKGROUP: u32 = 64;

/// One of the three passes a prefix fold takes on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixPass {
    /// One invocation a block: walk the block in place, leave its total.
    Blocks,
    /// One invocation: scan the blocks' totals.
    Tops,
    /// One invocation an element: take the block's offset.
    Offsets,
}

/// How two words meet under `combine`, by the word's scalar — the same
/// rule as `orla_rules::combine_words`, which is what holds the device
/// to the reference's words.
fn prefix_word(combine: Combine, scalar: orla_abi::Scalar, a: &str, b: &str) -> String {
    let float = matches!(scalar, orla_abi::Scalar::Float);
    match (combine, float) {
        (Combine::Sum, false) => {
            format!("bitcast<u32>(bitcast<i32>({a}) + bitcast<i32>({b}))")
        }
        (Combine::Sum, true) => {
            format!("bitcast<u32>(bitcast<f32>({a}) + bitcast<f32>({b}))")
        }
        // Floats meet by their bits' total order, as they do in a
        // scatter, so a not-a-number lands where the reference puts it.
        (Combine::Min, true) => {
            format!("select({a}, {b}, key_of(bitcast<i32>({b})) < key_of(bitcast<i32>({a})))")
        }
        (Combine::Max, true) => {
            format!("select({a}, {b}, key_of(bitcast<i32>({b})) > key_of(bitcast<i32>({a})))")
        }
        (Combine::Min, false) => {
            format!("bitcast<u32>(min(bitcast<i32>({a}), bitcast<i32>({b})))")
        }
        (Combine::Max, false) => {
            format!("bitcast<u32>(max(bitcast<i32>({a}), bitcast<i32>({b})))")
        }
        (Combine::Last, _) => b.to_owned(),
    }
}

/// Emits one of a prefix fold's three passes. The output is walked in
/// place as words, so the element may be anything the layout holds, and
/// nothing here needs shared memory or a barrier: a block is one
/// invocation's walk, the blocks' totals are another's, and the third
/// hands every element its block's offset.
pub fn emit_prefix(module: &Module, pass: PrefixPass) -> String {
    let Mode::Prefix { combine } = module.manifest.mode else {
        return String::new();
    };
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars().to_vec();
    let stride = scalars.len().max(1);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "@group(0) @binding({PREFIX_PLAN_BINDING}) var<uniform> plan: vec4<u32>;
@group(0) @binding({PREFIX_OUT_BINDING}) var<storage, read_write> out0: array<u32>;
@group(0) @binding({PREFIX_SUMS_BINDING}) var<storage, read_write> sums: array<u32>;
const STRIDE: u32 = {stride}u;
fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}"
    );
    let fold = (0..stride)
        .map(|k| {
            format!(
                "    r[{k}u] = {};",
                prefix_word(
                    combine,
                    scalars[k],
                    &format!("a[{k}u]"),
                    &format!("b[{k}u]")
                )
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let load = |from: &str| {
        (0..stride)
            .map(|k| format!("    v[{k}u] = {from}[b + {k}u];"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let store = |into: &str| {
        (0..stride)
            .map(|k| format!("    {into}[b + {k}u] = v[{k}u];"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let _ = writeln!(
        out,
        "alias Element = array<u32, {stride}u>;
fn fold(a: Element, b: Element) -> Element {{
    var r: Element;
{fold}
    return r;
}}
fn load_out(index: u32) -> Element {{
    let b = index * STRIDE;
    var v: Element;
{}
    return v;
}}
fn store_out(index: u32, v: Element) {{
    let b = index * STRIDE;
{}
}}
fn load_sum(index: u32) -> Element {{
    let b = index * STRIDE;
    var v: Element;
{}
    return v;
}}
fn store_sum(index: u32, v: Element) {{
    let b = index * STRIDE;
{}
}}",
        load("out0"),
        store("out0"),
        load("sums"),
        store("sums")
    );
    // plan.x: elements, plan.y: elements a block, plan.z: blocks.
    let body = match pass {
        PrefixPass::Blocks => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block = gid.x;
    if (block >= plan.z) {{
        return;
    }}
    let start = block * plan.y;
    let end = min(start + plan.y, plan.x);
    var running = load_out(start);
    for (var i = start + 1u; i < end; i = i + 1u) {{
        running = fold(running, load_out(i));
        store_out(i, running);
    }}
    store_sum(block, running);
}}"
        ),
        PrefixPass::Tops => "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var running = load_sum(0u);
    for (var i = 1u; i < plan.z; i = i + 1u) {
        running = fold(running, load_sum(i));
        store_sum(i, running);
    }
}"
        .to_owned(),
        PrefixPass::Offsets => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= plan.x) {{
        return;
    }}
    let block = i / plan.y;
    if (block == 0u) {{
        return;
    }}
    store_out(i, fold(load_sum(block - 1u), load_out(i)));
}}"
        ),
    };
    out.push_str(&body);
    out.push('\n');
    out
}

/// Emits the parallel apply pass: one invocation per pair, combining its
/// value into the target with atomics (`sum` of ints, `min` and `max` on
/// ordered keys) or recording the greatest source index per element
/// (`last`). Order-free, so the result is the sequential one.
pub fn emit_scatter_parallel(module: &Module) -> String {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let element = target.element();
    let stride = target.ty.schema.value_words();
    let mut out = scatter_header(module, Some(true), &winners_binding(false));
    let mut words = Vec::new();
    value_words(module, element, "p.f1", &mut words);
    let mut lines = Vec::new();
    match combine {
        Combine::Sum => {
            for (k, (expr, _)) in words.iter().enumerate() {
                lines.push(format!(
                    "    atomicAdd(&out0[b + {k}u], bitcast<i32>({expr}));"
                ));
            }
        }
        Combine::Min | Combine::Max => {
            let op = if combine == Combine::Min {
                "atomicMin"
            } else {
                "atomicMax"
            };
            for (k, (expr, ty)) in words.iter().enumerate() {
                let key = if *ty == Type::Float {
                    format!("key_of(bitcast<i32>({expr}))")
                } else {
                    format!("bitcast<i32>({expr})")
                };
                lines.push(format!("    {op}(&out0[b + {k}u], {key});"));
            }
        }
        Combine::Last => lines.push("    atomicMax(&winner[at], i + 1u);".to_owned()),
    }
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = u32(ctx[0].x) * u32(ctx[0].y) * u32(ctx[1].z);
    let i = gid.x;
    if (i >= n) {{
        return;
    }}
    let p = load_pair(i);
    {}
    let b = at * {stride}u;
{}
}}",
        scatter_address(module, "return;"),
        lines.join("\n")
    );
    out
}

/// Emits the gather pass of `last`: one invocation per target element,
/// taking the winning pair's value when there is one.
pub fn emit_scatter_gather(module: &Module) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), &winners_binding(true));
    let mut words = Vec::new();
    value_words(module, target.element(), "p.f1", &mut words);
    let stores = words
        .iter()
        .enumerate()
        .map(|(k, (expr, _))| format!("    out0[b + {k}u] = {expr};"))
        .collect::<Vec<_>>()
        .join("\n");
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x;
    if (e >= count) {{
        return;
    }}
    let w = winner[e];
    if (w == 0u) {{
        return;
    }}
    let p = load_pair(w - 1u);
    let b = e * {stride}u;
{stores}
}}"
    );
    out
}

/// Emits the pass that turns the target's float words into ordered keys
/// (`encode`) or back, around a parallel `min` or `max`.
pub fn emit_scatter_keys(module: &Module, encode: bool) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), "");
    // The key transform is an involution, so one function serves both ways.
    let _ = encode;
    let lines = target
        .ty
        .schema
        .scalars()
        .iter()
        .enumerate()
        .filter(|(_, scalar)| matches!(scalar, orla_abi::Scalar::Float))
        .map(|(k, _)| {
            format!("    out0[b + {k}u] = bitcast<u32>(key_of(bitcast<i32>(out0[b + {k}u])));")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x;
    if (e >= count) {{
        return;
    }}
    let b = e * {stride}u;
{lines}
}}"
    );
    out
}

/// Emits the sort's first pass: one invocation per slot of the padded
/// sort, writing a source's target index as its key and its index as its
/// order, the greatest key for a source outside the target or a pad.
pub fn emit_scatter_sort_keys(module: &Module) -> String {
    let mut out = scatter_header(module, None, &sort_bindings(false));
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= sort.x) {{
        return;
    }}
    let n = u32(ctx[0].x) * u32(ctx[0].y) * u32(ctx[1].z);
    order[i] = i;
    if (i >= n) {{
        keys[i] = 0xffffffffu;
        return;
    }}
    let p = load_pair(i);
    {}
    keys[i] = at;
}}",
        scatter_address(module, "keys[i] = 0xffffffffu; return;")
    );
    out
}

/// Emits one step of the bitonic sort over (key, order): the same shader
/// for every `k` and `j`, which the parameters carry.
pub fn emit_scatter_sort_step() -> String {
    format!(
        "{}
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let partner = i ^ sort.z;
    if (i >= sort.x || partner <= i) {{
        return;
    }}
    let ascending = (i & sort.y) == 0u;
    let ka = keys[i];
    let kb = keys[partner];
    let oa = order[i];
    let ob = order[partner];
    let a_greater = ka > kb || (ka == kb && oa > ob);
    if (ascending == a_greater) {{
        keys[i] = kb;
        keys[partner] = ka;
        order[i] = ob;
        order[partner] = oa;
    }}
}}",
        sort_bindings(false)
    )
}

/// Emits the fold: one invocation per target element finds its run in the
/// sorted keys and adds the pairs' values to the element in source order,
/// exactly as the CPU walks them.
pub fn emit_scatter_fold(module: &Module) -> String {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let element = target.element();
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), &sort_bindings(true));
    let (expr, _) = load_expr(module, element, "out0", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_out0(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
        wgsl_type(element)
    );
    let mut stores = Vec::new();
    store_lines(module, element, "v", 0, &mut stores);
    let _ = writeln!(
        out,
        "fn store_out0(index: u32, v: {}) {{
    let b = index * {stride}u;
{}
}}",
        wgsl_type(element),
        stores.join("\n")
    );
    let _ = writeln!(
        out,
        "fn combine(a: {ty}, b: {ty}) -> {ty} {{ return {}; }}",
        combine_expr(module, element, combine, "a", "b"),
        ty = wgsl_type(element)
    );
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x;
    if (e >= count) {{
        return;
    }}
    let n = sort.x;
    var lo = 0u;
    var hi = n;
    while (lo < hi) {{
        let mid = (lo + hi) / 2u;
        if (keys[mid] < e) {{
            lo = mid + 1u;
        }} else {{
            hi = mid;
        }}
    }}
    if (lo >= n || keys[lo] != e) {{
        return;
    }}
    var acc = load_out0(e);
    var pos = lo;
    loop {{
        if (pos >= n || keys[pos] != e) {{
            break;
        }}
        acc = combine(acc, load_pair(order[pos]).f1);
        pos = pos + 1u;
    }}
    store_out0(e, acc);
}}"
    );
    out
}

/// The type of one result of a function.
fn result_type_of(function: &Function, id: ValueId) -> Type {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .find_map(|(_, block)| find(&block.body, id))
                    .or_else(|| find(&otherwise.body, id)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, id).unwrap_or(Type::Float)
}

/// Workgroup length of the reduce step.
pub const REDUCE_WORKGROUP: u32 = 64;

/// Emits the reduce step of a reduce module: element `i` of the output
/// level is the run of at most [`orla_rules::reduce::FOLD`] elements
/// starting at `i * FOLD`, folded by the pairwise tree every backend
/// folds in — which is [`orla_rules::reduce::LEVELS_AT_ONCE`] of that
/// tree's levels in one pass, and the same values as walking them one
/// at a time. Run with the level's element count in the uniform until
/// one element is left.
pub fn emit_reduce(module: &Module) -> String {
    debug_assert_eq!(module.manifest.mode, Mode::Reduce);
    let Some(merge) = &module.merge else {
        return String::new();
    };
    let element = merge.args[0];
    let stride = module.manifest.outputs[0].ty.schema.value_words();
    let (count_binding, in_binding, out_binding) = reduce_bindings(module);
    let mut out = String::new();
    // The same context, params and inputs as the map shader: `merge` is a
    // plain function and may read any of them, or call one that does.
    emit_preamble(&mut out, module, false);
    let _ = writeln!(
        out,
        "@group(0) @binding({count_binding}) var<uniform> level: vec4<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({in_binding}) var<storage, read> fold_in: array<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({out_binding}) var<storage, read_write> fold_out: array<u32>;"
    );
    let (expr, _) = load_expr(module, element, "fold_in", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_level(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
        wgsl_type(element)
    );
    let mut stores = Vec::new();
    store_lines(module, element, "v", 0, &mut stores);
    let body = stores.join("\n").replace("out0[", "fold_out[");
    let _ = writeln!(
        out,
        "fn store_level(index: u32, v: {}) {{
    let b = index * {stride}u;
{body}
}}",
        wgsl_type(element)
    );
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(&mut out, module, function, &format!("f{index}"), false);
        out.push('\n');
    }
    emit_function(&mut out, module, merge, "merge", false);
    let fold = orla_rules::reduce::FOLD;
    let element = wgsl_type(element);
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({REDUCE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = level.x;
    let start = gid.x * {fold}u;
    if (start >= n) {{
        return;
    }}
    // The run this element folds, then that run folded by the tree: a
    // level of pairs, an odd last one carried up, until one is left.
    var held: array<{element}, {fold}>;
    var run = min({fold}u, n - start);
    var at = 0u;
    loop {{
        if (at >= run) {{ break; }}
        held[at] = load_level(start + at);
        at = at + 1u;
    }}
    loop {{
        if (run <= 1u) {{ break; }}
        let pairs = run / 2u;
        var k = 0u;
        loop {{
            if (k >= pairs) {{ break; }}
            held[k] = merge(held[2u * k], held[2u * k + 1u]);
            k = k + 1u;
        }}
        if ((run & 1u) == 1u) {{ held[pairs] = held[run - 1u]; }}
        run = (run + 1u) / 2u;
    }}
    store_level(gid.x, held[0]);
}}"
    );
    out
}

/// The fold pass's own bindings, after the inputs': the level count, the
/// level read and the level written.
pub fn reduce_bindings(module: &Module) -> (u32, u32, u32) {
    let base = INPUT_BINDING + module.manifest.inputs.len() as u32;
    (base, base + 1, base + 2)
}

/// Whether a type is floats all the way down, so it can be interpolated.
fn all_floats(module: &Module, ty: Type) -> bool {
    match ty {
        Type::Float
        | Type::Vec2
        | Type::Vec3
        | Type::Vec4
        | Type::Mat2
        | Type::Mat3
        | Type::Mat4 => true,
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .all(|field| all_floats(module, field.ty)),
        _ => false,
    }
}

/// `mix` over a type: WGSL's own for scalars and vectors, a generated
/// fieldwise one for structs of floats.
fn mix_call(module: &Module, ty: Type, a: &str, b: &str, t: &str) -> String {
    match ty {
        Type::Struct(index) => format!("mix_S{index}({a}, {b}, {t})"),
        _ if ty.order().is_some() => per_column(ty, |c| format!("mix({a}[{c}], {b}[{c}], {t})")),
        _ => {
            let _ = module;
            format!("mix({a}, {b}, {t})")
        }
    }
}

/// Emits `mix_S{k}` for every struct of floats, in table order so nested
/// structs come first.
fn emit_mixers(out: &mut String, module: &Module) {
    for (index, def) in module.structs.iter().enumerate() {
        if !all_floats(module, Type::Struct(index as u32)) {
            continue;
        }
        let fields = def
            .fields
            .iter()
            .enumerate()
            .map(|(k, field)| {
                mix_call(
                    module,
                    field.ty,
                    &format!("a{}", field_path(module, index as u32, k)),
                    &format!("b{}", field_path(module, index as u32, k)),
                    "t",
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "fn mix_S{index}(a: S{index}, b: S{index}, t: f32) -> S{index} {{ return S{index}({fields}); }}"
        );
    }
}

/// Emits `store_in{i}(index, v)` and `inside_in{i}(c)` for a memory: the
/// element store, and whether an address lands inside the memory (floored,
/// NaN never does), since a `put` outside is dropped.
fn emit_memory_writer(out: &mut String, module: &Module, index: usize, port: &orla_ir::Port) {
    let element = port.element();
    let slot = CONTEXT_HEAD + index;
    let encoding = input_encoding(module, index);
    let body = if encoding == Encoding::Word {
        let stride = port.ty.schema.value_words();
        let mut stores = Vec::new();
        store_lines(module, element, "v", 0, &mut stores);
        format!(
            "    let b = index * {stride}u;\n{}",
            stores.join("\n").replace("out0[", &format!("in{index}["))
        )
    } else {
        // One invocation walks a scan, so a narrow scalar is written by
        // reading its word, replacing its bytes and writing the word.
        let schema = &port.ty.schema;
        let mask = if encoding.bytes() == 1 {
            "0xFFu"
        } else {
            "0xFFFFu"
        };
        let mut scalars = Vec::new();
        scalar_paths(module, element, "v", &mut scalars);
        let mut lines = vec![format!("    let b = index * {}u;", schema.element_bytes())];
        for (s, (path, kind)) in scalars.iter().enumerate() {
            let offset = schema.offsets()[s];
            let raw = scalar_encode(encoding, *kind, path);
            lines.push(format!(
                "    {{ let at = b + {offset}u; let shift = (at & 3u) * 8u; in{index}[at >> 2u] = (in{index}[at >> 2u] & ~({mask} << shift)) | ({raw} << shift); }}"
            ));
        }
        lines.join("\n")
    };
    let _ = writeln!(
        out,
        "fn store_in{index}(index: u32, v: {}) {{\n{body}\n}}",
        wgsl_type(element)
    );
    let _ = match port.shape() {
        Shape::Rows => writeln!(
            out,
            "fn inside_in{index}(i: i32) -> bool {{ return i >= 0i && i < i32(ctx[{slot}].x); }}"
        ),
        Shape::Grid => writeln!(
            out,
            "fn inside_in{index}(c: vec2<f32>) -> bool {{
    let x = floor(c.x);
    let y = floor(c.y);
    return x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y;
}}"
        ),
        Shape::Volume => writeln!(
            out,
            "fn inside_in{index}(c: vec3<f32>) -> bool {{
    let x = floor(c.x);
    let y = floor(c.y);
    let z = floor(c.z);
    return x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y && z >= 0.0 && z < ctx[{slot}].z;
}}"
        ),
    };
}

/// Emits `linear_in{i}(c)`: the input interpolated at a fractional
/// position, element centres at half-integers, clamped at the edges.
fn emit_linear(out: &mut String, module: &Module, index: usize, port: &orla_ir::Port) {
    let element = port.element();
    if !all_floats(module, element) {
        return;
    }
    let slot = CONTEXT_HEAD + index;
    let ty = wgsl_type(element);
    let load = |x: &str, y: &str, z: &str| {
        format!("load_in{index}((({z}) * u32(h) + ({y})) * u32(w) + ({x}))")
    };
    let mix = |a: String, b: String, t: &str| mix_call(module, element, &a, &b, t);
    match port.shape() {
        Shape::Rows => {
            let _ = writeln!(
                out,
                "fn linear_in{index}(c: f32) -> {ty} {{
    let w = ctx[{slot}].x;
    let h = 1.0;
    let t = select(c - 0.5, -1.0, c != c);
    let lower = clamp(floor(t), 0.0, w - 1.0);
    let upper = min(lower + 1.0, w - 1.0);
    let f = select(clamp(t - floor(t), 0.0, 1.0), 0.0, t < 0.0);
    return {};
}}",
                mix(
                    load("u32(lower)", "0u", "0u"),
                    load("u32(upper)", "0u", "0u"),
                    "f"
                )
            );
        }
        Shape::Grid => {
            let _ = writeln!(
                out,
                "fn linear_in{index}(c: vec2<f32>) -> {ty} {{
    let w = ctx[{slot}].x;
    let h = ctx[{slot}].y;
    let t = select(c - vec2<f32>(0.5, 0.5), vec2<f32>(-1.0, -1.0), c != c);
    let lower = clamp(floor(t), vec2<f32>(0.0, 0.0), vec2<f32>(w - 1.0, h - 1.0));
    let upper = min(lower + vec2<f32>(1.0, 1.0), vec2<f32>(w - 1.0, h - 1.0));
    let f = select(clamp(t - floor(t), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)), vec2<f32>(0.0, 0.0), t < vec2<f32>(0.0, 0.0));
    let x0 = u32(lower.x); let x1 = u32(upper.x); let y0 = u32(lower.y); let y1 = u32(upper.y);
    let top = {};
    let bottom = {};
    return {};
}}",
                mix(load("x0", "y0", "0u"), load("x1", "y0", "0u"), "f.x"),
                mix(load("x0", "y1", "0u"), load("x1", "y1", "0u"), "f.x"),
                mix("top".into(), "bottom".into(), "f.y")
            );
        }
        Shape::Volume => {
            let _ = writeln!(
                out,
                "fn linear_in{index}(c: vec3<f32>) -> {ty} {{
    let w = ctx[{slot}].x;
    let h = ctx[{slot}].y;
    let d = ctx[{slot}].z;
    let t = select(c - vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(-1.0, -1.0, -1.0), c != c);
    let lower = clamp(floor(t), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(w - 1.0, h - 1.0, d - 1.0));
    let upper = min(lower + vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(w - 1.0, h - 1.0, d - 1.0));
    let f = select(clamp(t - floor(t), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)), vec3<f32>(0.0, 0.0, 0.0), t < vec3<f32>(0.0, 0.0, 0.0));
    let x0 = u32(lower.x); let x1 = u32(upper.x); let y0 = u32(lower.y); let y1 = u32(upper.y); let z0 = u32(lower.z); let z1 = u32(upper.z);
    let near_top = {};
    let near_bottom = {};
    let far_top = {};
    let far_bottom = {};
    let near = {};
    let far = {};
    return {};
}}",
                mix(load("x0", "y0", "z0"), load("x1", "y0", "z0"), "f.x"),
                mix(load("x0", "y1", "z0"), load("x1", "y1", "z0"), "f.x"),
                mix(load("x0", "y0", "z1"), load("x1", "y0", "z1"), "f.x"),
                mix(load("x0", "y1", "z1"), load("x1", "y1", "z1"), "f.x"),
                mix("near_top".into(), "near_bottom".into(), "f.y"),
                mix("far_top".into(), "far_bottom".into(), "f.y"),
                mix("near".into(), "far".into(), "f.z")
            );
        }
    }
}

/// How input `index` keeps its scalars: the one encoding of its port, or
/// whole words.
pub fn input_encoding(module: &Module, index: usize) -> Encoding {
    module.manifest.inputs[index]
        .ty
        .schema
        .uniform_encoding()
        .unwrap_or(Encoding::Word)
}

/// How output `k` keeps its scalars.
pub fn output_encoding(module: &Module, k: usize) -> Encoding {
    module.manifest.outputs[k]
        .ty
        .schema
        .uniform_encoding()
        .unwrap_or(Encoding::Word)
}

/// Whether any port keeps its scalars narrow, so the shader needs the
/// widening and narrowing helpers.
fn any_narrow(module: &Module) -> bool {
    let manifest = &module.manifest;
    manifest
        .inputs
        .iter()
        .chain(&manifest.outputs)
        .any(|port| !port.ty.schema.is_wide())
}

/// Whether a narrow output shares words between elements, so its
/// elements are stored by atomic or into a buffer cleared first; one
/// whose elements fill whole words is stored a word at a time.
pub fn output_shares_words(module: &Module, k: usize) -> bool {
    let schema = &module.manifest.outputs[k].ty.schema;
    !schema.is_wide() && !schema.element_bytes().is_multiple_of(4)
}

/// A `f32` constant by its bits, so the shader multiplies by exactly the
/// Host's own value.
fn f32_bits(value: f32) -> String {
    format!("bitcast<f32>({:#010x}u)", value.to_bits())
}

/// The widening and narrowing rules of `orla_abi::encoding`, as integer
/// arithmetic the shader compiler has no license to rewrite: a half
/// float both ways, and a normalized float narrowed.
fn emit_encodings(out: &mut String) {
    let _ = writeln!(
        out,
        "fn half_to_f32(h: u32) -> u32 {{
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1Fu;
    let m = h & 0x3FFu;
    if (e == 0u) {{
        if (m == 0u) {{ return sign; }}
        let shift = 10u - firstLeadingBit(m);
        return sign | ((113u - shift) << 23u) | (((m << shift) & 0x3FFu) << 13u);
    }}
    if (e == 0x1Fu) {{ return sign | 0x7F800000u | (m << 13u); }}
    return sign | ((e + 112u) << 23u) | (m << 13u);
}}
fn f32_to_half(x: u32) -> u32 {{
    let sign = (x >> 16u) & 0x8000u;
    let exponent = i32((x >> 23u) & 0xFFu);
    let mantissa = x & 0x7FFFFFu;
    if (exponent == 255i) {{
        return sign | 0x7C00u | select(0u, 0x200u | (mantissa >> 13u), mantissa != 0u);
    }}
    let e = exponent - 112i;
    if (e >= 31i) {{ return sign | 0x7C00u; }}
    var value = 0u;
    var dropped = 0u;
    var shift = 13u;
    if (e <= 0i) {{
        if (e < -10i) {{ return sign; }}
        let full = mantissa | 0x800000u;
        shift = u32(14i - e);
        value = full >> shift;
        dropped = full & ((1u << shift) - 1u);
    }} else {{
        value = (u32(e) << 10u) | (mantissa >> 13u);
        dropped = mantissa & 0x1FFFu;
    }}
    let half = 1u << (shift - 1u);
    let up = dropped > half || (dropped == half && (value & 1u) == 1u);
    return sign | (value + select(0u, 1u, up));
}}
fn encode_unit(x: f32, low: f32, scale: f32) -> u32 {{
    let nan = (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u;
    let c = select(clamp(x, low, 1.0), 0.0, nan);
    return bitcast<u32>(i32(round(c * scale)));
}}"
    );
}

/// An expression for scalar `k` (of kind `kind`) of a read at `base`:
/// a whole word at word `base + k`, or, for a narrow encoding, the
/// scalar's bytes at byte `base + k * size` widened by the Host's rule.
fn scalar_load(encoding: Encoding, kind: Scalar, buffer: &str, base: &str, k: usize) -> String {
    if encoding == Encoding::Word {
        let word = format!("{buffer}[{base} + {k}u]");
        return match kind {
            Scalar::Float => format!("bitcast<f32>({word})"),
            Scalar::Int => format!("bitcast<i32>({word})"),
            Scalar::Bool => format!("({word} != 0u)"),
            Scalar::Long => format!("long_of({word}, {buffer}[{base} + {}u])", k + 1),
            other => unreachable!("the GPU refuses a module holding a {other:?}"),
        };
    }
    let size = encoding.bytes();
    let at = format!("({base} + {}u)", k * size);
    let mask = if size == 1 { "0xFFu" } else { "0xFFFFu" };
    let raw = format!("(({buffer}[{at} >> 2u] >> (({at} & 3u) * 8u)) & {mask})");
    if kind == Scalar::Bool {
        return format!("({raw} != 0u)");
    }
    match encoding {
        Encoding::U8 | Encoding::U16 => format!("bitcast<i32>({raw})"),
        Encoding::I8 => format!("(bitcast<i32>({raw} << 24u) >> 24u)"),
        Encoding::I16 => format!("(bitcast<i32>({raw} << 16u) >> 16u)"),
        Encoding::Unorm8 => format!("(f32({raw}) * {})", f32_bits(encoding::UNORM8)),
        Encoding::Unorm16 => format!("(f32({raw}) * {})", f32_bits(encoding::UNORM16)),
        Encoding::Snorm8 => format!(
            "max(f32(bitcast<i32>({raw} << 24u) >> 24u) * {}, -1.0)",
            f32_bits(encoding::SNORM8)
        ),
        Encoding::F16 => format!("bitcast<f32>(half_to_f32({raw}))"),
        Encoding::Word => unreachable!("answered above"),
    }
}

/// An expression for the bits `value` (a scalar of kind `kind`) is kept
/// as in a narrow encoding, in the low bytes.
fn scalar_encode(encoding: Encoding, kind: Scalar, value: &str) -> String {
    if kind == Scalar::Bool {
        return format!("select(0u, 1u, {value})");
    }
    match encoding {
        Encoding::U8 | Encoding::I8 => format!("(bitcast<u32>({value}) & 0xFFu)"),
        Encoding::U16 | Encoding::I16 => format!("(bitcast<u32>({value}) & 0xFFFFu)"),
        Encoding::Unorm8 => format!("encode_unit({value}, 0.0, 255.0)"),
        Encoding::Unorm16 => format!("encode_unit({value}, 0.0, 65535.0)"),
        Encoding::Snorm8 => format!("(encode_unit({value}, -1.0, 127.0) & 0xFFu)"),
        Encoding::F16 => format!("f32_to_half(bitcast<u32>({value}))"),
        Encoding::Word => format!("bitcast<u32>({value})"),
    }
}

/// The scalars of a value of type `ty` at `path`, in layout order: an
/// expression for each and its kind.
fn scalar_paths(module: &Module, ty: Type, path: &str, out: &mut Vec<(String, Scalar)>) {
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            for (k, field) in def.fields.iter().enumerate() {
                scalar_paths(
                    module,
                    field.ty,
                    &format!("{path}{}", field_path(module, index, k)),
                    out,
                );
            }
        }
        other => out.extend(scalar_parts(other, path)),
    }
}

/// The scalars of a value that is not a struct, at `path`, in layout
/// order: itself, a vector's components, a matrix's entries column by
/// column; each an expression and its kind.
fn scalar_parts(ty: Type, path: &str) -> Vec<(String, Scalar)> {
    const LETTERS: [&str; 4] = ["x", "y", "z", "w"];
    match ty {
        Type::Scalar(kind) => vec![(path.to_owned(), kind)],
        Type::Vector(kind, n) => LETTERS
            .iter()
            .take(usize::from(n))
            .map(|letter| (format!("{path}.{letter}"), kind))
            .collect(),
        Type::Matrix(kind, n) => (0..n)
            .flat_map(|c| (0..n).map(move |r| (format!("{path}[{c}][{r}]"), kind)))
            .collect(),
        Type::Struct(_) => unreachable!("a struct's scalars are its fields'"),
    }
}

/// The words a scalar is stored as: a whole word, or two for a long.
fn stored_words(kind: Scalar, expr: &str) -> Vec<String> {
    match kind {
        Scalar::Bool => vec![format!("select(0u, 1u, {expr})")],
        Scalar::Long => vec![format!("long_low({expr})"), format!("long_high({expr})")],
        _ => vec![format!("bitcast<u32>({expr})")],
    }
}

/// The body of a store of `v` (a value of type `ty`) as element `index`
/// of output `k`, kept in a narrow encoding: whole words when elements
/// fill whole words, an atomic or into a cleared buffer when they share
/// them.
fn narrow_store_body(module: &Module, k: usize, ty: Type) -> String {
    let schema = &module.manifest.outputs[k].ty.schema;
    let encoding = output_encoding(module, k);
    let bytes = schema.element_bytes();
    let mut scalars = Vec::new();
    scalar_paths(module, ty, "v", &mut scalars);
    let bits: Vec<(usize, String)> = scalars
        .iter()
        .enumerate()
        .map(|(s, (path, kind))| (schema.offsets()[s], scalar_encode(encoding, *kind, path)))
        .collect();
    let mut lines = Vec::new();
    if bytes.is_multiple_of(4) {
        lines.push(format!("    let w = index * {}u;", bytes / 4));
        for word in 0..bytes / 4 {
            let parts: Vec<String> = bits
                .iter()
                .filter(|(offset, _)| offset / 4 == word)
                .map(|(offset, raw)| format!("({raw} << {}u)", (offset % 4) * 8))
                .collect();
            lines.push(format!("    out{k}[w + {word}u] = {};", parts.join(" | ")));
        }
    } else {
        lines.push(format!("    let b = index * {bytes}u;"));
        for (offset, raw) in &bits {
            lines.push(format!(
                "    atomicOr(&out{k}[(b + {offset}u) >> 2u], {raw} << (((b + {offset}u) & 3u) * 8u));"
            ));
        }
    }
    lines.join("\n")
}

/// An expression loading a value of type `ty` from `buffer` at `base`,
/// `offset` scalars in, and the scalars it consumes. `base` counts words
/// when the buffer keeps whole words and bytes when it keeps `encoding`.
fn load_expr(
    module: &Module,
    ty: Type,
    buffer: &str,
    base: &str,
    offset: usize,
    encoding: Encoding,
) -> (String, usize) {
    compose(module, ty, offset, &|kind, k| {
        scalar_load(encoding, kind, buffer, base, k)
    })
}

/// A value of type `ty` built from its scalars, `offset` in, each an
/// expression `scalar` gives for its kind and index; and the scalars it
/// takes.
fn compose(
    module: &Module,
    ty: Type,
    offset: usize,
    scalar: &dyn Fn(Scalar, usize) -> String,
) -> (String, usize) {
    // Each scalar from its words, `offset` in: one, or two for a long.
    let parts = |kind: Scalar, count: usize| -> (String, usize) {
        let width = kind.words();
        let parts = (0..count)
            .map(|k| scalar(kind, offset + k * width))
            .collect::<Vec<_>>()
            .join(", ");
        (parts, count * width)
    };
    match ty {
        Type::Scalar(kind) => (scalar(kind, offset), kind.words()),
        Type::Vector(kind, n) => {
            let (list, used) = parts(kind, usize::from(n));
            (format!("{}({list})", wgsl_type(ty)), used)
        }
        Type::Matrix(kind, n) => {
            let (list, used) = parts(kind, usize::from(n) * usize::from(n));
            (format!("{}({list})", wgsl_type(ty)), used)
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut parts = Vec::new();
            let mut used = 0;
            for field in &def.fields {
                let (expr, n) = compose(module, field.ty, offset + used, scalar);
                parts.push(expr);
                used += n;
            }
            (format!("S{index}({})", parts.join(", ")), used)
        }
    }
}

/// Where each parameter begins in the params uniform, in `vec4<u32>`
/// slots, and how many slots there are: a parameter is its words, from a
/// slot of its own, as many slots as its words fill.
pub fn param_slots(module: &Module) -> (Vec<usize>, usize) {
    let mut starts = Vec::with_capacity(module.manifest.params.len());
    let mut slots = 0;
    for param in &module.manifest.params {
        starts.push(slots);
        slots += param.default.words().len().div_ceil(4).max(1);
    }
    (starts, slots)
}

/// Statements storing `path` (a value of type `ty`) into `out0[b + ..]`.
fn store_lines(
    module: &Module,
    ty: Type,
    path: &str,
    offset: usize,
    out: &mut Vec<String>,
) -> usize {
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut used = 0;
            for (k, field) in def.fields.iter().enumerate() {
                used += store_lines(
                    module,
                    field.ty,
                    &format!("{path}{}", field_path(module, index, k)),
                    offset + used,
                    out,
                );
            }
            used
        }
        other => {
            let mut used = 0;
            for (expr, kind) in scalar_parts(other, path) {
                for word in stored_words(kind, &expr) {
                    out.push(format!("    out0[b + {}u] = {word};", offset + used));
                    used += 1;
                }
            }
            used
        }
    }
}

fn emit_function(out: &mut String, module: &Module, function: &Function, name: &str, entry: bool) {
    let args = if entry && module.manifest.mode == Mode::Scan {
        format!(
            "coord: vec3<f32>, element: u32, carried: {}",
            wgsl_type(module.manifest.outputs[0].element())
        )
    } else if entry {
        "coord: vec3<f32>, element: u32".to_owned()
    } else {
        function
            .args
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("a{index}: {}", wgsl_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    // A map's entry with several outputs stores each where it goes: WGSL
    // returns one value, and there is nothing to gain by packing them.
    let stores = entry && written_outputs(module) > 1;
    if stores {
        let _ = writeln!(out, "fn {name}({args}) {{");
    } else {
        let ret = result_type(function);
        let _ = writeln!(out, "fn {name}({args}) -> {} {{", wgsl_type(ret));
    }
    let mut emitter = Emitter {
        out,
        module,
        function,
        loops: Vec::new(),
    };
    emitter.instructions(&function.body, 1);
    if stores {
        for (k, result) in function.results.iter().enumerate() {
            let _ = writeln!(out, "    store_out{k}(element, v{});", result.0);
        }
    } else {
        let _ = writeln!(out, "    return v{};", function.results[0].0);
    }
    let _ = writeln!(out, "}}");
}

pub(crate) fn result_type(function: &Function) -> Type {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .find_map(|(_, block)| find(&block.body, id))
                    .or_else(|| find(&otherwise.body, id)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, function.results[0]).unwrap_or(Type::Float)
}

/// Whether a block is short plain arithmetic, which costs less computed
/// than branched around: no loop, branch, call or write.
fn speculable(block: &Block) -> bool {
    block.body.len() <= 16
        && block.body.iter().all(|instruction| {
            !matches!(
                instruction.op,
                Op::Loop { .. } | Op::If { .. } | Op::Match { .. } | Op::Put(..) | Op::Call(..)
            )
        })
}

/// Whether WGSL's `select` chooses between two values of a type: a
/// scalar or a vector, and never a 64-bit one.
fn selectable(ty: Type) -> bool {
    matches!(ty, Type::Scalar(_) | Type::Vector(..)) && !ty.is_64bit()
}

struct Emitter<'a> {
    out: &'a mut String,
    module: &'a Module,
    function: &'a Function,
    /// Ids of enclosing loop instructions, innermost last.
    loops: Vec<ValueId>,
}

impl Emitter<'_> {
    fn line(&mut self, depth: usize, text: &str) {
        let _ = writeln!(self.out, "{}{text}", "    ".repeat(depth));
    }

    fn instructions(&mut self, instructions: &[Instruction], depth: usize) {
        for instruction in instructions {
            self.instruction(instruction, depth);
        }
    }

    fn block(&mut self, block: &Block, depth: usize) -> String {
        self.instructions(&block.body, depth);
        format!("v{}", block.result.0)
    }

    fn instruction(&mut self, Instruction { id, ty, op }: &Instruction, depth: usize) {
        let v = |id: &ValueId| format!("v{}", id.0);
        let ty_name = wgsl_type(*ty);
        let expression = match op {
            Op::Const(value) => literal(value),
            Op::Param(index) => {
                // The parameter's words from its slots, as the type builds
                // them: the uniform holds words, as the interpreter does.
                let (starts, _) = param_slots(self.module);
                let start = starts[*index as usize];
                let ty = self.module.manifest.params[*index as usize].ty;
                let word = |kind: Scalar, k: usize| {
                    let word = format!("params[{}u][{}u]", start + k / 4, k % 4);
                    match kind {
                        Scalar::Float => format!("bitcast<f32>({word})"),
                        Scalar::Int => format!("bitcast<i32>({word})"),
                        Scalar::Bool => format!("({word} != 0u)"),
                        Scalar::Long => {
                            let high = k + 1;
                            let high = format!("params[{}u][{}u]", start + high / 4, high % 4);
                            format!("long_of({word}, {high})")
                        }
                        other => unreachable!("the GPU refuses a module holding a {other:?}"),
                    }
                };
                compose(self.module, ty, 0, &word).0
            }
            Op::Arg(index) => format!("a{index}"),
            // naga has no negation of a matrix; a scale by -1 flips the
            // same bits.
            Op::Unary(UnaryOp::Neg, a) if self.type_of(*a).order().is_some() => {
                format!("({} * -1.0)", v(a))
            }
            Op::Unary(UnaryOp::Neg, a) => format!("(-{})", v(a)),
            Op::Unary(UnaryOp::Not, a) => format!("(!{})", v(a)),
            Op::Binary(op, a, b) => {
                let symbol = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Rem => "%",
                    BinaryOp::Less => "<",
                    BinaryOp::LessEqual => "<=",
                    BinaryOp::Greater => ">",
                    BinaryOp::GreaterEqual => ">=",
                    BinaryOp::Equal => "==",
                    BinaryOp::NotEqual => "!=",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    BinaryOp::BitXor => "^",
                    BinaryOp::Shl => "<<",
                    BinaryOp::Shr => ">>",
                    BinaryOp::Ushr => ">>",
                };
                // A whole-vector comparison folds the componentwise one; a
                // shift takes its count unsigned, masked to the width.
                let vector = self.type_of(*a).lanes().is_some();
                let matrix = self.type_of(*a).order().map(usize::from);
                // An integer vector beside an Int: WGSL's bit operators
                // and shifts take two operands of one type, so the Int
                // widens; the shift count is unsigned, masked to the width.
                let int_vector = ty.int_components().is_some();
                let widen = |id: &ValueId| {
                    if self.type_of(*id) == *ty {
                        v(id)
                    } else {
                        format!("{}({})", wgsl_type(*ty), v(id))
                    }
                };
                let count = |id: &ValueId| {
                    let n = ty.int_components().unwrap_or(1);
                    // A whole vector converts; a scalar converts first,
                    // then splats, as WGSL does not do both in one.
                    let unsigned = if self.type_of(*id) == Type::Int {
                        format!("u32({})", v(id))
                    } else {
                        v(id)
                    };
                    format!("(vec{n}<u32>({unsigned}) & vec{n}<u32>(31u))")
                };
                // A long: the bit operators with a scalar widened beside a
                // vector, a shift's count modulo 64, and the division and
                // remainder the preamble writes.
                let long = self.type_of(*a).scalar() == Some(Type::Long);
                let lanes = ty.lanes().map(|(n, _)| n);
                let count64 = |id: &ValueId| match lanes {
                    // A scalar count converts first, then splats.
                    Some(n) if self.type_of(*id) == Type::Long => {
                        format!("(vec{n}<u32>(u32({})) & vec{n}<u32>(63u))", v(id))
                    }
                    Some(n) => format!("(vec{n}<u32>({}) & vec{n}<u32>(63u))", v(id)),
                    None => format!("(u32({}) & 63u)", v(id)),
                };
                let suffix = match lanes {
                    Some(n) => format!("x{n}"),
                    None => String::new(),
                };
                let unsigned = match lanes {
                    Some(n) => format!("vec{n}<u64>"),
                    None => "u64".to_owned(),
                };
                match op {
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor if long => {
                        format!("({} {symbol} {})", widen(a), widen(b))
                    }
                    BinaryOp::Shl | BinaryOp::Shr if long => {
                        format!("({} {symbol} {})", widen(a), count64(b))
                    }
                    BinaryOp::Ushr if long => format!(
                        "bitcast<{ty_name}>(bitcast<{unsigned}>({}) >> {})",
                        widen(a),
                        count64(b)
                    ),
                    BinaryOp::Div if long => {
                        format!("div_i64{suffix}({}, {})", widen(a), widen(b))
                    }
                    BinaryOp::Rem if long => {
                        format!("rem_i64{suffix}({}, {})", widen(a), widen(b))
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor if int_vector => {
                        format!("({} {symbol} {})", widen(a), widen(b))
                    }
                    BinaryOp::Shl | BinaryOp::Shr if int_vector => {
                        format!("({} {symbol} {})", widen(a), count(b))
                    }
                    BinaryOp::Ushr if int_vector => {
                        let n = ty.int_components().unwrap_or(1);
                        format!("vec{n}<i32>(vec{n}<u32>({}) >> {})", widen(a), count(b))
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual if matrix.is_some() => {
                        let n = matrix.unwrap_or(0);
                        let same = (0..n)
                            .map(|c| format!("all({}[{c}] == {}[{c}])", v(a), v(b)))
                            .collect::<Vec<_>>()
                            .join(" && ");
                        if *op == BinaryOp::Equal {
                            format!("({same})")
                        } else {
                            format!("(!({same}))")
                        }
                    }
                    BinaryOp::Rem if *ty != Type::Int && !int_vector => {
                        // Both operands at the result's width: a Float
                        // beside a vector broadcasts.
                        let widen = |id: &ValueId| {
                            if self.type_of(*id) == *ty {
                                v(id)
                            } else {
                                format!("{}({})", wgsl_type(*ty), v(id))
                            }
                        };
                        format!(
                            "fmod_{}({}, {})",
                            ty.components().unwrap_or(1),
                            widen(a),
                            widen(b)
                        )
                    }
                    // `isnan(x)` is `x != x`, which a compiler told
                    // there are no NaNs folds to false — and this one is
                    // told that. On the words it is the ones exponent
                    // with something in the mantissa, which nothing can
                    // fold away.
                    BinaryOp::NotEqual if a == b && self.type_of(*a) == Type::Float => {
                        format!("is_nan({})", v(a))
                    }
                    BinaryOp::Equal if a == b && self.type_of(*a) == Type::Float => {
                        format!("!is_nan({})", v(a))
                    }
                    BinaryOp::Equal if vector => format!("all({} == {})", v(a), v(b)),
                    BinaryOp::NotEqual if vector => format!("any({} != {})", v(a), v(b)),
                    BinaryOp::Shl | BinaryOp::Shr => {
                        format!("({} {symbol} (u32({}) & 31u))", v(a), v(b))
                    }
                    BinaryOp::Ushr => {
                        format!("i32(u32({}) >> (u32({}) & 31u))", v(a), v(b))
                    }
                    // The dividend through `whole_*`: a value the compiler
                    // narrowed (the top half of a word) divided wrongly.
                    BinaryOp::Div | BinaryOp::Rem if *ty == Type::Int || int_vector => {
                        let whole = match ty.int_components() {
                            Some(2) => "whole_v2",
                            Some(_) => "whole_v3",
                            None => "whole_i32",
                        };
                        format!("({whole}({}) {symbol} {})", widen(a), v(b))
                    }
                    _ => format!("({} {symbol} {})", v(a), v(b)),
                }
            }
            Op::Construct(parts) => format!(
                "{ty_name}({})",
                parts.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Extract(source, component) => {
                if let Type::Struct(index) = self.type_of(*source) {
                    format!(
                        "{}{}",
                        v(source),
                        field_path(self.module, index, *component as usize)
                    )
                } else if self.type_of(*source).order().is_some() {
                    format!("{}[{component}]", v(source))
                } else {
                    format!(
                        "{}.{}",
                        v(source),
                        ["x", "y", "z", "w"][*component as usize]
                    )
                }
            }
            Op::Subscript(array, index) => {
                let count = match self.type_of(*array) {
                    Type::Struct(struct_index) => {
                        self.module.structs[struct_index as usize].fields.len()
                    }
                    other => other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from),
                };
                format!("{}[clamp({}, 0, {})]", v(array), v(index), count - 1)
            }
            Op::Call(index, args) => format!(
                "f{index}({})",
                args.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args, *ty),
            Op::Put(input, index, value) => {
                // The store, when the address is inside; the value is the
                // instruction's own.
                self.line(
                    depth,
                    &format!(
                        "if (inside_in{input}({})) {{ store_in{input}(index_in{input}({}), {}); }}",
                        v(index),
                        v(index),
                        v(value)
                    ),
                );
                self.line(depth, &format!("let v{}: {ty_name} = {};", id.0, v(value)));
                return;
            }
            // Two short arms of plain arithmetic are both computed and
            // one is chosen: a branch costs a device more than the few
            // operations it would skip, and nothing a device computes
            // fails. A long is never chosen by `select`, which Metal
            // cannot pick for one.
            Op::If {
                condition,
                then,
                otherwise,
            } if speculable(then) && speculable(otherwise) && selectable(*ty) => {
                let chosen = self.block(then, depth);
                let other = self.block(otherwise, depth);
                format!("select({other}, {chosen}, {})", v(condition))
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                self.line(depth, &format!("var v{}: {ty_name};", id.0));
                self.line(depth, &format!("if ({}) {{", v(condition)));
                let result = self.block(then, depth + 1);
                self.line(depth + 1, &format!("v{} = {result};", id.0));
                self.line(depth, "} else {");
                let result = self.block(otherwise, depth + 1);
                self.line(depth + 1, &format!("v{} = {result};", id.0));
                self.line(depth, "}");
                return;
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                // WGSL's switch: every selector a literal, no fallthrough.
                self.line(depth, &format!("var v{}: {ty_name};", id.0));
                self.line(depth, &format!("switch ({}) {{", v(scrutinee)));
                for (values, block) in arms {
                    let selectors = values
                        .iter()
                        .map(|value| format!("{value}i"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.line(depth + 1, &format!("case {selectors}: {{"));
                    let result = self.block(block, depth + 2);
                    self.line(depth + 2, &format!("v{} = {result};", id.0));
                    self.line(depth + 1, "}");
                }
                self.line(depth + 1, "default: {");
                let result = self.block(otherwise, depth + 2);
                self.line(depth + 2, &format!("v{} = {result};", id.0));
                self.line(depth + 1, "}");
                self.line(depth, "}");
                return;
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                self.line(depth, &format!("var v{}: {ty_name} = {};", id.0, v(init)));
                let limit = match bound {
                    Some(bound) => format!("clamp({}, 0i, {count}i)", v(bound)),
                    None => format!("{count}i"),
                };
                // `it` and the value's number: a plain `i` would spell
                // `i32` at value 32, which WGSL reads as the type.
                self.line(
                    depth,
                    &format!(
                        "for (var it{0}: i32 = 0; it{0} < {limit}; it{0} = it{0} + 1) {{",
                        id.0
                    ),
                );
                self.loops.push(*id);
                if let Some(until) = until {
                    let done = self.block(until, depth + 1);
                    self.line(depth + 1, &format!("if ({done}) {{ break; }}"));
                }
                let next = self.block(body, depth + 1);
                self.line(depth + 1, &format!("v{} = {next};", id.0));
                self.loops.pop();
                self.line(depth, "}");
                return;
            }
            Op::Carried => match self.loops.last() {
                Some(id) => format!("v{}", id.0),
                None => "carried".to_owned(),
            },
            Op::Index => match self.loops.last() {
                Some(id) => format!("it{}", id.0),
                None => "i32(element)".to_owned(),
            },
            Op::Coord => match self.output_shape() {
                Shape::Volume => "coord".to_owned(),
                _ => "coord.xy".to_owned(),
            },
            Op::Domain => match self.output_shape() {
                Shape::Rows => "i32(ctx[0].x)".to_owned(),
                Shape::Grid => "ctx[0].xy".to_owned(),
                Shape::Volume => "vec3<f32>(ctx[0].xy, ctx[1].z)".to_owned(),
            },
            Op::Time => "ctx[0].z".to_owned(),
            Op::Rate => "ctx[1].w".to_owned(),
            Op::Pointer => "ctx[1].xy".to_owned(),
            Op::PointerDown => "button_down(0)".to_owned(),
            Op::Button(code) => format!("button_down({})", v(code)),
            Op::Wheel => "ctx[4].xy".to_owned(),
            Op::Stick(which) => format!("stick_at({})", v(which)),
            Op::Window => "ctx[4].zw".to_owned(),
            Op::Scale => "ctx[5].x".to_owned(),
            Op::Key(code) => format!("key_down({})", v(code)),
            Op::Element => "i32(element)".to_owned(),
            Op::Count(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                format!("i32(ctx[{slot}].x * ctx[{slot}].y * ctx[{slot}].z)")
            }
            Op::Size(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                match self.input_shape(*input) {
                    Shape::Rows => format!("i32(ctx[{slot}].x)"),
                    Shape::Grid => format!("ctx[{slot}].xy"),
                    Shape::Volume => format!("ctx[{slot}].xyz"),
                }
            }
            Op::Sample(input) => {
                // Validation made the input's shape the output's, so this
                // element's own address is the right one.
                let own = match self.output_shape() {
                    Shape::Rows => "i32(element)",
                    Shape::Grid => "coord.xy",
                    Shape::Volume => "coord",
                };
                format!("load_in{input}(index_in{input}({own}))")
            }
            Op::At(input, index) => format!("load_in{input}(index_in{input}({}))", v(index)),
            Op::Part { input, at, path } => {
                // The words this part is made of and nothing else:
                // where the path leads, held inside the element.
                let manifest = &self.module.manifest;
                let stride = manifest.inputs[*input as usize].ty.schema.value_words();
                let element = manifest.inputs[*input as usize].element();
                let layout = orla_ir::part_layout(&self.module.structs, element, path)
                    .expect("a path validation walked");
                let mut offset = format!("{}i", layout.base);
                for (index, count, part) in layout.steps {
                    let _ = write!(
                        offset,
                        " + clamp({}, 0i, {}i) * {part}i",
                        v(&index),
                        count as i64 - 1
                    );
                }
                let last = stride.saturating_sub(type_words(self.module, *ty));
                // The element the address names, or this element's
                // own where there is none, as a sample reads it.
                let element = match at {
                    Some(at) => v(at),
                    None => match self.output_shape() {
                        Shape::Rows => "i32(element)".to_owned(),
                        Shape::Grid => "coord.xy".to_owned(),
                        Shape::Volume => "coord".to_owned(),
                    },
                };
                let base = format!("b{}", id.0);
                // In words, or in bytes where the input keeps its scalars
                // narrow: the element's bytes, then the scalars'.
                let encoding = input_encoding(self.module, *input as usize);
                let (element_size, scalar_size) = if encoding == Encoding::Word {
                    (stride, 1)
                } else {
                    (
                        manifest.inputs[*input as usize].ty.schema.element_bytes(),
                        encoding.bytes(),
                    )
                };
                self.line(
                    depth,
                    &format!(
                        "let {base} = index_in{input}({element}) * {element_size}u + u32(clamp({offset}, 0i, {last}i)) * {scalar_size}u;"
                    ),
                );
                let (expression, _) =
                    load_expr(self.module, *ty, &format!("in{input}"), &base, 0, encoding);
                expression
            }
            Op::AtLinear(input, index) => format!("linear_in{input}({})", v(index)),
        };
        let _ = self.function;
        self.line(depth, &format!("let v{}: {ty_name} = {expression};", id.0));
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], ty: Type) -> String {
        let v = |id: &ValueId| format!("v{}", id.0);
        let list = args.iter().map(v).collect::<Vec<_>>().join(", ");
        let splat = |id: &ValueId, target: Type| {
            if target.lanes().is_some()
                && matches!(self.type_of(*id), Type::Float | Type::Int | Type::Long)
            {
                format!("{}({})", wgsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        let from = self.type_of(args[0]).scalar();
        let lanes = ty.lanes().map(|(n, _)| n);
        let suffix = |base: &str| match lanes {
            Some(n) => format!("{base}{n}"),
            None => base.to_owned(),
        };
        match intrinsic {
            Intrinsic::ToFloat => format!("{}({list})", wgsl_type(ty)),
            // A long keeps its low 32 bits, as the reference's does.
            Intrinsic::ToInt if from == Some(Type::Long) => format!("{}({list})", wgsl_type(ty)),
            // Saturating from a float, as `to_int` does; from an integer
            // or a Bool, WGSL's own.
            Intrinsic::ToLong if from == Some(Type::Float) => {
                format!("{}({list})", suffix("to_long"))
            }
            Intrinsic::ToLong => format!("{}({list})", wgsl_type(ty)),
            Intrinsic::Sign if from == Some(Type::Long) => match lanes {
                Some(n) => format!("sign_i64x{n}({list})"),
                None => format!("sign_i64({list})"),
            },
            // Saturating, as the reference converts; see `to_int` in
            // the preamble. A Bool is a 0 or a 1 and needs none of it.
            Intrinsic::ToInt => match (self.type_of(args[0]), ty) {
                (Type::Bool, _) => format!("{}({list})", wgsl_type(ty)),
                (_, Type::IVec2) => format!("to_int2({list})"),
                (_, Type::IVec3) => format!("to_int3({list})"),
                _ => format!("to_int({list})"),
            },
            Intrinsic::Pow => format!("pow({}, {})", v(&args[0]), splat(&args[1], ty)),
            // The angle at the origin, which a device's own does not
            // give; see `atan2_ref` in the preamble.
            Intrinsic::Atan2 => format!("atan2_ref({}, {})", v(&args[0]), v(&args[1])),
            // WGSL wants every operand the result's type; a Float broadcasts.
            Intrinsic::Min | Intrinsic::Max | Intrinsic::Clamp | Intrinsic::Mix => format!(
                "{}({})",
                intrinsic.name(),
                args.iter()
                    .map(|arg| splat(arg, ty))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            // WGSL's integer arithmetic wraps.
            Intrinsic::AddWrap => format!("({} + {})", v(&args[0]), v(&args[1])),
            Intrinsic::SubWrap => format!("({} - {})", v(&args[0]), v(&args[1])),
            Intrinsic::MulWrap => format!("({} * {})", v(&args[0]), v(&args[1])),
            Intrinsic::Popcount => format!("countOneBits({list})"),
            Intrinsic::Clz => format!("countLeadingZeros({list})"),
            Intrinsic::Ctz => format!("countTrailingZeros({list})"),
            Intrinsic::Rotl => format!("rotl_i32({list})"),
            Intrinsic::Rotr => format!("rotr_i32({list})"),
            Intrinsic::Ult => format!("(u32({}) < u32({}))", v(&args[0]), v(&args[1])),
            Intrinsic::Udiv => format!("i32(udiv_u32(u32({}), u32({})))", v(&args[0]), v(&args[1])),
            Intrinsic::Urem => format!("i32(urem_u32(u32({}), u32({})))", v(&args[0]), v(&args[1])),
            Intrinsic::Mulhi => format!("mulhi_i32({list})"),
            Intrinsic::Umulhi => format!(
                "i32(umulhi_u32(u32({}), u32({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Step => format!("step({}, {})", splat(&args[0], ty), v(&args[1])),
            Intrinsic::Smoothstep => format!(
                "smoothstep({}, {}, {})",
                splat(&args[0], ty),
                splat(&args[1], ty),
                v(&args[2])
            ),
            other => format!("{}({list})", other.name()),
        }
    }

    /// The shape the entry maps over: the output's, or the first input's
    /// for a reduction.
    fn output_shape(&self) -> Shape {
        let manifest = &self.module.manifest;
        match manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } | Mode::Prefix { .. } => {
                manifest.inputs.first().map_or(Shape::Rows, |p| p.shape())
            }
            Mode::Map => manifest.outputs.first().map_or(Shape::Grid, |p| p.shape()),
        }
    }

    fn input_shape(&self, input: u32) -> Shape {
        self.module.manifest.inputs[input as usize].shape()
    }

    fn type_of(&self, id: ValueId) -> Type {
        fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
            for instruction in instructions {
                if instruction.id == id {
                    return Some(instruction.ty);
                }
                let nested = match &instruction.op {
                    Op::If {
                        then, otherwise, ..
                    } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                    Op::Match {
                        arms, otherwise, ..
                    } => arms
                        .iter()
                        .find_map(|(_, block)| find(&block.body, id))
                        .or_else(|| find(&otherwise.body, id)),
                    Op::Loop { until, body, .. } => until
                        .as_ref()
                        .and_then(|block| find(&block.body, id))
                        .or_else(|| find(&body.body, id)),
                    _ => None,
                };
                if nested.is_some() {
                    return nested;
                }
            }
            None
        }
        find(&self.function.body, id).unwrap_or(Type::Float)
    }
}
