//! WGSL emission from validated Orla IR.
//!
//! Every value slot becomes a WGSL `let` (straight-line values) or `var`
//! (results of `if` and `loop`). Blocks map to WGSL blocks, so IR scoping
//! and WGSL scoping coincide. Resources are storage buffers of `u32` words,
//! one per scalar, plus two small uniform arrays for context and parameters.

use std::fmt::Write;

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
        Type::Float => "f32".into(),
        Type::Int => "i32".into(),
        Type::Bool => "bool".into(),
        Type::Vec2 => "vec2<f32>".into(),
        Type::Vec3 => "vec3<f32>".into(),
        Type::Vec4 => "vec4<f32>".into(),
        Type::IVec2 => "vec2<i32>".into(),
        Type::IVec3 => "vec3<i32>".into(),
        Type::Mat2 => "mat2x2<f32>".into(),
        Type::Mat3 => "mat3x3<f32>".into(),
        Type::Mat4 => "mat4x4<f32>".into(),
        Type::Struct(index) => format!("S{index}"),
    }
}

/// Whether any function of the module takes the inverse of a matrix of
/// the given type.
fn inverts(module: &Module, ty: Type) -> bool {
    fn walk(instructions: &[Instruction], ty: Type) -> bool {
        instructions
            .iter()
            .any(|instruction| match &instruction.op {
                Op::Intrinsic(Intrinsic::Inverse, _) => instruction.ty == ty,
                Op::If {
                    then, otherwise, ..
                } => walk(&then.body, ty) || walk(&otherwise.body, ty),
                Op::Match {
                    arms, otherwise, ..
                } => {
                    arms.iter().any(|(_, block)| walk(&block.body, ty)) || walk(&otherwise.body, ty)
                }
                Op::Loop { until, body, .. } => {
                    until.as_ref().is_some_and(|block| walk(&block.body, ty))
                        || walk(&body.body, ty)
                }
                _ => false,
            })
    }
    module
        .functions
        .iter()
        .chain([&module.entry])
        .chain(module.merge.iter())
        .chain(module.init.iter())
        .any(|function| walk(&function.body, ty))
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
    }
}

/// The preamble both shaders share: the context and params uniforms, the
/// output buffer when `with_output`, every input's buffer, the structs,
/// the key and button readers, every input's loader and address, the
/// normalizers, mixers, linear readers and memory writers. A `merge` may
/// read inputs, params and the context like any function, so the fold
/// shader carries the same.
fn emit_preamble(out: &mut String, module: &Module, with_output: bool) {
    let manifest = &module.manifest;
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    let params_len = manifest.params.len().max(1);
    let _ = writeln!(
        out,
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({PARAMS_BINDING}) var<uniform> params: array<vec4<f32>, {params_len}>;"
    );
    if with_output {
        let _ = writeln!(
            out,
            "@group(0) @binding({OUTPUT_BINDING}) var<storage, read_write> out0: array<u32>;"
        );
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
}}"
    );
    for (index, port) in manifest.inputs.iter().enumerate() {
        let element = port.element();
        let stride = port.ty.schema.words();
        let (expr, _) = load_expr(module, element, &format!("in{index}"), "b", 0);
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
    for ty in [Type::Vec2, Type::Vec3, Type::Vec4] {
        let name = wgsl_type(ty);
        let _ = writeln!(
            out,
            "fn safe_normalize_{}(v: {name}) -> {name} {{
    let l = length(v);
    if (l > 0.0) {{ return v / l; }}
    return {name}(0.0);
}}",
            ty.components().unwrap_or(0)
        );
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
    // The inverse of a matrix, which WGSL lacks: the adjugate over the
    // determinant, in the order the interpreter and the JIT compute it.
    // Only for the orders the Kernel inverts: the unrolled 4x4 costs the
    // shader compiler a quarter of a second.
    for ty in [Type::Mat2, Type::Mat3, Type::Mat4] {
        if !inverts(module, ty) {
            continue;
        }
        let name = wgsl_type(ty);
        let n = usize::from(ty.order().unwrap_or(0));
        let entries: Vec<String> = (0..n * n)
            .map(|k| format!("m[{}][{}]", k / n, k % n))
            .collect();
        let inverse = orla_rules::matrix::inverse(&mut orla_rules::matrix::Text, n, &entries);
        let _ = writeln!(
            out,
            "fn inverse_{n}(m: {name}) -> {name} {{
    return {name}({});
}}",
            inverse.join(", ")
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
    if let Some(port) = manifest.outputs.first() {
        // A scatter's map pass stores the entry's pairs, not elements.
        let element = if matches!(manifest.mode, Mode::Scatter { .. }) {
            result_type(&module.entry)
        } else {
            port.element()
        };
        let stride = type_words(module, element);
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
    store_out0(element, entry(coord, element));
}}"
    );
    out
}

/// Words a value of a type takes in a buffer.
pub fn type_words(module: &Module, ty: Type) -> usize {
    match ty {
        Type::Float | Type::Int | Type::Bool => 1,
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 => 4,
        Type::IVec2 => 2,
        Type::IVec3 => 3,
        Type::Mat2 => 4,
        Type::Mat3 => 9,
        Type::Mat4 => 16,
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .map(|field| type_words(module, field.ty))
            .sum(),
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
        Type::Float | Type::Int => out.push((format!("bitcast<u32>({path})"), ty)),
        Type::Bool => out.push((format!("select(0u, 1u, {path})"), Type::Bool)),
        Type::Vec2 | Type::Vec3 | Type::Vec4 | Type::IVec2 | Type::IVec3 => {
            let (n, lane) = ty.lanes().unwrap_or((4, Type::Float));
            for component in ["x", "y", "z", "w"].iter().take(n as usize) {
                out.push((format!("bitcast<u32>({path}.{component})"), lane));
            }
        }
        Type::Mat2 | Type::Mat3 | Type::Mat4 => {
            let n = usize::from(ty.order().unwrap_or(0));
            for c in 0..n {
                for r in 0..n {
                    out.push((format!("bitcast<u32>({path}[{c}][{r}])"), Type::Float));
                }
            }
        }
        Type::Struct(index) => {
            for (k, field) in module.structs[index as usize].fields.iter().enumerate() {
                let path = format!("{path}{}", field_path(module, index, k));
                value_words(module, field.ty, &path, out);
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
    let (expr, _) = load_expr(module, pair, "pairs", "b", 0);
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
    let stride = target.ty.schema.words();
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
    let stride = target.ty.schema.words();
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
    let stride = target.ty.schema.words();
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
    let stride = target.ty.schema.words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), &sort_bindings(true));
    let (expr, _) = load_expr(module, element, "out0", "b", 0);
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
    let stride = module.manifest.outputs[0].ty.schema.words();
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
    let (expr, _) = load_expr(module, element, "fold_in", "b", 0);
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
        Type::Int | Type::Bool | Type::IVec2 | Type::IVec3 => false,
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .all(|field| all_floats(module, field.ty)),
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
    let stride = port.ty.schema.words();
    let slot = CONTEXT_HEAD + index;
    let mut stores = Vec::new();
    store_lines(module, element, "v", 0, &mut stores);
    let body = stores.join("\n").replace("out0[", &format!("in{index}["));
    let _ = writeln!(
        out,
        "fn store_in{index}(index: u32, v: {}) {{
    let b = index * {stride}u;
{body}
}}",
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

/// An expression loading a value of type `ty` from `buffer[base + offset..]`,
/// and the words it consumes.
fn load_expr(
    module: &Module,
    ty: Type,
    buffer: &str,
    base: &str,
    offset: usize,
) -> (String, usize) {
    let word = |k: usize| format!("{buffer}[{base} + {k}u]");
    match ty {
        Type::Float => (format!("bitcast<f32>({})", word(offset)), 1),
        Type::Int => (format!("bitcast<i32>({})", word(offset)), 1),
        Type::Bool => (format!("({} != 0u)", word(offset)), 1),
        Type::Vec2 | Type::Vec3 | Type::Vec4 => {
            let n = ty.components().unwrap_or(4) as usize;
            let parts = (0..n)
                .map(|k| format!("bitcast<f32>({})", word(offset + k)))
                .collect::<Vec<_>>()
                .join(", ");
            (format!("{}({parts})", wgsl_type(ty)), n)
        }
        Type::IVec2 | Type::IVec3 => {
            let n = ty.int_components().unwrap_or(2) as usize;
            let parts = (0..n)
                .map(|k| format!("bitcast<i32>({})", word(offset + k)))
                .collect::<Vec<_>>()
                .join(", ");
            (format!("{}({parts})", wgsl_type(ty)), n)
        }
        Type::Mat2 | Type::Mat3 | Type::Mat4 => {
            let n = usize::from(ty.order().unwrap_or(0));
            let parts = (0..n * n)
                .map(|k| format!("bitcast<f32>({})", word(offset + k)))
                .collect::<Vec<_>>()
                .join(", ");
            (format!("{}({parts})", wgsl_type(ty)), n * n)
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut parts = Vec::new();
            let mut used = 0;
            for field in &def.fields {
                let (expr, n) = load_expr(module, field.ty, buffer, base, offset + used);
                parts.push(expr);
                used += n;
            }
            (format!("S{index}({})", parts.join(", ")), used)
        }
    }
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
        Type::Float => {
            out.push(format!("    out0[b + {offset}u] = bitcast<u32>({path});"));
            1
        }
        Type::Int => {
            out.push(format!("    out0[b + {offset}u] = bitcast<u32>({path});"));
            1
        }
        Type::Bool => {
            out.push(format!("    out0[b + {offset}u] = select(0u, 1u, {path});"));
            1
        }
        Type::Vec2 | Type::Vec3 | Type::Vec4 | Type::IVec2 | Type::IVec3 => {
            let n = ty.lanes().map_or(4, |(n, _)| n as usize);
            for (k, component) in ["x", "y", "z", "w"].iter().take(n).enumerate() {
                out.push(format!(
                    "    out0[b + {}u] = bitcast<u32>({path}.{component});",
                    offset + k
                ));
            }
            n
        }
        Type::Mat2 | Type::Mat3 | Type::Mat4 => {
            let n = usize::from(ty.order().unwrap_or(0));
            for c in 0..n {
                for r in 0..n {
                    out.push(format!(
                        "    out0[b + {}u] = bitcast<u32>({path}[{c}][{r}]);",
                        offset + c * n + r
                    ));
                }
            }
            n * n
        }
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
    let ret = result_type(function);
    let _ = writeln!(out, "fn {name}({args}) -> {} {{", wgsl_type(ret));
    let mut emitter = Emitter {
        out,
        module,
        function,
        loops: Vec::new(),
    };
    emitter.instructions(&function.body, 1);
    let _ = writeln!(out, "    return v{};", function.results[0].0);
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
                let slot = format!("params[{index}]");
                match self.module.manifest.params[*index as usize].default.ty() {
                    Type::Float => format!("{slot}.x"),
                    Type::Int => format!("bitcast<i32>({slot}.x)"),
                    Type::Bool => format!("({slot}.x != 0.0)"),
                    Type::Vec2 => format!("{slot}.xy"),
                    Type::Vec3 => format!("{slot}.xyz"),
                    Type::Vec4
                    | Type::IVec2
                    | Type::IVec3
                    | Type::Mat2
                    | Type::Mat3
                    | Type::Mat4
                    | Type::Struct(_) => slot,
                }
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
                match op {
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
                let stride = manifest.inputs[*input as usize].ty.schema.words();
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
                self.line(
                    depth,
                    &format!(
                        "let {base} = index_in{input}({element}) * {stride}u + u32(clamp({offset}, 0i, {last}i));"
                    ),
                );
                let (expression, _) = load_expr(self.module, *ty, &format!("in{input}"), &base, 0);
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
            if target.lanes().is_some() && matches!(self.type_of(*id), Type::Float | Type::Int) {
                format!("{}({})", wgsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        match intrinsic {
            Intrinsic::ToFloat | Intrinsic::ToInt => format!("{}({list})", wgsl_type(ty)),
            Intrinsic::Normalize => {
                format!("safe_normalize_{}({list})", ty.components().unwrap_or(0))
            }
            Intrinsic::Pow => format!("pow({}, {})", v(&args[0]), splat(&args[1], ty)),
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
            Intrinsic::Inverse => format!("inverse_{}({list})", ty.order().unwrap_or(0)),
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
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } => {
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
