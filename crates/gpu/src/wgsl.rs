//! WGSL emission from validated Orla IR.
//!
//! Every value slot becomes a WGSL `let` (straight-line values) or `var`
//! (results of `if` and `loop`). Blocks map to WGSL blocks, so IR scoping
//! and WGSL scoping coincide. Resources are storage buffers of `u32` words,
//! one per scalar, plus two small uniform arrays for context and parameters.

use std::fmt::Write;

use orla_ir::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Mode, Module, Op, Shape, Type, UnaryOp,
    Value, ValueId,
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
/// pointer_down]`, `[pointer.x, pointer.y, depth, rate]`, and two slots of
/// key bits. Each input then has `[width, height, depth, 0]`.
pub const CONTEXT_HEAD: usize = 4;

fn wgsl_type(ty: Type) -> String {
    match ty {
        Type::Float => "f32".into(),
        Type::Int => "i32".into(),
        Type::Bool => "bool".into(),
        Type::Vec2 => "vec2<f32>".into(),
        Type::Vec3 => "vec3<f32>".into(),
        Type::Vec4 => "vec4<f32>".into(),
        Type::Struct(index) => format!("S{index}"),
    }
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
        Value::Int(v) => format!("{v}i"),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2<f32>({}, {})", f(*x), f(*y)),
        Value::Vec3([x, y, z]) => format!("vec3<f32>({}, {}, {})", f(*x), f(*y), f(*z)),
        Value::Vec4([x, y, z, w]) => {
            format!("vec4<f32>({}, {}, {}, {})", f(*x), f(*y), f(*z), f(*w))
        }
    }
}

/// Emits a complete compute shader for the module.
pub fn emit(module: &Module) -> String {
    let manifest = &module.manifest;
    let mut out = String::new();
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
    let _ = writeln!(
        out,
        "@group(0) @binding({OUTPUT_BINDING}) var<storage, read_write> out0: array<u32>;"
    );
    for index in 0..manifest.inputs.len() {
        let _ = writeln!(
            out,
            "@group(0) @binding({}) var<storage, read> in{index}: array<u32>;",
            INPUT_BINDING + index as u32,
        );
    }
    out.push('\n');
    for (index, def) in module.structs.iter().enumerate() {
        let fields = def
            .fields
            .iter()
            .enumerate()
            .map(|(k, field)| format!("f{k}: {}", wgsl_type(field.ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "struct S{index} {{ {fields} }}");
    }
    let _ = writeln!(
        out,
        "fn key_down(code: i32) -> bool {{
    if (code < 0 || code > 255) {{ return false; }}
    let c = u32(code);
    let word = bitcast<u32>(ctx[2u + (c >> 7u)][(c >> 5u) & 3u]);
    return ((word >> (c & 31u)) & 1u) == 1u;
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
    emit_mixers(&mut out, module);
    for (index, port) in manifest.inputs.iter().enumerate() {
        emit_linear(&mut out, module, index, port);
    }
    out.push('\n');
    if let Some(port) = manifest.outputs.first() {
        let element = port.element();
        let stride = port.ty.schema.words();
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
    if (gid.x >= w || gid.y >= h || gid.z >= d) {{
        return;
    }}
    let element = (gid.z * h + gid.y) * w + gid.x;
    let coord = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z));
    store_out0(element, entry(coord, element));
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

/// Binding index of the reduce step's count uniform.
pub const REDUCE_COUNT_BINDING: u32 = 0;
/// Binding index of the reduce step's input level.
pub const REDUCE_IN_BINDING: u32 = 1;
/// Binding index of the reduce step's output level.
pub const REDUCE_OUT_BINDING: u32 = 2;
/// Workgroup length of the reduce step.
pub const REDUCE_WORKGROUP: u32 = 64;

/// Emits the reduce step of a reduce module: element `i` of the output
/// level is `merge(in[2i], in[2i + 1])`, or `in[2i]` alone at an odd tail.
/// Run with the level's element count in the uniform until one element
/// is left. Both backends fold in this same tree.
pub fn emit_reduce(module: &Module) -> String {
    debug_assert_eq!(module.manifest.mode, Mode::Reduce);
    let Some(merge) = &module.merge else {
        return String::new();
    };
    let element = merge.args[0];
    let stride = module.manifest.outputs[0].ty.schema.words();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "@group(0) @binding({REDUCE_COUNT_BINDING}) var<uniform> level: vec4<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({REDUCE_IN_BINDING}) var<storage, read> in0: array<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({REDUCE_OUT_BINDING}) var<storage, read_write> out0: array<u32>;"
    );
    // The map shader's context and params are absent here: `merge` is a
    // plain function, but it may call helpers and read params, so params
    // are bound too.
    let _ = writeln!(
        out,
        "@group(0) @binding(3) var<uniform> params: array<vec4<f32>, {}>;",
        module.manifest.params.len().max(1)
    );
    for (index, def) in module.structs.iter().enumerate() {
        let fields = def
            .fields
            .iter()
            .enumerate()
            .map(|(k, field)| format!("f{k}: {}", wgsl_type(field.ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "struct S{index} {{ {fields} }}");
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
    let (expr, _) = load_expr(module, element, "in0", "b", 0);
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
    let _ = writeln!(
        out,
        "fn store_out0(index: u32, v: {}) {{
    let b = index * {stride}u;
{}
}}",
        wgsl_type(element),
        stores.join("\n")
    );
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(&mut out, module, function, &format!("f{index}"), false);
        out.push('\n');
    }
    emit_function(&mut out, module, merge, "merge", false);
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({REDUCE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = level.x;
    let i = gid.x;
    if (i >= (n + 1u) / 2u) {{
        return;
    }}
    if (2u * i + 1u < n) {{
        store_out0(i, merge(load_level(2u * i), load_level(2u * i + 1u)));
    }} else {{
        store_out0(i, load_level(2u * i));
    }}
}}"
    );
    out
}

/// Whether a type is floats all the way down, so it can be interpolated.
fn all_floats(module: &Module, ty: Type) -> bool {
    match ty {
        Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4 => true,
        Type::Int | Type::Bool => false,
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
                    &format!("a.f{k}"),
                    &format!("b.f{k}"),
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
        Type::Vec2 | Type::Vec3 | Type::Vec4 => {
            let n = ty.components().unwrap_or(4) as usize;
            for (k, component) in ["x", "y", "z", "w"].iter().take(n).enumerate() {
                out.push(format!(
                    "    out0[b + {}u] = bitcast<u32>({path}.{component});",
                    offset + k
                ));
            }
            n
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut used = 0;
            for (k, field) in def.fields.iter().enumerate() {
                used += store_lines(
                    module,
                    field.ty,
                    &format!("{path}.f{k}"),
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

fn result_type(function: &Function) -> Type {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
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
                    Type::Vec4 | Type::Struct(_) => slot,
                }
            }
            Op::Arg(index) => format!("a{index}"),
            Op::Unary(UnaryOp::Neg, a) => format!("(-{})", v(a)),
            Op::Unary(UnaryOp::Not, a) => format!("(!{})", v(a)),
            Op::Binary(op, a, b) => {
                let symbol = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Less => "<",
                    BinaryOp::LessEqual => "<=",
                    BinaryOp::Greater => ">",
                    BinaryOp::GreaterEqual => ">=",
                    BinaryOp::Equal => "==",
                    BinaryOp::NotEqual => "!=",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                };
                format!("({} {symbol} {})", v(a), v(b))
            }
            Op::Construct(parts) => format!(
                "{ty_name}({})",
                parts.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Extract(source, component) => {
                if let Type::Struct(_) = self.type_of(*source) {
                    format!("{}.f{component}", v(source))
                } else {
                    format!(
                        "{}.{}",
                        v(source),
                        ["x", "y", "z", "w"][*component as usize]
                    )
                }
            }
            Op::Call(index, args) => format!(
                "f{index}({})",
                args.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args, *ty),
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
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                self.line(depth, &format!("var v{}: {ty_name} = {};", id.0, v(init)));
                self.line(
                    depth,
                    &format!(
                        "for (var i{0}: i32 = 0; i{0} < {count}i; i{0} = i{0} + 1) {{",
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
                Some(id) => format!("i{}", id.0),
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
            Op::PointerDown => "(ctx[0].w != 0.0)".to_owned(),
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
            Op::AtLinear(input, index) => format!("linear_in{input}({})", v(index)),
        };
        let _ = self.function;
        self.line(depth, &format!("let v{}: {ty_name} = {expression};", id.0));
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], ty: Type) -> String {
        let v = |id: &ValueId| format!("v{}", id.0);
        let list = args.iter().map(v).collect::<Vec<_>>().join(", ");
        let splat = |id: &ValueId, target: Type| {
            if target.components().is_some() && self.type_of(*id) == Type::Float {
                format!("{}({})", wgsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        match intrinsic {
            Intrinsic::ToFloat => format!("f32({list})"),
            Intrinsic::ToInt => format!("i32({list})"),
            Intrinsic::Normalize => {
                format!("safe_normalize_{}({list})", ty.components().unwrap_or(0))
            }
            Intrinsic::Pow => format!("pow({}, {})", v(&args[0]), splat(&args[1], ty)),
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
            Mode::Reduce | Mode::Scan => manifest.inputs.first().map_or(Shape::Rows, |p| p.shape()),
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
