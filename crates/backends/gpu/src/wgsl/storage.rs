//! Port storage in shaders: scalar encodings, and the loads and stores
//! that widen and narrow them.

use super::*;

/// Input `index`'s scalar encoding: its port's uniform one, else words.
pub fn input_encoding(module: &Module, index: usize) -> Encoding {
    module.manifest.inputs[index]
        .ty
        .schema
        .uniform_encoding()
        .unwrap_or(Encoding::Word)
}

/// Output `k`'s scalar encoding.
pub fn output_encoding(module: &Module, k: usize) -> Encoding {
    module.manifest.outputs[k]
        .ty
        .schema
        .uniform_encoding()
        .unwrap_or(Encoding::Word)
}

/// Whether any port is narrow, so the shader needs encoding helpers.
pub(crate) fn any_narrow(module: &Module) -> bool {
    let manifest = &module.manifest;
    manifest
        .inputs
        .iter()
        .chain(&manifest.outputs)
        .any(|port| !port.ty.schema.is_wide())
}

/// Whether a narrow output shares words between elements, so elements
/// are stored by atomic or into a cleared buffer, not a word at a time.
pub fn output_shares_words(module: &Module, k: usize) -> bool {
    let schema = &module.manifest.outputs[k].ty.schema;
    !schema.is_wide() && !schema.element_bytes().is_multiple_of(4)
}

/// Whether output `k` requires atomic-or word sharing when `shader_f16`
/// availability is taken into account: native `array<f16>` bindings write
/// 16-bit elements directly without buffer clears or word atomics.
pub fn output_shares_words_for(module: &Module, k: usize, shader_f16: bool) -> bool {
    if native_f16_output(module, k, shader_f16).is_some() {
        return false;
    }
    output_shares_words(module, k)
}

/// Native `SHADER_F16` storage array element type for a pure-`f16` port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeF16Binding {
    /// `array<f16>` (exact 2-byte scalar stride: `Float`, `Vec3`, matrices, structs, or `Op::Part`).
    Scalar,
    /// `array<vec2<f16>>` (4-byte vectorized load/store for whole `Vec2` elements).
    Vec2,
    /// `array<vec4<f16>>` (8-byte vectorized load/store for whole `Vec4` elements).
    Vec4,
}

impl NativeF16Binding {
    pub(crate) const fn wgsl_elem(self) -> &'static str {
        match self {
            Self::Scalar => "f16",
            Self::Vec2 => "vec2<f16>",
            Self::Vec4 => "vec4<f16>",
        }
    }
}

/// Indices of input ports read via `Op::Part` anywhere in `module`.
pub(crate) fn part_inputs(module: &Module) -> std::collections::BTreeSet<usize> {
    fn walk(body: &[Instruction], out: &mut std::collections::BTreeSet<usize>) {
        for instr in body {
            if let Op::Part { input, .. } = &instr.op {
                out.insert(*input as usize);
            }
            for block in instr.op.blocks() {
                walk(&block.body, out);
            }
        }
    }
    let mut out = std::collections::BTreeSet::new();
    for func in module
        .functions
        .iter()
        .chain([&module.entry])
        .chain(module.merge.iter())
        .chain(module.init.iter())
    {
        walk(&func.body, &mut out);
    }
    out
}

/// Native `f16` storage binding for input `index`, when `shader_f16` is
/// enabled and the port is an unpacked, untextured `Encoding::F16` buffer.
pub(crate) fn native_f16_input(
    module: &Module,
    index: usize,
    packs: &[Vec<usize>],
    textured: &[usize],
    shader_f16: bool,
    parts: &std::collections::BTreeSet<usize>,
) -> Option<NativeF16Binding> {
    if !shader_f16
        || input_encoding(module, index) != Encoding::F16
        || textured.contains(&index)
        || packs.iter().any(|group| group.contains(&index))
    {
        return None;
    }
    let element = module.manifest.inputs[index].element();
    if !parts.contains(&index) {
        if element == Type::Vec4 {
            return Some(NativeF16Binding::Vec4);
        }
        if element == Type::Vec2 {
            return Some(NativeF16Binding::Vec2);
        }
    }
    Some(NativeF16Binding::Scalar)
}

/// Native `f16` storage binding for written output `k`, when `shader_f16` is
/// enabled and the output is a non-scatter `Encoding::F16` buffer.
pub(crate) fn native_f16_output(
    module: &Module,
    k: usize,
    shader_f16: bool,
) -> Option<NativeF16Binding> {
    if !shader_f16
        || matches!(module.manifest.mode, Mode::Scatter { .. })
        || output_encoding(module, k) != Encoding::F16
    {
        return None;
    }
    match module.manifest.outputs[k].element() {
        Type::Vec4 => Some(NativeF16Binding::Vec4),
        Type::Vec2 => Some(NativeF16Binding::Vec2),
        _ => Some(NativeF16Binding::Scalar),
    }
}

/// Whether the emitted WGSL shader references `f16` types and therefore
/// requires a leading `enable f16;` directive.
pub(crate) fn shader_uses_f16(
    module: &Module,
    with_output: bool,
    packs: &[Vec<usize>],
    textured: &[usize],
    shader_f16: bool,
) -> bool {
    if !shader_f16 {
        return false;
    }
    let parts = part_inputs(module);
    let any_in = (0..module.manifest.inputs.len())
        .any(|i| native_f16_input(module, i, packs, textured, shader_f16, &parts).is_some());
    let any_out = with_output
        && (0..written_outputs(module)).any(|k| native_f16_output(module, k, shader_f16).is_some());
    any_in || any_out
}

/// Generates a native `array<f16>` / `array<vec2<f16>>` / `array<vec4<f16>>`
/// store body for `v: ty` into `buf` (`out{k}` or memory `in{index}`).
pub(crate) fn native_f16_store_body(
    module: &Module,
    buf: &str,
    ty: Type,
    binding: NativeF16Binding,
) -> String {
    match binding {
        NativeF16Binding::Vec4 => format!("    {buf}[index] = vec4<f16>(v);"),
        NativeF16Binding::Vec2 => format!("    {buf}[index] = vec2<f16>(v);"),
        NativeF16Binding::Scalar => {
            let mut scalars = Vec::new();
            scalar_paths(module, ty, "v", &mut scalars);
            let stride = scalars.len();
            let mut lines = Vec::with_capacity(stride + 1);
            lines.push(format!("    let b = index * {stride}u;"));
            for (s, (path, _)) in scalars.iter().enumerate() {
                lines.push(format!("    {buf}[b + {s}u] = f16({path});"));
            }
            lines.join("\n")
        }
    }
}

/// Generates a native `array<f16>` / `array<vec2<f16>>` / `array<vec4<f16>>`
/// load body for `load_in{index}(index: u32)`.
pub(crate) fn native_f16_load_body(
    module: &Module,
    buf: &str,
    ty: Type,
    binding: NativeF16Binding,
) -> String {
    match binding {
        NativeF16Binding::Vec4 => format!("    return vec4<f32>({buf}[index]);"),
        NativeF16Binding::Vec2 => format!("    return vec2<f32>({buf}[index]);"),
        NativeF16Binding::Scalar => {
            let stride = type_words(module, ty);
            let (expr, _) = compose(module, ty, 0, &|_, k| format!("f32({buf}[b + {k}u])"));
            format!("    let b = index * {stride}u;\n    return {expr};")
        }
    }
}

/// A `f32` constant written as its bits, exactly the Host's value.
pub(crate) fn f32_bits(value: f32) -> String {
    format!("bitcast<f32>({:#010x}u)", value.to_bits())
}

/// The `entelechy_abi::encoding` rules for half floats (both ways) and
/// normalized narrowing, in integer arithmetic the shader compiler keeps
/// exact.
pub(crate) fn emit_encodings(out: &mut String) {
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
fn load_f16_half(word: u32, hi: bool) -> f32 {{
    let u = unpack2x16float(word);
    return select(u.x, u.y, hi);
}}
fn encode_unit(x: f32, low: f32, scale: f32) -> u32 {{
    let nan = (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u;
    let c = select(clamp(x, low, 1.0), 0.0, nan);
    return bitcast<u32>(i32(round(c * scale)));
}}"
    );
}

/// Scalar `k` of a read at `base`: word `base + k`, or for a narrow
/// encoding the bytes at `base + k * size`, widened by the Host's rule.
pub(crate) fn scalar_load(
    encoding: Encoding,
    kind: Scalar,
    buffer: &str,
    base: &str,
    k: usize,
) -> String {
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
    if encoding == Encoding::F16 {
        return format!("load_f16_half({buffer}[{at} >> 2u], ({at} & 2u) != 0u)");
    }
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
        Encoding::F16 | Encoding::Word => unreachable!("answered above"),
    }
}

/// The narrow-encoded bits of scalar `value`, in the low bytes.
pub(crate) fn scalar_encode(encoding: Encoding, kind: Scalar, value: &str) -> String {
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

/// `(expression, kind)` of each scalar of a `ty` at `path`, in layout
/// order.
pub(crate) fn scalar_paths(module: &Module, ty: Type, path: &str, out: &mut Vec<(String, Scalar)>) {
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

/// `(expression, kind)` of each scalar of a non-struct at `path`, in
/// layout order; a matrix is column-major.
pub(crate) fn scalar_parts(ty: Type, path: &str) -> Vec<(String, Scalar)> {
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

/// The words a scalar is stored as: one, or two for a long.
pub(crate) fn stored_words(kind: Scalar, expr: &str) -> Vec<String> {
    match kind {
        Scalar::Bool => vec![format!("select(0u, 1u, {expr})")],
        Scalar::Long => vec![format!("long_low({expr})"), format!("long_high({expr})")],
        _ => vec![format!("bitcast<u32>({expr})")],
    }
}

/// The body storing `v: ty` as element `index` of narrow output `k`:
/// whole words when elements fill them, else an atomic or into a
/// cleared buffer.
pub(crate) fn narrow_store_body(module: &Module, k: usize, ty: Type) -> String {
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
        if encoding == Encoding::F16 {
            lines.push(format!("    let w = index * {}u;", bytes / 4));
            for word in 0..bytes / 4 {
                let lo = scalars
                    .iter()
                    .enumerate()
                    .find(|(s, _)| schema.offsets()[*s] == word * 4)
                    .map(|(_, (path, _))| path.as_str())
                    .unwrap_or("0.0");
                let hi = scalars
                    .iter()
                    .enumerate()
                    .find(|(s, _)| schema.offsets()[*s] == word * 4 + 2)
                    .map(|(_, (path, _))| path.as_str())
                    .unwrap_or("0.0");
                lines.push(format!(
                    "    out{k}[w + {word}u] = pack2x16float(vec2<f32>({lo}, {hi}));"
                ));
            }
        } else {
            lines.push(format!("    let w = index * {}u;", bytes / 4));
            for word in 0..bytes / 4 {
                let parts: Vec<String> = bits
                    .iter()
                    .filter(|(offset, _)| offset / 4 == word)
                    .map(|(offset, raw)| format!("({raw} << {}u)", (offset % 4) * 8))
                    .collect();
                lines.push(format!("    out{k}[w + {word}u] = {};", parts.join(" | ")));
            }
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

/// An expression loading a `ty` from `buffer` at `base`, `offset`
/// scalars in, and the scalars it consumes. `base` counts words, or
/// bytes for a narrow `encoding`.
pub(crate) fn load_expr(
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

/// A `ty` built from `scalar(kind, index)` expressions from `offset`,
/// and the words it takes.
pub(crate) fn compose(
    module: &Module,
    ty: Type,
    offset: usize,
    scalar: &dyn Fn(Scalar, usize) -> String,
) -> (String, usize) {
    // A long takes two words.
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

/// Each parameter's first `vec4<u32>` slot in the params uniform, and the
/// slot count. A parameter starts a slot and fills as many as its words
/// need.
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
pub(crate) fn store_lines(
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
