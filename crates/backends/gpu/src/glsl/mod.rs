//! GLSL ES 3.00 (`#version 300 es`) emission from validated Entelechy IR.
//!
//! Used by the page's WebGL2 driver for `Mode::Map` Kernels. Every Field's
//! storage words live in a 2D `RGBA32UI` texture (`ROW_TEXELS` texels wide,
//! four `u32` words per texel); context and parameters are `std140` `uvec4`
//! uniform blocks, keeping subnormal and NaN bit patterns intact on mobile
//! OpenGL ES 3.0 hardware.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use entelechy_abi::{Encoding, Scalar};
use entelechy_ir::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Module, Op, Shape, Type, UnaryOp, Value,
    ValueId,
};

mod emitter;
mod prelude;

use emitter::emit_function;
use prelude::emit_preamble;

use crate::wgsl::{
    CONTEXT_HEAD, FAILED_BUDGET, can_fail, input_encoding, output_encoding, param_slots,
    type_words, written_outputs,
};

/// Width in texels of a multi-row `RGBA32UI` word-stream texture (`2^11`,
/// guaranteed by OpenGL ES 3.0 / WebGL 2.0 `MAX_TEXTURE_SIZE >= 2048`).
pub const ROW_TEXELS: u32 = 2048;

/// Log2 of [`ROW_TEXELS`], for shift addressing (`texel >> ROW_SHIFT`).
pub const ROW_SHIFT: u32 = 11;

/// Bitmask for column addressing (`texel & ROW_MASK`).
pub const ROW_MASK: u32 = ROW_TEXELS - 1;

/// Runtime `1.0` in `u_ctx[5].z` that the shader compiler cannot fold.
pub(crate) const OPAQUE_ONE_EXPR: &str = "uintBitsToFloat(u_ctx[5].z)";

/// Fullscreen clip-space triangle vertex shader (`gl.drawArrays(TRIANGLES, 0, 3)`).
pub const VERTEX_SHADER: &str = "#version 300 es
precision highp float;
precision highp int;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
";

/// Fragment shader reducing a 2D `RGBA32UI` failure/work attachment by up to
/// `16x16` texels per pass: `.x` keeps the maximum failure word (first element
/// in row-major order), `.yz` sums 64-bit loop iterations (`low`, `high`).
pub const FAIL_REDUCE_SHADER: &str = "#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D u_src;
uniform ivec2 u_src_size;
layout(location = 0) out uvec4 out_fail;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * 16;
    uint word = 0u;
    uint work_lo = 0u;
    uint work_hi = 0u;
    for (int dy = 0; dy < 16; dy = dy + 1) {
        int y = base.y + dy;
        if (y >= u_src_size.y) { break; }
        for (int dx = 0; dx < 16; dx = dx + 1) {
            int x = base.x + dx;
            if (x >= u_src_size.x) { break; }
            uvec4 v = texelFetch(u_src, ivec2(x, y), 0);
            if (v.x > word) { word = v.x; }
            uint next_lo = work_lo + v.y;
            work_hi = work_hi + v.z + ((next_lo < work_lo) ? 1u : 0u);
            work_lo = next_lo;
        }
    }
    out_fail = uvec4(word, work_lo, work_hi, 0u);
}
";

/// Fragment shader comparing the first `u_words` 32-bit words of two resident
/// `RGBA32UI` textures, writing `1u` to `.x` where any word differs.
pub const SAME_SHADER: &str = "#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D u_a;
uniform usampler2D u_b;
uniform uint u_words;
layout(location = 0) out uvec4 out_diff;
void main() {
    uvec2 frag = uvec2(gl_FragCoord.xy);
    uint t = (frag.y << 11u) | frag.x;
    uint base = t << 2u;
    if (base >= u_words) {
        out_diff = uvec4(0u);
        return;
    }
    uvec4 a = texelFetch(u_a, ivec2(frag), 0);
    uvec4 b = texelFetch(u_b, ivec2(frag), 0);
    uint rem = u_words - base;
    bool diff = (a.x != b.x)
        || (rem > 1u && a.y != b.y)
        || (rem > 2u && a.z != b.z)
        || (rem > 3u && a.w != b.w);
    out_diff = uvec4(diff ? 1u : 0u, 0u, 0u, 0u);
}
";

/// Fragment shader presenting a resident `Image` (`RGBA32UI` `Vec4` words) and
/// optional `0x00RRGGBB` overlay onto the canvas's default framebuffer.
pub const BLIT_SHADER: &str = "#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D u_image;
uniform usampler2D u_overlay;
uniform vec4 u_frame[3];
layout(location = 0) out vec4 out_color;
void main() {
    vec2 pos = vec2(gl_FragCoord.x, u_frame[0].y - gl_FragCoord.y);
    if (pos.x < u_frame[2].y && pos.y < u_frame[2].z) {
        uint idx = uint(pos.y) * uint(u_frame[2].y) + uint(pos.x);
        uint t = idx >> 2u;
        uvec4 ov = texelFetch(u_overlay, ivec2(int(t & 2047u), int(t >> 11u)), 0);
        uint lane = idx & 3u;
        uint o = (lane == 0u) ? ov.x : ((lane == 1u) ? ov.y : ((lane == 2u) ? ov.z : ov.w));
        vec3 c = vec3(float((o >> 16u) & 255u), float((o >> 8u) & 255u), float(o & 255u)) / 255.0;
        out_color = vec4(c, 1.0);
        return;
    }
    vec2 p = pos - u_frame[1].xy;
    if (p.x < 0.0 || p.y < 0.0 || p.x >= u_frame[1].z || p.y >= u_frame[1].w) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    uint w = uint(u_frame[0].z);
    uint h = uint(u_frame[0].w);
    uint x = min(uint(p.x * u_frame[0].z / u_frame[1].z), w - 1u);
    uint y = min(uint(p.y * u_frame[0].w / u_frame[1].w), h - 1u);
    uint pixel = y * w + x;
    uvec4 raw_bits = texelFetch(u_image, ivec2(int(pixel & 2047u), int(pixel >> 11u)), 0);
    uvec3 rgb_bits = raw_bits.xyz;
    vec3 raw = uintBitsToFloat(rgb_bits);
    bvec3 nan = greaterThan(rgb_bits & uvec3(0x7FFFFFFFu), uvec3(0x7F800000u));
    vec3 clean = mix(raw, vec3(0.0), nan);
    out_color = vec4(clamp(clean, vec3(0.0), vec3(1.0)), 1.0);
}
";

/// Dimensions `(width, height)` in texels of the `RGBA32UI` texture holding
/// `words` 32-bit storage words.
pub fn texel_shape(words: usize) -> (u32, u32) {
    let texels = words.max(1).div_ceil(4);
    if texels <= ROW_TEXELS as usize {
        (texels as u32, 1)
    } else {
        let rows = texels.div_ceil(ROW_TEXELS as usize);
        (ROW_TEXELS, u32::try_from(rows).unwrap_or(u32::MAX))
    }
}

/// One fragment draw pass emitted for a `Mode::Map` Kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlslPass {
    /// Pruned `#version 300 es` fragment shader source.
    pub source: String,
    /// Manifest output indices bound to `COLOR_ATTACHMENT0..`.
    pub outputs: Vec<usize>,
    /// Whether `out_fail` is attached at the slot right after `outputs`.
    pub fails: bool,
}

/// Emits the primary `#version 300 es` fragment shader for `module`.
pub fn emit(module: &Module) -> String {
    emit_passes(module, &[], 4)
        .into_iter()
        .next()
        .map_or_else(String::new, |pass| pass.source)
}

/// Emits the fragment draw passes for `module` (`Mode::Map`), reading `packed`
/// inputs from one packed texture and attaching at most `max_draw_buffers`
/// color targets per pass.
pub fn emit_passes(module: &Module, packed: &[usize], max_draw_buffers: usize) -> Vec<GlslPass> {
    let written = written_outputs(module);
    let fails = can_fail(module);
    let mrt_fits = written > 1
        && written + usize::from(fails) <= max_draw_buffers.max(1)
        && module
            .manifest
            .outputs
            .iter()
            .take(written)
            .all(|port| port.ty.schema.is_wide() && port.ty.schema.value_words() == 4);
    if written == 1 || mrt_fits {
        let outputs: Vec<usize> = (0..written).collect();
        let raw = emit_pass_source(module, packed, &outputs, fails);
        return vec![GlslPass {
            source: pruned(&raw),
            outputs,
            fails,
        }];
    }
    (0..written)
        .map(|k| {
            let pass_fails = k == 0 && fails;
            let outputs = vec![k];
            let raw = emit_pass_source(module, packed, &outputs, pass_fails);
            GlslPass {
                source: pruned(&raw),
                outputs,
                fails: pass_fails,
            }
        })
        .collect()
}

fn emit_pass_source(module: &Module, packed: &[usize], outputs: &[usize], fails: bool) -> String {
    let mut out = String::with_capacity(4096);
    emit_preamble(&mut out, module, packed, outputs, fails);
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(
            &mut out,
            module,
            function,
            &function_name(index),
            None,
            packed,
        );
        out.push('\n');
    }
    emit_function(
        &mut out,
        module,
        &module.entry,
        "entry",
        Some(outputs),
        packed,
    );
    out.push('\n');
    emit_main(&mut out, module, outputs, fails);
    out
}

fn emit_main(out: &mut String, module: &Module, outputs: &[usize], fails: bool) {
    let manifest = &module.manifest;
    let first_port = &manifest.outputs[outputs[0]];
    let wide = first_port.ty.schema.is_wide();
    let stride_words = first_port.ty.schema.value_words();
    // Fast path: every bound output is wide with 4 words per element (1 texel = 1 element).
    if wide && stride_words == 4 {
        let _ = writeln!(
            out,
            "void main() {{
    uvec2 frag = uvec2(gl_FragCoord.xy);
    uint element = (frag.y << {ROW_SHIFT}u) | frag.x;
    uint w = u_ctx[0].x;
    uint h = u_ctx[0].y;
    uint d = u_ctx[1].z;
    uint total = w * h * d;
    if (element >= total) {{"
        );
        for (slot, _) in outputs.iter().enumerate() {
            let _ = writeln!(out, "        out{slot} = uvec4(0u);");
        }
        if fails {
            let _ = writeln!(out, "        out_fail = uvec4(0u);");
        }
        let _ = writeln!(
            out,
            "        return;
    }}
    uint row = element / w;
    uint x = element - row * w;
    uint z = row / h;
    uint y = row - z * h;
    current = element;
    failed_here = false;
    failed_code = 0u;
    begin_work();
    vec3 coord = vec3(float(x), float(y), float(z));"
        );
        if outputs.len() == 1 {
            let elem_ty = first_port.element();
            let _ = writeln!(
                out,
                "    {} r0 = entry(coord, element);",
                glsl_type(elem_ty)
            );
            let pack = pack_four_words(module, elem_ty, "r0");
            let _ = writeln!(out, "    out0 = {pack};");
        } else {
            for (slot, &out_idx) in outputs.iter().enumerate() {
                let elem_ty = manifest.outputs[out_idx].element();
                let _ = writeln!(out, "    {} r{slot};", glsl_type(elem_ty));
            }
            let args = (0..outputs.len())
                .map(|slot| format!("r{slot}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "    entry(coord, element, {args});");
            for (slot, &out_idx) in outputs.iter().enumerate() {
                let elem_ty = manifest.outputs[out_idx].element();
                let pack = pack_four_words(module, elem_ty, &format!("r{slot}"));
                let _ = writeln!(out, "    out{slot} = {pack};");
            }
        }
        if fails {
            let _ = writeln!(
                out,
                "    uint fw = (failed_here && failed_code != 0u) ? (0xFFFFFFFFu - (element * 4u + failed_code)) : 0u;
    out_fail = uvec4(fw, work, 0u, 0u);"
            );
        }
        let _ = writeln!(out, "}}");
        return;
    }

    // General path (1 output attachment per pass): arbitrary wide word stride S != 4, or narrow encoding.
    let out_idx = outputs[0];
    let elem_ty = first_port.element();
    if wide {
        if stride_words > 16 {
            emit_word_extractor(out, module, elem_ty);
        }
        let s = stride_words;
        let _ = writeln!(
            out,
            "void main() {{
    uvec2 frag = uvec2(gl_FragCoord.xy);
    uint t = (frag.y << {ROW_SHIFT}u) | frag.x;
    uint base_w = t << 2u;
    uint w = u_ctx[0].x;
    uint h = u_ctx[0].y;
    uint d = u_ctx[1].z;
    uint total = w * h * d;
    uint total_w = total * {s}u;
    if (base_w >= total_w) {{
        out0 = uvec4(0u);{clear_fail}
        return;
    }}
    uvec4 res = uvec4(0u);
    uint best_fail = 0u;
    uint work_lo = 0u;
    uint work_hi = 0u;
    uint e_start = base_w / {s}u;
    uint e_end = min((base_w + 3u) / {s}u + 1u, total);
    for (uint e = e_start; e < e_end; e = e + 1u) {{
        uint row = e / w;
        uint x = e - row * w;
        uint z = row / h;
        uint y = row - z * h;
        current = e;
        failed_here = false;
        failed_code = 0u;
        begin_work();
        vec3 coord = vec3(float(x), float(y), float(z));
        {ty} val = entry(coord, e);",
            clear_fail = if fails {
                "\n        out_fail = uvec4(0u);"
            } else {
                ""
            },
            ty = glsl_type(elem_ty),
        );
        if s <= 16 {
            let mut words = Vec::new();
            value_words_exprs(module, elem_ty, "val", &mut words);
            for (j, expr) in words.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "        {{
            uint gw = e * {s}u + {j}u;
            if (gw >= base_w && gw < base_w + 4u) {{
                uint lane = gw - base_w;
                uint wv = {expr};
                if (lane == 0u) {{ res.x = wv; }}
                else if (lane == 1u) {{ res.y = wv; }}
                else if (lane == 2u) {{ res.z = wv; }}
                else {{ res.w = wv; }}
            }}
        }}"
                );
            }
        } else {
            let _ = writeln!(
                out,
                "        for (uint lane = 0u; lane < 4u; lane = lane + 1u) {{
            uint gw = base_w + lane;
            if (gw >= e * {s}u && gw < (e + 1u) * {s}u) {{
                uint wv = extract_word_out(val, gw - e * {s}u);
                if (lane == 0u) {{ res.x = wv; }}
                else if (lane == 1u) {{ res.y = wv; }}
                else if (lane == 2u) {{ res.z = wv; }}
                else {{ res.w = wv; }}
            }}
        }}"
            );
        }
        if fails {
            let _ = writeln!(
                out,
                "        if (failed_here && failed_code != 0u) {{
            uint fw = 0xFFFFFFFFu - (e * 4u + failed_code);
            if (fw > best_fail) {{ best_fail = fw; }}
        }}
        if (((e * {s}u) >> 2u) == t) {{
            uint next_lo = work_lo + work;
            if (next_lo < work_lo) {{ work_hi = work_hi + 1u; }}
            work_lo = next_lo;
        }}"
            );
        }
        let _ = writeln!(out, "    }}\n    out0 = res;");
        if fails {
            let _ = writeln!(
                out,
                "    out_fail = uvec4(best_fail, work_lo, work_hi, 0u);"
            );
        }
        let _ = writeln!(out, "}}");
    } else {
        // Narrow encoding: 16 bytes per texel.
        let schema = &manifest.outputs[out_idx].ty.schema;
        let encoding = output_encoding(module, out_idx);
        let bytes = schema.element_bytes();
        let mut scalars = Vec::new();
        scalar_paths_glsl(module, elem_ty, "val", &mut scalars);
        let _ = writeln!(
            out,
            "void main() {{
    uvec2 frag = uvec2(gl_FragCoord.xy);
    uint t = (frag.y << {ROW_SHIFT}u) | frag.x;
    uint base_b = t << 4u;
    uint w = u_ctx[0].x;
    uint h = u_ctx[0].y;
    uint d = u_ctx[1].z;
    uint total = w * h * d;
    uint total_b = total * {bytes}u;
    if (base_b >= total_b) {{
        out0 = uvec4(0u);{clear_fail}
        return;
    }}
    uvec4 res = uvec4(0u);
    uint best_fail = 0u;
    uint work_lo = 0u;
    uint work_hi = 0u;
    uint e_start = base_b / {bytes}u;
    uint e_end = min((base_b + 15u) / {bytes}u + 1u, total);
    for (uint e = e_start; e < e_end; e = e + 1u) {{
        uint row = e / w;
        uint x = e - row * w;
        uint z = row / h;
        uint y = row - z * h;
        current = e;
        failed_here = false;
        failed_code = 0u;
        begin_work();
        vec3 coord = vec3(float(x), float(y), float(z));
        {ty} val = entry(coord, e);",
            clear_fail = if fails {
                "\n        out_fail = uvec4(0u);"
            } else {
                ""
            },
            ty = glsl_type(elem_ty),
        );
        for (s, (path, kind)) in scalars.iter().enumerate() {
            let offset = schema.offsets()[s];
            let encoded = prelude::scalar_encode_glsl(encoding, *kind, path);
            let _ = writeln!(
                out,
                "        {{
            uint gb = e * {bytes}u + {offset}u;
            if (gb >= base_b && gb < base_b + 16u) {{
                uint local = gb - base_b;
                uint shift = (local & 3u) * 8u;
                uint raw = ({encoded}) << shift;
                uint lane = local >> 2u;
                if (lane == 0u) {{ res.x = res.x | raw; }}
                else if (lane == 1u) {{ res.y = res.y | raw; }}
                else if (lane == 2u) {{ res.z = res.z | raw; }}
                else {{ res.w = res.w | raw; }}
            }}
        }}"
            );
        }
        if fails {
            let _ = writeln!(
                out,
                "        if (failed_here && failed_code != 0u) {{
            uint fw = 0xFFFFFFFFu - (e * 4u + failed_code);
            if (fw > best_fail) {{ best_fail = fw; }}
        }}
        if (((e * {bytes}u) >> 4u) == t) {{
            uint next_lo = work_lo + work;
            if (next_lo < work_lo) {{ work_hi = work_hi + 1u; }}
            work_lo = next_lo;
        }}"
            );
        }
        let _ = writeln!(out, "    }}\n    out0 = res;");
        if fails {
            let _ = writeln!(
                out,
                "    out_fail = uvec4(best_fail, work_lo, work_hi, 0u);"
            );
        }
        let _ = writeln!(out, "}}");
    }
}

fn pack_four_words(module: &Module, ty: Type, path: &str) -> String {
    match ty {
        Type::Vec4 => format!("floatBitsToUint({path})"),
        Type::IVec4 => format!("uvec4({path})"),
        _ => {
            let mut words = Vec::new();
            value_words_exprs(module, ty, path, &mut words);
            debug_assert_eq!(words.len(), 4);
            format!("uvec4({})", words.join(", "))
        }
    }
}

fn emit_word_extractor(out: &mut String, module: &Module, ty: Type) {
    let body = word_extract_body(module, ty, "v", "idx");
    let _ = writeln!(
        out,
        "uint extract_word_out({} v, uint idx) {{\n{body}\n    return 0u;\n}}",
        glsl_type(ty)
    );
}

fn word_extract_body(module: &Module, ty: Type, path: &str, idx_var: &str) -> String {
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            if let Some(elem) = def.element {
                let elem_w = type_words(module, elem);
                let elem_path = format!("{path}.m[int({idx_var} / {elem_w}u)]");
                if elem_w == 1 {
                    let mut single = Vec::new();
                    value_words_exprs(module, elem, &elem_path, &mut single);
                    format!("    return {};", single[0])
                } else {
                    let sub_idx = format!("({idx_var} % {elem_w}u)");
                    word_extract_body(module, elem, &elem_path, &sub_idx)
                }
            } else {
                let mut lines = Vec::new();
                let mut offset = 0usize;
                for (k, field) in def.fields.iter().enumerate() {
                    let fw = type_words(module, field.ty);
                    let fpath = format!("{path}{}", field_path(module, index, k));
                    let sub_idx = format!("({idx_var} - {offset}u)");
                    let inner = word_extract_body(module, field.ty, &fpath, &sub_idx);
                    lines.push(format!(
                        "    if ({idx_var} < {}u) {{\n{inner}\n    }}",
                        offset + fw
                    ));
                    offset += fw;
                }
                lines.join("\n")
            }
        }
        other => {
            let mut words = Vec::new();
            value_words_exprs(module, other, path, &mut words);
            let mut lines = Vec::new();
            for (k, w) in words.iter().enumerate() {
                lines.push(format!("    if ({idx_var} == {k}u) {{ return {w}; }}"));
            }
            lines.join("\n")
        }
    }
}

pub(crate) fn value_words_exprs(module: &Module, ty: Type, path: &str, out: &mut Vec<String>) {
    let mut scalars = Vec::new();
    scalar_paths_glsl(module, ty, path, &mut scalars);
    for (expr, kind) in scalars {
        match kind {
            Scalar::Float => out.push(format!("floatBitsToUint({expr})")),
            Scalar::Int => out.push(format!("uint({expr})")),
            Scalar::Bool => out.push(format!("({expr} ? 1u : 0u)")),
            Scalar::Long => {
                out.push(format!("({expr}).x"));
                out.push(format!("({expr}).y"));
            }
            Scalar::Double => unreachable!("the GPU refuses a module holding a Double"),
        }
    }
}

pub(crate) fn scalar_paths_glsl(
    module: &Module,
    ty: Type,
    path: &str,
    out: &mut Vec<(String, Scalar)>,
) {
    const LETTERS: [&str; 4] = ["x", "y", "z", "w"];
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            for (k, field) in def.fields.iter().enumerate() {
                scalar_paths_glsl(
                    module,
                    field.ty,
                    &format!("{path}{}", field_path(module, index, k)),
                    out,
                );
            }
        }
        Type::Scalar(kind) => out.push((path.to_owned(), kind)),
        Type::Vector(kind, n) => {
            for letter in LETTERS.iter().take(usize::from(n)) {
                out.push((format!("{path}.{letter}"), kind));
            }
        }
        Type::Matrix(kind, n) => {
            for c in 0..n {
                for r in 0..n {
                    out.push((format!("{path}[{c}][{r}]"), kind));
                }
            }
        }
    }
}

pub(crate) fn function_name(index: usize) -> String {
    format!("fn_{index}")
}

pub(crate) fn member_name(k: usize) -> String {
    format!("m_{k}")
}

pub(crate) fn field_path(module: &Module, index: u32, k: usize) -> String {
    if module.structs[index as usize].is_array() {
        format!(".m[{k}]")
    } else {
        format!(".{}", member_name(k))
    }
}

pub(crate) fn struct_decl(module: &Module, index: usize) -> String {
    let def = &module.structs[index];
    match def.element {
        Some(element) => format!(
            "struct S{index} {{ {} m[{}]; }};",
            glsl_type(element),
            def.fields.len().max(1)
        ),
        None => {
            let fields = def
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| format!("{} {};", glsl_type(field.ty), member_name(k)))
                .collect::<Vec<_>>()
                .join(" ");
            format!("struct S{index} {{ {fields} }};")
        }
    }
}

pub(crate) fn glsl_type(ty: Type) -> String {
    match ty {
        Type::Float => "float".into(),
        Type::Int => "int".into(),
        Type::Bool => "bool".into(),
        Type::Long => "uvec2".into(),
        Type::Vec2 => "vec2".into(),
        Type::Vec3 => "vec3".into(),
        Type::Vec4 => "vec4".into(),
        Type::IVec2 => "ivec2".into(),
        Type::IVec3 => "ivec3".into(),
        Type::IVec4 => "ivec4".into(),
        Type::Mat2 => "mat2".into(),
        Type::Mat3 => "mat3".into(),
        Type::Mat4 => "mat4".into(),
        Type::Struct(index) => format!("S{index}"),
        other => unreachable!("the WebGL2 emitter refuses {other}"),
    }
}

pub(crate) fn construct_type(module: &Module, ty: Type, parts: &[String]) -> String {
    if let Type::Struct(index) = ty
        && let Some(elem) = module.structs[index as usize].element
    {
        return format!(
            "S{index}({}[{}]({}))",
            glsl_type(elem),
            parts.len().max(1),
            parts.join(", ")
        );
    }
    format!("{}({})", glsl_type(ty), parts.join(", "))
}

pub(crate) fn literal(value: &Value) -> String {
    let f = |v: f32| {
        if v == 0.0 && v.is_sign_negative() {
            "uintBitsToFloat(0x80000000u)".to_owned()
        } else if v.is_finite() && v.abs() < 1.0e38 {
            format!("{v:?}")
        } else {
            format!("uintBitsToFloat({:#010x}u)", v.to_bits())
        }
    };
    let i = |v: i32| {
        if v == i32::MIN {
            "(-2147483647 - 1)".to_owned()
        } else {
            format!("{v}")
        }
    };
    let ints = |v: &[i32]| v.iter().map(|&x| i(x)).collect::<Vec<_>>().join(", ");
    let floats = |v: &[f32]| v.iter().map(|&x| f(x)).collect::<Vec<_>>().join(", ");
    match value {
        Value::Float(v) => f(*v),
        Value::Int(v) => i(*v),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2({}, {})", f(*x), f(*y)),
        Value::Vec3([x, y, z]) => format!("vec3({}, {}, {})", f(*x), f(*y), f(*z)),
        Value::Vec4([x, y, z, w]) => format!("vec4({}, {}, {}, {})", f(*x), f(*y), f(*z), f(*w)),
        Value::IVec2(v) => format!("ivec2({})", ints(v)),
        Value::IVec3(v) => format!("ivec3({})", ints(v)),
        Value::IVec4(v) => format!("ivec4({})", ints(v)),
        Value::Mat2(v) => format!("mat2({})", floats(v)),
        Value::Mat3(v) => format!("mat3({})", floats(&v[..])),
        Value::Mat4(v) => format!("mat4({})", floats(&v[..])),
        Value::Long(l) => {
            let bits = *l as u64;
            format!("uvec2({}u, {}u)", bits as u32, (bits >> 32) as u32)
        }
        Value::Record(_) => unreachable!("a constant of the IR is never a record"),
        other => unreachable!("the WebGL2 emitter refuses a module holding {other}"),
    }
}

/// `source` with only the top-level functions reachable from `main`.
pub fn pruned(source: &str) -> String {
    struct Item<'a> {
        lines: Vec<&'a str>,
        function: Option<&'a str>,
        entry: bool,
    }
    fn code(line: &str) -> &str {
        line.find("//").map_or(line, |at| &line[..at])
    }
    let braces = |line: &str| -> i64 {
        code(line).chars().fold(0, |depth, c| match c {
            '{' => depth + 1,
            '}' => depth - 1,
            _ => depth,
        })
    };
    fn words(text: &str) -> impl Iterator<Item = &str> {
        text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| w.starts_with(|c: char| c.is_alphabetic() || c == '_'))
    }

    let mut items = Vec::new();
    let mut lead: Vec<&str> = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let at_top = trimmed.len() == line.len();
        if at_top && trimmed.starts_with("//") {
            lead.push(line);
            continue;
        }
        let non_fn_prefix = trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("precision ")
            || trimmed.starts_with("layout(")
            || trimmed.starts_with("uniform ")
            || trimmed.starts_with("struct ");
        let fn_name = if at_top
            && !non_fn_prefix
            && code(line).contains('(')
            && !code(line).trim_end().ends_with(';')
        {
            code(line)
                .split('(')
                .next()
                .and_then(|head| head.split_whitespace().last())
                .filter(|name| !name.is_empty())
        } else {
            None
        };
        if let Some(name) = fn_name {
            let entry = name == "main";
            let mut item_lines = std::mem::take(&mut lead);
            item_lines.push(line);
            let mut depth = braces(line);
            let mut opened = code(line).contains('{');
            while !(opened && depth <= 0) {
                let Some(next) = lines.next() else { break };
                item_lines.push(next);
                depth += braces(next);
                opened |= code(next).contains('{');
            }
            items.push(Item {
                lines: item_lines,
                function: Some(name),
                entry,
            });
            continue;
        }
        for kept in lead.drain(..).chain(std::iter::once(line)) {
            items.push(Item {
                lines: vec![kept],
                function: None,
                entry: false,
            });
        }
    }
    for kept in lead {
        items.push(Item {
            lines: vec![kept],
            function: None,
            entry: false,
        });
    }

    let defined: BTreeMap<&str, usize> = items
        .iter()
        .enumerate()
        .filter_map(|(at, item)| item.function.map(|name| (name, at)))
        .collect();
    let mut reached = BTreeSet::new();
    let mut waiting: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.function.is_none() || item.entry)
        .map(|(at, _)| at)
        .collect();
    while let Some(at) = waiting.pop() {
        if !reached.insert(at) {
            continue;
        }
        for line in &items[at].lines {
            for word in words(code(line)) {
                if let Some(&callee) = defined.get(word)
                    && !reached.contains(&callee)
                {
                    waiting.push(callee);
                }
            }
        }
    }
    let mut out = String::with_capacity(source.len());
    for (at, item) in items.iter().enumerate() {
        if reached.contains(&at) {
            for line in &item.lines {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelechy_compiler::compile;
    use entelechy_ir::Mode;

    #[test]
    fn texel_shape_wraps_at_row_texels() {
        assert_eq!(texel_shape(0), (1, 1));
        assert_eq!(texel_shape(4), (1, 1));
        assert_eq!(texel_shape(5), (2, 1));
        assert_eq!(texel_shape(2048 * 4), (2048, 1));
        assert_eq!(texel_shape(2048 * 4 + 1), (2048, 2));
    }

    #[test]
    fn pruned_keeps_version_and_reachable_helpers_only() {
        let module = compile(
            "kernel t.img@1 {
                requires { field.read; field.write; time.read; }
                input src: Image;
                output dst: Image;
                fn tint(c: Vec4) -> Vec4 { return c * 0.5; }
                fn dead(x: Float) -> Float { return pow(x, 2.0); }
                compute {
                    return tint(sample(src)) + vec4(time(), 0.0, 0.0, 1.0);
                }
            }",
        )
        .unwrap();
        let shader = emit(&module);
        assert!(shader.starts_with("#version 300 es\n"), "{shader}");
        assert!(shader.contains("vec4 load_in0("), "{shader}");
        assert!(shader.contains("uint index_in0("), "{shader}");
        assert!(!shader.contains("pow_ref"), "{shader}");
        assert!(!shader.contains("clz_u32"), "{shader}");

        let synthetic = "#version 300 es\nfloat kept_fn(float x) {\n    return x * 2.0;\n}\nfloat dead_fn(float x) {\n    return x + 1.0;\n}\nvoid main() {\n    float y = kept_fn(1.0);\n}\n";
        let cut = pruned(synthetic);
        assert!(cut.starts_with("#version 300 es\n"), "{cut}");
        assert!(cut.contains("float kept_fn("), "{cut}");
        assert!(!cut.contains("dead_fn"), "{cut}");
    }

    #[test]
    fn every_map_conformance_case_emits_clean_essl_300() {
        for case in entelechy_conformance::cases()
            .into_iter()
            .chain(entelechy_conformance::long_scalar_cases())
        {
            let module = compile(&case.source).unwrap();
            if module.manifest.mode != Mode::Map
                || module.holds(Type::Double)
                || (module.holds(Type::Long) && !crate::wgsl::pairs_take(&module))
            {
                continue;
            }
            let accounted = entelechy_ir::account(&module).unwrap();
            let passes = emit_passes(&accounted, &[], 4);
            assert!(!passes.is_empty(), "{}", case.name);
            for pass in passes {
                assert!(
                    pass.source.starts_with("#version 300 es\n"),
                    "{}",
                    case.name
                );
                assert!(pass.source.contains("void main()"), "{}", case.name);
                // Guard against WGSL tokens leaking into GLSL ES 3.00.
                for forbidden in [
                    "bitcast<",
                    "vec2<",
                    "vec3<",
                    "vec4<",
                    "select(",
                    "atan2(",
                    "countOneBits(",
                    "countLeadingZeros(",
                    "countTrailingZeros(",
                    "firstLeadingBit(",
                    "fma(",
                    "var ",
                    "let ",
                    "fn ",
                ] {
                    assert!(
                        !pass.source.contains(forbidden),
                        "{} leaked WGSL token `{forbidden}`:\n{}",
                        case.name,
                        pass.source
                    );
                }
            }
        }
    }
}
