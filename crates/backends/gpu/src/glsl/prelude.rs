//! GLSL ES 3.00 preamble: uniforms, textures, structs, IEEE/integer/64-bit
//! helpers, narrow codecs, and input loaders.

use super::*;
use entelechy_abi::encoding;

const BIT_HELPERS: &str = "
uint popcount_u32(uint x) {
    uint v = x - ((x >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}
uint clz_u32(uint x) {
    if (x == 0u) { return 32u; }
    uint n = 0u;
    uint v = x;
    if ((v & 0xFFFF0000u) == 0u) { n = n + 16u; v = v << 16u; }
    if ((v & 0xFF000000u) == 0u) { n = n + 8u; v = v << 8u; }
    if ((v & 0xF0000000u) == 0u) { n = n + 4u; v = v << 4u; }
    if ((v & 0xC0000000u) == 0u) { n = n + 2u; v = v << 2u; }
    if ((v & 0x80000000u) == 0u) { n = n + 1u; }
    return n;
}
uint ctz_u32(uint x) {
    if (x == 0u) { return 32u; }
    return 31u - clz_u32(x & (0u - x));
}
uint first_leading_bit_u32(uint x) {
    return 31u - clz_u32(x);
}
";

const CHECKED_HELPERS: &str = "
int add_i32(int a, int b) {
    int r = int(uint(a) + uint(b));
    if (((a ^ r) & (b ^ r)) < 0) { fail(1u); }
    return r;
}
int sub_i32(int a, int b) {
    int r = int(uint(a) - uint(b));
    if (((a ^ b) & (a ^ r)) < 0) { fail(1u); }
    return r;
}
int mul_i32(int a, int b) {
    int r = int(uint(a) * uint(b));
    if (mulhi_i32(a, b) != (r >> 31)) { fail(1u); }
    return r;
}
int div_i32(int a, int b) {
    if (b == 0) { fail(2u); return a; }
    if (a == (-2147483647 - 1) && b == -1) { fail(1u); return a; }
    return whole_i32(a) / b;
}
int rem_i32(int a, int b) {
    if (b == 0) { fail(2u); return 0; }
    if (a == (-2147483647 - 1) && b == -1) { fail(1u); return 0; }
    return whole_i32(a) % b;
}
int neg_i32(int a) {
    if (a == (-2147483647 - 1)) { fail(1u); }
    return int(0u - uint(a));
}
uint udiv_checked(uint a, uint b) {
    if (b == 0u) { fail(2u); return a; }
    return udiv_u32(a, b);
}
uint urem_checked(uint a, uint b) {
    if (b == 0u) { fail(2u); return 0u; }
    return urem_u32(a, b);
}
";

fn checked_vector_helpers(n: u8) -> String {
    let letters = ["x", "y", "z", "w"];
    let mut out = String::new();
    for op in ["add", "sub", "mul", "div", "rem"] {
        let parts = letters
            .iter()
            .take(usize::from(n))
            .map(|c| format!("{op}_i32(a.{c}, b.{c})"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "ivec{n} {op}_i32x{n}(ivec{n} a, ivec{n} b) {{ return ivec{n}({parts}); }}"
        );
    }
    let parts = letters
        .iter()
        .take(usize::from(n))
        .map(|c| format!("neg_i32(a.{c})"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        out,
        "ivec{n} neg_i32x{n}(ivec{n} a) {{ return ivec{n}({parts}); }}"
    );
    out
}

const LONG_PAIR_HELPERS: &str = "
uvec2 long_of(uint low, uint high) { return uvec2(low, high); }
bool long_negative(uvec2 a) { return (a.y >> 31u) != 0u; }
bool long_is_min(uvec2 a) { return a.x == 0u && a.y == 0x80000000u; }
uvec2 long_from_int(int i) { return uvec2(uint(i), (i < 0) ? 0xFFFFFFFFu : 0u); }
uvec2 long_from_bool(bool b) { return uvec2(b ? 1u : 0u, 0u); }
bool long_eq(uvec2 a, uvec2 b) { return a.x == b.x && a.y == b.y; }
bool long_lt(uvec2 a, uvec2 b) {
    if (a.y != b.y) { return int(a.y) < int(b.y); }
    return a.x < b.x;
}
bool long_le(uvec2 a, uvec2 b) { return !long_lt(b, a); }
bool long_gt(uvec2 a, uvec2 b) { return long_lt(b, a); }
bool long_ge(uvec2 a, uvec2 b) { return !long_lt(a, b); }
bool ulong_lt(uvec2 a, uvec2 b) {
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}
uvec2 add_wrap_i64(uvec2 a, uvec2 b) {
    uint low = a.x + b.x;
    return uvec2(low, a.y + b.y + ((low < a.x) ? 1u : 0u));
}
uvec2 sub_wrap_i64(uvec2 a, uvec2 b) {
    return uvec2(a.x - b.x, a.y - b.y - ((a.x < b.x) ? 1u : 0u));
}
uvec2 neg_wrap_i64(uvec2 a) { return sub_wrap_i64(uvec2(0u, 0u), a); }
uvec2 wide_mul_u32(uint a, uint b) {
    uint a0 = a & 0xFFFFu;
    uint a1 = a >> 16u;
    uint b0 = b & 0xFFFFu;
    uint b1 = b >> 16u;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint middle = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    uint low = (p00 & 0xFFFFu) | (middle << 16u);
    return uvec2(low, a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (middle >> 16u));
}
uvec2 mul_wrap_i64(uvec2 a, uvec2 b) {
    uvec2 p = wide_mul_u32(a.x, b.x);
    return uvec2(p.x, p.y + a.x * b.y + a.y * b.x);
}
uvec2 add_i64(uvec2 a, uvec2 b) {
    uvec2 r = add_wrap_i64(a, b);
    if ((((a.y ^ r.y) & (b.y ^ r.y)) >> 31u) != 0u) { fail(1u); }
    return r;
}
uvec2 sub_i64(uvec2 a, uvec2 b) {
    uvec2 r = sub_wrap_i64(a, b);
    if ((((a.y ^ b.y) & (a.y ^ r.y)) >> 31u) != 0u) { fail(1u); }
    return r;
}
uvec2 neg_i64(uvec2 a) {
    if (long_is_min(a)) { fail(1u); }
    return neg_wrap_i64(a);
}
uvec2 ulong_abs(uvec2 a) {
    if (long_negative(a)) { return neg_wrap_i64(a); }
    return a;
}
uvec2 mul_i64(uvec2 a, uvec2 b) {
    uvec2 r = mul_wrap_i64(a, b);
    uvec2 x = ulong_abs(a);
    uvec2 y = ulong_abs(b);
    uvec2 p00 = wide_mul_u32(x.x, y.x);
    uvec2 p01 = wide_mul_u32(x.x, y.y);
    uvec2 p10 = wide_mul_u32(x.y, y.x);
    uvec2 p11 = wide_mul_u32(x.y, y.y);
    uvec2 w1 = add_wrap_i64(uvec2(p00.y, 0u), uvec2(p01.x, 0u));
    uvec2 w1b = add_wrap_i64(w1, uvec2(p10.x, 0u));
    uvec2 high = add_wrap_i64(add_wrap_i64(add_wrap_i64(p11, uvec2(p01.y, 0u)), uvec2(p10.y, 0u)), uvec2(w1b.y, 0u));
    uvec2 magnitude = uvec2(p00.x, w1b.x);
    bool negative = (long_negative(a) != long_negative(b)) && !(magnitude.x == 0u && magnitude.y == 0u);
    bool over = high.x != 0u || high.y != 0u;
    if (negative) {
        over = over || ulong_lt(uvec2(0u, 0x80000000u), magnitude);
    } else {
        over = over || ((magnitude.y >> 31u) != 0u);
    }
    if (over) { fail(1u); }
    return r;
}
uvec2 shl_i64(uvec2 a, uint count) {
    uint n = count & 63u;
    if (n == 0u) { return a; }
    if (n >= 32u) { return uvec2(0u, a.x << (n - 32u)); }
    return uvec2(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
uvec2 shr_ulong(uvec2 a, uint count) {
    uint n = count & 63u;
    if (n == 0u) { return a; }
    if (n >= 32u) { return uvec2(a.y >> (n - 32u), 0u); }
    return uvec2((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
uvec2 sar_i64(uvec2 a, uint count) {
    uint n = count & 63u;
    if (n == 0u) { return a; }
    uint fill = uint(int(a.y) >> 31);
    if (n >= 32u) { return uvec2(uint(int(a.y) >> int(n - 32u)), fill); }
    return uvec2((a.x >> n) | (a.y << (32u - n)), uint(int(a.y) >> int(n)));
}
uvec4 udivmod_pair(uvec2 n, uvec2 d) {
    uvec2 q = uvec2(0u, 0u);
    uvec2 r = uvec2(0u, 0u);
    for (int i = 63; i >= 0; i = i - 1) {
        r = shl_i64(r, 1u) | (shr_ulong(n, uint(i)) & uvec2(1u, 0u));
        if (!ulong_lt(r, d)) {
            r = sub_wrap_i64(r, d);
            q = q | shl_i64(uvec2(1u, 0u), uint(i));
        }
    }
    return uvec4(q, r);
}
uvec2 div_i64(uvec2 a, uvec2 b) {
    if (long_eq(b, uvec2(0u, 0u))) { fail(2u); return a; }
    if (long_eq(b, uvec2(0xFFFFFFFFu, 0xFFFFFFFFu))) {
        if (long_is_min(a)) { fail(1u); }
        return neg_wrap_i64(a);
    }
    uvec2 q = udivmod_pair(ulong_abs(a), ulong_abs(b)).xy;
    if (long_negative(a) != long_negative(b)) { return neg_wrap_i64(q); }
    return q;
}
uvec2 rem_i64(uvec2 a, uvec2 b) {
    if (long_eq(b, uvec2(0u, 0u))) { fail(2u); return uvec2(0u, 0u); }
    if (long_eq(b, uvec2(0xFFFFFFFFu, 0xFFFFFFFFu))) {
        if (long_is_min(a)) { fail(1u); }
        return uvec2(0u, 0u);
    }
    uvec2 r = udivmod_pair(ulong_abs(a), ulong_abs(b)).zw;
    if (long_negative(a)) { return neg_wrap_i64(r); }
    return r;
}
uvec2 sign_i64(uvec2 x) {
    if (long_negative(x)) { return uvec2(0xFFFFFFFFu, 0xFFFFFFFFu); }
    if (x.x == 0u && x.y == 0u) { return uvec2(0u, 0u); }
    return uvec2(1u, 0u);
}
uvec2 abs_i64(uvec2 x) { return ulong_abs(x); }
uvec2 min_i64(uvec2 a, uvec2 b) { if (long_lt(b, a)) { return b; } return a; }
uvec2 max_i64(uvec2 a, uvec2 b) { if (long_lt(a, b)) { return b; } return a; }
uvec2 clamp_i64(uvec2 x, uvec2 low, uvec2 high) { return min_i64(max_i64(x, low), high); }
float ulong_to_f32(uvec2 m) {
    if (m.y == 0u && m.x < 0x1000000u) { return float(m.x); }
    uint top;
    if (m.y != 0u) { top = 63u - clz_u32(m.y); } else { top = 31u - clz_u32(m.x); }
    uint shift = top - 23u;
    uint mantissa = shr_ulong(m, shift).x;
    uvec2 rest = m & sub_wrap_i64(shl_i64(uvec2(1u, 0u), shift), uvec2(1u, 0u));
    uvec2 half_bit = shl_i64(uvec2(1u, 0u), shift - 1u);
    uint exponent = top;
    if (ulong_lt(half_bit, rest) || (long_eq(rest, half_bit) && ((mantissa & 1u) == 1u))) {
        mantissa = mantissa + 1u;
        if (mantissa == 0x1000000u) { mantissa = 0x800000u; exponent = exponent + 1u; }
    }
    return uintBitsToFloat(((exponent + 127u) << 23u) | (mantissa & 0x7FFFFFu));
}
float long_to_f32(uvec2 x) {
    float f = ulong_to_f32(ulong_abs(x));
    if (long_negative(x)) { return -f; }
    return f;
}
uvec2 to_long(float x) {
    if (is_nan(x)) { return uvec2(0u, 0u); }
    if (x >= 9223372036854775808.0) { return uvec2(0xFFFFFFFFu, 0x7FFFFFFFu); }
    if (x <= -9223372036854775808.0) { return uvec2(0u, 0x80000000u); }
    float a = abs(x);
    uvec2 m;
    if (a < 4294967296.0) {
        m = uvec2(uint(a), 0u);
    } else {
        uint bits = floatBitsToUint(a);
        uint exponent = (bits >> 23u) - 127u;
        m = shl_i64(uvec2((bits & 0x7FFFFFu) | 0x800000u, 0u), exponent - 23u);
    }
    if (x < 0.0) { return neg_wrap_i64(m); }
    return m;
}
";

pub(crate) fn emit_preamble(
    out: &mut String,
    module: &Module,
    packed: &[usize],
    outputs: &[usize],
    fails: bool,
) {
    let manifest = &module.manifest;
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    let params_len = (param_slots(module).1 + packed.len().div_ceil(4)).max(1);
    let _ = writeln!(
        out,
        "#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
layout(std140) uniform CtxBlock {{ highp uvec4 u_ctx[{context_len}]; }};
layout(std140) uniform ParamsBlock {{ highp uvec4 u_params[{params_len}]; }};"
    );
    for (slot, _) in outputs.iter().enumerate() {
        let _ = writeln!(out, "layout(location = {slot}) out uvec4 out{slot};");
    }
    if fails {
        let _ = writeln!(
            out,
            "layout(location = {}) out uvec4 out_fail;",
            outputs.len()
        );
    }
    if !packed.is_empty() {
        let _ = writeln!(out, "uniform highp usampler2D u_inputs_packed;");
    }
    for index in 0..manifest.inputs.len() {
        if !packed.contains(&index) {
            let _ = writeln!(out, "uniform highp usampler2D u_in{index};");
        }
    }
    for index in 0..module.structs.len() {
        let _ = writeln!(out, "{}", struct_decl(module, index));
    }
    let _ = writeln!(
        out,
        "uint current;
bool failed_here;
uint failed_code;
uint work;
void begin_work() {{
    work = 0u;
}}"
    );
    if can_fail(module) {
        let _ = writeln!(
            out,
            "void fail(uint code) {{
    if (!failed_here) {{
        failed_here = true;
        failed_code = code;
    }}
}}"
        );
        if module.accounted() {
            let budget = manifest.budget;
            let _ = writeln!(
                out,
                "bool look() {{
    if (work > {budget}u) {{
        fail({FAILED_BUDGET}u);
        return true;
    }}
    if (work > u_ctx[5].w) {{
        failed_here = true;
        return true;
    }}
    return false;
}}"
            );
        }
    } else {
        let _ = writeln!(out, "void fail(uint code) {{}}");
    }

    // Core bit, NaN, context, and whole-dividend helpers (callees before callers).
    out.push_str(BIT_HELPERS);
    let _ = writeln!(
        out,
        "int whole_i32(int x) {{ return x * int({OPAQUE_ONE_EXPR}); }}
uint whole_u32(uint x) {{ return x * uint({OPAQUE_ONE_EXPR}); }}
ivec2 whole_v2(ivec2 x) {{ return x * int({OPAQUE_ONE_EXPR}); }}
ivec3 whole_v3(ivec3 x) {{ return x * int({OPAQUE_ONE_EXPR}); }}
ivec4 whole_v4(ivec4 x) {{ return x * int({OPAQUE_ONE_EXPR}); }}
bool is_nan(float x) {{
    return (floatBitsToUint(x) & 0x7FFFFFFFu) > 0x7F800000u;
}}
bvec2 is_nan2(vec2 x) {{
    return greaterThan(floatBitsToUint(x) & uvec2(0x7FFFFFFFu), uvec2(0x7F800000u));
}}
bvec3 is_nan3(vec3 x) {{
    return greaterThan(floatBitsToUint(x) & uvec3(0x7FFFFFFFu), uvec3(0x7F800000u));
}}
bvec4 is_nan4(vec4 x) {{
    return greaterThan(floatBitsToUint(x) & uvec4(0x7FFFFFFFu), uvec4(0x7F800000u));
}}
bool key_down(int code) {{
    if (code < 0 || code > 255) {{ return false; }}
    uint c = uint(code);
    uint slot = 2u + (c >> 7u);
    uint lane = (c >> 5u) & 3u;
    uvec4 v = u_ctx[slot];
    uint word = (lane == 0u) ? v.x : ((lane == 1u) ? v.y : ((lane == 2u) ? v.z : v.w));
    return ((word >> (c & 31u)) & 1u) == 1u;
}}
int typed_at(int k) {{
    if (k < 0 || k > 31) {{ return 0; }}
    uint c = uint(k);
    uvec4 v = u_ctx[8u + (c >> 2u)];
    uint lane = c & 3u;
    uint word = (lane == 0u) ? v.x : ((lane == 1u) ? v.y : ((lane == 2u) ? v.z : v.w));
    return int(word);
}}
vec2 stick_at(int which) {{
    if (which == 0) {{ return uintBitsToFloat(u_ctx[6].xy); }}
    if (which == 1) {{ return uintBitsToFloat(u_ctx[6].zw); }}
    if (which == 2) {{ return uintBitsToFloat(u_ctx[7].xy); }}
    return vec2(0.0, 0.0);
}}
bool button_down(int code) {{
    if (code < 0 || code > 23) {{ return false; }}
    return ((uint(uintBitsToFloat(u_ctx[0].w)) >> uint(code)) & 1u) == 1u;
}}
int abs_i32(int x) {{ return (x < 0) ? int(0u - uint(x)) : x; }}
ivec2 abs_i32x2(ivec2 x) {{ return ivec2(abs_i32(x.x), abs_i32(x.y)); }}
ivec3 abs_i32x3(ivec3 x) {{ return ivec3(abs_i32(x.x), abs_i32(x.y), abs_i32(x.z)); }}
ivec4 abs_i32x4(ivec4 x) {{ return ivec4(abs_i32(x.x), abs_i32(x.y), abs_i32(x.z), abs_i32(x.w)); }}
int to_int(float x) {{
    if (is_nan(x)) {{ return 0; }}
    if (x >= 2147483648.0) {{ return 2147483647; }}
    if (x <= -2147483648.0) {{ return (-2147483647 - 1); }}
    return int(x);
}}
ivec2 to_int2(vec2 v) {{ return ivec2(to_int(v.x), to_int(v.y)); }}
ivec3 to_int3(vec3 v) {{ return ivec3(to_int(v.x), to_int(v.y), to_int(v.z)); }}
ivec4 to_int4(vec4 v) {{ return ivec4(to_int(v.x), to_int(v.y), to_int(v.z), to_int(v.w)); }}
float pow_ref(float x, float y) {{
    uint xb = floatBitsToUint(x);
    uint yb = floatBitsToUint(y);
    uint xa = xb & 0x7FFFFFFFu;
    uint ya = yb & 0x7FFFFFFFu;
    if (xb >= 0x00800000u && xb < 0x7F800000u && ya >= 0x00800000u && ya < 0x7F800000u) {{
        return pow(x, y);
    }}
    if (ya == 0u || xb == 0x3F800000u) {{ return 1.0; }}
    if (xa > 0x7F800000u || ya > 0x7F800000u) {{ return uintBitsToFloat(whole_u32(0x7FC00000u)); }}
    float ay = uintBitsToFloat(ya);
    bool y_int = ya < 0x7F800000u && trunc(y) == y;
    bool y_odd = y_int && ay < 16777216.0 && ((uint(ay) & 1u) == 1u);
    uint odd_sign = y_odd ? (xb & 0x80000000u) : 0u;
    if (xa == 0u) {{
        return uintBitsToFloat(odd_sign | (((yb >> 31u) == 1u) ? 0x7F800000u : 0u));
    }}
    if (ya == 0x7F800000u && xa == 0x3F800000u) {{ return 1.0; }}
    if (ya == 0x7F800000u) {{
        return ((xa > 0x3F800000u) == ((yb >> 31u) == 0u)) ? uintBitsToFloat(whole_u32(0x7F800000u)) : 0.0;
    }}
    if (xa == 0x7F800000u) {{
        return uintBitsToFloat(odd_sign | (((yb >> 31u) == 1u) ? 0u : 0x7F800000u));
    }}
    if ((xb >> 31u) == 1u && !y_int) {{ return uintBitsToFloat(whole_u32(0x7FC00000u)); }}
    if ((xb >> 31u) == 1u) {{
        float p = pow(uintBitsToFloat(xa), y);
        return y_odd ? (-p) : p;
    }}
    return pow(x, y);
}}
vec2 pow_ref2(vec2 x, vec2 y) {{ return vec2(pow_ref(x.x, y.x), pow_ref(x.y, y.y)); }}
vec3 pow_ref3(vec3 x, vec3 y) {{ return vec3(pow_ref(x.x, y.x), pow_ref(x.y, y.y), pow_ref(x.z, y.z)); }}
vec4 pow_ref4(vec4 x, vec4 y) {{ return vec4(pow_ref(x.x, y.x), pow_ref(x.y, y.y), pow_ref(x.z, y.z), pow_ref(x.w, y.w)); }}
float atan2_ref(float y, float x) {{
    uint yb = floatBitsToUint(y);
    uint xb = floatBitsToUint(x);
    if ((yb & 0x7FFFFFFFu) == 0u && !is_nan(x)) {{
        uint turn = ((xb >> 31u) == 1u) ? 0x40490FDBu : 0u;
        return uintBitsToFloat((yb & 0x80000000u) | turn);
    }}
    return atan(y, x);
}}
float max_ref(float a, float b) {{
    float tie = uintBitsToFloat(floatBitsToUint(a) & floatBitsToUint(b));
    float m = (a == b) ? tie : max(a, b);
    return is_nan(a) ? b : (is_nan(b) ? a : m);
}}
float min_ref(float a, float b) {{
    float tie = uintBitsToFloat(floatBitsToUint(a) | floatBitsToUint(b));
    float m = (a == b) ? tie : min(a, b);
    return is_nan(a) ? b : (is_nan(b) ? a : m);
}}
float step_ref(float e, float x) {{
    return (!is_nan(e) && !is_nan(x) && x >= e) ? 1.0 : 0.0;
}}
float smoothstep_ref(float l, float h, float x) {{
    float t = min_ref(max_ref((x - l) / (h - l), 0.0), 1.0);
    return t * t * (3.0 - 2.0 * t);
}}
vec2 max_ref2(vec2 a, vec2 b) {{ return vec2(max_ref(a.x, b.x), max_ref(a.y, b.y)); }}
vec2 min_ref2(vec2 a, vec2 b) {{ return vec2(min_ref(a.x, b.x), min_ref(a.y, b.y)); }}
vec2 step_ref2(vec2 e, vec2 x) {{ return vec2(step_ref(e.x, x.x), step_ref(e.y, x.y)); }}
vec2 smoothstep_ref2(vec2 l, vec2 h, vec2 x) {{ return vec2(smoothstep_ref(l.x, h.x, x.x), smoothstep_ref(l.y, h.y, x.y)); }}
vec3 max_ref3(vec3 a, vec3 b) {{ return vec3(max_ref(a.x, b.x), max_ref(a.y, b.y), max_ref(a.z, b.z)); }}
vec3 min_ref3(vec3 a, vec3 b) {{ return vec3(min_ref(a.x, b.x), min_ref(a.y, b.y), min_ref(a.z, b.z)); }}
vec3 step_ref3(vec3 e, vec3 x) {{ return vec3(step_ref(e.x, x.x), step_ref(e.y, x.y), step_ref(e.z, x.z)); }}
vec3 smoothstep_ref3(vec3 l, vec3 h, vec3 x) {{ return vec3(smoothstep_ref(l.x, h.x, x.x), smoothstep_ref(l.y, h.y, x.y), smoothstep_ref(l.z, h.z, x.z)); }}
vec4 max_ref4(vec4 a, vec4 b) {{ return vec4(max_ref(a.x, b.x), max_ref(a.y, b.y), max_ref(a.z, b.z), max_ref(a.w, b.w)); }}
vec4 min_ref4(vec4 a, vec4 b) {{ return vec4(min_ref(a.x, b.x), min_ref(a.y, b.y), min_ref(a.z, b.z), min_ref(a.w, b.w)); }}
vec4 step_ref4(vec4 e, vec4 x) {{ return vec4(step_ref(e.x, x.x), step_ref(e.y, x.y), step_ref(e.z, x.z), step_ref(e.w, x.w)); }}
vec4 smoothstep_ref4(vec4 l, vec4 h, vec4 x) {{ return vec4(smoothstep_ref(l.x, h.x, x.x), smoothstep_ref(l.y, h.y, x.y), smoothstep_ref(l.z, h.z, x.z), smoothstep_ref(l.w, h.w, x.w)); }}
float fmod_1(float x, float y) {{
    uint xb = floatBitsToUint(x) & 0x7FFFFFFFu;
    uint yb = floatBitsToUint(y) & 0x7FFFFFFFu;
    if (xb >= 0x7F800000u || yb > 0x7F800000u || yb == 0u) {{
        return uintBitsToFloat(whole_u32(0x7FC00000u));
    }}
    if (yb == 0x7F800000u) {{ return x; }}
    float q = trunc(x / y);
    if (q == 0.0) {{ return x; }}
    float r;
    if (abs(x) < 1.0e18 && abs(y) > 1.0e-18 && abs(y) < 1.0e18) {{
        float qh = uintBitsToFloat(floatBitsToUint(q) & 0xFFFFF000u);
        float ql = q - qh;
        float yh = uintBitsToFloat(floatBitsToUint(y) & 0xFFFFF000u);
        float yl = y - yh;
        r = (((x - qh * yh) - qh * yl) - ql * yh) - ql * yl;
    }} else {{
        r = x - q * y;
    }}
    bool wrong = (r != 0.0) && ((r < 0.0) != (x < 0.0));
    return wrong ? (r + sign(x) * abs(y)) : r;
}}
vec2 fmod_2(vec2 x, vec2 y) {{ return vec2(fmod_1(x.x, y.x), fmod_1(x.y, y.y)); }}
vec3 fmod_3(vec3 x, vec3 y) {{ return vec3(fmod_1(x.x, y.x), fmod_1(x.y, y.y), fmod_1(x.z, y.z)); }}
vec4 fmod_4(vec4 x, vec4 y) {{ return vec4(fmod_1(x.x, y.x), fmod_1(x.y, y.y), fmod_1(x.z, y.z), fmod_1(x.w, y.w)); }}
int rotl_i32(int x, int n) {{
    uint k = uint(n) & 31u;
    return (k == 0u) ? x : int((uint(x) << k) | (uint(x) >> (32u - k)));
}}
int rotr_i32(int x, int n) {{
    uint k = uint(n) & 31u;
    return (k == 0u) ? x : int((uint(x) >> k) | (uint(x) << (32u - k)));
}}
uint umulhi_u32(uint a, uint b) {{
    uint al = a & 0xFFFFu;
    uint ah = a >> 16u;
    uint bl = b & 0xFFFFu;
    uint bh = b >> 16u;
    uint ll = al * bl;
    uint lh = al * bh;
    uint hl = ah * bl;
    uint hh = ah * bh;
    uint mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}}
int mulhi_i32(int a, int b) {{
    uint u = umulhi_u32(uint(a), uint(b));
    return int(u - ((a < 0) ? uint(b) : 0u) - ((b < 0) ? uint(a) : 0u));
}}
uint udiv_u32(uint narrow, uint b) {{
    uint a = whole_u32(narrow);
    if (b >= 0x80000000u) {{ return (a >= b) ? 1u : 0u; }}
    if (a < 0x80000000u) {{ return uint(int(a) / int(b)); }}
    uint q = uint(int(a >> 1u) / int(b)) << 1u;
    if (a - q * b >= b) {{ q = q + 1u; }}
    return q;
}}
uint urem_u32(uint a, uint b) {{
    return a - udiv_u32(a, b) * b;
}}"
    );

    if can_fail(module) {
        out.push_str(CHECKED_HELPERS);
        out.push_str(&checked_vector_helpers(2));
        out.push_str(&checked_vector_helpers(3));
        out.push_str(&checked_vector_helpers(4));
    }
    if module.holds(Type::Long) {
        out.push_str(LONG_PAIR_HELPERS);
    }
    if crate::wgsl::any_narrow(module) {
        emit_encodings(out);
    }
    emit_mixers(out, module);

    if !packed.is_empty() {
        let _ = writeln!(
            out,
            "uint fetch_packed(uint w) {{
    uint t = w >> 2u;
    uvec4 p = texelFetch(u_inputs_packed, ivec2(int(t & {ROW_MASK}u), int(t >> {ROW_SHIFT}u)), 0);
    uint lane = w & 3u;
    return (lane == 0u) ? p.x : ((lane == 1u) ? p.y : ((lane == 2u) ? p.z : p.w));
}}"
        );
    }
    for (index, port) in manifest.inputs.iter().enumerate() {
        if !packed.contains(&index) {
            let _ = writeln!(
                out,
                "uint fetch_in{index}(uint w) {{
    uint t = w >> 2u;
    uvec4 p = texelFetch(u_in{index}, ivec2(int(t & {ROW_MASK}u), int(t >> {ROW_SHIFT}u)), 0);
    uint lane = w & 3u;
    return (lane == 0u) ? p.x : ((lane == 1u) ? p.y : ((lane == 2u) ? p.z : p.w));
}}"
            );
        }
        let element = port.element();
        let encoding = input_encoding(module, index);
        if !packed.contains(&index) && encoding == Encoding::Word && element == Type::Vec4 {
            let _ = writeln!(
                out,
                "vec4 load_in{index}(uint index) {{
    return uintBitsToFloat(texelFetch(u_in{index}, ivec2(int(index & {ROW_MASK}u), int(index >> {ROW_SHIFT}u)), 0));
}}"
            );
        } else if !packed.contains(&index) && encoding == Encoding::Word && element == Type::IVec4 {
            let _ = writeln!(
                out,
                "ivec4 load_in{index}(uint index) {{
    return ivec4(texelFetch(u_in{index}, ivec2(int(index & {ROW_MASK}u), int(index >> {ROW_SHIFT}u)), 0));
}}"
            );
        } else {
            let stride = if encoding == Encoding::Word {
                port.ty.schema.value_words()
            } else {
                port.ty.schema.element_bytes()
            };
            let (fetcher, offset) = input_source_glsl(module, packed, index, encoding);
            let (expr, _) = load_expr_glsl(
                module,
                element,
                &fetcher,
                &format!("({offset}b)"),
                0,
                encoding,
            );
            let _ = writeln!(
                out,
                "{} load_in{index}(uint index) {{
    uint b = index * {stride}u;
    return {expr};
}}",
                glsl_type(element)
            );
        }

        let slot = CONTEXT_HEAD + index;
        let _ = match port.shape() {
            Shape::Rows => writeln!(
                out,
                "uint index_in{index}(int i) {{
    return uint(clamp(i, 0, int(u_ctx[{slot}].x) - 1));
}}"
            ),
            Shape::Grid => writeln!(
                out,
                "uint index_in{index}(vec2 c) {{
    uint w = u_ctx[{slot}].x;
    uint h = u_ctx[{slot}].y;
    uint x = min(uint(is_nan(c.x) ? 0.0 : max(floor(c.x), 0.0)), w - 1u);
    uint y = min(uint(is_nan(c.y) ? 0.0 : max(floor(c.y), 0.0)), h - 1u);
    return y * w + x;
}}"
            ),
            Shape::Volume => writeln!(
                out,
                "uint index_in{index}(vec3 c) {{
    uint w = u_ctx[{slot}].x;
    uint h = u_ctx[{slot}].y;
    uint d = u_ctx[{slot}].z;
    uint x = min(uint(is_nan(c.x) ? 0.0 : max(floor(c.x), 0.0)), w - 1u);
    uint y = min(uint(is_nan(c.y) ? 0.0 : max(floor(c.y), 0.0)), h - 1u);
    uint z = min(uint(is_nan(c.z) ? 0.0 : max(floor(c.z), 0.0)), d - 1u);
    return (z * h + y) * w + x;
}}"
            ),
        };
        emit_linear(out, module, index, port);
    }
    out.push('\n');
}

pub(crate) fn input_source_glsl(
    module: &Module,
    packed: &[usize],
    index: usize,
    encoding: Encoding,
) -> (String, String) {
    let Some(slot) = packed.iter().position(|&k| k == index) else {
        return (format!("fetch_in{index}"), String::new());
    };
    let first = param_slots(module).1;
    let lane = ["x", "y", "z", "w"][slot % 4];
    let offset = format!("u_params[{}u].{lane}", first + slot / 4);
    let offset = if encoding == Encoding::Word {
        offset
    } else {
        format!("({offset} * 4u)")
    };
    ("fetch_packed".to_owned(), format!("{offset} + "))
}

fn f32_bits_glsl(value: f32) -> String {
    format!("uintBitsToFloat({:#010x}u)", value.to_bits())
}

fn emit_encodings(out: &mut String) {
    let _ = writeln!(
        out,
        "uint half_to_f32(uint h) {{
    uint sign_bit = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0u) {{
        if (m == 0u) {{ return sign_bit; }}
        uint shift = 10u - first_leading_bit_u32(m);
        return sign_bit | ((113u - shift) << 23u) | (((m << shift) & 0x3FFu) << 13u);
    }}
    if (e == 0x1Fu) {{ return sign_bit | 0x7F800000u | (m << 13u); }}
    return sign_bit | ((e + 112u) << 23u) | (m << 13u);
}}
uint f32_to_half(uint x) {{
    uint sign_bit = (x >> 16u) & 0x8000u;
    int exponent = int((x >> 23u) & 0xFFu);
    uint mantissa = x & 0x7FFFFFu;
    if (exponent == 255) {{
        return sign_bit | 0x7C00u | ((mantissa != 0u) ? (0x200u | (mantissa >> 13u)) : 0u);
    }}
    int e = exponent - 112;
    if (e >= 31) {{ return sign_bit | 0x7C00u; }}
    uint value = 0u;
    uint dropped = 0u;
    uint shift = 13u;
    if (e <= 0) {{
        if (e < -10) {{ return sign_bit; }}
        uint full = mantissa | 0x800000u;
        shift = uint(14 - e);
        value = full >> shift;
        dropped = full & ((1u << shift) - 1u);
    }} else {{
        value = (uint(e) << 10u) | (mantissa >> 13u);
        dropped = mantissa & 0x1FFFu;
    }}
    uint half_bit = 1u << (shift - 1u);
    bool up = dropped > half_bit || (dropped == half_bit && ((value & 1u) == 1u));
    return sign_bit | (value + (up ? 1u : 0u));
}}
uint encode_unit(float x, float low, float scale) {{
    bool nan = (floatBitsToUint(x) & 0x7FFFFFFFu) > 0x7F800000u;
    float c = nan ? 0.0 : clamp(x, low, 1.0);
    return uint(int(roundEven(c * scale)));
}}"
    );
}

pub(crate) fn scalar_load_glsl(
    encoding: Encoding,
    kind: Scalar,
    fetcher: &str,
    base: &str,
    k: usize,
) -> String {
    if encoding == Encoding::Word {
        let word = format!("{fetcher}({base} + {k}u)");
        return match kind {
            Scalar::Float => format!("uintBitsToFloat({word})"),
            Scalar::Int => format!("int({word})"),
            Scalar::Bool => format!("({word} != 0u)"),
            Scalar::Long => format!("long_of({word}, {fetcher}({base} + {}u))", k + 1),
            other => unreachable!("the GPU refuses a module holding a {other:?}"),
        };
    }
    let size = encoding.bytes();
    let at = format!("({base} + {}u)", k * size);
    let mask = if size == 1 { "0xFFu" } else { "0xFFFFu" };
    let raw = format!("(({fetcher}({at} >> 2u) >> (({at} & 3u) * 8u)) & {mask})");
    if kind == Scalar::Bool {
        return format!("({raw} != 0u)");
    }
    match encoding {
        Encoding::U8 | Encoding::U16 => format!("int({raw})"),
        Encoding::I8 => format!("(int({raw} << 24u) >> 24)"),
        Encoding::I16 => format!("(int({raw} << 16u) >> 16)"),
        Encoding::Unorm8 => format!("(float({raw}) * {})", f32_bits_glsl(encoding::UNORM8)),
        Encoding::Unorm16 => format!("(float({raw}) * {})", f32_bits_glsl(encoding::UNORM16)),
        Encoding::Snorm8 => format!(
            "max(float(int({raw} << 24u) >> 24) * {}, -1.0)",
            f32_bits_glsl(encoding::SNORM8)
        ),
        Encoding::F16 => format!("uintBitsToFloat(half_to_f32({raw}))"),
        Encoding::Word => unreachable!("answered above"),
    }
}

pub(crate) fn scalar_encode_glsl(encoding: Encoding, kind: Scalar, value: &str) -> String {
    if kind == Scalar::Bool {
        return format!("({value} ? 1u : 0u)");
    }
    match encoding {
        Encoding::U8 | Encoding::I8 => format!("(uint({value}) & 0xFFu)"),
        Encoding::U16 | Encoding::I16 => format!("(uint({value}) & 0xFFFFu)"),
        Encoding::Unorm8 => format!("encode_unit({value}, 0.0, 255.0)"),
        Encoding::Unorm16 => format!("encode_unit({value}, 0.0, 65535.0)"),
        Encoding::Snorm8 => format!("(encode_unit({value}, -1.0, 127.0) & 0xFFu)"),
        Encoding::F16 => format!("f32_to_half(floatBitsToUint({value}))"),
        Encoding::Word => match kind {
            Scalar::Float => format!("floatBitsToUint({value})"),
            _ => format!("uint({value})"),
        },
    }
}

pub(crate) fn load_expr_glsl(
    module: &Module,
    ty: Type,
    fetcher: &str,
    base: &str,
    offset: usize,
    encoding: Encoding,
) -> (String, usize) {
    compose_glsl(module, ty, offset, &|kind, k| {
        scalar_load_glsl(encoding, kind, fetcher, base, k)
    })
}

pub(crate) fn compose_glsl(
    module: &Module,
    ty: Type,
    offset: usize,
    scalar: &dyn Fn(Scalar, usize) -> String,
) -> (String, usize) {
    let parts = |kind: Scalar, count: usize| -> (Vec<String>, usize) {
        let width = kind.words();
        let list = (0..count)
            .map(|k| scalar(kind, offset + k * width))
            .collect();
        (list, count * width)
    };
    match ty {
        Type::Scalar(kind) => (scalar(kind, offset), kind.words()),
        Type::Vector(kind, n) => {
            let (list, used) = parts(kind, usize::from(n));
            (construct_type(module, ty, &list), used)
        }
        Type::Matrix(kind, n) => {
            let (list, used) = parts(kind, usize::from(n) * usize::from(n));
            (construct_type(module, ty, &list), used)
        }
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let mut list = Vec::with_capacity(def.fields.len());
            let mut used = 0;
            for field in &def.fields {
                let (expr, n) = compose_glsl(module, field.ty, offset + used, scalar);
                list.push(expr);
                used += n;
            }
            (construct_type(module, ty, &list), used)
        }
    }
}

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

fn mix_call(module: &Module, ty: Type, a: &str, b: &str, t: &str) -> String {
    match ty {
        Type::Struct(index) => format!("mix_S{index}({a}, {b}, {t})"),
        _ if ty.order().is_some() => {
            let n = usize::from(ty.order().unwrap_or(0));
            let cols: Vec<String> = (0..n)
                .map(|c| format!("mix({a}[{c}], {b}[{c}], {t})"))
                .collect();
            construct_type(module, ty, &cols)
        }
        _ => format!("mix({a}, {b}, {t})"),
    }
}

fn emit_mixers(out: &mut String, module: &Module) {
    for (index, def) in module.structs.iter().enumerate() {
        if !all_floats(module, Type::Struct(index as u32)) {
            continue;
        }
        let fields: Vec<String> = def
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
            .collect();
        let body = construct_type(module, Type::Struct(index as u32), &fields);
        let _ = writeln!(
            out,
            "S{index} mix_S{index}(S{index} a, S{index} b, float t) {{ return {body}; }}"
        );
    }
}

fn emit_linear(out: &mut String, module: &Module, index: usize, port: &entelechy_ir::Port) {
    let element = port.element();
    if !all_floats(module, element) {
        return;
    }
    let slot = CONTEXT_HEAD + index;
    let ty = glsl_type(element);
    let load =
        |x: &str, y: &str, z: &str| format!("load_in{index}((({z}) * h + ({y})) * w + ({x}))");
    let mix = |a: String, b: String, t: &str| mix_call(module, element, &a, &b, t);
    match port.shape() {
        Shape::Rows => {
            let _ = writeln!(
                out,
                "{ty} linear_in{index}(float c) {{
    uint w = u_ctx[{slot}].x;
    uint h = 1u;
    float t = is_nan(c) ? -1.0 : (c - 0.5);
    uint lower = min(uint(max(floor(t), 0.0)), w - 1u);
    uint upper = min(lower + 1u, w - 1u);
    float f = (t < 0.0) ? 0.0 : clamp(t - floor(t), 0.0, 1.0);
    return {};
}}",
                mix(load("lower", "0u", "0u"), load("upper", "0u", "0u"), "f")
            );
        }
        Shape::Grid => {
            let _ = writeln!(
                out,
                "{ty} linear_in{index}(vec2 c) {{
    uint w = u_ctx[{slot}].x;
    uint h = u_ctx[{slot}].y;
    vec2 t = mix(c - vec2(0.5, 0.5), vec2(-1.0, -1.0), is_nan2(c));
    uvec2 lower = min(uvec2(max(floor(t), vec2(0.0, 0.0))), uvec2(w - 1u, h - 1u));
    uvec2 upper = min(lower + uvec2(1u, 1u), uvec2(w - 1u, h - 1u));
    vec2 f = mix(clamp(t - floor(t), vec2(0.0, 0.0), vec2(1.0, 1.0)), vec2(0.0, 0.0), lessThan(t, vec2(0.0, 0.0)));
    uint x0 = lower.x; uint x1 = upper.x; uint y0 = lower.y; uint y1 = upper.y;
    {ty} top = {};
    {ty} bottom = {};
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
                "{ty} linear_in{index}(vec3 c) {{
    uint w = u_ctx[{slot}].x;
    uint h = u_ctx[{slot}].y;
    uint d = u_ctx[{slot}].z;
    vec3 t = mix(c - vec3(0.5, 0.5, 0.5), vec3(-1.0, -1.0, -1.0), is_nan3(c));
    uvec3 lower = min(uvec3(max(floor(t), vec3(0.0, 0.0, 0.0))), uvec3(w - 1u, h - 1u, d - 1u));
    uvec3 upper = min(lower + uvec3(1u, 1u, 1u), uvec3(w - 1u, h - 1u, d - 1u));
    vec3 f = mix(clamp(t - floor(t), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0)), vec3(0.0, 0.0, 0.0), lessThan(t, vec3(0.0, 0.0, 0.0)));
    uint x0 = lower.x; uint x1 = upper.x; uint y0 = lower.y; uint y1 = upper.y; uint z0 = lower.z; uint z1 = upper.z;
    {ty} near_top = {};
    {ty} near_bottom = {};
    {ty} far_top = {};
    {ty} far_bottom = {};
    {ty} near = {};
    {ty} far = {};
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
