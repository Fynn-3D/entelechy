//! The WGSL preamble of every shader: context, structs, checked integer
//! and 64-bit helpers, input loaders, and memory writers.

use super::*;

/// Checked integer arithmetic: WGSL's results (wrapping, `x / 0` is `x`,
/// `x % 0` is zero), recording the element and code where the reference
/// fails.
pub(crate) const CHECKED_HELPERS: &str = "
fn add_i32(a: i32, b: i32) -> i32 {
    let r = a + b;
    if (((a ^ r) & (b ^ r)) < 0) { fail(1u); }
    return r;
}
fn sub_i32(a: i32, b: i32) -> i32 {
    let r = a - b;
    if (((a ^ b) & (a ^ r)) < 0) { fail(1u); }
    return r;
}
fn mul_i32(a: i32, b: i32) -> i32 {
    let r = a * b;
    if (mulhi_i32(a, b) != (r >> 31u)) { fail(1u); }
    return r;
}
fn div_i32(a: i32, b: i32) -> i32 {
    if (b == 0) { fail(2u); return a; }
    if (a == i32(-2147483647 - 1) && b == -1) { fail(1u); return a; }
    return whole_i32(a) / b;
}
fn rem_i32(a: i32, b: i32) -> i32 {
    if (b == 0) { fail(2u); return 0; }
    if (a == i32(-2147483647 - 1) && b == -1) { fail(1u); return 0; }
    return whole_i32(a) % b;
}
fn neg_i32(a: i32) -> i32 {
    if (a == i32(-2147483647 - 1)) { fail(1u); }
    return 0 - a;
}
fn udiv_checked(a: u32, b: u32) -> u32 {
    if (b == 0u) { fail(2u); return a; }
    return udiv_u32(a, b);
}
fn urem_checked(a: u32, b: u32) -> u32 {
    if (b == 0u) { fail(2u); return 0u; }
    return urem_u32(a, b);
}
";

/// Checked arithmetic componentwise over an `n`-lane integer vector.
pub(crate) fn checked_vector_helpers(n: u8) -> String {
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
            "fn {op}_i32x{n}(a: vec{n}<i32>, b: vec{n}<i32>) -> vec{n}<i32> {{ return vec{n}<i32>({parts}); }}"
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
        "fn neg_i32x{n}(a: vec{n}<i32>) -> vec{n}<i32> {{ return vec{n}<i32>({parts}); }}"
    );
    out
}

/// `Long` helpers by the reference's rules: checked arithmetic, division
/// and remainder (naga's 64-bit ones fail to compile on the device),
/// sign, and float conversion, all free of `select` on 64-bit values.
pub(crate) const LONG_HELPERS: &str = "
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
    if (b == 0li) { fail(2u); return a; }
    if (b == -1li) {
        if (a == -9223372036854775807li - 1li) { fail(1u); }
        return -a;
    }
    var ua = bitcast<u64>(a);
    if (a < 0li) { ua = bitcast<u64>(-a); }
    var ub = bitcast<u64>(b);
    if (b < 0li) { ub = bitcast<u64>(-b); }
    let q = bitcast<i64>(udivmod_u64(ua, ub).x);
    if ((a < 0li) != (b < 0li)) { return -q; }
    return q;
}
fn rem_i64(a: i64, b: i64) -> i64 {
    if (b == 0li) { fail(2u); return 0li; }
    if (b == -1li) {
        if (a == -9223372036854775807li - 1li) { fail(1u); }
        return 0li;
    }
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
// The high 64 bits of a 128-bit product, from 32-bit halves.
fn umulhi_u64(a: u64, b: u64) -> u64 {
    let a0 = a & 0xFFFFFFFFlu;
    let a1 = a >> 32u;
    let b0 = b & 0xFFFFFFFFlu;
    let b1 = b >> 32u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let middle = (p00 >> 32u) + (p01 & 0xFFFFFFFFlu) + (p10 & 0xFFFFFFFFlu);
    return a1 * b1 + (p01 >> 32u) + (p10 >> 32u) + (middle >> 32u);
}
fn mulhi_i64(a: i64, b: i64) -> i64 {
    var hi = bitcast<i64>(umulhi_u64(bitcast<u64>(a), bitcast<u64>(b)));
    if (a < 0li) { hi = hi - b; }
    if (b < 0li) { hi = hi - a; }
    return hi;
}
fn add_i64(a: i64, b: i64) -> i64 {
    let r = a + b;
    if (((a ^ r) & (b ^ r)) < 0li) { fail(1u); }
    return r;
}
fn sub_i64(a: i64, b: i64) -> i64 {
    let r = a - b;
    if (((a ^ b) & (a ^ r)) < 0li) { fail(1u); }
    return r;
}
fn mul_i64(a: i64, b: i64) -> i64 {
    let r = a * b;
    if (mulhi_i64(a, b) != (r >> 63u)) { fail(1u); }
    return r;
}
fn neg_i64(a: i64) -> i64 {
    if (a == -9223372036854775807li - 1li) { fail(1u); }
    return 0li - a;
}
fn add_i64x2(a: vec2<i64>, b: vec2<i64>) -> vec2<i64> { return vec2<i64>(add_i64(a.x, b.x), add_i64(a.y, b.y)); }
fn add_i64x3(a: vec3<i64>, b: vec3<i64>) -> vec3<i64> { return vec3<i64>(add_i64(a.x, b.x), add_i64(a.y, b.y), add_i64(a.z, b.z)); }
fn sub_i64x2(a: vec2<i64>, b: vec2<i64>) -> vec2<i64> { return vec2<i64>(sub_i64(a.x, b.x), sub_i64(a.y, b.y)); }
fn sub_i64x3(a: vec3<i64>, b: vec3<i64>) -> vec3<i64> { return vec3<i64>(sub_i64(a.x, b.x), sub_i64(a.y, b.y), sub_i64(a.z, b.z)); }
fn mul_i64x2(a: vec2<i64>, b: vec2<i64>) -> vec2<i64> { return vec2<i64>(mul_i64(a.x, b.x), mul_i64(a.y, b.y)); }
fn mul_i64x3(a: vec3<i64>, b: vec3<i64>) -> vec3<i64> { return vec3<i64>(mul_i64(a.x, b.x), mul_i64(a.y, b.y), mul_i64(a.z, b.z)); }
fn neg_i64x2(a: vec2<i64>) -> vec2<i64> { return vec2<i64>(neg_i64(a.x), neg_i64(a.y)); }
fn neg_i64x3(a: vec3<i64>) -> vec3<i64> { return vec3<i64>(neg_i64(a.x), neg_i64(a.y), neg_i64(a.z)); }
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
";

/// A Long's words where it is a pair of them, `i64` the pair: as
/// [`LONG_WORDS`], also used by the scatter passes.
pub(crate) const LONG_PAIR_WORDS: &str = "
alias i64 = vec2<u32>;
fn long_of(low: u32, high: u32) -> i64 { return vec2<u32>(low, high); }
fn long_low(x: i64) -> u32 { return x.x; }
fn long_high(x: i64) -> u32 { return x.y; }
";

/// `Long` on a device without 64-bit integers: `i64` is a pair of words,
/// `vec2<u32>(low, high)`, two's complement ([`LONG_PAIR_WORDS`]), and
/// every operation the emitter writes on a scalar Long is a helper here,
/// by the same rules as [`LONG_HELPERS`]. The words in memory are the
/// same as the device's own.
pub(crate) const LONG_PAIR_HELPERS: &str = "
fn long_negative(a: i64) -> bool { return (a.y >> 31u) != 0u; }
fn long_is_min(a: i64) -> bool { return a.x == 0u && a.y == 0x80000000u; }
fn long_from_int(i: i32) -> i64 { return vec2<u32>(bitcast<u32>(i), select(0u, 0xFFFFFFFFu, i < 0)); }
fn long_from_bool(b: bool) -> i64 { return vec2<u32>(select(0u, 1u, b), 0u); }
fn long_eq(a: i64, b: i64) -> bool { return a.x == b.x && a.y == b.y; }
fn long_lt(a: i64, b: i64) -> bool {
    if (a.y != b.y) { return bitcast<i32>(a.y) < bitcast<i32>(b.y); }
    return a.x < b.x;
}
fn long_le(a: i64, b: i64) -> bool { return !long_lt(b, a); }
fn long_gt(a: i64, b: i64) -> bool { return long_lt(b, a); }
fn long_ge(a: i64, b: i64) -> bool { return !long_lt(a, b); }
// Unsigned order, for magnitudes.
fn ulong_lt(a: i64, b: i64) -> bool {
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}
fn add_wrap_i64(a: i64, b: i64) -> i64 {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}
fn sub_wrap_i64(a: i64, b: i64) -> i64 {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn neg_wrap_i64(a: i64) -> i64 { return sub_wrap_i64(vec2<u32>(0u, 0u), a); }
// The full product of two words.
fn wide_mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let middle = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let low = (p00 & 0xFFFFu) | (middle << 16u);
    return vec2<u32>(low, a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (middle >> 16u));
}
fn mul_wrap_i64(a: i64, b: i64) -> i64 {
    let p = wide_mul_u32(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}
fn add_i64(a: i64, b: i64) -> i64 {
    let r = add_wrap_i64(a, b);
    if ((((a.y ^ r.y) & (b.y ^ r.y)) >> 31u) != 0u) { fail(1u); }
    return r;
}
fn sub_i64(a: i64, b: i64) -> i64 {
    let r = sub_wrap_i64(a, b);
    if ((((a.y ^ b.y) & (a.y ^ r.y)) >> 31u) != 0u) { fail(1u); }
    return r;
}
fn neg_i64(a: i64) -> i64 {
    if (long_is_min(a)) { fail(1u); }
    return neg_wrap_i64(a);
}
// The magnitude as an unsigned pair (the most negative's is 2^63).
fn ulong_abs(a: i64) -> i64 {
    if (long_negative(a)) { return neg_wrap_i64(a); }
    return a;
}
// Whether the product of magnitudes fits under 2^64, and fits the sign.
fn mul_i64(a: i64, b: i64) -> i64 {
    let r = mul_wrap_i64(a, b);
    let x = ulong_abs(a);
    let y = ulong_abs(b);
    // The high 64 bits of the 128-bit product of magnitudes, by words.
    let p00 = wide_mul_u32(x.x, y.x);
    let p01 = wide_mul_u32(x.x, y.y);
    let p10 = wide_mul_u32(x.y, y.x);
    let p11 = wide_mul_u32(x.y, y.y);
    let w1 = add_wrap_i64(vec2<u32>(p00.y, 0u), vec2<u32>(p01.x, 0u));
    let w1b = add_wrap_i64(w1, vec2<u32>(p10.x, 0u));
    let high = add_wrap_i64(add_wrap_i64(add_wrap_i64(p11, vec2<u32>(p01.y, 0u)), vec2<u32>(p10.y, 0u)), vec2<u32>(w1b.y, 0u));
    let magnitude = vec2<u32>(p00.x, w1b.x);
    let negative = long_negative(a) != long_negative(b) && !(magnitude.x == 0u && magnitude.y == 0u);
    var over = high.x != 0u || high.y != 0u;
    if (negative) {
        over = over || ulong_lt(vec2<u32>(0u, 0x80000000u), magnitude);
    } else {
        over = over || (magnitude.y >> 31u) != 0u;
    }
    if (over) { fail(1u); }
    return r;
}
fn shl_i64(a: i64, count: u32) -> i64 {
    let n = count & 63u;
    if (n == 0u) { return a; }
    if (n >= 32u) { return vec2<u32>(0u, a.x << (n - 32u)); }
    return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn shr_ulong(a: i64, count: u32) -> i64 {
    let n = count & 63u;
    if (n == 0u) { return a; }
    if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn sar_i64(a: i64, count: u32) -> i64 {
    let n = count & 63u;
    if (n == 0u) { return a; }
    let fill = bitcast<u32>(bitcast<i32>(a.y) >> 31u);
    if (n >= 32u) { return vec2<u32>(bitcast<u32>(bitcast<i32>(a.y) >> (n - 32u)), fill); }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), bitcast<u32>(bitcast<i32>(a.y) >> n));
}
// The quotient and remainder of magnitudes, a bit at a time.
fn udivmod_pair(n: i64, d: i64) -> vec4<u32> {
    var q = vec2<u32>(0u, 0u);
    var r = vec2<u32>(0u, 0u);
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        r = shl_i64(r, 1u) | (shr_ulong(n, u32(i)) & vec2<u32>(1u, 0u));
        if (!ulong_lt(r, d)) {
            r = sub_wrap_i64(r, d);
            q = q | shl_i64(vec2<u32>(1u, 0u), u32(i));
        }
    }
    return vec4<u32>(q, r);
}
fn div_i64(a: i64, b: i64) -> i64 {
    if (long_eq(b, vec2<u32>(0u, 0u))) { fail(2u); return a; }
    if (long_eq(b, vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu))) {
        if (long_is_min(a)) { fail(1u); }
        return neg_wrap_i64(a);
    }
    let q = udivmod_pair(ulong_abs(a), ulong_abs(b)).xy;
    if (long_negative(a) != long_negative(b)) { return neg_wrap_i64(q); }
    return q;
}
fn rem_i64(a: i64, b: i64) -> i64 {
    if (long_eq(b, vec2<u32>(0u, 0u))) { fail(2u); return vec2<u32>(0u, 0u); }
    if (long_eq(b, vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu))) {
        if (long_is_min(a)) { fail(1u); }
        return vec2<u32>(0u, 0u);
    }
    let r = udivmod_pair(ulong_abs(a), ulong_abs(b)).zw;
    if (long_negative(a)) { return neg_wrap_i64(r); }
    return r;
}
fn sign_i64(x: i64) -> i64 {
    if (long_negative(x)) { return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu); }
    if (x.x == 0u && x.y == 0u) { return vec2<u32>(0u, 0u); }
    return vec2<u32>(1u, 0u);
}
fn abs_i64(x: i64) -> i64 { return ulong_abs(x); }
fn min_i64(a: i64, b: i64) -> i64 { if (long_lt(b, a)) { return b; } return a; }
fn max_i64(a: i64, b: i64) -> i64 { if (long_lt(a, b)) { return b; } return a; }
fn clamp_i64(x: i64, low: i64, high: i64) -> i64 { return min_i64(max_i64(x, low), high); }
// An unsigned pair to the nearest Float, ties to even.
fn ulong_to_f32(m: i64) -> f32 {
    if (m.y == 0u && m.x < 0x1000000u) { return f32(m.x); }
    var top: u32;
    if (m.y != 0u) { top = 63u - countLeadingZeros(m.y); } else { top = 31u - countLeadingZeros(m.x); }
    let shift = top - 23u;
    var mantissa = shr_ulong(m, shift).x;
    let rest = m & sub_wrap_i64(shl_i64(vec2<u32>(1u, 0u), shift), vec2<u32>(1u, 0u));
    let half = shl_i64(vec2<u32>(1u, 0u), shift - 1u);
    var exponent = top;
    if (ulong_lt(half, rest) || (long_eq(rest, half) && (mantissa & 1u) == 1u)) {
        mantissa = mantissa + 1u;
        if (mantissa == 0x1000000u) { mantissa = 0x800000u; exponent = exponent + 1u; }
    }
    return bitcast<f32>(((exponent + 127u) << 23u) | (mantissa & 0x7FFFFFu));
}
fn long_to_f32(x: i64) -> f32 {
    let f = ulong_to_f32(ulong_abs(x));
    if (long_negative(x)) { return -f; }
    return f;
}
// A Float to a Long as the reference converts it: truncated, saturating
// at the ends, and zero for what is not a number.
fn to_long(x: f32) -> i64 {
    if (is_nan(x)) { return vec2<u32>(0u, 0u); }
    if (x >= 9223372036854775808.0) { return vec2<u32>(0xFFFFFFFFu, 0x7FFFFFFFu); }
    if (x <= -9223372036854775808.0) { return vec2<u32>(0u, 0x80000000u); }
    let a = abs(x);
    var m: i64;
    if (a < 4294967296.0) {
        m = vec2<u32>(u32(a), 0u);
    } else {
        let bits = bitcast<u32>(a);
        let exponent = (bits >> 23u) - 127u;
        m = shl_i64(vec2<u32>((bits & 0x7FFFFFu) | 0x800000u, 0u), exponent - 23u);
    }
    if (x < 0.0) { return neg_wrap_i64(m); }
    return m;
}
";

/// A Long's words, also used by the scatter passes.
pub(crate) const LONG_WORDS: &str = "
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

/// The preamble of the map and fold shaders: uniforms, outputs when
/// `with_output`, inputs, structs, input readers and addresses, helpers,
/// mixers, linear readers and memory writers. With `longs` in pairs, a
/// Long is a pair of words (`i64` an alias of `vec2<u32>`).
pub(crate) fn emit_preamble(
    out: &mut String,
    module: &Module,
    with_output: bool,
    packed: &[usize],
    longs: Longs,
) {
    let manifest = &module.manifest;
    if longs == Longs::Pairs && module.holds(Type::Long) {
        out.push_str(LONG_PAIR_WORDS.trim_start());
    }
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    // Packed input offsets follow the parameters, four per slot.
    let params_len = (param_slots(module).1 + packed.len().div_ceil(4)).max(1);
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
    if !packed.is_empty() {
        let _ = writeln!(
            out,
            "@group(0) @binding({PACKED_BINDING}) var<storage, read> inputs_packed: array<u32>;"
        );
    }
    for (index, port) in manifest.inputs.iter().enumerate() {
        if packed.contains(&index) {
            continue;
        }
        // A memory's binding is read-write.
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
fn typed_at(k: i32) -> i32 {{
    if (k < 0 || k > 31) {{ return 0; }}
    let c = u32(k);
    return bitcast<i32>(ctx[8u + (c >> 2u)][c & 3u]);
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
fn max_ref(a: f32, b: f32) -> f32 {{
    let tie = bitcast<f32>(bitcast<u32>(a) & bitcast<u32>(b));
    let m = select(max(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn min_ref(a: f32, b: f32) -> f32 {{
    let tie = bitcast<f32>(bitcast<u32>(a) | bitcast<u32>(b));
    let m = select(min(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn smoothstep_ref(l: f32, h: f32, x: f32) -> f32 {{
    let t = min_ref(max_ref((x - l) / (h - l), f32(0.0)), f32(1.0));
    return t * t * (3.0 - 2.0 * t);
}}
fn max_ref2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    let tie = bitcast<vec2<f32>>(bitcast<vec2<u32>>(a) & bitcast<vec2<u32>>(b));
    let m = select(max(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn min_ref2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    let tie = bitcast<vec2<f32>>(bitcast<vec2<u32>>(a) | bitcast<vec2<u32>>(b));
    let m = select(min(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn smoothstep_ref2(l: vec2<f32>, h: vec2<f32>, x: vec2<f32>) -> vec2<f32> {{
    let t = min_ref2(max_ref2((x - l) / (h - l), vec2<f32>(0.0)), vec2<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}}
fn max_ref3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {{
    let tie = bitcast<vec3<f32>>(bitcast<vec3<u32>>(a) & bitcast<vec3<u32>>(b));
    let m = select(max(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn min_ref3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {{
    let tie = bitcast<vec3<f32>>(bitcast<vec3<u32>>(a) | bitcast<vec3<u32>>(b));
    let m = select(min(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn smoothstep_ref3(l: vec3<f32>, h: vec3<f32>, x: vec3<f32>) -> vec3<f32> {{
    let t = min_ref3(max_ref3((x - l) / (h - l), vec3<f32>(0.0)), vec3<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}}
fn max_ref4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {{
    let tie = bitcast<vec4<f32>>(bitcast<vec4<u32>>(a) & bitcast<vec4<u32>>(b));
    let m = select(max(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn min_ref4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {{
    let tie = bitcast<vec4<f32>>(bitcast<vec4<u32>>(a) | bitcast<vec4<u32>>(b));
    let m = select(min(a, b), tie, a == b);
    return select(select(m, a, b != b), b, a != a);
}}
fn smoothstep_ref4(l: vec4<f32>, h: vec4<f32>, x: vec4<f32>) -> vec4<f32> {{
    let t = min_ref4(max_ref4((x - l) / (h - l), vec4<f32>(0.0)), vec4<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}}
fn to_int2(v: vec2<f32>) -> vec2<i32> {{
    return vec2<i32>(to_int(v.x), to_int(v.y));
}}
fn to_int3(v: vec3<f32>) -> vec3<i32> {{
    return vec3<i32>(to_int(v.x), to_int(v.y), to_int(v.z));
}}
fn to_int4(v: vec4<f32>) -> vec4<i32> {{
    return vec4<i32>(to_int(v.x), to_int(v.y), to_int(v.z), to_int(v.w));
}}"
    );
    // This invocation's element (set by `main`), its failure flag and
    // its work count.
    out.push_str(
        "var<private> current: u32;\nvar<private> failed_here: bool;\nvar<private> work: u32;\n",
    );
    // The next work count that needs a look: the end of the block or one
    // past the budget, so an iteration costs one add and one compare.
    let budget = manifest.budget;
    let _ = writeln!(
        out,
        "var<private> next_look: u32;
var<private> looked: u32;
fn begin_work() {{
    work = 0u;
    looked = 0u;
    next_look = {}u;
}}",
        entelechy_ir::WORK_BLOCK.min(budget + 1)
    );
    if can_fail(module) {
        // An invocation records only its first failure, where the
        // reference stops.
        let _ = writeln!(
            out,
            "struct Failure {{ word: atomic<u32>, blocks: atomic<u32>, limit: u32, left: atomic<u32>, total: atomic<u32> }}
@group(0) @binding({FAILURE_BINDING}) var<storage, read_write> failure: Failure;
fn fail(code: u32) {{
    if (!failed_here) {{
        failed_here = true;
        atomicMax(&failure.word, 0xFFFFFFFFu - (current * 4u + code));
    }}
}}
// Hands the run the blocks this element filled; true once the run has
// passed its total, which the record's count says, not the element.
fn spend(blocks: u32) -> bool {{
    if (atomicAdd(&failure.blocks, blocks) + blocks <= failure.limit) {{
        return false;
    }}
    failed_here = true;
    return true;
}}"
        );
        if module.accounted() {
            // `Op::Look`: past the budget the element fails; otherwise the
            // blocks since the last look go to the run's total, and past
            // the run's limit the count is pinned past the budget.
            let _ = writeln!(
                out,
                "fn look() -> bool {{
    if (work < next_look) {{
        return false;
    }}
    if (work > {budget}u) {{
        fail({FAILED_BUDGET}u);
        return true;
    }}
    if (spend((work >> {shift}u) - (looked >> {shift}u))) {{
        work = {past}u;
        return true;
    }}
    looked = work;
    next_look = min((work | {mask}u) + 1u, {past}u);
    return false;
}}
// What this invocation's finished elements ran short of their blocks,
// kept, and handed to the run's record in blocks and as it is.
var<private> owed: u32;
var<workgroup> leftover: atomic<u32>;
fn keep_left() {{
    owed = owed + (work & {mask}u);
    if (owed >= {handed}u) {{
        settle_left();
    }}
}}
fn settle_left() {{
    if (owed > 0u) {{
        spend(owed >> {shift}u);
        atomicAdd(&failure.left, owed & {mask}u);
        owed = 0u;
    }}
}}",
                shift = entelechy_ir::WORK_BLOCK.trailing_zeros(),
                mask = entelechy_ir::WORK_BLOCK - 1,
                past = budget + 1,
                handed = 64 * entelechy_ir::WORK_BLOCK,
            );
        }
        out.push_str(CHECKED_HELPERS);
        out.push_str(&checked_vector_helpers(2));
        out.push_str(&checked_vector_helpers(3));
        out.push_str(&checked_vector_helpers(4));
    } else {
        out.push_str("fn fail(code: u32) {}\n");
    }
    if module.holds(Type::Long) {
        if longs == Longs::Pairs {
            out.push_str(LONG_PAIR_HELPERS);
        } else {
            out.push_str(LONG_HELPERS);
            out.push_str(LONG_WORDS);
        }
    }
    for (index, port) in manifest.inputs.iter().enumerate() {
        let element = port.element();
        // Words per element, or bytes for a narrow input.
        let encoding = input_encoding(module, index);
        let stride = if encoding == Encoding::Word {
            port.ty.schema.value_words()
        } else {
            port.ty.schema.element_bytes()
        };
        let (buffer, offset) = input_source(module, packed, index, encoding);
        let (expr, _) = load_expr(
            module,
            element,
            &buffer,
            &format!("({offset}b)"),
            0,
            encoding,
        );
        let _ = writeln!(
            out,
            "fn load_in{index}(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
            wgsl_type(element)
        );
        // The linear index of an address, each axis floored (NaN counts as
        // zero) and clamped into the input's extent as an integer: `u32()`
        // saturates, and `min` keeps the last element.
        let slot = CONTEXT_HEAD + index;
        let _ = match port.shape() {
            Shape::Rows => writeln!(
                out,
                "fn index_in{index}(i: i32) -> u32 {{
    return u32(clamp(i, 0i, i32(bitcast<u32>(ctx[{slot}].x)) - 1i));
}}"
            ),
            Shape::Grid => writeln!(
                out,
                "fn index_in{index}(c: vec2<f32>) -> u32 {{
    let w = bitcast<u32>(ctx[{slot}].x);
    let h = bitcast<u32>(ctx[{slot}].y);
    let x = min(u32(select(floor(c.x), 0.0, c.x != c.x)), w - 1u);
    let y = min(u32(select(floor(c.y), 0.0, c.y != c.y)), h - 1u);
    return y * w + x;
}}"
            ),
            Shape::Volume => writeln!(
                out,
                "fn index_in{index}(c: vec3<f32>) -> u32 {{
    let w = bitcast<u32>(ctx[{slot}].x);
    let h = bitcast<u32>(ctx[{slot}].y);
    let d = bitcast<u32>(ctx[{slot}].z);
    let x = min(u32(select(floor(c.x), 0.0, c.x != c.x)), w - 1u);
    let y = min(u32(select(floor(c.y), 0.0, c.y != c.y)), h - 1u);
    let z = min(u32(select(floor(c.z), 0.0, c.z != c.z)), d - 1u);
    return (z * h + y) * w + x;
}}"
            ),
        };
    }
    // The exact truncated float remainder (WGSL's `%` rounds the
    // quotient first): `fma` keeps it exact, and a wrong-signed result
    // is corrected by one divisor.
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
    // Dividends the compiler cannot narrow (see `OPAQUE_ONE`).
    let _ = writeln!(
        out,
        "fn whole_i32(x: i32) -> i32 {{ return x * i32({OPAQUE_ONE}); }}
fn whole_u32(x: u32) -> u32 {{ return x * u32({OPAQUE_ONE}); }}
fn whole_v2(x: vec2<i32>) -> vec2<i32> {{ return x * i32({OPAQUE_ONE}); }}
fn whole_v3(x: vec3<i32>) -> vec3<i32> {{ return x * i32({OPAQUE_ONE}); }}"
    );
    // Rotations by a runtime count (a shift by the width is invalid),
    // the high half of a 64-bit product from 16-bit halves, and unsigned
    // division through the signed one: native `u32 / u32` is wrong on
    // Metal for dividends with the top bit set.
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

/// Whether a type is all floats, so it can be interpolated.
pub(crate) fn all_floats(module: &Module, ty: Type) -> bool {
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

/// `mix` over a type: WGSL's for scalars and vectors, per column for
/// matrices, generated `mix_S{k}` for float structs.
pub(crate) fn mix_call(module: &Module, ty: Type, a: &str, b: &str, t: &str) -> String {
    match ty {
        Type::Struct(index) => format!("mix_S{index}({a}, {b}, {t})"),
        _ if ty.order().is_some() => per_column(ty, |c| format!("mix({a}[{c}], {b}[{c}], {t})")),
        _ => {
            let _ = module;
            format!("mix({a}, {b}, {t})")
        }
    }
}

/// Emits `mix_S{k}` for every float struct, in table order (nested
/// first).
pub(crate) fn emit_mixers(out: &mut String, module: &Module) {
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

/// Emits a memory's `store_in{i}(index, v)` and `inside_in{i}(c)`, the
/// latter true for a floored address inside it (never NaN); a `put`
/// outside is dropped.
pub(crate) fn emit_memory_writer(
    out: &mut String,
    module: &Module,
    index: usize,
    port: &entelechy_ir::Port,
) {
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
        // A scan runs in one invocation, so a narrow scalar is written by
        // read-modify-write of its word.
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
            "fn inside_in{index}(i: i32) -> bool {{ return i >= 0i && i < i32(bitcast<u32>(ctx[{slot}].x)); }}"
        ),
        Shape::Grid => writeln!(
            out,
            "fn inside_in{index}(c: vec2<f32>) -> bool {{
    let x = floor(c.x);
    let y = floor(c.y);
    return x >= 0.0 && x < f32(bitcast<u32>(ctx[{slot}].x)) && y >= 0.0 && y < f32(bitcast<u32>(ctx[{slot}].y));
}}"
        ),
        Shape::Volume => writeln!(
            out,
            "fn inside_in{index}(c: vec3<f32>) -> bool {{
    let x = floor(c.x);
    let y = floor(c.y);
    let z = floor(c.z);
    return x >= 0.0 && x < f32(bitcast<u32>(ctx[{slot}].x)) && y >= 0.0 && y < f32(bitcast<u32>(ctx[{slot}].y)) && z >= 0.0 && z < f32(bitcast<u32>(ctx[{slot}].z));
}}"
        ),
    };
}

/// Emits `linear_in{i}(c)`: the input interpolated at a fractional
/// position, element centres at half-integers, clamped at the edges.
pub(crate) fn emit_linear(
    out: &mut String,
    module: &Module,
    index: usize,
    port: &entelechy_ir::Port,
) {
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
    let w = bitcast<u32>(ctx[{slot}].x);
    let h = 1u;
    let t = select(c - 0.5, -1.0, c != c);
    let lower = min(u32(floor(t)), w - 1u);
    let upper = min(lower + 1u, w - 1u);
    let f = select(clamp(t - floor(t), 0.0, 1.0), 0.0, t < 0.0);
    return {};
}}",
                mix(load("lower", "0u", "0u"), load("upper", "0u", "0u"), "f")
            );
        }
        Shape::Grid => {
            let _ = writeln!(
                out,
                "fn linear_in{index}(c: vec2<f32>) -> {ty} {{
    let w = bitcast<u32>(ctx[{slot}].x);
    let h = bitcast<u32>(ctx[{slot}].y);
    let t = select(c - vec2<f32>(0.5, 0.5), vec2<f32>(-1.0, -1.0), c != c);
    let lower = min(vec2<u32>(floor(t)), vec2<u32>(w - 1u, h - 1u));
    let upper = min(lower + vec2<u32>(1u, 1u), vec2<u32>(w - 1u, h - 1u));
    let f = select(clamp(t - floor(t), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)), vec2<f32>(0.0, 0.0), t < vec2<f32>(0.0, 0.0));
    let x0 = lower.x; let x1 = upper.x; let y0 = lower.y; let y1 = upper.y;
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
    let w = bitcast<u32>(ctx[{slot}].x);
    let h = bitcast<u32>(ctx[{slot}].y);
    let d = bitcast<u32>(ctx[{slot}].z);
    let t = select(c - vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(-1.0, -1.0, -1.0), c != c);
    let lower = min(vec3<u32>(floor(t)), vec3<u32>(w - 1u, h - 1u, d - 1u));
    let upper = min(lower + vec3<u32>(1u, 1u, 1u), vec3<u32>(w - 1u, h - 1u, d - 1u));
    let f = select(clamp(t - floor(t), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)), vec3<f32>(0.0, 0.0, 0.0), t < vec3<f32>(0.0, 0.0, 0.0));
    let x0 = lower.x; let x1 = upper.x; let y0 = lower.y; let y1 = upper.y; let z0 = lower.z; let z1 = upper.z;
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
