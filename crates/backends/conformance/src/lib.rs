#![warn(missing_docs)]
//! The feature matrix: every Kernel language feature under every
//! execution mode, as complete Kernels with inputs, params and context,
//! so a backend is held to the interpreter across the cross product.
//!
//! Feature rows hold a backend to the interpreter on the words it
//! computes. Semantic rows (untaken work, edges, truncation, float
//! edges) carry work that fails if it runs, so a finished run is the
//! proof; the interpreter and the JIT report the failure, and the GPU
//! agrees on the words.
//!
//! A feature is a snippet computing an `Out` from the environment every
//! mode's wrapper provides: `i: Int`, `x: Float`, the inputs `rows`,
//! `ints` and `cells`, `sampled` (the element's own sample of an input
//! of the output's shape), the params `k`, `n`, `flag` and `v`, and
//! three functions. The modes are a map over rows, a map over a grid, a
//! reduction, a scan with a memory, a scatter, an emitting scatter, a
//! prefix, and a map writing four named outputs.

use entelechy_abi::{Context, Domain, Encoding, InputState, Scalar, Schema, Value};

pub mod random;

/// One Kernel of the matrix and everything it runs with.
#[derive(Clone, Debug)]
pub struct Case {
    /// `<feature> in <mode>`.
    pub name: String,
    /// The Kernel's source.
    pub source: String,
    /// The inputs, in port order: layout, extent, words.
    pub inputs: Vec<(Schema, Domain, Vec<u32>)>,
    /// The params, in declaration order.
    pub params: Vec<Value>,
    /// The invocation's domain.
    pub domain: Domain,
    /// The context every case evaluates under.
    pub context: Context,
}

/// The features: a name, `let`s that may use the environment, and the
/// `Out` they yield.
pub fn features() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // `max` of the two zeros is the positive one, `min` the negative,
        // in either order; `clamp` is `max` then `min`. Compared by bits.
        (
            "signed zeros in min max clamp",
            "let nz = from_bits((1 << 31) + i * 0); let pz = from_bits(i * 0);",
            "Out { f: from_bits((bits(max(nz, pz)) >>> 31) + (bits(max(pz, nz)) >>> 31) * 2 + 0x3F800000), i: (bits(min(pz, nz)) >>> 31) + (bits(min(nz, pz)) >>> 31) * 2 + (bits(clamp(nz, pz, 1.0)) >>> 31) * 4, b: bits(min(pz, nz)) < 0, v: vec3(1.0, 2.0, 3.0) }",
        ),
        (
            "float arithmetic",
            "let f = x * 2.0 - 1.0 / (x + 1.0) + x % 0.3;",
            "Out { f: f, i: 0, b: f > 0.5, v: vec3(x, 1.0, 2.0) * 0.5 - vec3(0.25, 0.0, 0.0) }",
        ),
        (
            "int arithmetic",
            "let a = i * 3 - 7 / (i % 5 + 1) + (i % 3) * -2;",
            "Out { f: float(a), i: a, b: a < 0, v: vec3(a, 1, 2) }",
        ),
        (
            "bits shifts wraps",
            "let bits = (((i << 3) ^ (i >> 1)) | (n & 5) ^ ((0 - i) >>> 3)) & 0xFF; let wrapped = add_wrap(0x7FFFFFF0, i) >>> 28; let m = mul_wrap(0x40000001, i);",
            "Out { f: float(bits), i: bits + wrapped + (m >>> 29) + (sub_wrap(0x80000000, i) >>> 30), b: (bits & 1) == 1, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "four-integer vectors",
            "let q = ivec4(ivec2(i, -i), 3, n) * 2 + ivec4(1, 1, 1, 1); let c = clamp(abs(q - ivec4(5, 5, 5, 5)), 0, 9); let r = int(vec4(x, x * 2.0, -x, 0.5) * 3.0); let s = q.wzyx + ivec3(i, 1, 2).xyzz;",
            "Out { f: float(s.x + r.y) + float(c).w, i: min(q, s).z + max(c, 1).x + sign(r).z + r.x, b: q == q.xyzw && s != ivec4(0, 0, 0, 0), v: vec3(float(c).xyz) }",
        ),
        (
            "comparisons and logic",
            "let a = x > 1.0 && !(i == 3) || x <= 0.2; let b = (i != 4) == (x >= 0.5);",
            "Out { f: 0.0, i: int(a) + int(b) * 2, b: a != b, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "vectors",
            "let w = normalize(vec3(x, 1.0, 2.0)) * length(v) + cross(v, vec3(0.0, 1.0, 0.0)) + reflect(vec3(x, -1.0, 0.0), vec3(0.0, 1.0, 0.0)); let d = dot(w, v) + dot(vec2(x, 1.0), vec2(1.0, x)); let vv = vec4(x, 1.0, 2.0, 3.0) * 0.5 + vec4(1, 2, 3, 4);",
            "Out { f: d + vv.w + vv.x, i: 0, b: vec2(x, 1.0) == vec2(x, 1.0) && w != vec3(0.0, 0.0, 0.0), v: mix(w, v, k) + clamp(w, 0.0, 1.0) + step(0.5, w) + smoothstep(0.0, 2.0, w) }",
        ),
        (
            "trig exp pow",
            "let t = sin(x) + cos(x) * tan(x * 0.1) + asin(fract(x * 0.13)) + acos(fract(x * 0.17)) + atan(x) + atan2(x, 3.0) + exp(k * x) + log(x + 1.0) + pow(x + 1.0, 1.5) + sqrt(x);",
            "Out { f: t, i: 0, b: t > 3.0, v: sin(vec3(x, 2.0 * x, 3.0 * x)) + pow(vec3(x + 1.0, 2.0, 3.0), vec3(1.5, 0.5, 2.0)) }",
        ),
        (
            "rounding min max",
            "let r = floor(x * 0.9) + ceil(x * 0.2) + round(x * 0.5) + sign(x - 5.0) + abs(x - 6.0) + max(x, 3.0) + min(x, 4.0);",
            "Out { f: r + fract(x * 7.0), i: clamp(i * 3, 2, 20) + max(i, 4) + min(i, 3) + abs(i - 4) + sign(i - 2), b: false, v: floor(vec3(x, 2.5, 3.7)) + fract(vec3(x, 2.5, -3.7)) }",
        ),
        (
            "conversions",
            "let a = float(i) + float(i > 3); let b = int(x * 1.5) + int(x > 2.0);",
            "Out { f: a, i: b, b: i > 3, v: vec3(i, i + 1, i + 2) }",
        ),
        (
            "structs",
            "let c = Cell { a: x, b: i, on: i % 2 == 0 }; let c2 = Cell { ..c, b: c.b + 1 }; let { a, b: tally, on } = c2; let { x: vx, y: vy } = v * 2.0;",
            "Out { f: a * 2.0 + vx + vy, i: tally, b: on, v: vec3(c.a, float(c.b), 0.0) }",
        ),
        (
            // An enum's variants are its `Int`s; a `match` naming every
            // variant gives a value that is none of them its last arm.
            "enums",
            "let m = if i % 5 == 4 { 9 } else { i % 3 }; let named = match m { Mood.Calm => 10, Mood.Wary | Mood.Angry => 20 + m }; let stepped = next(next(m)); let tagged = Tag { mood: Mood.Angry, n: i };",
            "Out { f: float(named), i: named + stepped * 100 + start * 1000 + tagged.mood, b: m == Mood.Angry, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "if chain",
            "let g = if x < 1.0 { 1 } else if x < 3.0 { let mid = i * 2; mid } else { 3 };",
            "Out { f: if g == 1 { x } else { -x }, i: g, b: g == 3, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // What is not taken is not computed: every division here
            // fails if its side runs, so a finished run is the proof.
            "not taken",
            "let zero = i - i; let branch = if i == 0 { -1 } else { 100 / i }; let arm = match i % 4 { 0 => -1, _ => 100 / (i % 4) }; let guarded = i != 0 && 100 / i > 3; let either = i == 0 || 100 / i > 3; let early = loop 8 with a = 1 until a > 0 { 100 / zero }; let never = loop i % 1 with a = 2 { 100 / zero };",
            "Out { f: float(branch + arm), i: branch + arm + early + never + int(guarded) + int(either) * 2, b: guarded != either, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // Every read past an edge is the edge: a Field clamps, an
            // array clamps, a matrix's column clamps.
            "edges",
            "let far = at(rows, i - 60) + at(rows, i + 60); let corner = at(cells, vec2(float(i) - 20.0, float(i) + 20.0)).a; let lin = at_linear(rows, float(i) - 8.5) + at_linear(rows, float(i) + 44.5); let table = [3, 1, 4, 1]; let picked = table[i - 20] + table[i + 20]; let frame = mat3(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0), vec3(7.0, 8.0, 9.0)); let column = frame[i - 20].x + frame[i + 20].z;",
            "Out { f: far + corner + lin + column, i: picked, b: far > lin, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // A part of an element reads exactly the words its field or
            // index names. The index clamps as an array's, the element
            // address as a Field's.
            "parts of an element",
            "let patch = at(parts, i % 8); let p = patch.points[i % 4] + patch.points[i - 9] + patch.points[i + 9]; let deep = patch.links[i % 3][i % 2] + patch.links[i + 40][1] + patch.links[0 - i][0]; let far = at(parts, i + 100).label + int(at(parts, 0 - i).points[2].z * 10.0); let corner = at(parts, i % 8).points[0].y; let here = near.points[i % 4] + near.points[i + 9]; let own = near.links[i % 3][i % 2] + near.label;",
            "Out { f: p.x + p.z + corner + here.y, i: deep + far + patch.label + own, b: p.y > 0.0, v: p + here }",
        ),
        (
            // Infinity from x / 0, NaN from 0 / 0, and NaN equal to and
            // ordered against nothing. Compared by what the tests say,
            // since a NaN payload is the backend's; `sqrt` of a negative
            // is outside the contract (see the documents' caveat).
            "float edges",
            "let zero = x - x; let up = 1.0 / zero; let down = (0.0 - 1.0) / zero; let undefined = zero / zero; let same = undefined == undefined; let ordered = undefined < 1.0 || undefined > 1.0; let big = up > 1000000000.0;",
            "Out { f: 0.0, i: int(isinf(up)) + int(isinf(down)) * 2 + int(isnan(undefined)) * 4 + int(same) * 8 + int(ordered) * 16 + int(big) * 32 + int(up > down) * 64, b: isnan(undefined), v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // Float to Int at the ends of the range and at NaN.
            "conversion edges",
            "let zero = x - x; let up = 1.0 / zero; let undefined = zero / zero; let huge = 1000000000000000.0 * 1000000000000000.0; let top = int(huge) / 1000 + int(up) / 2000 + int(2147483647.0) / 4000 + int(2147483648.0) / 8000; let bottom = int(0.0 - huge) / 1000 + int(0.0 - up) / 2000 + int(0.0 - 2147483648.0) / 4000; let none = int(undefined); let closest = int(1.9999999) + int(0.0 - 1.9999999) * 3 + int(16777217.0) / 16;",
            "Out { f: float(none), i: top + bottom + none * 3 + closest, b: none == 0, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // `atan2` at the origin, and on the negative axis where a
            // zero's sign decides the side.
            "angles at the origin",
            "let zero = x - x; let minus = zero * (0.0 - 1.0); let one = 1.0 + zero; let a = atan2(zero, zero) + atan2(zero, minus) * 2.0 + atan2(minus, zero) * 4.0; let b = atan2(minus, minus) * 8.0 + atan2(minus, one) * 16.0 + atan2(minus, 0.0 - one) * 32.0; let c = atan2(zero, one) + atan2(one, zero) * 2.0;",
            "Out { f: a + b + c, i: int(isnan(a)) + int(isnan(b)) * 2, b: a + b < 0.0, v: vec3(a, b, c) }",
        ),
        (
            // NaN through the choosing functions (`min`, `max`, `clamp`,
            // `smoothstep`, `step`).
            "not a number through",
            "let zero = x - x; let undefined = zero / zero; let one = 1.0 + zero; let chosen = min(undefined, one) + min(one, undefined) * 2.0 + max(undefined, one) * 4.0 + max(one, undefined) * 8.0; let held = clamp(undefined, zero, one) + saturate(undefined) * 2.0 + smoothstep(zero, one, undefined) * 4.0; let stepped = step(undefined, one) + step(one, undefined) * 2.0; let through = mix(zero, one, undefined) + abs(undefined) + floor(undefined);",
            "Out { f: chosen + held + stepped, i: int(isnan(through)) + int(isnan(held)) * 2 + int(isnan(chosen)) * 4, b: isnan(through), v: vec3(chosen, held, stepped) }",
        ),
        (
            // A quotient truncates toward zero and a remainder takes the
            // dividend's sign, floats included; the unsigned pair reads
            // the same bits as a number without one.
            "signs",
            "let a = i - 24; let q = a / 5; let r = a % 5; let fq = (x - 7.5) / 2.0; let fr = (x - 7.5) % 2.0; let u = udiv(a, 5) >>> 26; let ur = urem(a, 5); let neg = (0 - a) / 5;",
            "Out { f: fq + fr, i: q + r * 3 + u + ur * 5 + neg * 7, b: r < 0, v: vec3(float(q), fr, float(ur)) }",
        ),
        (
            // `&&` and `||` evaluate the right side only when the left
            // leaves the answer open.
            "short circuit",
            "let guarded = i != 0 && 100 / i > 3; let either = i == 0 || 100 / max(i, 1) < 2; let mixed = (x > 0.5 || i > 2) && !(i == 1 && x < 0.1); let chain = i > 0 && i < 9 && x > 0.0 && guarded;",
            "Out { f: if chain { x } else { -x }, i: int(guarded) * 2 + int(either) * 3 + int(mixed) * 5, b: guarded || mixed, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "match",
            "let dense = match i % 7 { 0 => 1, 1 | 2 => { let two = i * 2; two }, 3 => 30, 5 => 50, _ => -1 }; let sparse = match i * 1000 { 0 => 7, 3000 => 8, 9000 => 9, -1 => 10, _ => 11 }; let ranged = match i % 11 { 0..4 => 1, 4..=6 => 2, -3..0 | 9 => 3, 8 => 4, _ => 5 };",
            "Out { f: float(dense), i: dense * 3 + sparse + ranged * 7, b: sparse == 11, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "loops",
            "let walked = loop 12 with acc = Cell { a: 0.0, b: 0, on: false } until acc.b > n { Cell { a: acc.a + sin(acc.a + x) * 0.5 + float(index()), b: acc.b + 1, on: !acc.on } }; let nested = loop 3 with outer = i { loop 2 with inner = outer { inner + outer + index() } }; let counted = loop n with s = 0 { s + index() }; let clamped = loop i % 5 - 1 with s = 1 { s * 2 }; let deep = loop n + 1 with outer = 0 { outer + loop i % 3 with inner = 1 { inner + index() } };",
            "Out { f: walked.a, i: walked.b + nested + counted + clamped + deep, b: walked.on, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // A `break` ends its own loop with its value; outer loops
            // continue.
            "break",
            "let first = loop 20 with best = -1 { let c = i + index(); if c % 7 == 0 { break c; } best }; let armed = loop 8 with acc = 0 { match index() { 0 | 1 => acc + 1, 5 => { break acc * 100; }, _ => acc + 2 } }; let nested = loop 4 with outer = 0 { let inner = loop 4 with a = 0 { if a > 2 { break a; } a + 1 }; if outer > 5 { break outer; } outer + inner }; let tested = loop 12 with a = x until a > 3.0 { if a > 2.0 { break a * 10.0; } a + 0.7 }; let counted = loop n with a = 0 { if index() > i % 5 { break a * 2; } a + index() }; let never = loop 5 with a = 1 { if i < 0 { break 99; } a * 2 };",
            "Out { f: tested, i: first + armed + nested + counted + never, b: tested > 20.0, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // One function, lowered for each type a call gives it.
            "type parameters",
            "let f = twice(x * 0.5); let a = twice(i); let w = twice(v); let c = twice(Cell { a: x, b: i, on: true }.a); let chosen = pick(x, 1.5, i % 2 == 0); let which = pick(i, n, flag) + width(x) + width(ivec2(i, 2)) + width(vec4(x, 1.0, 2.0, 3.0));",
            "Out { f: f + c + chosen, i: a + which, b: w.x > 1.0, v: w }",
        ),
        (
            "calls",
            "let w = twist(v + vec3(x, 0.0, 0.0), k * x); let p = peek(cells, vec2(x * 0.7, x * 0.3)); let s = score(Cell { a: x, b: i, on: true });",
            "Out { f: p + s.a, i: s.b, b: s.on, v: w }",
        ),
        (
            "field reads",
            "let c = at(cells, vec2(x * 0.7, x * 0.3)); let r = at(rows, i % 32) + at(rows, -3) + at(rows, 99); let lin = at_linear(rows, x * 0.37 + 0.2); let sz = size(cells);",
            "Out { f: c.a + r + lin + sz.x + sampled, i: c.b + count(rows) + count(cells) + at(ints, i % 32), b: c.on, v: vec3(sz.x, sz.y, 0.0) }",
        ),
        (
            "context",
            "let t = time() + rate() + window().x + window().y + scale() + pointer().x + pointer().y + wheel().x + wheel().y; let keys = int(key(65)) + int(key(37)) * 2 + int(pointer_down()) * 4 + int(button(1)) * 8 + int(button(2)) * 16; let pad = stick(0) + stick(1) * 2.0 + stick(2) * 4.0 + stick(9); let held = int(button(8)) + int(button(11)) * 2 + int(button(16)) * 4 + int(button(23)) * 8; let chars = typed(0) * 3 + typed(1) + typed(2) + typed(-1) + typed(40) + typed(i % 3);",
            "Out { f: t + pad.x + pad.y, i: keys + held * 32 + pads() * 1024 + chars, b: key(32), v: vec3(window().x, scale(), pad.y) }",
        ),
        (
            "params",
            "let scaled = v * k;",
            "Out { f: k * x + v.y, i: n + int(flag), b: flag && n > 3, v: scaled }",
        ),
        (
            "more math",
            // Makes no NaN: a GPU driver's fast math may fold `inf * 0`.
            // `mod` by 0.71 keeps every remainder off an exact multiple,
            // where a contracted multiply-subtract may land either side.
            "let m = trunc(x * 3.3) + exp2(x * 0.5) + log2(x + 1.0) + sinh(x * 0.2) + cosh(x * 0.2) + tanh(x) + inversesqrt(x + 1.0) + distance(v, vec3(x, 0.0, 0.0)) + radians(x) + degrees(x) + saturate(x - 2.0) + fma(x, 2.0, 1.0) + mod(x, 0.71) + mod(-x, 0.71) - trunc(-x * 1.5);",
            "Out { f: m, i: int(isinf(x / 0.0)) * 2 + int(isnan(x)) * 4 + int(isinf(x)) * 8, b: isnan(x) || isinf(x), v: refract(normalize(vec3(x, -1.0, 0.0)), vec3(0.0, 1.0, 0.0), 0.7) + saturate(vec3(x, -1.0, 2.0)) + mod(vec3(x, 2.0, -3.0), 0.71) + fma(v, v, v) + trunc(vec3(x, -x, 2.5)) }",
        ),
        (
            "int primitives",
            "let bitsy = popcount(i * 77) + clz(i + 1) + ctz(i * 4 + 8) + rotl(i, 5) + (rotr(0x80000001, i) & 0xFFFF) + (udiv(0xFFFFFFF0, i + 1) & 0xFFFF) + (urem(0xFFFFFFF0, i + 3) & 0xFFFF) + mulhi(0x40000000, i * 1000) + umulhi(0xF0000000, i * 1000 + 1) + clz(0) + ctz(0) + rotl(i, 32) + rotr(i, 0) + mulhi(-3, 0x60000000) + (rotr(0x80000001, i) >>> 16);",
            "Out { f: 0.0, i: bitsy, b: ult(0x80000000, i) != ult(i, 0x80000000), v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "arrays",
            "let lut = [3, 1, 4, 1, 5, 9, 2, 6]; let fl = [x, x * 2.0, 3.0]; let reps = [i; 4]; let grid = [[1, 2], [3, 4], [5, 6]]; let cells = [Cell { a: x, b: i, on: true }, Cell { a: 1.0, b: 2, on: false }]; let t = Table { rows: [i, i + 1, i + 2, i + 3], scale: 0.5 }; let rolled = roll(t.rows, i % 4); let vs = [v, vec3(x, 0.0, 1.0)]; let built = [each 5 { let k = index() * 2; k + i }]; let squares = [each 3 { loop 4 with s = 0 { s + index() * built[index()] } }]; let turned = loop 3 with arr = [1, 2, 3, 4] { [each 4 { arr[(index() + 1) % 4] + arr[i % 4] }] }; let picked = turned[i % 4] + lut[i % 8] + lut[2] + lut[-i] + lut[i + 100] + reps[i % 3] + grid[i % 3][(i + 1) % 2] + grid[2][0] + cells[i % 2].b + rolled[0] + rolled[i % 4] + t.rows[3] + built[i % 5] + squares[i % 3];",
            "Out { f: fl[i % 3] + fl[1] * t.scale + cells[i % 2].a + vs[i % 2].y + vs[1].z, i: picked, b: cells[(i + 1) % 2].on, v: vs[i % 2] + mix(vs[0], vs[1], k) }",
        ),
        (
            "matrices",
            "let m2 = spin(x); let m3 = mat3(vec3(x + 2.0, 1.0, 0.0), vec3(0.0, x + 2.0, 1.0), vec3(1.0, 0.0, x + 2.0)); let m4 = mat4(vec4(2.0, 0.0, 0.0, 0.0), vec4(0.0, 3.0, 0.0, 0.0), vec4(0.0, 0.0, 4.0, 0.0), vec4(x, 1.0, 2.0, 1.0)); let ident = inverse(m3) * m3; let back = inverse(m4) * m4; let sum = m3 + transpose(m3) - mat3(1.0) * 0.5; let sc = 2.0 * m2 * 0.5; let mv = m3 * v; let vm = v * m3; let mm = (m3 * m3)[i % 3]; let neg = -m2; let col = m4[3]; let spun = loop 3 with r = m2 { r * m2 }; let fr = Frame { basis: m3, origin: v }; let mq = m4 * vec4(1.0, 2.0, 3.0, 1.0);",
            "Out { f: determinant(m2) + determinant(m3) + determinant(m4) + ident[1][1] + ident[0][2] + back[2][2] + back[3].x + sum[i % 3][(i + 1) % 3] + sc[0].x + mv.x + vm.y + mm.z + neg[1].y + col.w + m3[i % 3].x + v[i % 3] + mq.z + m2[1][0] + (m3 * 2.0)[2][2] + (2.0 * m3)[0][0] + spun[0].x + fr.basis[1].y + fr.origin.z + inverse(m2)[0].x, i: int(m2 == m2) + int(m3 != m3) * 2 + int(-m2 == m2 * -1.0) * 4 + int(neg != m2) * 8, b: sum == transpose(sum), v: mv + vm + mm + vec3(mq.x, mq.y, mq.z) + transpose(m3)[i % 3] + inverse(m4)[3].x * v }",
        ),
        (
            "integer vectors",
            "let c = ivec2(i % 16, i / 16); let d = ivec3(i, -i, 3); let sum = c + ivec2(1, 2) * 3 - d.x; let quot = (d / 2 + d % 3) * ivec3(2, 3, 4) - 7 / (d + ivec3(99, 99, 99)); let bits = ((c << 2) ^ (d.y & 5) | ivec2(1, 0)) >> 1; let ush = ivec2(-8, 16) >>> 2; let m = min(max(d, ivec3(2, 1, 0)), 10) + clamp(d, -2, 2) + clamp(d, ivec3(0, 0, 0), ivec3(1, 1, 1)) + abs(ivec3(-1, 2, -3)) + sign(ivec3(-5, 0, 5)); let cellf = float(c) * 0.5 + vec2(0.25, 0.25); let back = int(cellf) + int(vec3(1.9, -1.9, 0.5)).y; let site = Site { cell: walk(c, ivec2(3, 1)), weight: i }; let looked = at(cells, site.cell).b + at(cells, c).b + at(cells, ivec2(-3, 99)).b; let isz = isize(cells);",
            "Out { f: cellf.x + cellf.y + float(back).x + float(d).z, i: sum.x + sum.y + quot.x + quot.y + quot.z + bits.x + bits.y + ush.x + ush.y + m.x + m.y + m.z + site.cell.x * 100 + site.weight + looked + c[i % 2] + (-c).y + isz.x * 1000 + isz.y, b: c == ivec2(i % 16, i / 16) && d != ivec3(0, 0, 0) && !(c != c), v: vec3(float(m).x, float(sum).y, float(c[1])) }",
        ),
        (
            "narrow division",
            // The top half of a word, then a signed division by a
            // constant: the dividend a shader compiler may narrow to
            // sixteen bits, which the GPU backend guards.
            "let word = add_wrap(mul_wrap(i, 0x10001), 0x12345); let hi = word >>> 16; let hv = ivec2(word, word * 3) >>> 16; let narrow = hi % 7 + hi / 7 + (hi % 6) * 10 + urem(hi, 7) + udiv(hi, 3) + ((word >> 16) & 0xFFFF) % 7 + (hv % 7).x + (hv / 7).y;",
            "Out { f: 0.0, i: narrow, b: hi % 7 == 0, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "swizzles and joins",
            "let p = vec4(v, x); let q = vec4(vec2(x, 2.0), 3, 4.0); let s = vec3(i, vec2(0.5, 0.25)); let w = p.wzyx + q.xyzw + vec4(s.zz, s.xy) + vec4(p.rgb, p.a); let iv = ivec3(ivec2(i, 2), 3).zx + ivec3(1, ivec2(i, i)).yz; let m2 = mat2(w.xy, w.zw); let t = m2 * v.xy + p.ba;",
            "Out { f: w.x + w.w + t.x + t.y + s.y, i: iv.x * 10 + iv.y, b: p.xyz == v && q.gb == vec2(2.0, 3.0), v: w.yzw + vec3(s.xy, 1.0) + p.rrr }",
        ),
        (
            "matrix, integer vector, struct and array parameters",
            "let q = m3 * vec3(x, 1.0, 2.0); let s = site.cell + spot; let w = steps[i % 4] + site.weight;",
            "Out { f: q.x + q.y * 0.5 + q.z, i: s.x * 100 + s.y + w, b: s.x > s.y, v: q }",
        ),
        (
            "float bits",
            "let fb = bits(x); let back = from_bits(fb ^ 1);",
            "Out { f: back, i: fb - bits(from_bits(fb)), b: bits(x * 2.0) > fb, v: vec3(from_bits(fb + 3), x, 0.0) }",
        ),
        (
            "hex and negatives",
            "let h = (0xFF & i) | 0x80000000; let neg = -i * -1 - -3;",
            "Out { f: -x, i: (h >>> 24) + neg, b: h < 0, v: vec3(-1.0, 0x10, -0.5) }",
        ),
    ]
}

/// A mode's wrapper: the snippet's lets and `Out` to a whole Kernel.
pub type Wrapper = fn(&str, &str) -> String;

/// The modes: a name and a wrapper taking the snippet's lets and `Out`.
pub fn modes() -> Vec<(&'static str, Wrapper)> {
    vec![
        ("map over rows", map_rows),
        ("map over a grid", map_grid),
        ("reduce", reduce),
        ("scan", scan),
        ("scatter", scatter),
        ("scatter emitting", scatter_emitting),
        ("prefix", prefix),
        ("map into several outputs", map_several),
    ]
}

const HEAD: &str = "requires { field.read; field.write; input.read; time.read; }
    struct Cell { a: Float; b: Int; on: Bool; }
    struct Out { f: Float; i: Int; b: Bool; v: Vec3; }
    param k: Float = 0.3;
    param n: Int = 7;
    param flag: Bool = true;
    param v: Vec3 = vec3(1.0, 2.0, 3.0);
    param m3: Mat3 = mat3(1.0);
    param spot: IVec2 = ivec2(0, 0);
    param site: Site = Site { cell: ivec2(0, 0), weight: 0 };
    param steps: [Int; 4] = [0; 4];
    fn twist(p: Vec3, t: Float) -> Vec3 { return vec3(p.x * cos(t) - p.y * sin(t), p.x * sin(t) + p.y * cos(t), p.z); }
    fn peek(g: Grid<Cell>, c: Vec2) -> Float { return at(g, c).a; }
    fn score(c: Cell) -> Cell { return Cell { a: c.a * 2.0, b: c.b + 1, on: !c.on }; }
    struct Table { rows: [Int; 4]; scale: Float; }
    struct Patch { points: [Vec3; 4]; links: [[Int; 2]; 3]; label: Int; }
    fn roll(r: [Int; 4], by: Int) -> [Int; 4] { return [r[by % 4], r[(by + 1) % 4], r[(by + 2) % 4], r[(by + 3) % 4]]; }
    struct Frame { basis: Mat3; origin: Vec3; }
    fn spin(a: Float) -> Mat2 { return mat2(vec2(cos(a), sin(a)), vec2(-sin(a), cos(a))); }
    struct Site { cell: IVec2; weight: Int; }
    fn walk(c: IVec2, d: IVec2) -> IVec2 { return (c + d) % ivec2(16, 4); }
    fn twice<T>(value: T) -> T { return value + value; }
    fn pick<T>(a: T, b: T, first: Bool) -> T { return if first { a } else { b }; }
    fn width<T>(value: T) -> Int { let doubled = twice(value); return int(doubled == twice(value)); }
    type Mood = enum { Calm, Wary, Angry };
    struct Tag { mood: Mood; n: Int; }
    param start: Mood = Mood.Wary;
    fn next(m: Mood) -> Mood { return match m { Mood.Calm => Mood.Wary, Mood.Wary => Mood.Angry, Mood.Angry => Mood.Calm }; }
";

const ENV: &str =
    "let x = float(i) * 0.37 + 0.1; let sampled = sample(rows); let near = sample(parts);";
const GRID_ENV: &str =
    "let x = float(i) * 0.37 + 0.1; let sampled = sample(cells).a; let near = sample(patches);";

fn map_rows(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.map_rows@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    output result: Rows<Out>;
    compute {{ let i = element(); {ENV} {lets} let out = {out}; return out; }} }}"
    )
}

fn map_grid(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.map_grid@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    output result: Grid<Out>;
    compute {{ let c0 = icoord(); let i = c0.x + c0.y * idomain().x; {GRID_ENV} {lets} let out = {out}; return out; }} }}"
    )
}

/// A map writing four outputs from one element (the whole `Out`, a
/// vector, an integer, a flag), named in an order other than declared.
fn map_several(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.map_several@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    output result: Grid<Out>; output lane: Grid<Vec3>; output bits: Grid<Int>; output seen: Grid<Bool>;
    compute {{ let c0 = icoord(); let i = c0.x + c0.y * idomain().x; {GRID_ENV} {lets} let out = {out}; return {{ lane: out.v * 2.0 + vec3(x, 0.0, 1.0), seen: out.b != (i % 2 == 0), result: out, bits: out.i ^ 5 }}; }} }}"
    )
}

fn reduce(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.reduce@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    output result: Rows<Out>;
    compute {{ let i = element(); {ENV} {lets} let out = {out}; return out; }}
    merge(a, b) {{ return Out {{ f: a.f + b.f, i: (a.i & 0xFFFF) + (b.i & 0xFFFF), b: a.b || b.b, v: a.v + b.v }}; }} }}"
    )
}

fn scan(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.scan@1 {{ {HEAD}
    input ticks: Rows<Int>; input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    memory mem: Rows<Int>;
    output result: Rows<Out>;
    fn store(m: Rows<Int>, a: Int, v: Int) -> Int {{ return put(m, a, v); }}
    scan with acc = Out {{ f: 0.0, i: 0, b: false, v: vec3(0.0, 0.0, 0.0) }} {{
        let i = index(); {ENV}
        put(mem, i % 16, i);
        store(mem, (i + 1) % 16, at(mem, i % 16) + 1);
        {lets} let out = {out};
        return Out {{ f: out.f + acc.f * 0.5, i: add_wrap(out.i, acc.i), b: out.b != acc.b, v: out.v + acc.v * 0.5 }};
    }} }}"
    )
}

/// A prefix folding with the named rule `max` over a float, an int and
/// a vector; every grouping gives the same result.
fn prefix(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.prefix@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    struct Run {{ f: Float; i: Int; v: Vec3; }}
    output result: Rows<Run>;
    prefix with max {{ let i = element(); {ENV} {lets} let out = {out}; return Run {{ f: out.f, i: out.i, v: out.v }}; }} }}"
    )
}

fn scatter(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.scatter@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>; input target: Rows<Out>;
    output result: Rows<Out>;
    struct Pair {{ at: Int; value: Out; }}
    scatter into target with last {{ let i = element(); {ENV} {lets} let out = {out}; return Pair {{ at: (i * 7) % 16, value: out }}; }} }}"
    )
}

/// A scatter under `max` whose element emits a pair per `fan` before
/// the one it returns; the result is independent of landing order.
fn scatter_emitting(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.scatter_emitting@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>;
    struct Run {{ f: Float; i: Int; v: Vec3; }}
    input target: Rows<Run>;
    output result: Rows<Run>;
    struct Pair {{ at: Int; value: Run; }}
    scatter into target with max {{ let i = element(); {ENV} {lets} let out = {out};
        let _fanned = loop i % 3 with fan = 0 {{
            emit(Pair {{ at: (i * 5 + fan * 3) % 16, value: Run {{ f: out.f, i: out.i ^ fan, v: out.v * float(fan) }} }});
            fan + 1
        }};
        return Pair {{ at: (i * 7) % 16, value: Run {{ f: out.f, i: out.i, v: out.v }} }}; }} }}"
    )
}

/// The count of elements the map, reduce, scan and scatter run over.
pub const COUNT: u32 = 48;

fn rows() -> (Schema, Domain, Vec<u32>) {
    (
        Schema::new(vec![Scalar::Float]),
        Domain::Rows(32),
        (0..32)
            .map(|i| ((i as f32) * 0.21).sin().to_bits())
            .collect(),
    )
}

fn ints() -> (Schema, Domain, Vec<u32>) {
    (
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(32),
        (0..32u32).map(|i| (i * 3) % 11).collect(),
    )
}

fn cells() -> (Schema, Domain, Vec<u32>) {
    (
        Schema::new(vec![Scalar::Float, Scalar::Int, Scalar::Bool]),
        Domain::Grid(16, 4),
        (0..64)
            .flat_map(|i| {
                [
                    ((i as f32) * 0.21).sin().to_bits(),
                    (i % 5) as u32,
                    (i % 3 == 0) as u32,
                ]
            })
            .collect(),
    )
}

/// A Field whose element holds arrays: what a part is taken out of.
fn parts_input() -> (Schema, Domain, Vec<u32>) {
    let mut scalars = vec![Scalar::Float; 12];
    scalars.extend(std::iter::repeat_n(Scalar::Int, 7));
    let words = (0..8u32)
        .flat_map(|e| {
            let points =
                (0..12).map(move |k| ((e as f32) * 0.31 + (k as f32) * 0.17).sin().to_bits());
            let links = (0..6).map(move |k| ((e + k) % 7) as i32 as u32);
            points.chain(links).chain(std::iter::once(e * 3))
        })
        .collect();
    (Schema::new(scalars), Domain::Rows(8), words)
}

/// The same element over a grid, read by `sample`.
fn patches_input() -> (Schema, Domain, Vec<u32>) {
    let (schema, _, _) = parts_input();
    let words = (0..64u32)
        .flat_map(|e| {
            let points =
                (0..12).map(move |k| ((e as f32) * 0.11 + (k as f32) * 0.23).cos().to_bits());
            let links = (0..6).map(move |k| ((e + k * 2) % 5) as i32 as u32);
            points.chain(links).chain(std::iter::once(e % 9))
        })
        .collect();
    (schema, Domain::Grid(16, 4), words)
}

fn out_schema() -> Schema {
    Schema::new(vec![
        Scalar::Float,
        Scalar::Int,
        Scalar::Bool,
        Scalar::Float,
        Scalar::Float,
        Scalar::Float,
    ])
}

fn params() -> Vec<Value> {
    vec![
        Value::Float(0.3),
        Value::Int(7),
        Value::Bool(true),
        Value::Vec3([1.0, 2.0, 3.0]),
        // Not the defaults, so a backend that read the defaults shows.
        Value::Mat3([1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 0.25, 0.0, 1.0]),
        Value::IVec2([3, -2]),
        Value::Record(vec![Value::IVec2([5, 7]), Value::Int(-4)]),
        Value::Record(vec![
            Value::Int(11),
            Value::Int(-3),
            Value::Int(8),
            Value::Int(1),
        ]),
        // `start`, a `Mood`: `Angry`, not the default.
        Value::Int(2),
    ]
}

/// The context every case runs under: time, rate, window, pointer with
/// two buttons held, wheel, three keys down, two pads connected, and two
/// characters typed.
pub fn context() -> Context {
    let mut input = InputState::default();
    input.pointer = [12.5, 7.25];
    input.pointer_down = true;
    input.pointer_middle = true;
    input.wheel = [1.0, -2.0];
    input.set_key(65, true);
    input.set_key(37, true);
    input.set_key(32, true);
    // Pad: sticks off centre, a trigger half down, three buttons held.
    input.sticks = [[0.625, -0.375], [-0.8125, 0.25], [0.5, 0.0]];
    input.set_button(entelechy_abi::pad::SOUTH, true);
    input.set_button(entelechy_abi::pad::NORTH, true);
    input.set_button(entelechy_abi::pad::START, true);
    input.pads = 2;
    input.type_text("你a");
    Context {
        time: 1.5,
        rate: 48000.0,
        input,
        window: [320.0, 200.0],
        scale: 2.0,
    }
}

/// Every feature under every mode.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (feature, lets, result) in features() {
        for (mode, wrap) in modes() {
            out.push(case_in(feature, mode, wrap(lets, result)));
        }
    }
    out.extend(narrow_cases());
    out
}

/// A Kernel of `mode` with that mode's inputs, domain, params and
/// context.
fn case_in(feature: &str, mode: &str, source: String) -> Case {
    let (inputs, domain) = match mode {
        "map over rows" => (
            vec![rows(), ints(), cells(), parts_input(), patches_input()],
            Domain::Rows(COUNT),
        ),
        "map over a grid" => (
            vec![rows(), ints(), cells(), parts_input(), patches_input()],
            Domain::Grid(16, 3),
        ),
        "map into several outputs" => (
            vec![rows(), ints(), cells(), parts_input(), patches_input()],
            Domain::Grid(16, 3),
        ),
        "reduce" => (
            vec![rows(), ints(), cells(), parts_input(), patches_input()],
            Domain::Rows(1),
        ),
        // A prefix maps over its first input, as a scan does.
        "prefix" => (
            vec![rows(), ints(), cells(), parts_input(), patches_input()],
            Domain::Rows(COUNT),
        ),
        "scan" => (
            vec![
                (
                    Schema::new(vec![Scalar::Int]),
                    Domain::Rows(COUNT),
                    vec![0; COUNT as usize],
                ),
                rows(),
                ints(),
                cells(),
                parts_input(),
                patches_input(),
                (
                    Schema::new(vec![Scalar::Int]),
                    Domain::Rows(16),
                    vec![0; 16],
                ),
            ],
            Domain::Rows(COUNT),
        ),
        // The target holds `Run`s starting below most pairs.
        "scatter emitting" => (
            vec![
                rows(),
                ints(),
                cells(),
                parts_input(),
                patches_input(),
                (
                    Schema::new(vec![
                        Scalar::Float,
                        Scalar::Int,
                        Scalar::Float,
                        Scalar::Float,
                        Scalar::Float,
                    ]),
                    Domain::Rows(16),
                    (0..16)
                        .flat_map(|_| {
                            let low = (-1.0e6f32).to_bits();
                            [low, i32::MIN as u32 + 1, low, low, low]
                        })
                        .collect(),
                ),
            ],
            Domain::Rows(16),
        ),
        _ => (
            vec![
                rows(),
                ints(),
                cells(),
                parts_input(),
                patches_input(),
                (out_schema(), Domain::Rows(16), vec![0; 16 * 6]),
            ],
            Domain::Rows(16),
        ),
    };
    Case {
        name: format!("{feature} in {mode}"),
        source,
        inputs,
        params: params(),
        domain,
        context: context(),
    }
}

/// The 64-bit features: a name, `let`s over the wide environment, and
/// the `Wide` they yield. The environment is `i: Int`, `x: Double` and
/// `n: Long` from the element, the inputs `rows: Rows<Double>` and `ids:
/// Rows<Long>`, the params `big`, `origin`, `scale`, `frame` and
/// `steps`, and two functions. The JIT is held to the interpreter; the
/// GPU refuses these rows.
pub fn wide_features() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "double arithmetic",
            "let far = 500000.123456789 + x; let diff = (far - 500000.0) * 1000.0; let q = diff / 3.0 + diff % 0.7 - (-x) + 2 * x; let near = far - 0.1 > 500000.0;",
            "Wide { d: q, l: long(diff), w: dvec3(diff, x, -x) * 0.5 - origin * scale, b: q > 1.0 && q != x && near, i: int(q) }",
        ),
        (
            "long arithmetic",
            "let a = big * n + 7 - n * n * 1000000007l; let q = a / 97l + a % 97 + (0l - a) / 5; let bits = (((a << 20) ^ (a >> 3)) | (n & 5l)) & 0xFFFFFFFFFFl; let u = a >>> 40; let wrapped = (add_wrap(9223372036854775807l, n) >>> 60) + (sub_wrap(-9223372036854775807l, n) >>> 60) + (mul_wrap(big, big) >>> 40); let e = 5000000000 + n; let s = (n << 63) >>> 63;",
            "Wide { d: double(a) * 0.5, l: q + bits + u + wrapped + e + s + steps[i % 3] + abs(-n) + sign(a) + min(a, 3l) + max(n, 2l) + clamp(a, -5l, 5l), w: dvec3(double(q), 0.0, 1.0), b: a > big && a != 0l && n <= 40, i: int(a % 1000l) }",
        ),
        (
            "double vectors and matrices",
            "let p = origin + dvec3(x, 1.0, 2.0) * scale; let local = p - origin; let n3 = normalize(local); let c = cross(local, dvec3(0.0, 0.0, 1.0)); let m = spin(x) * dmat2(2.0); let r = frame * local; let t = transpose(dmat3(dvec3(x + 2.0, 1.0, 0.0), dvec3(0.0, x + 2.0, 1.0), dvec3(1.0, 0.0, x + 2.0))); let inv = inverse(t) * t; let det = determinant(t) + determinant(m); let dm4 = inverse(dmat4(dvec4(2.0, 0.0, 0.0, 0.0), dvec4(0.0, 3.0, 0.0, 0.0), dvec4(0.0, 0.0, 4.0, 0.0), dvec4(x, 1.0, 2.0, 1.0)))[3]; let sw = local.zyx + dvec3(local.xy, 1.0); let col = t[i % 3]; let mv = local * t; let neg = -m; let scaled = 2.0 * t * 0.5 + t - t;",
            "Wide { d: dot(local, n3) + length(c) + det + inv[1][1] + dm4.x + m[1][0] + sw.x + col[i % 3] + mv.y + neg[0].y + scaled[2][2] + dot(dvec2(x, 1.0), dvec2(1.0, x)) + distance(p, origin), l: 0l, w: reflect(n3, dvec3(0.0, 1.0, 0.0)) + r + c + lerp(local, sw, 0.25), b: local == local && n3 != c && t == transpose(transpose(t)), i: int(t == t) + int(m != m) * 2 }",
        ),
        (
            "double math",
            "let t = sin(x) + cos(x) * tan(x * 0.1) + asin(fract(x * 0.13)) + acos(fract(x * 0.17)) + atan(x) + atan2(x, 3.0) + exp(x * 0.1) + log(x + 1.0) + pow(x + 1.0, 1.5) + sqrt(x) + exp2(x * 0.5) + log2(x + 1.0) + sinh(x * 0.2) + cosh(x * 0.2) + tanh(x) + floor(x) + ceil(x) + round(x) + trunc(-x) + sign(x - 5.0) + abs(x - 6.0) + min(x, 3.0) + max(x, 4.0) + clamp(x, 1.0, 2.0) + mix(x, 2.0, 0.25) + step(0.5, x) + smoothstep(0.0, 2.0, x) + fma(x, 2.0, 1.0) + mod(x, 0.71) + radians(x) + degrees(x) + inversesqrt(x + 1.0) + saturate(x - 2.0);",
            "Wide { d: t, l: long(t * 1000.0), w: sin(dvec3(x, 2.0 * x, 3.0 * x)) + pow(dvec3(x + 1.0, 2.0, 3.0), dvec3(1.5, 0.5, 2.0)) + clamp(dvec3(x, -x, 0.5), 0.0, 1.0) + floor(dvec3(x, -x, 2.5)) + min(dvec3(x, 1.0, 2.0), 1.5) + mix(dvec3(x, 0.0, 1.0), dvec3(1.0, 1.0, 1.0), dvec3(0.5, 0.25, 0.0)), b: isnan(t) || isinf(t), i: int(t) }",
        ),
        (
            "long vectors",
            "let a = lvec2(n, big) + lvec2(1l, 2l) * 3; let b = lvec3(a, -n) % lvec3(7l, 11l, 13l); let c = (a << lvec2(3l, 40l)) ^ lvec2(n, n); let widened = lvec2(ivec2(i, -i)) + lvec2(1, 2); let parts = a.yx - lvec2(1l, 1l); let clamped = clamp(lvec3(n, -n, big), -10l, 10l) + clamp(lvec3(n, n, n), lvec3(0l, 1l, 2l), lvec3(3l, 4l, 5l));",
            "Wide { d: double(a.y), l: a.x + b.x + b.y + b.z + c.x + (c.y >>> 30) + widened.y + abs(lvec2(-n, n)).x + min(a, lvec2(5l, 6l)).y + max(a, 4l).x + parts.x + clamped.x + clamped.z + sign(lvec2(-n, 0l)).x + a[i % 2], w: dvec3(double(b.x), double(c.x), 0.0), b: a == a && b != lvec3(0l, 0l, 0l), i: int(b.z) }",
        ),
        (
            "64-bit conversions",
            "let f = float(x); let dd = double(f) + double(i) + double(n) + double(i > 3); let back = int(dd) + int(big) + int(x * 1.5) + int(n); let lng = long(x * 1000.0) + long(i) + long(i > 2) + long(f) + long(-x); let vv = float(dvec3(x, 1.0, 2.0)); let dv = double(vec3(f, 1.0, 2.0)) + double(ivec3(i, -i, 1)); let iv = int(dvec2(x, -x)); let lv = long(vec2(f, -f)) + long(dvec2(x, 2.5)) + long(ivec2(i, 3)); let mm = float(dmat2(x)); let dm = double(mat2(f)); let zero = x - x; let nan = zero / zero; let edge = int(1.0 / zero) / 4 + int(nan) + int(double(big) * 1000000000000.0) / 4 + int(-1.0 / zero) / 4; let ledge = long(1.0 / zero) / 8 + long(nan) + long(-1.0 / zero) / 8;",
            "Wide { d: dd + double(vv.x) + dv.y + double(mm[0][0]) + dm[1][1], l: lng + long(iv.x) + lv.y + lv.x + ledge, w: dv, b: f > 1.0, i: back + edge }",
        ),
        (
            "64-bit structs, arrays and params",
            "let s = Site { at: dvec2(x, -x) + origin.xy, id: n * big }; let arr = [x, x * 2.0, 3.0d]; let longs = [n, big, -n]; let picked = arr[i % 3] + arr[i + 10] + double(longs[i % 3]) + double(steps[i % 3]); let moved = Site { ..s, id: s.id + 1l }; let table = [1.5d, 2.25d, -0.125d]; let each_one = [each 3 { double(index()) * x }];",
            "Wide { d: picked + s.at.y - origin.y + moved.at.x + table[i % 3] + table[i - 40] + each_one[i % 3], l: moved.id + longs[i % 3] + longs[i + 10], w: dvec3(s.at, 1.0) - dvec3(origin.xy, 0.0), b: s.id > 0l, i: int(s.id % 1000l) }",
        ),
        (
            "64-bit field reads and control",
            "let r = at(rows, i % 16) + at(rows, i + 100) + at(rows, -5); let id = at(ids, i % 16) * 3l; let pick = if r > 0.5 { r * 2.0 } else { -r }; let arm = match i % 3 { 0 => id, 1 => -id, _ => 7l }; let walked = loop 5 with acc = x { acc * 0.5 + r }; let counted = loop 4 with acc = n { if acc > 100l { break acc; } acc * 3l };",
            "Wide { d: r + pick + walked, l: id + arm + counted, w: dvec3(r, at(rows, 3), walked), b: r > 0.0, i: int(id % 100l) }",
        ),
    ]
}

const WIDE_HEAD: &str = "requires { field.read; field.write; }
    struct Wide { d: Double; l: Long; w: DVec3; b: Bool; i: Int; }
    struct Site { at: DVec2; id: Long; }
    param big: Long = 5000000000l;
    param origin: DVec3 = dvec3(500000.125, -4000000.0625, 12.5);
    param scale: Double = 0.001;
    param frame: DMat3 = dmat3(1.0);
    param steps: [Long; 3] = [1l, -2l, 3l];
    fn lerp(a: DVec3, b: DVec3, t: Double) -> DVec3 { return a + (b - a) * t; }
    fn spin(a: Double) -> DMat2 { return dmat2(dvec2(cos(a), sin(a)), dvec2(-sin(a), cos(a))); }
";

const WIDE_ENV: &str = "let x = double(i) * 0.37 + 0.1; let n = long(i);";

/// The 64-bit modes: a name, a wrapper, and whether its first input is
/// a scan's row of ticks or a scatter's target follows the inputs.
fn wide_modes() -> Vec<(&'static str, Wrapper)> {
    fn map_rows(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.map_rows@1 {{ {WIDE_HEAD} input rows: Rows<Double>; input ids: Rows<Long>;
    output result: Rows<Wide>;
    compute {{ let i = element(); {WIDE_ENV} {lets} return {out}; }} }}"
        )
    }
    fn map_grid(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.map_grid@1 {{ {WIDE_HEAD} input rows: Rows<Double>; input ids: Rows<Long>;
    output result: Grid<Wide>; output lane: Grid<DVec3>;
    compute {{ let c0 = icoord(); let i = c0.x + c0.y * idomain().x; {WIDE_ENV} {lets} let out = {out}; return {{ lane: out.w * 2.0, result: out }}; }} }}"
        )
    }
    fn reduce(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.reduce@1 {{ {WIDE_HEAD} input rows: Rows<Double>; input ids: Rows<Long>;
    output result: Rows<Wide>;
    compute {{ let i = element(); {WIDE_ENV} {lets} return {out}; }}
    merge(a, b) {{ return Wide {{ d: a.d + b.d, l: add_wrap(a.l, b.l), w: a.w + b.w, b: a.b || b.b, i: (a.i & 0xFFFF) + (b.i & 0xFFFF) }}; }} }}"
        )
    }
    fn scan(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.scan@1 {{ {WIDE_HEAD} input ticks: Rows<Int>; input rows: Rows<Double>; input ids: Rows<Long>;
    output result: Rows<Wide>;
    scan with acc = Wide {{ d: 0.0, l: 0l, w: dvec3(0.0, 0.0, 0.0), b: false, i: 0 }} {{
        let i = index(); {WIDE_ENV} {lets} let out = {out};
        return Wide {{ d: out.d + acc.d * 0.5, l: add_wrap(out.l, acc.l), w: out.w + acc.w * 0.5, b: out.b != acc.b, i: add_wrap(out.i, acc.i) }};
    }} }}"
        )
    }
    fn scatter(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.scatter@1 {{ {WIDE_HEAD} input rows: Rows<Double>; input ids: Rows<Long>; input target: Rows<Acc>;
    struct Acc {{ d: Double; l: Long; w: DVec3; }}
    struct Pair {{ at: Int; value: Acc; }}
    output result: Rows<Acc>;
    scatter into target with sum {{ let i = element(); {WIDE_ENV} {lets} let out = {out}; return Pair {{ at: (i * 7) % 5, value: Acc {{ d: out.d, l: out.l % 1000000l, w: out.w }} }}; }} }}"
        )
    }
    fn prefix(lets: &str, out: &str) -> String {
        format!(
            "kernel wide.prefix@1 {{ {WIDE_HEAD} input rows: Rows<Double>; input ids: Rows<Long>;
    struct Run {{ d: Double; l: Long; w: DVec3; }}
    output result: Rows<Run>;
    prefix with max {{ let i = element(); {WIDE_ENV} {lets} let out = {out}; return Run {{ d: out.d, l: out.l, w: out.w }}; }} }}"
        )
    }
    vec![
        ("map over rows", map_rows as Wrapper),
        ("map over a grid", map_grid),
        ("reduce", reduce),
        ("scan", scan),
        ("scatter", scatter),
        ("prefix", prefix),
    ]
}

fn wide_rows() -> (Schema, Domain, Vec<u32>) {
    let values =
        (0..16).map(|i| Value::Double((f64::from(i) * 0.21).sin() * 500000.0 + 0.123456789));
    (
        Schema::new(vec![Scalar::Double; 2]),
        Domain::Rows(16),
        values.flat_map(|v| v.words()).collect(),
    )
}

fn wide_ids() -> (Schema, Domain, Vec<u32>) {
    let values = (0..16i64).map(|i| Value::Long(i * 3_000_000_019 - 20_000_000_000));
    (
        Schema::new(vec![Scalar::Long; 2]),
        Domain::Rows(16),
        values.flat_map(|v| v.words()).collect(),
    )
}

fn wide_params() -> Vec<Value> {
    vec![
        Value::Long(5_000_000_001),
        // Not the defaults, so a backend that read the defaults shows.
        Value::DVec3([500000.1234567, -4000000.0625, 12.5]),
        Value::Double(0.0011),
        Value::DMat3([1.0, 0.5, 0.0, -0.5, 1.0, 0.0, 0.25, 0.0, 1.0]),
        Value::Record(vec![Value::Long(4), Value::Long(-5), Value::Long(6)]),
    ]
}

/// Every 64-bit feature under every mode it runs in.
pub fn wide_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (feature, lets, result) in wide_features() {
        for (mode, wrap) in wide_modes() {
            let (inputs, domain) = match mode {
                "map over rows" => (vec![wide_rows(), wide_ids()], Domain::Rows(COUNT)),
                "map over a grid" => (vec![wide_rows(), wide_ids()], Domain::Grid(16, 3)),
                "scan" => (
                    vec![
                        (
                            Schema::new(vec![Scalar::Int]),
                            Domain::Rows(COUNT),
                            vec![0; COUNT as usize],
                        ),
                        wide_rows(),
                        wide_ids(),
                    ],
                    Domain::Rows(COUNT),
                ),
                "scatter" => {
                    let mut acc = vec![Scalar::Double; 2];
                    acc.extend([Scalar::Long; 2]);
                    acc.extend([Scalar::Double; 6]);
                    (
                        vec![
                            wide_rows(),
                            wide_ids(),
                            (Schema::new(acc), Domain::Rows(5), vec![0; 5 * 10]),
                        ],
                        Domain::Rows(5),
                    )
                }
                "reduce" => (vec![wide_rows(), wide_ids()], Domain::Rows(1)),
                _ => (vec![wide_rows(), wide_ids()], Domain::Rows(16)),
            };
            out.push(Case {
                name: format!("{feature} in {mode}"),
                source: wrap(lets, result),
                inputs,
                params: wide_params(),
                domain,
                context: context(),
            });
        }
    }
    out
}

/// The `Long`-only features, which a device with 64-bit integers runs
/// in a map and a reduction: a name, `let`s over the environment
/// (`i: Int`, `n: Long`, the input `ids: Rows<Long>`, the params `big`,
/// `origin` and `steps`), and the `Lng` they yield.
pub fn long_features() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "long arithmetic",
            "let a = big * n + 7 - n * n * 1000000007l; let q = a / 97l + a % 97 + (0l - a) / 5; let bits = (((a << 20) ^ (a >> 3)) | (n & 5l)) & 0xFFFFFFFFFFl; let u = a >>> 40; let wrapped = (add_wrap(9223372036854775807l, n) >>> 60) + (sub_wrap(-9223372036854775807l, n) >>> 60) + (mul_wrap(big, big) >>> 40); let e = 5000000000 + n; let s = (n << 63) >>> 63; let big_q = (-9223372036854775807l - 1) / (n + 1) + 9223372036854775807l % (n + 2);",
            "Lng { l: q + bits + u + wrapped + e + s + steps[i % 3] + abs(-n) + sign(a) + min(a, 3l) + max(n, 2l) + clamp(a, -5l, 5l) + big_q / 1000000000l, v: lvec3(q, a, -n), f: float(a) * 0.5, i: int(a % 1000l), b: a > big && a != 0l && n <= 40 }",
        ),
        (
            "long vectors",
            "let a = lvec2(n, big) + lvec2(1l, 2l) * 3; let b = lvec3(a, -n) % lvec3(7l, 11l, 13l) + lvec3(a, n) / 3l; let c = (a << lvec2(3l, 40l)) ^ lvec2(n, n); let d = (a >> 2l) + (lvec2(-n, n) >>> 60l); let widened = lvec2(ivec2(i, -i)) + lvec2(1, 2); let parts = a.yx - lvec2(1l, 1l); let clamped = clamp(lvec3(n, -n, big), -10l, 10l) + clamp(lvec3(n, n, n), lvec3(0l, 1l, 2l), lvec3(3l, 4l, 5l));",
            "Lng { l: a.x + b.x + b.y + b.z + c.x + (c.y >>> 30) + d.x + d.y + widened.y + abs(lvec2(-n, n)).x + min(a, lvec2(5l, 6l)).y + max(a, 4l).x + parts.x + clamped.x + clamped.z + sign(lvec2(-n, 0l)).x + a[i % 2], v: lvec3(b.x, c.x, (a & 255l).y), f: float(b.z), i: int(b.z), b: a == a && b != lvec3(0l, 0l, 0l) }",
        ),
        (
            "long conversions",
            // Floats with exact products, so contraction keeps every
            // truncation.
            "let f = float(i) * 0.25 + 0.5; let back = int(big) + int(n * 3l); let lng = long(f * 1024.0) + long(i) + long(i > 2) + long(-f); let lv = long(vec2(f, -f)) + long(ivec2(i, 3)); let zero = f - f; let ledge = long(1.0 / zero) / 8 + long(zero / zero) + long(-1.0 / zero) / 8; let fl = float(lvec2(n, big)); let iv = int(lvec3(big, n, -n));",
            "Lng { l: lng + lv.y + lv.x + ledge, v: lvec3(lv, n), f: fl.y + float(big) + fl.x, i: back + iv.x + iv.z, b: f > 1.0 }",
        ),
        (
            "long structs, arrays and params",
            "let s = Site { at: lvec2(n, -n) + origin, id: n * big }; let longs = [n, big, -n]; let picked = longs[i % 3] + longs[i + 10] + steps[i % 3]; let moved = Site { ..s, id: s.id + 1l }; let each_one = [each 3 { long(index()) * n }];",
            "Lng { l: picked + moved.id + each_one[i % 3] + s.at.x, v: lvec3(s.at, 1l), f: 0.0, i: int(s.id % 1000l), b: s.id > 0l }",
        ),
        (
            "long reads and control",
            "let id = at(ids, i % 16) * 3l; let arm = match i % 3 { 0 => id, 1 => -id, _ => 7l }; let counted = loop 4 with acc = n { if acc > 100l { break acc; } acc * 3l }; let picked = if id > 0 { id / 7 } else { id % 7 };",
            "Lng { l: id + arm + counted + picked, v: lvec3(id, arm, counted), f: float(id), i: int(id % 100l), b: id > 0l }",
        ),
    ]
}

const LONG_HEAD: &str = "requires { field.read; field.write; }
    struct Lng { l: Long; v: LVec3; f: Float; i: Int; b: Bool; }
    struct Site { at: LVec2; id: Long; }
    param big: Long = 5000000000l;
    param origin: LVec2 = lvec2(500000123456789, -4000000987654321);
    param steps: [Long; 3] = [1l, -2l, 3l];
";

/// Every `Long` feature in a map over rows, a map over a grid, and a
/// reduction.
pub fn long_cases() -> Vec<Case> {
    fn map_rows(lets: &str, out: &str) -> String {
        format!(
            "kernel long.map_rows@1 {{ {LONG_HEAD} input ids: Rows<Long>;
    output result: Rows<Lng>;
    compute {{ let i = element(); let n = long(i); {lets} return {out}; }} }}"
        )
    }
    fn map_grid(lets: &str, out: &str) -> String {
        format!(
            "kernel long.map_grid@1 {{ {LONG_HEAD} input ids: Rows<Long>;
    output result: Grid<Lng>; output lane: Grid<LVec2>;
    compute {{ let c0 = icoord(); let i = c0.x + c0.y * idomain().x; let n = long(i); {lets} let out = {out}; return {{ lane: out.v.xy * 2, result: out }}; }} }}"
        )
    }
    fn reduce(lets: &str, out: &str) -> String {
        format!(
            "kernel long.reduce@1 {{ {LONG_HEAD} input ids: Rows<Long>;
    output result: Rows<Lng>;
    compute {{ let i = element(); let n = long(i); {lets} return {out}; }}
    merge(a, b) {{ return Lng {{ l: add_wrap(a.l, b.l), v: lvec3(add_wrap(a.v.x, b.v.x), add_wrap(a.v.y, b.v.y), add_wrap(a.v.z, b.v.z)), f: a.f + b.f, i: (a.i & 0xFFFF) + (b.i & 0xFFFF), b: a.b || b.b }}; }} }}"
        )
    }
    let modes: [(&str, Wrapper); 3] = [
        ("map over rows", map_rows),
        ("map over a grid", map_grid),
        ("reduce", reduce),
    ];
    let mut out = Vec::new();
    for (feature, lets, result) in long_features() {
        for (mode, wrap) in modes {
            let domain = match mode {
                "map over rows" => Domain::Rows(COUNT),
                "map over a grid" => Domain::Grid(16, 3),
                _ => Domain::Rows(1),
            };
            // A reduction folds over its first input's rows.
            let ids = if mode == "reduce" {
                let values =
                    (0..COUNT as i64).map(|i| Value::Long(i * 3_000_000_019 - 20_000_000_000));
                (
                    Schema::new(vec![Scalar::Long; 2]),
                    Domain::Rows(COUNT),
                    values.flat_map(|v| v.words()).collect(),
                )
            } else {
                wide_ids()
            };
            out.push(Case {
                name: format!("{feature} in {mode}"),
                source: wrap(lets, result),
                inputs: vec![ids],
                params: vec![
                    Value::Long(5_000_000_001),
                    Value::LVec2([500000123456789, -4000000987654322]),
                    Value::Record(vec![Value::Long(4), Value::Long(-5), Value::Long(6)]),
                ],
                domain,
                context: context(),
            });
        }
    }
    out
}

/// Compares two outputs word by word under a layout: an Int or Bool word
/// must match exactly, a Float word within `relative` of the larger
/// magnitude plus `absolute` (NaN matches NaN). Returns the first
/// mismatch as text.
pub fn compare(
    scalars: &[Scalar],
    reference: &[u32],
    candidate: &[u32],
    relative: f32,
    absolute: f32,
) -> Option<String> {
    if reference.len() != candidate.len() {
        return Some(format!(
            "{} words against {}",
            reference.len(),
            candidate.len()
        ));
    }
    let stride = scalars.len().max(1);
    let wide = |words: &[u32], index: usize| {
        u64::from(words[index]) | u64::from(words.get(index + 1).copied().unwrap_or(0)) << 32
    };
    let mut index = 0;
    while index < reference.len() {
        let scalar = scalars[index % stride];
        let (a, b) = (reference[index], candidate[index]);
        let (wa, wb) = (wide(reference, index), wide(candidate, index));
        // Exact equality first: infinities differ by NaN.
        let near = |fa: f64, fb: f64, relative: f64, absolute: f64| {
            fa == fb
                || (fa.is_nan() && fb.is_nan())
                || (fa - fb).abs() <= absolute + relative * fa.abs().max(fb.abs())
        };
        let same = match scalar {
            Scalar::Float => near(
                f64::from(f32::from_bits(a)),
                f64::from(f32::from_bits(b)),
                f64::from(relative),
                f64::from(absolute),
            ),
            // The float rule, scaled to double precision.
            Scalar::Double => near(
                f64::from_bits(wa),
                f64::from_bits(wb),
                f64::from(relative) * 1e-9,
                f64::from(absolute) * 1e-9,
            ),
            Scalar::Long => wa == wb,
            _ => a == b,
        };
        if !same {
            let shown = |w: u32, wide: u64| match scalar {
                Scalar::Float => format!("{}", f32::from_bits(w)),
                Scalar::Int => format!("{}", w as i32),
                Scalar::Bool => format!("{}", w != 0),
                Scalar::Double => format!("{}", f64::from_bits(wide)),
                Scalar::Long => format!("{}", wide as i64),
            };
            return Some(format!(
                "element {} word {} ({scalar:?}): {} against {}",
                index / stride,
                index % stride,
                shown(a, wa),
                shown(b, wb)
            ));
        }
        index += scalar.words();
    }
    None
}

/// Floats that test a narrow encoding's rule: exact values, ties, values
/// past the range on both sides, and for a half the subnormals and the
/// infinities. NaN is outside a device's promise, so none appears.
fn narrow_floats(count: usize) -> Vec<f32> {
    let edges = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        0.25,
        127.5 / 255.0,
        1.5 / 255.0,
        2.0,
        -3.0,
        0.1,
        1.0 / 3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2f32.powi(-20),
        65504.0,
        1.0 + 2f32.powi(-11),
        -0.75,
        0.999,
        1e-9,
    ];
    (0..count)
        .map(|i| edges[i % edges.len()] * if i >= edges.len() { 0.5 } else { 1.0 })
        .collect()
}

/// Integers that test an integer encoding: its bounds, one past each, and
/// what wraps.
fn narrow_ints(count: usize) -> Vec<i32> {
    let edges = [
        0, 1, -1, 127, 128, -128, -129, 255, 256, 300, -300, 32767, 32768, -32768, 65535, 65536,
    ];
    (0..count)
        .map(|i| edges[i % edges.len()] + (i / edges.len()) as i32)
        .collect()
}

/// Every narrow encoding read by `sample`, `at` over rows and a grid, a
/// part of an element and (floats) `at_linear`, and written by a map and
/// by a scan's steps into a memory. Inputs are Kernel words narrowed by
/// the Host's rule.
pub fn narrow_cases() -> Vec<Case> {
    let mut out = Vec::new();
    let words = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    for encoding in Encoding::NARROW {
        let e = encoding.name();
        let kept = |scalars: Vec<Scalar>| {
            Schema::new(scalars)
                .with_encoding(encoding)
                .expect("the encoding fits")
        };
        if encoding.is_float() {
            let map = format!(
                "kernel matrix.narrow_{e}@1 {{
    requires {{ field.read; field.write; }}
    input rows: Rows<Float as {e}>;
    input pairs: Rows<Vec2 as {e}>;
    input grid: Grid<Vec4 as {e}>;
    output kept: Rows<Float as {e}>;
    output halved: Rows<Float as {e}>;
    output widened: Rows<Vec4>;
    output pair: Rows<Vec2 as {e}>;
    compute {{
        let i = element();
        let r = sample(rows);
        let far = at(rows, 15 - i);
        let g = at(grid, vec2(float(i % 4), float(i / 4)));
        let l = at_linear(grid, vec2(float(i % 4) + 0.25, 1.5));
        let y = sample(pairs).y;
        return {{ kept: far, halved: r * 0.5, widened: vec4(r, g.w, l.x, y), pair: vec2(y, r) }};
    }}
}}"
            );
            out.push(Case {
                name: format!("narrow {e} in a map"),
                source: map,
                inputs: vec![
                    (
                        kept(vec![Scalar::Float]),
                        Domain::Rows(16),
                        words(&narrow_floats(16)),
                    ),
                    (
                        kept(vec![Scalar::Float; 2]),
                        Domain::Rows(16),
                        words(
                            &narrow_floats(32)[..]
                                .iter()
                                .rev()
                                .copied()
                                .collect::<Vec<_>>(),
                        ),
                    ),
                    (
                        kept(vec![Scalar::Float; 4]),
                        Domain::Grid(4, 4),
                        words(&narrow_floats(64)),
                    ),
                ],
                params: Vec::new(),
                domain: Domain::Rows(16),
                context: context(),
            });
            let scan = format!(
                "kernel matrix.narrow_{e}_scan@1 {{
    requires {{ field.read; field.write; }}
    input ticks: Rows<Int>;
    memory ram: Rows<Float as {e}>;
    output result: Rows<Float as {e}>;
    scan with acc = 0.0 {{
        let i = index();
        put(ram, i % 8, at(ram, (i + 3) % 8) * 0.5);
        return at(ram, i % 8);
    }}
}}"
            );
            out.push(Case {
                name: format!("narrow {e} in a scan's memory"),
                source: scan,
                inputs: vec![
                    (
                        Schema::new(vec![Scalar::Int]),
                        Domain::Rows(20),
                        vec![0; 20],
                    ),
                    (
                        kept(vec![Scalar::Float]),
                        Domain::Rows(8),
                        words(&narrow_floats(8)),
                    ),
                ],
                params: Vec::new(),
                domain: Domain::Rows(20),
                context: context(),
            });
        } else {
            let ints = |count: usize| {
                narrow_ints(count)
                    .into_iter()
                    .map(|v| v as u32)
                    .collect::<Vec<u32>>()
            };
            let map = format!(
                "kernel matrix.narrow_{e}@1 {{
    requires {{ field.read; field.write; }}
    input rows: Rows<Int as {e}>;
    input pairs: Rows<IVec2 as {e}>;
    input flags: Rows<Bool as {e}>;
    input grid: Grid<Int as {e}>;
    output kept: Rows<Int as {e}>;
    output widened: Rows<IVec3>;
    output flag: Rows<Bool as {e}>;
    compute {{
        let i = element();
        let r = sample(rows);
        let far = at(rows, 15 - i);
        let g = at(grid, vec2(float(i % 4), float(i / 4)));
        let y = sample(pairs).y;
        return {{ kept: add_wrap(far, 300 * i), widened: ivec3(r, g, y), flag: !sample(flags) || i == 3 }};
    }}
}}"
            );
            out.push(Case {
                name: format!("narrow {e} in a map"),
                source: map,
                inputs: vec![
                    (kept(vec![Scalar::Int]), Domain::Rows(16), ints(16)),
                    (kept(vec![Scalar::Int; 2]), Domain::Rows(16), ints(32)),
                    (
                        kept(vec![Scalar::Bool]),
                        Domain::Rows(16),
                        (0..16).map(|i| u32::from(i % 3 == 0)).collect(),
                    ),
                    (kept(vec![Scalar::Int]), Domain::Grid(4, 4), ints(16)),
                ],
                params: Vec::new(),
                domain: Domain::Rows(16),
                context: context(),
            });
            let scan = format!(
                "kernel matrix.narrow_{e}_scan@1 {{
    requires {{ field.read; field.write; }}
    input ticks: Rows<Int>;
    memory ram: Rows<Int as {e}>;
    output result: Rows<Int as {e}>;
    scan with acc = 0 {{
        let i = index();
        put(ram, i % 8, add_wrap(at(ram, (i + 3) % 8), i * 37));
        return add_wrap(acc, at(ram, i % 8));
    }}
}}"
            );
            out.push(Case {
                name: format!("narrow {e} in a scan's memory"),
                source: scan,
                inputs: vec![
                    (
                        Schema::new(vec![Scalar::Int]),
                        Domain::Rows(20),
                        vec![0; 20],
                    ),
                    (kept(vec![Scalar::Int]), Domain::Rows(8), ints(8)),
                ],
                params: Vec::new(),
                domain: Domain::Rows(20),
                context: context(),
            });
        }
    }
    out
}

/// An element a scatter's target may hold: its name, the declarations
/// it needs, its type, a value of it built from `{k}`, and the scalars
/// of its words, a 64-bit one twice.
struct ElementShape {
    name: &'static str,
    decls: &'static str,
    ty: &'static str,
    value: &'static str,
    scalars: Vec<Scalar>,
}

fn element_shapes() -> Vec<ElementShape> {
    vec![
        ElementShape {
            name: "an Int",
            decls: "",
            ty: "Int",
            value: "{k} % 2001 - 1000",
            scalars: vec![Scalar::Int],
        },
        ElementShape {
            name: "a Float",
            decls: "",
            ty: "Float",
            value: "float({k} % 2001 - 1000) * 0.125",
            scalars: vec![Scalar::Float],
        },
        ElementShape {
            name: "a Long",
            decls: "",
            ty: "Long",
            value: "long({k} % 2001 - 1000) * 3000000001l",
            scalars: vec![Scalar::Long; 2],
        },
        ElementShape {
            name: "a Vec3",
            decls: "",
            ty: "Vec3",
            value: "vec3(float({k} % 17) - 8.0, float({k} % 5), -0.5 * float({k} % 11))",
            scalars: vec![Scalar::Float; 3],
        },
        ElementShape {
            name: "two Longs",
            decls: "struct Deep { near: Long; id: Long; }",
            ty: "Deep",
            value: "Deep { near: long({k} % 97) << 33, id: -long({k}) }",
            scalars: vec![Scalar::Long; 4],
        },
        ElementShape {
            name: "a wide record",
            decls: "struct Wide { a: Vec4; b: IVec3; c: Float; d: Vec4; e: Int; f: Vec4; g: Vec4; }",
            ty: "Wide",
            value: "Wide { a: vec4(float({k} % 7), 1.0, -2.0, float({k} % 3)), b: ivec3({k} % 9, -({k} % 4), 5), c: float({k} % 13) * 0.5, d: vec4(0.25, float({k} % 5), 3.0, -1.0), e: {k} % 1000 - 500, f: vec4(float({k} % 11), 0.0, 0.0, 1.0), g: vec4(-float({k} % 6), 2.0, float({k} % 8), 0.5) }",
            scalars: {
                let mut scalars = vec![Scalar::Float; 4];
                scalars.extend([Scalar::Int; 3]);
                scalars.push(Scalar::Float);
                scalars.extend([Scalar::Float; 4]);
                scalars.push(Scalar::Int);
                scalars.extend([Scalar::Float; 8]);
                scalars
            },
        },
    ]
}

/// The elements a shape case runs over: enough for the interpreter and
/// the JIT to split them among threads and the JIT to part the target.
pub const SHAPE_COUNT: u32 = 20_000;

/// Every element shape a scatter's target may hold, under every rule,
/// into rows and into a volume, returning its pairs and emitting more
/// where the rule lets it. The one input is a row of Ints, so every pair
/// is wider than anything the Kernel reads.
pub fn shape_cases() -> Vec<Case> {
    let keys = (
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(SHAPE_COUNT),
        (0..SHAPE_COUNT)
            .map(|i| i.wrapping_mul(2_654_435_761) >> 8)
            .collect::<Vec<u32>>(),
    );
    // Target: declaration, extent, and an address from `{k}` (some
    // outside).
    let targets = [
        ("rows", "Rows", "Int", Domain::Rows(61), "{k} % 67 - 3"),
        (
            "a volume",
            "Volume",
            "Vec3",
            Domain::Volume(4, 3, 5),
            "vec3(float({k} % 5), float({k} % 4), float({k} % 6))",
        ),
    ];
    let mut out = Vec::new();
    for shape in element_shapes() {
        let floats = shape.scalars.contains(&Scalar::Float);
        for (where_, kind, address_ty, extent, address) in &targets {
            for rule in ["sum", "min", "max", "last"] {
                let order_free = rule == "min" || rule == "max" || (rule == "sum" && !floats);
                for emits in [false, true] {
                    if emits && !order_free {
                        continue;
                    }
                    let fan = if emits {
                        format!(
                            "let _fan = loop {{k}} % 3 with j = 0 {{ emit(Pair {{ at: {}, value: {} }}); j + 1 }};",
                            address.replace("{k}", "({k} + j * 7)"),
                            shape.value.replace("{k}", "({k} + j)"),
                        )
                    } else {
                        String::new()
                    };
                    let source = format!(
                        "kernel shape.scatter@1 {{ requires {{ field.read; field.write; }}
    {decls}
    input keys: Rows<Int>; input target: {kind}<{ty}>; output result: {kind}<{ty}>;
    struct Pair {{ at: {address_ty}; value: {ty}; }}
    scatter into target with {rule} {{
        let k = sample(keys);
        {fan}
        return Pair {{ at: {at}, value: {value} }};
    }} }}",
                        decls = shape.decls,
                        ty = shape.ty,
                        at = address.replace("{k}", "k"),
                        value = shape.value.replace("{k}", "k"),
                        fan = fan.replace("{k}", "k"),
                    );
                    // Starts mid-range, so every rule keeps some and
                    // takes some.
                    let start: Vec<u32> = shape
                        .scalars
                        .iter()
                        .enumerate()
                        .map(|(k, scalar)| match scalar {
                            Scalar::Float => 0.25f32.to_bits(),
                            Scalar::Long if k % 2 == 1 => 0,
                            _ => 5,
                        })
                        .collect();
                    let target = (
                        Schema::new(shape.scalars.clone()),
                        *extent,
                        start.repeat(extent.count()),
                    );
                    out.push(Case {
                        name: format!(
                            "{} into {where_} with {rule}{}",
                            shape.name,
                            if emits { ", emitting" } else { "" }
                        ),
                        source,
                        inputs: vec![keys.clone(), target],
                        params: Vec::new(),
                        domain: Domain::Rows(SHAPE_COUNT),
                        context: Context::default(),
                    });
                }
            }
        }
    }
    out
}

/// What an integer sum case must come to on every backend: the output's
/// words, or an overflow at the element named, `(x, y)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Summed {
    /// The output's words.
    Words(Vec<u32>),
    /// An overflow at this element.
    OverflowAt(u32, u32),
}

impl Summed {
    /// Where `outcome` differs from what the case must come to, as text.
    pub fn judge(
        &self,
        outcome: &Result<Vec<u32>, entelechy_rules::ExecutionError>,
    ) -> Option<String> {
        use entelechy_rules::{ArithmeticError, ExecutionError};
        match (self, outcome) {
            (Self::Words(want), Ok(got)) if want == got => None,
            (
                Self::OverflowAt(x, y),
                Err(ExecutionError::Arithmetic {
                    x: gx,
                    y: gy,
                    error: ArithmeticError::Overflow,
                }),
            ) if (x, y) == (gx, gy) => None,
            (want, got) => Some(format!("wanted {want:?}, got {got:?}")),
        }
    }
}

/// Integer sums under a scatter and a prefix: each element's whole
/// total, in any adding order. Overflow is reported at the first element
/// whose total leaves the range; a total that wraps and returns fits.
pub fn sum_cases() -> Vec<(Case, Summed)> {
    const MAX: i32 = i32::MAX;
    const MIN: i32 = i32::MIN;
    let rows = |values: &[i32]| {
        (
            Schema::new(vec![Scalar::Int]),
            Domain::Rows(values.len() as u32),
            values.iter().map(|v| *v as u32).collect::<Vec<u32>>(),
        )
    };
    let words = |values: &[i32]| Summed::Words(values.iter().map(|v| *v as u32).collect());
    let scatter = |name: &str, emits: bool, ats: &[i32], values: &[i32], target: &[i32], summed| {
        // Emitting: half the value emitted, half returned.
        let body = if emits {
            "let i = element(); let v = at(values, i); emit(Pair { at: at(ats, i), value: v / 2 }); return Pair { at: at(ats, i), value: v - v / 2 };"
        } else {
            "let i = element(); return Pair { at: at(ats, i), value: at(values, i) };"
        };
        let source = format!(
            "kernel sums.scatter@1 {{ requires {{ field.read; field.write; }}
    input ats: Rows<Int>; input values: Rows<Int>; input target: Rows<Int>;
    output result: Rows<Int>;
    struct Pair {{ at: Int; value: Int; }}
    scatter into target with sum {{ {body} }} }}"
        );
        (
            Case {
                name: format!("integer sum: {name}"),
                source,
                inputs: vec![rows(ats), rows(values), rows(target)],
                params: Vec::new(),
                domain: Domain::Rows(ats.len() as u32),
                context: context(),
            },
            summed,
        )
    };
    let prefix = |name: &str, values: &[i32], summed| {
        (
            Case {
                name: format!("integer prefix: {name}"),
                source: "kernel sums.prefix@1 { requires { field.read; field.write; }
    input values: Rows<Int>;
    output result: Rows<Int>;
    prefix with sum { return at(values, element()); } }"
                    .to_owned(),
                inputs: vec![rows(values)],
                params: Vec::new(),
                domain: Domain::Rows(values.len() as u32),
                context: context(),
            },
            summed,
        )
    };
    // 48 values of +-2e9 into one element: total zero, and every order
    // wraps on the way.
    let swings: Vec<i32> = (0..48)
        .map(|i| {
            if i % 2 == 0 {
                2_000_000_000
            } else {
                -2_000_000_000
            }
        })
        .collect();
    // A row crossing a device's blocks: `MIN`, then `each` per element.
    // A block's own total wraps; every prefix fits at a million each
    // and overflows at 1.1 million.
    let long = |each: i32| {
        std::iter::once(MIN)
            .chain(std::iter::repeat_n(each, 4095))
            .collect::<Vec<i32>>()
    };
    let prefixes = |values: &[i32]| {
        let mut total = 0i32;
        values
            .iter()
            .map(|v| {
                total += v;
                total
            })
            .collect::<Vec<i32>>()
    };
    // A record of a float (summed in element order) and an integer
    // (summed whole).
    let mixed = |name: &str, values: &[i32], target: &[i32], summed| {
        let record = |ints: &[i32]| {
            (
                Schema::new(vec![Scalar::Float, Scalar::Int]),
                Domain::Rows(ints.len() as u32),
                ints.iter()
                    .flat_map(|v| [0.5f32.to_bits(), *v as u32])
                    .collect::<Vec<u32>>(),
            )
        };
        (
            Case {
                name: format!("integer sum beside a float: {name}"),
                source: "kernel sums.mixed@1 { requires { field.read; field.write; }
    struct Both { f: Float; i: Int; }
    input values: Rows<Int>; input target: Rows<Both>;
    output result: Rows<Both>;
    struct Pair { at: Int; value: Both; }
    scatter into target with sum { let i = element(); return Pair { at: 0, value: Both { f: 0.25, i: at(values, i) } }; } }"
                    .to_owned(),
                inputs: vec![rows(values), record(target)],
                params: Vec::new(),
                domain: Domain::Rows(values.len() as u32),
                context: context(),
            },
            summed,
        )
    };
    let fits = long(1_000_000);
    let passes = long(1_100_000);
    let first_past = (1..passes.len())
        .find(|&k| i64::from(MIN) + 1_100_000 * k as i64 > i64::from(MAX))
        .unwrap() as u32;
    vec![
        scatter(
            "passes the end and comes back",
            false,
            &[0, 1, 0, 1, 0, 2],
            &[MAX, 7, 1, 3, -1, -4],
            &[0, 0, 0, 0],
            words(&[MAX, 10, -4, 0]),
        ),
        scatter(
            "the first element whose total does not fit",
            false,
            &[2, 1, 2, 1],
            &[MAX, MAX, 1, 1],
            &[0, 0, 0, 0],
            Summed::OverflowAt(1, 0),
        ),
        scatter(
            "below the least there is",
            false,
            &[0, 0],
            &[-1, -1],
            &[MIN + 1, 0],
            Summed::OverflowAt(0, 0),
        ),
        scatter(
            "swings each way to nothing",
            false,
            &[3; 48],
            &swings,
            &[5, 0, 0, 0],
            words(&[5, 0, 0, 0]),
        ),
        scatter(
            "emitted, passes the end and comes back",
            true,
            &[0, 1, 0, 1, 0, 2],
            &[MAX, 7, 1, 3, -1, -4],
            &[0, 0, 0, 0],
            words(&[MAX, 10, -4, 0]),
        ),
        mixed(
            "passes the end and comes back",
            &[MAX, 1, -1],
            &[0],
            Summed::Words(vec![1.25f32.to_bits(), MAX as u32]),
        ),
        mixed("does not fit", &[MAX, 1], &[0], Summed::OverflowAt(0, 0)),
        prefix(
            "a block's total passes the end",
            &fits,
            words(&prefixes(&fits)),
        ),
        prefix(
            "the first prefix that does not fit",
            &passes,
            Summed::OverflowAt(first_past, 0),
        ),
        prefix(
            "passes the end at the second",
            &[MAX, 1, -1, 5],
            Summed::OverflowAt(1, 0),
        ),
    ]
}

impl Case {
    /// The words a Host needs to run this case on the interpreter and a
    /// backend beside it: its inputs and outputs, twice over, and never
    /// less than the matrix's usual room.
    pub fn room(&self) -> usize {
        let inputs: usize = self.inputs.iter().map(|(_, _, words)| words.len()).sum();
        (1 << 22).max(4 * (inputs + 4 * self.domain.count()))
    }
}

/// Extents past 2^24, which a 32-bit float cannot hold whole, under every
/// mode: each element, count, size and domain exact, as the interpreter
/// gives them. Each case reads a Field of ones as long as its domain.
pub fn large_cases() -> Vec<Case> {
    let rows = (1u32 << 24) + 1;
    let ints = Schema::new(vec![Scalar::Int]);
    let ones = |domain: Domain| (ints.clone(), domain, vec![1u32; domain.count()]);
    let case = |name: &str, source: &str, inputs, domain| Case {
        name: format!("{name} past 2^24"),
        source: source.to_owned(),
        inputs,
        params: Vec::new(),
        domain,
        context: Context::default(),
    };
    let side = 4097;
    vec![
        case(
            "map over rows",
            "kernel large.map@1 { requires { field.read; field.write; } input ones: Rows<Int>; output result: Rows<Int>;
                compute { return count(ones) + size(ones) - domain() - element() + sample(ones) + at(ones, element() + 1); } }",
            vec![ones(Domain::Rows(rows))],
            Domain::Rows(rows),
        ),
        case(
            "map over a grid",
            "kernel large.grid@1 { requires { field.read; field.write; } input ones: Grid<Int>; output result: Grid<Int>;
                compute { let s = isize(ones); let c = icoord(); return s.x * s.y - count(ones) + c.y * s.x + c.x + sample(ones); } }",
            vec![ones(Domain::Grid(side, side))],
            Domain::Grid(side, side),
        ),
        // A float address far past a wide grid's edge reads its last
        // element: the clamp is an integer's, not a float's, whose
        // `width - 1` rounds up to the width past 2^24.
        {
            let width = (1u32 << 25) + 4096;
            let floats = Schema::new(vec![Scalar::Float]);
            let places: Vec<u32> = (0..width).map(|i| (i as f32).to_bits()).collect();
            case(
                "a float address past a grid's edge",
                "kernel large.edge@1 { requires { field.read; field.write; } input g: Grid<Float>; output result: Rows<Float>;
                    compute { let x = 1.0e10 + float(element()); return at(g, vec2(x, 0.0)) + at_linear(g, vec2(x, 0.5)) + at(g, vec2(float(element()) * 1.0e7, 0.0)); } }",
                vec![(floats, Domain::Grid(width, 1), places)],
                Domain::Rows(8),
            )
        },
        case(
            "reduce",
            "kernel large.reduce@1 { requires { field.read; field.write; } input ones: Rows<Int>; output result: Rows<Int>;
                compute { return element() + sample(ones) + count(ones); } merge(a, b) { return max(a, b); } }",
            vec![ones(Domain::Rows(rows))],
            Domain::Rows(1),
        ),
        case(
            "prefix",
            "kernel large.prefix@1 { requires { field.read; field.write; } input ones: Rows<Int>; output result: Rows<Int>;
                prefix with sum { return sample(ones); } }",
            vec![ones(Domain::Rows(rows))],
            Domain::Rows(rows),
        ),
        case(
            "scatter",
            "kernel large.scatter@1 { requires { field.read; field.write; } input ones: Rows<Int>; input slots: Rows<Int>; output result: Rows<Int>;
                struct Put { slot: Int; value: Int; }
                scatter into slots with max { return Put { slot: element() % 4, value: element() + sample(ones) }; } }",
            vec![ones(Domain::Rows(rows)), ones(Domain::Rows(4))],
            Domain::Rows(rows),
        ),
    ]
}

/// Every case of the matrix whose words a run computes: features, wide,
/// narrow and shaped ones, and integer sums, for [`unrepeated`].
pub fn repeat_cases() -> Vec<Case> {
    cases()
        .into_iter()
        .chain(wide_cases())
        .chain(long_cases())
        .chain(narrow_cases())
        .chain(shape_cases())
        .chain(sum_cases().into_iter().map(|(case, _)| case))
        .collect()
}

/// Where `backend` writes other words, or fails otherwise, on a second run
/// of a case with the same inputs than on the first: a backend that says it
/// repeats exactly ([`entelechy_rules::Backend::repeats_exactly`]) answers none.
/// Each case comes with its module; a case the backend declines or cannot
/// prepare is left to the matrix.
pub fn unrepeated(
    backend: &dyn entelechy_rules::Backend,
    cases: &[(Case, entelechy_ir::Validated)],
) -> Vec<String> {
    use entelechy_abi::{Capability, FieldData, Handle, Host, Invocation, Scope};
    let words = |host: &Host, handle: Handle| -> Result<Vec<u32>, String> {
        let schema = host.schema(handle).map_err(|e| e.to_string())?;
        let count = host.domain(handle).map_err(|e| e.to_string())?.count();
        match host.resident(handle).map_err(|e| e.to_string())? {
            Some(device) => Ok(backend
                .download(device, schema.storage_words(count) * 4)?
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            None => Ok(host
                .data(handle)
                .map_err(|e| e.to_string())?
                .storage()
                .to_vec()),
        }
    };
    let mut failures = Vec::new();
    for (case, module) in cases {
        if backend.declines(module).is_some() {
            continue;
        }
        let Ok(prepared) = backend.prepare(module) else {
            continue;
        };
        let mut host = Host::new(case.room());
        let inputs = case
            .inputs
            .iter()
            .map(|(schema, domain, values)| {
                host.adopt(FieldData::from_values(schema.clone(), *domain, values).unwrap())
                    .unwrap()
            })
            .collect();
        let invocation = Invocation {
            inputs,
            params: case.params.clone(),
            domain: case.domain,
            context: case.context,
        };
        let mut runs: Vec<Result<Vec<Vec<u32>>, String>> = Vec::new();
        for _ in 0..2 {
            let outputs = {
                let mut scope = Scope::new(&mut host, Capability::ALL.into_iter().collect());
                backend.execute(prepared.as_ref(), module, &invocation, &mut scope)
            };
            runs.push(
                outputs
                    .map_err(|error| error.to_string())
                    .and_then(|outputs| outputs.iter().map(|h| words(&host, *h)).collect()),
            );
        }
        if runs[0] != runs[1] {
            failures.push(format!(
                "{}: a second run differs from the first",
                case.name
            ));
        }
    }
    failures
}
