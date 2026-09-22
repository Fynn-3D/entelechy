#![warn(missing_docs)]
//! The feature matrix: every feature of the Kernel language under every
//! execution mode, as complete Kernels with their inputs, params and
//! context, so a backend can be held to the interpreter across the
//! whole cross product rather than where an example happens to look.
//!
//! The rows fall in two parts. The features are what the language can
//! write, one row a form, which holds a backend to the interpreter on
//! the words it computes. The semantics are what the language promises
//! about what it computes: what is not taken is not computed, a read
//! past an edge is the edge, a quotient truncates toward zero, and a
//! float says what it says where no number answers. Those rows carry
//! work that would fail if it ran, so a run that finishes is the proof;
//! the interpreter and the JIT report such a failure, which is what
//! makes the row bite, and the GPU agrees on the words either way.
//!
//! A feature is a snippet that computes an `Out` from an environment
//! every mode's wrapper provides: `i: Int`, `x: Float`, the inputs
//! `rows`, `ints` and `cells`, `sampled` (the element's own sample of an
//! input of the output's shape), the params `k`, `n`, `flag` and `v`,
//! and three functions. A mode's wrapper puts the snippet in a map over
//! rows, a map over a grid, a reduction, a scan with a memory, and a
//! scatter, and returns what that mode returns.

use orla_abi::{Context, Domain, InputState, Scalar, Schema, Value};

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
            "if chain",
            "let g = if x < 1.0 { 1 } else if x < 3.0 { let mid = i * 2; mid } else { 3 };",
            "Out { f: if g == 1 { x } else { -x }, i: g, b: g == 3, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // What is not taken is not computed: every division here
            // would fail if the side holding it ran, so a run that
            // finishes is the proof. The GPU reports no failure, but
            // the interpreter and the JIT do, and all three must agree
            // on the words.
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
            // A part of an element is read on its own: the words a
            // field or an index names and no others, wherever they
            // begin. An index past the end clamps as an array's does,
            // and the element the address names clamps as a Field's.
            "parts of an element",
            "let patch = at(parts, i % 8); let p = patch.points[i % 4] + patch.points[i - 9] + patch.points[i + 9]; let deep = patch.links[i % 3][i % 2] + patch.links[i + 40][1] + patch.links[0 - i][0]; let far = at(parts, i + 100).label + int(at(parts, 0 - i).points[2].z * 10.0); let corner = at(parts, i % 8).points[0].y; let here = near.points[i % 4] + near.points[i + 9]; let own = near.links[i % 3][i % 2] + near.label;",
            "Out { f: p.x + p.z + corner + here.y, i: deep + far + patch.label + own, b: p.y > 0.0, v: p + here }",
        ),
        (
            // What a float does where no number answers: an infinity
            // where a positive is divided by nothing, a not-a-number
            // where nothing is, and nothing equals or orders against a
            // not-a-number. The words themselves are not compared,
            // since a payload is a backend's own business; what it says
            // about them is. `sqrt` of a negative is left out: the CPU
            // and the JIT have NaN there and a driver's fast math need
            // not, which the caveat in the documents names.
            "float edges",
            "let zero = x - x; let up = 1.0 / zero; let down = (0.0 - 1.0) / zero; let undefined = zero / zero; let same = undefined == undefined; let ordered = undefined < 1.0 || undefined > 1.0; let big = up > 1000000000.0;",
            "Out { f: 0.0, i: int(isinf(up)) + int(isinf(down)) * 2 + int(isnan(undefined)) * 4 + int(same) * 8 + int(ordered) * 16 + int(big) * 32 + int(up > down) * 64, b: isnan(undefined), v: vec3(0.0, 0.0, 0.0) }",
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
            // `&&` and `||` ask the other side only where the first
            // settles nothing, which is why the division here is safe.
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
            // A `break` stops its own loop and gives its value: every
            // shape of it, and the loops around it untouched.
            "break",
            "let first = loop 20 with best = -1 { let c = i + index(); if c % 7 == 0 { break c; } best }; let armed = loop 8 with acc = 0 { match index() { 0 | 1 => acc + 1, 5 => { break acc * 100; }, _ => acc + 2 } }; let nested = loop 4 with outer = 0 { let inner = loop 4 with a = 0 { if a > 2 { break a; } a + 1 }; if outer > 5 { break outer; } outer + inner }; let tested = loop 12 with a = x until a > 3.0 { if a > 2.0 { break a * 10.0; } a + 0.7 }; let counted = loop n with a = 0 { if index() > i % 5 { break a * 2; } a + index() }; let never = loop 5 with a = 1 { if i < 0 { break 99; } a * 2 };",
            "Out { f: tested, i: first + armed + nested + counted + never, b: tested > 20.0, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            // One function written once, lowered for each type a call
            // gives it.
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
            "let t = time() + rate() + window().x + window().y + scale() + pointer().x + pointer().y + wheel().x + wheel().y; let keys = int(key(65)) + int(key(37)) * 2 + int(pointer_down()) * 4 + int(button(1)) * 8 + int(button(2)) * 16;",
            "Out { f: t, i: keys, b: key(32), v: vec3(window().x, scale(), 0.0) }",
        ),
        (
            "params",
            "let scaled = v * k;",
            "Out { f: k * x + v.y, i: n + int(flag), b: flag && n > 3, v: scaled }",
        ),
        (
            "more math",
            // No NaN is made here: the GPU's driver may fold `inf * 0`
            // under fast math where the CPU has NaN; a NaN that arrives
            // in a Field is seen alike (the address tests cover that).
            // `mod` by 0.71, not 0.7: at an exact multiple the remainder
            // is a knife-edge (0 or nearly the divisor) that the GPU's
            // contracted multiply-subtract lands on the other side of.
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
            "let lut = [3, 1, 4, 1, 5, 9, 2, 6]; let fl = [x, x * 2.0, 3.0]; let reps = [i; 4]; let grid = [[1, 2], [3, 4], [5, 6]]; let cells = [Cell { a: x, b: i, on: true }, Cell { a: 1.0, b: 2, on: false }]; let t = Table { rows: [i, i + 1, i + 2, i + 3], scale: 0.5 }; let rolled = roll(t.rows, i % 4); let vs = [v, vec3(x, 0.0, 1.0)]; let built = [each 5 { let k = index() * 2; k + i }]; let squares = [each 3 { loop 4 with s = 0 { s + index() * built[index()] } }]; let picked = lut[i % 8] + lut[2] + lut[-i] + lut[i + 100] + reps[i % 3] + grid[i % 3][(i + 1) % 2] + grid[2][0] + cells[i % 2].b + rolled[0] + rolled[i % 4] + t.rows[3] + built[i % 5] + squares[i % 3];",
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
            // constant: Metal narrowed the dividend to sixteen bits and
            // divided wrongly until the GPU backend guarded its dividends.
            "let word = add_wrap(mul_wrap(i, 0x10001), 0x12345); let hi = word >>> 16; let hv = ivec2(word, word * 3) >>> 16; let narrow = hi % 7 + hi / 7 + (hi % 6) * 10 + urem(hi, 7) + udiv(hi, 3) + ((word >> 16) & 0xFFFF) % 7 + (hv % 7).x + (hv / 7).y;",
            "Out { f: 0.0, i: narrow, b: hi % 7 == 0, v: vec3(0.0, 0.0, 0.0) }",
        ),
        (
            "swizzles and joins",
            "let p = vec4(v, x); let q = vec4(vec2(x, 2.0), 3, 4.0); let s = vec3(i, vec2(0.5, 0.25)); let w = p.wzyx + q.xyzw + vec4(s.zz, s.xy) + vec4(p.rgb, p.a); let iv = ivec3(ivec2(i, 2), 3).zx + ivec3(1, ivec2(i, i)).yz; let m2 = mat2(w.xy, w.zw); let t = m2 * v.xy + p.ba;",
            "Out { f: w.x + w.w + t.x + t.y + s.y, i: iv.x * 10 + iv.y, b: p.xyz == v && q.gb == vec2(2.0, 3.0), v: w.yzw + vec3(s.xy, 1.0) + p.rrr }",
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
    ]
}

const HEAD: &str = "requires { field.read; field.write; input.read; time.read; }
    struct Cell { a: Float; b: Int; on: Bool; }
    struct Out { f: Float; i: Int; b: Bool; v: Vec3; }
    param k: Float = 0.3;
    param n: Int = 7;
    param flag: Bool = true;
    param v: Vec3 = vec3(1.0, 2.0, 3.0);
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

fn scatter(lets: &str, out: &str) -> String {
    format!(
        "kernel matrix.scatter@1 {{ {HEAD}
    input rows: Rows<Float>; input ints: Rows<Int>; input cells: Grid<Cell>; input parts: Rows<Patch>; input patches: Grid<Patch>; input target: Rows<Out>;
    output result: Rows<Out>;
    struct Pair {{ at: Int; value: Out; }}
    scatter into target with last {{ let i = element(); {ENV} {lets} let out = {out}; return Pair {{ at: (i * 7) % 16, value: out }}; }} }}"
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

/// The same element over a grid, which a sample reads at its own place.
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
    ]
}

/// The context every case runs under: a time, a rate, a window, the
/// pointer with two buttons held, a wheel, and three keys down.
pub fn context() -> Context {
    let mut input = InputState::default();
    input.pointer = [12.5, 7.25];
    input.pointer_down = true;
    input.pointer_middle = true;
    input.wheel = [1.0, -2.0];
    input.set_key(65, true);
    input.set_key(37, true);
    input.set_key(32, true);
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
            let source = wrap(lets, result);
            let (inputs, domain) = match mode {
                "map over rows" => (
                    vec![rows(), ints(), cells(), parts_input(), patches_input()],
                    Domain::Rows(COUNT),
                ),
                "map over a grid" => (
                    vec![rows(), ints(), cells(), parts_input(), patches_input()],
                    Domain::Grid(16, 3),
                ),
                "reduce" => (
                    vec![rows(), ints(), cells(), parts_input(), patches_input()],
                    Domain::Rows(1),
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
            out.push(Case {
                name: format!("{feature} in {mode}"),
                source,
                inputs,
                params: params(),
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
    for (index, (a, b)) in reference.iter().zip(candidate).enumerate() {
        let scalar = scalars[index % stride];
        let same = match scalar {
            Scalar::Float => {
                let (fa, fb) = (f32::from_bits(*a), f32::from_bits(*b));
                (fa.is_nan() && fb.is_nan())
                    || (fa - fb).abs() <= absolute + relative * fa.abs().max(fb.abs())
            }
            _ => a == b,
        };
        if !same {
            let shown = |w: u32| match scalar {
                Scalar::Float => format!("{}", f32::from_bits(w)),
                Scalar::Int => format!("{}", w as i32),
                Scalar::Bool => format!("{}", w != 0),
            };
            return Some(format!(
                "element {} word {} ({scalar:?}): {} against {}",
                index / stride,
                index % stride,
                shown(*a),
                shown(*b)
            ));
        }
    }
    None
}
