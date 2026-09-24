//! Random Kernels for differential testing: typed expressions over the
//! matrix's environment, wrapped in every execution mode, so a backend is
//! held to the interpreter on programs nobody wrote by hand. The same seed
//! gives the same programs, so a difference found is a difference kept.
//!
//! What the expressions reach is the language as the matrix's features
//! use it: float, integer, boolean and vector arithmetic, comparisons,
//! the math builtins, conversions, `if`, `loop` with a carried value, and
//! `let`s. Integer arithmetic may overflow or divide by zero: a program
//! that fails must fail alike, at the same element, on every backend.

use crate::{Case, case_in, modes};

/// A small deterministic generator (SplitMix64): no dependency, and the
/// same numbers on every machine.
pub struct Rng(u64);

impl Rng {
    /// A generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.bits() % n.max(1)
    }

    /// One of `items`.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    F,
    I,
    B,
    V,
}

/// The names in scope, by type, and whether integers and booleans may
/// be made from floats.
struct Scope {
    names: Vec<(String, Ty)>,
    /// Integers and booleans from integers and booleans alone: what a
    /// device promises exactly, however its floats round.
    exact: bool,
}

impl Scope {
    fn of(&self, ty: Ty) -> Vec<&str> {
        self.names
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

/// Writes one expression of `ty`, at most `depth` deep.
fn expr(rng: &mut Rng, scope: &Scope, ty: Ty, depth: u32) -> String {
    if depth == 0 || rng.below(4) == 0 {
        return leaf(rng, scope, ty);
    }
    let d = depth - 1;
    let e = |rng: &mut Rng, ty: Ty| expr(rng, scope, ty, d);
    match ty {
        Ty::F => match rng.below(22) {
            0 => format!("({} + {})", e(rng, Ty::F), e(rng, Ty::F)),
            1 => format!("({} - {})", e(rng, Ty::F), e(rng, Ty::F)),
            2 => format!("({} * {})", e(rng, Ty::F), e(rng, Ty::F)),
            3 => format!("({} / {})", e(rng, Ty::F), e(rng, Ty::F)),
            4 => format!("({} % {})", e(rng, Ty::F), e(rng, Ty::F)),
            5 => format!(
                "{}({})",
                rng.pick(&[
                    "sin", "cos", "abs", "floor", "ceil", "fract", "round", "sign", "tanh", "exp2"
                ]),
                e(rng, Ty::F)
            ),
            6 => format!("sqrt(abs({}))", e(rng, Ty::F)),
            7 => format!(
                "{}({}, {})",
                rng.pick(&["min", "max", "step", "atan2", "pow"]),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            8 => format!(
                "clamp({}, {}, {})",
                e(rng, Ty::F),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            9 => format!(
                "smoothstep({}, {}, {})",
                e(rng, Ty::F),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            10 => format!(
                "mix({}, {}, {})",
                e(rng, Ty::F),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            11 => format!("float({})", e(rng, Ty::I)),
            12 => format!("dot({}, {})", e(rng, Ty::V), e(rng, Ty::V)),
            13 => format!("length({})", e(rng, Ty::V)),
            14 => format!("{}.{}", paren(e(rng, Ty::V)), rng.pick(&["x", "y", "z"])),
            15 => format!(
                "if {} {{ {} }} else {{ {} }}",
                e(rng, Ty::B),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            16 => format!("(0.0 - {})", e(rng, Ty::F)),
            17 => format!(
                "loop {} with acc = {} {{ acc * 0.5 + {} + float(index()) }}",
                1 + rng.below(4),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            18 => format!("log(abs({}) + 1.0)", e(rng, Ty::F)),
            19 => format!(
                "fma({}, {}, {})",
                e(rng, Ty::F),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            _ => format!("({} * 0.5 + {})", e(rng, Ty::F), e(rng, Ty::F)),
        },
        Ty::I => match rng.below(18) {
            0 => format!("({} + {})", e(rng, Ty::I), e(rng, Ty::I)),
            1 => format!("({} - {})", e(rng, Ty::I), e(rng, Ty::I)),
            2 => format!("({} * {})", e(rng, Ty::I), e(rng, Ty::I)),
            3 => format!("({} / {})", e(rng, Ty::I), e(rng, Ty::I)),
            4 => format!("({} % {})", e(rng, Ty::I), e(rng, Ty::I)),
            5 => format!(
                "({} {} {})",
                e(rng, Ty::I),
                rng.pick(&["&", "|", "^"]),
                e(rng, Ty::I)
            ),
            6 => format!(
                "({} {} {})",
                e(rng, Ty::I),
                rng.pick(&["<<", ">>", ">>>"]),
                rng.below(34)
            ),
            7 => format!(
                "{}({}, {})",
                rng.pick(&["add_wrap", "sub_wrap", "mul_wrap"]),
                e(rng, Ty::I),
                e(rng, Ty::I)
            ),
            8 => format!(
                "{}({}, {})",
                rng.pick(&["min", "max"]),
                e(rng, Ty::I),
                e(rng, Ty::I)
            ),
            9 => format!("abs({})", e(rng, Ty::I)),
            10 if !scope.exact => format!("int({})", e(rng, Ty::F)),
            11 => format!("int({})", e(rng, Ty::B)),
            12 => format!(
                "if {} {{ {} }} else {{ {} }}",
                e(rng, Ty::B),
                e(rng, Ty::I),
                e(rng, Ty::I)
            ),
            13 => format!("clamp({}, -100, 100)", e(rng, Ty::I)),
            14 => format!(
                "{}({})",
                rng.pick(&["popcount", "clz", "ctz"]),
                e(rng, Ty::I)
            ),
            15 => format!(
                "loop {} with acc = {} {{ add_wrap(acc, {} + index()) }}",
                1 + rng.below(4),
                e(rng, Ty::I),
                e(rng, Ty::I)
            ),
            16 => format!("(0 - {})", e(rng, Ty::I)),
            _ => format!(
                "({} {} {})",
                e(rng, Ty::I),
                rng.pick(&["&", "^"]),
                rng.below(256)
            ),
        },
        Ty::B => match rng.below(8) {
            0 if !scope.exact => format!(
                "({} {} {})",
                e(rng, Ty::F),
                rng.pick(&["<", "<=", ">", ">=", "==", "!="]),
                e(rng, Ty::F)
            ),
            1 => format!(
                "({} {} {})",
                e(rng, Ty::I),
                rng.pick(&["<", "<=", ">", ">=", "==", "!="]),
                e(rng, Ty::I)
            ),
            2 => format!("({} && {})", e(rng, Ty::B), e(rng, Ty::B)),
            3 => format!("({} || {})", e(rng, Ty::B), e(rng, Ty::B)),
            4 => format!("!{}", paren(e(rng, Ty::B))),
            5 => format!("({} == {})", e(rng, Ty::B), e(rng, Ty::B)),
            6 if !scope.exact => format!("({} != {})", e(rng, Ty::V), e(rng, Ty::V)),
            _ => format!(
                "if {} {{ {} }} else {{ {} }}",
                e(rng, Ty::B),
                e(rng, Ty::B),
                e(rng, Ty::B)
            ),
        },
        Ty::V => match rng.below(10) {
            0 => format!(
                "vec3({}, {}, {})",
                e(rng, Ty::F),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
            1 => format!("({} + {})", e(rng, Ty::V), e(rng, Ty::V)),
            2 => format!("({} - {})", e(rng, Ty::V), e(rng, Ty::V)),
            3 => format!("({} * {})", e(rng, Ty::V), e(rng, Ty::F)),
            4 => format!("cross({}, {})", e(rng, Ty::V), e(rng, Ty::V)),
            5 => format!("normalize({})", e(rng, Ty::V)),
            6 => format!(
                "mix({}, {}, {})",
                e(rng, Ty::V),
                e(rng, Ty::V),
                e(rng, Ty::F)
            ),
            7 => format!(
                "{}({})",
                rng.pick(&["abs", "floor", "fract"]),
                e(rng, Ty::V)
            ),
            8 => format!(
                "if {} {{ {} }} else {{ {} }}",
                e(rng, Ty::B),
                e(rng, Ty::V),
                e(rng, Ty::V)
            ),
            _ => format!(
                "clamp({}, {}, {})",
                e(rng, Ty::V),
                e(rng, Ty::F),
                e(rng, Ty::F)
            ),
        },
    }
}

fn paren(text: String) -> String {
    format!("({text})")
}

/// A name in scope, or a literal.
fn leaf(rng: &mut Rng, scope: &Scope, ty: Ty) -> String {
    let names = scope.of(ty);
    if !names.is_empty() && rng.below(3) != 0 {
        return (*rng.pick(&names)).to_owned();
    }
    match ty {
        Ty::F => rng
            .pick(&[
                "0.0", "1.0", "0.5", "2.5", "-1.25", "0.001", "1000.0", "3.14159", "-0.0", "0.1",
            ])
            .to_string(),
        // A written number meets its type where it is used, and two meeting
        // alone are exact, a Long past an Int's range: meeting `i` makes
        // it an Int.
        Ty::I => format!(
            "({} + 0 * i)",
            rng.pick(&[
                "0",
                "1",
                "-1",
                "2",
                "7",
                "255",
                "-100",
                "65536",
                "2147483647",
                "-2147483647"
            ])
        ),
        Ty::B => rng.pick(&["true", "false"]).to_string(),
        Ty::V => rng
            .pick(&[
                "vec3(0.0, 0.0, 0.0)",
                "vec3(1.0, 2.0, 3.0)",
                "vec3(-0.5, 0.25, 4.0)",
            ])
            .to_string(),
    }
}

/// `count` random programs from `seed`, each under a mode it picks: a
/// few `let`s and the `Out` they give, wrapped as the matrix wraps its
/// features, with the inputs, params and context that mode runs with.
pub fn cases(seed: u64, count: usize) -> Vec<Case> {
    generate(seed, count, false)
}

/// Programs whose integers and booleans are made from integers and
/// booleans alone, so their `i` and `b` must match exactly on a device
/// whose floats only match to rounding.
pub fn exact_cases(seed: u64, count: usize) -> Vec<Case> {
    generate(seed, count, true)
}

fn generate(seed: u64, count: usize, exact: bool) -> Vec<Case> {
    let mut rng = Rng::new(seed);
    let modes = modes();
    (0..count)
        .map(|n| {
            // The environment every mode's wrapper gives, and the params.
            let mut scope = Scope {
                names: vec![
                    ("i".into(), Ty::I),
                    ("x".into(), Ty::F),
                    ("sampled".into(), Ty::F),
                    ("k".into(), Ty::F),
                    ("n".into(), Ty::I),
                    ("flag".into(), Ty::B),
                    ("v".into(), Ty::V),
                    ("time()".into(), Ty::F),
                    ("pointer().x".into(), Ty::F),
                ],
                exact,
            };
            let mut lets = String::new();
            for l in 0..rng.below(5) {
                let ty = *rng.pick(&[Ty::F, Ty::I, Ty::B, Ty::V]);
                let value = expr(&mut rng, &scope, ty, 3);
                let name = format!("r{l}");
                lets.push_str(&format!("let {name} = {value}; "));
                scope.names.push((name, ty));
            }
            let out = format!(
                "Out {{ f: {}, i: {}, b: {}, v: {} }}",
                expr(&mut rng, &scope, Ty::F, 4),
                expr(&mut rng, &scope, Ty::I, 4),
                expr(&mut rng, &scope, Ty::B, 3),
                expr(&mut rng, &scope, Ty::V, 3),
            );
            let (mode, wrap) = rng.pick(&modes);
            case_in(&format!("random {seed}:{n}"), mode, wrap(&lets, &out))
        })
        .collect()
}
