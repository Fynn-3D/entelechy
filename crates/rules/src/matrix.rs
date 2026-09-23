//! Matrix arithmetic, once for every backend: the formulas are written
//! over an [`Arith`], so the interpreter runs them on floats, the JIT
//! emits them as instructions and the GPU prints them as WGSL, all in
//! one order of operations. Matrices are column-major: entry (row `r`,
//! column `c`) of an order `n` matrix is word `c * n + r`.

/// The four operations the formulas need, over some representation of
/// a float.
pub trait Arith {
    /// A float, as this backend holds it.
    type V: Clone;
    /// `a + b`.
    fn add(&mut self, a: Self::V, b: Self::V) -> Self::V;
    /// `a - b`.
    fn sub(&mut self, a: Self::V, b: Self::V) -> Self::V;
    /// `a * b`.
    fn mul(&mut self, a: Self::V, b: Self::V) -> Self::V;
    /// `a / b`.
    fn div(&mut self, a: Self::V, b: Self::V) -> Self::V;
    /// `-a`.
    fn neg(&mut self, a: Self::V) -> Self::V;
}

/// The formulas on plain floats.
pub struct Floats;

impl Arith for Floats {
    type V = f32;
    fn add(&mut self, a: f32, b: f32) -> f32 {
        a + b
    }
    fn sub(&mut self, a: f32, b: f32) -> f32 {
        a - b
    }
    fn mul(&mut self, a: f32, b: f32) -> f32 {
        a * b
    }
    fn div(&mut self, a: f32, b: f32) -> f32 {
        a / b
    }
    fn neg(&mut self, a: f32) -> f32 {
        -a
    }
}

/// Doubles, for the interpreter's matrices of them.
pub struct Doubles;

impl Arith for Doubles {
    type V = f64;
    fn add(&mut self, a: f64, b: f64) -> f64 {
        a + b
    }
    fn sub(&mut self, a: f64, b: f64) -> f64 {
        a - b
    }
    fn mul(&mut self, a: f64, b: f64) -> f64 {
        a * b
    }
    fn div(&mut self, a: f64, b: f64) -> f64 {
        a / b
    }
    fn neg(&mut self, a: f64) -> f64 {
        -a
    }
}

/// The formulas as text: every operation parenthesised, for a shader.
pub struct Text;

impl Arith for Text {
    type V = String;
    fn add(&mut self, a: String, b: String) -> String {
        format!("({a} + {b})")
    }
    fn sub(&mut self, a: String, b: String) -> String {
        format!("({a} - {b})")
    }
    fn mul(&mut self, a: String, b: String) -> String {
        format!("({a} * {b})")
    }
    fn div(&mut self, a: String, b: String) -> String {
        format!("({a} / {b})")
    }
    fn neg(&mut self, a: String) -> String {
        format!("(-{a})")
    }
}

/// The sum of the products `x[k] * y[k]`, first product first.
fn dot<A: Arith>(a: &mut A, pairs: impl Iterator<Item = (A::V, A::V)>) -> A::V {
    let mut sum = None;
    for (x, y) in pairs {
        let product = a.mul(x, y);
        sum = Some(match sum {
            None => product,
            Some(sum) => a.add(sum, product),
        });
    }
    sum.expect("a matrix has an order of at least one")
}

/// `x * y` for two matrices of order `n`.
pub fn mul_mat_mat<A: Arith>(a: &mut A, n: usize, x: &[A::V], y: &[A::V]) -> Vec<A::V> {
    let mut out = Vec::with_capacity(n * n);
    for c in 0..n {
        for r in 0..n {
            out.push(dot(
                a,
                (0..n).map(|k| (x[k * n + r].clone(), y[c * n + k].clone())),
            ));
        }
    }
    out
}

/// `x * v` for a matrix of order `n` and a column vector.
pub fn mul_mat_vec<A: Arith>(a: &mut A, n: usize, x: &[A::V], v: &[A::V]) -> Vec<A::V> {
    (0..n)
        .map(|r| dot(a, (0..n).map(|c| (x[c * n + r].clone(), v[c].clone()))))
        .collect()
}

/// `v * x` for a row vector and a matrix of order `n`.
pub fn mul_vec_mat<A: Arith>(a: &mut A, n: usize, v: &[A::V], x: &[A::V]) -> Vec<A::V> {
    (0..n)
        .map(|c| dot(a, (0..n).map(|r| (v[r].clone(), x[c * n + r].clone()))))
        .collect()
}

/// The matrix with rows and columns exchanged.
pub fn transpose<V: Clone>(n: usize, x: &[V]) -> Vec<V> {
    let mut out = Vec::with_capacity(n * n);
    for c in 0..n {
        for r in 0..n {
            out.push(x[r * n + c].clone());
        }
    }
    out
}

/// The matrix without row `row` and column `col`.
fn minor<V: Clone>(n: usize, x: &[V], row: usize, col: usize) -> Vec<V> {
    let mut out = Vec::with_capacity((n - 1) * (n - 1));
    for c in 0..n {
        if c == col {
            continue;
        }
        for r in 0..n {
            if r != row {
                out.push(x[c * n + r].clone());
            }
        }
    }
    out
}

/// The determinant of a matrix of order `n`, expanded along the first
/// row.
pub fn determinant<A: Arith>(a: &mut A, n: usize, x: &[A::V]) -> A::V {
    if n == 1 {
        return x[0].clone();
    }
    let mut total = None;
    for c in 0..n {
        let sub = minor(n, x, 0, c);
        let d = determinant(a, n - 1, &sub);
        let term = a.mul(x[c * n].clone(), d);
        total = Some(match (total, c % 2) {
            (None, _) => term,
            (Some(total), 0) => a.add(total, term),
            (Some(total), _) => a.sub(total, term),
        });
    }
    total.expect("a matrix has an order of at least one")
}

/// The inverse of a matrix of order `n` (two or more): the adjugate over
/// the determinant. A singular matrix divides by zero.
pub fn inverse<A: Arith>(a: &mut A, n: usize, x: &[A::V]) -> Vec<A::V> {
    let det = determinant(a, n, x);
    let mut out = Vec::with_capacity(n * n);
    for c in 0..n {
        for r in 0..n {
            // Entry (r, c) of the inverse is cofactor (c, r) over the
            // determinant.
            let sub = minor(n, x, c, r);
            let d = determinant(a, n - 1, &sub);
            let cofactor = if (r + c) % 2 == 0 { d } else { a.neg(d) };
            out.push(a.div(cofactor, det.clone()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_transposes_determinants_and_inverses_agree_with_hand_values() {
        let mut f = Floats;
        // Column-major: columns (1, 2), (3, 4) is the matrix [[1, 3], [2, 4]].
        let m = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mul_mat_vec(&mut f, 2, &m, &[1.0, 1.0]), vec![4.0, 6.0]);
        assert_eq!(mul_vec_mat(&mut f, 2, &[1.0, 1.0], &m), vec![3.0, 7.0]);
        assert_eq!(mul_mat_mat(&mut f, 2, &m, &m), vec![7.0, 10.0, 15.0, 22.0]);
        assert_eq!(transpose(2, &m), vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(determinant(&mut f, 2, &m), -2.0);
        assert_eq!(inverse(&mut f, 2, &m), vec![-2.0, 1.0, 1.5, -0.5]);
        let r = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(determinant(&mut f, 3, &r), 1.0);
        assert_eq!(inverse(&mut f, 3, &r), transpose(3, &r));
        let s = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        assert_eq!(determinant(&mut f, 4, &s), 24.0);
        let inv = inverse(&mut f, 4, &s);
        let identity = mul_mat_mat(&mut f, 4, &inv, &s);
        for (k, v) in identity.iter().enumerate() {
            let expected = if k % 5 == 0 { 1.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-6, "{identity:?}");
        }
        let mut t = Text;
        assert_eq!(
            determinant(
                &mut t,
                2,
                &["a".to_string(), "b".into(), "c".into(), "d".into()]
            ),
            "((a * d) - (c * b))"
        );
    }
}
