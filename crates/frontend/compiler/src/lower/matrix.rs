//! The determinant and the inverse, as scalar formulas over an
//! [`Arith`]: the lowering writes them into the IR one operation at a
//! time, so every backend computes them in one order of operations.
//! Matrices are column-major: entry (row `r`, column `c`) of an order
//! `n` matrix is entry `c * n + r`.

/// The operations the formulas need, over some representation of a
/// scalar.
pub(super) trait Arith {
    /// A scalar, as the formulas are written over it.
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
pub(super) fn determinant<A: Arith>(a: &mut A, n: usize, x: &[A::V]) -> A::V {
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
pub(super) fn inverse<A: Arith>(a: &mut A, n: usize, x: &[A::V]) -> Vec<A::V> {
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

    /// The formulas on plain floats.
    struct Floats;

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

    #[test]
    fn determinants_and_inverses_agree_with_hand_values() {
        let mut f = Floats;
        // Column-major: columns (1, 2), (3, 4) is the matrix [[1, 3], [2, 4]].
        let m = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(determinant(&mut f, 2, &m), -2.0);
        assert_eq!(inverse(&mut f, 2, &m), vec![-2.0, 1.0, 1.5, -0.5]);
        // A rotation's inverse is its transpose.
        let r = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(determinant(&mut f, 3, &r), 1.0);
        assert_eq!(
            inverse(&mut f, 3, &r),
            vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        let s = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        assert_eq!(determinant(&mut f, 4, &s), 24.0);
        let inv = inverse(&mut f, 4, &s);
        assert_eq!(&inv[..3], &[0.5, 0.0, 0.0]);
        assert_eq!(&inv[12..], &[-0.5, -2.0 / 3.0, -0.75, 1.0]);
    }
}
