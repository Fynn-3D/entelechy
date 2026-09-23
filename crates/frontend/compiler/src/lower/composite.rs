//! The builtins that are formulas over the arithmetic, written into the
//! IR as that arithmetic: products of matrices, `dot`, `cross`,
//! `length`, `normalize`, `reflect`, `transpose`, `determinant` and
//! `inverse`. The IR then holds only operations every backend has, and
//! each of these is one formula, in one order of operations, wherever
//! it runs.

use orla_ir::{BinaryOp, Block as IrBlock, Intrinsic, Op, Type, ValueId};

use super::Builder;
use super::matrix::{self, Arith};

impl Builder<'_> {
    /// Part `k` of a value: a vector's component or a matrix's column.
    fn part(&mut self, value: ValueId, k: u8) -> ValueId {
        let ty = self.ty(value);
        let part = ty
            .column()
            .or_else(|| ty.lanes().map(|(_, lane)| lane))
            .expect("a vector or a matrix");
        self.emit(part, Op::Extract(value, u32::from(k)))
    }

    fn binary(&mut self, ty: Type, op: BinaryOp, a: ValueId, b: ValueId) -> ValueId {
        self.emit(ty, Op::Binary(op, a, b))
    }

    /// `a op b`, the result the first operand's type.
    fn same(&mut self, op: BinaryOp, a: ValueId, b: ValueId) -> ValueId {
        let ty = self.ty(a);
        self.binary(ty, op, a, b)
    }

    /// The sum of a vector's components, the first first.
    fn sum_of(&mut self, vector: ValueId) -> ValueId {
        let (n, _) = self.ty(vector).lanes().expect("a vector");
        let mut sum = self.part(vector, 0);
        for k in 1..n {
            let next = self.part(vector, k);
            sum = self.same(BinaryOp::Add, sum, next);
        }
        sum
    }

    /// `dot(a, b)`: the componentwise product, summed first to last.
    pub(super) fn dot(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let product = self.same(BinaryOp::Mul, a, b);
        self.sum_of(product)
    }

    /// `m * v`: each column scaled by its component of `v`, added first
    /// to last, which is every row's sum of products in that order.
    fn matrix_vector(&mut self, m: ValueId, v: ValueId) -> ValueId {
        let n = self.ty(m).order().expect("a matrix");
        let mut sum = None;
        for c in 0..n {
            let column = self.part(m, c);
            let scale = self.part(v, c);
            let term = self.same(BinaryOp::Mul, column, scale);
            sum = Some(match sum {
                None => term,
                Some(sum) => self.same(BinaryOp::Add, sum, term),
            });
        }
        sum.expect("a matrix has an order of at least two")
    }

    /// The product of two operands when one is a matrix and it is a
    /// product of matrices: `m * m`, `m * v` (a column) or `v * m` (a
    /// row). `None` for any other pair, a scale by a scalar included.
    pub(super) fn matrix_product(&mut self, a: ValueId, b: ValueId) -> Option<ValueId> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        match (ta.order(), tb.order()) {
            (Some(n), Some(_)) if ta == tb => {
                let columns = (0..n)
                    .map(|c| {
                        let column = self.part(b, c);
                        self.matrix_vector(a, column)
                    })
                    .collect();
                Some(self.emit(ta, Op::Construct(columns)))
            }
            (Some(_), None) if ta.column() == Some(tb) => Some(self.matrix_vector(a, b)),
            (None, Some(n)) if tb.column() == Some(ta) => {
                let parts = (0..n)
                    .map(|c| {
                        let column = self.part(b, c);
                        self.dot(a, column)
                    })
                    .collect();
                Some(self.emit(ta, Op::Construct(parts)))
            }
            _ => None,
        }
    }

    /// A builtin that is a formula, written as one; `None` when `name` is
    /// not one of them or the arguments do not fit it, which the caller
    /// reports.
    pub(super) fn formula(&mut self, name: &str, ids: &[ValueId]) -> Option<ValueId> {
        let types: Vec<Type> = ids.iter().map(|id| self.ty(*id)).collect();
        // A vector of a float kind, of either width.
        let vector = |ty: Type| ty.lanes().is_some() && ty.is_float_kind();
        Some(match (name, types.as_slice()) {
            ("dot", [a, b]) if a == b && vector(*a) => self.dot(ids[0], ids[1]),
            ("length", [a]) if vector(*a) => self.length(ids[0]),
            ("normalize", [a]) if vector(*a) => self.normalize(ids[0]),
            ("reflect", [a, b]) if a == b && vector(*a) => {
                // incident - 2 * dot(normal, incident) * normal
                let (incident, normal) = (ids[0], ids[1]);
                let d = self.dot(normal, incident);
                let kind = self.ty(d);
                let two = self.number(kind, 2.0);
                let scale = self.same(BinaryOp::Mul, two, d);
                let along = self.same(BinaryOp::Mul, normal, scale);
                self.same(BinaryOp::Sub, incident, along)
            }
            ("cross", [a, b]) if a == b && matches!(*a, Type::Vec3 | Type::DVec3) => {
                // a.yzx * b.zxy - a.zxy * b.yzx
                let (a, b) = (ids[0], ids[1]);
                let turned = |this: &mut Self, v: ValueId, order: [u8; 3]| {
                    let parts = order.iter().map(|k| this.part(v, *k)).collect();
                    let ty = this.ty(v);
                    this.emit(ty, Op::Construct(parts))
                };
                let (a1, b1) = (turned(self, a, [1, 2, 0]), turned(self, b, [2, 0, 1]));
                let (a2, b2) = (turned(self, a, [2, 0, 1]), turned(self, b, [1, 2, 0]));
                let left = self.same(BinaryOp::Mul, a1, b1);
                let right = self.same(BinaryOp::Mul, a2, b2);
                self.same(BinaryOp::Sub, left, right)
            }
            ("transpose", [m]) if m.order().is_some() => {
                let n = m.order().unwrap_or(0);
                let column = m.column().expect("a matrix");
                let columns = (0..n)
                    .map(|c| {
                        let parts = (0..n)
                            .map(|k| {
                                let source = self.part(ids[0], k);
                                self.part(source, c)
                            })
                            .collect();
                        self.emit(column, Op::Construct(parts))
                    })
                    .collect();
                self.emit(*m, Op::Construct(columns))
            }
            ("determinant", [m]) if m.order().is_some() => {
                let (n, entries) = self.entries(ids[0]);
                let kind = m.scalar().expect("a matrix");
                matrix::determinant(
                    &mut Scalars {
                        builder: self,
                        kind,
                    },
                    n,
                    &entries,
                )
            }
            ("inverse", [m]) if m.order().is_some() => {
                let (n, entries) = self.entries(ids[0]);
                let kind = m.scalar().expect("a matrix");
                let inverse = matrix::inverse(
                    &mut Scalars {
                        builder: self,
                        kind,
                    },
                    n,
                    &entries,
                );
                let column = m.column().expect("a matrix");
                let columns = inverse
                    .chunks(n)
                    .map(|parts| self.emit(column, Op::Construct(parts.to_vec())))
                    .collect();
                self.emit(*m, Op::Construct(columns))
            }
            _ => return None,
        })
    }

    /// `sqrt(dot(v, v))`.
    pub(super) fn length(&mut self, v: ValueId) -> ValueId {
        let squared = self.dot(v, v);
        let kind = self.ty(squared);
        self.emit(kind, Op::Intrinsic(Intrinsic::Sqrt, vec![squared]))
    }

    /// `v / length(v)`, and the zero vector for the zero vector.
    fn normalize(&mut self, v: ValueId) -> ValueId {
        let ty = self.ty(v);
        let (n, kind) = ty.lanes().expect("a vector");
        let length = self.length(v);
        let zero = self.number(kind, 0.0);
        let positive = self.binary(Type::Bool, BinaryOp::Greater, length, zero);
        let id = self.reserve(ty);
        self.buffers.push(Vec::new());
        let scaled = self.same(BinaryOp::Div, v, length);
        let then = IrBlock {
            body: self.buffers.pop().expect("the nested buffer is open"),
            result: scaled,
        };
        self.buffers.push(Vec::new());
        let zero = self.number(kind, 0.0);
        let nothing = self.emit(ty, Op::Construct(vec![zero; usize::from(n)]));
        let otherwise = IrBlock {
            body: self.buffers.pop().expect("the nested buffer is open"),
            result: nothing,
        };
        self.place(
            id,
            ty,
            Op::If {
                condition: positive,
                then,
                otherwise,
            },
        )
    }

    /// A matrix's order and its entries, column by column.
    fn entries(&mut self, m: ValueId) -> (usize, Vec<ValueId>) {
        let n = self.ty(m).order().expect("a matrix");
        let mut entries = Vec::with_capacity(usize::from(n) * usize::from(n));
        for c in 0..n {
            let column = self.part(m, c);
            for r in 0..n {
                entries.push(self.part(column, r));
            }
        }
        (usize::from(n), entries)
    }
}

/// The formulas' arithmetic on scalars of one kind, written into the IR.
struct Scalars<'b, 'a> {
    builder: &'b mut Builder<'a>,
    kind: Type,
}

impl Arith for Scalars<'_, '_> {
    type V = ValueId;
    fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.binary(self.kind, BinaryOp::Add, a, b)
    }
    fn sub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.binary(self.kind, BinaryOp::Sub, a, b)
    }
    fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.binary(self.kind, BinaryOp::Mul, a, b)
    }
    fn div(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.binary(self.kind, BinaryOp::Div, a, b)
    }
    fn neg(&mut self, a: ValueId) -> ValueId {
        self.builder
            .emit(self.kind, Op::Unary(orla_ir::UnaryOp::Neg, a))
    }
}
