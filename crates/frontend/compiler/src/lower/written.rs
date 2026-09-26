//! Numbers written without a width. A written number is exact until it
//! meets a width: arithmetic between written numbers is done here, on
//! the exact values, and gives a written number; beside a `Double` one
//! is a `Double`, beside a `Long` a `Long`; and a vector, a matrix or an
//! array built of written numbers alone takes the kind it meets as a
//! whole. Where nothing names a width, a written float is a `Float` and
//! a written integer an `Int` (a `Long` when it does not fit). A `let`
//! names one, and fixes its width: every use sees one value.

use orla_ir::{BinaryOp, Block as IrBlock, Number, Op, Type, Value, ValueId};

use super::Builder;

impl Builder<'_> {
    /// A number written without a width, emitted at its default width.
    pub(super) fn written(&mut self, number: Number) -> ValueId {
        let (ty, value) = match number {
            Number::Double(v) => (Type::Float, Value::Float(v as f32)),
            Number::Long(v) => match i32::try_from(v) {
                Ok(v) => (Type::Int, Value::Int(v)),
                Err(_) => (Type::Long, Value::Long(v)),
            },
            _ => unreachable!("only numbers are written"),
        };
        let id = self.emit(ty, Op::Const(value));
        self.literals.insert(id.0, number);
        id
    }

    /// Whether a value is written numbers all the way down, with nothing
    /// naming it yet.
    pub(super) fn is_written(&self, id: ValueId) -> bool {
        self.literals.contains_key(&id.0) || self.aggregates.contains(&id.0)
    }

    /// `a op b` on two written numbers, done on their exact values: a
    /// written number, or `None` for an operator that is not arithmetic,
    /// an operand that is not a written number, or integer arithmetic
    /// that fails, which is left to fail where it runs.
    pub(super) fn fold_written(&mut self, op: BinaryOp, a: ValueId, b: ValueId) -> Option<ValueId> {
        let (x, y) = (*self.literals.get(&a.0)?, *self.literals.get(&b.0)?);
        let truncated = self.truncated.contains(&a.0) || self.truncated.contains(&b.0);
        let number = match (x, y) {
            (Number::Long(x), Number::Long(y)) => {
                // A division that leaves a remainder is whole, not exact:
                // it stays an integer, and says so to whatever meets it.
                let dropped = op == BinaryOp::Div && y != 0 && x % y != 0;
                let folded = self.written(Number::Long(match op {
                    BinaryOp::Add => x.checked_add(y)?,
                    BinaryOp::Sub => x.checked_sub(y)?,
                    BinaryOp::Mul => x.checked_mul(y)?,
                    BinaryOp::Div if y != 0 => x.checked_div(y)?,
                    BinaryOp::Rem if y != 0 => x.checked_rem(y)?,
                    _ => return None,
                }));
                if truncated || dropped {
                    self.truncated.insert(folded.0);
                }
                return Some(folded);
            }
            // A truncated integer beside a written float is not folded:
            // an integer meets a float, and is refused where it does.
            _ if truncated => return None,
            _ => {
                let exact = |n: Number| match n {
                    Number::Long(v) => v as f64,
                    Number::Double(v) => v,
                    _ => f64::NAN,
                };
                let (x, y) = (exact(x), exact(y));
                Number::Double(match op {
                    BinaryOp::Add => x + y,
                    BinaryOp::Sub => x - y,
                    BinaryOp::Mul => x * y,
                    BinaryOp::Div => x / y,
                    BinaryOp::Rem => x % y,
                    _ => return None,
                })
            }
        };
        Some(self.written(number))
    }

    /// Whether `id` is a written integer a division left a remainder of,
    /// which no float takes.
    pub(super) fn is_truncated(&self, id: ValueId) -> bool {
        self.truncated.contains(&id.0)
    }

    /// What a refusal says of a truncated written integer meeting a float.
    pub(super) fn truncated_hint(&self, ids: &[ValueId]) -> &'static str {
        if ids.iter().any(|id| self.is_truncated(*id)) {
            "; a division of whole numbers that leaves a remainder is whole, not a fraction: write `1.0 / 2` for a half"
        } else {
            ""
        }
    }

    /// `-a` of a written number, as a written number.
    pub(super) fn negate_written(&mut self, a: ValueId) -> Option<ValueId> {
        let number = match *self.literals.get(&a.0)? {
            Number::Long(v) => Number::Long(v.checked_neg()?),
            Number::Double(v) => Number::Double(-v),
            _ => return None,
        };
        Some(self.written(number))
    }

    /// `id` as a value of `want`, when it is written numbers that can be
    /// one: a number of the 64-bit kind `want` is, or a vector, a matrix
    /// or an array of them. Anything else is left as it is, for the
    /// caller's check to name.
    pub(super) fn adapt(&mut self, id: ValueId, want: Type) -> ValueId {
        self.retype(id, want, &mut []);
        id
    }

    /// Adapts the written values among `ids` to the kind of the first of
    /// them that is not written, when a written number can be of it:
    /// `x + 0.1` and `v * vec3(1.0, 2.0, 3.0)` beside doubles, `x * 2`
    /// beside a float.
    pub(super) fn adapt_all(&mut self, ids: &mut [ValueId]) {
        let Some(kind) = ids
            .iter()
            .filter(|id| !self.is_written(**id))
            .find_map(|id| self.ty(*id).scalar())
            .filter(|scalar| matches!(*scalar, Type::Float | Type::Double | Type::Long))
        else {
            return;
        };
        for id in ids {
            if let Some(want) = self.ty(*id).with_scalar(kind) {
                self.retype(*id, want, &mut []);
            }
        }
    }

    /// Values that must be of one type (an array's elements, an `if`'s
    /// branches): the written ones take the type of the first that is not
    /// written, their instructions found in the open blocks or in
    /// `blocks`.
    pub(super) fn unify(&mut self, ids: &[ValueId], blocks: &mut [&mut IrBlock]) {
        let Some(want) = ids
            .iter()
            .find(|id| !self.is_written(**id))
            .map(|id| self.ty(*id))
        else {
            return;
        };
        for id in ids {
            self.retype(*id, want, blocks);
        }
    }

    /// Retypes a written value as `want`, when every part of it can be:
    /// nothing changes unless all of it does.
    fn retype(&mut self, id: ValueId, want: Type, blocks: &mut [&mut IrBlock]) -> bool {
        if self.ty(id) == want {
            return true;
        }
        if !self.fits(id, want, blocks) {
            return false;
        }
        self.rewrite(id, want, blocks);
        true
    }

    /// Whether a written value can be retyped as `want`.
    fn fits(&self, id: ValueId, want: Type, blocks: &[&mut IrBlock]) -> bool {
        if self.ty(id) == want {
            return true;
        }
        if !self.locatable(id, blocks) {
            return false;
        }
        if self.literals.contains_key(&id.0) {
            return self.as_value(id, want).is_some();
        }
        if !self.aggregates.contains(&id.0) {
            return false;
        }
        let Some(parts) = self.parts_of(id, blocks) else {
            return false;
        };
        let Some(wanted) = self.part_types(self.ty(id), want, parts.len()) else {
            return false;
        };
        parts
            .iter()
            .zip(wanted)
            .all(|(part, ty)| self.fits(*part, ty, blocks))
    }

    /// Retypes a value `fits` said can be.
    fn rewrite(&mut self, id: ValueId, want: Type, blocks: &mut [&mut IrBlock]) {
        if self.ty(id) == want {
            return;
        }
        let op = match self.as_value(id, want) {
            Some(value) => Some(Op::Const(value)),
            None => {
                let parts = self.parts_of(id, blocks).expect("fits found them");
                let wanted = self
                    .part_types(self.ty(id), want, parts.len())
                    .expect("fits found them");
                for (part, ty) in parts.iter().zip(wanted) {
                    self.rewrite(*part, ty, blocks);
                }
                None
            }
        };
        let instruction = self
            .buffers
            .iter_mut()
            .flat_map(|buffer| buffer.iter_mut())
            .chain(blocks.iter_mut().flat_map(|block| block.body.iter_mut()))
            .find(|instruction| instruction.id == id)
            .expect("fits found it");
        instruction.ty = want;
        if let Some(op) = op {
            instruction.op = op;
        }
        self.types[id.0 as usize] = want;
        self.literals.remove(&id.0);
        self.aggregates.remove(&id.0);
    }

    /// A written number as a value of `want`: a float as a `Double`, an
    /// integer as a `Double` or a `Long`.
    fn as_value(&self, id: ValueId, want: Type) -> Option<Value> {
        match (self.literals.get(&id.0)?, want) {
            (Number::Double(v), Type::Double) => Some(Value::Double(*v)),
            (Number::Double(v), Type::Float) => Some(Value::Float(*v as f32)),
            (Number::Long(_), Type::Double | Type::Float) if self.is_truncated(id) => None,
            (Number::Long(v), Type::Double) => Some(Value::Double(*v as f64)),
            (Number::Long(v), Type::Float) => Some(Value::Float(*v as f32)),
            (Number::Long(v), Type::Long) => Some(Value::Long(*v)),
            _ => None,
        }
    }

    /// Whether a value's instruction is in an open block or in `blocks`,
    /// where a rewrite can reach it.
    fn locatable(&self, id: ValueId, blocks: &[&mut IrBlock]) -> bool {
        self.buffers
            .iter()
            .flat_map(|buffer| buffer.iter())
            .chain(blocks.iter().flat_map(|block| block.body.iter()))
            .any(|instruction| instruction.id == id)
    }

    /// The parts a written aggregate was built from.
    fn parts_of(&self, id: ValueId, blocks: &[&mut IrBlock]) -> Option<Vec<ValueId>> {
        self.buffers
            .iter()
            .flat_map(|buffer| buffer.iter())
            .chain(blocks.iter().flat_map(|block| block.body.iter()))
            .find(|instruction| instruction.id == id)
            .and_then(|instruction| match &instruction.op {
                Op::Construct(parts) => Some(parts.clone()),
                _ => None,
            })
    }

    /// What each of `n` parts must be for a value of type `have` to be
    /// one of `want`: of one shape, a vector of a vector's, a matrix of a
    /// matrix's, an array of an array's.
    fn part_types(&self, have: Type, want: Type, n: usize) -> Option<Vec<Type>> {
        match (have, want) {
            (Type::Vector(_, a), Type::Vector(kind, b)) if a == b && usize::from(b) == n => {
                Some(vec![Type::Scalar(kind); n])
            }
            (Type::Matrix(_, a), Type::Matrix(_, b)) if a == b && usize::from(b) == n => {
                Some(vec![want.column()?; n])
            }
            (Type::Struct(a), Type::Struct(b)) => {
                let table = self.structs();
                let (have, want) = (&table[a as usize], &table[b as usize]);
                (have.is_array() && want.is_array() && want.fields.len() == n)
                    .then(|| want.fields.iter().map(|field| field.ty).collect())
            }
            _ => None,
        }
    }
}
