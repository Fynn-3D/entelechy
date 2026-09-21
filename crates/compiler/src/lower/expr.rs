//! Expressions to instructions: every form of the language, typed as
//! it is lowered, with the diagnostics a mistake in one gets.

use orla_ir::{BinaryOp, MAX_LOOP_COUNT, Op, Type, UnaryOp, Value, ValueId};
use orla_language::{Expr, ExprKind, Span};

use crate::CompileError;

use super::declare::component_index;
use super::{Builder, Local, semantic, subscript_element};

impl Builder<'_> {
    /// Lowers an expression that must produce a plain value (not an image).
    pub(super) fn value(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        match &expr.kind {
            ExprKind::Float(value) => Ok(self.emit(Type::Float, Op::Const(Value::Float(*value)))),
            ExprKind::Int(value) => Ok(self.emit(Type::Int, Op::Const(Value::Int(*value)))),
            ExprKind::Bool(value) => Ok(self.emit(Type::Bool, Op::Const(Value::Bool(*value)))),
            ExprKind::Name(name) => self.name(name, expr.span),
            ExprKind::Neg(inner) => self.unary(UnaryOp::Neg, "-", inner, expr.span),
            ExprKind::Not(inner) => self.unary(UnaryOp::Not, "!", inner, expr.span),
            ExprKind::Binary(op, left, right) => {
                let op = match op {
                    orla_language::BinaryOp::Add => BinaryOp::Add,
                    orla_language::BinaryOp::Sub => BinaryOp::Sub,
                    orla_language::BinaryOp::Mul => BinaryOp::Mul,
                    orla_language::BinaryOp::Div => BinaryOp::Div,
                    orla_language::BinaryOp::Rem => BinaryOp::Rem,
                    orla_language::BinaryOp::Less => BinaryOp::Less,
                    orla_language::BinaryOp::LessEqual => BinaryOp::LessEqual,
                    orla_language::BinaryOp::Greater => BinaryOp::Greater,
                    orla_language::BinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
                    orla_language::BinaryOp::Equal => BinaryOp::Equal,
                    orla_language::BinaryOp::NotEqual => BinaryOp::NotEqual,
                    orla_language::BinaryOp::And => BinaryOp::And,
                    orla_language::BinaryOp::Or => BinaryOp::Or,
                    orla_language::BinaryOp::BitAnd => BinaryOp::BitAnd,
                    orla_language::BinaryOp::BitOr => BinaryOp::BitOr,
                    orla_language::BinaryOp::BitXor => BinaryOp::BitXor,
                    orla_language::BinaryOp::Shl => BinaryOp::Shl,
                    orla_language::BinaryOp::Shr => BinaryOp::Shr,
                    orla_language::BinaryOp::Ushr => BinaryOp::Ushr,
                };
                let left = self.value(left)?;
                let right = self.value(right)?;
                let (a, b) = (self.ty(left), self.ty(right));
                let ty = op.result_type(a, b).ok_or_else(|| {
                    semantic(
                        expr.span,
                        format!("cannot apply `{}` to {a} and {b}", op.symbol()),
                    )
                })?;
                Ok(self.emit(ty, Op::Binary(op, left, right)))
            }
            ExprKind::Member(inner, member) => {
                let inner = self.value(inner)?;
                let ty = self.ty(inner);
                if let Type::Struct(index) = ty {
                    let table = self.structs();
                    let def = &table[index as usize];
                    if def.is_array() {
                        return Err(semantic(
                            expr.span,
                            format!("`{}` is an array; take an element with `[index]`", def.name),
                        ));
                    }
                    let field = def
                        .fields
                        .iter()
                        .position(|field| &field.name == member)
                        .ok_or_else(|| {
                            semantic(expr.span, format!("`{}` has no field `{member}`", def.name))
                        })?;
                    let field_ty = def.fields[field].ty;
                    drop(table);
                    return Ok(self.emit(field_ty, Op::Extract(inner, field as u32)));
                }
                if ty.order().is_some() {
                    return Err(semantic(
                        expr.span,
                        format!("{ty} is a matrix; take a column with `[index]`"),
                    ));
                }
                let (count, lane) = ty.lanes().ok_or_else(|| {
                    semantic(
                        expr.span,
                        format!(
                            "`.{member}` needs a vector or struct, found {}",
                            self.describe(ty)
                        ),
                    )
                })?;
                // One letter is a component; two to four are a swizzle,
                // a vector of those components in that order, from one
                // alphabet (xyzw or rgba).
                let letters: Vec<(u8, bool)> = member
                    .chars()
                    .map(component_index)
                    .collect::<Option<Vec<_>>>()
                    .filter(|letters| {
                        (1..=4).contains(&letters.len())
                            && letters.iter().all(|(_, colour)| *colour == letters[0].1)
                    })
                    .ok_or_else(|| {
                        semantic(
                            expr.span,
                            format!(
                                "`.{member}` is not a component or a swizzle; use x, y, z, w or r, g, b, a, one to four of one alphabet"
                            ),
                        )
                    })?;
                if let Some((component, _)) = letters.iter().find(|(c, _)| *c >= count) {
                    let letter = member
                        .chars()
                        .find(|l| component_index(*l).is_some_and(|(c, _)| c == *component))
                        .unwrap_or('?');
                    return Err(semantic(
                        expr.span,
                        format!("{ty} has no component `.{letter}`"),
                    ));
                }
                if letters.len() == 1 {
                    return Ok(self.emit(lane, Op::Extract(inner, u32::from(letters[0].0))));
                }
                let result = match (lane, letters.len()) {
                    (Type::Float, 2) => Type::Vec2,
                    (Type::Float, 3) => Type::Vec3,
                    (Type::Float, 4) => Type::Vec4,
                    (Type::Int, 2) => Type::IVec2,
                    (Type::Int, 3) => Type::IVec3,
                    _ => {
                        return Err(semantic(
                            expr.span,
                            format!(
                                "`.{member}` would be an integer vector of four, which there is none"
                            ),
                        ));
                    }
                };
                let parts = letters
                    .iter()
                    .map(|(component, _)| {
                        self.emit(lane, Op::Extract(inner, u32::from(*component)))
                    })
                    .collect();
                Ok(self.emit(result, Op::Construct(parts)))
            }
            ExprKind::Array(elements) => {
                let mut ids = Vec::with_capacity(elements.len());
                for element in elements {
                    ids.push(self.value(element)?);
                }
                let first = self.ty(ids[0]);
                for (id, element) in ids.iter().zip(elements) {
                    if self.ty(*id) != first {
                        return Err(semantic(
                            element.span,
                            format!(
                                "an array's elements are all one type: this one is {}, the first {}",
                                self.describe(self.ty(*id)),
                                self.describe(first)
                            ),
                        ));
                    }
                }
                let ty = self.array_type(first, ids.len() as u32, expr.span)?;
                Ok(self.emit(ty, Op::Construct(ids)))
            }
            ExprKind::Repeat(element, count) => {
                let id = self.value(element)?;
                let ty = self.array_type(self.ty(id), count.value, count.span)?;
                Ok(self.emit(ty, Op::Construct(vec![id; count.value as usize])))
            }
            ExprKind::Each(count, body) => {
                // Unrolled: the body once per element, `index()` a literal.
                let mut ids = Vec::with_capacity(count.value as usize);
                for position in 0..count.value {
                    self.each.push((position, self.loops.len()));
                    let id = self.block(body);
                    self.each.pop();
                    ids.push(id?);
                }
                let ty = self.array_type(self.ty(ids[0]), count.value, count.span)?;
                Ok(self.emit(ty, Op::Construct(ids)))
            }
            ExprKind::Subscript(array, index) => {
                let array_id = self.value(array)?;
                let array_ty = self.ty(array_id);
                let Some((element, count)) = subscript_element(array_ty, &self.structs()) else {
                    return Err(semantic(
                        array.span,
                        format!(
                            "`[index]` takes an element of an array, a column of a matrix or a component of a vector, found {}",
                            self.describe(array_ty)
                        ),
                    ));
                };
                // A literal index is a field: checked here, free to run.
                let literal = match &index.kind {
                    ExprKind::Int(value) => Some(*value),
                    ExprKind::Neg(inner) => match &inner.kind {
                        ExprKind::Int(value) => Some(value.wrapping_neg()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(literal) = literal {
                    if literal < 0 || literal as usize >= count {
                        return Err(semantic(
                            index.span,
                            format!(
                                "index {literal} is outside {}, whose indices are 0 to {}",
                                self.describe(array_ty),
                                count - 1
                            ),
                        ));
                    }
                    return Ok(self.emit(element, Op::Extract(array_id, literal as u32)));
                }
                let index_id = self.value(index)?;
                if self.ty(index_id) != Type::Int {
                    return Err(semantic(
                        index.span,
                        format!(
                            "an array takes an Int index, found {}",
                            self.describe(self.ty(index_id))
                        ),
                    ));
                }
                Ok(self.emit(element, Op::Subscript(array_id, index_id)))
            }
            ExprKind::Call(name, args) => self.call(name, args, expr.span),
            ExprKind::Struct { name, base, fields } => {
                let index = self
                    .structs()
                    .iter()
                    .position(|def| def.name == name.value)
                    .ok_or_else(|| {
                        semantic(name.span, format!("`{}` is not a struct", name.value))
                    })?;
                let ty = Type::Struct(index as u32);
                let base_id = match base {
                    Some(base) => {
                        let id = self.value(base)?;
                        if self.ty(id) != ty {
                            return Err(semantic(
                                base.span,
                                format!(
                                    "`..base` must be a `{}`, found {}",
                                    name.value,
                                    self.describe(self.ty(id))
                                ),
                            ));
                        }
                        Some(id)
                    }
                    None => None,
                };
                let field_count = self.structs()[index].fields.len();
                let mut given: Vec<Option<ValueId>> = vec![None; field_count];
                for (field, value) in fields {
                    let position = self.structs()[index]
                        .fields
                        .iter()
                        .position(|f| f.name == field.value)
                        .ok_or_else(|| {
                            semantic(
                                field.span,
                                format!("`{}` has no field `{}`", name.value, field.value),
                            )
                        })?;
                    if given[position].is_some() {
                        return Err(semantic(
                            field.span,
                            format!("field `{}` is set twice", field.value),
                        ));
                    }
                    let id = self.value(value)?;
                    let expected = self.structs()[index].fields[position].ty;
                    if self.ty(id) != expected {
                        return Err(semantic(
                            value.span,
                            format!(
                                "`{}.{}` expects {}, found {}",
                                name.value,
                                field.value,
                                self.describe(expected),
                                self.describe(self.ty(id))
                            ),
                        ));
                    }
                    given[position] = Some(id);
                }
                let mut parts = Vec::with_capacity(field_count);
                for (position, slot) in given.into_iter().enumerate() {
                    let id = match (slot, base_id) {
                        (Some(id), _) => id,
                        (None, Some(base)) => {
                            let field_ty = self.structs()[index].fields[position].ty;
                            self.emit(field_ty, Op::Extract(base, position as u32))
                        }
                        (None, None) => {
                            return Err(semantic(
                                name.span,
                                format!(
                                    "`{}` needs field `{}`; set it or start from `..base`",
                                    name.value,
                                    self.structs()[index].fields[position].name
                                ),
                            ));
                        }
                    };
                    parts.push(id);
                }
                Ok(self.emit(ty, Op::Construct(parts)))
            }
            ExprKind::If {
                condition,
                then,
                otherwise,
            } => {
                let condition_id = self.value(condition)?;
                if self.ty(condition_id) != Type::Bool {
                    return Err(semantic(
                        condition.span,
                        format!(
                            "`if` needs a Bool condition, found {}",
                            self.ty(condition_id)
                        ),
                    ));
                }
                let id = self.reserve(Type::Bool);
                let then_block = self.nested(then)?;
                let otherwise_block = self.nested(otherwise)?;
                let (a, b) = (self.ty(then_block.result), self.ty(otherwise_block.result));
                if a != b {
                    return Err(semantic(
                        otherwise.result.span,
                        format!(
                            "`if` branches give {} and {}; they must agree",
                            self.describe(a),
                            self.describe(b)
                        ),
                    ));
                }
                Ok(self.place(
                    id,
                    a,
                    Op::If {
                        condition: condition_id,
                        then: then_block,
                        otherwise: otherwise_block,
                    },
                ))
            }
            ExprKind::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let scrutinee_id = self.value(scrutinee)?;
                if self.ty(scrutinee_id) != Type::Int {
                    return Err(semantic(
                        scrutinee.span,
                        format!(
                            "`match` takes an Int, found {}",
                            self.describe(self.ty(scrutinee_id))
                        ),
                    ));
                }
                let id = self.reserve(Type::Int);
                let mut seen: std::collections::BTreeMap<i32, Span> =
                    std::collections::BTreeMap::new();
                let mut lowered = Vec::with_capacity(arms.len());
                let mut ty: Option<(Type, Span)> = None;
                for arm in arms {
                    for value in &arm.values {
                        if seen.insert(value.value, value.span).is_some() {
                            return Err(semantic(
                                value.span,
                                format!("`match` already has an arm for {}", value.value),
                            ));
                        }
                    }
                    let block = self.nested(&arm.body)?;
                    let arm_ty = self.ty(block.result);
                    if let Some((first, _)) = ty
                        && first != arm_ty
                    {
                        return Err(semantic(
                            arm.body.result.span,
                            format!(
                                "`match` arms give {} and {}; they must agree",
                                self.describe(first),
                                self.describe(arm_ty)
                            ),
                        ));
                    }
                    ty = Some((arm_ty, arm.body.result.span));
                    lowered.push((arm.values.iter().map(|v| v.value).collect(), block));
                }
                let otherwise_block = self.nested(otherwise)?;
                let otherwise_ty = self.ty(otherwise_block.result);
                if let Some((first, _)) = ty
                    && first != otherwise_ty
                {
                    return Err(semantic(
                        otherwise.result.span,
                        format!(
                            "`match` arms give {} and {}; they must agree",
                            self.describe(first),
                            self.describe(otherwise_ty)
                        ),
                    ));
                }
                Ok(self.place(
                    id,
                    otherwise_ty,
                    Op::Match {
                        scrutinee: scrutinee_id,
                        arms: lowered,
                        otherwise: otherwise_block,
                    },
                ))
            }
            ExprKind::Loop {
                count,
                name,
                init,
                until,
                body,
            } => {
                if count.value > MAX_LOOP_COUNT {
                    return Err(semantic(
                        count.span,
                        format!("`loop` bound {} exceeds {MAX_LOOP_COUNT}", count.value),
                    ));
                }
                let init_id = self.value(init)?;
                let carried = self.ty(init_id);
                // The values the enclosing loop (or the scan) carries are
                // read here, before this loop begins, and stand under their
                // names inside it: they do not change while it runs, and
                // `Carried` inside would mean this loop's own value.
                let outer: Vec<(String, Type)> = {
                    let mut seen = std::collections::BTreeSet::new();
                    self.locals
                        .iter()
                        .rev()
                        .filter_map(|(local_name, local)| match local {
                            Local::Carried { depth, ty } if *depth == self.loops.len() => seen
                                .insert(local_name.clone())
                                .then(|| (local_name.clone(), *ty)),
                            Local::Value(_) => {
                                seen.insert(local_name.clone());
                                None
                            }
                            _ => None,
                        })
                        .collect()
                };
                let shadows = outer.len();
                for (outer_name, ty) in outer {
                    let value = self.emit(ty, Op::Carried);
                    self.locals.push((outer_name, Local::Value(value)));
                }
                let id = self.reserve(carried);
                self.loops.push(carried);
                self.locals.push((
                    name.value.clone(),
                    Local::Carried {
                        depth: self.loops.len(),
                        ty: carried,
                    },
                ));
                let lowered = (|| {
                    let until_block = match until {
                        Some(until) => {
                            let block = self.nested_expr(until)?;
                            if self.ty(block.result) != Type::Bool {
                                return Err(semantic(
                                    until.span,
                                    format!(
                                        "`until` needs a Bool, found {}",
                                        self.ty(block.result)
                                    ),
                                ));
                            }
                            Some(block)
                        }
                        None => None,
                    };
                    let body_block = self.nested(body)?;
                    let next = self.ty(body_block.result);
                    if next != carried {
                        return Err(semantic(
                            body.result.span,
                            format!(
                                "loop carries {} but the body gives {}",
                                self.describe(carried),
                                self.describe(next)
                            ),
                        ));
                    }
                    Ok((until_block, body_block))
                })();
                self.locals.pop();
                self.locals.truncate(self.locals.len() - shadows);
                self.loops.pop();
                let (until_block, body_block) = lowered?;
                Ok(self.place(
                    id,
                    carried,
                    Op::Loop {
                        count: count.value,
                        init: init_id,
                        until: until_block,
                        body: body_block,
                    },
                ))
            }
        }
    }

    /// A float constant.
    pub(super) fn constant(&mut self, value: f32) -> ValueId {
        self.emit(Type::Float, Op::Const(orla_ir::Value::Float(value)))
    }

    pub(super) fn unary(
        &mut self,
        op: UnaryOp,
        symbol: &str,
        inner: &Expr,
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let inner = self.value(inner)?;
        let ty = self.ty(inner);
        let result = op
            .result_type(ty)
            .ok_or_else(|| semantic(span, format!("cannot apply `{symbol}` to {ty}")))?;
        Ok(self.emit(result, Op::Unary(op, inner)))
    }
}
