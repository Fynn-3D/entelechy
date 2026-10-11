//! Expressions to instructions: every form of the language, typed and
//! diagnosed as it lowers.

use entelechy_ir::{BinaryOp, Block as IrBlock, MAX_LOOP_COUNT, Op, Type, UnaryOp, Value, ValueId};
use entelechy_language::{Block, Expr, ExprKind, Span};

use crate::CompileError;

use super::declare::component_index;
use super::{Builder, Local, semantic, subscript_element};

impl Builder<'_> {
    /// Lowers an expression that must produce a plain value (not an image).
    pub(super) fn value(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        let id = self.value_here(expr)?;
        // Where the value is written, the first expression that makes it.
        self.places.entry(id).or_insert(expr.span);
        Ok(id)
    }

    /// [`Builder::value`], its place unrecorded.
    fn value_here(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        match &expr.kind {
            ExprKind::Float(value) => {
                let id = self.written(entelechy_ir::Number::Double(*value));
                self.mark_large(id, expr.span);
                Ok(id)
            }
            ExprKind::Int(value) => Ok(self.written(entelechy_ir::Number::Long(*value))),
            ExprKind::Double(value) => {
                Ok(self.emit(Type::Double, Op::Const(Value::Double(*value))))
            }
            ExprKind::Long(value) => Ok(self.emit(Type::Long, Op::Const(Value::Long(*value)))),
            ExprKind::Bool(value) => Ok(self.emit(Type::Bool, Op::Const(Value::Bool(*value)))),
            ExprKind::Name(name) => self.name(name, expr.span),
            ExprKind::Neg(inner) => self.unary(UnaryOp::Neg, "-", inner, expr.span),
            ExprKind::Not(inner) => self.unary(UnaryOp::Not, "!", inner, expr.span),
            // `&&` and `||` short-circuit, lowered as an `if` whose one
            // arm runs: `n != 0 && x / n > 1` divides only when `n` is
            // not zero.
            ExprKind::Binary(
                op @ (entelechy_language::BinaryOp::And | entelechy_language::BinaryOp::Or),
                left,
                right,
            ) => {
                let and = matches!(op, entelechy_language::BinaryOp::And);
                let symbol = if and { "&&" } else { "||" };
                let condition = self.value(left)?;
                if self.ty(condition) != Type::Bool {
                    return Err(semantic(
                        left.span,
                        format!(
                            "`{symbol}` takes Bool on the left, found {}",
                            self.describe(self.ty(condition))
                        ),
                    ));
                }
                // The IR numbers the value before its blocks, and the
                // blocks in written order.
                let id = self.reserve(Type::Bool);
                let (then, otherwise) = if and {
                    let other = self.nested_expr(right)?;
                    let settled = self.constant_block(false);
                    (other, settled)
                } else {
                    let settled = self.constant_block(true);
                    let other = self.nested_expr(right)?;
                    (settled, other)
                };
                let other = if and { &then } else { &otherwise };
                if self.ty(other.result) != Type::Bool {
                    return Err(semantic(
                        right.span,
                        format!(
                            "`{symbol}` takes Bool on the right, found {}",
                            self.describe(self.ty(other.result))
                        ),
                    ));
                }
                Ok(self.place(
                    id,
                    Type::Bool,
                    Op::If {
                        condition,
                        then,
                        otherwise,
                    },
                ))
            }
            ExprKind::Binary(op, left, right) => {
                let op = match op {
                    entelechy_language::BinaryOp::Add => BinaryOp::Add,
                    entelechy_language::BinaryOp::Sub => BinaryOp::Sub,
                    entelechy_language::BinaryOp::Mul => BinaryOp::Mul,
                    entelechy_language::BinaryOp::Div => BinaryOp::Div,
                    entelechy_language::BinaryOp::Rem => BinaryOp::Rem,
                    entelechy_language::BinaryOp::Less => BinaryOp::Less,
                    entelechy_language::BinaryOp::LessEqual => BinaryOp::LessEqual,
                    entelechy_language::BinaryOp::Greater => BinaryOp::Greater,
                    entelechy_language::BinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
                    entelechy_language::BinaryOp::Equal => BinaryOp::Equal,
                    entelechy_language::BinaryOp::NotEqual => BinaryOp::NotEqual,
                    entelechy_language::BinaryOp::And => BinaryOp::And,
                    entelechy_language::BinaryOp::Or => BinaryOp::Or,
                    entelechy_language::BinaryOp::BitAnd => BinaryOp::BitAnd,
                    entelechy_language::BinaryOp::BitOr => BinaryOp::BitOr,
                    entelechy_language::BinaryOp::BitXor => BinaryOp::BitXor,
                    entelechy_language::BinaryOp::Shl => BinaryOp::Shl,
                    entelechy_language::BinaryOp::Shr => BinaryOp::Shr,
                    entelechy_language::BinaryOp::Ushr => BinaryOp::Ushr,
                };
                let left = self.value(left)?;
                let right = self.value(right)?;
                // Written numbers meet as their exact values.
                if let Some(folded) = self.fold_written(op, left, right) {
                    self.mark_large(folded, expr.span);
                    return Ok(folded);
                }
                // What was written, for a refusal to name.
                let written = (self.ty(left), self.ty(right));
                let mut pair = [left, right];
                self.adapt_all(&mut pair);
                let [left, right] = pair;
                // A product of matrices is its formula.
                if op == BinaryOp::Mul
                    && let Some(product) = self.matrix_product(left, right)
                {
                    return Ok(product);
                }
                let ty = op
                    .result_type(self.ty(left), self.ty(right))
                    .ok_or_else(|| {
                        let (a, b) = written;
                        // Two widths of one kind meet through a conversion.
                        let widths = [a.scalar(), b.scalar()];
                        let hint = self.truncated_hint(&[left, right]);
                        // Two kinds of number meet through a conversion:
                        // each that makes the operation hold is named, a
                        // scalar beside a vector converting alone.
                        let convert = |ty: Option<Type>| match ty {
                            Some(Type::Float) => Some("float"),
                            Some(Type::Int) => Some("int"),
                            Some(Type::Double) => Some("double"),
                            Some(Type::Long) => Some("long"),
                            _ => None,
                        };
                        let vector = |ty: Type| ty.scalar() != Some(ty);
                        let called = |ty: Type| match convert(Some(ty)) {
                            Some(name) => name.to_owned(),
                            None => ty.to_string(),
                        };
                        // `from` converted to `kind`, where `op` then holds.
                        let joins = |from: Type, other: Type, kind: Type, first: bool| {
                            let to = from.with_scalar(kind)?;
                            let (x, y) = if first { (to, other) } else { (other, to) };
                            op.result_type(x, y)?;
                            Some(format!(
                                "`{}(..)` the {}",
                                convert(Some(kind))?,
                                called(from)
                            ))
                        };
                        let hint = match (widths[0], widths[1]) {
                            _ if !hint.is_empty() => hint.to_owned(),
                            (Some(x), Some(y))
                                if x != y
                                    && convert(Some(x)).is_some()
                                    && convert(Some(y)).is_some() =>
                            {
                                let mut ways = Vec::new();
                                if !vector(b) || vector(a) {
                                    ways.extend(joins(b, a, x, false));
                                }
                                if !vector(a) || vector(b) {
                                    ways.extend(joins(a, b, y, true));
                                }
                                match ways.as_slice() {
                                    [] => String::new(),
                                    [one] => format!("; convert: {one}"),
                                    [one, other, ..] => format!(
                                        "; they meet through a conversion: {one}, or {other}"
                                    ),
                                }
                            }
                            _ => String::new(),
                        };
                        semantic(
                            expr.span,
                            format!("cannot apply `{}` to {a} and {b}{hint}", op.symbol()),
                        )
                    })?;
                Ok(self.emit(ty, Op::Binary(op, left, right)))
            }
            ExprKind::Member(inner, member) => {
                // An enum's variant, `Phase.Walk`, is its `Int`.
                if let ExprKind::Name(name) = &inner.kind
                    && !self.binds(name)
                    && self.shared.structs.is_enum(name)
                {
                    let value = super::variant_of(self.shared.structs, name, member, expr.span)?;
                    return Ok(self.emit(Type::Int, Op::Const(Value::Int(value))));
                }
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
                // One letter is a component; two to four are a swizzle
                // from one alphabet (xyzw or rgba).
                let letters: Vec<(u8, bool)> = member
                    .chars()
                    .map(component_index)
                    .collect::<Option<Vec<_>>>()
                    .filter(|letters| {
                        (1..=4).contains(&letters.len())
                            && letters.iter().all(|(_, color)| *color == letters[0].1)
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
                let result = match Type::vector(lane, letters.len() as u8) {
                    Some(result) => result,
                    None => {
                        return Err(semantic(
                            expr.span,
                            format!(
                                "`.{member}` would be a vector of four {lane}s, which there is none"
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
                self.unify(&ids, &mut []);
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
                // The count is checked before the first copy, and the
                // running total of copies as they are written, since
                // nested builders multiply.
                super::check_array_count(count.value, count.span)?;
                // Unrolled: the body once per element, `index()` a literal.
                let mut ids = Vec::with_capacity(count.value as usize);
                for position in 0..count.value {
                    self.each.push((position, self.loops.len()));
                    let id = self.block(body);
                    self.each.pop();
                    ids.push(id?);
                    if self.types.len() > super::MAX_UNROLLED_VALUES {
                        return Err(super::semantic(
                            count.span,
                            format!(
                                "this `[each ...]` writes its body out past {} values; nest fewer or smaller builders, or use a loop",
                                super::MAX_UNROLLED_VALUES
                            ),
                        ));
                    }
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
                // A literal index is a field, checked here.
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
                    .shared
                    .structs
                    .find(&name.value)
                    .map(|index| index as usize)
                    .ok_or_else(|| {
                        // A struct its document does not see is written as
                        // its source writes it, and not offered as itself.
                        let written = entelechy_language::written_name(&name.value);
                        let structs = self.structs();
                        let known: Vec<&str> = structs
                            .iter()
                            .map(|def| def.name.as_str())
                            .filter(|known| *known != written)
                            .collect();
                        semantic(
                            name.span,
                            format!(
                                "`{written}` is not a struct{}",
                                entelechy_language::did_you_mean(written, known)
                            ),
                        )
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
                    let id = self.adapt(id, expected);
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
            ExprKind::Record(fields) => {
                let manifest = self.shared.manifest;
                let Some(record) = self.shared.record else {
                    let why = match manifest.outputs.len() {
                        0 | 1 => "this Kernel has one output, and gives its value".to_owned(),
                        _ => format!(
                            "several outputs are a map's, and this Kernel is {}",
                            super::a(manifest.mode.name())
                        ),
                    };
                    return Err(semantic(
                        expr.span,
                        format!("`{{ name: .. }}` gives a Kernel's outputs by name; {why}"),
                    ));
                };
                let names: Vec<&str> = manifest.outputs.iter().map(|p| p.name.as_str()).collect();
                let mut given: Vec<Option<ValueId>> = vec![None; names.len()];
                for (field, value) in fields {
                    let Some(position) = names.iter().position(|name| *name == field.value) else {
                        return Err(semantic(
                            field.span,
                            format!(
                                "`{}` is not an output of this Kernel{}",
                                field.value,
                                entelechy_language::did_you_mean(
                                    &field.value,
                                    names.iter().copied()
                                )
                            ),
                        ));
                    };
                    if given[position].is_some() {
                        return Err(semantic(
                            field.span,
                            format!("output `{}` is given twice", field.value),
                        ));
                    }
                    let id = self.value(value)?;
                    let expected = manifest.outputs[position].element();
                    let id = self.adapt(id, expected);
                    if self.ty(id) != expected {
                        return Err(semantic(
                            value.span,
                            format!(
                                "output `{}` needs {} per element, found {}",
                                field.value,
                                super::a(&self.describe(expected)),
                                self.describe(self.ty(id))
                            ),
                        ));
                    }
                    given[position] = Some(id);
                }
                let mut parts = Vec::with_capacity(names.len());
                for (position, slot) in given.into_iter().enumerate() {
                    let Some(id) = slot else {
                        return Err(semantic(
                            expr.span,
                            format!(
                                "every output is given once; `{}` is missing",
                                names[position]
                            ),
                        ));
                    };
                    parts.push(id);
                }
                Ok(self.emit(Type::Struct(record), Op::Construct(parts)))
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
                let mut then_block = self.nested(then)?;
                let mut otherwise_block = self.nested(otherwise)?;
                let results = [then_block.result, otherwise_block.result];
                self.unify(&results, &mut [&mut then_block, &mut otherwise_block]);
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
                let (arms, otherwise) = self.arms(arms, otherwise.as_deref(), expr.span)?;
                let mut blocks = Vec::with_capacity(arms.len());
                for (_, body) in &arms {
                    blocks.push(self.nested(body)?);
                }
                let mut otherwise_block = self.nested(otherwise)?;
                {
                    let mut results: Vec<ValueId> =
                        blocks.iter().map(|block| block.result).collect();
                    results.push(otherwise_block.result);
                    let mut all: Vec<&mut IrBlock> = blocks.iter_mut().collect();
                    all.push(&mut otherwise_block);
                    self.unify(&results, &mut all);
                }
                let mut lowered = Vec::with_capacity(arms.len());
                let mut ty: Option<(Type, Span)> = None;
                for ((values, body), block) in arms.into_iter().zip(blocks) {
                    let arm_ty = self.ty(block.result);
                    if let Some((first, _)) = ty
                        && first != arm_ty
                    {
                        return Err(semantic(
                            body.result.span,
                            format!(
                                "`match` arms give {} and {}; they must agree",
                                self.describe(first),
                                self.describe(arm_ty)
                            ),
                        ));
                    }
                    ty = Some((arm_ty, body.result.span));
                    lowered.push((values, block));
                }
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
            ExprKind::Break(_) => Err(semantic(
                expr.span,
                "`break` gives a `loop` its value; write it among a loop body's statements, as `if done { break x; }`",
            )),
            ExprKind::Loop {
                count,
                name,
                init,
                until,
                body,
            } => {
                // A literal count is the bound; any other `Int` is read
                // once, capped at the largest bound.
                let (bound, run_time) = match &count.kind {
                    ExprKind::Int(literal) => {
                        let literal = *literal as u32;
                        if literal > MAX_LOOP_COUNT {
                            return Err(semantic(
                                count.span,
                                format!("`loop` bound {literal} exceeds {MAX_LOOP_COUNT}"),
                            ));
                        }
                        (literal, None)
                    }
                    _ => {
                        let id = self.value(count)?;
                        if self.ty(id) != Type::Int {
                            return Err(semantic(
                                count.span,
                                format!(
                                    "`loop` takes an Int count, found {}",
                                    self.describe(self.ty(id))
                                ),
                            ));
                        }
                        (MAX_LOOP_COUNT, Some(id))
                    }
                };
                let init_id = self.value(init)?;
                let carried = self.ty(init_id);
                // A body that breaks carries the value and whether the
                // loop stopped; the `until` test reads the second, so a
                // `break` exists only in the compiler.
                let pair = breaks(body).then(|| self.pair_type(carried));
                let init_id = match pair {
                    None => init_id,
                    Some(pair) => {
                        let running = self.emit(Type::Bool, Op::Const(Value::Bool(false)));
                        self.emit(pair, Op::Construct(vec![init_id, running]))
                    }
                };
                let held = pair.unwrap_or(carried);
                // The enclosing loop's (or scan's) carried values are read
                // before this loop and stand under their names inside it;
                // `Carried` inside means this loop's own value.
                let outer: Vec<(String, Type, Option<Type>)> = {
                    let mut seen = std::collections::BTreeSet::new();
                    self.locals
                        .iter()
                        .rev()
                        .filter_map(|(local_name, local)| match local {
                            Local::Carried { depth, ty, pair } if *depth == self.loops.len() => {
                                seen.insert(local_name.clone())
                                    .then(|| (local_name.clone(), *ty, *pair))
                            }
                            Local::Value(_) => {
                                seen.insert(local_name.clone());
                                None
                            }
                            _ => None,
                        })
                        .collect()
                };
                let shadows = outer.len();
                for (outer_name, ty, pair) in outer {
                    let value = self.read_carried(ty, pair);
                    self.locals.push((outer_name, Local::Value(value)));
                }
                let id = self.reserve(held);
                self.loops.push(held);
                self.locals.push((
                    name.value.clone(),
                    Local::Carried {
                        depth: self.loops.len(),
                        ty: carried,
                        pair,
                    },
                ));
                let lowered = (|| {
                    let until_block = match pair {
                        None => match until {
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
                        },
                        Some(pair) => Some(self.until_or_stopped(until.as_deref(), pair)?),
                    };
                    let body_block = match pair {
                        None => {
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
                            body_block
                        }
                        Some(pair) => self.nested_with_break(body, carried, pair)?,
                    };
                    Ok((until_block, body_block))
                })();
                self.locals.pop();
                self.locals.truncate(self.locals.len() - shadows);
                self.loops.pop();
                let (until_block, body_block) = lowered?;
                let value = self.place(
                    id,
                    held,
                    Op::Loop {
                        count: bound,
                        bound: run_time,
                        init: init_id,
                        until: until_block,
                        body: body_block,
                    },
                );
                Ok(match pair {
                    None => value,
                    Some(_) => self.emit(carried, Op::Extract(value, 0)),
                })
            }
        }
    }

    /// The exit test of a loop whose body breaks: stopped, or the written
    /// `until`.
    fn until_or_stopped(
        &mut self,
        until: Option<&Expr>,
        pair: Type,
    ) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let built = (|| {
            let carried = self.emit(pair, Op::Carried);
            let stopped = self.emit(Type::Bool, Op::Extract(carried, 1));
            let Some(until) = until else {
                return Ok(stopped);
            };
            let id = self.reserve(Type::Bool);
            let then = self.constant_block(true);
            let otherwise = self.nested_expr(until)?;
            if self.ty(otherwise.result) != Type::Bool {
                return Err(semantic(
                    until.span,
                    format!("`until` needs a Bool, found {}", self.ty(otherwise.result)),
                ));
            }
            Ok(self.place(
                id,
                Type::Bool,
                Op::If {
                    condition: stopped,
                    then,
                    otherwise,
                },
            ))
        })();
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: built?,
        })
    }

    /// A loop body that may `break`, lowered into its own block giving the
    /// next carried value and whether the loop stopped.
    fn nested_with_break(
        &mut self,
        block: &Block,
        carried: Type,
        pair: Type,
    ) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let result = self.scoped(block, |builder, result| {
            builder.result_with_break(result, carried, pair)
        });
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: result?,
        })
    }

    /// The result of a loop body that may `break`: every path gives the
    /// carried value and whether it stopped, one type across paths.
    fn result_with_break(
        &mut self,
        expr: &Expr,
        carried: Type,
        pair: Type,
    ) -> Result<ValueId, CompileError> {
        match &expr.kind {
            ExprKind::Break(value) => {
                let id = self.value(value)?;
                let id = self.adapt(id, carried);
                if self.ty(id) != carried {
                    return Err(semantic(
                        value.span,
                        format!(
                            "loop carries {} but `break` gives {}",
                            self.describe(carried),
                            self.describe(self.ty(id))
                        ),
                    ));
                }
                let stopped = self.emit(Type::Bool, Op::Const(Value::Bool(true)));
                Ok(self.emit(pair, Op::Construct(vec![id, stopped])))
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
                let id = self.reserve(pair);
                let then_block = self.nested_with_break(then, carried, pair)?;
                let otherwise_block = self.nested_with_break(otherwise, carried, pair)?;
                Ok(self.place(
                    id,
                    pair,
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
                let id = self.reserve(pair);
                let (arms, otherwise) = self.arms(arms, otherwise.as_deref(), expr.span)?;
                let mut lowered = Vec::with_capacity(arms.len());
                for (values, body) in arms {
                    let block = self.nested_with_break(body, carried, pair)?;
                    lowered.push((values, block));
                }
                let otherwise_block = self.nested_with_break(otherwise, carried, pair)?;
                Ok(self.place(
                    id,
                    pair,
                    Op::Match {
                        scrutinee: scrutinee_id,
                        arms: lowered,
                        otherwise: otherwise_block,
                    },
                ))
            }
            _ => {
                let id = self.value(expr)?;
                if self.ty(id) != carried {
                    return Err(semantic(
                        expr.span,
                        format!(
                            "loop carries {} but the body gives {}",
                            self.describe(carried),
                            self.describe(self.ty(id))
                        ),
                    ));
                }
                let running = self.emit(Type::Bool, Op::Const(Value::Bool(false)));
                Ok(self.emit(pair, Op::Construct(vec![id, running])))
            }
        }
    }

    /// A `match`'s arms, each with the values it takes, and the block a
    /// value no arm takes runs: the `_` arm, or, where the arms name every
    /// variant of one enum, the last arm, which takes the values that are
    /// no variant too. A value taken twice, and a variant left out, are
    /// refused.
    #[allow(clippy::type_complexity)]
    fn arms<'m>(
        &self,
        arms: &'m [entelechy_language::MatchArm],
        otherwise: Option<&'m Block>,
        span: Span,
    ) -> Result<(Vec<(Vec<i32>, &'m Block)>, &'m Block), CompileError> {
        let mut seen = std::collections::BTreeSet::new();
        // The enum the arms name.
        let mut named: Option<&str> = None;
        let mut resolved = Vec::with_capacity(arms.len());
        for arm in arms {
            let mut values: Vec<(i32, Span, String)> = arm
                .values
                .iter()
                .map(|value| (value.value, value.span, value.value.to_string()))
                .collect();
            for (name, variant) in &arm.variants {
                if let Some(first) = named
                    && first != name.value
                {
                    return Err(semantic(
                        name.span,
                        format!(
                            "a `match` names the variants of one enum, and these are of `{first}` and `{}`",
                            name.value
                        ),
                    ));
                }
                named = Some(&name.value);
                let value = super::variant_of(
                    self.shared.structs,
                    &name.value,
                    &variant.value,
                    variant.span,
                )?;
                values.push((
                    value,
                    variant.span,
                    format!("{}.{}", name.value, variant.value),
                ));
            }
            for (value, span, written) in &values {
                if !seen.insert(*value) {
                    return Err(semantic(
                        *span,
                        format!("`match` already has an arm for {written}"),
                    ));
                }
            }
            resolved.push((
                values.into_iter().map(|(value, ..)| value).collect(),
                &arm.body,
            ));
        }
        if let Some(otherwise) = otherwise {
            return Ok((resolved, otherwise));
        }
        let name = named.expect("the parser keeps a `_` arm where no arm names a variant");
        let variants = self
            .shared
            .structs
            .variants(name)
            .expect("its variants were found above");
        let missing: Vec<String> = variants
            .iter()
            .filter(|(_, number)| !seen.contains(number))
            .map(|(variant, _)| format!("`{name}.{variant}`"))
            .collect();
        if !missing.is_empty() {
            return Err(semantic(
                span,
                format!(
                    "this `match` leaves out {}; name every variant, or end with `_ => ...`",
                    missing.join(", ")
                ),
            ));
        }
        let (_, last) = resolved.pop().expect("a `match` has an arm");
        Ok((resolved, last))
    }

    /// A constant truth-value block: the settled arm of `&&` or `||`.
    fn constant_block(&mut self, value: bool) -> IrBlock {
        self.buffers.push(Vec::new());
        let result = self.emit(Type::Bool, Op::Const(entelechy_ir::Value::Bool(value)));
        let body = self.buffers.pop().expect("the nested buffer is open");
        IrBlock { body, result }
    }

    /// A float constant.
    pub(super) fn constant(&mut self, value: f32) -> ValueId {
        self.emit(Type::Float, Op::Const(entelechy_ir::Value::Float(value)))
    }

    /// A constant of a float kind: a `Float`, or a `Double` exact.
    pub(super) fn number(&mut self, kind: Type, value: f64) -> ValueId {
        match kind {
            Type::Double => self.emit(Type::Double, Op::Const(entelechy_ir::Value::Double(value))),
            _ => self.constant(value as f32),
        }
    }

    pub(super) fn unary(
        &mut self,
        op: UnaryOp,
        symbol: &str,
        inner: &Expr,
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let inner = self.value(inner)?;
        // A negated written number stays written: `-0.1` beside a
        // `Double` is the double nearest minus a tenth.
        if op == UnaryOp::Neg
            && let Some(negated) = self.negate_written(inner)
        {
            self.mark_large(negated, span);
            return Ok(negated);
        }
        let ty = self.ty(inner);
        let result = op
            .result_type(ty)
            .ok_or_else(|| semantic(span, format!("cannot apply `{symbol}` to {ty}")))?;
        Ok(self.emit(result, Op::Unary(op, inner)))
    }
}

/// Whether a block's result, or an arm of the `if`s and `match`es it
/// ends in, may `break`. A nested `loop` holds its own breaks.
fn breaks(block: &Block) -> bool {
    fn of(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Break(_) => true,
            ExprKind::If {
                then, otherwise, ..
            } => breaks(then) || breaks(otherwise),
            ExprKind::Match {
                arms, otherwise, ..
            } => {
                arms.iter().any(|arm| breaks(&arm.body)) || otherwise.as_deref().is_some_and(breaks)
            }
            _ => false,
        }
    }
    of(&block.result)
}
