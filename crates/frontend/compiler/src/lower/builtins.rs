//! Calls by name: the dispatched Field, memory, context and construction
//! builtins, builtins composed from other operations, and the
//! document's own functions, which precede the math intrinsics.

use orla_ir::{
    BinaryOp, Block as IrBlock, Capability, Intrinsic, Mode, Op, Shape, ShapeIndex, Type, Value,
    ValueId,
};
use orla_language::{Expr, Span};

use crate::CompileError;

use super::{Builder, semantic};

/// Builtins a document's function of the same name stands for: those the
/// language gained after sources could already name a function so, which
/// keep their own.
pub(super) const YIELDING: [&str; 1] = ["pads"];

/// The builtins composed from other operations, typed by their arguments;
/// see [`Builder::composed`].
pub(super) const COMPOSED: [&str; 10] = [
    "zero",
    "saturate",
    "distance",
    "radians",
    "degrees",
    "inversesqrt",
    "mod",
    "isnan",
    "isinf",
    "refract",
];

/// The builtins lowered as formulas; see [`composite`](super::composite).
pub(super) const FORMULAS: [&str; 8] = [
    "dot",
    "cross",
    "length",
    "normalize",
    "reflect",
    "transpose",
    "determinant",
    "inverse",
];

impl Builder<'_> {
    /// The builtins lowered to the other operations they spell.
    pub(super) fn composed(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<ValueId, CompileError> {
        // `zero()` has no value alone: it is a struct literal's base, where
        // the literal's struct gives it one.
        if name == "zero" {
            return Err(semantic(
                span,
                "`zero()` is a struct's every field zero, written as the base of its literal: `Name { ..zero(), field: value }`",
            ));
        }
        let mut ids = args
            .iter()
            .map(|arg| self.value(arg))
            .collect::<Result<Vec<_>, _>>()?;
        self.adapt_all(&mut ids);
        let types: Vec<Type> = ids.iter().map(|id| self.ty(*id)).collect();
        // A float of either width, or a vector of them.
        let float_like = |t: Type| t.is_float_kind() && t.order().is_none();
        let vector = |t: Type| t.lanes().is_some() && float_like(t);
        let list = types
            .iter()
            .map(|ty| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let refused = semantic(span, format!("`{name}` cannot be applied to ({list})"));
        // The first argument's scalar kind, for the composition's
        // constants.
        let kind = types
            .first()
            .and_then(|ty| ty.scalar())
            .unwrap_or(Type::Float);
        match (name, types.as_slice()) {
            ("saturate", [a]) if float_like(*a) => {
                let zero = self.number(kind, 0.0);
                let one = self.number(kind, 1.0);
                Ok(self.emit(*a, Op::Intrinsic(Intrinsic::Clamp, vec![ids[0], zero, one])))
            }
            ("distance", [a, b]) if a == b && vector(*a) => {
                let difference = self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], ids[1]));
                Ok(self.length(difference))
            }
            ("radians", [a]) if float_like(*a) => {
                let scale = self.number(kind, std::f64::consts::PI / 180.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("degrees", [a]) if float_like(*a) => {
                let scale = self.number(kind, 180.0 / std::f64::consts::PI);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("inversesqrt", [a]) if float_like(*a) => {
                let root = self.emit(*a, Op::Intrinsic(Intrinsic::Sqrt, vec![ids[0]]));
                let one = self.number(kind, 1.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Div, one, root)))
            }
            ("mod", [a, b]) if float_like(*a) && (a == b || *b == kind) => {
                // x - y * floor(x / y), with y's sign. At an exact
                // multiple a backend may give 0 or nearly y.
                let quotient = self.emit(*a, Op::Binary(BinaryOp::Div, ids[0], ids[1]));
                let floored = self.emit(*a, Op::Intrinsic(Intrinsic::Floor, vec![quotient]));
                let product = self.emit(*a, Op::Binary(BinaryOp::Mul, floored, ids[1]));
                Ok(self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], product)))
            }
            ("isnan", [Type::Float | Type::Double]) => {
                Ok(self.emit(Type::Bool, Op::Binary(BinaryOp::NotEqual, ids[0], ids[0])))
            }
            ("isinf", [a @ (Type::Float | Type::Double)]) => {
                let magnitude = self.emit(*a, Op::Intrinsic(Intrinsic::Abs, vec![ids[0]]));
                let largest = match *a {
                    Type::Double => self.number(kind, f64::MAX),
                    _ => self.constant(f32::MAX),
                };
                Ok(self.emit(
                    Type::Bool,
                    Op::Binary(BinaryOp::Greater, magnitude, largest),
                ))
            }
            ("refract", [i, n, Type::Float]) if i == n && i.components().is_some() => {
                // k = 1 - eta^2 (1 - dot(n, i)^2); below zero, the zero
                // vector; else eta i - (eta dot(n, i) + sqrt(k)) n.
                let (incident, normal, eta) = (ids[0], ids[1], ids[2]);
                let d = self.dot(normal, incident);
                let one = self.constant(1.0);
                let dd = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, d, d));
                let rest = self.emit(Type::Float, Op::Binary(BinaryOp::Sub, one, dd));
                let eta2 = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta, eta));
                let scaled = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta2, rest));
                let k = self.emit(Type::Float, Op::Binary(BinaryOp::Sub, one, scaled));
                let zero = self.constant(0.0);
                let total = self.emit(Type::Bool, Op::Binary(BinaryOp::Less, k, zero));
                let id = self.reserve(*i);
                self.buffers.push(Vec::new());
                let parts = vec![zero; i.components().unwrap_or(0) as usize];
                let nothing = self.emit(*i, Op::Construct(parts));
                let then = IrBlock {
                    body: self.buffers.pop().expect("the nested buffer is open"),
                    result: nothing,
                };
                self.buffers.push(Vec::new());
                let bent = self.emit(*i, Op::Binary(BinaryOp::Mul, incident, eta));
                let along = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta, d));
                let root = self.emit(Type::Float, Op::Intrinsic(Intrinsic::Sqrt, vec![k]));
                let sum = self.emit(Type::Float, Op::Binary(BinaryOp::Add, along, root));
                let back = self.emit(*i, Op::Binary(BinaryOp::Mul, normal, sum));
                let out = self.emit(*i, Op::Binary(BinaryOp::Sub, bent, back));
                let otherwise = IrBlock {
                    body: self.buffers.pop().expect("the nested buffer is open"),
                    result: out,
                };
                Ok(self.place(
                    id,
                    *i,
                    Op::If {
                        condition: total,
                        then,
                        otherwise,
                    },
                ))
            }
            _ => Err(refused),
        }
    }

    pub(super) fn call(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<ValueId, CompileError> {
        match name {
            "put" => {
                // A write belongs to a step: the entry, or a function it
                // calls with the memory as a Field parameter.
                if self.name == "init" {
                    return Err(semantic(
                        span,
                        "`put` writes step by step; the initial carried value is before the first step",
                    ));
                }
                self.require(name, Capability::FieldWrite, span)?;
                self.arity(name, args, 3, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let (port_name, port_ty) = (port.name.clone(), port.ty.clone());
                if !self.manifest.memories.contains(&port_name) {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`put` writes a memory, and `{port_name}` is not one; declare it with `memory {port_name}: {port_ty};`"
                        ),
                    ));
                }
                if self.manifest.mode != Mode::Scan {
                    return Err(semantic(
                        span,
                        "`put` writes step by step, which only `scan` does",
                    ));
                }
                let address = self.value(&args[1])?;
                let expected = port_ty.index_type();
                let address = self.float_address(address, expected);
                if self.ty(address) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`put` addresses `{port_name}`, {}, with {}; it takes {}",
                            super::a(&port_ty.shape.to_string()),
                            self.describe(self.ty(address)),
                            self.describe(expected)
                        ),
                    ));
                }
                let value = self.value(&args[2])?;
                let element = port_ty.element;
                let value = self.adapt(value, element);
                if self.ty(value) != element {
                    return Err(semantic(
                        args[2].span,
                        format!(
                            "`put` writes {} into `{port_name}`, whose element is {}",
                            self.describe(self.ty(value)),
                            self.describe(element)
                        ),
                    ));
                }
                Ok(self.emit(element, Op::Put(input, address, value)))
            }
            "emit" => {
                self.entry_only(name, span)?;
                self.require(name, Capability::FieldWrite, span)?;
                self.arity(name, args, 1, span)?;
                let Mode::Scatter { combine, .. } = self.manifest.mode else {
                    return Err(semantic(
                        span,
                        "`emit` gives a scatter more pairs than the one it returns, which only `scatter` does",
                    ));
                };
                let element = self.manifest.outputs[0].element();
                if !combine.order_free(element, &self.shared.structs.borrow()) {
                    return Err(semantic(
                        span,
                        format!(
                            "`emit` takes a rule whose answer is the same in any order, and `with {}` over {} is not one: scatter `with min`, `with max`, or a `sum` of integers",
                            combine.name(),
                            self.describe(element)
                        ),
                    ));
                }
                let pair = self.value(&args[0])?;
                let pair = self.scatter_pair(pair, args[0].span)?;
                let ty = self.ty(pair);
                Ok(self.emit(ty, Op::Emit(pair)))
            }
            "sample" => {
                self.entry_only(name, span)?;
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                if port.shape() != self.output_shape() {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`sample` reads `{}` at this element's own address, but it is {} and the output is {}; use `at`",
                            port.name,
                            port.shape(),
                            self.output_shape()
                        ),
                    ));
                }
                let element = port.element();
                Ok(self.emit(element, Op::Sample(input)))
            }
            "at" | "fetch" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 2, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let element = port.element();
                let expected = port.ty.index_type();
                let (port_name, shape) = (port.name.clone(), port.shape());
                let index = self.value(&args[1])?;
                let index = self.float_address(index, expected);
                if self.ty(index) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`{name}` addresses `{port_name}`, a {shape}, with {expected} or its integer vector, found {}",
                            self.describe(self.ty(index))
                        ),
                    ));
                }
                Ok(self.emit(element, Op::At(input, index)))
            }
            "at_linear" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 2, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let element = port.element();
                let expected = orla_ir::linear_index_type(port.shape());
                let (port_name, shape) = (port.name.clone(), port.shape());
                let index = self.value(&args[1])?;
                if self.ty(index) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`{name}` addresses `{port_name}`, a {shape}, with {expected}, found {}",
                            self.describe(self.ty(index))
                        ),
                    ));
                }
                let layout = self.manifest.inputs[input as usize].ty.schema.clone();
                if !layout
                    .scalars()
                    .iter()
                    .all(|s| *s == orla_ir::Scalar::Float)
                {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`at_linear` interpolates floats, but `{port_name}` lays out as {layout}"
                        ),
                    ));
                }
                Ok(self.emit(element, Op::AtLinear(input, index)))
            }
            "count" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, _) = self.input_index(&args[0], name)?;
                Ok(self.emit(Type::Int, Op::Count(input)))
            }
            "size" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let ty = port.ty.index_type();
                Ok(self.emit(ty, Op::Size(input)))
            }
            "element" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Int, Op::Element))
            }
            "coord" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                let shape = self.output_shape();
                if shape == Shape::Rows {
                    return Err(semantic(
                        span,
                        "`coord` is for grids and volumes; over rows use `element()`",
                    ));
                }
                Ok(self.emit(shape.index_type(), Op::Coord))
            }
            "domain" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                let ty = self.output_shape().index_type();
                Ok(self.emit(ty, Op::Domain))
            }
            "time" => {
                self.require(name, Capability::TimeRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Time))
            }
            "rate" => {
                self.require(name, Capability::TimeRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Rate))
            }
            "window" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Window))
            }
            "scale" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Scale))
            }
            "pointer" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Pointer))
            }
            "pointer_down" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Bool, Op::PointerDown))
            }
            "button" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 1, span)?;
                let code = self.value(&args[0])?;
                if self.ty(code) != Type::Int {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`button` needs an Int number, found {}",
                            self.describe(self.ty(code))
                        ),
                    ));
                }
                Ok(self.emit(Type::Bool, Op::Button(code)))
            }
            "stick" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 1, span)?;
                let which = self.value(&args[0])?;
                if self.ty(which) != Type::Int {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`stick` needs an Int number, found {}",
                            self.describe(self.ty(which))
                        ),
                    ));
                }
                Ok(self.emit(Type::Vec2, Op::Stick(which)))
            }
            "wheel" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Wheel))
            }
            "key" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 1, span)?;
                let code = self.value(&args[0])?;
                if self.ty(code) != Type::Int {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`key` needs an Int code, found {}",
                            self.describe(self.ty(code))
                        ),
                    ));
                }
                Ok(self.emit(Type::Bool, Op::Key(code)))
            }
            "index" => {
                self.arity(name, args, 0, span)?;
                if let Some((position, depth)) = self.each.last().copied()
                    && depth == self.loops.len()
                {
                    return Ok(self.emit(Type::Int, Op::Const(Value::Int(position as i32))));
                }
                let scanning = self.is_entry && self.manifest.mode == Mode::Scan;
                if self.loops.is_empty() && !scanning {
                    return Err(semantic(
                        span,
                        "`index()` is only available inside a loop, an `[each ...]`, or a `scan`",
                    ));
                }
                Ok(self.emit(Type::Int, Op::Index))
            }
            "vec2" | "vec3" | "vec4" | "ivec2" | "ivec3" | "ivec4" | "dvec2" | "dvec3"
            | "dvec4" | "lvec2" | "lvec3" => {
                let (kind, wanted) = match name {
                    "vec2" => (Type::Float, 2),
                    "vec3" => (Type::Float, 3),
                    "vec4" => (Type::Float, 4),
                    "ivec2" => (Type::Int, 2),
                    "ivec3" => (Type::Int, 3),
                    "ivec4" => (Type::Int, 4),
                    "dvec2" => (Type::Double, 2),
                    "dvec3" => (Type::Double, 3),
                    "dvec4" => (Type::Double, 4),
                    "lvec2" => (Type::Long, 2),
                    _ => (Type::Long, 3),
                };
                let ty = Type::vector(kind, wanted).unwrap_or(kind);
                // Widening components convert: `Int` into a float vector,
                // 32-bit kin into a 64-bit vector. Narrowing is explicit.
                let takes = |from: Type| match kind {
                    Type::Float => matches!(from, Type::Float | Type::Int),
                    Type::Int => from == Type::Int,
                    Type::Double => {
                        matches!(from, Type::Double | Type::Float | Type::Int | Type::Long)
                    }
                    _ => matches!(from, Type::Long | Type::Int),
                };
                // Components from scalars and shorter vectors, in order:
                // `vec4(v3, 1.0)`, `vec3(1.0, v2)`.
                let wanted = usize::from(wanted);
                let mut parts = Vec::with_capacity(wanted);
                let mut given = Vec::with_capacity(args.len());
                for arg in args {
                    let part = self.value(arg)?;
                    let part = self.adapt(part, kind);
                    if matches!(kind, Type::Float | Type::Double) && self.is_truncated(part) {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` would take a whole number that is not what was written{}",
                                self.truncated_hint(&[part])
                            ),
                        ));
                    }
                    let part_ty = self.ty(part);
                    given.push(part_ty.to_string());
                    let (from, count) = match part_ty.shape() {
                        Some((from, count)) if part_ty.order().is_none() && takes(from) => {
                            (from, count)
                        }
                        _ => {
                            let found = self.describe(part_ty);
                            let message = match kind {
                                Type::Float => format!(
                                    "`{name}` components must be Float, Int or a shorter vector, found {found}"
                                ),
                                Type::Int => format!(
                                    "`{name}` components must be Int or a shorter integer vector, found {found}; a Float truncates with int()"
                                ),
                                Type::Double => format!(
                                    "`{name}` components must be numbers or shorter vectors of them, found {found}"
                                ),
                                _ => format!(
                                    "`{name}` components must be Long, Int or a shorter integer vector, found {found}; a float truncates with long()"
                                ),
                            };
                            return Err(semantic(arg.span, message));
                        }
                    };
                    for k in 0..count {
                        let component = if count == 1 {
                            part
                        } else {
                            self.emit(from, Op::Extract(part, u32::from(k)))
                        };
                        parts.push(self.converted(component, kind));
                    }
                }
                if parts.len() != wanted {
                    return Err(semantic(
                        span,
                        format!(
                            "`{name}` takes {wanted} components, found {} in ({})",
                            parts.len(),
                            given.join(", ")
                        ),
                    ));
                }
                Ok(self.emit(ty, Op::Construct(parts)))
            }
            "icoord" | "idomain" => {
                // The float coordinate or extent converted; both are
                // whole.
                let inner = if name == "icoord" { "coord" } else { "domain" };
                let float = self.call(inner, args, span)?;
                let ty = self.ty(float);
                Ok(match ty.other_kind() {
                    Some(integer) => {
                        self.emit(integer, Op::Intrinsic(Intrinsic::ToInt, vec![float]))
                    }
                    None => float,
                })
            }
            "isize" => {
                let float = self.call("size", args, span)?;
                let ty = self.ty(float);
                Ok(match ty.other_kind() {
                    Some(integer) => {
                        self.emit(integer, Op::Intrinsic(Intrinsic::ToInt, vec![float]))
                    }
                    None => float,
                })
            }
            "mat2" | "mat3" | "mat4" | "dmat2" | "dmat3" | "dmat4" => {
                let (kind, order) = match name {
                    "mat2" => (Type::Float, 2),
                    "mat3" => (Type::Float, 3),
                    "mat4" => (Type::Float, 4),
                    "dmat2" => (Type::Double, 2),
                    "dmat3" => (Type::Double, 3),
                    _ => (Type::Double, 4),
                };
                let column = Type::vector(kind, order).unwrap_or(kind);
                let ty = column.matrix_of().unwrap_or(column);
                let order = usize::from(order);
                if args.len() == 1 {
                    // One number: that value down the diagonal.
                    let diagonal = self.value(&args[0])?;
                    let diagonal = self.adapt(diagonal, kind);
                    if self.ty(diagonal) != kind {
                        return Err(semantic(
                            args[0].span,
                            format!(
                                "`{name}` takes {order} {column} columns, or one {kind} for the diagonal; found {}",
                                self.describe(self.ty(diagonal))
                            ),
                        ));
                    }
                    let zero = match kind {
                        Type::Double => self.emit(Type::Double, Op::Const(Value::Double(0.0))),
                        _ => self.constant(0.0),
                    };
                    let columns = (0..order)
                        .map(|c| {
                            let parts = (0..order)
                                .map(|r| if r == c { diagonal } else { zero })
                                .collect();
                            self.emit(column, Op::Construct(parts))
                        })
                        .collect();
                    return Ok(self.emit(ty, Op::Construct(columns)));
                }
                self.arity(name, args, order, span)?;
                let mut columns = Vec::with_capacity(order);
                for arg in args {
                    let part = self.value(arg)?;
                    if self.ty(part) != column {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` takes {column} columns, found {}",
                                self.describe(self.ty(part))
                            ),
                        ));
                    }
                    columns.push(part);
                }
                Ok(self.emit(ty, Op::Construct(columns)))
            }
            _ => {
                // The document's functions first: one may stand for a
                // math builtin of its name.
                let found = self.function_index(name, span)?.map(|index| {
                    let functions = self.shared.functions.borrow();
                    let function = &functions[index as usize].1;
                    (
                        index as usize,
                        function.args.clone(),
                        function.result_type(),
                    )
                });
                if let Some((index, expected, ret)) = found {
                    // A function of a builtin's name stands for it in its
                    // own document: a call that fits the builtin and not the
                    // function says which it met.
                    let hiding = |error: CompileError| match error {
                        CompileError::Semantic {
                            span,
                            message,
                            declaration,
                        } if crate::help::builtin(name).is_some() => CompileError::Semantic {
                            span,
                            message: format!(
                                "{message}; this `{name}` is the document's own function, which hides the builtin `{name}`"
                            ),
                            declaration,
                        },
                        other => other,
                    };
                    self.arity(name, args, expected.len(), span)
                        .map_err(hiding)?;
                    let mut ids = Vec::with_capacity(args.len());
                    for (arg, expected) in args.iter().zip(expected) {
                        let id = self.value(arg)?;
                        let id = self.adapt(id, expected);
                        if self.ty(id) != expected {
                            return Err(hiding(semantic(
                                arg.span,
                                format!(
                                    "`{name}` expects {} here, found {}",
                                    self.describe(expected),
                                    self.describe(self.ty(id))
                                ),
                            )));
                        }
                        ids.push(id);
                    }
                    self.called(index as u32, span)?;
                    return Ok(self.emit(ret, Op::Call(index as u32, ids)));
                }
                if let Some((_, template)) = self.shared.templates.get(name) {
                    return self.instantiate(template, args, span);
                }
                if YIELDING.contains(&name) {
                    self.require(name, Capability::InputRead, span)?;
                    self.arity(name, args, 0, span)?;
                    return Ok(self.emit(Type::Int, Op::Pads));
                }
                if COMPOSED.contains(&name) {
                    return self.composed(name, args, span);
                }
                if FORMULAS.contains(&name) {
                    let mut ids = args
                        .iter()
                        .map(|arg| self.value(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.adapt_all(&mut ids);
                    return self.formula(name, &ids).ok_or_else(|| {
                        let list = ids
                            .iter()
                            .map(|id| self.ty(*id).to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        semantic(span, format!("`{name}` cannot be applied to ({list})"))
                    });
                }
                if let Some(intrinsic) = Intrinsic::from_name(name) {
                    let mut ids = args
                        .iter()
                        .map(|arg| self.value(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.adapt_all(&mut ids);
                    let types = ids.iter().map(|id| self.ty(*id)).collect::<Vec<_>>();
                    // A value converted to its own kind is itself, a
                    // written number's width fixed as a `let` fixes it.
                    let kind = match intrinsic {
                        Intrinsic::ToFloat => Some(Type::Float),
                        Intrinsic::ToInt => Some(Type::Int),
                        Intrinsic::ToDouble => Some(Type::Double),
                        Intrinsic::ToLong => Some(Type::Long),
                        _ => None,
                    };
                    if let (Some(kind), [id], [ty]) = (kind, ids.as_slice(), types.as_slice())
                        && ty.scalar() == Some(kind)
                    {
                        self.fix_width(*id)?;
                        return Ok(*id);
                    }
                    let ty = intrinsic.result_type(&types).ok_or_else(|| {
                        let list = types
                            .iter()
                            .map(|ty| ty.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        semantic(span, format!("`{name}` cannot be applied to ({list})"))
                    })?;
                    return Ok(self.emit(ty, Op::Intrinsic(intrinsic, ids)));
                }
                // The Kernel's functions, then every builtin via `help`.
                let mut declared: Vec<&str> = self
                    .shared
                    .plain
                    .keys()
                    .chain(self.shared.templates.keys())
                    .copied()
                    .collect();
                declared.sort_unstable();
                let near = orla_language::nearest(name, declared)
                    .map(str::to_owned)
                    .or_else(|| {
                        orla_language::nearest(name, crate::help::similar(name)).map(str::to_owned)
                    });
                let suggestion = orla_language::suggestion(near.as_deref());
                Err(semantic(
                    span,
                    format!(
                        "`{name}` is not a builtin, intrinsic, or function of this document or one it uses{suggestion}"
                    ),
                ))
            }
        }
    }
}
