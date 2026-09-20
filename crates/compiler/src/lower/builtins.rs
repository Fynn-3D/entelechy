//! Calls by name: the Field, memory, context and construction builtins
//! the lowering dispatches, the builtins composed from other operations
//! so every backend has them, and, before the math intrinsics, the
//! document's own functions.

use orla_ir::{
    BinaryOp, Block as IrBlock, Capability, Intrinsic, Mode, Op, Shape, ShapeIndex, Type, ValueId,
};
use orla_language::{Expr, Span};

use crate::CompileError;

use super::{Builder, result_type, semantic};

impl Builder<'_> {
    /// The builtins that are spellings of other operations: lowered to
    /// those operations, so every backend has them by construction.
    pub(super) fn composed(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let ids = args
            .iter()
            .map(|arg| self.value(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let types: Vec<Type> = ids.iter().map(|id| self.ty(*id)).collect();
        let float_like = |t: Type| matches!(t, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4);
        let vector = |t: Type| t.components().is_some();
        let list = types
            .iter()
            .map(|ty| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let refused = semantic(span, format!("`{name}` cannot be applied to ({list})"));
        match (name, types.as_slice()) {
            ("saturate", [a]) if float_like(*a) => {
                let zero = self.constant(0.0);
                let one = self.constant(1.0);
                Ok(self.emit(*a, Op::Intrinsic(Intrinsic::Clamp, vec![ids[0], zero, one])))
            }
            ("distance", [a, b]) if a == b && vector(*a) => {
                let difference = self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], ids[1]));
                Ok(self.emit(
                    Type::Float,
                    Op::Intrinsic(Intrinsic::Length, vec![difference]),
                ))
            }
            ("radians", [a]) if float_like(*a) => {
                let scale = self.constant(std::f32::consts::PI / 180.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("degrees", [a]) if float_like(*a) => {
                let scale = self.constant(180.0 / std::f32::consts::PI);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("inversesqrt", [a]) if float_like(*a) => {
                let root = self.emit(*a, Op::Intrinsic(Intrinsic::Sqrt, vec![ids[0]]));
                let one = self.constant(1.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Div, one, root)))
            }
            ("mod", [a, b]) if float_like(*a) && (a == b || *b == Type::Float) => {
                // x - y * floor(x / y): the remainder that takes y's sign.
                // At an exact multiple it is a knife-edge, 0 or nearly y,
                // as floor is at an integer; backends may fall either way.
                let quotient = self.emit(*a, Op::Binary(BinaryOp::Div, ids[0], ids[1]));
                let floored = self.emit(*a, Op::Intrinsic(Intrinsic::Floor, vec![quotient]));
                let product = self.emit(*a, Op::Binary(BinaryOp::Mul, floored, ids[1]));
                Ok(self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], product)))
            }
            ("isnan", [Type::Float]) => {
                Ok(self.emit(Type::Bool, Op::Binary(BinaryOp::NotEqual, ids[0], ids[0])))
            }
            ("isinf", [Type::Float]) => {
                let magnitude = self.emit(Type::Float, Op::Intrinsic(Intrinsic::Abs, vec![ids[0]]));
                let largest = self.constant(f32::MAX);
                Ok(self.emit(
                    Type::Bool,
                    Op::Binary(BinaryOp::Greater, magnitude, largest),
                ))
            }
            ("refract", [i, n, Type::Float]) if i == n && vector(*i) => {
                // k = 1 - eta^2 (1 - dot(n, i)^2); below zero, the zero
                // vector; else eta i - (eta dot(n, i) + sqrt(k)) n.
                let (incident, normal, eta) = (ids[0], ids[1], ids[2]);
                let d = self.emit(
                    Type::Float,
                    Op::Intrinsic(Intrinsic::Dot, vec![normal, incident]),
                );
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
                // A write belongs to a step: the entry, or a function the
                // step calls with the memory as a Field parameter. The
                // carried value's start is before every step.
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
                            "`put` addresses `{port_name}`, a {}, with {}; it takes {}",
                            port_ty.shape,
                            self.describe(self.ty(address)),
                            self.describe(expected)
                        ),
                    ));
                }
                let value = self.value(&args[2])?;
                let element = port_ty.element;
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
                let scanning = self.is_entry && self.manifest.mode == Mode::Scan;
                if self.loops.is_empty() && !scanning {
                    return Err(semantic(
                        span,
                        "`index()` is only available inside a loop or a `scan`",
                    ));
                }
                Ok(self.emit(Type::Int, Op::Index))
            }
            "vec2" | "vec3" | "vec4" => {
                let ty = match name {
                    "vec2" => Type::Vec2,
                    "vec3" => Type::Vec3,
                    _ => Type::Vec4,
                };
                // Components from scalars and shorter vectors, in order,
                // as many as the vector has: `vec4(v3, 1.0)`, `vec3(1.0,
                // v2)`. An Int component widens; a cell address needs no
                // `float()` at every use.
                let wanted = ty.components().unwrap_or(0) as usize;
                let mut parts = Vec::with_capacity(wanted);
                let mut given = Vec::with_capacity(args.len());
                for arg in args {
                    let part = self.value(arg)?;
                    let part_ty = self.ty(part);
                    given.push(part_ty.to_string());
                    match part_ty {
                        Type::Float => parts.push(part),
                        Type::Int => parts.push(
                            self.emit(Type::Float, Op::Intrinsic(Intrinsic::ToFloat, vec![part])),
                        ),
                        Type::Vec2 | Type::Vec3 | Type::Vec4 => {
                            for k in 0..part_ty.components().unwrap_or(0) {
                                parts.push(self.emit(Type::Float, Op::Extract(part, u32::from(k))));
                            }
                        }
                        other => {
                            return Err(semantic(
                                arg.span,
                                format!(
                                    "`{name}` components must be Float, Int or a shorter vector, found {other}"
                                ),
                            ));
                        }
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
            "ivec2" | "ivec3" => {
                let ty = if name == "ivec2" {
                    Type::IVec2
                } else {
                    Type::IVec3
                };
                let wanted = ty.int_components().unwrap_or(0) as usize;
                let mut parts = Vec::with_capacity(wanted);
                let mut given = Vec::with_capacity(args.len());
                for arg in args {
                    let part = self.value(arg)?;
                    let part_ty = self.ty(part);
                    given.push(part_ty.to_string());
                    match part_ty {
                        Type::Int => parts.push(part),
                        Type::IVec2 | Type::IVec3 => {
                            for k in 0..part_ty.int_components().unwrap_or(0) {
                                parts.push(self.emit(Type::Int, Op::Extract(part, u32::from(k))));
                            }
                        }
                        other => {
                            return Err(semantic(
                                arg.span,
                                format!(
                                    "`{name}` components must be Int or a shorter integer vector, found {}; a Float truncates with int()",
                                    self.describe(other)
                                ),
                            ));
                        }
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
                // The float coordinate or extent, converted: every value
                // either takes is a whole number.
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
            "mat2" | "mat3" | "mat4" => {
                let ty = match name {
                    "mat2" => Type::Mat2,
                    "mat3" => Type::Mat3,
                    _ => Type::Mat4,
                };
                let order = ty.order().unwrap_or(0) as usize;
                let column = ty.column().unwrap_or(Type::Float);
                if args.len() == 1 {
                    // One Float: that value down the diagonal.
                    let diagonal = self.value(&args[0])?;
                    if self.ty(diagonal) != Type::Float {
                        return Err(semantic(
                            args[0].span,
                            format!(
                                "`{name}` takes {order} {column} columns, or one Float for the diagonal; found {}",
                                self.describe(self.ty(diagonal))
                            ),
                        ));
                    }
                    let zero = self.constant(0.0);
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
                // A function of the document first: one may take a math
                // builtin's name and stand for it.
                let found = {
                    let functions = self.shared.functions.borrow();
                    functions.iter().position(|(f, _)| f == name).map(|index| {
                        (
                            index,
                            functions[index].1.args.clone(),
                            result_type(&functions[index].1),
                        )
                    })
                };
                if let Some((index, expected, ret)) = found {
                    self.arity(name, args, expected.len(), span)?;
                    let mut ids = Vec::with_capacity(args.len());
                    for (arg, expected) in args.iter().zip(expected) {
                        let id = self.value(arg)?;
                        if self.ty(id) != expected {
                            return Err(semantic(
                                arg.span,
                                format!(
                                    "`{name}` expects {} here, found {}",
                                    self.describe(expected),
                                    self.describe(self.ty(id))
                                ),
                            ));
                        }
                        ids.push(id);
                    }
                    return Ok(self.emit(ret, Op::Call(index as u32, ids)));
                }
                if let Some(template) = self.shared.templates.iter().find(|t| t.name.value == name)
                {
                    return self.instantiate(template, args, span);
                }
                if matches!(
                    name,
                    "saturate"
                        | "distance"
                        | "radians"
                        | "degrees"
                        | "inversesqrt"
                        | "mod"
                        | "isnan"
                        | "isinf"
                        | "refract"
                ) {
                    return self.composed(name, args, span);
                }
                if let Some(intrinsic) = Intrinsic::from_name(name) {
                    let ids = args
                        .iter()
                        .map(|arg| self.value(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let types = ids.iter().map(|id| self.ty(*id)).collect::<Vec<_>>();
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
                Err(semantic(
                    span,
                    format!("`{name}` is not a builtin, intrinsic, or function defined above"),
                ))
            }
        }
    }
}
