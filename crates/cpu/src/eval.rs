//! The IR interpreter.

use std::fmt;

use orla_abi::{ImageHandle, Invocation, Scope, Type, Value};
use orla_ir::{BinaryOp, Function, Intrinsic, Op, UnaryOp, Validated};

use crate::ExecutionError;

/// Integer arithmetic that cannot produce a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Integer division or remainder by zero.
    DivisionByZero,
    /// Integer result does not fit in 32 bits.
    Overflow,
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => f.write_str("integer division by zero"),
            Self::Overflow => f.write_str("integer overflow"),
        }
    }
}

pub(crate) struct Machine<'m> {
    module: &'m Validated,
    invocation: &'m Invocation,
    entry_values: Vec<Value>,
}

impl<'m> Machine<'m> {
    pub(crate) fn new(module: &'m Validated, invocation: &'m Invocation) -> Self {
        Self {
            module,
            invocation,
            entry_values: Vec::with_capacity(module.entry.body.len()),
        }
    }

    pub(crate) fn run_entry(
        &mut self,
        scope: &mut Scope<'_>,
        x: u32,
        y: u32,
    ) -> Result<Vec<Value>, ExecutionError> {
        let mut values = std::mem::take(&mut self.entry_values);
        values.clear();
        let entry = &self.module.entry;
        let result = self.run(scope, entry, &[], x, y, &mut values);
        self.entry_values = values;
        let values = &self.entry_values;
        result.map(|()| {
            entry
                .results
                .iter()
                .map(|id| values[id.0 as usize])
                .collect()
        })
    }

    fn call(
        &self,
        scope: &mut Scope<'_>,
        function: &Function,
        args: &[Value],
        x: u32,
        y: u32,
    ) -> Result<Value, ExecutionError> {
        let mut values = Vec::with_capacity(function.body.len());
        self.run(scope, function, args, x, y, &mut values)?;
        Ok(values[function.results[0].0 as usize])
    }

    fn run(
        &self,
        scope: &mut Scope<'_>,
        function: &Function,
        args: &[Value],
        x: u32,
        y: u32,
        values: &mut Vec<Value>,
    ) -> Result<(), ExecutionError> {
        let arithmetic = |error| ExecutionError::Arithmetic { x, y, error };
        for instruction in &function.body {
            let value = match &instruction.op {
                Op::Const(value) => *value,
                Op::Param(index) => self.invocation.params[*index as usize],
                Op::Arg(index) => args[*index as usize],
                Op::Unary(UnaryOp::Neg, a) => negate(values[a.0 as usize]).map_err(arithmetic)?,
                Op::Binary(op, a, b) => {
                    binary(*op, values[a.0 as usize], values[b.0 as usize]).map_err(arithmetic)?
                }
                Op::Construct(parts) => {
                    let mut components = [0.0; 4];
                    for (slot, part) in components.iter_mut().zip(parts) {
                        *slot = float(values[part.0 as usize]);
                    }
                    match instruction.ty {
                        Type::Vec2 => Value::Vec2([components[0], components[1]]),
                        _ => Value::Vec4(components),
                    }
                }
                Op::Extract(vector, component) => {
                    Value::Float(components(values[vector.0 as usize])[*component as usize])
                }
                Op::Call(callee, call_args) => {
                    let call_args: Vec<Value> =
                        call_args.iter().map(|id| values[id.0 as usize]).collect();
                    let callee = &self.module.functions[*callee as usize];
                    self.call(scope, callee, &call_args, x, y)?
                }
                Op::Intrinsic(intrinsic, call_args) => {
                    let call_args: Vec<Value> =
                        call_args.iter().map(|id| values[id.0 as usize]).collect();
                    intrinsic_value(*intrinsic, &call_args)
                }
                Op::Coord => Value::Vec2([x as f32, y as f32]),
                Op::Domain => Value::Vec2([
                    self.invocation.domain.width as f32,
                    self.invocation.domain.height as f32,
                ]),
                Op::ImageSize(input) => {
                    let (width, height) = scope.image_size(self.input(*input))?;
                    Value::Vec2([width as f32, height as f32])
                }
                Op::ImageSample(input) => {
                    let image = self.input(*input);
                    Value::Vec4(fetch_clamped(scope, image, x as f32, y as f32)?)
                }
                Op::ImageFetch(input, coord) => {
                    let image = self.input(*input);
                    let Value::Vec2([fx, fy]) = values[coord.0 as usize] else {
                        unreachable!("validation guarantees a Vec2 coordinate")
                    };
                    Value::Vec4(fetch_clamped(scope, image, fx, fy)?)
                }
            };
            values.push(value);
        }
        Ok(())
    }

    fn input(&self, index: u32) -> ImageHandle {
        self.invocation.inputs[index as usize]
    }
}

fn fetch_clamped(
    scope: &Scope<'_>,
    image: ImageHandle,
    fx: f32,
    fy: f32,
) -> Result<[f32; 4], ExecutionError> {
    let (width, height) = scope.image_size(image)?;
    let clamp = |value: f32, limit: u32| -> u32 {
        if value.is_nan() {
            0
        } else {
            value.floor().clamp(0.0, (limit - 1) as f32) as u32
        }
    };
    Ok(scope.image_read(image, clamp(fx, width), clamp(fy, height))?)
}

fn float(value: Value) -> f32 {
    match value {
        Value::Float(v) => v,
        Value::Int(v) => v as f32,
        Value::Vec2([x, _]) => x,
        Value::Vec4([x, ..]) => x,
    }
}

fn components(value: Value) -> [f32; 4] {
    match value {
        Value::Float(v) => [v, v, v, v],
        Value::Int(v) => [v as f32; 4],
        Value::Vec2([x, y]) => [x, y, 0.0, 0.0],
        Value::Vec4(v) => v,
    }
}

fn negate(value: Value) -> Result<Value, ArithmeticError> {
    Ok(match value {
        Value::Float(v) => Value::Float(-v),
        Value::Int(v) => Value::Int(v.checked_neg().ok_or(ArithmeticError::Overflow)?),
        Value::Vec2([x, y]) => Value::Vec2([-x, -y]),
        Value::Vec4([x, y, z, w]) => Value::Vec4([-x, -y, -z, -w]),
    })
}

fn binary(op: BinaryOp, a: Value, b: Value) -> Result<Value, ArithmeticError> {
    if let (Value::Int(a), Value::Int(b)) = (a, b) {
        let result = match op {
            BinaryOp::Add => a.checked_add(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Sub => a.checked_sub(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Mul => a.checked_mul(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Div => {
                if b == 0 {
                    return Err(ArithmeticError::DivisionByZero);
                }
                a.checked_div(b).ok_or(ArithmeticError::Overflow)?
            }
        };
        return Ok(Value::Int(result));
    }
    let apply = |a: f32, b: f32| match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
    };
    Ok(match (a, b) {
        (Value::Float(a), Value::Float(b)) => Value::Float(apply(a, b)),
        (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([apply(a[0], b[0]), apply(a[1], b[1])]),
        (Value::Vec2(a), Value::Float(b)) => Value::Vec2([apply(a[0], b), apply(a[1], b)]),
        (Value::Float(a), Value::Vec2(b)) => Value::Vec2([apply(a, b[0]), apply(a, b[1])]),
        (Value::Vec4(a), Value::Vec4(b)) => Value::Vec4(std::array::from_fn(|i| apply(a[i], b[i]))),
        (Value::Vec4(a), Value::Float(b)) => Value::Vec4(a.map(|a| apply(a, b))),
        (Value::Float(a), Value::Vec4(b)) => Value::Vec4(b.map(|b| apply(a, b))),
        _ => unreachable!("validation rejects mixed operand types"),
    })
}

fn map1(value: Value, f: impl Fn(f32) -> f32) -> Value {
    match value {
        Value::Float(v) => Value::Float(f(v)),
        Value::Vec2(v) => Value::Vec2(v.map(&f)),
        Value::Vec4(v) => Value::Vec4(v.map(&f)),
        Value::Int(v) => Value::Int(v),
    }
}

fn map2(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    match (a, b) {
        (Value::Float(a), Value::Float(b)) => Value::Float(f(a, b)),
        (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([f(a[0], b[0]), f(a[1], b[1])]),
        (Value::Vec4(a), Value::Vec4(b)) => Value::Vec4(std::array::from_fn(|i| f(a[i], b[i]))),
        _ => unreachable!("validation rejects mixed intrinsic operands"),
    }
}

fn intrinsic_value(intrinsic: Intrinsic, args: &[Value]) -> Value {
    match (intrinsic, args) {
        (Intrinsic::Floor, [a]) => map1(*a, f32::floor),
        (Intrinsic::Fract, [a]) => map1(*a, |v| v - v.floor()),
        (Intrinsic::Abs, [Value::Int(a)]) => Value::Int(a.wrapping_abs()),
        (Intrinsic::Abs, [a]) => map1(*a, f32::abs),
        (Intrinsic::Sqrt, [a]) => map1(*a, f32::sqrt),
        (Intrinsic::Sin, [a]) => map1(*a, f32::sin),
        (Intrinsic::Cos, [a]) => map1(*a, f32::cos),
        (Intrinsic::Min, [Value::Int(a), Value::Int(b)]) => Value::Int((*a).min(*b)),
        (Intrinsic::Max, [Value::Int(a), Value::Int(b)]) => Value::Int((*a).max(*b)),
        (Intrinsic::Min, [a, b]) => map2(*a, *b, f32::min),
        (Intrinsic::Max, [a, b]) => map2(*a, *b, f32::max),
        (Intrinsic::Clamp, [Value::Int(v), Value::Int(low), Value::Int(high)]) => {
            Value::Int((*v).max(*low).min(*high))
        }
        (Intrinsic::Clamp, [v, low, high]) => {
            let low = map2(*v, *low, f32::max);
            map2(low, *high, f32::min)
        }
        (Intrinsic::Mix, [a, b, Value::Float(t)]) => map2(*a, *b, |a, b| a + (b - a) * t),
        (Intrinsic::ToFloat, [Value::Int(v)]) => Value::Float(*v as f32),
        (Intrinsic::ToInt, [Value::Float(v)]) => Value::Int(*v as i32),
        _ => unreachable!("validation guarantees intrinsic signatures"),
    }
}
