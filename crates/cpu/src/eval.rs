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

/// Interpreter state for one thread. All frames live on one value stack:
/// a frame holds its arguments followed by its instruction values, so
/// calls allocate nothing.
pub(crate) struct Machine<'m> {
    module: &'m Validated,
    invocation: &'m Invocation,
    stack: Vec<Value>,
}

impl<'m> Machine<'m> {
    pub(crate) fn new(module: &'m Validated, invocation: &'m Invocation) -> Self {
        Self {
            module,
            invocation,
            stack: Vec::with_capacity(module.entry.body.len() * 2),
        }
    }

    /// Runs the entry point for one pixel and returns its results.
    pub(crate) fn run_entry(
        &mut self,
        scope: &Scope<'_>,
        x: u32,
        y: u32,
    ) -> Result<impl Iterator<Item = &Value>, ExecutionError> {
        self.stack.clear();
        let entry = &self.module.entry;
        run(
            self.module,
            self.invocation,
            scope,
            entry,
            0,
            &mut self.stack,
            x,
            y,
        )?;
        let stack = &self.stack;
        Ok(entry.results.iter().map(move |id| &stack[id.0 as usize]))
    }
}

/// Executes `function` whose arguments already sit at `stack[base..]`.
/// Instruction values follow the arguments.
#[allow(clippy::too_many_arguments)]
fn run(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
    function: &Function,
    base: usize,
    stack: &mut Vec<Value>,
    x: u32,
    y: u32,
) -> Result<(), ExecutionError> {
    let arithmetic = |error| ExecutionError::Arithmetic { x, y, error };
    let values = base + function.args.len();
    for instruction in &function.body {
        let at = |id: &orla_ir::ValueId| stack[values + id.0 as usize];
        let value = match &instruction.op {
            Op::Const(value) => *value,
            Op::Param(index) => invocation.params[*index as usize],
            Op::Arg(index) => stack[base + *index as usize],
            Op::Unary(UnaryOp::Neg, a) => negate(at(a)).map_err(arithmetic)?,
            Op::Binary(op, a, b) => binary(*op, at(a), at(b)).map_err(arithmetic)?,
            Op::Construct(parts) => {
                let mut components = [0.0; 4];
                for (slot, part) in components.iter_mut().zip(parts) {
                    *slot = float(at(part));
                }
                from_parts(components, instruction.ty)
            }
            Op::Extract(vector, component) => {
                Value::Float(components(at(vector))[*component as usize])
            }
            Op::Call(callee, call_args) => {
                let callee = &module.functions[*callee as usize];
                let frame = stack.len();
                for arg in call_args {
                    let value = stack[values + arg.0 as usize];
                    stack.push(value);
                }
                run(module, invocation, scope, callee, frame, stack, x, y)?;
                let result = stack[frame + callee.args.len() + callee.results[0].0 as usize];
                stack.truncate(frame);
                result
            }
            Op::Intrinsic(intrinsic, call_args) => {
                let mut args = [Value::Float(0.0); 3];
                for (slot, arg) in args.iter_mut().zip(call_args) {
                    *slot = at(arg);
                }
                intrinsic_value(*intrinsic, &args[..call_args.len()])
            }
            Op::Coord => Value::Vec2([x as f32, y as f32]),
            Op::Domain => Value::Vec2([
                invocation.domain.width as f32,
                invocation.domain.height as f32,
            ]),
            Op::Time => Value::Float(invocation.time),
            Op::ImageSize(input) => {
                let (width, height) = scope.image_size(invocation.inputs[*input as usize])?;
                Value::Vec2([width as f32, height as f32])
            }
            Op::ImageSample(input) => {
                let image = invocation.inputs[*input as usize];
                Value::Vec4(fetch_clamped(scope, image, x as f32, y as f32)?)
            }
            Op::ImageFetch(input, coord) => {
                let image = invocation.inputs[*input as usize];
                let Value::Vec2([fx, fy]) = at(coord) else {
                    unreachable!("validation guarantees a Vec2 coordinate")
                };
                Value::Vec4(fetch_clamped(scope, image, fx, fy)?)
            }
        };
        stack.push(value);
    }
    Ok(())
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
        Value::Vec3([x, ..]) => x,
        Value::Vec4([x, ..]) => x,
    }
}

/// Components padded to four; scalars broadcast.
fn components(value: Value) -> [f32; 4] {
    match value {
        Value::Float(v) => [v, v, v, v],
        Value::Int(v) => [v as f32; 4],
        Value::Vec2([x, y]) => [x, y, 0.0, 0.0],
        Value::Vec3([x, y, z]) => [x, y, z, 0.0],
        Value::Vec4(v) => v,
    }
}

fn arity(value: Value) -> usize {
    value.ty().components().unwrap_or(1) as usize
}

fn from_parts(parts: [f32; 4], ty: Type) -> Value {
    match ty {
        Type::Vec2 => Value::Vec2([parts[0], parts[1]]),
        Type::Vec3 => Value::Vec3([parts[0], parts[1], parts[2]]),
        Type::Vec4 => Value::Vec4(parts),
        Type::Int => Value::Int(parts[0] as i32),
        Type::Float | Type::Image => Value::Float(parts[0]),
    }
}

fn negate(value: Value) -> Result<Value, ArithmeticError> {
    Ok(match value {
        Value::Int(v) => Value::Int(v.checked_neg().ok_or(ArithmeticError::Overflow)?),
        other => map1(other, |v| -v),
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
    Ok(map2(a, b, apply))
}

/// Applies `f` componentwise; the result takes the vector type of either
/// operand (scalars broadcast).
fn map2(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    let ty = if arity(a) >= arity(b) { a.ty() } else { b.ty() };
    let (a, b) = (components(a), components(b));
    from_parts(std::array::from_fn(|i| f(a[i], b[i])), ty)
}

fn map1(value: Value, f: impl Fn(f32) -> f32) -> Value {
    from_parts(components(value).map(f), value.ty())
}

fn dot(a: Value, b: Value) -> f32 {
    let (n, a, b) = (arity(a), components(a), components(b));
    (0..n).map(|i| a[i] * b[i]).sum()
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
        (Intrinsic::Exp, [a]) => map1(*a, f32::exp),
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
        (Intrinsic::Dot, [a, b]) => Value::Float(dot(*a, *b)),
        (Intrinsic::Cross, [a, b]) => {
            let (a, b) = (components(*a), components(*b));
            Value::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])
        }
        (Intrinsic::Length, [a]) => Value::Float(dot(*a, *a).sqrt()),
        (Intrinsic::Normalize, [a]) => {
            let length = dot(*a, *a).sqrt();
            if length > 0.0 {
                map1(*a, |v| v / length)
            } else {
                map1(*a, |_| 0.0)
            }
        }
        (Intrinsic::Pow, [a, b]) => map2(*a, *b, f32::powf),
        (Intrinsic::Step, [edge, x]) => {
            map2(*edge, *x, |edge, x| if x >= edge { 1.0 } else { 0.0 })
        }
        (Intrinsic::Smoothstep, [low, high, x]) => {
            let t = map2(
                map2(*x, *low, |x, low| x - low),
                map2(*high, *low, |h, l| h - l),
                |a, b| (a / b).clamp(0.0, 1.0),
            );
            map1(t, |t| t * t * (3.0 - 2.0 * t))
        }
        (Intrinsic::Reflect, [incident, normal]) => {
            let scale = 2.0 * dot(*normal, *incident);
            map2(*incident, *normal, |i, n| i - scale * n)
        }
        _ => unreachable!("validation guarantees intrinsic signatures"),
    }
}
