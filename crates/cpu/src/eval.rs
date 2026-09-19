//! The IR interpreter.

use std::fmt;

use orla_abi::{ImageHandle, Invocation, Scope, Type, Value};
use orla_ir::{BinaryOp, Block, Function, Instruction, Intrinsic, Op, UnaryOp, Validated};

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
/// a frame holds its arguments followed by one slot per instruction of the
/// function, nested blocks included, so calls and loops allocate nothing.
pub(crate) struct Machine<'m> {
    module: &'m Validated,
    invocation: &'m Invocation,
    stack: Vec<Value>,
    entry_slots: usize,
}

impl<'m> Machine<'m> {
    pub(crate) fn new(module: &'m Validated, invocation: &'m Invocation) -> Self {
        let entry_slots = module.entry.slot_count();
        Self {
            module,
            invocation,
            stack: Vec::with_capacity(entry_slots * 2),
            entry_slots,
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
        self.stack.resize(self.entry_slots, Value::Float(0.0));
        let entry = &self.module.entry;
        let mut frame = Frame {
            module: self.module,
            invocation: self.invocation,
            scope,
            x,
            y,
            base: 0,
            values: 0,
            loops: Vec::new(),
        };
        frame.run(entry, &mut self.stack)?;
        let stack = &self.stack;
        Ok(entry.results.iter().map(move |id| &stack[id.0 as usize]))
    }
}

struct LoopState {
    carried: Value,
    index: i32,
}

/// One activation of a function.
struct Frame<'a, 'm> {
    module: &'m Validated,
    invocation: &'m Invocation,
    scope: &'a Scope<'a>,
    x: u32,
    y: u32,
    /// Where the arguments start.
    base: usize,
    /// Where the instruction slots start.
    values: usize,
    loops: Vec<LoopState>,
}

impl<'m> Frame<'_, 'm> {
    fn run(
        &mut self,
        function: &'m Function,
        stack: &mut Vec<Value>,
    ) -> Result<(), ExecutionError> {
        self.instructions(&function.body, stack)
    }

    fn instructions(
        &mut self,
        instructions: &'m [Instruction],
        stack: &mut Vec<Value>,
    ) -> Result<(), ExecutionError> {
        for instruction in instructions {
            let value = self.instruction(instruction, stack)?;
            stack[self.values + instruction.id.0 as usize] = value;
        }
        Ok(())
    }

    fn block(&mut self, block: &'m Block, stack: &mut Vec<Value>) -> Result<Value, ExecutionError> {
        self.instructions(&block.body, stack)?;
        Ok(stack[self.values + block.result.0 as usize])
    }

    fn instruction(
        &mut self,
        instruction: &'m Instruction,
        stack: &mut Vec<Value>,
    ) -> Result<Value, ExecutionError> {
        let (x, y) = (self.x, self.y);
        let arithmetic = |error| ExecutionError::Arithmetic { x, y, error };
        let values = self.values;
        let at = |stack: &Vec<Value>, id: &orla_ir::ValueId| stack[values + id.0 as usize];
        Ok(match &instruction.op {
            Op::Const(value) => *value,
            Op::Param(index) => self.invocation.params[*index as usize],
            Op::Arg(index) => stack[self.base + *index as usize],
            Op::Unary(UnaryOp::Neg, a) => negate(at(stack, a)).map_err(arithmetic)?,
            Op::Unary(UnaryOp::Not, a) => match at(stack, a) {
                Value::Bool(value) => Value::Bool(!value),
                _ => unreachable!("validation guarantees a Bool operand"),
            },
            Op::Binary(op, a, b) => binary(*op, at(stack, a), at(stack, b)).map_err(arithmetic)?,
            Op::Construct(parts) => {
                let mut components = [0.0; 4];
                for (slot, part) in components.iter_mut().zip(parts) {
                    *slot = float(at(stack, part));
                }
                from_parts(components, instruction.ty)
            }
            Op::Extract(vector, component) => {
                Value::Float(components(at(stack, vector))[*component as usize])
            }
            Op::Call(callee, call_args) => {
                let callee = &self.module.functions[*callee as usize];
                let frame_base = stack.len();
                for arg in call_args {
                    let value = stack[values + arg.0 as usize];
                    stack.push(value);
                }
                let frame_values = stack.len();
                stack.resize(frame_values + callee.slot_count(), Value::Float(0.0));
                let mut frame = Frame {
                    module: self.module,
                    invocation: self.invocation,
                    scope: self.scope,
                    x,
                    y,
                    base: frame_base,
                    values: frame_values,
                    loops: Vec::new(),
                };
                frame.run(callee, stack)?;
                let result = stack[frame_values + callee.results[0].0 as usize];
                stack.truncate(frame_base);
                result
            }
            Op::Intrinsic(intrinsic, call_args) => {
                let mut args = [Value::Float(0.0); 3];
                for (slot, arg) in args.iter_mut().zip(call_args) {
                    *slot = at(stack, arg);
                }
                intrinsic_value(*intrinsic, &args[..call_args.len()])
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => match at(stack, condition) {
                Value::Bool(true) => self.block(then, stack)?,
                Value::Bool(false) => self.block(otherwise, stack)?,
                _ => unreachable!("validation guarantees a Bool condition"),
            },
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                self.loops.push(LoopState {
                    carried: at(stack, init),
                    index: 0,
                });
                let outcome: Result<(), ExecutionError> = (|| {
                    for index in 0..*count {
                        self.loops.last_mut().expect("loop is open").index = index as i32;
                        if let Some(until) = until
                            && self.block(until, stack)? == Value::Bool(true)
                        {
                            break;
                        }
                        let next = self.block(body, stack)?;
                        self.loops.last_mut().expect("loop is open").carried = next;
                    }
                    Ok(())
                })();
                let state = self.loops.pop().expect("loop is open");
                outcome?;
                state.carried
            }
            Op::Carried => {
                self.loops
                    .last()
                    .expect("validation guarantees a loop")
                    .carried
            }
            Op::Index => Value::Int(
                self.loops
                    .last()
                    .expect("validation guarantees a loop")
                    .index,
            ),
            Op::Coord => Value::Vec2([x as f32, y as f32]),
            Op::Domain => Value::Vec2([
                self.invocation.domain.width as f32,
                self.invocation.domain.height as f32,
            ]),
            Op::Time => Value::Float(self.invocation.time),
            Op::ImageSize(input) => {
                let (width, height) = self
                    .scope
                    .image_size(self.invocation.inputs[*input as usize])?;
                Value::Vec2([width as f32, height as f32])
            }
            Op::ImageSample(input) => {
                let image = self.invocation.inputs[*input as usize];
                Value::Vec4(fetch_clamped(self.scope, image, x as f32, y as f32)?)
            }
            Op::ImageFetch(input, coord) => {
                let image = self.invocation.inputs[*input as usize];
                let Value::Vec2([fx, fy]) = at(stack, coord) else {
                    unreachable!("validation guarantees a Vec2 coordinate")
                };
                Value::Vec4(fetch_clamped(self.scope, image, fx, fy)?)
            }
        })
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
        Value::Bool(v) => v as u8 as f32,
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
        Value::Bool(v) => [v as u8 as f32; 4],
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
        Type::Bool => Value::Bool(parts[0] != 0.0),
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
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {}
        BinaryOp::And => {
            return Ok(Value::Bool(
                a == Value::Bool(true) && b == Value::Bool(true),
            ));
        }
        BinaryOp::Or => {
            return Ok(Value::Bool(
                a == Value::Bool(true) || b == Value::Bool(true),
            ));
        }
        BinaryOp::Equal => return Ok(Value::Bool(equal(a, b))),
        BinaryOp::NotEqual => return Ok(Value::Bool(!equal(a, b))),
        BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
            let ordering = match (a, b) {
                (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
                _ => unreachable!("validation guarantees scalar comparison operands"),
            };
            // NaN compares false, as in IEEE 754 and every GPU language.
            let holds = match (op, ordering) {
                (_, None) => false,
                (BinaryOp::Less, Some(o)) => o.is_lt(),
                (BinaryOp::LessEqual, Some(o)) => o.is_le(),
                (BinaryOp::Greater, Some(o)) => o.is_gt(),
                (BinaryOp::GreaterEqual, Some(o)) => o.is_ge(),
                _ => unreachable!(),
            };
            return Ok(Value::Bool(holds));
        }
    }
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
            _ => unreachable!(),
        };
        return Ok(Value::Int(result));
    }
    let apply = |a: f32, b: f32| match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        _ => unreachable!(),
    };
    Ok(map2(a, b, apply))
}

fn equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => unreachable!("validation guarantees matching equality operands"),
    }
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
        (Intrinsic::ToFloat, [Value::Bool(v)]) => Value::Float(*v as u8 as f32),
        (Intrinsic::ToInt, [Value::Float(v)]) => Value::Int(*v as i32),
        (Intrinsic::ToInt, [Value::Bool(v)]) => Value::Int(*v as i32),
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
        (Intrinsic::Atan2, [Value::Float(y), Value::Float(x)]) => Value::Float(y.atan2(*x)),
        _ => unreachable!("validation guarantees intrinsic signatures"),
    }
}
