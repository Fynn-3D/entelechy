#![warn(missing_docs)]
//! Orla IR (L3).
//!
//! The IR is the true internal representation of a Kernel. It is typed,
//! SSA-like (every instruction defines exactly one value and may only refer
//! to values defined before it in the same or an enclosing block),
//! structured (control flow is `if` and bounded `loop` expressions, never
//! jumps), deterministic, inspectable, and independent of any backend. Host
//! access appears only as explicit operations, so a validator can prove
//! which capabilities a Kernel needs before it ever runs.

mod display;
mod validate;

use std::fmt;
use std::str::FromStr;

pub use orla_abi::{Capabilities, Capability, Type, Value};
pub use validate::{Validated, ValidationError, validate};

/// Stable identity of a Kernel: a dotted name and a version, e.g.
/// `image.grayscale@1`. Graphs bind to this, never to source paths.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId {
    name: String,
    version: u32,
}

impl KernelId {
    /// Builds an id from its parts, checking the name shape.
    pub fn new(name: &str, version: u32) -> Result<Self, InvalidKernelId> {
        let valid_segment = |segment: &str| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                && !segment.starts_with(|c: char| c.is_ascii_digit())
        };
        if name.split('.').all(valid_segment) {
            Ok(Self {
                name: name.to_owned(),
                version,
            })
        } else {
            Err(InvalidKernelId(format!("{name}@{version}")))
        }
    }

    /// The dotted name without the version.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The version.
    pub fn version(&self) -> u32 {
        self.version
    }
}

impl fmt::Display for KernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// The text is not a valid `name@version` Kernel id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidKernelId(pub String);

impl fmt::Display for InvalidKernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is not a kernel id of the form `dotted.name@version`",
            self.0
        )
    }
}

impl std::error::Error for InvalidKernelId {}

impl FromStr for KernelId {
    type Err = InvalidKernelId;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (name, version) = text
            .rsplit_once('@')
            .ok_or_else(|| InvalidKernelId(text.to_owned()))?;
        let version = version
            .parse::<u32>()
            .map_err(|_| InvalidKernelId(text.to_owned()))?;
        Self::new(name, version)
    }
}

/// A named, typed input or output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    /// Port name.
    pub name: String,
    /// Port type.
    pub ty: Type,
}

/// A named parameter with a default value.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Parameter name.
    pub name: String,
    /// Default value; its type is the parameter type.
    pub default: Value,
}

/// The stable contract between a Kernel and the runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    /// Kernel identity.
    pub id: KernelId,
    /// Capabilities the Kernel declares.
    pub requires: Capabilities,
    /// Inputs in declaration order.
    pub inputs: Vec<Port>,
    /// Parameters in declaration order.
    pub params: Vec<Param>,
    /// Outputs in declaration order.
    pub outputs: Vec<Port>,
}

/// Index of a value within a function. Ids are assigned in pre-order over
/// the function's instructions, nested blocks included, so `ValueId(n)` is
/// also the value's slot in a flat frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Binary operations on scalars, vectors, and truth values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `a < b`
    Less,
    /// `a <= b`
    LessEqual,
    /// `a > b`
    Greater,
    /// `a >= b`
    GreaterEqual,
    /// `a == b`
    Equal,
    /// `a != b`
    NotEqual,
    /// `a && b`
    And,
    /// `a || b`
    Or,
}

impl BinaryOp {
    /// Result type of `a op b`, or `None` when the operands cannot combine.
    /// Arithmetic: vectors combine componentwise with vectors of the same
    /// size or with a float on either side; integers only with integers.
    /// Comparisons take two scalars of one type and give `Bool`; equality
    /// also compares `Bool`s. `&&` and `||` take `Bool`s.
    pub fn result_type(self, a: Type, b: Type) -> Option<Type> {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => match (a, b) {
                (Type::Float, Type::Float) => Some(Type::Float),
                (Type::Int, Type::Int) => Some(Type::Int),
                (Type::Vec2, Type::Vec2)
                | (Type::Vec2, Type::Float)
                | (Type::Float, Type::Vec2) => Some(Type::Vec2),
                (Type::Vec3, Type::Vec3)
                | (Type::Vec3, Type::Float)
                | (Type::Float, Type::Vec3) => Some(Type::Vec3),
                (Type::Vec4, Type::Vec4)
                | (Type::Vec4, Type::Float)
                | (Type::Float, Type::Vec4) => Some(Type::Vec4),
                _ => None,
            },
            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual => match (a, b) {
                (Type::Float, Type::Float) | (Type::Int, Type::Int) => Some(Type::Bool),
                _ => None,
            },
            Self::Equal | Self::NotEqual => match (a, b) {
                (Type::Float, Type::Float) | (Type::Int, Type::Int) | (Type::Bool, Type::Bool) => {
                    Some(Type::Bool)
                }
                _ => None,
            },
            Self::And | Self::Or => match (a, b) {
                (Type::Bool, Type::Bool) => Some(Type::Bool),
                _ => None,
            },
        }
    }

    /// Source spelling.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-a` on numbers and vectors.
    Neg,
    /// `!a` on truth values.
    Not,
}

impl UnaryOp {
    /// Result type, or `None` when the operand does not fit.
    pub fn result_type(self, a: Type) -> Option<Type> {
        match (self, a) {
            (Self::Neg, Type::Float | Type::Int | Type::Vec2 | Type::Vec3 | Type::Vec4) => Some(a),
            (Self::Not, Type::Bool) => Some(Type::Bool),
            _ => None,
        }
    }
}

/// Pure math intrinsics. Their signatures are the single source of truth
/// shared by the compiler, the validator, and every backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    /// Largest integer value not greater than the input, componentwise.
    Floor,
    /// Fractional part, componentwise.
    Fract,
    /// Absolute value, componentwise.
    Abs,
    /// Square root, componentwise.
    Sqrt,
    /// Sine, componentwise.
    Sin,
    /// Cosine, componentwise.
    Cos,
    /// Componentwise minimum.
    Min,
    /// Componentwise maximum.
    Max,
    /// `clamp(x, low, high)` componentwise.
    Clamp,
    /// `mix(a, b, t)` linear interpolation with a float `t`.
    Mix,
    /// Convert `Int` or `Bool` (as 0 or 1) to `Float`.
    ToFloat,
    /// Convert `Float` (truncating toward zero) or `Bool` (as 0 or 1) to `Int`.
    ToInt,
    /// Dot product of two vectors of the same size.
    Dot,
    /// Cross product of two `Vec3`.
    Cross,
    /// Euclidean length of a vector.
    Length,
    /// Unit vector in the same direction; the zero vector stays zero.
    Normalize,
    /// `pow(base, exponent)` componentwise; a `Float` exponent broadcasts.
    Pow,
    /// `e^x` componentwise.
    Exp,
    /// `step(edge, x)`: 1 where `x >= edge`, else 0, componentwise; a
    /// `Float` edge broadcasts.
    Step,
    /// `smoothstep(low, high, x)` componentwise; `Float` edges broadcast.
    Smoothstep,
    /// `reflect(incident, normal)` for vectors of the same size.
    Reflect,
    /// `atan2(y, x)` on floats.
    Atan2,
}

impl Intrinsic {
    /// All intrinsics with their source names.
    pub const ALL: [(Intrinsic, &'static str); 22] = [
        (Intrinsic::Floor, "floor"),
        (Intrinsic::Fract, "fract"),
        (Intrinsic::Abs, "abs"),
        (Intrinsic::Sqrt, "sqrt"),
        (Intrinsic::Sin, "sin"),
        (Intrinsic::Cos, "cos"),
        (Intrinsic::Min, "min"),
        (Intrinsic::Max, "max"),
        (Intrinsic::Clamp, "clamp"),
        (Intrinsic::Mix, "mix"),
        (Intrinsic::ToFloat, "float"),
        (Intrinsic::ToInt, "int"),
        (Intrinsic::Dot, "dot"),
        (Intrinsic::Cross, "cross"),
        (Intrinsic::Length, "length"),
        (Intrinsic::Normalize, "normalize"),
        (Intrinsic::Pow, "pow"),
        (Intrinsic::Exp, "exp"),
        (Intrinsic::Step, "step"),
        (Intrinsic::Smoothstep, "smoothstep"),
        (Intrinsic::Reflect, "reflect"),
        (Intrinsic::Atan2, "atan2"),
    ];

    /// Source spelling.
    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(intrinsic, _)| *intrinsic == self)
            .map(|(_, name)| *name)
            .unwrap_or("?")
    }

    /// Looks up an intrinsic by source name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|(_, candidate)| *candidate == name)
            .map(|(intrinsic, _)| *intrinsic)
    }

    /// Result type for the given argument types, or `None` if the call is
    /// ill-typed.
    pub fn result_type(self, args: &[Type]) -> Option<Type> {
        let float_like =
            |ty: Type| matches!(ty, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4);
        let vector = |ty: Type| ty.components().is_some();
        match (self, args) {
            (
                Self::Floor
                | Self::Fract
                | Self::Abs
                | Self::Sqrt
                | Self::Sin
                | Self::Cos
                | Self::Exp,
                [a],
            ) if float_like(*a) => Some(*a),
            (Self::Abs, [Type::Int]) => Some(Type::Int),
            (Self::Min | Self::Max, [a, b]) if a == b && (float_like(*a) || *a == Type::Int) => {
                Some(*a)
            }
            (Self::Clamp, [a, b, c]) if a == b && b == c && (float_like(*a) || *a == Type::Int) => {
                Some(*a)
            }
            (Self::Mix, [a, b, Type::Float]) if a == b && float_like(*a) => Some(*a),
            (Self::ToFloat, [Type::Int | Type::Bool]) => Some(Type::Float),
            (Self::ToInt, [Type::Float | Type::Bool]) => Some(Type::Int),
            (Self::Dot, [a, b]) if a == b && vector(*a) => Some(Type::Float),
            (Self::Cross, [Type::Vec3, Type::Vec3]) => Some(Type::Vec3),
            (Self::Length, [a]) if vector(*a) => Some(Type::Float),
            (Self::Normalize, [a]) if vector(*a) => Some(*a),
            (Self::Reflect, [a, b]) if a == b && vector(*a) => Some(*a),
            (Self::Pow, [a, b]) if (a == b || *b == Type::Float) && float_like(*a) => Some(*a),
            (Self::Step, [a, b]) if (a == b || *a == Type::Float) && float_like(*b) => Some(*b),
            (Self::Smoothstep, [a, b, c])
                if a == b && (b == c || *a == Type::Float) && float_like(*c) =>
            {
                Some(*c)
            }
            (Self::Atan2, [Type::Float, Type::Float]) => Some(Type::Float),
            _ => None,
        }
    }
}

/// A nested block: instructions and the value the block yields.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Instructions in order.
    pub body: Vec<Instruction>,
    /// The value the block yields; defined in the block or an enclosing one.
    pub result: ValueId,
}

/// Largest iteration bound a `loop` may declare.
pub const MAX_LOOP_COUNT: u32 = 65_536;

/// One SSA operation. Host access is explicit and named.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// A literal.
    Const(Value),
    /// The Kernel parameter at the given manifest index.
    Param(u32),
    /// The current function's argument at the given index.
    Arg(u32),
    /// Unary operation.
    Unary(UnaryOp, ValueId),
    /// Binary operation.
    Binary(BinaryOp, ValueId, ValueId),
    /// Build a vector from float components; the instruction type says which.
    Construct(Vec<ValueId>),
    /// Extract a float component from a vector.
    Extract(ValueId, u8),
    /// Call a Kernel-defined function by index with arguments.
    Call(u32, Vec<ValueId>),
    /// A pure intrinsic.
    Intrinsic(Intrinsic, Vec<ValueId>),
    /// Evaluate `then` or `otherwise` depending on a `Bool` and yield that
    /// block's result. Only the taken block runs.
    If {
        /// The condition.
        condition: ValueId,
        /// Block taken when the condition holds.
        then: Block,
        /// Block taken otherwise.
        otherwise: Block,
    },
    /// Run `body` at most `count` times, threading one carried value: it
    /// starts as `init`, each iteration reads it through [`Op::Carried`] and
    /// yields its replacement. Before each iteration `until` is evaluated
    /// with the current carried value; when it holds, the loop stops. The
    /// instruction's value is the final carried value.
    Loop {
        /// Compile-time iteration bound, at most [`MAX_LOOP_COUNT`].
        count: u32,
        /// Initial carried value.
        init: ValueId,
        /// Optional exit test, yielding `Bool`.
        until: Option<Block>,
        /// The body, yielding the next carried value.
        body: Block,
    },
    /// The carried value of the innermost enclosing loop.
    Carried,
    /// The iteration index of the innermost enclosing loop as `Int`.
    Index,
    /// The current output pixel as `Vec2`. Entry point only.
    Coord,
    /// The output domain size as `Vec2`. Entry point only.
    Domain,
    /// Session time in seconds as `Float`. Context: `time.read`.
    Time,
    /// Size of an input image as `Vec2`. Host call: `image.read`.
    ImageSize(u32),
    /// Pixel of an input image at the current coordinate as `Vec4`.
    /// Host call: `image.read`. Entry point only.
    ImageSample(u32),
    /// Pixel of an input image at a `Vec2` coordinate (clamped to the image)
    /// as `Vec4`. Host call: `image.read`.
    ImageFetch(u32, ValueId),
}

/// An instruction: a typed value defined by an operation.
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    /// The value this instruction defines; its pre-order position.
    pub id: ValueId,
    /// The type of the defined value.
    pub ty: Type,
    /// How the value is computed.
    pub op: Op,
}

/// A function: arguments, a top-level block, and results. Function `i` may
/// only call functions with index less than `i`, which rules out recursion
/// by construction.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    /// Name for inspection and diagnostics.
    pub name: String,
    /// Argument types.
    pub args: Vec<Type>,
    /// Top-level instructions.
    pub body: Vec<Instruction>,
    /// Values returned, in order.
    pub results: Vec<ValueId>,
}

impl Function {
    /// Number of value slots the function defines, nested blocks included.
    pub fn slot_count(&self) -> usize {
        fn count(instructions: &[Instruction]) -> usize {
            instructions
                .iter()
                .map(|instruction| {
                    1 + match &instruction.op {
                        Op::If {
                            then, otherwise, ..
                        } => count(&then.body) + count(&otherwise.body),
                        Op::Loop { until, body, .. } => {
                            until.as_ref().map_or(0, |block| count(&block.body)) + count(&body.body)
                        }
                        _ => 0,
                    }
                })
                .sum()
        }
        count(&self.body)
    }
}

/// A complete compiled Kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// The manifest.
    pub manifest: Manifest,
    /// Helper functions in dependency order.
    pub functions: Vec<Function>,
    /// The entry point, executed once per output pixel. Its results map to
    /// `manifest.outputs` in order; image outputs receive the `Vec4` result.
    pub entry: Function,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ids_parse_and_display() {
        let id: KernelId = "image.grayscale@2".parse().unwrap();
        assert_eq!(id.name(), "image.grayscale");
        assert_eq!(id.version(), 2);
        assert_eq!(id.to_string(), "image.grayscale@2");
        assert!("Grayscale@1".parse::<KernelId>().is_err());
        assert!("image.grayscale".parse::<KernelId>().is_err());
        assert!("image..blur@1".parse::<KernelId>().is_err());
        assert!("image.1blur@1".parse::<KernelId>().is_err());
    }

    #[test]
    fn operator_typing_is_explicit() {
        assert_eq!(
            BinaryOp::Mul.result_type(Type::Vec4, Type::Float),
            Some(Type::Vec4)
        );
        assert_eq!(BinaryOp::Add.result_type(Type::Int, Type::Float), None);
        assert_eq!(BinaryOp::Add.result_type(Type::Vec2, Type::Vec4), None);
        assert_eq!(
            BinaryOp::Less.result_type(Type::Float, Type::Float),
            Some(Type::Bool)
        );
        assert_eq!(BinaryOp::Less.result_type(Type::Vec2, Type::Vec2), None);
        assert_eq!(
            BinaryOp::Equal.result_type(Type::Bool, Type::Bool),
            Some(Type::Bool)
        );
        assert_eq!(BinaryOp::And.result_type(Type::Int, Type::Int), None);
        assert_eq!(UnaryOp::Not.result_type(Type::Float), None);
        assert_eq!(
            Intrinsic::Mix.result_type(&[Type::Vec4, Type::Vec4, Type::Float]),
            Some(Type::Vec4)
        );
        assert_eq!(
            Intrinsic::Mix.result_type(&[Type::Vec4, Type::Vec4, Type::Int]),
            None
        );
        assert_eq!(
            Intrinsic::ToInt.result_type(&[Type::Float]),
            Some(Type::Int)
        );
        assert_eq!(Intrinsic::from_name("floor"), Some(Intrinsic::Floor));
    }
}
