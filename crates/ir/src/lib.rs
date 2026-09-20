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
mod fuse;
mod validate;

use std::fmt;
use std::str::FromStr;

pub use fuse::{FuseError, InputUse, fuse, input_use, samples_only};
pub use orla_abi::{
    Capabilities, Capability, PortType, Scalar, Schema, Shape, ShapeIndex, Type, Value,
};
pub use validate::{Validated, ValidationError, linear_index_type, validate};

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
    /// What the port carries.
    pub ty: PortType,
}

impl Port {
    /// The value type the entry point produces or `sample` yields for this
    /// port: its element type.
    pub fn element(&self) -> Type {
        self.ty.element
    }

    /// How the port's elements are addressed.
    pub fn shape(&self) -> Shape {
        self.ty.shape
    }
}

/// The storage layout of a value type: vectors and structs flattened to
/// scalars in declaration order. `None` for unknown structs.
pub fn schema_of(ty: Type, structs: &[StructDef]) -> Option<Schema> {
    fn push(ty: Type, structs: &[StructDef], out: &mut Vec<Scalar>) -> Option<()> {
        match ty {
            Type::Float => out.push(Scalar::Float),
            Type::Int => out.push(Scalar::Int),
            Type::Bool => out.push(Scalar::Bool),
            Type::Vec2 => out.extend([Scalar::Float; 2]),
            Type::Vec3 => out.extend([Scalar::Float; 3]),
            Type::Vec4 => out.extend([Scalar::Float; 4]),
            Type::Struct(index) => {
                for field in &structs.get(index as usize)?.fields {
                    push(field.ty, structs, out)?;
                }
            }
        }
        Some(())
    }
    let mut scalars = Vec::new();
    push(ty, structs, &mut scalars)?;
    Some(Schema::new(scalars))
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
    /// How the Kernel executes.
    pub mode: Mode,
    /// Capabilities the Kernel declares.
    pub requires: Capabilities,
    /// Inputs in declaration order.
    pub inputs: Vec<Port>,
    /// Parameters in declaration order.
    pub params: Vec<Param>,
    /// Outputs in declaration order.
    pub outputs: Vec<Port>,
    /// Names of the memories: ports that appear in `inputs` (bound by the
    /// graph, read with `at`) and in `outputs` (what the steps left there),
    /// which an ordered Kernel writes step by step with [`Op::Put`]. They
    /// come last in both lists, in one order.
    pub memories: Vec<String>,
}

/// How a Kernel executes over its domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The entry runs once per element of the output domain.
    #[default]
    Map,
    /// The entry runs once per element of the first input's domain and
    /// the results are folded pairwise by the module's `merge` function
    /// into the single element of a `Rows` output. `merge` must be
    /// associative and commutative: the fold's tree is the backend's.
    Reduce,
    /// The entry runs once per element of the first input's rows, in
    /// order, reading the previous step's result through [`Op::Carried`]
    /// (the module's `init` before the first) and writing each step's
    /// result as that element of a `Rows` output of the same count.
    Scan,
    /// The entry runs once per element of the first input's domain and
    /// yields an address into the `base` input and a value; the output is
    /// `base` with every value combined into the element at its address,
    /// as if the sources were applied one after another in element order.
    /// An address outside `base` is dropped. `base` has the output's
    /// shape and layout, and sizes the output.
    Scatter {
        /// Index of the input the output starts from.
        base: u32,
        /// How a value meets the element already there.
        combine: Combine,
    },
}

impl Mode {
    /// Stable spelling used in manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::Reduce => "reduce",
            Self::Scan => "scan",
            Self::Scatter { .. } => "scatter",
        }
    }

    /// Whether the entry maps over the first input rather than the output.
    pub fn maps_over_input(self) -> bool {
        !matches!(self, Self::Map)
    }
}

/// How a scattered value combines with the element at its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combine {
    /// Fieldwise addition; `Int` fields overflow as the backend's integers do.
    Sum,
    /// Fieldwise minimum.
    Min,
    /// Fieldwise maximum.
    Max,
    /// The value replaces the element; the last source in element order wins.
    Last,
}

impl Combine {
    /// Stable spelling used in `.orla` sources.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Last => "last",
        }
    }

    /// The spelling's meaning, if it is one.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "sum" => Some(Self::Sum),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "last" => Some(Self::Last),
            _ => None,
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scatter { base, combine } => {
                write!(f, "scatter into input {base} with {}", combine.name())
            }
            _ => f.write_str(self.name()),
        }
    }
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
    /// `a % b`: the remainder with the sign of `a`, as `/` truncates.
    Rem,
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
    /// `a & b` on integers.
    BitAnd,
    /// `a | b` on integers.
    BitOr,
    /// `a ^ b` on integers.
    BitXor,
    /// `a << b` on integers; the count is taken modulo 32.
    Shl,
    /// `a >> b` on integers, arithmetic; the count is taken modulo 32.
    Shr,
    /// `a >>> b` on integers, logical (zeros shifted in); the count is
    /// taken modulo 32.
    Ushr,
}

impl BinaryOp {
    /// Result type of `a op b`, or `None` when the operands cannot combine.
    /// Arithmetic: vectors combine componentwise with vectors of the same
    /// size or with a float on either side; integers only with integers.
    /// Comparisons take two scalars of one type and give `Bool`; equality
    /// also compares `Bool`s. `&&` and `||` take `Bool`s. The bit
    /// operators and shifts take two `Int`s.
    pub fn result_type(self, a: Type, b: Type) -> Option<Type> {
        match self {
            Self::BitAnd | Self::BitOr | Self::BitXor | Self::Shl | Self::Shr | Self::Ushr => {
                match (a, b) {
                    (Type::Int, Type::Int) => Some(Type::Int),
                    _ => None,
                }
            }
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem => match (a, b) {
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
            // Vectors compare whole: equal when every component is.
            Self::Equal | Self::NotEqual => match (a, b) {
                (Type::Float, Type::Float)
                | (Type::Int, Type::Int)
                | (Type::Bool, Type::Bool)
                | (Type::Vec2, Type::Vec2)
                | (Type::Vec3, Type::Vec3)
                | (Type::Vec4, Type::Vec4) => Some(Type::Bool),
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
            Self::Rem => "%",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::And => "&&",
            Self::Or => "||",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Ushr => ">>>",
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
    /// Smallest integer value not less than the input, componentwise.
    Ceil,
    /// Nearest integer, ties to even, componentwise.
    Round,
    /// -1, 0, or 1 by the sign, componentwise; `Int` stays `Int`.
    Sign,
    /// Natural logarithm, componentwise.
    Log,
    /// Tangent, componentwise.
    Tan,
    /// Arc sine, componentwise.
    Asin,
    /// Arc cosine, componentwise.
    Acos,
    /// Arc tangent of one value, componentwise; `atan2` takes two.
    Atan,
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
    /// Componentwise minimum; a `Float` beside a vector broadcasts.
    Min,
    /// Componentwise maximum; a `Float` beside a vector broadcasts.
    Max,
    /// `clamp(x, low, high)` componentwise; `Float` bounds broadcast.
    Clamp,
    /// `mix(a, b, t)` linear interpolation, `t` a `Float` or componentwise.
    Mix,
    /// `add_wrap(a, b)` on `Int`s: the sum modulo 2^32, never a failure.
    AddWrap,
    /// `sub_wrap(a, b)` on `Int`s: the difference modulo 2^32.
    SubWrap,
    /// `mul_wrap(a, b)` on `Int`s: the low 32 bits of the product.
    MulWrap,
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
    /// The integer part, toward zero, componentwise.
    Trunc,
    /// 2 raised to x, componentwise.
    Exp2,
    /// The base-2 logarithm, componentwise.
    Log2,
    /// The hyperbolic sine, componentwise.
    Sinh,
    /// The hyperbolic cosine, componentwise.
    Cosh,
    /// The hyperbolic tangent, componentwise.
    Tanh,
    /// `fma(a, b, c)`: a * b + c rounded once, componentwise.
    Fma,
    /// The count of one bits of an Int.
    Popcount,
    /// The count of leading zero bits of an Int, 32 for zero.
    Clz,
    /// The count of trailing zero bits of an Int, 32 for zero.
    Ctz,
    /// `rotl(x, n)`: the bits rotated left by n modulo 32.
    Rotl,
    /// `rotr(x, n)`: the bits rotated right by n modulo 32.
    Rotr,
    /// `ult(a, b)`: whether a is below b as unsigned 32-bit values.
    Ult,
    /// `udiv(a, b)`: the unsigned quotient; a zero divisor is a failure.
    Udiv,
    /// `urem(a, b)`: the unsigned remainder; a zero divisor is a failure.
    Urem,
    /// `mulhi(a, b)`: the high 32 bits of the signed 64-bit product.
    Mulhi,
    /// `umulhi(a, b)`: the high 32 bits of the unsigned 64-bit product.
    Umulhi,
}

impl Intrinsic {
    /// All intrinsics with their source names.
    pub const ALL: [(Intrinsic, &'static str); 50] = [
        (Intrinsic::Floor, "floor"),
        (Intrinsic::Ceil, "ceil"),
        (Intrinsic::Round, "round"),
        (Intrinsic::Sign, "sign"),
        (Intrinsic::Log, "log"),
        (Intrinsic::Tan, "tan"),
        (Intrinsic::Asin, "asin"),
        (Intrinsic::Acos, "acos"),
        (Intrinsic::Atan, "atan"),
        (Intrinsic::Fract, "fract"),
        (Intrinsic::Abs, "abs"),
        (Intrinsic::Sqrt, "sqrt"),
        (Intrinsic::Sin, "sin"),
        (Intrinsic::Cos, "cos"),
        (Intrinsic::Min, "min"),
        (Intrinsic::Max, "max"),
        (Intrinsic::Clamp, "clamp"),
        (Intrinsic::Mix, "mix"),
        (Intrinsic::AddWrap, "add_wrap"),
        (Intrinsic::SubWrap, "sub_wrap"),
        (Intrinsic::MulWrap, "mul_wrap"),
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
        (Intrinsic::Trunc, "trunc"),
        (Intrinsic::Exp2, "exp2"),
        (Intrinsic::Log2, "log2"),
        (Intrinsic::Sinh, "sinh"),
        (Intrinsic::Cosh, "cosh"),
        (Intrinsic::Tanh, "tanh"),
        (Intrinsic::Fma, "fma"),
        (Intrinsic::Popcount, "popcount"),
        (Intrinsic::Clz, "clz"),
        (Intrinsic::Ctz, "ctz"),
        (Intrinsic::Rotl, "rotl"),
        (Intrinsic::Rotr, "rotr"),
        (Intrinsic::Ult, "ult"),
        (Intrinsic::Udiv, "udiv"),
        (Intrinsic::Urem, "urem"),
        (Intrinsic::Mulhi, "mulhi"),
        (Intrinsic::Umulhi, "umulhi"),
    ];

    /// Source spelling.
    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(intrinsic, _)| *intrinsic == self)
            .map(|(_, name)| *name)
            .unwrap_or("?")
    }

    /// One sentence on what the intrinsic computes, for `help`.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Floor => "The largest integer not greater than x, componentwise.",
            Self::Ceil => "The smallest integer not less than x, componentwise.",
            Self::Round => "The nearest integer, ties to even, componentwise.",
            Self::Sign => "-1, 0 or 1 by the sign of x, componentwise; an Int stays an Int.",
            Self::Log => "The natural logarithm, componentwise.",
            Self::Tan => "The tangent, componentwise.",
            Self::Asin => "The arc sine, componentwise.",
            Self::Acos => "The arc cosine, componentwise.",
            Self::Atan => "The arc tangent of one value, componentwise; atan2 takes two.",
            Self::Fract => "The fractional part, x - floor(x), componentwise.",
            Self::Abs => "The absolute value, componentwise; an Int stays an Int.",
            Self::Sqrt => "The square root, componentwise.",
            Self::Sin => "The sine, componentwise.",
            Self::Cos => "The cosine, componentwise.",
            Self::Min => {
                "The smaller of a and b, componentwise; a Float beside a vector broadcasts."
            }
            Self::Max => {
                "The larger of a and b, componentwise; a Float beside a vector broadcasts."
            }
            Self::Clamp => "x held between low and high, componentwise; Float bounds broadcast.",
            Self::Mix => "a + (b - a) * t, componentwise; t a Float or componentwise.",
            Self::AddWrap => "a + b modulo 2^32 on Ints: never a failure, unlike +.",
            Self::SubWrap => "a - b modulo 2^32 on Ints: never a failure, unlike -.",
            Self::MulWrap => "The low 32 bits of a * b on Ints: never a failure, unlike *.",
            Self::ToFloat => "An Int, or a Bool as 0 or 1, as a Float.",
            Self::ToInt => "A Float truncated toward zero, or a Bool as 0 or 1, as an Int.",
            Self::Dot => "The dot product of two vectors of one size.",
            Self::Cross => "The cross product of two Vec3.",
            Self::Length => "The Euclidean length of a vector.",
            Self::Normalize => "The unit vector in v's direction; the zero vector stays zero.",
            Self::Pow => "base raised to exponent, componentwise; a Float exponent broadcasts.",
            Self::Exp => "e raised to x, componentwise.",
            Self::Step => "1 where x >= edge, else 0, componentwise; a Float edge broadcasts.",
            Self::Smoothstep => {
                "0 below low, 1 above high, a smooth curve between, componentwise; Float edges broadcast."
            }
            Self::Reflect => {
                "incident reflected off the surface with unit normal normal: incident - 2 * dot(normal, incident) * normal."
            }
            Self::Atan2 => "The angle of the point (x, y) from the x axis, in -pi..pi.",
            Self::Trunc => "The integer part of x, toward zero, componentwise.",
            Self::Exp2 => "2 raised to x, componentwise.",
            Self::Log2 => "The base-2 logarithm, componentwise.",
            Self::Sinh => "The hyperbolic sine, componentwise.",
            Self::Cosh => "The hyperbolic cosine, componentwise.",
            Self::Tanh => "The hyperbolic tangent, componentwise.",
            Self::Fma => "a * b + c rounded once, componentwise.",
            Self::Popcount => "How many bits of x are one.",
            Self::Clz => "How many zero bits lead x; 32 for zero.",
            Self::Ctz => "How many zero bits trail x; 32 for zero.",
            Self::Rotl => "x's bits rotated left by n modulo 32.",
            Self::Rotr => "x's bits rotated right by n modulo 32.",
            Self::Ult => "Whether a is below b as unsigned 32-bit values.",
            Self::Udiv => {
                "The quotient of a and b as unsigned 32-bit values; a zero b is a failure, like /."
            }
            Self::Urem => {
                "The remainder of a and b as unsigned 32-bit values; a zero b is a failure, like %."
            }
            Self::Mulhi => "The high 32 bits of the signed 64-bit product of a and b.",
            Self::Umulhi => "The high 32 bits of the unsigned 64-bit product of a and b.",
        }
    }

    /// The names of the arguments, for `help`.
    pub fn argument_names(self) -> &'static [&'static str] {
        match self {
            Self::Min
            | Self::Max
            | Self::AddWrap
            | Self::SubWrap
            | Self::MulWrap
            | Self::Dot
            | Self::Cross => &["a", "b"],
            Self::Clamp => &["x", "low", "high"],
            Self::Mix => &["a", "b", "t"],
            Self::Pow => &["base", "exponent"],
            Self::Step => &["edge", "x"],
            Self::Smoothstep => &["low", "high", "x"],
            Self::Reflect => &["incident", "normal"],
            Self::Atan2 => &["y", "x"],
            Self::Fma => &["a", "b", "c"],
            Self::Rotl | Self::Rotr => &["x", "n"],
            Self::Ult | Self::Udiv | Self::Urem | Self::Mulhi | Self::Umulhi => &["a", "b"],
            Self::Length | Self::Normalize => &["v"],
            _ => &["x"],
        }
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
                | Self::Ceil
                | Self::Round
                | Self::Sign
                | Self::Fract
                | Self::Abs
                | Self::Sqrt
                | Self::Sin
                | Self::Cos
                | Self::Tan
                | Self::Asin
                | Self::Acos
                | Self::Atan
                | Self::Exp
                | Self::Log,
                [a],
            ) if float_like(*a) => Some(*a),
            (Self::Abs | Self::Sign, [Type::Int]) => Some(Type::Int),
            (Self::Min | Self::Max, [a, b]) if a == b && (float_like(*a) || *a == Type::Int) => {
                Some(*a)
            }
            (Self::Min | Self::Max, [a, Type::Float]) if vector(*a) => Some(*a),
            (Self::AddWrap | Self::SubWrap | Self::MulWrap, [Type::Int, Type::Int]) => {
                Some(Type::Int)
            }
            (Self::Clamp, [a, b, c]) if a == b && b == c && (float_like(*a) || *a == Type::Int) => {
                Some(*a)
            }
            (Self::Clamp, [a, Type::Float, Type::Float]) if vector(*a) => Some(*a),
            (Self::Mix, [a, b, c]) if a == b && (*c == Type::Float || c == a) && float_like(*a) => {
                Some(*a)
            }
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
            (Self::Trunc | Self::Exp2 | Self::Log2 | Self::Sinh | Self::Cosh | Self::Tanh, [a])
                if float_like(*a) =>
            {
                Some(*a)
            }
            (Self::Fma, [a, b, c]) if a == b && b == c && float_like(*a) => Some(*a),
            (Self::Popcount | Self::Clz | Self::Ctz, [Type::Int]) => Some(Type::Int),
            (
                Self::Rotl | Self::Rotr | Self::Udiv | Self::Urem | Self::Mulhi | Self::Umulhi,
                [Type::Int, Type::Int],
            ) => Some(Type::Int),
            (Self::Ult, [Type::Int, Type::Int]) => Some(Type::Bool),
            _ => None,
        }
    }
}

/// One field of a struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// Field name.
    pub name: String,
    /// Field type: plain, or an earlier struct in the table.
    pub ty: Type,
}

/// A Kernel-defined record type. Struct `i` may only contain structs with
/// index less than `i`, which rules out recursion by construction.
///
/// A fixed-size array `[T; N]` is a struct of `N` fields of one type `T`,
/// named `0` to `N - 1`, with `element` set to `T`: it lays out, builds
/// and extracts like any struct, and may also be indexed at run time
/// with [`Op::Index`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDef {
    /// Name for inspection and diagnostics; an array's is `[T; N]`.
    pub name: String,
    /// Fields in declaration order.
    pub fields: Vec<Field>,
    /// The element type when the struct is an array, whose fields are
    /// all of it.
    pub element: Option<Type>,
}

/// Largest element count an array type may declare.
pub const MAX_ARRAY_COUNT: u32 = 4096;

impl StructDef {
    /// A named record.
    pub fn record(name: impl Into<String>, fields: Vec<Field>) -> Self {
        Self {
            name: name.into(),
            fields,
            element: None,
        }
    }

    /// The array `[element; count]`, named by the element's spelling.
    pub fn array(element: Type, element_name: &str, count: u32) -> Self {
        Self {
            name: format!("[{element_name}; {count}]"),
            fields: (0..count)
                .map(|k| Field {
                    name: k.to_string(),
                    ty: element,
                })
                .collect(),
            element: Some(element),
        }
    }

    /// Whether the struct is an array, and of what.
    pub fn is_array(&self) -> bool {
        self.element.is_some()
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
    /// Build a vector from float components or a struct from its fields; the
    /// instruction type says which.
    Construct(Vec<ValueId>),
    /// Extract a float component from a vector, a field from a struct, or
    /// an element of an array at a constant index.
    Extract(ValueId, u32),
    /// The element of an array at an `Int` index, clamped into the
    /// array's range, as the array's element type.
    Subscript(ValueId, ValueId),
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
    /// Evaluate the arm whose values hold the `Int` `scrutinee`, else
    /// `otherwise`, and yield that block's result. Only that block runs.
    /// No value appears in two arms.
    Match {
        /// The `Int` matched.
        scrutinee: ValueId,
        /// The arms in order: the values each takes, and its block.
        arms: Vec<(Vec<i32>, Block)>,
        /// The block taken when no arm holds the value.
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
    /// The carried value of the innermost enclosing loop, or, outside any
    /// loop in a scan's entry, the scan's carried value.
    Carried,
    /// The iteration index of the innermost enclosing loop as `Int`, or,
    /// outside any loop in a scan's entry, the element index.
    Index,
    /// The current element's coordinate: `Vec2` over a grid, `Vec3` over a
    /// volume; rows have `element()` instead. Entry point only.
    Coord,
    /// The output domain's extent: `Int` for rows, `Vec2` for a grid,
    /// `Vec3` for a volume. Entry point only.
    Domain,
    /// Session time in seconds as `Float`. Context: `time.read`.
    Time,
    /// The sound device's clock in samples per second as `Float`, zero
    /// without a device. Context: `time.read`.
    Rate,
    /// Pointer position in output pixels as `Vec2`. Context: `input.read`.
    Pointer,
    /// Whether the pointer button is held, as `Bool`. Context: `input.read`.
    PointerDown,
    /// Whether the pointer button with the given `Int` number is held, as
    /// `Bool`: 0 left, 1 right, 2 middle. Context: `input.read`.
    Button(ValueId),
    /// What the wheel scrolled since the last evaluation, as `Vec2`.
    /// Context: `input.read`.
    Wheel,
    /// The window's size in physical pixels as `Vec2`, zero by zero
    /// without a window. Context: `input.read`.
    Window,
    /// The window's scale factor, physical pixels per logical pixel, as
    /// `Float`; one without a window. Context: `input.read`.
    Scale,
    /// Whether the key with the given `Int` code is held, as `Bool`.
    /// Context: `input.read`.
    Key(ValueId),
    /// The current element's linear index as `Int`. Entry point only.
    Element,
    /// The element of an input at the current coordinate, clamped to the
    /// input's extent, as its element type. The input's shape must match
    /// the output's. Host call: `field.read`. Entry point only.
    Sample(u32),
    /// The element of an input at an address of the input's index type
    /// (`Int`, `Vec2`, or `Vec3`; clamped), as its element type. Host call:
    /// `field.read`.
    At(u32, ValueId),
    /// The element of an input at an address of the input's index type,
    /// interpolated linearly between the neighbouring elements (bilinear
    /// over a grid, trilinear over a volume; clamped at the edges), as
    /// its element type. Only for inputs whose layout is all floats.
    /// Host call: `field.read`.
    AtLinear(u32, ValueId),
    /// Number of elements of an input as `Int`. Host call: `field.read`.
    Count(u32),
    /// Extent of an input as its index type. Host call: `field.read`.
    Size(u32),
    /// Writes a value at an address of the input's index type into a
    /// memory input, in step order, later reads of it seeing the write; an
    /// address outside the memory is dropped. Yields the value written.
    /// Host call: `field.write`. Entry point of an ordered Kernel only.
    Put(u32, ValueId, ValueId),
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
    /// Named values, for inspection: the `let` bindings in source order,
    /// each with the value it names. No part of the program's meaning;
    /// a trace reads them.
    pub bindings: Vec<(ValueId, String)>,
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
                        Op::Match {
                            arms, otherwise, ..
                        } => {
                            arms.iter()
                                .map(|(_, block)| count(&block.body))
                                .sum::<usize>()
                                + count(&otherwise.body)
                        }
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
    /// Struct types in dependency order.
    pub structs: Vec<StructDef>,
    /// Helper functions in dependency order.
    pub functions: Vec<Function>,
    /// The entry point, executed once per element (of the output's domain
    /// in [`Mode::Map`], of the first input's in [`Mode::Reduce`]). Its
    /// results map to `manifest.outputs` in order.
    pub entry: Function,
    /// In [`Mode::Reduce`], the fold: `(T, T) -> T` over the entry's
    /// result type `T`. Present exactly when the mode is `Reduce`.
    pub merge: Option<Function>,
    /// In [`Mode::Scan`], the carried value before the first element:
    /// `() -> T`. Present exactly when the mode is `Scan`.
    pub init: Option<Function>,
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
