#![warn(missing_docs)]
//! Entelechy IR (L3).
//!
//! The typed, SSA-like, structured, backend-independent form of a Kernel.
//! Each instruction defines one value and refers only to values defined
//! before it in its block or an enclosing one; control flow is `if` and
//! bounded `loop`. Host access is explicit operations, so validation proves
//! a Kernel's capabilities before it runs.

mod account;
mod display;
pub use display::{NamedManifest, port_type_name, type_name};
pub use size::{Carried, Sizes, carried, sizes};
mod fuse;
mod group;
mod lanes;
mod optimize;
mod size;
mod validate;

use std::fmt;
use std::str::FromStr;

pub use account::{Renumbering, account, account_renumbered};
pub use entelechy_abi::{
    Capabilities, Capability, Encoding, Number, PortType, Scalar, Schema, Shape, ShapeIndex, Type,
    Value,
};
pub use fuse::{
    FuseError, InputUse, counted, counts_only, extent_params, fuse, fuse_indexed, fuse_outputs,
    fused_name, input_use, offset_read, row_reads, samples_only,
};
pub use group::group_fault;
pub use lanes::lanes;
pub use optimize::{carried_builders, optimize, renumbered};
pub use validate::{Validated, ValidationError, linear_index_type, validate};

/// The IR version, raised when a module's validity or meaning changes.
pub const IR_VERSION: u32 = 1;

/// Stable identity of a Kernel, `name@version` (`image.grayscale@1`).
/// Graphs bind to it, never to source paths.
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
    /// The element type the entry produces or `sample` yields for this port.
    pub fn element(&self) -> Type {
        self.ty.element
    }

    /// How the port's elements are addressed.
    pub fn shape(&self) -> Shape {
        self.ty.shape
    }
}

/// The kind of word a value type's scalars are kept as.
pub fn scalar_kind(ty: Type) -> Option<Scalar> {
    Some(match ty.scalar()? {
        Type::Float => Scalar::Float,
        Type::Int => Scalar::Int,
        Type::Bool => Scalar::Bool,
        Type::Double => Scalar::Double,
        Type::Long => Scalar::Long,
        _ => return None,
    })
}

/// The storage layout of a value type: vectors and structs flattened to
/// scalars in declaration order. `None` for unknown structs.
pub fn schema_of(ty: Type, structs: &[StructDef]) -> Option<Schema> {
    fn push(ty: Type, structs: &[StructDef], out: &mut Vec<Scalar>) -> Option<()> {
        match ty {
            Type::Struct(index) => {
                for field in &structs.get(index as usize)?.fields {
                    push(field.ty, structs, out)?;
                }
            }
            other => {
                let words = other.words()?;
                out.extend(std::iter::repeat_n(scalar_kind(other)?, words));
            }
        }
        Some(())
    }
    let mut scalars = Vec::new();
    push(ty, structs, &mut scalars)?;
    Some(Schema::new(scalars))
}

/// The name of each number an element of type `ty` holds, in the order
/// [`schema_of`] lays them out: a struct's fields by name, nested ones
/// joined by a dot; a vector's lanes `x` to `w`; a matrix's numbered from
/// 0; a value of one number unnamed (`""`). `None` for unknown structs.
pub fn number_names(ty: Type, structs: &[StructDef]) -> Option<Vec<String>> {
    fn joined(name: &str, part: &str) -> String {
        if name.is_empty() {
            part.to_owned()
        } else {
            format!("{name}.{part}")
        }
    }
    fn push(ty: Type, structs: &[StructDef], name: &str, out: &mut Vec<String>) -> Option<()> {
        match ty {
            Type::Struct(index) => {
                for field in &structs.get(index as usize)?.fields {
                    push(field.ty, structs, &joined(name, &field.name), out)?;
                }
            }
            other => {
                let (_, lanes) = other.shape()?;
                match lanes {
                    1 => out.push(name.to_owned()),
                    2..=4 => out.extend(
                        ["x", "y", "z", "w"][..usize::from(lanes)]
                            .iter()
                            .map(|lane| joined(name, lane)),
                    ),
                    _ => out.extend((0..lanes).map(|i| joined(name, &i.to_string()))),
                }
            }
        }
        Some(())
    }
    let mut names = Vec::new();
    push(ty, structs, "", &mut names)?;
    Some(names)
}

/// A named parameter of any value type, with a default value.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Parameter name.
    pub name: String,
    /// Its type: any value type, the module's structs and arrays included.
    pub ty: Type,
    /// Default value, of that type.
    pub default: Value,
}

/// A value of type `ty` with every word zero.
pub fn template(ty: Type, structs: &[StructDef]) -> Value {
    match ty {
        Type::Float => Value::Float(0.0),
        Type::Int => Value::Int(0),
        Type::Bool => Value::Bool(false),
        Type::Vec2 => Value::Vec2([0.0; 2]),
        Type::Vec3 => Value::Vec3([0.0; 3]),
        Type::Vec4 => Value::Vec4([0.0; 4]),
        Type::IVec2 => Value::IVec2([0; 2]),
        Type::IVec3 => Value::IVec3([0; 3]),
        Type::IVec4 => Value::IVec4([0; 4]),
        Type::Mat2 => Value::Mat2([0.0; 4]),
        Type::Mat3 => Value::Mat3(Box::new([0.0; 9])),
        Type::Mat4 => Value::Mat4(Box::new([0.0; 16])),
        Type::Double => Value::Double(0.0),
        Type::Long => Value::Long(0),
        Type::DVec2 => Value::DVec2([0.0; 2]),
        Type::DVec3 => Value::DVec3(Box::new([0.0; 3])),
        Type::DVec4 => Value::DVec4(Box::new([0.0; 4])),
        Type::LVec2 => Value::LVec2([0; 2]),
        Type::LVec3 => Value::LVec3(Box::new([0; 3])),
        Type::DMat2 => Value::DMat2(Box::new([0.0; 4])),
        Type::DMat3 => Value::DMat3(Box::new([0.0; 9])),
        Type::DMat4 => Value::DMat4(Box::new([0.0; 16])),
        Type::Struct(index) => Value::Record(
            structs[index as usize]
                .fields
                .iter()
                .map(|field| template(field.ty, structs))
                .collect(),
        ),
        other => unreachable!("{other:?} is no type a source can write"),
    }
}

/// `value` retyped as `ty` when their layouts match (see
/// [`Number::as_kind`](entelechy_abi::Number::as_kind)); otherwise `None`.
pub fn value_as(value: &Value, ty: Type, structs: &[StructDef]) -> Option<Value> {
    value.shaped_like(&template(ty, structs))
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
    /// Names of the memories: ports in both `inputs` (bound by the graph,
    /// read with `at`) and `outputs`, which an ordered Kernel writes step by
    /// step with [`Op::Put`]. They come last in both lists, in one order.
    pub memories: Vec<String>,
    /// Loop iterations one element (a scan step, a reduction merge) may run,
    /// nested loops included; past it the element fails on every backend.
    /// See [`Module::counts_work`].
    pub budget: u32,
}

/// How a Kernel executes over its domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The entry runs once per element of the output domain.
    #[default]
    Map,
    /// The entry runs per element of the first input; `merge` folds the
    /// results pairwise into the single element of a `Rows` output. `merge`
    /// is associative and commutative: the backend picks the fold's tree.
    Reduce,
    /// The entry runs per row of the first input, in order, reading the
    /// previous result through [`Op::Carried`] (`init` before the first) and
    /// writing each result to that element of a `Rows` output.
    Scan,
    /// The entry runs per element of the first input, yielding an address
    /// into `base` and a value. The output is `base` with each value combined
    /// into its address as if applied in element order; under
    /// [`Combine::Sum`] an integer is the element's whole total. Addresses
    /// outside `base` are dropped; `base` has the output's shape and layout.
    Scatter {
        /// Index of the input the output starts from.
        base: u32,
        /// How a value meets the element already there.
        combine: Combine,
    },
    /// The entry runs per row of the first input; each element of the `Rows`
    /// output is its value combined by [`Combine`] with every value before
    /// it (an inclusive prefix fold). The rule is order-free (integer `sum`,
    /// `min`, `max`), so any grouping gives the reference's words.
    Prefix {
        /// How a value meets everything before it.
        combine: Combine,
    },
    /// The entry runs once per element of the output domain, as a map
    /// does, the elements in groups of `size` (by linear index: element
    /// `e` is lane `e % size` of group `e / size`). Lanes of a group hand
    /// each other values through [`Op::Exchange`], which every lane
    /// reaches alike; a group past the domain's end runs whole, and its
    /// lanes past the end write nothing.
    Group {
        /// Lanes in a group, 1 to [`MAX_GROUP`].
        size: u32,
    },
}

/// The most lanes a group holds: what every device runs together.
pub const MAX_GROUP: u32 = 256;

impl Mode {
    /// Stable spelling used in manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::Reduce => "reduce",
            Self::Scan => "scan",
            Self::Scatter { .. } => "scatter",
            Self::Prefix { .. } => "prefix",
            Self::Group { .. } => "group",
        }
    }
}

/// How a scattered value combines with the element at its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combine {
    /// Fieldwise addition. A float adds in element order; an integer is the
    /// element's whole total and fails, at the first element, only when that
    /// total overflows (`entelechy_rules::Sums`).
    Sum,
    /// Fieldwise minimum.
    Min,
    /// Fieldwise maximum.
    Max,
    /// The value replaces the element; the last source in element order wins.
    Last,
}

impl Combine {
    /// Stable spelling used in `.ent` sources.
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

    /// Whether the rule's answer over `element` is independent of pair
    /// order: `min` and `max` always, `sum` when no scalar is a float. Only
    /// such a rule takes pairs a scatter emits.
    pub fn order_free(self, element: Type, structs: &[StructDef]) -> bool {
        match self {
            Self::Min | Self::Max => true,
            Self::Sum => schema_of(element, structs).is_some_and(|schema| {
                !schema
                    .scalars()
                    .iter()
                    .any(|s| matches!(s, Scalar::Float | Scalar::Double))
            }),
            Self::Last => false,
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scatter { base, combine } => {
                write!(f, "scatter into input {base} with {}", combine.name())
            }
            Self::Prefix { combine } => write!(f, "prefix with {}", combine.name()),
            _ => f.write_str(self.name()),
        }
    }
}

/// Index of a value within a function, assigned in pre-order over its
/// instructions, nested blocks included; also its slot in a flat frame.
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
    /// `a >>> b` on integers, logical; the count is taken modulo 32.
    Ushr,
}

impl BinaryOp {
    /// Result type of `a op b`, or `None` when the operands cannot combine.
    /// Both sides share one scalar kind (`Float`/`Double` and `Int`/`Long`
    /// meet only through a conversion). Vectors combine componentwise with
    /// same-size vectors or a scalar of their kind. Matrices add and
    /// subtract with their own order and scale by a scalar; the compiler
    /// writes their products out. Comparisons take two scalars of one type
    /// and give `Bool`; equality also compares `Bool`s and whole vectors and
    /// matrices. `&&` and `||` take `Bool`s. Bit operators and shifts take
    /// integers of either width and integer vectors, a scalar broadcasting.
    pub fn result_type(self, a: Type, b: Type) -> Option<Type> {
        let kind = a.scalar()?;
        if b.scalar()? != kind {
            return None;
        }
        let integer = matches!(kind, Type::Int | Type::Long);
        let float = matches!(kind, Type::Float | Type::Double);
        // Same shape, or a scalar beside the other.
        let broadcast = || {
            if a == b || b == kind {
                Some(a)
            } else if a == kind {
                Some(b)
            } else {
                None
            }
        };
        match self {
            // A matrix adds and subtracts with its own type and scales by
            // a scalar.
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem
                if a.order().is_some() || b.order().is_some() =>
            {
                match self {
                    Self::Add | Self::Sub if a == b => Some(a),
                    Self::Mul if a == kind => Some(b),
                    Self::Mul if b == kind => Some(a),
                    _ => None,
                }
            }
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem if integer || float => {
                broadcast()
            }
            Self::BitAnd | Self::BitOr | Self::BitXor | Self::Shl | Self::Shr | Self::Ushr
                if integer =>
            {
                broadcast()
            }
            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual
                if a == b && a == kind && (integer || float) =>
            {
                Some(Type::Bool)
            }
            // Vectors and matrices are equal when every component is.
            Self::Equal | Self::NotEqual if a == b => Some(Type::Bool),
            Self::And | Self::Or if a == Type::Bool && b == Type::Bool => Some(Type::Bool),
            _ => None,
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
        match (self, a.scalar()?) {
            (Self::Neg, Type::Float | Type::Int | Type::Double | Type::Long) => Some(a),
            (Self::Not, Type::Bool) if a == Type::Bool => Some(Type::Bool),
            _ => None,
        }
    }
}

/// Pure math intrinsics; their signatures are the one source for the
/// compiler, the validator and every backend.
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
    /// Convert any number, or a `Bool` as 0 or 1, to `Double`.
    ToDouble,
    /// Convert any number (floats truncating) or `Bool` (0 or 1) to `Long`.
    ToLong,
    /// `pow(base, exponent)` componentwise; a `Float` exponent broadcasts.
    Pow,
    /// `e^x` componentwise.
    Exp,
    /// `step(edge, x)`: 1 where `x >= edge`, else 0; a `Float` edge
    /// broadcasts.
    Step,
    /// `smoothstep(low, high, x)` componentwise; `Float` edges broadcast.
    Smoothstep,
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
    /// `bits(x)`: a Float's 32 bits as an Int, unchanged.
    Bits,
    /// `from_bits(x)`: an Int's 32 bits as a Float, unchanged.
    FromBits,
}

impl Intrinsic {
    /// All intrinsics with their source names.
    pub const ALL: [(Intrinsic, &'static str); 49] = [
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
        (Intrinsic::ToDouble, "double"),
        (Intrinsic::ToLong, "long"),
        (Intrinsic::Pow, "pow"),
        (Intrinsic::Exp, "exp"),
        (Intrinsic::Step, "step"),
        (Intrinsic::Smoothstep, "smoothstep"),
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
        (Intrinsic::Bits, "bits"),
        (Intrinsic::FromBits, "from_bits"),
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
            Self::Abs => "The absolute value, componentwise; an Int or IVec stays one.",
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
            Self::ToFloat => {
                "An Int, a Double, a Long, or a Bool as 0 or 1, as a Float; a vector or matrix componentwise."
            }
            Self::ToInt => {
                "A float truncated toward zero, a Long wrapped to 32 bits, or a Bool as 0 or 1, as an Int; a vector componentwise."
            }
            Self::ToDouble => {
                "A Float, an Int, a Long, or a Bool as 0 or 1, as a Double; a vector or matrix componentwise."
            }
            Self::ToLong => {
                "A float truncated toward zero, an Int, or a Bool as 0 or 1, as a Long; a vector componentwise."
            }
            Self::Pow => "base raised to exponent, componentwise; a Float exponent broadcasts.",
            Self::Exp => "e raised to x, componentwise.",
            Self::Step => "1 where x >= edge, else 0, componentwise; a Float edge broadcasts.",
            Self::Smoothstep => {
                "0 below low, 1 above high, a smooth curve between, componentwise; Float edges broadcast."
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
            Self::Bits => {
                "x's 32 bits as an Int, unchanged: for a Float of no sign, the Ints order as the Floats do."
            }
            Self::FromBits => "x's 32 bits as a Float, unchanged; bits(from_bits(x)) is x.",
        }
    }

    /// The names of the arguments, for `help`.
    pub fn argument_names(self) -> &'static [&'static str] {
        match self {
            Self::Min | Self::Max | Self::AddWrap | Self::SubWrap | Self::MulWrap => &["a", "b"],
            Self::Clamp => &["x", "low", "high"],
            Self::Mix => &["a", "b", "t"],
            Self::Pow => &["base", "exponent"],
            Self::Step => &["edge", "x"],
            Self::Smoothstep => &["low", "high", "x"],
            Self::Atan2 => &["y", "x"],
            Self::Fma => &["a", "b", "c"],
            Self::Rotl | Self::Rotr => &["x", "n"],
            Self::Ult | Self::Udiv | Self::Urem | Self::Mulhi | Self::Umulhi => &["a", "b"],
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

    /// Result type for the argument types, or `None` if ill-typed.
    pub fn result_type(self, args: &[Type]) -> Option<Type> {
        // A float of either width, or a vector of them.
        let float_like = |ty: Type| ty.is_float_kind() && ty.order().is_none();
        // An integer of either width, or a vector of them.
        let int_like = |ty: Type| ty.is_int_kind();
        let vector = |ty: Type| ty.lanes().is_some();
        let scalar = |ty: Type| ty.scalar().unwrap_or(ty);
        // A conversion to `to`: the argument's shape in `to`'s kind.
        let convert = |to: Type| match args {
            [a] if a.scalar().is_some_and(|from| from != to) => a.with_scalar(to),
            _ => None,
        };
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
                | Self::Log
                | Self::Trunc
                | Self::Exp2
                | Self::Log2
                | Self::Sinh
                | Self::Cosh
                | Self::Tanh,
                [a],
            ) if float_like(*a) => Some(*a),
            (Self::Abs | Self::Sign, [a]) if int_like(*a) => Some(*a),
            (Self::Min | Self::Max, [a, b]) if a == b && (float_like(*a) || int_like(*a)) => {
                Some(*a)
            }
            (Self::Min | Self::Max, [a, b])
                if vector(*a) && *b == scalar(*a) && (float_like(*a) || int_like(*a)) =>
            {
                Some(*a)
            }
            (Self::AddWrap | Self::SubWrap | Self::MulWrap, [a, b])
                if a == b && matches!(*a, Type::Int | Type::Long) =>
            {
                Some(*a)
            }
            (Self::Clamp, [a, b, c]) if a == b && b == c && (float_like(*a) || int_like(*a)) => {
                Some(*a)
            }
            (Self::Clamp, [a, b, c])
                if vector(*a) && b == c && *b == scalar(*a) && (float_like(*a) || int_like(*a)) =>
            {
                Some(*a)
            }
            (Self::Mix, [a, b, c]) if a == b && (*c == scalar(*a) || c == a) && float_like(*a) => {
                Some(*a)
            }
            (Self::ToFloat, _) => convert(Type::Float),
            (Self::ToInt, _) => convert(Type::Int),
            (Self::ToDouble, _) => convert(Type::Double),
            (Self::ToLong, _) => convert(Type::Long),
            (Self::Pow, [a, b]) if (a == b || *b == scalar(*a)) && float_like(*a) => Some(*a),
            (Self::Step, [a, b]) if (a == b || *a == scalar(*b)) && float_like(*b) => Some(*b),
            (Self::Smoothstep, [a, b, c])
                if a == b && (b == c || *a == scalar(*c)) && float_like(*c) =>
            {
                Some(*c)
            }
            (Self::Atan2, [a, b]) if a == b && matches!(*a, Type::Float | Type::Double) => Some(*a),
            (Self::Fma, [a, b, c]) if a == b && b == c && float_like(*a) => Some(*a),
            (Self::Popcount | Self::Clz | Self::Ctz, [Type::Int]) => Some(Type::Int),
            (
                Self::Rotl | Self::Rotr | Self::Udiv | Self::Urem | Self::Mulhi | Self::Umulhi,
                [Type::Int, Type::Int],
            ) => Some(Type::Int),
            (Self::Ult, [Type::Int, Type::Int]) => Some(Type::Bool),
            (Self::Bits, [Type::Float]) => Some(Type::Int),
            (Self::FromBits, [Type::Int]) => Some(Type::Float),
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

/// A Kernel-defined record type. Struct `i` contains only structs below
/// `i`, so none is recursive.
///
/// An array `[T; N]` is a struct of `N` fields of type `T` named `0` to
/// `N - 1`, with `element` set to `T`; [`Op::Subscript`] also indexes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDef {
    /// Name for inspection and diagnostics; an array's is `[T; N]`.
    pub name: String,
    /// Fields in declaration order.
    pub fields: Vec<Field>,
    /// The element type when the struct is an array.
    pub element: Option<Type>,
}

/// Largest element count an array type may declare.
pub const MAX_ARRAY_COUNT: u32 = 4096;

/// The most parts a value may hold: its words and the structs and arrays
/// around them. Every walk over a value's type is bounded by it.
pub const MAX_VALUE_PARTS: usize = 1 << 16;

/// The deepest structs and arrays may nest in a type.
pub const MAX_TYPE_DEPTH: usize = 64;

/// The parts a value of `ty` holds and how deep its structs nest, each
/// struct in `structs` it reaches within [`MAX_VALUE_PARTS`] and
/// [`MAX_TYPE_DEPTH`].
pub fn measure(structs: &[StructDef], ty: Type) -> (usize, usize) {
    match ty {
        Type::Struct(index) => {
            measure_struct(&structs[index as usize], &|field| measure(structs, field))
        }
        other => (other.words().unwrap_or(1), 0),
    }
}

/// A struct's parts and depth, from what `field` measures its fields'
/// types at; an array's elements are measured once.
pub fn measure_struct(def: &StructDef, field: &dyn Fn(Type) -> (usize, usize)) -> (usize, usize) {
    if let Some(element) = def.element {
        let (parts, depth) = field(element);
        return (
            parts.saturating_mul(def.fields.len()).saturating_add(1),
            depth + 1,
        );
    }
    def.fields.iter().fold((1, 1), |(parts, depth), each| {
        let (more, deep) = field(each.ty);
        (parts.saturating_add(more), depth.max(deep + 1))
    })
}

/// Why a value of `ty` may not be: it holds past [`MAX_VALUE_PARTS`] or
/// nests past [`MAX_TYPE_DEPTH`].
pub fn too_large((parts, depth): (usize, usize)) -> Option<String> {
    if parts > MAX_VALUE_PARTS {
        return Some(format!(
            "a value of it holds {parts} words and structs; a value holds at most {MAX_VALUE_PARTS}"
        ));
    }
    (depth > MAX_TYPE_DEPTH)
        .then(|| format!("it nests {depth} deep; a type nests at most {MAX_TYPE_DEPTH}"))
}

/// Words a value of this type takes in the flat layout the Host, backends
/// and IR share: a struct's fields end to end, a matrix's columns.
pub fn words(structs: &[StructDef], ty: Type) -> usize {
    match ty {
        Type::Struct(index) => structs[index as usize]
            .fields
            .iter()
            .map(|field| words(structs, field.ty))
            .sum(),
        other => other.words().unwrap_or(1),
    }
}

/// Where part `k` (struct field, matrix column or vector component) of a
/// value of this type begins and its type, as [`Op::Extract`] takes it.
pub fn part_at(structs: &[StructDef], ty: Type, k: usize) -> Option<(usize, Type)> {
    match ty {
        Type::Struct(index) => {
            let def = structs.get(index as usize)?;
            let field = def.fields.get(k)?;
            let offset = def.fields[..k]
                .iter()
                .map(|field| words(structs, field.ty))
                .sum();
            Some((offset, field.ty))
        }
        other => {
            let (count, part) = match (other.order(), other.column()) {
                (Some(order), Some(column)) => (usize::from(order), column),
                _ => {
                    let (count, lane) = other.lanes()?;
                    (usize::from(count), lane)
                }
            };
            (k < count).then(|| (k * words(structs, part), part))
        }
    }
}

/// The part type, count and word stride of an indexed value of this type,
/// as [`Op::Subscript`] takes them, clamping the index into `0..count`.
pub fn elements_of(structs: &[StructDef], ty: Type) -> Option<(Type, usize, usize)> {
    match ty {
        Type::Struct(index) => {
            let def = structs.get(index as usize)?;
            let element = def.element?;
            Some((element, def.fields.len(), words(structs, element)))
        }
        other => {
            let part = other
                .column()
                .or_else(|| other.lanes().map(|(_, lane)| lane))?;
            let count = other
                .order()
                .map(usize::from)
                .or_else(|| other.lanes().map(|(n, _)| usize::from(n)))?;
            Some((part, count, words(structs, part)))
        }
    }
}

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

impl Op {
    /// The blocks the op carries, in order: an `if`'s two, a `match`'s arms
    /// then its default, a `loop`'s test then its body. Every walk over
    /// nested instructions goes through here.
    pub fn blocks(&self) -> Vec<&Block> {
        match self {
            Op::If {
                then, otherwise, ..
            } => vec![then, otherwise],
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter()
                .map(|(_, block)| block)
                .chain(std::iter::once(otherwise))
                .collect(),
            Op::Loop { until, body, .. } => until.iter().chain(std::iter::once(body)).collect(),
            Op::Each { body, .. } | Op::Lanes { body } => vec![body],
            _ => Vec::new(),
        }
    }

    /// [`Op::blocks`], to change them.
    pub fn blocks_mut(&mut self) -> Vec<&mut Block> {
        match self {
            Op::If {
                then, otherwise, ..
            } => vec![then, otherwise],
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter_mut()
                .map(|(_, block)| block)
                .chain(std::iter::once(otherwise))
                .collect(),
            Op::Loop { until, body, .. } => until.iter_mut().chain(std::iter::once(body)).collect(),
            Op::Each { body, .. } | Op::Lanes { body } => vec![body],
            _ => Vec::new(),
        }
    }
}

/// Largest iteration bound a `loop` may write: a literal count, and an
/// `[each]`'s. A count read at run time is bounded by [`MAX_BUDGET`]
/// instead, so the element's work budget is what stops it.
pub const MAX_LOOP_COUNT: u32 = 65_536;

/// The default work budget: loop iterations an element may run.
pub const DEFAULT_BUDGET: u32 = 1 << 24;

/// The largest work budget a Kernel may declare.
pub const MAX_BUDGET: u32 = 1 << 30;

/// The loop iterations an element adds to its run's total at a time. The
/// total counts whole blocks: one shared addition per block, and a Kernel
/// whose elements cannot fill one is not counted.
pub const WORK_BLOCK: u32 = 1024;

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
    /// Build a vector, integer vector, matrix (from columns) or struct
    /// (from fields); the instruction type says which.
    Construct(Vec<ValueId>),
    /// Extract a vector component, matrix column, struct field or array
    /// element at a constant index.
    Extract(ValueId, u32),
    /// The element of an array, the column of a matrix or the component
    /// of a vector at an `Int` index, clamped into range.
    Subscript(ValueId, ValueId),
    /// Call a Kernel-defined function by index with arguments.
    Call(u32, Vec<ValueId>),
    /// A pure intrinsic.
    Intrinsic(Intrinsic, Vec<ValueId>),
    /// Yield `then` or `otherwise` by a `Bool`; only the taken block runs.
    If {
        /// The condition.
        condition: ValueId,
        /// Block taken when the condition holds.
        then: Block,
        /// Block taken otherwise.
        otherwise: Block,
    },
    /// Yield the arm whose values hold the `Int` `scrutinee`, else
    /// `otherwise`; only that block runs. No value appears in two arms.
    Match {
        /// The `Int` matched.
        scrutinee: ValueId,
        /// The arms in order: the values each takes, and its block.
        arms: Vec<(Vec<i32>, Block)>,
        /// The block taken when no arm holds the value.
        otherwise: Block,
    },
    /// Run `body` at most `count` times, threading one carried value from
    /// `init`: each iteration reads it through [`Op::Carried`] and yields
    /// its replacement. `until`, tested on the carried value before each
    /// iteration, stops the loop when it holds. Yields the final value.
    Loop {
        /// Compile-time iteration bound: at most [`MAX_LOOP_COUNT`] alone,
        /// at most [`MAX_BUDGET`] beside a run-time `bound`.
        count: u32,
        /// An optional run-time `Int`, read once before the loop: the loop
        /// runs its value clamped to `0..=count` times. Its `count` is the
        /// largest budget, so no element runs it that far: the budget fails
        /// the element first, named, rather than the count stopping short.
        bound: Option<ValueId>,
        /// Initial carried value.
        init: ValueId,
        /// Optional exit test, yielding `Bool`.
        until: Option<Block>,
        /// The body, yielding the next carried value.
        body: Block,
    },
    /// Build an array of `count` elements: `body` runs once for each, in
    /// order, [`Op::Index`] its position, and yields that element. An
    /// `[each]` not written out: its runs are not loop iterations, and
    /// count as no work but what their own loops do.
    Each {
        /// How many elements, at most [`MAX_LOOP_COUNT`].
        count: u32,
        /// The body, yielding one element.
        body: Block,
    },
    /// The carried value of the innermost enclosing loop (an [`Op::Each`]
    /// carries none), or, outside any loop in a scan's entry, the scan's
    /// carried value.
    Carried,
    /// The iteration index of the innermost enclosing loop or position of
    /// the innermost enclosing [`Op::Each`] as `Int`, or, outside both in
    /// a scan's entry, the element index.
    Index,
    /// The current element's coordinate: `Vec2` over a grid, `Vec3` over a
    /// volume; rows have `element()` instead. Entry point only.
    Coord,
    /// The output domain's extent: `Int` for rows, `Vec2` for a grid,
    /// `Vec3` for a volume. Entry point only.
    Domain,
    /// Session time in seconds as `Float`, the clock rounded to it.
    /// Context: `time.read`.
    Time,
    /// Session time in seconds as `Double`, the clock whole. Context:
    /// `time.read`.
    Clock,
    /// The sound device's clock in samples per second as `Float`, zero
    /// without a device. Context: `time.read`.
    Rate,
    /// The samples (per channel) the sound device holds and has not yet
    /// played, as `Int`, zero without a device. Context: `time.read`.
    Queued,
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
    /// How many pads are connected, as `Int`. Context: `input.read`.
    Pads,
    /// A pad's axis pair by number as `Vec2`: 0 left stick, 1 right, 2 the
    /// triggers. Sticks run `-1..1` (`y` away from the person), triggers
    /// `0..1`; zero without a pad. Context: `input.read`.
    Stick(ValueId),
    /// The window's size in physical pixels as `Vec2`, zero by zero
    /// without a window. Context: `input.read`.
    Window,
    /// The window's scale factor, physical pixels per logical pixel, as
    /// `Float`; one without a window. Context: `input.read`.
    Scale,
    /// Whether the key with the given `Int` code is held, as `Bool`.
    /// Context: `input.read`.
    Key(ValueId),
    /// The character typed since the last evaluation at the given `Int`
    /// place, as an `Int` code point; 0 after the last. Context:
    /// `input.read`.
    Typed(ValueId),
    /// The current element's linear index as `Int`. Entry point only.
    Element,
    /// The current element's place in its group, `0` to the group's size
    /// less one, as `Int`. A group's entry only.
    Lane,
    /// The value (the first operand) lane `lane` of this group gave at
    /// this point, the lane wrapped round the group's size: every lane
    /// gives its own and takes another's, so no two write one place. Every
    /// lane reaches it alike (its control flow depends on nothing a lane
    /// holds alone), in a group's entry only, and in no function it calls.
    Exchange(ValueId, ValueId),
    /// The element of an input at the current coordinate, clamped to the
    /// input's extent, as its element type. The input's shape must match
    /// the output's. Host call: `field.read`. Entry point only.
    Sample(u32),
    /// The element of an input at an address of the input's index type
    /// (`Int`, `Vec2`, or `Vec3`; clamped), as its element type. Host call:
    /// `field.read`.
    At(u32, ValueId),
    /// Part of one input element, named by a path; a backend reads only the
    /// words it names. The element is the one at the address, as [`Op::At`]
    /// takes it, or else this element's own, as [`Op::Sample`] reads it
    /// (entry point only). Host call: `field.read`.
    Part {
        /// The input read, by manifest index.
        input: u32,
        /// The address, of the input's index type; `None` for this element.
        at: Option<ValueId>,
        /// The way from the whole element to the part, outermost first.
        path: Vec<Step>,
    },
    /// The input element at an address of its index type, interpolated
    /// linearly (bilinear on a grid, trilinear on a volume; clamped at the
    /// edges). All-float layouts only. Host call: `field.read`.
    AtLinear(u32, ValueId),
    /// Number of elements of an input as `Int`. Host call: `field.read`.
    Count(u32),
    /// Extent of an input as its index type. Host call: `field.read`.
    Size(u32),
    /// Writes a value into a memory input at an address of its index type,
    /// in step order; later reads see it, an address outside is dropped.
    /// Yields the value. Host call: `field.write`. Ordered Kernel entry only.
    Put(u32, ValueId, ValueId),
    /// Emits one more scatter pair (target address, element value), laid
    /// out as the pair the entry returns. An element's pairs are those it
    /// emits, in order, then the one it returns; an address outside the
    /// target is dropped. Yields the pair. Host call: `field.write`. Only in
    /// a scatter's entry and its callees, under an order-free rule.
    Emit(ValueId),
    /// Adds the body iterations of the loop just before it (the operand) to
    /// the element's work and yields them as `Int`. Placed only by
    /// [`account()`]; see [`Module::worst_work`].
    Spend(ValueId),
    /// Checks the element's work, as `Bool`: `true` (a failure, recorded
    /// where seen; the element stops its loops) once it passes its Kernel's
    /// budget or its run the Host's bound. Otherwise hands the run's total
    /// the [`WORK_BLOCK`]s filled since the last check. Placed only by
    /// [`account()`].
    Look,
    /// Runs `body` once for each lane of the group, in lane order, as that
    /// lane: its element, place, reads, [`Op::Lane`], work and failures
    /// are the lane's. Yields the array of the lanes' results. A lane that
    /// fails is held, the lowest one held, until the next [`Op::Meet`],
    /// and runs no more. Not a loop: [`Op::Index`] and [`Op::Carried`] in
    /// it are those of the loop around it. Placed only by [`lanes()`].
    Lanes {
        /// The body, yielding one lane's element.
        body: Block,
    },
    /// The end of a stretch of the group between exchanges: once a lane
    /// has failed, the run fails here with the lowest lane held. At an
    /// exchange, the operand is the array of what the lanes give there, for
    /// a trace to answer a lane's exchanges from; at the entry's end there
    /// is none. Yields `Bool` `true`. Placed only by [`lanes()`].
    Meet(Option<ValueId>),
}

/// One step from a whole value towards a part, as [`Op::Part`] takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// A field of a record, a column of a matrix or a component of a
    /// vector, by index: what [`Op::Extract`] takes.
    Field(u32),
    /// The array element, matrix column or vector component at a clamped
    /// `Int`: what [`Op::Subscript`] takes.
    Index(ValueId),
}

/// Where a path through an element leads.
#[derive(Clone, Debug, PartialEq)]
pub struct PartLayout {
    /// The words before the part that are known without running.
    pub base: usize,
    /// The index steps resolved at run time: each index's value, choice
    /// count and word stride.
    pub steps: Vec<(ValueId, usize, usize)>,
    /// What the path names.
    pub ty: Type,
}

/// Where a path through an element of this type leads; `None` when the
/// path does not fit the type.
pub fn part_layout(structs: &[StructDef], element: Type, path: &[Step]) -> Option<PartLayout> {
    let mut ty = element;
    let mut base = 0;
    let mut steps = Vec::new();
    for step in path {
        match step {
            Step::Field(k) => {
                let (offset, field) = part_at(structs, ty, *k as usize)?;
                base += offset;
                ty = field;
            }
            Step::Index(index) => {
                let (part, count, stride) = elements_of(structs, ty)?;
                steps.push((*index, count, stride));
                ty = part;
            }
        }
    }
    Some(PartLayout { base, steps, ty })
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

/// A function: arguments, a top-level block and results. Function `i`
/// calls only functions below `i`, so none is recursive.
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
    /// The `let` bindings in source order with the values they name, for
    /// inspection and traces; they carry no meaning.
    pub bindings: Vec<(ValueId, String)>,
}

impl Function {
    /// Number of value slots the function defines, nested blocks included.
    pub fn slot_count(&self) -> usize {
        fn count(instructions: &[Instruction]) -> usize {
            instructions
                .iter()
                .map(|instruction| {
                    1 + instruction
                        .op
                        .blocks()
                        .iter()
                        .map(|block| count(&block.body))
                        .sum::<usize>()
                })
                .sum()
        }
        count(&self.body)
    }

    /// The type of the instruction defining `id` at any depth; `None` for
    /// an argument or a value the function lacks.
    pub fn type_of(&self, id: ValueId) -> Option<Type> {
        fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
            instructions.iter().find_map(|instruction| {
                if instruction.id == id {
                    return Some(instruction.ty);
                }
                instruction
                    .op
                    .blocks()
                    .into_iter()
                    .find_map(|block| find(&block.body, id))
            })
        }
        find(&self.body, id)
    }

    /// The type of the first result, as a caller receives it; `Float` when
    /// no instruction defines it.
    pub fn result_type(&self) -> Type {
        self.results
            .first()
            .and_then(|&result| self.type_of(result))
            .unwrap_or(Type::Float)
    }
}

impl Function {
    /// Whether any argument or computed value matches `holds`.
    pub fn any_type(&self, holds: &dyn Fn(Type) -> bool) -> bool {
        fn walk(instructions: &[Instruction], holds: &dyn Fn(Type) -> bool) -> bool {
            instructions.iter().any(|instruction| {
                holds(instruction.ty)
                    || instruction
                        .op
                        .blocks()
                        .iter()
                        .any(|block| walk(&block.body, holds))
            })
        }
        self.args.iter().any(|ty| holds(*ty)) || walk(&self.body, holds)
    }
}

/// The most loop iterations a block can run; `functions` holds each
/// called function's own.
fn block_work(instructions: &[Instruction], functions: &[u64]) -> u64 {
    instructions.iter().fold(0u64, |sum, instruction| {
        let work = match &instruction.op {
            Op::Loop {
                count, until, body, ..
            } => {
                let inner = block_work(&body.body, functions)
                    .saturating_add(until.as_ref().map_or(0, |b| block_work(&b.body, functions)));
                u64::from(*count).saturating_mul(inner.saturating_add(1))
            }
            // Its runs are not iterations: the loops in it, each run.
            Op::Each { count, body } => {
                u64::from(*count).saturating_mul(block_work(&body.body, functions))
            }
            // A group's lanes: each lane's own.
            Op::Lanes { body } => block_work(&body.body, functions),
            Op::If {
                then, otherwise, ..
            } => block_work(&then.body, functions).max(block_work(&otherwise.body, functions)),
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter()
                .map(|(_, block)| block_work(&block.body, functions))
                .fold(block_work(&otherwise.body, functions), u64::max),
            Op::Call(index, _) => functions.get(*index as usize).copied().unwrap_or(0),
            _ => 0,
        };
        sum.saturating_add(work)
    })
}

/// The iterations [`Module::typical_work`] counts for a loop whose count
/// is read at run time or that may stop early.
pub const TYPICAL_OPEN_LOOP: u32 = 4;

fn typical_block(instructions: &[Instruction], functions: &[u64], params: &[Value]) -> u64 {
    let mut defined = std::collections::HashMap::new();
    defining(instructions, &mut defined);
    typical_in(instructions, functions, &defined, params)
}

/// The count a loop read at run time takes when it is a parameter given
/// in `params`, clamped to the loop's `bound`.
fn param_count(
    bound: Option<ValueId>,
    count: u32,
    defined: &std::collections::HashMap<ValueId, &Op>,
    params: &[Value],
) -> Option<u32> {
    let Some(Op::Param(index)) = defined.get(&bound?) else {
        return None;
    };
    match params.get(*index as usize)? {
        Value::Int(n) => Some((*n).clamp(0, count as i32) as u32),
        _ => None,
    }
}

/// Whether a loop in `instructions` counts a parameter, without stopping
/// early: [`Module::typical_work_given`] counts it at the parameter's value.
fn counts_param(
    instructions: &[Instruction],
    defined: &std::collections::HashMap<ValueId, &Op>,
) -> bool {
    instructions.iter().any(|instruction| {
        let own = matches!(
            &instruction.op,
            Op::Loop { bound: Some(bound), until: None, .. }
                if matches!(defined.get(bound), Some(Op::Param(_)))
        );
        own || instruction
            .op
            .blocks()
            .into_iter()
            .any(|block| counts_param(&block.body, defined))
    })
}

/// Every value's defining operation, nested blocks included.
fn defining<'a>(
    instructions: &'a [Instruction],
    out: &mut std::collections::HashMap<ValueId, &'a Op>,
) {
    for instruction in instructions {
        out.insert(instruction.id, &instruction.op);
        for block in instruction.op.blocks() {
            defining(&block.body, out);
        }
    }
}

/// Whether a loop's count is a number of rounds of the loop limit, a
/// count divided by [`MAX_LOOP_COUNT`] or more (`(n + 65535) / 65536`):
/// one round for any count one loop can take.
fn counts_rounds(bound: ValueId, defined: &std::collections::HashMap<ValueId, &Op>) -> bool {
    let Some(Op::Binary(BinaryOp::Div, _, by)) = defined.get(&bound) else {
        return false;
    };
    matches!(defined.get(by), Some(Op::Const(Value::Int(limit))) if i64::from(*limit) >= i64::from(MAX_LOOP_COUNT))
}

fn typical_in(
    instructions: &[Instruction],
    functions: &[u64],
    defined: &std::collections::HashMap<ValueId, &Op>,
    params: &[Value],
) -> u64 {
    instructions.iter().fold(0u64, |sum, instruction| {
        let work = match &instruction.op {
            Op::Loop {
                count,
                bound,
                until,
                body,
                ..
            } => {
                let inner = typical_in(&body.body, functions, defined, params).saturating_add(
                    until
                        .as_ref()
                        .map_or(0, |b| typical_in(&b.body, functions, defined, params)),
                );
                let given = until
                    .is_none()
                    .then(|| param_count(*bound, *count, defined, params))
                    .flatten();
                let runs = if let Some(given) = given {
                    given
                } else if bound.is_some_and(|bound| counts_rounds(bound, defined)) {
                    1
                } else if bound.is_some() || until.is_some() {
                    (*count).min(TYPICAL_OPEN_LOOP)
                } else {
                    *count
                };
                u64::from(runs).saturating_mul(inner.saturating_add(1))
            }
            Op::Each { count, body } => {
                u64::from(*count).saturating_mul(typical_in(&body.body, functions, defined, params))
            }
            Op::Lanes { body } => typical_in(&body.body, functions, defined, params),
            Op::If {
                then, otherwise, ..
            } => typical_in(&then.body, functions, defined, params).max(typical_in(
                &otherwise.body,
                functions,
                defined,
                params,
            )),
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter()
                .map(|(_, block)| typical_in(&block.body, functions, defined, params))
                .fold(
                    typical_in(&otherwise.body, functions, defined, params),
                    u64::max,
                ),
            Op::Call(index, _) => functions.get(*index as usize).copied().unwrap_or(0),
            _ => 0,
        };
        sum.saturating_add(work).saturating_add(1)
    })
}

/// The most iterations one run of a loop, nested ones included, may take
/// while its work is checked only at its end. A loop that can run more
/// also checks at the start of each iteration, so an element runs at most
/// this far past a bound. An innermost loop costs one addition.
pub const LOOK_WORK: u64 = 1 << 20;

/// Whether a loop of `count` iterations checks its element's work at each
/// iteration; `functions` is [`Module::functions_work`]. See [`LOOK_WORK`].
pub fn loop_looks(count: u32, until: Option<&Block>, body: &Block, functions: &[u64]) -> bool {
    let inner = block_work(&body.body, functions)
        .saturating_add(until.map_or(0, |until| block_work(&until.body, functions)));
    u64::from(count).saturating_mul(inner.saturating_add(1)) > LOOK_WORK
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

impl Module {
    /// The most loop iterations one element can run: every loop at its bound
    /// ([`MAX_LOOP_COUNT`] for a run-time count), nested loops multiplied,
    /// the longer branch arm, calls counted where made. Saturates.
    pub fn worst_work(&self) -> u64 {
        let per_function = self.functions_work();
        std::iter::once(&self.entry)
            .chain(&self.merge)
            .chain(&self.init)
            .map(|function| block_work(&function.body, &per_function))
            .max()
            .unwrap_or(0)
    }

    /// The operations one element takes as a rule, at least one: each
    /// instruction one; a loop of a written count runs its body that many
    /// times, one whose count is read at run time, or that may stop early,
    /// [`TYPICAL_OPEN_LOOP`] times at most, and one counting rounds of the
    /// loop limit once; the longer branch arm; a call
    /// what its function takes. An estimate of size, for choosing where a
    /// run goes, never a bound.
    pub fn typical_work(&self) -> u64 {
        self.typical_work_given(&[])
    }

    /// [`Module::typical_work`] for a run given `params`: a loop that
    /// counts a parameter, and does not stop early, runs its body that
    /// parameter's value of times, clamped to its bound.
    pub fn typical_work_given(&self, params: &[Value]) -> u64 {
        let mut per_function: Vec<u64> = Vec::with_capacity(self.functions.len());
        for function in &self.functions {
            let work = typical_block(&function.body, &per_function, params);
            per_function.push(work);
        }
        std::iter::once(&self.entry)
            .chain(&self.merge)
            .chain(&self.init)
            .map(|function| typical_block(&function.body, &per_function, params))
            .max()
            .unwrap_or(0)
            .max(1)
    }

    /// Whether a run's [`Module::typical_work_given`] depends on its
    /// parameters: some loop counts one.
    pub fn typical_reads_params(&self) -> bool {
        self.functions
            .iter()
            .chain(std::iter::once(&self.entry))
            .chain(&self.merge)
            .chain(&self.init)
            .any(|function| {
                let mut defined = std::collections::HashMap::new();
                defining(&function.body, &mut defined);
                counts_param(&function.body, &defined)
            })
    }

    /// Whether [`account()`] has placed the module's work checks.
    pub fn accounted(&self) -> bool {
        std::iter::once(&self.entry)
            .chain(&self.merge)
            .chain(&self.init)
            .any(|function| match function.body.last() {
                Some(i) if i.op == Op::Look => true,
                // A group's entry its lanes run in looks in its last lanes.
                Some(i) if i.op == Op::Meet(None) => function.body.iter().any(|i| {
                    matches!(&i.op, Op::Lanes { body } if body.body.iter().any(|i| i.op == Op::Look))
                }),
                _ => false,
            })
    }

    /// For each function, the most loop iterations one call may run.
    pub fn functions_work(&self) -> Vec<u64> {
        let mut per_function: Vec<u64> = Vec::with_capacity(self.functions.len());
        for function in &self.functions {
            let work = block_work(&function.body, &per_function);
            per_function.push(work);
        }
        per_function
    }

    /// Whether a backend counts this module's work as it runs: only when
    /// its worst case can pass its budget or fill a [`WORK_BLOCK`].
    pub fn counts_work(&self) -> bool {
        let worst = self.worst_work();
        worst > u64::from(self.manifest.budget) || worst >= u64::from(WORK_BLOCK)
    }

    /// Whether any port, parameter, struct field or computed value is made
    /// of `scalar`.
    pub fn holds(&self, scalar: Type) -> bool {
        let made_of = |ty: Type| ty.scalar() == Some(scalar);
        let kind = scalar_kind(scalar);
        let manifest = &self.manifest;
        manifest
            .inputs
            .iter()
            .chain(&manifest.outputs)
            .any(|port| port.ty.schema.scalars().iter().any(|s| Some(*s) == kind))
            || manifest.params.iter().any(|param| made_of(param.ty))
            || self
                .structs
                .iter()
                .any(|def| def.fields.iter().any(|field| made_of(field.ty)))
            || self
                .functions
                .iter()
                .chain(std::iter::once(&self.entry))
                .chain(&self.merge)
                .chain(&self.init)
                .any(|function| function.any_type(&made_of))
    }
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
