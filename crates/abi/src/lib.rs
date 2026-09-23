#![warn(missing_docs)]
//! Capability ABI (L2).
//!
//! This crate defines the world visible to a Kernel: value types, opaque
//! handles, named capabilities, and a [`Scope`] that gates every Host call on
//! an explicit grant. A Kernel never touches the Host directly; backends call
//! the Host on its behalf through a `Scope` built from the capabilities the
//! Kernel declared. No capability means no access.

pub mod names;

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

pub use orla_host::{
    Domain, Encoding, FieldData, Handle, Host, HostError, Image, Pixel, Scalar, Schema, Shape,
};
pub use orla_host::{encoding, file};

/// A named authority a Kernel may declare and be granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Read elements and extents of input Fields, images included.
    FieldRead,
    /// Create and write output Fields, images included.
    FieldWrite,
    /// Read the session clock.
    TimeRead,
    /// Read pointer and key state, and the window they arrive through:
    /// its size in pixels and its scale factor.
    InputRead,
}

impl Capability {
    /// All capabilities known to this ABI version.
    pub const ALL: [Capability; 4] = [
        Capability::FieldRead,
        Capability::FieldWrite,
        Capability::TimeRead,
        Capability::InputRead,
    ];

    /// Stable spelling used in `.orla` sources and manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::FieldRead => "field.read",
            Self::FieldWrite => "field.write",
            Self::TimeRead => "time.read",
            Self::InputRead => "input.read",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The name does not correspond to any capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownCapability(pub String);

impl fmt::Display for UnknownCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown capability `{}`", self.0)
    }
}

impl std::error::Error for UnknownCapability {}

impl FromStr for Capability {
    type Err = UnknownCapability;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|capability| capability.name() == name)
            .ok_or_else(|| UnknownCapability(name.to_owned()))
    }
}

/// An explicit set of capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Capabilities(BTreeSet<Capability>);

impl Capabilities {
    /// The empty set: no access at all.
    pub fn none() -> Self {
        Self::default()
    }

    /// Whether the set grants `capability`.
    pub fn contains(&self, capability: Capability) -> bool {
        self.0.contains(&capability)
    }

    /// Adds a capability.
    pub fn insert(&mut self, capability: Capability) {
        self.0.insert(capability);
    }

    /// Iterates capabilities in stable order.
    pub fn iter(&self) -> impl Iterator<Item = Capability> + '_ {
        self.0.iter().copied()
    }

    /// Whether every capability in `self` is also in `other`.
    pub fn is_subset(&self, other: &Capabilities) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl FromIterator<Capability> for Capabilities {
    fn from_iter<I: IntoIterator<Item = Capability>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for capability in self.iter() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "{capability}")?;
        }
        Ok(())
    }
}

/// The version of the Capability ABI: what a Kernel may ask of the Host
/// and how values cross. Bumped when a Kernel built against the old one
/// would no longer bind.
pub const ABI_VERSION: u32 = 1;

/// Types that cross the Kernel boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// 32-bit float.
    Float,
    /// 32-bit signed integer.
    Int,
    /// A truth value.
    Bool,
    /// Two floats.
    Vec2,
    /// Three floats.
    Vec3,
    /// Four floats.
    Vec4,
    /// Two integers: a grid coordinate, a cell address.
    IVec2,
    /// Three integers: a volume coordinate.
    IVec3,
    /// Two columns of two floats, column-major.
    Mat2,
    /// Three columns of three floats, column-major.
    Mat3,
    /// Four columns of four floats, column-major.
    Mat4,
    /// 64-bit float: two words, the low one first.
    Double,
    /// 64-bit signed integer: two words, the low one first.
    Long,
    /// Two doubles.
    DVec2,
    /// Three doubles.
    DVec3,
    /// Four doubles.
    DVec4,
    /// Two longs.
    LVec2,
    /// Three longs.
    LVec3,
    /// Two columns of two doubles, column-major.
    DMat2,
    /// Three columns of three doubles, column-major.
    DMat3,
    /// Four columns of four doubles, column-major.
    DMat4,
    /// A Kernel-defined record, by index into the module's struct table.
    /// Structs are internal to a Kernel: they never cross the ABI as
    /// parameters or ports.
    Struct(u32),
}

impl Type {
    /// Stable spelling used in `.orla` sources; structs print by index.
    pub fn name(self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Int => "Int",
            Self::Bool => "Bool",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Vec4 => "Vec4",
            Self::IVec2 => "IVec2",
            Self::IVec3 => "IVec3",
            Self::Mat2 => "Mat2",
            Self::Mat3 => "Mat3",
            Self::Mat4 => "Mat4",
            Self::Double => "Double",
            Self::Long => "Long",
            Self::DVec2 => "DVec2",
            Self::DVec3 => "DVec3",
            Self::DVec4 => "DVec4",
            Self::LVec2 => "LVec2",
            Self::LVec3 => "LVec3",
            Self::DMat2 => "DMat2",
            Self::DMat3 => "DMat3",
            Self::DMat4 => "DMat4",
            Self::Struct(_) => "struct",
        }
    }

    /// What a value of this type is made of, and how many: the scalar
    /// type of a vector's components or a matrix's entries, or a scalar
    /// itself once. `None` for a struct.
    pub fn shape(self) -> Option<(Type, u8)> {
        Some(match self {
            Self::Float | Self::Int | Self::Bool | Self::Double | Self::Long => (self, 1),
            Self::Vec2 => (Self::Float, 2),
            Self::Vec3 => (Self::Float, 3),
            Self::Vec4 | Self::Mat2 => (Self::Float, 4),
            Self::Mat3 => (Self::Float, 9),
            Self::Mat4 => (Self::Float, 16),
            Self::IVec2 => (Self::Int, 2),
            Self::IVec3 => (Self::Int, 3),
            Self::DVec2 => (Self::Double, 2),
            Self::DVec3 => (Self::Double, 3),
            Self::DVec4 | Self::DMat2 => (Self::Double, 4),
            Self::DMat3 => (Self::Double, 9),
            Self::DMat4 => (Self::Double, 16),
            Self::LVec2 => (Self::Long, 2),
            Self::LVec3 => (Self::Long, 3),
            Self::Struct(_) => return None,
        })
    }

    /// The scalar type a value of this type is made of: itself for a
    /// scalar, the components' for a vector or a matrix.
    pub fn scalar(self) -> Option<Type> {
        self.shape().map(|(scalar, _)| scalar)
    }

    /// Whether this is a 64-bit scalar, or a vector or matrix of them.
    pub fn is_64bit(self) -> bool {
        matches!(self.scalar(), Some(Self::Double | Self::Long))
    }

    /// Whether this is a float of either width, or a vector or matrix of
    /// them.
    pub fn is_float_kind(self) -> bool {
        matches!(self.scalar(), Some(Self::Float | Self::Double))
    }

    /// Whether this is an integer of either width, or a vector of them.
    pub fn is_int_kind(self) -> bool {
        matches!(self.scalar(), Some(Self::Int | Self::Long))
    }

    /// Words a value of this type takes: two per 64-bit scalar. `None`
    /// for a struct, whose words its table knows.
    pub fn words(self) -> Option<usize> {
        let (scalar, n) = self.shape()?;
        let width = if scalar.is_64bit() { 2 } else { 1 };
        Some(usize::from(n) * width)
    }

    /// The type of this shape made of `scalar` instead: `Vec3` of
    /// `Double` is `DVec3`, `Double` of `Int` is `Int`. `None` when there
    /// is no such type (`Vec4` of `Int`).
    pub fn with_scalar(self, scalar: Type) -> Option<Type> {
        if let Some(n) = self.order() {
            return Self::matrix_of_order(scalar, n);
        }
        let (_, n) = self.shape()?;
        Self::vector(scalar, n)
    }

    /// The vector of `n` components of `scalar`, or the scalar itself
    /// when `n` is one.
    pub fn vector(scalar: Type, n: u8) -> Option<Type> {
        Some(match (scalar, n) {
            (Self::Float | Self::Int | Self::Bool | Self::Double | Self::Long, 1) => scalar,
            (Self::Float, 2) => Self::Vec2,
            (Self::Float, 3) => Self::Vec3,
            (Self::Float, 4) => Self::Vec4,
            (Self::Int, 2) => Self::IVec2,
            (Self::Int, 3) => Self::IVec3,
            (Self::Double, 2) => Self::DVec2,
            (Self::Double, 3) => Self::DVec3,
            (Self::Double, 4) => Self::DVec4,
            (Self::Long, 2) => Self::LVec2,
            (Self::Long, 3) => Self::LVec3,
            _ => return None,
        })
    }

    fn matrix_of_order(scalar: Type, n: u8) -> Option<Type> {
        Some(match (scalar, n) {
            (Self::Float, 2) => Self::Mat2,
            (Self::Float, 3) => Self::Mat3,
            (Self::Float, 4) => Self::Mat4,
            (Self::Double, 2) => Self::DMat2,
            (Self::Double, 3) => Self::DMat3,
            (Self::Double, 4) => Self::DMat4,
            _ => return None,
        })
    }

    /// The order of a square matrix type: its columns, each a vector of
    /// as many floats or doubles.
    pub fn order(self) -> Option<u8> {
        match self {
            Self::Mat2 | Self::DMat2 => Some(2),
            Self::Mat3 | Self::DMat3 => Some(3),
            Self::Mat4 | Self::DMat4 => Some(4),
            _ => None,
        }
    }

    /// The column type of a matrix type.
    pub fn column(self) -> Option<Type> {
        Self::vector(self.scalar()?, self.order()?)
    }

    /// The matrix type whose columns are this vector type.
    pub fn matrix_of(self) -> Option<Type> {
        let (n, scalar) = self.lanes()?;
        Self::matrix_of_order(scalar, n)
    }

    /// Number of float components for 32-bit float vector types.
    pub fn components(self) -> Option<u8> {
        match self {
            Self::Vec2 => Some(2),
            Self::Vec3 => Some(3),
            Self::Vec4 => Some(4),
            _ => None,
        }
    }

    /// Number of integer components for 32-bit integer vector types.
    pub fn int_components(self) -> Option<u8> {
        match self {
            Self::IVec2 => Some(2),
            Self::IVec3 => Some(3),
            _ => None,
        }
    }

    /// Number of components of a vector of any kind, and their type.
    pub fn lanes(self) -> Option<(u8, Type)> {
        match self.shape()? {
            (_, 1) => None,
            _ if self.order().is_some() => None,
            (scalar, n) => Some((n, scalar)),
        }
    }

    /// The float vector with as many components as this integer vector,
    /// and the other way round: what `float` and `int` convert between.
    pub fn other_kind(self) -> Option<Type> {
        match self {
            Self::Vec2 => Some(Self::IVec2),
            Self::Vec3 => Some(Self::IVec3),
            Self::IVec2 => Some(Self::Vec2),
            Self::IVec3 => Some(Self::Vec3),
            _ => None,
        }
    }

    /// Whether values of this type are plain data that can cross the ABI
    /// (not Kernel-internal structs).
    pub fn is_value(self) -> bool {
        !matches!(self, Self::Struct(_))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Struct(index) => write!(f, "S{index}"),
            other => f.write_str(other.name()),
        }
    }
}

/// The name does not correspond to any type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownType(pub String);

impl fmt::Display for UnknownType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown type `{}`", self.0)
    }
}

impl std::error::Error for UnknownType {}

impl Type {
    /// Every type a name may spell, for a diagnostic that wants to
    /// suggest the one that was meant.
    pub const NAMED: [Self; 21] = [
        Self::Float,
        Self::Int,
        Self::Bool,
        Self::Vec2,
        Self::Vec3,
        Self::Vec4,
        Self::IVec2,
        Self::IVec3,
        Self::Mat2,
        Self::Mat3,
        Self::Mat4,
        Self::Double,
        Self::Long,
        Self::DVec2,
        Self::DVec3,
        Self::DVec4,
        Self::LVec2,
        Self::LVec3,
        Self::DMat2,
        Self::DMat3,
        Self::DMat4,
    ];
}

impl FromStr for Type {
    type Err = UnknownType;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::NAMED
            .into_iter()
            .find(|ty| ty.name() == name)
            .ok_or_else(|| UnknownType(name.to_owned()))
    }
}

/// A value crossing the Kernel boundary: a parameter, a constant. Any
/// value a Kernel can hold, a struct's fields and an array's elements as
/// a [`Value::Record`] in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A float.
    Float(f32),
    /// An integer.
    Int(i32),
    /// A truth value.
    Bool(bool),
    /// A two-component vector.
    Vec2([f32; 2]),
    /// A three-component vector.
    Vec3([f32; 3]),
    /// A four-component vector.
    Vec4([f32; 4]),
    /// A two-component integer vector.
    IVec2([i32; 2]),
    /// A three-component integer vector.
    IVec3([i32; 3]),
    /// A 2x2 matrix, column by column.
    Mat2([f32; 4]),
    /// A 3x3 matrix, column by column.
    Mat3([f32; 9]),
    /// A 4x4 matrix, column by column.
    Mat4([f32; 16]),
    /// A 64-bit float.
    Double(f64),
    /// A 64-bit integer.
    Long(i64),
    /// Two doubles.
    DVec2([f64; 2]),
    /// Three doubles.
    DVec3([f64; 3]),
    /// Four doubles.
    DVec4([f64; 4]),
    /// Two longs.
    LVec2([i64; 2]),
    /// Three longs.
    LVec3([i64; 3]),
    /// A 2x2 matrix of doubles, column by column.
    DMat2([f64; 4]),
    /// A 3x3 matrix of doubles, column by column.
    DMat3([f64; 9]),
    /// A 4x4 matrix of doubles, column by column.
    DMat4([f64; 16]),
    /// A struct's fields, or an array's elements, in order. Its type is
    /// the struct or array a Kernel declares; what it must match is the
    /// layout, never a name.
    Record(Vec<Value>),
}

/// One scalar of a value, whatever its kind: what a value is taken
/// apart into to be laid out, converted, or rebuilt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    /// An `f32`.
    Float(f32),
    /// An `i32`.
    Int(i32),
    /// A truth value.
    Bool(bool),
    /// An `f64`.
    Double(f64),
    /// An `i64`.
    Long(i64),
}

impl Number {
    /// The kind of word this scalar is kept as.
    pub fn kind(self) -> Scalar {
        match self {
            Self::Float(_) => Scalar::Float,
            Self::Int(_) => Scalar::Int,
            Self::Bool(_) => Scalar::Bool,
            Self::Double(_) => Scalar::Double,
            Self::Long(_) => Scalar::Long,
        }
    }

    /// Its words: one, or two for a 64-bit scalar, the low one first.
    pub fn push_words(self, out: &mut Vec<u32>) {
        match self {
            Self::Float(v) => out.push(v.to_bits()),
            Self::Int(v) => out.push(v as u32),
            Self::Bool(v) => out.push(u32::from(v)),
            Self::Double(v) => split(v.to_bits(), out),
            Self::Long(v) => split(v as u64, out),
        }
    }

    /// This scalar as one of `kind`, when it converts: an integer to a
    /// float of either width or to a wider integer, a float to the other
    /// width, a long to an integer that holds it. A written number has no
    /// width of its own; the place it is given to decides. This is how a
    /// value given is taken, never how a Kernel computes: `double(x)` in
    /// a Kernel is the float's own value.
    pub fn as_kind(self, kind: Scalar) -> Option<Number> {
        Some(match (self, kind) {
            (Self::Float(v), Scalar::Float) => Self::Float(v),
            // A float given where a double is wanted is the decimal it
            // reads as: `0.001` written where no width was said is the
            // double nearest a thousandth, not the float's nearest.
            (Self::Float(v), Scalar::Double) => {
                Self::Double(v.to_string().parse().unwrap_or(f64::from(v)))
            }
            (Self::Double(v), Scalar::Double) => Self::Double(v),
            (Self::Double(v), Scalar::Float) => Self::Float(v as f32),
            (Self::Int(v), Scalar::Int) => Self::Int(v),
            (Self::Int(v), Scalar::Float) => Self::Float(v as f32),
            (Self::Int(v), Scalar::Double) => Self::Double(f64::from(v)),
            (Self::Int(v), Scalar::Long) => Self::Long(i64::from(v)),
            (Self::Long(v), Scalar::Long) => Self::Long(v),
            (Self::Long(v), Scalar::Int) => Self::Int(i32::try_from(v).ok()?),
            (Self::Long(v), Scalar::Float) => Self::Float(v as f32),
            (Self::Long(v), Scalar::Double) => Self::Double(v as f64),
            (Self::Bool(v), Scalar::Bool) => Self::Bool(v),
            _ => return None,
        })
    }
}

fn split(bits: u64, out: &mut Vec<u32>) {
    out.push(bits as u32);
    out.push((bits >> 32) as u32);
}

impl Value {
    /// The type of a value that is not a record; a record's type is a
    /// struct of the Kernel that takes it.
    pub fn ty(&self) -> Option<Type> {
        Some(match self {
            Self::Float(_) => Type::Float,
            Self::Int(_) => Type::Int,
            Self::Bool(_) => Type::Bool,
            Self::Vec2(_) => Type::Vec2,
            Self::Vec3(_) => Type::Vec3,
            Self::Vec4(_) => Type::Vec4,
            Self::IVec2(_) => Type::IVec2,
            Self::IVec3(_) => Type::IVec3,
            Self::Mat2(_) => Type::Mat2,
            Self::Mat3(_) => Type::Mat3,
            Self::Mat4(_) => Type::Mat4,
            Self::Double(_) => Type::Double,
            Self::Long(_) => Type::Long,
            Self::DVec2(_) => Type::DVec2,
            Self::DVec3(_) => Type::DVec3,
            Self::DVec4(_) => Type::DVec4,
            Self::LVec2(_) => Type::LVec2,
            Self::LVec3(_) => Type::LVec3,
            Self::DMat2(_) => Type::DMat2,
            Self::DMat3(_) => Type::DMat3,
            Self::DMat4(_) => Type::DMat4,
            Self::Record(_) => return None,
        })
    }

    /// Its scalars in layout order.
    pub fn numbers(&self) -> Vec<Number> {
        let mut out = Vec::new();
        self.push_numbers(&mut out);
        out
    }

    fn push_numbers(&self, out: &mut Vec<Number>) {
        let floats =
            |out: &mut Vec<Number>, v: &[f32]| out.extend(v.iter().map(|x| Number::Float(*x)));
        let ints = |out: &mut Vec<Number>, v: &[i32]| out.extend(v.iter().map(|x| Number::Int(*x)));
        let doubles =
            |out: &mut Vec<Number>, v: &[f64]| out.extend(v.iter().map(|x| Number::Double(*x)));
        let longs =
            |out: &mut Vec<Number>, v: &[i64]| out.extend(v.iter().map(|x| Number::Long(*x)));
        match self {
            Self::Float(v) => out.push(Number::Float(*v)),
            Self::Int(v) => out.push(Number::Int(*v)),
            Self::Bool(v) => out.push(Number::Bool(*v)),
            Self::Double(v) => out.push(Number::Double(*v)),
            Self::Long(v) => out.push(Number::Long(*v)),
            Self::Vec2(v) => floats(out, v),
            Self::Vec3(v) => floats(out, v),
            Self::Vec4(v) => floats(out, v),
            Self::Mat2(v) => floats(out, v),
            Self::Mat3(v) => floats(out, v),
            Self::Mat4(v) => floats(out, v),
            Self::IVec2(v) => ints(out, v),
            Self::IVec3(v) => ints(out, v),
            Self::DVec2(v) => doubles(out, v),
            Self::DVec3(v) => doubles(out, v),
            Self::DVec4(v) => doubles(out, v),
            Self::DMat2(v) => doubles(out, v),
            Self::DMat3(v) => doubles(out, v),
            Self::DMat4(v) => doubles(out, v),
            Self::LVec2(v) => longs(out, v),
            Self::LVec3(v) => longs(out, v),
            Self::Record(members) => {
                for member in members {
                    member.push_numbers(out);
                }
            }
        }
    }

    /// The Kernel's words, one per 32-bit scalar and two per 64-bit one,
    /// in layout order.
    pub fn words(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for number in self.numbers() {
            number.push_words(&mut out);
        }
        out
    }

    /// What each word is, in layout order.
    pub fn scalars(&self) -> Vec<Scalar> {
        let mut out = Vec::new();
        for number in self.numbers() {
            let kind = number.kind();
            out.extend(std::iter::repeat_n(kind, kind.words()));
        }
        out
    }

    /// This value in `template`'s shape, when its scalars are the
    /// template's in order, each converting to the template's kind
    /// ([`Number::as_kind`]). What a value must match is a layout, never
    /// a spelling: `[[1, 0], [0, 1]]` is a `Mat2`, and `[vec3(0.0, 1.0,
    /// 0.0), 2]` a struct of a `Vec3` and a `Float`.
    pub fn shaped_like(&self, template: &Value) -> Option<Value> {
        let words = widened(self, &template.scalars())?;
        let mut words = words.into_iter();
        let shaped = template.rebuilt(&mut words);
        Some(shaped)
    }

    /// A value of this one's shape made of `words`, taken in order.
    pub fn rebuilt(&self, words: &mut impl Iterator<Item = u32>) -> Value {
        let mut next = || words.next().unwrap_or(0);
        macro_rules! lanes {
            ($n:expr, $zero:expr, $read:expr) => {{
                let mut out = [$zero; $n];
                for slot in &mut out {
                    *slot = $read(&mut next);
                }
                out
            }};
        }
        let float = |next: &mut dyn FnMut() -> u32| f32::from_bits(next());
        let int = |next: &mut dyn FnMut() -> u32| next() as i32;
        let wide = |next: &mut dyn FnMut() -> u32| {
            let low = u64::from(next());
            low | (u64::from(next()) << 32)
        };
        let double = |next: &mut dyn FnMut() -> u32| f64::from_bits(wide(next));
        let long = |next: &mut dyn FnMut() -> u32| wide(next) as i64;
        match self {
            Self::Float(_) => Self::Float(float(&mut next)),
            Self::Int(_) => Self::Int(int(&mut next)),
            Self::Bool(_) => Self::Bool(next() != 0),
            Self::Vec2(_) => Self::Vec2(lanes!(2, 0.0, float)),
            Self::Vec3(_) => Self::Vec3(lanes!(3, 0.0, float)),
            Self::Vec4(_) => Self::Vec4(lanes!(4, 0.0, float)),
            Self::IVec2(_) => Self::IVec2(lanes!(2, 0, int)),
            Self::IVec3(_) => Self::IVec3(lanes!(3, 0, int)),
            Self::Mat2(_) => Self::Mat2(lanes!(4, 0.0, float)),
            Self::Mat3(_) => Self::Mat3(lanes!(9, 0.0, float)),
            Self::Mat4(_) => Self::Mat4(lanes!(16, 0.0, float)),
            Self::Double(_) => Self::Double(double(&mut next)),
            Self::Long(_) => Self::Long(long(&mut next)),
            Self::DVec2(_) => Self::DVec2(lanes!(2, 0.0, double)),
            Self::DVec3(_) => Self::DVec3(lanes!(3, 0.0, double)),
            Self::DVec4(_) => Self::DVec4(lanes!(4, 0.0, double)),
            Self::LVec2(_) => Self::LVec2(lanes!(2, 0, long)),
            Self::LVec3(_) => Self::LVec3(lanes!(3, 0, long)),
            Self::DMat2(_) => Self::DMat2(lanes!(4, 0.0, double)),
            Self::DMat3(_) => Self::DMat3(lanes!(9, 0.0, double)),
            Self::DMat4(_) => Self::DMat4(lanes!(16, 0.0, double)),
            Self::Record(members) => {
                Self::Record(members.iter().map(|member| member.rebuilt(words)).collect())
            }
        }
    }
}

/// `value`'s words as a layout of `wanted` scalars: as many scalars, in
/// the same order, each converting to the kind wanted
/// ([`Number::as_kind`]); `None` when they are another layout.
pub fn widened(value: &Value, wanted: &[Scalar]) -> Option<Vec<u32>> {
    let mut kinds = Vec::with_capacity(wanted.len());
    let mut k = 0;
    while k < wanted.len() {
        kinds.push(wanted[k]);
        k += wanted[k].words();
    }
    let numbers = value.numbers();
    if numbers.len() != kinds.len() {
        return None;
    }
    let mut out = Vec::with_capacity(wanted.len());
    for (number, kind) in numbers.into_iter().zip(kinds) {
        number.as_kind(kind)?.push_words(&mut out);
    }
    Some(out)
}

impl fmt::Display for Value {
    /// The literal that reads back as this value, in a scene or a replay.
    /// A double outside a constructor that says so carries a `d`, and a
    /// long an `l`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |f: &mut fmt::Formatter<'_>, name: &str, parts: Vec<String>| {
            write!(f, "{name}({})", parts.join(", "))
        };
        fn numbers<T: fmt::Debug>(v: &[T]) -> Vec<String> {
            v.iter().map(|x| format!("{x:?}")).collect()
        }
        fn columns<T: fmt::Debug>(v: &[T], n: usize, name: &str) -> Vec<String> {
            v.chunks(n)
                .map(|column| format!("{name}{n}({})", numbers(column).join(", ")))
                .collect()
        }
        match self {
            Self::Float(value) => write!(f, "{value:?}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Double(value) => write!(f, "{value:?}d"),
            Self::Long(value) => write!(f, "{value}l"),
            Self::Vec2(v) => list(f, "vec2", numbers(v)),
            Self::Vec3(v) => list(f, "vec3", numbers(v)),
            Self::Vec4(v) => list(f, "vec4", numbers(v)),
            Self::IVec2(v) => list(f, "ivec2", numbers(v)),
            Self::IVec3(v) => list(f, "ivec3", numbers(v)),
            Self::Mat2(v) => list(f, "mat2", columns(v, 2, "vec")),
            Self::Mat3(v) => list(f, "mat3", columns(v, 3, "vec")),
            Self::Mat4(v) => list(f, "mat4", columns(v, 4, "vec")),
            Self::DVec2(v) => list(f, "dvec2", numbers(v)),
            Self::DVec3(v) => list(f, "dvec3", numbers(v)),
            Self::DVec4(v) => list(f, "dvec4", numbers(v)),
            Self::LVec2(v) => list(f, "lvec2", numbers(v)),
            Self::LVec3(v) => list(f, "lvec3", numbers(v)),
            Self::DMat2(v) => list(f, "dmat2", columns(v, 2, "dvec")),
            Self::DMat3(v) => list(f, "dmat3", columns(v, 3, "dvec")),
            Self::DMat4(v) => list(f, "dmat4", columns(v, 4, "dvec")),
            Self::Record(members) => write!(
                f,
                "[{}]",
                members
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// What a port carries: a Field of `element`s in one [`Shape`]. An image
/// is the grid of `Vec4`; it has no representation of its own.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PortType {
    /// The element type as the Kernel sees it (`Type::Struct` indexes the
    /// Kernel's struct table).
    pub element: Type,
    /// How elements are addressed.
    pub shape: Shape,
    /// The element's storage layout.
    pub schema: Schema,
}

impl PortType {
    /// The image port: a grid of `Vec4`.
    pub fn image() -> Self {
        Self {
            element: Type::Vec4,
            shape: Shape::Grid,
            schema: Schema::pixel(),
        }
    }

    /// Whether this is the image port.
    pub fn is_image(&self) -> bool {
        self.shape == Shape::Grid && self.element == Type::Vec4 && self.schema.is_wide()
    }

    /// The value type that addresses one element: `Int`, `Vec2`, or `Vec3`.
    pub fn index_type(&self) -> Type {
        self.shape.index_type()
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_image() {
            return f.write_str("Image");
        }
        match self.schema.uniform_encoding() {
            _ if self.schema.is_wide() => write!(f, "{}<{}>", self.shape, self.element),
            Some(encoding) => write!(f, "{}<{} as {encoding}>", self.shape, self.element),
            None => write!(f, "{}<{} as {}>", self.shape, self.element, self.schema),
        }
    }
}

/// Value types that address a shape's elements.
pub trait ShapeIndex {
    /// `Int` for rows, `Vec2` for grids, `Vec3` for volumes.
    fn index_type(self) -> Type;
}

impl ShapeIndex for Shape {
    fn index_type(self) -> Type {
        match self {
            Shape::Rows => Type::Int,
            Shape::Grid => Type::Vec2,
            Shape::Volume => Type::Vec3,
        }
    }
}

/// Pointer and key state observed for one evaluation. Key codes follow the
/// common browser key codes: letters and digits by their ASCII uppercase
/// value, space 32, enter 13, escape 27, arrows 37 to 40; 256 codes fit.
/// Pointer buttons are numbered 0 left, 1 right, 2 middle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputState {
    /// Pointer position in output pixels.
    pub pointer: [f32; 2],
    /// Whether the primary pointer button is held.
    pub pointer_down: bool,
    /// Whether the secondary (right) pointer button is held.
    pub pointer_right: bool,
    /// Whether the middle pointer button is held.
    pub pointer_middle: bool,
    /// What the wheel scrolled since the last evaluation, in lines: `x`
    /// sideways, `y` positive away from the person.
    pub wheel: [f32; 2],
    /// A pad's analog axes, each in `-1..1`: the left stick, the right
    /// stick, and the two triggers as one pair (`x` left, `y` right,
    /// both in `0..1`). Zero without a pad.
    pub sticks: [[f32; 2]; 3],
    /// A pad's buttons as bits, the lowest being [`pad::SOUTH`].
    pad_buttons: u32,
    keys: [u32; 8],
}

/// How many buttons `InputState::button` knows: three of a pointer's,
/// then a pad's from 8.
pub const BUTTONS: i32 = 24;

/// A pad's buttons, by the number `button()` takes.
pub mod pad {
    /// The lower face button (A, Cross).
    pub const SOUTH: i32 = 8;
    /// The right face button (B, Circle).
    pub const EAST: i32 = 9;
    /// The left face button (X, Square).
    pub const WEST: i32 = 10;
    /// The upper face button (Y, Triangle).
    pub const NORTH: i32 = 11;
    /// The left shoulder.
    pub const LEFT_SHOULDER: i32 = 12;
    /// The right shoulder.
    pub const RIGHT_SHOULDER: i32 = 13;
    /// The left stick pressed in.
    pub const LEFT_THUMB: i32 = 14;
    /// The right stick pressed in.
    pub const RIGHT_THUMB: i32 = 15;
    /// Start.
    pub const START: i32 = 16;
    /// Select.
    pub const SELECT: i32 = 17;
}

impl InputState {
    /// Whether the button numbered `code` is held: 0 left, 1 right, 2
    /// middle of a pointer, then a pad's from 8 (see [`pad`]); any other
    /// number is never held.
    pub fn button(&self, code: i32) -> bool {
        if !(0..BUTTONS).contains(&code) {
            return false;
        }
        (self.button_bits() >> code) & 1 == 1
    }

    /// Holds or releases a button; numbers outside `0..BUTTONS` are
    /// ignored, and the pointer's three keep their own fields.
    pub fn set_button(&mut self, code: i32, down: bool) {
        match code {
            0 => self.pointer_down = down,
            1 => self.pointer_right = down,
            2 => self.pointer_middle = down,
            code if (8..BUTTONS).contains(&code) => {
                let bit = 1 << (code - 8);
                if down {
                    self.pad_buttons |= bit;
                } else {
                    self.pad_buttons &= !bit;
                }
            }
            _ => {}
        }
    }

    /// The held buttons as bits, `button(n)` at bit `n`.
    pub fn button_bits(&self) -> u32 {
        (self.pointer_down as u32)
            | (self.pointer_right as u32) << 1
            | (self.pointer_middle as u32) << 2
            | (self.pad_buttons & 0xFFFF) << 8
    }

    /// One of the pad's axis pairs, by the number `stick()` takes: 0 the
    /// left stick, 1 the right, 2 the triggers. Any other number is a
    /// pair of zeros.
    pub fn stick(&self, which: i32) -> [f32; 2] {
        match which {
            0..=2 => self.sticks[which as usize],
            _ => [0.0, 0.0],
        }
    }

    /// Whether the key with `code` is held; codes outside `0..256` are never held.
    pub fn key(&self, code: i32) -> bool {
        if !(0..256).contains(&code) {
            return false;
        }
        let code = code as u32;
        (self.keys[(code >> 5) as usize] >> (code & 31)) & 1 == 1
    }

    /// Presses or releases a key; codes outside `0..256` are ignored.
    pub fn set_key(&mut self, code: i32, down: bool) {
        if !(0..256).contains(&code) {
            return;
        }
        let code = code as u32;
        let word = &mut self.keys[(code >> 5) as usize];
        if down {
            *word |= 1 << (code & 31);
        } else {
            *word &= !(1 << (code & 31));
        }
    }

    /// The key set as eight 32-bit words, bit `code` set when held.
    pub fn key_words(&self) -> [u32; 8] {
        self.keys
    }

    /// A value that changes whenever any observable input changes.
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |word: u32| {
            hash ^= word as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        feed(self.pointer[0].to_bits());
        feed(self.pointer[1].to_bits());
        feed(self.button_bits());
        feed(self.wheel[0].to_bits());
        feed(self.wheel[1].to_bits());
        for pair in self.sticks {
            feed(pair[0].to_bits());
            feed(pair[1].to_bits());
        }
        feed(self.pad_buttons);
        for word in self.keys {
            feed(word);
        }
        hash
    }
}

/// `C` in `Y = K(X, P, C)`: what the world looks like for one evaluation.
/// Context is not a resource; Kernels read it through builtins behind
/// `time.read` and `input.read`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Context {
    /// Session time in seconds.
    pub time: f32,
    /// The sound device's clock in samples per second, or zero when no
    /// device is driving the evaluation. A device fact, like time.
    pub rate: f32,
    /// Pointer and key state.
    pub input: InputState,
    /// The window's size in physical pixels, or zero by zero when no
    /// window is open. A device fact, like the sound card's rate: a
    /// node's `domain` may take it, so a picture fills the window at the
    /// window's resolution.
    pub window: [f32; 2],
    /// The window's scale factor: physical pixels per logical pixel, so a
    /// Kernel can lay out a bar at a constant physical size. One without
    /// a window.
    pub scale: f32,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            time: 0.0,
            rate: 0.0,
            input: InputState::default(),
            window: [0.0, 0.0],
            scale: 1.0,
        }
    }
}

impl Context {
    /// Whether a window is open: both sides of `window` are positive.
    pub fn has_window(&self) -> bool {
        self.window[0] > 0.0 && self.window[1] > 0.0
    }

    /// A hash of what `input.read` observes besides time: the input state,
    /// the window's size, and its scale.
    pub fn input_fingerprint(&self) -> u64 {
        let mut hash = self.input.fingerprint();
        for word in [
            self.window[0].to_bits(),
            self.window[1].to_bits(),
            self.scale.to_bits(),
        ] {
            hash ^= word as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}

/// Everything a Kernel receives for one execution: `X`, `P` and `C` in
/// `Y = K(X, P, C)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    /// Field inputs in manifest order.
    pub inputs: Vec<Handle>,
    /// Parameter values in manifest order.
    pub params: Vec<Value>,
    /// Output domain.
    pub domain: Domain,
    /// Time and input for this execution.
    pub context: Context,
}

/// Errors a Kernel can observe when calling the Host through the ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbiError {
    /// The Kernel was not granted the capability the call requires.
    Denied(Capability),
    /// The Host rejected the call.
    Host(HostError),
}

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Denied(capability) => write!(f, "capability `{capability}` was not granted"),
            Self::Host(error) => write!(f, "host rejected the call: {error}"),
        }
    }
}

impl std::error::Error for AbiError {}

impl From<HostError> for AbiError {
    fn from(error: HostError) -> Self {
        Self::Host(error)
    }
}

/// A capability-gated view of the Host.
///
/// Every method checks its capability before touching the Host. Backends
/// construct one `Scope` per execution from the Kernel manifest.
pub struct Scope<'a> {
    host: &'a mut Host,
    granted: Capabilities,
}

impl<'a> Scope<'a> {
    /// Wraps the Host with the given grant.
    pub fn new(host: &'a mut Host, granted: Capabilities) -> Self {
        Self { host, granted }
    }

    /// The capabilities this scope grants.
    pub fn granted(&self) -> &Capabilities {
        &self.granted
    }

    fn require(&self, capability: Capability) -> Result<(), AbiError> {
        if self.granted.contains(capability) {
            Ok(())
        } else {
            Err(AbiError::Denied(capability))
        }
    }

    /// The extent of an input. Requires `field.read`.
    pub fn size(&self, field: Handle) -> Result<Domain, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.domain(field)?)
    }

    /// The element layout of an input. Requires `field.read`.
    pub fn schema(&self, field: Handle) -> Result<Schema, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.schema(field)?.clone())
    }

    /// One element of an input as the Kernel's words, widened if it is
    /// kept narrow, by linear index. Requires `field.read`. Fails while
    /// the Field is resident on a device.
    pub fn element(
        &self,
        field: Handle,
        index: u32,
    ) -> Result<std::borrow::Cow<'_, [u32]>, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.element(field, index)?)
    }

    /// Every element of an input as the Kernel's words, widened where it
    /// is kept narrow. Requires `field.read`.
    pub fn values(&self, field: Handle) -> Result<std::borrow::Cow<'_, [u32]>, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.data(field)?.values())
    }

    /// An input's memory as kept: the Kernel's words for a wide schema,
    /// packed scalars for a narrow one ([`Schema::load`] reads them).
    /// Requires `field.read`.
    pub fn storage(&self, field: Handle) -> Result<&[u32], AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.data(field)?.storage())
    }

    /// The device object of a resident input, if any. Requires `field.read`.
    pub fn resident(
        &self,
        field: Handle,
    ) -> Result<Option<&(dyn std::any::Any + Send + Sync)>, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.resident(field)?)
    }

    /// Moves an input's words onto the device, replacing the Host copy.
    /// Residency is a storage detail, so reading authority suffices.
    pub fn make_resident(
        &mut self,
        field: Handle,
        device: Box<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.make_resident(field, device)?)
    }

    /// A new zeroed Field. Requires `field.write`.
    pub fn create(&mut self, schema: Schema, domain: Domain) -> Result<Handle, AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.create(schema, domain)?)
    }

    /// A new Field whose words are whatever buffer it was given: for a
    /// backend that writes every one of them before anything reads it.
    /// Requires `field.write`.
    pub fn create_for_fill(&mut self, schema: Schema, domain: Domain) -> Result<Handle, AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.create_for_fill(schema, domain)?)
    }

    /// The memory of a Field this scope created, as kept, to fill in
    /// place: what a backend writes into when it would otherwise fill a
    /// buffer of its own and copy it in. Requires `field.write`.
    pub fn storage_mut(&mut self, field: Handle) -> Result<&mut [u32], AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.storage_mut(field)?)
    }

    /// Replaces every element of a Field this scope created with the
    /// Kernel's words, narrowed where the Field keeps them narrow.
    /// Requires `field.write`.
    pub fn write_values(&mut self, field: Handle, values: &[u32]) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.write_values(field, values)?)
    }

    /// Replaces the memory of a Field this scope created, as kept.
    /// Requires `field.write`.
    pub fn write_storage(&mut self, field: Handle, storage: &[u32]) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.write_storage(field, storage)?)
    }

    /// Replaces one element of a Field this scope created. Requires
    /// `field.write`.
    pub fn write_element(
        &mut self,
        field: Handle,
        index: u32,
        words: &[u32],
    ) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.write_element(field, index, words)?)
    }

    /// A new Field whose words live on the device. Requires `field.write`.
    pub fn adopt_resident(
        &mut self,
        schema: Schema,
        domain: Domain,
        device: Box<dyn std::any::Any + Send + Sync>,
    ) -> Result<Handle, AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.adopt_resident(schema, domain, device)?)
    }

    /// Releases a Field this scope created. Requires `field.write`.
    pub fn release(&mut self, field: Handle) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.release(field)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buttons_and_the_wheel_are_input() {
        let mut input = InputState::default();
        let quiet = input.fingerprint();
        input.pointer_right = true;
        assert!(input.button(1) && !input.button(0) && !input.button(7));
        assert_eq!(input.button_bits(), 0b010);
        let right = input.fingerprint();
        assert_ne!(right, quiet);
        input.wheel = [0.0, -1.0];
        assert_ne!(input.fingerprint(), right);
    }

    #[test]
    fn capabilities_round_trip_through_names() {
        for capability in Capability::ALL {
            assert_eq!(capability.name().parse::<Capability>(), Ok(capability));
        }
        assert_eq!(
            "file.write".parse::<Capability>(),
            Err(UnknownCapability("file.write".into()))
        );
    }

    #[test]
    fn scope_denies_ungranted_calls_before_touching_the_host() {
        let mut host = Host::new(64);
        let image = host.create_image(2, 2).unwrap();
        let mut scope = Scope::new(&mut host, Capabilities::none());
        assert_eq!(
            scope.element(image, 0),
            Err(AbiError::Denied(Capability::FieldRead))
        );
        assert_eq!(
            scope.create(Schema::pixel(), Domain::Rows(1)),
            Err(AbiError::Denied(Capability::FieldWrite))
        );
        assert_eq!(host.live(), 1);
    }

    #[test]
    fn scope_forwards_granted_calls_and_host_errors() {
        let mut host = Host::new(64);
        let image = host.create_image(2, 2).unwrap();
        let granted: Capabilities = [Capability::FieldRead, Capability::FieldWrite]
            .into_iter()
            .collect();
        let mut scope = Scope::new(&mut host, granted);
        scope
            .write_element(image, 1, &[0.5f32.to_bits(); 4])
            .unwrap();
        assert_eq!(&*scope.element(image, 1).unwrap(), &[0.5f32.to_bits(); 4]);
        assert_eq!(scope.size(image).unwrap(), Domain::Grid(2, 2));
        assert_eq!(scope.schema(image).unwrap(), Schema::pixel());
        assert!(matches!(
            scope.element(image, 9),
            Err(AbiError::Host(HostError::ElementOutOfBounds { .. }))
        ));
        let created = scope
            .create(Schema::new(vec![Scalar::Int]), Domain::Volume(1, 2, 3))
            .unwrap();
        assert_eq!(scope.size(created).unwrap().count(), 6);
        scope.release(created).unwrap();
        assert_eq!(host.live(), 1);
        assert_eq!(PortType::image().to_string(), "Image");
        let rows = PortType {
            element: Type::Vec2,
            shape: Shape::Rows,
            schema: Schema::new(vec![Scalar::Float; 2]),
        };
        assert_eq!(rows.to_string(), "Rows<Vec2>");
        assert_eq!(rows.index_type(), Type::Int);
    }
}
