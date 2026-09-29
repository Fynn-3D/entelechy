#![warn(missing_docs)]
//! Capability ABI (L2).
//!
//! The world visible to a Kernel: value types, opaque handles, named
//! capabilities, and a [`Scope`] that gates every Host call on an explicit
//! grant. Backends call the Host for a Kernel only through a `Scope` built
//! from its declared capabilities. No capability means no access.

pub mod names;

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

pub use orla_host::{DEFAULT_WORK, MAX_WORK, encoding};
pub use orla_host::{
    Domain, Encoding, FieldData, Handle, Host, HostError, Image, Pixel, Scalar, Schema, Shape,
};

/// A named authority a Kernel may declare and be granted; what each
/// grants is [`Capability::doc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// `field.read`.
    FieldRead,
    /// `field.write`.
    FieldWrite,
    /// `time.read`.
    TimeRead,
    /// `input.read`.
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

    /// What it grants, one sentence: the part of the world a Kernel that
    /// declares it may touch.
    pub fn doc(self) -> &'static str {
        match self {
            Self::FieldRead => {
                "Read the elements and extents of the Fields bound to its inputs, images included."
            }
            Self::FieldWrite => "Create and write its output Fields, images included.",
            Self::TimeRead => "Read the session clock and the sound rate.",
            Self::InputRead => {
                "Read the pointer, buttons, keys, pad and wheel, and the window's pixel size and scale."
            }
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
/// and how values cross.
pub const ABI_VERSION: u32 = 1;

/// Types that cross the Kernel boundary, by structure: a [`Scalar`], a
/// vector of two to four, a square float-kind matrix, or a Kernel's own
/// struct. Each named type has a same-named constant (`Type::Vec3`),
/// usable as a pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// One scalar.
    Scalar(Scalar),
    /// Two to four components of one kind (see [`Type::vector`]).
    Vector(Scalar, u8),
    /// A square float-kind matrix of the given order, column-major.
    Matrix(Scalar, u8),
    /// A Kernel-defined record, by index into the module's struct table.
    /// Structs stay inside a Kernel: they never cross the ABI as
    /// parameters or ports.
    Struct(u32),
}

#[allow(non_upper_case_globals)]
impl Type {
    /// 32-bit float.
    pub const Float: Type = Type::Scalar(Scalar::Float);
    /// 32-bit signed integer.
    pub const Int: Type = Type::Scalar(Scalar::Int);
    /// A truth value.
    pub const Bool: Type = Type::Scalar(Scalar::Bool);
    /// 64-bit float: two words, the low one first.
    pub const Double: Type = Type::Scalar(Scalar::Double);
    /// 64-bit signed integer: two words, the low one first.
    pub const Long: Type = Type::Scalar(Scalar::Long);
    /// Two floats.
    pub const Vec2: Type = Type::Vector(Scalar::Float, 2);
    /// Three floats.
    pub const Vec3: Type = Type::Vector(Scalar::Float, 3);
    /// Four floats.
    pub const Vec4: Type = Type::Vector(Scalar::Float, 4);
    /// Two integers: a grid coordinate, a cell address.
    pub const IVec2: Type = Type::Vector(Scalar::Int, 2);
    /// Three integers: a volume coordinate.
    pub const IVec3: Type = Type::Vector(Scalar::Int, 3);
    /// Four integers.
    pub const IVec4: Type = Type::Vector(Scalar::Int, 4);
    /// Two doubles.
    pub const DVec2: Type = Type::Vector(Scalar::Double, 2);
    /// Three doubles.
    pub const DVec3: Type = Type::Vector(Scalar::Double, 3);
    /// Four doubles.
    pub const DVec4: Type = Type::Vector(Scalar::Double, 4);
    /// Two longs.
    pub const LVec2: Type = Type::Vector(Scalar::Long, 2);
    /// Three longs.
    pub const LVec3: Type = Type::Vector(Scalar::Long, 3);
    /// Two columns of two floats.
    pub const Mat2: Type = Type::Matrix(Scalar::Float, 2);
    /// Three columns of three floats.
    pub const Mat3: Type = Type::Matrix(Scalar::Float, 3);
    /// Four columns of four floats.
    pub const Mat4: Type = Type::Matrix(Scalar::Float, 4);
    /// Two columns of two doubles.
    pub const DMat2: Type = Type::Matrix(Scalar::Double, 2);
    /// Three columns of three doubles.
    pub const DMat3: Type = Type::Matrix(Scalar::Double, 3);
    /// Four columns of four doubles.
    pub const DMat4: Type = Type::Matrix(Scalar::Double, 4);

    /// Every named type with its spelling.
    pub const NAMED: [(Self, &'static str); 22] = [
        (Self::Float, "Float"),
        (Self::Int, "Int"),
        (Self::Bool, "Bool"),
        (Self::Vec2, "Vec2"),
        (Self::Vec3, "Vec3"),
        (Self::Vec4, "Vec4"),
        (Self::IVec2, "IVec2"),
        (Self::IVec3, "IVec3"),
        (Self::IVec4, "IVec4"),
        (Self::Mat2, "Mat2"),
        (Self::Mat3, "Mat3"),
        (Self::Mat4, "Mat4"),
        (Self::Double, "Double"),
        (Self::Long, "Long"),
        (Self::DVec2, "DVec2"),
        (Self::DVec3, "DVec3"),
        (Self::DVec4, "DVec4"),
        (Self::LVec2, "LVec2"),
        (Self::LVec3, "LVec3"),
        (Self::DMat2, "DMat2"),
        (Self::DMat3, "DMat3"),
        (Self::DMat4, "DMat4"),
    ];
}

impl Type {
    /// Stable spelling used in `.orla` sources; structs print by index.
    pub fn name(self) -> &'static str {
        Self::NAMED
            .iter()
            .find(|(ty, _)| *ty == self)
            .map_or("struct", |(_, name)| name)
    }

    /// The scalar type and count a value is made of; `None` for a struct.
    pub fn shape(self) -> Option<(Type, u8)> {
        Some(match self {
            Self::Scalar(kind) => (Self::Scalar(kind), 1),
            Self::Vector(kind, n) => (Self::Scalar(kind), n),
            Self::Matrix(kind, n) => (Self::Scalar(kind), n * n),
            Self::Struct(_) => return None,
        })
    }

    /// The scalar type a value of this type is made of.
    pub fn scalar(self) -> Option<Type> {
        self.kind().map(Self::Scalar)
    }

    /// The kind of scalar a value of this type is made of.
    pub fn kind(self) -> Option<Scalar> {
        match self {
            Self::Scalar(kind) | Self::Vector(kind, _) | Self::Matrix(kind, _) => Some(kind),
            Self::Struct(_) => None,
        }
    }

    /// Whether this is a 64-bit scalar, or a vector or matrix of them.
    pub fn is_64bit(self) -> bool {
        matches!(self.kind(), Some(Scalar::Double | Scalar::Long))
    }

    /// Whether this is made of floats of either width.
    pub fn is_float_kind(self) -> bool {
        matches!(self.kind(), Some(Scalar::Float | Scalar::Double))
    }

    /// Whether this is an integer of either width, or a vector of them.
    pub fn is_int_kind(self) -> bool {
        matches!(self.kind(), Some(Scalar::Int | Scalar::Long))
    }

    /// Words a value takes, two per 64-bit scalar; `None` for a struct.
    pub fn words(self) -> Option<usize> {
        let (_, n) = self.shape()?;
        Some(usize::from(n) * self.kind()?.words())
    }

    /// This shape made of `scalar` (`Vec3` of `Double` is `DVec3`), if
    /// such a type exists.
    pub fn with_scalar(self, scalar: Type) -> Option<Type> {
        let Self::Scalar(kind) = scalar else {
            return None;
        };
        match self {
            Self::Scalar(_) => Some(scalar),
            Self::Vector(_, n) => Self::vector(scalar, n),
            Self::Matrix(_, n) => Self::matrix(kind, n),
            Self::Struct(_) => None,
        }
    }

    /// The vector of `n` components of `scalar` (the scalar when `n` is
    /// one). Floats and doubles come in 2 to 4, integers and longs in 2
    /// or 3, booleans alone.
    pub fn vector(scalar: Type, n: u8) -> Option<Type> {
        let Self::Scalar(kind) = scalar else {
            return None;
        };
        match (kind, n) {
            (_, 1) => Some(scalar),
            (Scalar::Float | Scalar::Double | Scalar::Int, 2..=4) | (Scalar::Long, 2..=3) => {
                Some(Self::Vector(kind, n))
            }
            _ => None,
        }
    }

    /// The square matrix of order `n` of a float kind, if it exists.
    pub fn matrix(kind: Scalar, n: u8) -> Option<Type> {
        (matches!(kind, Scalar::Float | Scalar::Double) && (2..=4).contains(&n))
            .then_some(Self::Matrix(kind, n))
    }

    /// The order of a square matrix type.
    pub fn order(self) -> Option<u8> {
        match self {
            Self::Matrix(_, n) => Some(n),
            _ => None,
        }
    }

    /// The column type of a matrix type.
    pub fn column(self) -> Option<Type> {
        match self {
            Self::Matrix(kind, n) => Some(Self::Vector(kind, n)),
            _ => None,
        }
    }

    /// The matrix type whose columns are this vector type.
    pub fn matrix_of(self) -> Option<Type> {
        match self {
            Self::Vector(kind, n) => Self::matrix(kind, n),
            _ => None,
        }
    }

    /// Number of components of a 32-bit float vector.
    pub fn components(self) -> Option<u8> {
        match self {
            Self::Vector(Scalar::Float, n) => Some(n),
            _ => None,
        }
    }

    /// Number of components of a 32-bit integer vector.
    pub fn int_components(self) -> Option<u8> {
        match self {
            Self::Vector(Scalar::Int, n) => Some(n),
            _ => None,
        }
    }

    /// Number of components of a vector of any kind, and their type.
    pub fn lanes(self) -> Option<(u8, Type)> {
        match self {
            Self::Vector(kind, n) => Some((n, Self::Scalar(kind))),
            _ => None,
        }
    }

    /// The same-width vector of the other 32-bit kind (float and int).
    pub fn other_kind(self) -> Option<Type> {
        match self {
            Self::Vector(Scalar::Float, n) => Self::vector(Self::Int, n),
            Self::Vector(Scalar::Int, n) => Self::vector(Self::Float, n),
            _ => None,
        }
    }

    /// Whether values of this type can cross the ABI (every non-struct).
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

impl FromStr for Type {
    type Err = UnknownType;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::NAMED
            .iter()
            .find(|(_, spelled)| *spelled == name)
            .map(|(ty, _)| *ty)
            .ok_or_else(|| UnknownType(name.to_owned()))
    }
}

/// A value crossing the Kernel boundary: a parameter, a constant; structs
/// and arrays as a [`Value::Record`].
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
    /// A four-component integer vector.
    IVec4([i32; 4]),
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
    /// A struct's fields or an array's elements in order, matched by
    /// layout, never by name.
    Record(Vec<Value>),
}

/// One scalar of a value, of any kind.
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

    /// The numbers of one element's Kernel words as `schema` lays them
    /// out, in order: what [`Number::push_words`] wrote, read back.
    pub fn of_element(schema: &Schema, element: &[u32]) -> Vec<Number> {
        let scalars = schema.scalars();
        let word = |i: usize| element.get(i).copied().unwrap_or(0);
        let wide = |i: usize| u64::from(word(i)) | (u64::from(word(i + 1)) << 32);
        let mut out = Vec::with_capacity(scalars.len());
        let mut i = 0;
        while i < scalars.len() {
            out.push(match scalars[i] {
                Scalar::Float => Self::Float(f32::from_bits(word(i))),
                Scalar::Int => Self::Int(word(i) as i32),
                Scalar::Bool => Self::Bool(word(i) != 0),
                Scalar::Double => Self::Double(f64::from_bits(wide(i))),
                Scalar::Long => Self::Long(wide(i) as i64),
            });
            i += scalars[i].words();
        }
        out
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

    /// This given value as one of `kind`, when it converts: an integer to
    /// either float or a wider integer, a float to the other width, a
    /// long to an `Int` that holds it. Kernel arithmetic is separate.
    pub fn as_kind(self, kind: Scalar) -> Option<Number> {
        Some(match (self, kind) {
            (Self::Float(v), Scalar::Float) => Self::Float(v),
            // A float becomes the double of its decimal spelling: `0.001`
            // is the double nearest a thousandth.
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
    /// The type of a non-record value.
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
            Self::IVec4(_) => Type::IVec4,
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
            Self::IVec4(v) => ints(out, v),
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

    /// This value in `template`'s shape, when its scalars convert in
    /// order to the template's ([`Number::as_kind`]). Layout decides:
    /// `[[1, 0], [0, 1]]` is a `Mat2`.
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
            Self::IVec4(_) => Self::IVec4(lanes!(4, 0, int)),
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

/// `value`'s words laid out as `wanted`, each scalar converted in order
/// ([`Number::as_kind`]); `None` for another layout.
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
    /// The literal that reads back as this value; a bare double carries
    /// `d`, a long `l`.
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
            Self::IVec4(v) => list(f, "ivec4", numbers(v)),
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
/// is the grid of `Vec4`.
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

/// Pointer and key state for one evaluation. Key codes are browser key
/// codes (ASCII uppercase letters and digits, space 32, enter 13, escape
/// 27, arrows 37 to 40), 0 to 255. Pointer buttons: 0 left, 1 right,
/// 2 middle.
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
    /// Wheel scroll since the last evaluation, in lines: `x` sideways,
    /// `y` positive away from the person.
    pub wheel: [f32; 2],
    /// A pad's axes in `-1..1`: left stick, right stick, and the triggers
    /// as one pair (`x` left, `y` right, in `0..1`). Zero without a pad.
    pub sticks: [[f32; 2]; 3],
    /// A pad's buttons as bits, the lowest being [`pad::SOUTH`].
    pad_buttons: u32,
    keys: [u32; 8],
}

/// Button numbers `InputState::button` knows: a pointer's 0 to 2, a pad's
/// from 8.
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
    /// Whether the button numbered `code` is held: a pointer's 0 to 2, a
    /// pad's from 8 (see [`pad`]); other numbers are never held.
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

    /// A pad axis pair: 0 left stick, 1 right, 2 triggers; zeros otherwise.
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

/// `C` in `Y = K(X, P, C)`: the world for one evaluation, read by Kernels
/// through builtins behind `time.read` and `input.read`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Context {
    /// Session time in seconds.
    pub time: f32,
    /// The driving sound device's samples per second; zero without one.
    pub rate: f32,
    /// Pointer and key state.
    pub input: InputState,
    /// The window's size in physical pixels; zero by zero without one.
    /// A node's `domain` may take it.
    pub window: [f32; 2],
    /// Physical pixels per logical pixel; one without a window.
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

    /// A hash of what `input.read` observes: input, window size, scale.
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

    /// The loop iterations this run may take over all its elements
    /// ([`Host::work`]). A bound; it needs no capability.
    pub fn work(&self) -> u64 {
        self.host.work()
    }

    /// The words the Host may still allocate under its budget. A bound;
    /// it needs no capability.
    pub fn room(&self) -> usize {
        self.host
            .budget_words()
            .saturating_sub(self.host.used_words())
    }

    /// A bound handle's shape and element layout, to check a binding
    /// before a run. A check; it needs no capability and reads no word.
    pub fn bound(&self, field: Handle) -> Result<(Shape, Schema), AbiError> {
        Ok((
            self.host.domain(field)?.shape(),
            self.host.schema(field)?.clone(),
        ))
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

    /// One element of an input by linear index, as widened words.
    /// Requires `field.read`. Fails while the Field is device-resident.
    pub fn element(
        &self,
        field: Handle,
        index: u32,
    ) -> Result<std::borrow::Cow<'_, [u32]>, AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.element(field, index)?)
    }

    /// Every element of an input as widened words. Requires `field.read`.
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

    /// Gives an input's words a copy on the device, beside the Host's.
    /// Requires `field.read`.
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

    /// Whether these Fields would fit under the Host's budget now
    /// ([`Host::can_hold`]), asked before device allocation. Requires
    /// `field.write`.
    pub fn can_hold(&self, fields: &[(&Schema, Domain)]) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.can_hold(fields)?)
    }

    /// A new Field with unspecified words, for a backend that writes every
    /// word before any read. Requires `field.write`.
    pub fn create_for_fill(&mut self, schema: Schema, domain: Domain) -> Result<Handle, AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.create_for_fill(schema, domain)?)
    }

    /// The memory of a Field this scope created, as kept, to fill in
    /// place. Requires `field.write`.
    pub fn storage_mut(&mut self, field: Handle) -> Result<&mut [u32], AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.storage_mut(field)?)
    }

    /// Replaces every element of a Field this scope created with wide
    /// words, narrowed as the Field keeps them. Requires `field.write`.
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

    /// An element's words read back as the numbers that wrote them, a
    /// 64-bit pair as one.
    #[test]
    fn an_elements_words_read_back_as_its_numbers() {
        let numbers = [
            Number::Float(1.5),
            Number::Int(-3),
            Number::Bool(true),
            Number::Double(0.1),
            Number::Long(-5_000_000_000),
        ];
        let mut words = Vec::new();
        for number in numbers {
            number.push_words(&mut words);
        }
        let schema = Schema::new(
            numbers
                .iter()
                .flat_map(|n| vec![n.kind(); n.kind().words()])
                .collect(),
        );
        assert_eq!(Number::of_element(&schema, &words), numbers);
    }

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
