#![warn(missing_docs)]
//! Capability ABI (L2).
//!
//! This crate defines the world visible to a Kernel: value types, opaque
//! handles, named capabilities, and a [`Scope`] that gates every Host call on
//! an explicit grant. A Kernel never touches the Host directly; backends call
//! the Host on its behalf through a `Scope` built from the capabilities the
//! Kernel declared. No capability means no access.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

pub use orla_host::file;
pub use orla_host::{
    Domain, FieldData, Handle, Host, HostError, Image, Pixel, Scalar, Schema, Shape,
};

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
            Self::Struct(_) => "struct",
        }
    }

    /// The order of a square matrix type: its columns, each a vector of
    /// as many floats.
    pub fn order(self) -> Option<u8> {
        match self {
            Self::Mat2 => Some(2),
            Self::Mat3 => Some(3),
            Self::Mat4 => Some(4),
            _ => None,
        }
    }

    /// The column type of a matrix type.
    pub fn column(self) -> Option<Type> {
        match self {
            Self::Mat2 => Some(Self::Vec2),
            Self::Mat3 => Some(Self::Vec3),
            Self::Mat4 => Some(Self::Vec4),
            _ => None,
        }
    }

    /// The matrix type whose columns are this vector type.
    pub fn matrix_of(self) -> Option<Type> {
        match self {
            Self::Vec2 => Some(Self::Mat2),
            Self::Vec3 => Some(Self::Mat3),
            Self::Vec4 => Some(Self::Mat4),
            _ => None,
        }
    }

    /// Number of float components for vector types.
    pub fn components(self) -> Option<u8> {
        match self {
            Self::Vec2 => Some(2),
            Self::Vec3 => Some(3),
            Self::Vec4 => Some(4),
            _ => None,
        }
    }

    /// Number of integer components for integer vector types.
    pub fn int_components(self) -> Option<u8> {
        match self {
            Self::IVec2 => Some(2),
            Self::IVec3 => Some(3),
            _ => None,
        }
    }

    /// Number of components of a vector of either kind, and their type.
    pub fn lanes(self) -> Option<(u8, Type)> {
        self.components()
            .map(|n| (n, Self::Float))
            .or_else(|| self.int_components().map(|n| (n, Self::Int)))
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

impl FromStr for Type {
    type Err = UnknownType;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        [
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
        ]
        .into_iter()
        .find(|ty| ty.name() == name)
        .ok_or_else(|| UnknownType(name.to_owned()))
    }
}

/// A plain value crossing the Kernel boundary (parameters, constants).
#[derive(Clone, Copy, Debug, PartialEq)]
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
}

impl Value {
    /// The type of the value.
    pub fn ty(&self) -> Type {
        match self {
            Self::Float(_) => Type::Float,
            Self::Int(_) => Type::Int,
            Self::Bool(_) => Type::Bool,
            Self::Vec2(_) => Type::Vec2,
            Self::Vec3(_) => Type::Vec3,
            Self::Vec4(_) => Type::Vec4,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(value) => write!(f, "{value:?}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Vec2([x, y]) => write!(f, "vec2({x:?}, {y:?})"),
            Self::Vec3([x, y, z]) => write!(f, "vec3({x:?}, {y:?}, {z:?})"),
            Self::Vec4([x, y, z, w]) => write!(f, "vec4({x:?}, {y:?}, {z:?}, {w:?})"),
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
        self.shape == Shape::Grid && self.element == Type::Vec4
    }

    /// The value type that addresses one element: `Int`, `Vec2`, or `Vec3`.
    pub fn index_type(&self) -> Type {
        self.shape.index_type()
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_image() {
            f.write_str("Image")
        } else {
            write!(f, "{}<{}>", self.shape, self.element)
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
    keys: [u32; 8],
}

impl InputState {
    /// Whether the pointer button numbered `code` is held: 0 left, 1
    /// right, 2 middle; any other number is never held.
    pub fn button(&self, code: i32) -> bool {
        match code {
            0 => self.pointer_down,
            1 => self.pointer_right,
            2 => self.pointer_middle,
            _ => false,
        }
    }

    /// The held buttons as bits, `button(n)` at bit `n`.
    pub fn button_bits(&self) -> u32 {
        (self.pointer_down as u32)
            | (self.pointer_right as u32) << 1
            | (self.pointer_middle as u32) << 2
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

    /// One element of an input as words, by linear index. Requires
    /// `field.read`. Fails while the Field is resident on a device.
    pub fn element(&self, field: Handle, index: u32) -> Result<&[u32], AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.element(field, index)?)
    }

    /// Every word of an input. Requires `field.read`.
    pub fn words(&self, field: Handle) -> Result<&[u32], AbiError> {
        self.require(Capability::FieldRead)?;
        Ok(self.host.data(field)?.words())
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

    /// Replaces every word of a Field this scope created. Requires
    /// `field.write`.
    pub fn write_all(&mut self, field: Handle, words: &[u32]) -> Result<(), AbiError> {
        self.require(Capability::FieldWrite)?;
        Ok(self.host.write_words(field, words)?)
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
        assert_eq!(scope.element(image, 1).unwrap(), &[0.5f32.to_bits(); 4]);
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
