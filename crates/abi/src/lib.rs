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

pub use orla_host::{Host, HostError, Image, ImageHandle, Pixel};

/// A named authority a Kernel may declare and be granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Read pixels and dimensions of input images.
    ImageRead,
    /// Create and write output images.
    ImageWrite,
    /// Read the session clock.
    TimeRead,
}

impl Capability {
    /// All capabilities known to this ABI version.
    pub const ALL: [Capability; 3] = [
        Capability::ImageRead,
        Capability::ImageWrite,
        Capability::TimeRead,
    ];

    /// Stable spelling used in `.orla` sources and manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::ImageRead => "image.read",
            Self::ImageWrite => "image.write",
            Self::TimeRead => "time.read",
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
    /// An image resource, only ever seen through a handle.
    Image,
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
            Self::Image => "Image",
            Self::Struct(_) => "struct",
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

    /// Whether values of this type are plain data that can cross the ABI
    /// (not resources, not Kernel-internal structs).
    pub fn is_value(self) -> bool {
        !matches!(self, Self::Image | Self::Struct(_))
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
            Self::Image,
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

/// The output domain of one Kernel execution: the grid of pixels the entry
/// point runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domain {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

/// Everything a Kernel receives for one execution: `X`, `P` and `C` in
/// `Y = K(X, P, C)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    /// Image inputs in manifest order.
    pub inputs: Vec<ImageHandle>,
    /// Parameter values in manifest order.
    pub params: Vec<Value>,
    /// Output domain.
    pub domain: Domain,
    /// Session time in seconds. Context, not a resource; Kernels read it
    /// through `time()` and must declare `time.read`.
    pub time: f32,
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

    /// Dimensions of an image. Requires `image.read`.
    pub fn image_size(&self, image: ImageHandle) -> Result<(u32, u32), AbiError> {
        self.require(Capability::ImageRead)?;
        Ok(self.host.image_size(image)?)
    }

    /// One pixel of an image. Requires `image.read`.
    pub fn image_read(&self, image: ImageHandle, x: u32, y: u32) -> Result<Pixel, AbiError> {
        self.require(Capability::ImageRead)?;
        Ok(self.host.read_pixel(image, x, y)?)
    }

    /// Every pixel of an image, row-major. Requires `image.read`.
    pub fn image_pixels(&self, image: ImageHandle) -> Result<&[Pixel], AbiError> {
        self.require(Capability::ImageRead)?;
        Ok(self.host.image(image)?.pixels())
    }

    /// Replaces every pixel of an image. Requires `image.write`.
    pub fn image_write_all(
        &mut self,
        image: ImageHandle,
        pixels: &[Pixel],
    ) -> Result<(), AbiError> {
        self.require(Capability::ImageWrite)?;
        Ok(self.host.write_image(image, pixels)?)
    }

    /// A new black image. Requires `image.write`.
    pub fn image_create(&mut self, width: u32, height: u32) -> Result<ImageHandle, AbiError> {
        self.require(Capability::ImageWrite)?;
        Ok(self.host.create_image(width, height)?)
    }

    /// Writes one pixel. Requires `image.write`.
    pub fn image_write(
        &mut self,
        image: ImageHandle,
        x: u32,
        y: u32,
        pixel: Pixel,
    ) -> Result<(), AbiError> {
        self.require(Capability::ImageWrite)?;
        Ok(self.host.write_pixel(image, x, y, pixel)?)
    }

    /// Releases an image this scope created. Requires `image.write`.
    pub fn image_release(&mut self, image: ImageHandle) -> Result<(), AbiError> {
        self.require(Capability::ImageWrite)?;
        Ok(self.host.release_image(image)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            scope.image_read(image, 0, 0),
            Err(AbiError::Denied(Capability::ImageRead))
        );
        assert_eq!(
            scope.image_create(1, 1),
            Err(AbiError::Denied(Capability::ImageWrite))
        );
        assert_eq!(host.live_images(), 1);
    }

    #[test]
    fn scope_forwards_granted_calls_and_host_errors() {
        let mut host = Host::new(64);
        let image = host.create_image(2, 2).unwrap();
        let granted: Capabilities = [Capability::ImageRead, Capability::ImageWrite]
            .into_iter()
            .collect();
        let mut scope = Scope::new(&mut host, granted);
        scope.image_write(image, 1, 0, [0.5; 4]).unwrap();
        assert_eq!(scope.image_read(image, 1, 0).unwrap(), [0.5; 4]);
        assert_eq!(scope.image_size(image).unwrap(), (2, 2));
        assert!(matches!(
            scope.image_read(image, 5, 5),
            Err(AbiError::Host(HostError::OutOfBounds { .. }))
        ));
        let created = scope.image_create(1, 1).unwrap();
        scope.image_release(created).unwrap();
        assert_eq!(host.live_images(), 1);
    }
}
