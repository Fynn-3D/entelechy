#![warn(missing_docs)]
//! Host Core (L1).
//!
//! The Host owns primitive resources and hands out opaque handles. It knows
//! nothing about Kernels, Graphs, or products. Every resource has one owner
//! (the Host), an explicit budget, and a handle that becomes invalid after
//! release. Nothing here should ever grow product logic; if a feature can be
//! a Kernel, it does not belong in this crate.
//!
//! There is one kind of resource: a Field, which is a [`Domain`] of elements
//! laid out by a [`Schema`], one 32-bit word per scalar. An image is the
//! Field whose domain is a grid and whose elements are four floats; the
//! pixel helpers here are views of that case, not a second store.

pub mod file;

use std::any::Any;
use std::fmt;

/// Opaque handle to a resource owned by the Host: any Field, an image
/// included. The architecture documents list handle kinds (image, buffer,
/// texture); the Typed Field decision converged them into this one.
///
/// The handle carries a generation so that a released slot cannot be reached
/// through a stale copy of the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Handle {
    index: u32,
    generation: u32,
}

impl fmt::Display for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "field#{}.{}", self.index, self.generation)
    }
}

/// One RGBA pixel with linear float components.
pub type Pixel = [f32; 4];

/// One scalar of an element, as stored: every scalar is one 32-bit word, so
/// integers cross the boundary exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// An `f32` word.
    Float,
    /// An `i32` word.
    Int,
    /// A `0` or `1` word.
    Bool,
}

/// The flattened element layout of a Field: vectors and structs become
/// their scalars in declaration order. This is the complete schema the
/// Host and backends need; names live in the Kernel that declared it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schema(Vec<Scalar>);

impl Schema {
    /// Builds a schema from scalars in order.
    pub fn new(scalars: Vec<Scalar>) -> Self {
        Self(scalars)
    }

    /// The layout of a pixel: four floats.
    pub fn pixel() -> Self {
        Self(vec![Scalar::Float; 4])
    }

    /// The scalars in order.
    pub fn scalars(&self) -> &[Scalar] {
        &self.0
    }

    /// Words per element.
    pub fn words(&self) -> usize {
        self.0.len()
    }

    /// Whether elements are pixels: four floats.
    pub fn is_pixel(&self) -> bool {
        self.0.len() == 4 && self.0.iter().all(|s| *s == Scalar::Float)
    }

    /// The layout in words, for a message a person reads: "four floats
    /// (a pixel)", "two floats and an integer". The spelling `Display`
    /// writes is what a file and a wire carry; this is what a
    /// diagnostic says.
    pub fn describe(&self) -> String {
        if self.is_pixel() {
            return "four floats (a pixel)".to_owned();
        }
        if self.0.is_empty() {
            return "nothing".to_owned();
        }
        let mut runs: Vec<(Scalar, usize)> = Vec::new();
        for scalar in &self.0 {
            match runs.last_mut() {
                Some((last, count)) if last == scalar => *count += 1,
                _ => runs.push((*scalar, 1)),
            }
        }
        let counted = |scalar: Scalar, count: usize| {
            let name = match (scalar, count) {
                (Scalar::Float, 1) => "float",
                (Scalar::Float, _) => "floats",
                (Scalar::Int, 1) => "integer",
                (Scalar::Int, _) => "integers",
                (Scalar::Bool, 1) => "boolean",
                (Scalar::Bool, _) => "booleans",
            };
            let how_many = match count {
                1 => "one".to_owned(),
                2 => "two".to_owned(),
                3 => "three".to_owned(),
                4 => "four".to_owned(),
                other => other.to_string(),
            };
            format!("{how_many} {name}")
        };
        let parts: Vec<String> = runs
            .into_iter()
            .map(|(scalar, count)| counted(scalar, count))
            .collect();
        match parts.len() {
            1 => parts[0].clone(),
            _ => format!(
                "{} and {}",
                parts[..parts.len() - 1].join(", "),
                parts[parts.len() - 1]
            ),
        }
    }
}

impl std::str::FromStr for Schema {
    type Err = String;

    /// Parses the spelling `Display` writes: scalars as `f`, `i`, `b`,
    /// separated by commas.
    fn from_str(text: &str) -> Result<Self, String> {
        let mut scalars = Vec::new();
        for part in text.split(',') {
            scalars.push(match part.trim() {
                "f" => Scalar::Float,
                "i" => Scalar::Int,
                "b" => Scalar::Bool,
                other => return Err(format!("`{other}` is not a scalar (f, i, or b)")),
            });
        }
        Ok(Self(scalars))
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, scalar) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            f.write_str(match scalar {
                Scalar::Float => "f",
                Scalar::Int => "i",
                Scalar::Bool => "b",
            })?;
        }
        Ok(())
    }
}

/// How a Field's elements are addressed. The shapes are the dense lattices
/// a Kernel can be mapped over; anything irregular is rows plus indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shape {
    /// One dimension, addressed by an index.
    Rows,
    /// Two dimensions, addressed by a coordinate pair.
    Grid,
    /// Three dimensions, addressed by a coordinate triple.
    Volume,
}

impl Shape {
    /// Stable spelling used in `.orla` sources.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rows => "Rows",
            Self::Grid => "Grid",
            Self::Volume => "Volume",
        }
    }

    /// Number of dimensions.
    pub fn dims(self) -> u8 {
        match self {
            Self::Rows => 1,
            Self::Grid => 2,
            Self::Volume => 3,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The extent of a Field: how many elements, in which shape. Every side is
/// at least one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    /// `count` elements in a row.
    Rows(u32),
    /// `width` by `height` elements.
    Grid(u32, u32),
    /// `width` by `height` by `depth` elements.
    Volume(u32, u32, u32),
}

impl Domain {
    /// The shape.
    pub fn shape(self) -> Shape {
        match self {
            Self::Rows(_) => Shape::Rows,
            Self::Grid(..) => Shape::Grid,
            Self::Volume(..) => Shape::Volume,
        }
    }

    /// Elements along the first axis.
    pub fn width(self) -> u32 {
        match self {
            Self::Rows(n) => n,
            Self::Grid(w, _) | Self::Volume(w, _, _) => w,
        }
    }

    /// Elements along the second axis; one for rows.
    pub fn height(self) -> u32 {
        match self {
            Self::Rows(_) => 1,
            Self::Grid(_, h) | Self::Volume(_, h, _) => h,
        }
    }

    /// Elements along the third axis; one below a volume.
    pub fn depth(self) -> u32 {
        match self {
            Self::Volume(_, _, d) => d,
            _ => 1,
        }
    }

    /// Total number of elements.
    pub fn count(self) -> usize {
        self.width() as usize * self.height() as usize * self.depth() as usize
    }

    /// Whether any side is zero.
    pub fn is_empty(self) -> bool {
        self.width() == 0 || self.height() == 0 || self.depth() == 0
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rows(n) => write!(f, "{n}"),
            Self::Grid(w, h) => write!(f, "vec2({w}, {h})"),
            Self::Volume(w, h, d) => write!(f, "vec3({w}, {h}, {d})"),
        }
    }
}

/// Elements laid out by a [`Schema`] over a [`Domain`], one word per scalar,
/// first axis fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldData {
    schema: Schema,
    domain: Domain,
    words: Vec<u32>,
}

impl FieldData {
    /// A zeroed Field.
    pub fn new(schema: Schema, domain: Domain) -> Result<Self, HostError> {
        if domain.is_empty() || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let words = vec![0; domain.count() * schema.words()];
        Ok(Self {
            schema,
            domain,
            words,
        })
    }

    /// Wraps existing words; the length must be `count * schema.words()`.
    pub fn from_words(schema: Schema, domain: Domain, words: Vec<u32>) -> Result<Self, HostError> {
        if domain.is_empty() || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let expected = domain.count() * schema.words();
        if words.len() != expected {
            return Err(HostError::WordCount {
                expected,
                actual: words.len(),
            });
        }
        Ok(Self {
            schema,
            domain,
            words,
        })
    }

    /// A grid of pixels from an image.
    pub fn from_image(image: Image) -> Self {
        let words = image
            .pixels
            .iter()
            .flat_map(|pixel| pixel.iter().map(|v| v.to_bits()))
            .collect();
        Self {
            schema: Schema::pixel(),
            domain: Domain::Grid(image.width, image.height),
            words,
        }
    }

    /// The element layout.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The extent.
    pub fn domain(&self) -> Domain {
        self.domain
    }

    /// Number of elements.
    pub fn count(&self) -> u32 {
        self.domain.count() as u32
    }

    /// All words, element after element.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The words of one element by linear index.
    pub fn element(&self, index: u32) -> Result<&[u32], HostError> {
        let count = self.domain.count();
        if index as usize >= count {
            return Err(HostError::ElementOutOfBounds {
                index,
                count: count as u32,
            });
        }
        let stride = self.schema.words();
        Ok(&self.words[index as usize * stride..(index as usize + 1) * stride])
    }

    /// The pixels of a Field of pixels, as an image of `width` by
    /// `height * depth` rows.
    pub fn to_image(&self) -> Result<Image, HostError> {
        if !self.schema.is_pixel() {
            return Err(HostError::NotPixels);
        }
        let pixels = self
            .words
            .chunks_exact(4)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                    f32::from_bits(c[3]),
                ]
            })
            .collect();
        Image::from_pixels(
            self.domain.width(),
            self.domain.height() * self.domain.depth(),
            pixels,
        )
    }
}

/// A dense row-major RGBA float grid: the pixel view of a Field, used at
/// observation boundaries (files, windows, tests).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<Pixel>,
}

impl Image {
    /// Creates a black, fully transparent image.
    pub fn new(width: u32, height: u32) -> Result<Self, HostError> {
        if width == 0 || height == 0 {
            return Err(HostError::ZeroSize);
        }
        let count = width as usize * height as usize;
        Ok(Self {
            width,
            height,
            pixels: vec![[0.0; 4]; count],
        })
    }

    /// Wraps existing pixel data; the length must match the dimensions.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<Pixel>) -> Result<Self, HostError> {
        if width == 0 || height == 0 {
            return Err(HostError::ZeroSize);
        }
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(HostError::PixelCount {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major pixel data.
    pub fn pixels(&self) -> &[Pixel] {
        &self.pixels
    }
}

/// Errors reported by the Host. They are always explicit; the Host never
/// silently clamps, recovers, or reallocates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The handle does not refer to a live resource.
    InvalidHandle(Handle),
    /// Allocating the resource would exceed the Host budget.
    BudgetExceeded {
        /// Words requested by the allocation.
        requested: usize,
        /// Words still available under the budget.
        available: usize,
    },
    /// A pixel coordinate lies outside the image.
    OutOfBounds {
        /// Requested column.
        x: u32,
        /// Requested row.
        y: u32,
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
    /// An element index lies outside the Field.
    ElementOutOfBounds {
        /// Requested element.
        index: u32,
        /// Elements in the Field.
        count: u32,
    },
    /// Fields must have at least one element and one word per element.
    ZeroSize,
    /// Pixel data length does not match the dimensions.
    PixelCount {
        /// Pixels implied by the dimensions.
        expected: usize,
        /// Pixels supplied.
        actual: usize,
    },
    /// Word data length does not match the domain and schema.
    WordCount {
        /// Words implied by the domain and schema.
        expected: usize,
        /// Words supplied.
        actual: usize,
    },
    /// The Field's words live on a device; materialize it before reading or
    /// writing them from the CPU.
    Resident(Handle),
    /// The Field's elements are not pixels, so it has no image view.
    NotPixels,
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle(handle) => write!(f, "{handle} is not a live resource"),
            Self::BudgetExceeded {
                requested,
                available,
            } => write!(
                f,
                "resource budget exceeded: requested {requested} words, {available} available"
            ),
            Self::OutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(f, "pixel ({x}, {y}) is outside a {width}x{height} image"),
            Self::ElementOutOfBounds { index, count } => {
                write!(f, "element {index} is outside a Field of {count}")
            }
            Self::ZeroSize => write!(f, "fields must have at least one element"),
            Self::PixelCount { expected, actual } => {
                write!(f, "expected {expected} pixels, received {actual}")
            }
            Self::WordCount { expected, actual } => {
                write!(f, "expected {expected} words, received {actual}")
            }
            Self::Resident(handle) => write!(
                f,
                "{handle} is resident on a device; materialize it before touching its elements"
            ),
            Self::NotPixels => write!(f, "the elements are not pixels, so there is no image view"),
        }
    }
}

impl std::error::Error for HostError {}

/// Where a Field's words are.
enum Storage {
    /// Nothing; the slot is free.
    Free,
    /// Words in Host memory.
    Data(FieldData),
    /// Words on a device, held by an object only that backend understands.
    Resident {
        schema: Schema,
        domain: Domain,
        device: Box<dyn Any + Send + Sync>,
    },
}

struct Slot {
    generation: u32,
    storage: Storage,
    /// Budget charged when the resource was adopted, returned on release.
    charge: usize,
}

impl Slot {
    fn live(&self) -> bool {
        !matches!(self.storage, Storage::Free)
    }

    fn describe(&self) -> Option<(&Schema, Domain)> {
        match &self.storage {
            Storage::Free => None,
            Storage::Data(data) => Some((&data.schema, data.domain)),
            Storage::Resident { schema, domain, .. } => Some((schema, *domain)),
        }
    }
}

/// What a buffer holds when it is handed to a caller that says it will
/// write every word: nothing any Field ever had, so leaving some of
/// them shows.
const POISON: u32 = 0xDEAD_BEEF;

/// The Host: a bounded store of primitive resources.
pub struct Host {
    slots: Vec<Slot>,
    free: Vec<u32>,
    budget_words: usize,
    used_words: usize,
    /// Word buffers a released Field gave back, largest last. A scene
    /// makes and drops the same Fields every frame, and asking the
    /// system for a megabyte and touching it again costs more than the
    /// work that fills it; these are handed out again by `create`.
    spare: Vec<Vec<u32>>,
}

impl Host {
    /// Creates a Host that will never hold more than `budget_words` live
    /// words; a pixel is four words.
    pub fn new(budget_words: usize) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            spare: Vec::new(),
            budget_words,
            used_words: 0,
        }
    }

    /// Number of live resources, resident ones included.
    pub fn live(&self) -> usize {
        self.slots.iter().filter(|slot| slot.live()).count()
    }

    /// Number of live resources whose words are on a device.
    pub fn resident_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot.storage, Storage::Resident { .. }))
            .count()
    }

    /// Words currently allocated.
    pub fn used_words(&self) -> usize {
        self.used_words
    }

    /// Total word budget.
    pub fn budget_words(&self) -> usize {
        self.budget_words
    }

    /// Allocates a zeroed Field.
    pub fn create(&mut self, schema: Schema, domain: Domain) -> Result<Handle, HostError> {
        let data = match self.take_spare(&schema, domain)? {
            Some(mut words) => {
                words.fill(0);
                FieldData::from_words(schema, domain, words)?
            }
            None => FieldData::new(schema, domain)?,
        };
        self.adopt(data)
    }

    /// A Field whose words are whatever the buffer it was given holds:
    /// for a backend that writes every one of them before anything
    /// reads it, which is what a map over a domain does. Anything else
    /// wants [`Host::create`].
    pub fn create_for_fill(&mut self, schema: Schema, domain: Domain) -> Result<Handle, HostError> {
        let data = match self.take_spare(&schema, domain)? {
            Some(mut words) => {
                // The caller says it writes every word. Where that is
                // checked, the words it was given are nothing any Field
                // ever held, so a caller that leaves some of them is a
                // test that fails rather than one that passes on what
                // another Field happened to leave there.
                if cfg!(debug_assertions) {
                    words.fill(POISON);
                }
                FieldData::from_words(schema, domain, words)?
            }
            None => FieldData::new(schema, domain)?,
        };
        self.adopt(data)
    }

    /// A buffer of exactly the words this Field needs, from the ones
    /// released Fields gave back.
    fn take_spare(
        &mut self,
        schema: &Schema,
        domain: Domain,
    ) -> Result<Option<Vec<u32>>, HostError> {
        if domain.is_empty() || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let wanted = domain.count() * schema.words();
        let at = self.spare.iter().position(|buffer| buffer.len() == wanted);
        Ok(at.map(|at| self.spare.swap_remove(at)))
    }

    /// Takes ownership of existing data.
    pub fn adopt(&mut self, data: FieldData) -> Result<Handle, HostError> {
        let requested = data.words.len();
        self.place(requested, Storage::Data(data))
    }

    /// Takes ownership of a Field whose words live on a device. `device`
    /// is whatever the owning backend needs to find them again.
    pub fn adopt_resident(
        &mut self,
        schema: Schema,
        domain: Domain,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<Handle, HostError> {
        if domain.is_empty() || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let requested = domain.count() * schema.words();
        self.place(
            requested,
            Storage::Resident {
                schema,
                domain,
                device,
            },
        )
    }

    /// Allocates a black image: a grid of pixels.
    pub fn create_image(&mut self, width: u32, height: u32) -> Result<Handle, HostError> {
        self.create(Schema::pixel(), Domain::Grid(width, height))
    }

    /// Takes ownership of an image as a grid of pixels.
    pub fn adopt_image(&mut self, image: Image) -> Result<Handle, HostError> {
        self.adopt(FieldData::from_image(image))
    }

    fn place(&mut self, requested: usize, storage: Storage) -> Result<Handle, HostError> {
        let available = self.budget_words.saturating_sub(self.used_words);
        if requested > available {
            return Err(HostError::BudgetExceeded {
                requested,
                available,
            });
        }
        self.used_words += requested;
        let index = match self.free.pop() {
            Some(index) => {
                let slot = &mut self.slots[index as usize];
                slot.generation = slot.generation.wrapping_add(1);
                slot.storage = storage;
                slot.charge = requested;
                index
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    storage,
                    charge: requested,
                });
                (self.slots.len() - 1) as u32
            }
        };
        Ok(Handle {
            index,
            generation: self.slots[index as usize].generation,
        })
    }

    /// The element layout, resident or not.
    pub fn schema(&self, handle: Handle) -> Result<&Schema, HostError> {
        Ok(self.slot(handle)?.describe().expect("live slot").0)
    }

    /// The extent, resident or not.
    pub fn domain(&self, handle: Handle) -> Result<Domain, HostError> {
        Ok(self.slot(handle)?.describe().expect("live slot").1)
    }

    /// Width and height, resident or not; a volume reports its first two
    /// axes.
    pub fn image_size(&self, handle: Handle) -> Result<(u32, u32), HostError> {
        let domain = self.domain(handle)?;
        Ok((domain.width(), domain.height()))
    }

    /// Borrows the data behind a handle; fails with [`HostError::Resident`]
    /// when its words are on a device.
    pub fn data(&self, handle: Handle) -> Result<&FieldData, HostError> {
        match &self.slot(handle)?.storage {
            Storage::Data(data) => Ok(data),
            Storage::Resident { .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
        }
    }

    fn data_mut(&mut self, handle: Handle) -> Result<&mut FieldData, HostError> {
        match &mut self.slot_mut(handle)?.storage {
            Storage::Data(data) => Ok(data),
            Storage::Resident { .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
        }
    }

    /// The words of one element by linear index.
    pub fn element(&self, handle: Handle, index: u32) -> Result<&[u32], HostError> {
        self.data(handle)?.element(index)
    }

    /// The words of a Field, to write in place. A backend that fills a
    /// Field it just created writes here rather than into a buffer of
    /// its own that it then copies in: for a million elements that is
    /// two copies of every word that nobody needs.
    pub fn words_mut(&mut self, handle: Handle) -> Result<&mut [u32], HostError> {
        Ok(&mut self.data_mut(handle)?.words)
    }

    /// Replaces every word; the length must match.
    pub fn write_words(&mut self, handle: Handle, words: &[u32]) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        if words.len() != data.words.len() {
            return Err(HostError::WordCount {
                expected: data.words.len(),
                actual: words.len(),
            });
        }
        data.words.copy_from_slice(words);
        Ok(())
    }

    /// Replaces the words of one element.
    pub fn write_element(
        &mut self,
        handle: Handle,
        index: u32,
        words: &[u32],
    ) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        let stride = data.schema.words();
        let count = data.domain.count();
        if index as usize >= count {
            return Err(HostError::ElementOutOfBounds {
                index,
                count: count as u32,
            });
        }
        if words.len() != stride {
            return Err(HostError::WordCount {
                expected: stride,
                actual: words.len(),
            });
        }
        let start = index as usize * stride;
        data.words[start..start + stride].copy_from_slice(words);
        Ok(())
    }

    /// The device object of a resident Field, or `None` when its words are
    /// in Host memory.
    pub fn resident(&self, handle: Handle) -> Result<Option<&(dyn Any + Send + Sync)>, HostError> {
        Ok(match &self.slot(handle)?.storage {
            Storage::Resident { device, .. } => Some(device.as_ref()),
            _ => None,
        })
    }

    /// Moves a Field's words onto a device: the Host drops its copy and
    /// keeps `device` in its place. Layout, extent, and identity are
    /// unchanged.
    pub fn make_resident(
        &mut self,
        handle: Handle,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (schema, domain) = slot.describe().expect("live slot");
        let schema = schema.clone();
        slot.storage = Storage::Resident {
            schema,
            domain,
            device,
        };
        Ok(())
    }

    /// Brings a resident Field back into Host memory with the words the
    /// backend downloaded; the device object is dropped. A no-op for a
    /// Field already in memory.
    pub fn materialize(&mut self, handle: Handle, words: Vec<u32>) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (schema, domain) = match &slot.storage {
            Storage::Resident { schema, domain, .. } => (schema.clone(), *domain),
            Storage::Data(_) => return Ok(()),
            Storage::Free => return Err(HostError::InvalidHandle(handle)),
        };
        slot.storage = Storage::Data(FieldData::from_words(schema, domain, words)?);
        Ok(())
    }

    /// Releases a Field, resident or not. The handle is invalid afterwards.
    pub fn release(&mut self, handle: Handle) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let storage = std::mem::replace(&mut slot.storage, Storage::Free);
        let charge = std::mem::take(&mut slot.charge);
        self.used_words -= charge;
        self.free.push(handle.index);
        // The words go into the spare pile for the next Field of that
        // size, which is almost always the same node a frame later.
        if let Storage::Data(data) = storage {
            self.keep_spare(data.words);
        }
        Ok(())
    }

    /// Keeps a released buffer for the next Field of its size. The pile
    /// holds a handful of them and no more words than an eighth of the
    /// budget, so a scene of large Fields does not leave the system
    /// holding what it is no longer using; the rest goes back.
    fn keep_spare(&mut self, words: Vec<u32>) {
        const KEPT: usize = 8;
        let room = self.budget_words / 8;
        if words.is_empty() || words.len() > room {
            return;
        }
        self.spare.push(words);
        let mut held: usize = self.spare.iter().map(Vec::len).sum();
        while self.spare.len() > KEPT || held > room {
            // The smallest is the one worth least.
            let Some(smallest) = self
                .spare
                .iter()
                .enumerate()
                .min_by_key(|(_, buffer)| buffer.len())
                .map(|(at, _)| at)
            else {
                break;
            };
            held -= self.spare[smallest].len();
            self.spare.swap_remove(smallest);
        }
    }

    /// The image view of a Field of pixels: a copy, since the Host stores
    /// words. Fails with [`HostError::Resident`] when the words are on a
    /// device and [`HostError::NotPixels`] for other layouts.
    pub fn image(&self, handle: Handle) -> Result<Image, HostError> {
        self.data(handle)?.to_image()
    }

    fn pixel_offset(data: &FieldData, x: u32, y: u32) -> Result<usize, HostError> {
        if !data.schema.is_pixel() {
            return Err(HostError::NotPixels);
        }
        let width = data.domain.width();
        let height = data.domain.height() * data.domain.depth();
        if x >= width || y >= height {
            return Err(HostError::OutOfBounds {
                x,
                y,
                width,
                height,
            });
        }
        Ok((y as usize * width as usize + x as usize) * 4)
    }

    /// Reads one pixel of a Field of pixels.
    pub fn read_pixel(&self, handle: Handle, x: u32, y: u32) -> Result<Pixel, HostError> {
        let data = self.data(handle)?;
        let offset = Self::pixel_offset(data, x, y)?;
        let w = &data.words[offset..offset + 4];
        Ok([
            f32::from_bits(w[0]),
            f32::from_bits(w[1]),
            f32::from_bits(w[2]),
            f32::from_bits(w[3]),
        ])
    }

    /// Writes one pixel of a Field of pixels.
    pub fn write_pixel(
        &mut self,
        handle: Handle,
        x: u32,
        y: u32,
        pixel: Pixel,
    ) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        let offset = Self::pixel_offset(data, x, y)?;
        for (k, v) in pixel.iter().enumerate() {
            data.words[offset + k] = v.to_bits();
        }
        Ok(())
    }

    /// Replaces every pixel of a Field of pixels; the length must match.
    pub fn write_image(&mut self, handle: Handle, pixels: &[Pixel]) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        if !data.schema.is_pixel() {
            return Err(HostError::NotPixels);
        }
        if pixels.len() * 4 != data.words.len() {
            return Err(HostError::PixelCount {
                expected: data.words.len() / 4,
                actual: pixels.len(),
            });
        }
        for (slot, value) in data
            .words
            .iter_mut()
            .zip(pixels.iter().flat_map(|p| p.iter()))
        {
            *slot = value.to_bits();
        }
        Ok(())
    }

    fn slot(&self, handle: Handle) -> Result<&Slot, HostError> {
        self.slots
            .get(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation && slot.live())
            .ok_or(HostError::InvalidHandle(handle))
    }

    fn slot_mut(&mut self, handle: Handle) -> Result<&mut Slot, HostError> {
        self.slots
            .get_mut(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation && slot.live())
            .ok_or(HostError::InvalidHandle(handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words a released Field gave back are handed out again, and
    /// what the pile holds is bounded: a handful of buffers and an
    /// eighth of the budget, so a scene of large Fields does not leave
    /// the system holding what nothing is using.
    #[test]
    fn released_words_come_back_and_the_pile_is_bounded() {
        let mut host = Host::new(4096);
        let schema = Schema::new(vec![Scalar::Int]);
        let first = host.create(schema.clone(), Domain::Rows(256)).unwrap();
        host.write_words(first, &[7; 256]).unwrap();
        host.release(first).unwrap();
        // A plain `create` is zeroed however the words come to it.
        let again = host.create(schema.clone(), Domain::Rows(256)).unwrap();
        assert!(host.data(again).unwrap().words().iter().all(|w| *w == 0));
        host.release(again).unwrap();

        // Nothing live, and the pile holds no more than an eighth of
        // the budget however much is released.
        for _ in 0..32 {
            let big = host.create(schema.clone(), Domain::Rows(400)).unwrap();
            host.release(big).unwrap();
        }
        let held: usize = host.spare.iter().map(Vec::len).sum();
        assert!(held <= 4096 / 8, "the pile holds {held} words");
        assert_eq!(host.live(), 0);

        // A buffer larger than the pile takes is not kept at all.
        let huge = host.create(schema, Domain::Rows(2048)).unwrap();
        host.release(huge).unwrap();
        assert!(host.spare.iter().all(|buffer| buffer.len() <= 4096 / 8));
    }

    #[test]
    fn handles_become_invalid_after_release() {
        let mut host = Host::new(64);
        let image = host.create_image(2, 2).unwrap();
        host.write_pixel(image, 1, 1, [1.0, 0.5, 0.25, 1.0])
            .unwrap();
        assert_eq!(host.read_pixel(image, 1, 1).unwrap(), [1.0, 0.5, 0.25, 1.0]);
        host.release(image).unwrap();
        assert_eq!(
            host.read_pixel(image, 0, 0),
            Err(HostError::InvalidHandle(image))
        );
        assert_eq!(host.release(image), Err(HostError::InvalidHandle(image)));
        // A reused slot receives a new generation, so the old handle stays dead.
        let reused = host.create_image(1, 1).unwrap();
        assert_ne!(reused, image);
        assert!(host.read_pixel(image, 0, 0).is_err());
        assert_eq!(host.live(), 1);
    }

    #[test]
    fn budget_is_enforced_in_words_and_returned() {
        let mut host = Host::new(32);
        let first = host.create_image(2, 2).unwrap();
        assert_eq!(
            host.create_image(3, 3),
            Err(HostError::BudgetExceeded {
                requested: 36,
                available: 16
            })
        );
        host.release(first).unwrap();
        assert_eq!(host.used_words(), 0);
        assert!(host.create_image(2, 4).is_ok());
    }

    #[test]
    fn resident_fields_keep_identity_extent_and_budget_but_hide_words() {
        let mut host = Host::new(32);
        let image = host.create_image(2, 2).unwrap();
        host.write_pixel(image, 0, 0, [1.0; 4]).unwrap();
        host.make_resident(image, Box::new(42u32)).unwrap();
        assert_eq!(host.domain(image).unwrap(), Domain::Grid(2, 2));
        assert_eq!(host.resident_count(), 1);
        assert_eq!(host.used_words(), 16);
        assert_eq!(
            host.read_pixel(image, 0, 0),
            Err(HostError::Resident(image))
        );
        assert_eq!(host.image(image).unwrap_err(), HostError::Resident(image));
        let device = host.resident(image).unwrap().unwrap();
        assert_eq!(device.downcast_ref::<u32>(), Some(&42));
        host.materialize(image, vec![0.5f32.to_bits(); 16]).unwrap();
        assert_eq!(host.read_pixel(image, 1, 1).unwrap(), [0.5; 4]);
        assert!(host.resident(image).unwrap().is_none());
        let resident = host
            .adopt_resident(Schema::pixel(), Domain::Grid(2, 2), Box::new(7u8))
            .unwrap();
        assert_eq!(host.live(), 2);
        host.release(resident).unwrap();
        host.release(image).unwrap();
        assert_eq!(host.used_words(), 0);
    }

    #[test]
    fn every_shape_shares_the_store_and_images_are_a_view() {
        let mut host = Host::new(64);
        let schema = Schema::new(vec![Scalar::Float, Scalar::Int, Scalar::Bool]);
        let rows = host.create(schema.clone(), Domain::Rows(4)).unwrap();
        assert_eq!(host.domain(rows).unwrap(), Domain::Rows(4));
        assert_eq!(host.used_words(), 12);
        assert_eq!(host.image(rows).unwrap_err(), HostError::NotPixels);
        assert_eq!(host.schema(rows).unwrap(), &schema);
        let words: Vec<u32> = (0..12).collect();
        host.write_words(rows, &words).unwrap();
        assert_eq!(host.element(rows, 2).unwrap(), &[6, 7, 8]);
        host.write_element(rows, 3, &[1, 2, 3]).unwrap();
        assert_eq!(host.element(rows, 3).unwrap(), &[1, 2, 3]);
        assert_eq!(
            host.element(rows, 4).unwrap_err(),
            HostError::ElementOutOfBounds { index: 4, count: 4 }
        );
        host.make_resident(rows, Box::new(1u8)).unwrap();
        assert_eq!(host.schema(rows).unwrap(), &schema);
        assert_eq!(host.data(rows).unwrap_err(), HostError::Resident(rows));
        host.materialize(rows, words.clone()).unwrap();
        assert_eq!(host.data(rows).unwrap().words(), &words[..]);

        let volume = host
            .create(Schema::new(vec![Scalar::Float]), Domain::Volume(2, 2, 3))
            .unwrap();
        assert_eq!(host.domain(volume).unwrap().count(), 12);
        assert_eq!(host.image_size(volume).unwrap(), (2, 2));

        // A grid of four floats is an image, whichever way it was made.
        let grid = host.create(Schema::pixel(), Domain::Grid(2, 1)).unwrap();
        host.write_pixel(grid, 1, 0, [0.25, 0.5, 0.75, 1.0])
            .unwrap();
        let image = host.image(grid).unwrap();
        assert_eq!(image.pixels()[1], [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(host.element(grid, 1).unwrap()[0], 0.25f32.to_bits());
        host.release(rows).unwrap();
        host.release(volume).unwrap();
        host.release(grid).unwrap();
        assert_eq!(host.used_words(), 0);
    }

    #[test]
    fn bounds_and_sizes_are_checked() {
        let mut host = Host::new(32);
        assert_eq!(host.create_image(0, 4), Err(HostError::ZeroSize));
        let image = host.create_image(2, 2).unwrap();
        assert_eq!(
            host.read_pixel(image, 2, 0),
            Err(HostError::OutOfBounds {
                x: 2,
                y: 0,
                width: 2,
                height: 2
            })
        );
        assert_eq!(
            Image::from_pixels(2, 2, vec![[0.0; 4]; 3]).unwrap_err(),
            HostError::PixelCount {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(
            host.write_words(image, &[0; 3]).unwrap_err(),
            HostError::WordCount {
                expected: 16,
                actual: 3
            }
        );
    }
}
