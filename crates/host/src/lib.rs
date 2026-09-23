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
//! laid out by a [`Schema`]: a Kernel sees one 32-bit word per scalar (two
//! for a 64-bit one), and a Field may keep a scalar narrower
//! ([`Encoding`]). An image is the Field whose domain is a grid and whose
//! elements are four floats; the pixel helpers here are views of that
//! case, not a second store.

pub mod encoding;
pub mod file;

pub use encoding::Encoding;

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

/// What one word of an element is to a Kernel: a float, an integer or a
/// boolean, each computed in one 32-bit word, or a word of a 64-bit
/// float or integer, which is two words, the low one first. How a word
/// is kept in memory is its [`Encoding`], which may be narrower for the
/// 32-bit kinds and is always a whole word for the 64-bit ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// An `f32`.
    Float,
    /// An `i32`.
    Int,
    /// A `0` or `1`.
    Bool,
    /// One of the two words of an `f64`; they come in pairs, low first.
    Double,
    /// One of the two words of an `i64`; they come in pairs, low first.
    Long,
}

impl Scalar {
    fn letter(self) -> &'static str {
        match self {
            Self::Float => "f",
            Self::Int => "i",
            Self::Bool => "b",
            Self::Double => "d",
            Self::Long => "l",
        }
    }

    /// Words a scalar of this kind takes: two for the 64-bit kinds.
    pub fn words(self) -> usize {
        match self {
            Self::Double | Self::Long => 2,
            _ => 1,
        }
    }

    fn noun(self) -> (&'static str, &'static str) {
        match self {
            Self::Float => ("float", "floats"),
            Self::Int => ("integer", "integers"),
            Self::Bool => ("boolean", "booleans"),
            Self::Double => ("double", "doubles"),
            Self::Long => ("long", "longs"),
        }
    }
}

/// The flattened element layout of a Field: vectors and structs become
/// their scalars in declaration order, each with the encoding it is kept
/// in. This is the complete schema the Host and backends need; names live
/// in the Kernel that declared it.
///
/// An element is its scalars' bytes one after another, and elements one
/// after another with nothing between, little-endian. Every scalar sits
/// at a multiple of its own size, so none crosses a 32-bit word.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schema {
    scalars: Vec<Scalar>,
    encodings: Vec<Encoding>,
    /// Each scalar's byte offset in the element.
    offsets: Vec<usize>,
    /// Bytes an element takes.
    bytes: usize,
}

impl Schema {
    /// Builds a schema from scalars in order, each a whole word.
    pub fn new(scalars: Vec<Scalar>) -> Self {
        let encodings = vec![Encoding::Word; scalars.len()];
        Self::encoded(scalars, encodings).expect("words always lay out")
    }

    /// Builds a schema whose scalars are kept as `encodings`. Refused when
    /// an encoding does not fit its scalar, or a scalar would not sit at
    /// a multiple of its size in every element.
    pub fn encoded(scalars: Vec<Scalar>, encodings: Vec<Encoding>) -> Result<Self, String> {
        if scalars.len() != encodings.len() {
            return Err(format!(
                "{} scalars and {} encodings",
                scalars.len(),
                encodings.len()
            ));
        }
        let mut offsets = Vec::with_capacity(scalars.len());
        let mut bytes = 0;
        let mut widest = 1;
        let mut k = 0;
        while k < scalars.len() {
            let scalar = scalars[k];
            if scalar.words() == 2 && scalars.get(k + 1) != Some(&scalar) {
                return Err(format!(
                    "a {} is two words, and word {k} has no partner",
                    scalar.noun().0
                ));
            }
            k += scalar.words();
        }
        for (scalar, encoding) in scalars.iter().zip(&encodings) {
            if !encoding.fits(*scalar) {
                return Err(format!(
                    "a {} cannot be kept as `{encoding}`",
                    scalar.noun().0
                ));
            }
            let size = encoding.bytes();
            if bytes % size != 0 {
                return Err(format!(
                    "a `{encoding}` scalar would sit at byte {bytes}, which is not a multiple of its size"
                ));
            }
            offsets.push(bytes);
            bytes += size;
            widest = widest.max(size);
        }
        if bytes % widest != 0 {
            return Err(format!(
                "an element of {bytes} bytes would leave the next one's scalars off their size"
            ));
        }
        Ok(Self {
            scalars,
            encodings,
            offsets,
            bytes,
        })
    }

    /// The same scalars, every one kept as `encoding`.
    pub fn with_encoding(&self, encoding: Encoding) -> Result<Self, String> {
        Self::encoded(self.scalars.clone(), vec![encoding; self.scalars.len()])
    }

    /// The layout of a pixel: four floats.
    pub fn pixel() -> Self {
        Self::new(vec![Scalar::Float; 4])
    }

    /// The scalars in order, as a Kernel sees them.
    pub fn scalars(&self) -> &[Scalar] {
        &self.scalars
    }

    /// How each scalar is kept.
    pub fn encodings(&self) -> &[Encoding] {
        &self.encodings
    }

    /// Each scalar's byte offset within an element.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The one encoding every scalar is kept in, when there is one.
    pub fn uniform_encoding(&self) -> Option<Encoding> {
        let first = *self.encodings.first()?;
        self.encodings.iter().all(|e| *e == first).then_some(first)
    }

    /// Words a Kernel sees per element: one per scalar, however it is kept.
    pub fn value_words(&self) -> usize {
        self.scalars.len()
    }

    /// Bytes an element takes in memory.
    pub fn element_bytes(&self) -> usize {
        self.bytes
    }

    /// Whether every scalar is a whole word, so memory holds the Kernel's
    /// own words and nothing is widened or narrowed.
    pub fn is_wide(&self) -> bool {
        self.encodings.iter().all(|e| *e == Encoding::Word)
    }

    /// Words of memory `count` elements take: their bytes, rounded up to
    /// a whole word.
    pub fn storage_words(&self, count: usize) -> usize {
        (count * self.bytes).div_ceil(4)
    }

    /// The Kernel's word for scalar `scalar` of element `element`, read
    /// from `storage` and widened.
    #[inline]
    pub fn load(&self, storage: &[u32], element: usize, scalar: usize) -> u32 {
        let encoding = self.encodings[scalar];
        let byte = element * self.bytes + self.offsets[scalar];
        let bits = encoding::load(storage, byte, encoding.bytes());
        encoding.decode(self.scalars[scalar], bits)
    }

    /// Narrows `word` into scalar `scalar` of element `element`.
    #[inline]
    pub fn store(&self, storage: &mut [u32], element: usize, scalar: usize, word: u32) {
        let encoding = self.encodings[scalar];
        let byte = element * self.bytes + self.offsets[scalar];
        encoding::store(storage, byte, encoding.bytes(), encoding.encode(word));
    }

    /// The memory `count` elements of the Kernel's words take, narrowed.
    /// A wide schema's memory is its words.
    pub fn pack(&self, values: &[u32]) -> Vec<u32> {
        if self.is_wide() {
            return values.to_vec();
        }
        let per = self.value_words().max(1);
        let count = values.len() / per;
        let mut storage = vec![0; self.storage_words(count)];
        for (element, words) in values.chunks_exact(per).enumerate() {
            for (scalar, word) in words.iter().enumerate() {
                self.store(&mut storage, element, scalar, *word);
            }
        }
        storage
    }

    /// The Kernel's words of `count` elements kept in `storage`, widened.
    pub fn unpack(&self, storage: &[u32], count: usize) -> Vec<u32> {
        if self.is_wide() {
            return storage[..count * self.value_words()].to_vec();
        }
        let mut values = Vec::with_capacity(count * self.value_words());
        for element in 0..count {
            for scalar in 0..self.value_words() {
                values.push(self.load(storage, element, scalar));
            }
        }
        values
    }

    /// Whether elements are pixels: four floats, however they are kept.
    pub fn is_pixel(&self) -> bool {
        self.scalars.len() == 4 && self.scalars.iter().all(|s| *s == Scalar::Float)
    }

    /// The layout, for a message a person reads: "four floats (a pixel)",
    /// "two floats and an integer", "four floats kept as unorm8". The
    /// spelling `Display` writes is what a file and a wire carry; this is
    /// what a diagnostic says.
    pub fn describe(&self) -> String {
        let kept = match self.uniform_encoding() {
            Some(Encoding::Word) | None if self.is_wide() => String::new(),
            Some(encoding) => format!(" kept as {encoding}"),
            None => format!(" kept as {self}"),
        };
        if self.is_pixel() {
            return format!("four floats (a pixel){kept}");
        }
        if self.scalars.is_empty() {
            return "nothing".to_owned();
        }
        let mut runs: Vec<(Scalar, usize)> = Vec::new();
        for scalar in &self.scalars {
            match runs.last_mut() {
                Some((last, count)) if last == scalar => *count += 1,
                _ => runs.push((*scalar, 1)),
            }
        }
        for (scalar, count) in &mut runs {
            *count /= scalar.words();
        }
        let counted = |scalar: Scalar, count: usize| {
            let (one, many) = scalar.noun();
            let name = if count == 1 { one } else { many };
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
        let listed = match parts.len() {
            1 => parts[0].clone(),
            _ => format!(
                "{} and {}",
                parts[..parts.len() - 1].join(", "),
                parts[parts.len() - 1]
            ),
        };
        format!("{listed}{kept}")
    }
}

impl std::str::FromStr for Schema {
    type Err = String;

    /// Parses the spelling `Display` writes: scalars as `f`, `i`, `b`, `d`,
    /// `l`, separated by commas, each followed by `:` and its encoding when
    /// it is narrower than a word (`f:unorm8`). A `d` or an `l` is one
    /// 64-bit scalar, which is two words.
    fn from_str(text: &str) -> Result<Self, String> {
        let mut scalars = Vec::new();
        let mut encodings = Vec::new();
        for part in text.split(',') {
            let (letter, kept) = match part.trim().split_once(':') {
                Some((letter, kept)) => (letter, Some(kept)),
                None => (part.trim(), None),
            };
            let scalar = match letter {
                "f" => Scalar::Float,
                "i" => Scalar::Int,
                "b" => Scalar::Bool,
                "d" => Scalar::Double,
                "l" => Scalar::Long,
                other => return Err(format!("`{other}` is not a scalar (f, i, b, d, or l)")),
            };
            let encoding = match kept {
                None => Encoding::Word,
                Some(name) => {
                    Encoding::parse(name).ok_or_else(|| format!("`{name}` is not an encoding"))?
                }
            };
            for _ in 0..scalar.words() {
                scalars.push(scalar);
                encodings.push(encoding);
            }
        }
        Self::encoded(scalars, encodings)
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut k = 0;
        while k < self.scalars.len() {
            let (scalar, encoding) = (self.scalars[k], self.encodings[k]);
            if k > 0 {
                f.write_str(",")?;
            }
            f.write_str(scalar.letter())?;
            if encoding != Encoding::Word {
                write!(f, ":{encoding}")?;
            }
            k += scalar.words();
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

/// Elements laid out by a [`Schema`] over a [`Domain`], first axis
/// fastest. What it holds is its storage: the Kernel's words for a wide
/// schema, narrower scalars packed for a narrow one.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldData {
    schema: Schema,
    domain: Domain,
    storage: Vec<u32>,
}

impl FieldData {
    /// A zeroed Field.
    pub fn new(schema: Schema, domain: Domain) -> Result<Self, HostError> {
        if domain.is_empty() || schema.value_words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let storage = vec![0; schema.storage_words(domain.count())];
        Ok(Self {
            schema,
            domain,
            storage,
        })
    }

    /// Wraps existing storage; the length must be what the schema lays
    /// the domain out in ([`Schema::storage_words`]).
    pub fn from_storage(
        schema: Schema,
        domain: Domain,
        storage: Vec<u32>,
    ) -> Result<Self, HostError> {
        if domain.is_empty() || schema.value_words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let expected = schema.storage_words(domain.count());
        if storage.len() != expected {
            return Err(HostError::WordCount {
                expected,
                actual: storage.len(),
            });
        }
        Ok(Self {
            schema,
            domain,
            storage,
        })
    }

    /// A Field of the Kernel's words, one per scalar per element, kept as
    /// the schema says: narrowed where it is narrow.
    pub fn from_values(schema: Schema, domain: Domain, values: &[u32]) -> Result<Self, HostError> {
        if domain.is_empty() || schema.value_words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let expected = domain.count() * schema.value_words();
        if values.len() != expected {
            return Err(HostError::WordCount {
                expected,
                actual: values.len(),
            });
        }
        let storage = schema.pack(values);
        Ok(Self {
            schema,
            domain,
            storage,
        })
    }

    /// A grid of pixels from an image.
    pub fn from_image(image: Image) -> Self {
        let storage = image
            .pixels
            .iter()
            .flat_map(|pixel| pixel.iter().map(|v| v.to_bits()))
            .collect();
        Self {
            schema: Schema::pixel(),
            domain: Domain::Grid(image.width, image.height),
            storage,
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

    /// The memory as kept: the Kernel's words when the schema is wide,
    /// packed scalars when it is narrow. What a file or a device holds.
    pub fn storage(&self) -> &[u32] {
        &self.storage
    }

    /// The Kernel's words, element after element: the storage itself
    /// when it is wide, widened otherwise.
    pub fn values(&self) -> std::borrow::Cow<'_, [u32]> {
        if self.schema.is_wide() {
            std::borrow::Cow::Borrowed(&self.storage)
        } else {
            std::borrow::Cow::Owned(self.schema.unpack(&self.storage, self.domain.count()))
        }
    }

    /// The Kernel's words of one element by linear index.
    pub fn element(&self, index: u32) -> Result<std::borrow::Cow<'_, [u32]>, HostError> {
        let count = self.domain.count();
        if index as usize >= count {
            return Err(HostError::ElementOutOfBounds {
                index,
                count: count as u32,
            });
        }
        let stride = self.schema.value_words();
        let index = index as usize;
        if self.schema.is_wide() {
            return Ok(std::borrow::Cow::Borrowed(
                &self.storage[index * stride..(index + 1) * stride],
            ));
        }
        Ok(std::borrow::Cow::Owned(
            (0..stride)
                .map(|scalar| self.schema.load(&self.storage, index, scalar))
                .collect(),
        ))
    }

    /// The pixels of a Field of pixels, as an image of `width` by
    /// `height * depth` rows, widened when they are kept narrow.
    pub fn to_image(&self) -> Result<Image, HostError> {
        if !self.schema.is_pixel() {
            return Err(HostError::NotPixels);
        }
        let pixels = self
            .values()
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
                FieldData::from_storage(schema, domain, words)?
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
                FieldData::from_storage(schema, domain, words)?
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
        if domain.is_empty() || schema.value_words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let wanted = schema.storage_words(domain.count());
        let at = self.spare.iter().position(|buffer| buffer.len() == wanted);
        Ok(at.map(|at| self.spare.swap_remove(at)))
    }

    /// Takes ownership of existing data.
    pub fn adopt(&mut self, data: FieldData) -> Result<Handle, HostError> {
        let requested = data.storage.len();
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
        if domain.is_empty() || schema.value_words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let requested = schema.storage_words(domain.count());
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

    /// The Kernel's words of one element by linear index, widened when the
    /// Field keeps them narrow.
    pub fn element(
        &self,
        handle: Handle,
        index: u32,
    ) -> Result<std::borrow::Cow<'_, [u32]>, HostError> {
        self.data(handle)?.element(index)
    }

    /// A Field's storage, to write in place. A backend that fills a Field
    /// it just created writes here rather than into a buffer of its own
    /// that it then copies in: for a million elements that is two copies
    /// of every word that nobody needs. It is the Field's memory as kept,
    /// packed when the schema is narrow.
    pub fn storage_mut(&mut self, handle: Handle) -> Result<&mut [u32], HostError> {
        Ok(&mut self.data_mut(handle)?.storage)
    }

    /// Replaces the whole storage, as kept; the length must match.
    pub fn write_storage(&mut self, handle: Handle, storage: &[u32]) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        if storage.len() != data.storage.len() {
            return Err(HostError::WordCount {
                expected: data.storage.len(),
                actual: storage.len(),
            });
        }
        data.storage.copy_from_slice(storage);
        Ok(())
    }

    /// Replaces every element with the Kernel's words, narrowed as the
    /// schema keeps them; one word per scalar per element.
    pub fn write_values(&mut self, handle: Handle, values: &[u32]) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        let expected = data.domain.count() * data.schema.value_words();
        if values.len() != expected {
            return Err(HostError::WordCount {
                expected,
                actual: values.len(),
            });
        }
        if data.schema.is_wide() {
            data.storage.copy_from_slice(values);
        } else {
            data.storage = data.schema.pack(values);
        }
        Ok(())
    }

    /// Replaces one element with the Kernel's words, narrowed as the
    /// schema keeps them.
    pub fn write_element(
        &mut self,
        handle: Handle,
        index: u32,
        words: &[u32],
    ) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        let stride = data.schema.value_words();
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
        for (scalar, word) in words.iter().enumerate() {
            data.schema
                .store(&mut data.storage, index as usize, scalar, *word);
        }
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
        slot.storage = Storage::Data(FieldData::from_storage(schema, domain, words)?);
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
            self.keep_spare(data.storage);
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

    /// The element a pixel is, in a Field of pixels.
    fn pixel_index(data: &FieldData, x: u32, y: u32) -> Result<usize, HostError> {
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
        Ok(y as usize * width as usize + x as usize)
    }

    /// Reads one pixel of a Field of pixels, widened if it is kept narrow.
    pub fn read_pixel(&self, handle: Handle, x: u32, y: u32) -> Result<Pixel, HostError> {
        let data = self.data(handle)?;
        let index = Self::pixel_index(data, x, y)?;
        let channel = |k: usize| f32::from_bits(data.schema.load(&data.storage, index, k));
        Ok([channel(0), channel(1), channel(2), channel(3)])
    }

    /// Writes one pixel of a Field of pixels, narrowed if it is kept so.
    pub fn write_pixel(
        &mut self,
        handle: Handle,
        x: u32,
        y: u32,
        pixel: Pixel,
    ) -> Result<(), HostError> {
        let data = self.data_mut(handle)?;
        let index = Self::pixel_index(data, x, y)?;
        for (k, v) in pixel.iter().enumerate() {
            data.schema.store(&mut data.storage, index, k, v.to_bits());
        }
        Ok(())
    }

    /// Replaces every pixel of a Field of pixels; the length must match.
    pub fn write_image(&mut self, handle: Handle, pixels: &[Pixel]) -> Result<(), HostError> {
        let data = self.data(handle)?;
        if !data.schema.is_pixel() {
            return Err(HostError::NotPixels);
        }
        let count = data.domain.count();
        if pixels.len() != count {
            return Err(HostError::PixelCount {
                expected: count,
                actual: pixels.len(),
            });
        }
        let values: Vec<u32> = pixels
            .iter()
            .flat_map(|p| p.iter().map(|v| v.to_bits()))
            .collect();
        self.write_values(handle, &values)
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
        host.write_storage(first, &[7; 256]).unwrap();
        host.release(first).unwrap();
        // A plain `create` is zeroed however the words come to it.
        let again = host.create(schema.clone(), Domain::Rows(256)).unwrap();
        assert!(host.data(again).unwrap().storage().iter().all(|w| *w == 0));
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
        host.write_storage(rows, &words).unwrap();
        assert_eq!(&*host.element(rows, 2).unwrap(), &[6, 7, 8]);
        host.write_element(rows, 3, &[1, 2, 3]).unwrap();
        assert_eq!(&*host.element(rows, 3).unwrap(), &[1, 2, 3]);
        assert_eq!(
            host.element(rows, 4).unwrap_err(),
            HostError::ElementOutOfBounds { index: 4, count: 4 }
        );
        host.make_resident(rows, Box::new(1u8)).unwrap();
        assert_eq!(host.schema(rows).unwrap(), &schema);
        assert_eq!(host.data(rows).unwrap_err(), HostError::Resident(rows));
        host.materialize(rows, words.clone()).unwrap();
        assert_eq!(host.data(rows).unwrap().storage(), &words[..]);

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
            host.write_storage(image, &[0; 3]).unwrap_err(),
            HostError::WordCount {
                expected: 16,
                actual: 3
            }
        );
    }

    #[test]
    fn a_narrow_field_is_charged_and_kept_as_its_bytes_and_read_as_values() {
        // A 3 by 2 picture of bytes: six pixels of four bytes, six words,
        // where the same picture of floats is twenty-four.
        let mut host = Host::new(64);
        let bytes = Schema::pixel().with_encoding(Encoding::Unorm8).unwrap();
        assert!(bytes.is_pixel() && !bytes.is_wide());
        assert_eq!(bytes.element_bytes(), 4);
        assert_eq!(bytes.to_string(), "f:unorm8,f:unorm8,f:unorm8,f:unorm8");
        assert_eq!(bytes.describe(), "four floats (a pixel) kept as unorm8");
        assert_eq!(bytes.to_string().parse::<Schema>().unwrap(), bytes);
        let picture = host.create(bytes, Domain::Grid(3, 2)).unwrap();
        assert_eq!(host.used_words(), 6);
        host.write_pixel(picture, 2, 1, [1.0, 0.5, -3.0, 0.25])
            .unwrap();
        let pixel = host.read_pixel(picture, 2, 1).unwrap();
        assert_eq!(pixel[0], 1.0);
        assert_eq!(pixel[2], 0.0, "clamped");
        assert_eq!(pixel[1], 128.0 * encoding::UNORM8);
        assert_eq!(host.data(picture).unwrap().storage()[5], 0x40_00_80_FF);
        let image = host.image(picture).unwrap();
        assert_eq!(image.pixels()[5], pixel);

        // Rows of bytes: five of them fill two words, and the last
        // three bytes of the second stay zero.
        let row = Schema::new(vec![Scalar::Int])
            .with_encoding(Encoding::I8)
            .unwrap();
        let values: Vec<u32> = [1i32, -1, 300, 127, -128]
            .iter()
            .map(|v| *v as u32)
            .collect();
        let data = FieldData::from_values(row, Domain::Rows(5), &values).unwrap();
        assert_eq!(data.storage(), &[0x7F_2C_FF_01, 0x80]);
        let back: Vec<i32> = data.values().iter().map(|w| *w as i32).collect();
        assert_eq!(back, [1, -1, 44, 127, -128]);

        // What cannot be kept is refused where the schema is made.
        assert!(
            Schema::new(vec![Scalar::Int])
                .with_encoding(Encoding::F16)
                .is_err()
        );
        assert!(
            Schema::encoded(
                vec![Scalar::Float, Scalar::Float],
                vec![Encoding::Unorm8, Encoding::F16]
            )
            .is_err(),
            "a half at byte one"
        );
    }

    #[test]
    fn a_64_bit_scalar_is_two_words_written_as_one_letter() {
        let schema: Schema = "d,l,f".parse().unwrap();
        assert_eq!(
            schema.scalars(),
            &[
                Scalar::Double,
                Scalar::Double,
                Scalar::Long,
                Scalar::Long,
                Scalar::Float
            ]
        );
        assert_eq!(schema.value_words(), 5);
        assert_eq!(schema.element_bytes(), 20);
        assert_eq!(schema.to_string(), "d,l,f");
        assert_eq!(schema.describe(), "one double, one long and one float");
        // A word without its partner is no scalar, and a 64-bit scalar
        // is kept whole.
        assert!(Schema::encoded(vec![Scalar::Double], vec![Encoding::Word]).is_err());
        assert!(
            Schema::encoded(vec![Scalar::Double, Scalar::Long], vec![Encoding::Word; 2]).is_err()
        );
        assert!("d:f16".parse::<Schema>().is_err());
        assert!("l:u8".parse::<Schema>().is_err());
    }
}
