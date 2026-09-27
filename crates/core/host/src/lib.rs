#![warn(missing_docs)]
//! Host Core (L1).
//!
//! The Host owns primitive resources and hands out opaque handles; it
//! holds no product logic. Every resource has one owner, an explicit
//! budget, and a handle that is invalid after release.
//!
//! The one resource is a Field: a [`Domain`] of elements laid out by a
//! [`Schema`]. A Kernel sees one 32-bit word per scalar (two for a 64-bit
//! one); memory may keep a scalar narrower ([`Encoding`]). An image is a
//! grid of four-float elements; the pixel helpers are views of it.

pub mod encoding;

pub use encoding::Encoding;

use std::any::Any;
use std::fmt;

/// Opaque handle to a Host-owned Field, images included. Its generation
/// keeps a released slot out of reach of stale copies.
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

/// What one word of an element is to a Kernel. `Double` and `Long` words
/// come in pairs, low first, and are kept whole; the 32-bit kinds may be
/// kept narrower ([`Encoding`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// An `f32`.
    Float,
    /// An `i32`.
    Int,
    /// A `0` or `1`.
    Bool,
    /// One word of an `f64`, low first.
    Double,
    /// One word of an `i64`, low first.
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

    /// Words a scalar of this kind takes.
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

/// The flattened element layout of a Field: scalars in declaration order,
/// each with its encoding. Names live in the declaring Kernel.
///
/// Elements are packed little-endian, scalar after scalar and element
/// after element. Every scalar sits at a multiple of its own size, so
/// none crosses a 32-bit word.
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
    /// an encoding does not fit its scalar or a scalar would be misaligned.
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

    /// Words a Kernel sees per element: one per scalar.
    pub fn value_words(&self) -> usize {
        self.scalars.len()
    }

    /// Bytes an element takes in memory.
    pub fn element_bytes(&self) -> usize {
        self.bytes
    }

    /// Whether every scalar is kept as a whole word.
    pub fn is_wide(&self) -> bool {
        self.encodings.iter().all(|e| *e == Encoding::Word)
    }

    /// Words of memory `count` elements take, rounded up to a whole word.
    pub fn storage_words(&self, count: usize) -> usize {
        // Counted in 64 bits, so a 32-bit `usize` cannot overflow.
        let words = (count as u64).saturating_mul(self.bytes as u64).div_ceil(4);
        usize::try_from(words).unwrap_or(usize::MAX)
    }

    /// The Kernel's word for scalar `scalar` of element `element`, widened.
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

    /// The memory `count` elements of Kernel words take, narrowed.
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

    /// Whether a Field of this layout over `domain` is a picture: a Grid
    /// of pixels. The one rule for what is shown, read and written as an
    /// image; a row or a volume of pixels is not one.
    pub fn is_picture(&self, domain: Domain) -> bool {
        self.is_pixel() && domain.shape() == Shape::Grid
    }

    /// The layout for a diagnostic ("four floats kept as unorm8"); `Display`
    /// writes the spelling files and the wire carry.
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

    /// Parses the spelling `Display` writes: `f`, `i`, `b`, `d`, `l`, comma
    /// separated, each with `:encoding` when narrower than a word
    /// (`f:unorm8`). `d` and `l` are 64-bit scalars of two words.
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

/// How a Field's elements are addressed: a dense lattice a Kernel maps
/// over. Irregular data is rows plus indices.
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

/// The extent of a Field. Every side is at least one.
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

    /// Total number of elements, saturating at `usize::MAX` (past
    /// [`MAX_ELEMENTS`]).
    pub fn count(self) -> usize {
        (self.width() as usize)
            .saturating_mul(self.height() as usize)
            .saturating_mul(self.depth() as usize)
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

/// The most elements one Field holds: elements are found by a 32-bit
/// index everywhere past the Host.
pub const MAX_ELEMENTS: usize = u32::MAX as usize;

/// The words of memory `schema` over `domain` takes, when the domain is
/// allowed: not empty, at most [`MAX_ELEMENTS`]. Allocates nothing.
fn words_for(schema: &Schema, domain: Domain) -> Result<usize, HostError> {
    if domain.is_empty() || schema.value_words() == 0 {
        return Err(HostError::ZeroSize);
    }
    // Counted in 64 bits, so a saturated 32-bit count is refused.
    let exact = u64::from(domain.width())
        .saturating_mul(u64::from(domain.height()))
        .saturating_mul(u64::from(domain.depth()));
    if exact > MAX_ELEMENTS as u64 {
        return Err(HostError::TooManyElements(domain));
    }
    Ok(schema.storage_words(domain.count()))
}

/// Elements laid out by a [`Schema`] over a [`Domain`], first axis
/// fastest. Storage holds Kernel words for a wide schema, packed scalars
/// for a narrow one.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldData {
    schema: Schema,
    domain: Domain,
    storage: Vec<u32>,
}

impl FieldData {
    /// A zeroed Field.
    pub fn new(schema: Schema, domain: Domain) -> Result<Self, HostError> {
        let storage = vec![0; words_for(&schema, domain)?];
        Ok(Self {
            schema,
            domain,
            storage,
        })
    }

    /// Wraps existing storage of [`Schema::storage_words`] length.
    pub fn from_storage(
        schema: Schema,
        domain: Domain,
        storage: Vec<u32>,
    ) -> Result<Self, HostError> {
        let expected = words_for(&schema, domain)?;
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

    /// A Field from the Kernel's words, one per scalar per element, narrowed
    /// as the schema says.
    pub fn from_values(schema: Schema, domain: Domain, values: &[u32]) -> Result<Self, HostError> {
        words_for(&schema, domain)?;
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

    /// The memory as kept, as a file or a device holds it.
    pub fn storage(&self) -> &[u32] {
        &self.storage
    }

    /// The Kernel's words, element after element, widened when narrow.
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
    /// `height * depth` rows, widened.
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

/// A dense row-major RGBA float grid: the pixel view of a Field at
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

    /// Wraps existing pixel data of the image's length.
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

/// Errors reported by the Host. The Host never silently clamps, recovers,
/// or reallocates.
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
    /// The domain holds more than [`MAX_ELEMENTS`].
    TooManyElements(Domain),
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
    /// The Field's words are on a device; materialize it first.
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
            Self::TooManyElements(domain) => write!(
                f,
                "a Field over {domain} holds more than {MAX_ELEMENTS} elements"
            ),
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
    /// Words on a device, held by an object its backend understands.
    Resident {
        schema: Schema,
        domain: Domain,
        device: Box<dyn Any + Send + Sync>,
    },
}

struct Slot {
    generation: u32,
    storage: Storage,
    /// Budget charged on adoption, returned on release.
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

/// What a buffer holds when handed to a caller that writes every word,
/// so a word left unwritten shows.
const POISON: u32 = 0xDEAD_BEEF;

/// The default loop iterations of one Kernel run, all elements together.
pub const DEFAULT_WORK: u64 = 4_000_000_000;

/// The most loop iterations a Host may let one run take.
pub const MAX_WORK: u64 = 1 << 40;

/// The Host: a bounded store of primitive resources, and a bound on the
/// compute one run may spend.
pub struct Host {
    slots: Vec<Slot>,
    free: Vec<u32>,
    budget_words: usize,
    used_words: usize,
    /// Budget words set aside for another Host ([`Host::reserve`]).
    reserved_words: usize,
    /// The loop iterations one run of a Kernel may take.
    work: u64,
    /// Word buffers released Fields gave back, largest last, reused by
    /// `create`.
    spare: Vec<Vec<u32>>,
}

impl Host {
    /// Creates a Host holding at most `budget_words` live words; a pixel is
    /// four words.
    pub fn new(budget_words: usize) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            spare: Vec::new(),
            budget_words,
            used_words: 0,
            reserved_words: 0,
            work: DEFAULT_WORK,
        }
    }

    /// The loop iterations one Kernel run may take, all elements together;
    /// it keeps a run from holding a device until a driver resets it.
    pub fn work(&self) -> u64 {
        self.work
    }

    /// Sets [`Host::work`], clamped to `1..=MAX_WORK`.
    pub fn set_work(&mut self, iterations: u64) {
        self.work = iterations.clamp(1, MAX_WORK);
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

    /// A Field whose words are whatever its buffer holds, for a backend that
    /// writes every word before any is read, as a map over a domain does.
    pub fn create_for_fill(&mut self, schema: Schema, domain: Domain) -> Result<Handle, HostError> {
        let data = match self.take_spare(&schema, domain)? {
            Some(mut words) => {
                // Poisoned where checked, so a word left unwritten fails
                // a test.
                if cfg!(debug_assertions) {
                    words.fill(POISON);
                }
                FieldData::from_storage(schema, domain, words)?
            }
            None => FieldData::new(schema, domain)?,
        };
        self.adopt(data)
    }

    /// A buffer of exactly this Field's words, reused when one is spare.
    fn take_spare(
        &mut self,
        schema: &Schema,
        domain: Domain,
    ) -> Result<Option<Vec<u32>>, HostError> {
        let wanted = words_for(schema, domain)?;
        // Charged before anything is allocated.
        self.within_budget(wanted)?;
        let at = self.spare.iter().position(|buffer| buffer.len() == wanted);
        Ok(at.map(|at| self.spare.swap_remove(at)))
    }

    /// Takes ownership of existing data.
    pub fn adopt(&mut self, data: FieldData) -> Result<Handle, HostError> {
        let requested = data.storage.len();
        self.place(requested, Storage::Data(data))
    }

    /// Takes ownership of a Field whose words are on a device; `device` is
    /// what the owning backend needs to find them.
    pub fn adopt_resident(
        &mut self,
        schema: Schema,
        domain: Domain,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<Handle, HostError> {
        let requested = words_for(&schema, domain)?;
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

    /// Whether Fields of these schemas over these domains fit the budget now;
    /// a backend allocating elsewhere asks this first.
    pub fn can_hold(&self, fields: &[(&Schema, Domain)]) -> Result<(), HostError> {
        let mut requested = 0usize;
        for (schema, domain) in fields {
            requested = requested.saturating_add(words_for(schema, *domain)?);
        }
        self.within_budget(requested)
    }

    /// Sets aside room for Fields of these schemas over these domains held
    /// elsewhere (another Host, on another thread) and answers the words set
    /// aside. The budget counts them with this Host's Fields until
    /// [`Host::unreserve`].
    pub fn reserve(&mut self, fields: &[(&Schema, Domain)]) -> Result<usize, HostError> {
        let mut requested = 0usize;
        for (schema, domain) in fields {
            requested = requested.saturating_add(words_for(schema, *domain)?);
        }
        self.within_budget(requested)?;
        self.reserved_words += requested;
        Ok(requested)
    }

    /// Gives back words [`Host::reserve`] set aside.
    pub fn unreserve(&mut self, words: usize) {
        self.reserved_words = self.reserved_words.saturating_sub(words);
    }

    /// Whether `requested` more words fit under the budget.
    fn within_budget(&self, requested: usize) -> Result<(), HostError> {
        let available = self
            .budget_words
            .saturating_sub(self.used_words)
            .saturating_sub(self.reserved_words);
        if requested > available {
            return Err(HostError::BudgetExceeded {
                requested,
                available,
            });
        }
        Ok(())
    }

    fn place(&mut self, requested: usize, storage: Storage) -> Result<Handle, HostError> {
        self.within_budget(requested)?;
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

    /// Width and height, resident or not; a volume reports its first two.
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

    /// The Kernel's words of one element by linear index, widened.
    pub fn element(
        &self,
        handle: Handle,
        index: u32,
    ) -> Result<std::borrow::Cow<'_, [u32]>, HostError> {
        self.data(handle)?.element(index)
    }

    /// A Field's storage as kept, to write in place: a backend fills a new
    /// Field here.
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

    /// Replaces every element with the Kernel's words, one per scalar per
    /// element, narrowed as kept.
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

    /// Replaces one element with the Kernel's words, narrowed as kept.
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

    /// The device object of a resident Field; `None` in Host memory.
    pub fn resident(&self, handle: Handle) -> Result<Option<&(dyn Any + Send + Sync)>, HostError> {
        Ok(match &self.slot(handle)?.storage {
            Storage::Resident { device, .. } => Some(device.as_ref()),
            _ => None,
        })
    }

    /// Moves a Field's words onto a device: the Host drops its copy and keeps
    /// `device`. Layout, extent and identity stay.
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

    /// Brings a resident Field back into Host memory with the downloaded
    /// words, dropping the device object. A Field in memory stays as it is.
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
        // The words go to the spare pile for the next Field of that size.
        if let Storage::Data(data) = storage {
            self.keep_spare(data.storage);
        }
        Ok(())
    }

    /// Keeps a released buffer for reuse. The pile holds a handful of
    /// buffers and at most an eighth of the budget; the rest is freed.
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

    /// A copy of a Field of pixels as an image. Fails with
    /// [`HostError::Resident`] on a device, [`HostError::NotPixels`] for
    /// other layouts.
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

    /// Reads one pixel, widened.
    pub fn read_pixel(&self, handle: Handle, x: u32, y: u32) -> Result<Pixel, HostError> {
        let data = self.data(handle)?;
        let index = Self::pixel_index(data, x, y)?;
        let channel = |k: usize| f32::from_bits(data.schema.load(&data.storage, index, k));
        Ok([channel(0), channel(1), channel(2), channel(3)])
    }

    /// Writes one pixel, narrowed as kept.
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

    /// A picture is a Grid of pixels; a row or a volume of them is not,
    /// nor a Grid of anything else.
    #[test]
    fn a_picture_is_a_grid_of_pixels() {
        let pixel = Schema::pixel();
        assert!(pixel.is_picture(Domain::Grid(4, 4)));
        assert!(!pixel.is_picture(Domain::Rows(16)));
        assert!(!pixel.is_picture(Domain::Volume(4, 4, 4)));
        assert!(!Schema::new(vec![Scalar::Float]).is_picture(Domain::Grid(4, 4)));
    }

    /// Released words are reused, and the pile is bounded to a handful of
    /// buffers and an eighth of the budget.
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

        // The pile holds at most an eighth of the budget.
        for _ in 0..32 {
            let big = host.create(schema.clone(), Domain::Rows(400)).unwrap();
            host.release(big).unwrap();
        }
        let held: usize = host.spare.iter().map(Vec::len).sum();
        assert!(held <= 4096 / 8, "the pile holds {held} words");
        assert_eq!(host.live(), 0);

        // A buffer larger than the pile allows is freed.
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

    /// Reserved room counts against the budget until given back; a reserve
    /// past it is refused whole.
    #[test]
    fn reserved_room_counts_against_the_budget_until_given_back() {
        let mut host = Host::new(32);
        let first = host.create_image(2, 2).unwrap();
        let pixel = Schema::pixel();
        assert_eq!(
            host.reserve(&[(&pixel, Domain::Grid(3, 1)), (&pixel, Domain::Grid(3, 1))]),
            Err(HostError::BudgetExceeded {
                requested: 24,
                available: 16
            })
        );
        let reserved = host.reserve(&[(&pixel, Domain::Grid(3, 1))]).unwrap();
        assert_eq!(reserved, 12);
        assert!(host.create_image(1, 2).is_err(), "4 words are left");
        assert!(host.create_image(1, 1).is_ok());
        host.unreserve(reserved);
        assert!(host.create_image(1, 2).is_ok());
        host.release(first).unwrap();
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

    /// A domain is refused for its cost before allocation: past the budget,
    /// past 32-bit indices, past `usize` overflow.
    #[test]
    fn a_domain_is_refused_before_it_is_allocated() {
        let mut host = Host::new(1 << 20);
        let int = Schema::new(vec![Scalar::Int]);
        // Four billion words: refused, not tried.
        assert_eq!(
            host.create(int.clone(), Domain::Rows(u32::MAX)),
            Err(HostError::BudgetExceeded {
                requested: u32::MAX as usize,
                available: 1 << 20
            })
        );
        for huge in [
            Domain::Grid(u32::MAX, u32::MAX),
            Domain::Volume(1 << 22, 1 << 21, 1 << 21),
            Domain::Volume(u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert_eq!(
                host.create(int.clone(), huge),
                Err(HostError::TooManyElements(huge))
            );
            assert_eq!(
                host.create_for_fill(int.clone(), huge),
                Err(HostError::TooManyElements(huge))
            );
            assert_eq!(
                host.can_hold(&[(&int, huge)]),
                Err(HostError::TooManyElements(huge))
            );
            assert!(FieldData::from_values(int.clone(), huge, &[]).is_err());
        }
        // Several are charged as one.
        let half = Domain::Rows(1 << 19);
        assert!(host.can_hold(&[(&int, half)]).is_ok());
        assert!(
            host.can_hold(&[(&int, half), (&int, half), (&int, half)])
                .is_err()
        );
        assert_eq!(host.used_words(), 0);
    }

    #[test]
    fn a_narrow_field_is_charged_and_kept_as_its_bytes_and_read_as_values() {
        // A 3 by 2 picture of bytes takes six words; of floats,
        // twenty-four.
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

        // Five byte rows fill two words; the last three bytes stay zero.
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
        // A lone word of a pair is refused; a 64-bit scalar is kept whole.
        assert!(Schema::encoded(vec![Scalar::Double], vec![Encoding::Word]).is_err());
        assert!(
            Schema::encoded(vec![Scalar::Double, Scalar::Long], vec![Encoding::Word; 2]).is_err()
        );
        assert!("d:f16".parse::<Schema>().is_err());
        assert!("l:u8".parse::<Schema>().is_err());
    }
}
