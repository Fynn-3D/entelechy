#![warn(missing_docs)]
//! Host Core (L1).
//!
//! The Host owns primitive resources and hands out opaque handles. It knows
//! nothing about Kernels, Graphs, or products. Every resource has one owner
//! (the Host), an explicit budget, and a handle that becomes invalid after
//! release. Nothing here should ever grow product logic; if a feature can be
//! a Kernel, it does not belong in this crate.

use std::any::Any;
use std::fmt;

/// Opaque handle to an image owned by the Host.
///
/// The handle carries a generation so that a released slot cannot be reached
/// through a stale copy of the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageHandle {
    index: u32,
    generation: u32,
}

impl fmt::Display for ImageHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "image#{}.{}", self.index, self.generation)
    }
}

/// One RGBA pixel with linear float components.
pub type Pixel = [f32; 4];

/// One scalar of a Field element, as stored: every scalar is one 32-bit
/// word, so integers cross the boundary exactly.
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

    /// The scalars in order.
    pub fn scalars(&self) -> &[Scalar] {
        &self.0
    }

    /// Words per element.
    pub fn words(&self) -> usize {
        self.0.len()
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

/// Rows of elements laid out by a [`Schema`], one word per scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldData {
    schema: Schema,
    count: u32,
    words: Vec<u32>,
}

impl FieldData {
    /// A zeroed Field.
    pub fn new(schema: Schema, count: u32) -> Result<Self, HostError> {
        if count == 0 || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let words = vec![0; count as usize * schema.words()];
        Ok(Self {
            schema,
            count,
            words,
        })
    }

    /// Wraps existing words; the length must be `count * schema.words()`.
    pub fn from_words(schema: Schema, count: u32, words: Vec<u32>) -> Result<Self, HostError> {
        if count == 0 || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let expected = count as usize * schema.words();
        if words.len() != expected {
            return Err(HostError::PixelCount {
                expected,
                actual: words.len(),
            });
        }
        Ok(Self {
            schema,
            count,
            words,
        })
    }

    /// The element layout.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Number of elements.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// All words, element after element.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The words of one element.
    pub fn element(&self, index: u32) -> Result<&[u32], HostError> {
        if index >= self.count {
            return Err(HostError::OutOfBounds {
                x: index,
                y: 0,
                width: self.count,
                height: 1,
            });
        }
        let stride = self.schema.words();
        Ok(&self.words[index as usize * stride..(index as usize + 1) * stride])
    }
}

/// Image storage: a dense row-major RGBA float grid.
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

    fn offset(&self, x: u32, y: u32) -> Result<usize, HostError> {
        if x >= self.width || y >= self.height {
            return Err(HostError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        Ok(y as usize * self.width as usize + x as usize)
    }
}

/// Errors reported by the Host. They are always explicit; the Host never
/// silently clamps, recovers, or reallocates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The handle does not refer to a live resource.
    InvalidHandle(ImageHandle),
    /// Allocating the resource would exceed the Host budget.
    BudgetExceeded {
        /// Pixels requested by the allocation.
        requested: usize,
        /// Pixels still available under the budget.
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
    /// Images must have at least one pixel in each dimension.
    ZeroSize,
    /// Pixel data length does not match the dimensions.
    PixelCount {
        /// Pixels implied by the dimensions.
        expected: usize,
        /// Pixels supplied.
        actual: usize,
    },
    /// The image's pixels live on a device; materialize it before reading
    /// or writing them from the CPU.
    Resident(ImageHandle),
    /// The handle is a Field, not an image.
    NotAnImage(ImageHandle),
    /// The handle is an image, not a Field.
    NotAField(ImageHandle),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle(handle) => write!(f, "{handle} is not a live image"),
            Self::BudgetExceeded {
                requested,
                available,
            } => write!(
                f,
                "image budget exceeded: requested {requested} pixels, {available} available"
            ),
            Self::OutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(f, "pixel ({x}, {y}) is outside a {width}x{height} image"),
            Self::ZeroSize => write!(f, "images must have non-zero width and height"),
            Self::PixelCount { expected, actual } => {
                write!(f, "expected {expected} pixels, received {actual}")
            }
            Self::Resident(handle) => write!(
                f,
                "{handle} is resident on a device; materialize it before touching its pixels"
            ),
            Self::NotAnImage(handle) => write!(f, "{handle} is a Field, not an image"),
            Self::NotAField(handle) => write!(f, "{handle} is an image, not a Field"),
        }
    }
}

impl std::error::Error for HostError {}

/// Where an image's pixels are.
enum Storage {
    /// Nothing; the slot is free.
    Free,
    /// Pixels in Host memory.
    Pixels(Image),
    /// Field words in Host memory.
    Field(FieldData),
    /// Pixels or words on a device, held by an object only that backend
    /// understands. `schema` is `Some` for a Field.
    Resident {
        width: u32,
        height: u32,
        schema: Option<Schema>,
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
    fn size(&self) -> Option<(u32, u32)> {
        match &self.storage {
            Storage::Free => None,
            Storage::Pixels(image) => Some((image.width, image.height)),
            Storage::Field(field) => Some((field.count, 1)),
            Storage::Resident { width, height, .. } => Some((*width, *height)),
        }
    }
}

/// The Host: a bounded store of primitive resources.
pub struct Host {
    slots: Vec<Slot>,
    free: Vec<u32>,
    budget_pixels: usize,
    used_pixels: usize,
}

impl Host {
    /// Creates a Host that will never hold more than `budget_pixels` live pixels.
    pub fn new(budget_pixels: usize) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            budget_pixels,
            used_pixels: 0,
        }
    }

    /// Number of live images, resident ones included.
    pub fn live_images(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.size().is_some())
            .count()
    }

    /// Number of live images whose pixels are on a device.
    pub fn resident_images(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot.storage, Storage::Resident { .. }))
            .count()
    }

    /// Pixels currently allocated.
    pub fn used_pixels(&self) -> usize {
        self.used_pixels
    }

    /// Total pixel budget.
    pub fn budget_pixels(&self) -> usize {
        self.budget_pixels
    }

    /// Allocates a black image of the given size.
    pub fn create_image(&mut self, width: u32, height: u32) -> Result<ImageHandle, HostError> {
        self.adopt_image(Image::new(width, height)?)
    }

    /// Takes ownership of an existing image.
    pub fn adopt_image(&mut self, image: Image) -> Result<ImageHandle, HostError> {
        let requested = image.pixels.len();
        self.adopt(requested, Storage::Pixels(image))
    }

    /// Takes ownership of an image whose pixels live on a device. `device`
    /// is whatever the owning backend needs to find them again.
    pub fn adopt_resident(
        &mut self,
        width: u32,
        height: u32,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<ImageHandle, HostError> {
        if width == 0 || height == 0 {
            return Err(HostError::ZeroSize);
        }
        let requested = width as usize * height as usize;
        self.adopt(
            requested,
            Storage::Resident {
                width,
                height,
                schema: None,
                device,
            },
        )
    }

    /// Allocates a zeroed Field of `count` elements.
    pub fn create_field(&mut self, schema: Schema, count: u32) -> Result<ImageHandle, HostError> {
        self.adopt_field(FieldData::new(schema, count)?)
    }

    /// Takes ownership of an existing Field.
    pub fn adopt_field(&mut self, field: FieldData) -> Result<ImageHandle, HostError> {
        let requested = field.words.len().div_ceil(4);
        self.adopt(requested, Storage::Field(field))
    }

    /// Takes ownership of a Field whose words live on a device.
    pub fn adopt_resident_field(
        &mut self,
        schema: Schema,
        count: u32,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<ImageHandle, HostError> {
        if count == 0 || schema.words() == 0 {
            return Err(HostError::ZeroSize);
        }
        let requested = (count as usize * schema.words()).div_ceil(4);
        self.adopt(
            requested,
            Storage::Resident {
                width: count,
                height: 1,
                schema: Some(schema),
                device,
            },
        )
    }

    /// The schema of a Field, resident or not; `None` for an image.
    pub fn schema(&self, handle: ImageHandle) -> Result<Option<&Schema>, HostError> {
        Ok(match &self.slot(handle)?.storage {
            Storage::Field(field) => Some(&field.schema),
            Storage::Resident { schema, .. } => schema.as_ref(),
            _ => None,
        })
    }

    /// Borrows the Field behind a handle; fails with [`HostError::Resident`]
    /// when its words are on a device.
    pub fn field(&self, handle: ImageHandle) -> Result<&FieldData, HostError> {
        match &self.slot(handle)?.storage {
            Storage::Field(field) => Ok(field),
            Storage::Resident {
                schema: Some(_), ..
            } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
            _ => Err(HostError::NotAField(handle)),
        }
    }

    /// Replaces every word of a Field; the length must match.
    pub fn write_field(&mut self, handle: ImageHandle, words: &[u32]) -> Result<(), HostError> {
        let field = match &mut self.slot_mut(handle)?.storage {
            Storage::Field(field) => field,
            Storage::Resident {
                schema: Some(_), ..
            } => return Err(HostError::Resident(handle)),
            Storage::Free => return Err(HostError::InvalidHandle(handle)),
            _ => return Err(HostError::NotAField(handle)),
        };
        if words.len() != field.words.len() {
            return Err(HostError::PixelCount {
                expected: field.words.len(),
                actual: words.len(),
            });
        }
        field.words.copy_from_slice(words);
        Ok(())
    }

    /// Brings a resident Field back into Host memory with the words the
    /// backend downloaded.
    pub fn materialize_words(
        &mut self,
        handle: ImageHandle,
        words: Vec<u32>,
    ) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (count, schema) = match &slot.storage {
            Storage::Resident {
                width,
                schema: Some(schema),
                ..
            } => (*width, schema.clone()),
            Storage::Field(_) => return Ok(()),
            _ => return Err(HostError::NotAField(handle)),
        };
        slot.storage = Storage::Field(FieldData::from_words(schema, count, words)?);
        Ok(())
    }

    fn adopt(&mut self, requested: usize, storage: Storage) -> Result<ImageHandle, HostError> {
        let available = self.budget_pixels.saturating_sub(self.used_pixels);
        if requested > available {
            return Err(HostError::BudgetExceeded {
                requested,
                available,
            });
        }
        self.used_pixels += requested;
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
        Ok(ImageHandle {
            index,
            generation: self.slots[index as usize].generation,
        })
    }

    /// The device object of a resident image, or `None` when its pixels are
    /// in Host memory.
    pub fn resident(
        &self,
        handle: ImageHandle,
    ) -> Result<Option<&(dyn Any + Send + Sync)>, HostError> {
        Ok(match &self.slot(handle)?.storage {
            Storage::Resident { device, .. } => Some(device.as_ref()),
            _ => None,
        })
    }

    /// Moves an image's pixels onto a device: the Host drops its copy and
    /// keeps `device` in its place. Size and identity are unchanged.
    pub fn make_resident(
        &mut self,
        handle: ImageHandle,
        device: Box<dyn Any + Send + Sync>,
    ) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (width, height) = slot.size().ok_or(HostError::InvalidHandle(handle))?;
        let schema = match &slot.storage {
            Storage::Field(field) => Some(field.schema.clone()),
            Storage::Resident { schema, .. } => schema.clone(),
            _ => None,
        };
        slot.storage = Storage::Resident {
            width,
            height,
            schema,
            device,
        };
        Ok(())
    }

    /// Brings a resident image back into Host memory with the pixels the
    /// backend downloaded; the device object is dropped.
    pub fn materialize(
        &mut self,
        handle: ImageHandle,
        pixels: Vec<Pixel>,
    ) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (width, height) = slot.size().ok_or(HostError::InvalidHandle(handle))?;
        if matches!(
            slot.storage,
            Storage::Field(_)
                | Storage::Resident {
                    schema: Some(_),
                    ..
                }
        ) {
            return Err(HostError::NotAnImage(handle));
        }
        let image = Image::from_pixels(width, height, pixels)?;
        slot.storage = Storage::Pixels(image);
        Ok(())
    }

    /// Releases an image, resident or not. The handle is invalid afterwards.
    pub fn release_image(&mut self, handle: ImageHandle) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        slot.size().ok_or(HostError::InvalidHandle(handle))?;
        slot.storage = Storage::Free;
        let charge = std::mem::take(&mut slot.charge);
        self.used_pixels -= charge;
        self.free.push(handle.index);
        Ok(())
    }

    /// Borrows the image behind a handle; fails with
    /// [`HostError::Resident`] when its pixels are on a device.
    pub fn image(&self, handle: ImageHandle) -> Result<&Image, HostError> {
        match &self.slot(handle)?.storage {
            Storage::Pixels(image) => Ok(image),
            Storage::Resident { schema: None, .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
            _ => Err(HostError::NotAnImage(handle)),
        }
    }

    /// Image dimensions as `(width, height)`, resident or not.
    pub fn image_size(&self, handle: ImageHandle) -> Result<(u32, u32), HostError> {
        self.slot(handle)?
            .size()
            .ok_or(HostError::InvalidHandle(handle))
    }

    /// Reads one pixel.
    pub fn read_pixel(&self, handle: ImageHandle, x: u32, y: u32) -> Result<Pixel, HostError> {
        let image = self.image(handle)?;
        let offset = image.offset(x, y)?;
        Ok(image.pixels[offset])
    }

    /// Writes one pixel.
    pub fn write_pixel(
        &mut self,
        handle: ImageHandle,
        x: u32,
        y: u32,
        pixel: Pixel,
    ) -> Result<(), HostError> {
        let image = self.image_mut(handle)?;
        let offset = image.offset(x, y)?;
        image.pixels[offset] = pixel;
        Ok(())
    }

    /// Replaces every pixel of an image; the slice length must match.
    pub fn write_image(&mut self, handle: ImageHandle, pixels: &[Pixel]) -> Result<(), HostError> {
        let image = self.image_mut(handle)?;
        if pixels.len() != image.pixels.len() {
            return Err(HostError::PixelCount {
                expected: image.pixels.len(),
                actual: pixels.len(),
            });
        }
        image.pixels.copy_from_slice(pixels);
        Ok(())
    }

    fn slot(&self, handle: ImageHandle) -> Result<&Slot, HostError> {
        self.slots
            .get(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation && slot.size().is_some())
            .ok_or(HostError::InvalidHandle(handle))
    }

    fn slot_mut(&mut self, handle: ImageHandle) -> Result<&mut Slot, HostError> {
        self.slots
            .get_mut(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation && slot.size().is_some())
            .ok_or(HostError::InvalidHandle(handle))
    }

    fn image_mut(&mut self, handle: ImageHandle) -> Result<&mut Image, HostError> {
        match &mut self.slot_mut(handle)?.storage {
            Storage::Pixels(image) => Ok(image),
            Storage::Resident { schema: None, .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
            _ => Err(HostError::NotAnImage(handle)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_become_invalid_after_release() {
        let mut host = Host::new(16);
        let image = host.create_image(2, 2).unwrap();
        host.write_pixel(image, 1, 1, [1.0, 0.5, 0.25, 1.0])
            .unwrap();
        assert_eq!(host.read_pixel(image, 1, 1).unwrap(), [1.0, 0.5, 0.25, 1.0]);
        host.release_image(image).unwrap();
        assert_eq!(
            host.read_pixel(image, 0, 0),
            Err(HostError::InvalidHandle(image))
        );
        assert_eq!(
            host.release_image(image),
            Err(HostError::InvalidHandle(image))
        );
        // A reused slot receives a new generation, so the old handle stays dead.
        let reused = host.create_image(1, 1).unwrap();
        assert_ne!(reused, image);
        assert!(host.read_pixel(image, 0, 0).is_err());
        assert_eq!(host.live_images(), 1);
    }

    #[test]
    fn budget_is_enforced_and_returned() {
        let mut host = Host::new(8);
        let first = host.create_image(2, 2).unwrap();
        assert_eq!(
            host.create_image(3, 3),
            Err(HostError::BudgetExceeded {
                requested: 9,
                available: 4
            })
        );
        host.release_image(first).unwrap();
        assert_eq!(host.used_pixels(), 0);
        assert!(host.create_image(2, 4).is_ok());
    }

    #[test]
    fn resident_images_keep_identity_size_and_budget_but_hide_pixels() {
        let mut host = Host::new(8);
        let image = host.create_image(2, 2).unwrap();
        host.write_pixel(image, 0, 0, [1.0; 4]).unwrap();
        host.make_resident(image, Box::new(42u32)).unwrap();
        assert_eq!(host.image_size(image).unwrap(), (2, 2));
        assert_eq!(host.resident_images(), 1);
        assert_eq!(host.used_pixels(), 4);
        assert_eq!(
            host.read_pixel(image, 0, 0),
            Err(HostError::Resident(image))
        );
        assert_eq!(host.image(image).unwrap_err(), HostError::Resident(image));
        let device = host.resident(image).unwrap().unwrap();
        assert_eq!(device.downcast_ref::<u32>(), Some(&42));
        host.materialize(image, vec![[0.5; 4]; 4]).unwrap();
        assert_eq!(host.read_pixel(image, 1, 1).unwrap(), [0.5; 4]);
        assert!(host.resident(image).unwrap().is_none());
        let resident = host.adopt_resident(2, 2, Box::new(7u8)).unwrap();
        assert_eq!(host.live_images(), 2);
        host.release_image(resident).unwrap();
        host.release_image(image).unwrap();
        assert_eq!(host.used_pixels(), 0);
    }

    #[test]
    fn fields_share_the_store_with_images() {
        let mut host = Host::new(8);
        let schema = Schema::new(vec![Scalar::Float, Scalar::Int, Scalar::Bool]);
        let field = host.create_field(schema.clone(), 4).unwrap();
        assert_eq!(host.image_size(field).unwrap(), (4, 1));
        assert_eq!(
            host.used_pixels(),
            3,
            "twelve words round up to three pixels"
        );
        assert_eq!(host.image(field).unwrap_err(), HostError::NotAnImage(field));
        assert_eq!(host.schema(field).unwrap(), Some(&schema));
        let words: Vec<u32> = (0..12).collect();
        host.write_field(field, &words).unwrap();
        assert_eq!(host.field(field).unwrap().element(2).unwrap(), &[6, 7, 8]);
        host.make_resident(field, Box::new(1u8)).unwrap();
        assert_eq!(host.schema(field).unwrap(), Some(&schema));
        assert_eq!(host.field(field).unwrap_err(), HostError::Resident(field));
        assert_eq!(
            host.materialize(field, vec![]).unwrap_err(),
            HostError::NotAnImage(field)
        );
        host.materialize_words(field, words.clone()).unwrap();
        assert_eq!(host.field(field).unwrap().words(), &words[..]);
        let image = host.create_image(1, 1).unwrap();
        assert_eq!(host.field(image).unwrap_err(), HostError::NotAField(image));
        host.release_image(field).unwrap();
        host.release_image(image).unwrap();
        assert_eq!(host.used_pixels(), 0);
    }

    #[test]
    fn bounds_and_sizes_are_checked() {
        let mut host = Host::new(8);
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
    }
}
