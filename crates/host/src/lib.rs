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
    /// Pixels on a device, held by an object only that backend understands.
    Resident {
        width: u32,
        height: u32,
        device: Box<dyn Any + Send + Sync>,
    },
}

struct Slot {
    generation: u32,
    storage: Storage,
}

impl Slot {
    fn size(&self) -> Option<(u32, u32)> {
        match &self.storage {
            Storage::Free => None,
            Storage::Pixels(image) => Some((image.width, image.height)),
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
                device,
            },
        )
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
                index
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    storage,
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
        slot.storage = Storage::Resident {
            width,
            height,
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
        let image = Image::from_pixels(width, height, pixels)?;
        slot.storage = Storage::Pixels(image);
        Ok(())
    }

    /// Releases an image, resident or not. The handle is invalid afterwards.
    pub fn release_image(&mut self, handle: ImageHandle) -> Result<(), HostError> {
        let slot = self.slot_mut(handle)?;
        let (width, height) = slot.size().ok_or(HostError::InvalidHandle(handle))?;
        slot.storage = Storage::Free;
        self.used_pixels -= width as usize * height as usize;
        self.free.push(handle.index);
        Ok(())
    }

    /// Borrows the image behind a handle; fails with
    /// [`HostError::Resident`] when its pixels are on a device.
    pub fn image(&self, handle: ImageHandle) -> Result<&Image, HostError> {
        match &self.slot(handle)?.storage {
            Storage::Pixels(image) => Ok(image),
            Storage::Resident { .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
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
            Storage::Resident { .. } => Err(HostError::Resident(handle)),
            Storage::Free => Err(HostError::InvalidHandle(handle)),
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
