//! The Host's devices a scene's output goes to: `output name <- source to
//! device;`. The language knows no device; this table is the one place
//! that does, each with the structure of Field it takes. A device comes
//! with a primitive of the Host's IO, never for one scene's sake.

use entelechy_kernel::{Scalar, Schema, Shape};

/// A device an output goes to.
pub struct Device {
    /// Its name, as a scene writes it after `to`.
    pub name: &'static str,
    /// What it takes, as a refusal says it.
    pub takes: &'static str,
    /// Whether it shows what it takes every frame (the window's company),
    /// rather than carrying it out once after a commit.
    pub every_frame: bool,
    /// Whether a Field of this shape and element layout is one it takes.
    fits: fn(Shape, &Schema) -> bool,
}

impl Device {
    /// Whether a Field of `shape` and `schema` is one this device takes.
    pub fn fits(&self, shape: Shape, schema: &Schema) -> bool {
        (self.fits)(shape, schema)
    }
}

/// The system clipboard: text, put on it after a commit that changed it.
pub const CLIPBOARD: &str = "clipboard";
/// The sound card: a frame's samples.
pub const SOUND: &str = "sound";
/// The system's pointer: its picture.
pub const POINTER: &str = "pointer";
/// The pixel of the pointer's picture it points with.
pub const POINTER_TIP: &str = "pointer_tip";

/// Every device, in the order a refusal lists them.
pub const DEVICES: [Device; 4] = [
    Device {
        name: CLIPBOARD,
        every_frame: false,
        takes: "text, a `Rows<Int>` of characters to the first 0",
        fits: |shape, schema| shape == Shape::Rows && schema.scalars() == [Scalar::Int],
    },
    Device {
        name: SOUND,
        every_frame: true,
        takes: "a frame's samples, rows of one float (mono) or two (stereo)",
        fits: |shape, schema| {
            shape == Shape::Rows
                && (1..=2).contains(&schema.scalars().len())
                && schema.scalars().iter().all(|s| *s == Scalar::Float)
        },
    },
    Device {
        name: POINTER,
        every_frame: true,
        takes: "a picture, a `Grid` of four floats a pixel",
        fits: |shape, schema| shape == Shape::Grid && schema.is_pixel(),
    },
    Device {
        name: POINTER_TIP,
        every_frame: true,
        takes: "rows whose first is two numbers, the pixel across and down from the pointer picture's top left",
        fits: |shape, schema| {
            shape == Shape::Rows
                && schema.scalars().len() == 2
                && schema
                    .scalars()
                    .iter()
                    .all(|s| matches!(s, Scalar::Int | Scalar::Float))
        },
    },
];

/// The device named `name`, or why there is none: the nearest name the
/// table knows, when one is near.
pub fn named(name: &str) -> Result<&'static Device, String> {
    DEVICES.iter().find(|d| d.name == name).ok_or_else(|| {
        let names = DEVICES.iter().map(|d| d.name);
        let known: Vec<&str> = names.clone().collect();
        format!(
            "no device `{name}`: an output goes `to` {}{}",
            known.join(", "),
            entelechy_language::did_you_mean(name, names)
        )
    })
}

/// Why device `name` does not take a Field of `shape` and `schema`, or
/// `None` when it does.
pub fn fault(name: &str, shape: Shape, schema: &Schema) -> Option<String> {
    let device = match named(name) {
        Ok(device) => device,
        Err(why) => return Some(why),
    };
    (!device.fits(shape, schema)).then(|| {
        format!(
            "`to {name}` takes {}; this is {:?} of {}",
            device.takes,
            shape,
            schema.describe()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(scalars: &[Scalar]) -> Schema {
        Schema::new(scalars.to_vec())
    }

    #[test]
    fn each_device_takes_its_structure_and_refuses_another() {
        let float = Scalar::Float;
        assert_eq!(fault(SOUND, Shape::Rows, &schema(&[float, float])), None);
        assert_eq!(fault(SOUND, Shape::Rows, &schema(&[float])), None);
        assert!(fault(SOUND, Shape::Grid, &schema(&[float, float])).is_some());
        assert!(fault(SOUND, Shape::Rows, &schema(&[float; 3])).is_some());
        assert_eq!(fault(POINTER, Shape::Grid, &schema(&[float; 4])), None);
        assert!(fault(POINTER, Shape::Rows, &schema(&[float; 4])).is_some());
        assert_eq!(
            fault(POINTER_TIP, Shape::Rows, &schema(&[Scalar::Int; 2])),
            None
        );
        assert_eq!(fault(CLIPBOARD, Shape::Rows, &schema(&[Scalar::Int])), None);
        assert!(fault(CLIPBOARD, Shape::Rows, &schema(&[float])).is_some());
    }

    #[test]
    fn a_device_the_table_does_not_know_is_refused_with_the_nearest() {
        let why = fault("sond", Shape::Rows, &schema(&[Scalar::Float])).unwrap();
        assert!(why.contains("no device `sond`"), "{why}");
        assert!(why.contains("sound"), "{why}");
    }
}
