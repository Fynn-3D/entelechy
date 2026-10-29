//! The Host's devices a scene's outputs go to and its inputs are fed by:
//! `output name <- source to device;`, `input name: Type = device;`. The
//! language knows no device; this table is the one place that does, each
//! with the structure of Field it takes and gives. A device comes with a
//! primitive of the Host's IO, never for one scene's sake.

use entelechy_kernel::{Scalar, Schema, Shape};

/// A structure of Field a device takes or gives.
pub struct Port {
    /// What it is, as a refusal says it.
    pub says: &'static str,
    /// Whether a Field of this shape and element layout is one.
    fits: fn(Shape, &Schema) -> bool,
}

impl Port {
    /// Whether a Field of `shape` and `schema` is one.
    pub fn fits(&self, shape: Shape, schema: &Schema) -> bool {
        (self.fits)(shape, schema)
    }
}

/// A device: what an output gives it (`takes`) and what it feeds an input
/// (`gives`), either or both.
pub struct Device {
    /// Its name, as a scene writes it after `to` or `=`.
    pub name: &'static str,
    /// What an output gives it; `None` when no output goes to it.
    pub takes: Option<Port>,
    /// What it feeds an input; `None` when no input reads it.
    pub gives: Option<Port>,
    /// Whether it shows what it takes every frame (the window's company),
    /// rather than carrying it out once after a commit.
    pub every_frame: bool,
}

/// Text, a character a row to the first 0.
fn text(shape: Shape, schema: &Schema) -> bool {
    shape == Shape::Rows && schema.scalars() == [Scalar::Int]
}

/// The system clipboard: text, put on it after a commit that changed it,
/// and read from it when it changes.
pub const CLIPBOARD: &str = "clipboard";
/// The Kernels the session watches preparing and the scene they are for.
pub const PROGRESS: &str = "progress";
/// The sound card: a frame's samples.
pub const SOUND: &str = "sound";
/// The system's pointer: its picture.
pub const POINTER: &str = "pointer";
/// The pixel of the pointer's picture it points with.
pub const POINTER_TIP: &str = "pointer_tip";

/// Every device, in the order a refusal lists them.
pub const DEVICES: [Device; 5] = [
    Device {
        name: CLIPBOARD,
        every_frame: false,
        takes: Some(Port {
            says: "text, a `Rows<Int>` of characters to the first 0",
            fits: text,
        }),
        gives: Some(Port {
            says: "text, a `Rows<Int>` of characters and a 0 after the last",
            fits: text,
        }),
    },
    Device {
        name: PROGRESS,
        every_frame: false,
        takes: None,
        gives: Some(Port {
            says: "words, a `Rows<Int>` of the Kernels preparing and the scene they are for",
            fits: text,
        }),
    },
    Device {
        name: SOUND,
        every_frame: true,
        takes: Some(Port {
            says: "a frame's samples, rows of one float (mono) or two (stereo)",
            fits: |shape, schema| {
                shape == Shape::Rows
                    && (1..=2).contains(&schema.scalars().len())
                    && schema.scalars().iter().all(|s| *s == Scalar::Float)
            },
        }),
        gives: None,
    },
    Device {
        name: POINTER,
        every_frame: true,
        takes: Some(Port {
            says: "a picture, a `Grid` of four floats a pixel",
            fits: |shape, schema| shape == Shape::Grid && schema.is_pixel(),
        }),
        gives: None,
    },
    Device {
        name: POINTER_TIP,
        every_frame: true,
        takes: Some(Port {
            says: "rows whose first is two numbers, the pixel across and down from the pointer picture's top left",
            fits: |shape, schema| {
                shape == Shape::Rows
                    && schema.scalars().len() == 2
                    && schema
                        .scalars()
                        .iter()
                        .all(|s| matches!(s, Scalar::Int | Scalar::Float))
            },
        }),
        gives: None,
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

/// Why device `name` does not take a Field of `shape` and `schema` from an
/// output, or `None` when it does.
pub fn fault(name: &str, shape: Shape, schema: &Schema) -> Option<String> {
    let device = match named(name) {
        Ok(device) => device,
        Err(why) => return Some(why),
    };
    let Some(port) = &device.takes else {
        return Some(format!("`{name}` feeds an input and takes no output"));
    };
    (!port.fits(shape, schema)).then(|| {
        format!(
            "`to {name}` takes {}; this is {shape:?} of {}",
            port.says,
            schema.describe()
        )
    })
}

/// Why device `name` does not feed an input declared `ty`, or `None` when
/// it does.
pub fn feed_fault(name: &str, ty: &str) -> Option<String> {
    let device = match named(name) {
        Ok(device) => device,
        Err(why) => return Some(why),
    };
    let Some(port) = &device.gives else {
        return Some(format!("`{name}` takes an output and feeds no input"));
    };
    let fits = declared(ty, false)
        .and_then(|(shape, schema)| Some(port.fits(shape, &schema?)))
        .unwrap_or(false);
    (!fits).then(|| format!("`= {name}` gives {}, not `{ty}`", port.says))
}

/// The shape and, where the element is one of the language's own kept as
/// its words, the element layout of a Field declared `ty` (aliases
/// resolved): `Image` is a `Grid` of `Vec4`; with `bytes`, a `Rows` of
/// `Int` unless the element says how its bytes are kept.
pub fn declared(ty: &str, bytes: bool) -> Option<(Shape, Option<Schema>)> {
    let ty = ty.trim();
    let shape = if bytes || ty.starts_with("Rows<") {
        Shape::Rows
    } else if ty.starts_with("Grid<") || ty == "Image" {
        Shape::Grid
    } else if ty.starts_with("Volume<") {
        Shape::Volume
    } else {
        return None;
    };
    let written = ty
        .split_once('<')
        .and_then(|(_, rest)| rest.strip_suffix('>'))
        .map(str::trim);
    let element = if ty == "Image" {
        "Vec4"
    } else if bytes {
        written.unwrap_or("Int")
    } else {
        written.unwrap_or("")
    };
    let schema = element
        .parse::<entelechy_ir::Type>()
        .ok()
        .filter(|_| !element.contains(" as "))
        .and_then(|element| entelechy_ir::schema_of(element, &[]));
    Some((shape, schema))
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
        assert!(
            fault(PROGRESS, Shape::Rows, &schema(&[Scalar::Int]))
                .unwrap()
                .contains("takes no output")
        );
    }

    #[test]
    fn each_device_feeds_its_structure_and_none_another() {
        assert_eq!(feed_fault(CLIPBOARD, "Rows<Int>"), None);
        assert_eq!(feed_fault(PROGRESS, "Rows<Int>"), None);
        assert!(
            feed_fault(CLIPBOARD, "Rows<Float>")
                .unwrap()
                .contains("not `Rows<Float>`")
        );
        assert!(
            feed_fault(SOUND, "Rows<Float>")
                .unwrap()
                .contains("feeds no input")
        );
    }

    #[test]
    fn a_device_the_table_does_not_know_is_refused_with_the_nearest() {
        let why = fault("sond", Shape::Rows, &schema(&[Scalar::Float])).unwrap();
        assert!(why.contains("no device `sond`"), "{why}");
        assert!(why.contains("sound"), "{why}");
        let why = feed_fault("clipbord", "Rows<Int>").unwrap();
        assert!(why.contains("clipboard"), "{why}");
    }
}
