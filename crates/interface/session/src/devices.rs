//! Host device names and the Field structures their output ports accept.
//!
//! Source checking and diagnostics live in `entelechy_documents::devices`;
//! runtime presentation and device feeds read this table whether source
//! documents are compiled in or not.

use entelechy_kernel::{Scalar, Schema, Shape};

/// The clipboard: text, a character a row to the first 0.
pub(crate) const CLIPBOARD: &str = "clipboard";
/// The Kernels the session watches preparing and the scene they are for.
pub(crate) const PROGRESS: &str = "progress";
/// The sound card: a frame's samples.
pub(crate) const SOUND: &str = "sound";
/// The pointer's picture.
pub(crate) const POINTER: &str = "pointer";
/// The pixel of the pointer's picture it points with.
pub(crate) const POINTER_TIP: &str = "pointer_tip";

/// Whether `device` presents its output every frame alongside the picture.
pub(crate) fn every_frame(device: &str) -> bool {
    matches!(device, SOUND | POINTER | POINTER_TIP)
}

/// Whether an output sent `to` `device` accepts a Field of `shape` and
/// `schema`.
pub(crate) fn fits_output(device: &str, shape: Shape, schema: &Schema) -> bool {
    match device {
        CLIPBOARD => shape == Shape::Rows && schema.scalars() == [Scalar::Int],
        SOUND => {
            shape == Shape::Rows
                && (1..=2).contains(&schema.scalars().len())
                && schema.scalars().iter().all(|s| *s == Scalar::Float)
        }
        POINTER => shape == Shape::Grid && schema.is_pixel(),
        POINTER_TIP => {
            shape == Shape::Rows
                && schema.scalars().len() == 2
                && schema
                    .scalars()
                    .iter()
                    .all(|s| matches!(s, Scalar::Int | Scalar::Float))
        }
        _ => false,
    }
}

#[cfg(all(test, feature = "documents"))]
mod tests {
    use super::*;

    #[test]
    fn runtime_devices_match_document_device_table() {
        let schemas = [
            Schema::new(vec![Scalar::Int]),
            Schema::new(vec![Scalar::Int, Scalar::Int]),
            Schema::new(vec![Scalar::Float]),
            Schema::new(vec![Scalar::Float, Scalar::Float]),
            Schema::pixel(),
        ];
        let shapes = [Shape::Rows, Shape::Grid, Shape::Volume];
        for entry in &entelechy_documents::devices::DEVICES {
            assert_eq!(every_frame(entry.name), entry.every_frame, "{}", entry.name);
            for shape in shapes {
                for schema in &schemas {
                    let expected = entry
                        .takes
                        .as_ref()
                        .is_some_and(|port| port.fits(shape, schema));
                    assert_eq!(
                        fits_output(entry.name, shape, schema),
                        expected,
                        "{} {shape:?} {schema}",
                        entry.name
                    );
                }
            }
        }
    }
}
