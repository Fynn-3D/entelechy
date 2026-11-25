//! The importer's readers (glTF/GLB, symphonia audio, standard MIDI, SVG,
//! TrueType and OpenType fonts), a library so the exporter's tests can
//! read back what it wrote. The font reader is always there; the rest
//! are the `audio`, `models` and `vector` features, on by default.

#[cfg(feature = "audio")]
pub mod audio;
pub mod font;
#[cfg(feature = "models")]
pub mod gltf;
pub mod midi;
#[cfg(feature = "vector")]
pub mod svg;
pub mod texture;
