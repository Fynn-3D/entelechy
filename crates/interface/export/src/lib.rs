//! A mesh and a rig as GLB, and a font as TrueType or WOFF, from the
//! Host's files; and a scene or compiled `.bundle` as one offline page, a
//! standalone native desktop application (`.app`, `.exe`, `.bin`), or a
//! mobile bridge project (`.ios`, `.android`). Sounds, scores and pictures
//! are written by `entelechy_formats::encode`.

pub mod codesign;
pub mod font;
pub mod gltf;
pub mod native;
pub mod page;
pub mod sha;
