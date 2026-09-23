//! The file formats Orla reads and writes outside its own: PNG, PPM and
//! WAV decoded into Host Fields and encoded out of them, the inflate PNG
//! needs, and the base64 the protocol and glTF carry bytes in.
//!
//! Each is pure bytes in and bytes or a Field out, with no session, no
//! runtime and no dependency beyond the Host, so an importer, an
//! exporter and a session share one decoder and none of them owns it.

#![warn(missing_docs)]

pub mod base64;
pub mod inflate;
pub mod png;
pub mod ppm;
pub mod wav;
