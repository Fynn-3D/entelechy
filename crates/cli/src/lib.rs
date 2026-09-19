#![warn(missing_docs)]
//! Interface layer: a file-based session over the Graph and Kernel runtimes.
//!
//! The session resolves `.orla` sources from a directory and a `.osn` scene
//! from a file, publishes changed Kernels atomically, and keeps the last
//! successful output when a change is rejected. Source resolution lives
//! here, not in the runtimes.

pub mod ppm;
pub mod replay;
pub mod session;
