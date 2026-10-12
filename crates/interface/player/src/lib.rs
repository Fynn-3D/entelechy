#![warn(missing_docs)]
//! The Player's devices, the sound card and the pads, and the pointer
//! held in its window, free of windowing for headless tests. Their use is
//! `entelechy_session::play`'s.

pub mod audio;
pub mod gamepad;
pub mod pointer;
