#![warn(missing_docs)]
//! The Player's devices, kept free of windowing so they can be tested
//! headlessly: the sound card and the pads. What the Player does with
//! them is the session's play (`orla_session::play`), which every
//! interface that plays a scene shares.

pub mod audio;
pub mod gamepad;
