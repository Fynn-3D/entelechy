#![warn(missing_docs)]
//! The command line's own commands over a session: packing a package,
//! rendering frames to files, replaying recorded input, and scaffolding
//! a scene to start from. The session they drive is `orla-session`.

pub mod package;
pub mod render;
/// Replays live in the session, where the Player plays them too.
pub use orla_session::replay;
pub mod scaffold;
