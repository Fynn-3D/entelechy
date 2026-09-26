//! The machine's clocks. `wasm32-unknown-unknown` has none: there,
//! nothing is timed or dated.

use std::time::{Duration, Instant};

/// Whether this machine has clocks to ask.
const CLOCKS: bool = !cfg!(all(target_arch = "wasm32", target_os = "unknown"));

/// A moment to measure from; `None` where there is no clock.
pub fn now() -> Option<Instant> {
    CLOCKS.then(Instant::now)
}

/// How long since `started`; zero where there is no clock.
pub fn since(started: Option<Instant>) -> Duration {
    started.map(|started| started.elapsed()).unwrap_or_default()
}

/// Seconds since the Unix epoch; `None` where there is no clock.
pub fn wall() -> Option<u64> {
    if !CLOCKS {
        return None;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .ok()
}
