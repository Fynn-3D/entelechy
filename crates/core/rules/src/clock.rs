//! The machine's clock, which every layer times by. `wasm32-unknown-unknown`
//! has none of its own: there the host gives one ([`give`]), and until it
//! does nothing is timed or dated.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Whether this machine has a clock of its own.
const OWN: bool = !cfg!(all(target_arch = "wasm32", target_os = "unknown"));

/// A moment on the machine's clock, from the clock's first reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Moment(Duration);

impl Moment {
    /// How long since this moment.
    pub fn elapsed(self) -> Duration {
        now().map_or(Duration::ZERO, |now| now.since(self))
    }

    /// How long from `earlier` to this moment; zero when it is later.
    pub fn since(self, earlier: Moment) -> Duration {
        self.0.saturating_sub(earlier.0)
    }
}

/// The clock a host without one of its own gives: seconds, rising.
static GIVEN: OnceLock<fn() -> f64> = OnceLock::new();
/// Where this machine's own clock starts.
static START: OnceLock<Instant> = OnceLock::new();

/// Gives the clock for a machine without one (a page's
/// `performance.now()`), in seconds. Once: a later clock is not taken.
pub fn give(seconds: fn() -> f64) {
    let _ = GIVEN.set(seconds);
}

/// The moment now; `None` where there is no clock.
pub fn now() -> Option<Moment> {
    if OWN {
        return Some(Moment(START.get_or_init(Instant::now).elapsed()));
    }
    let seconds = GIVEN.get()?();
    (seconds.is_finite() && seconds >= 0.0).then(|| Moment(Duration::from_secs_f64(seconds)))
}

/// How long since `started`; zero where there is no clock.
pub fn since(started: Option<Moment>) -> Duration {
    started.map(Moment::elapsed).unwrap_or_default()
}

/// Seconds since the Unix epoch; `None` where the machine keeps no date.
pub fn wall() -> Option<u64> {
    if !OWN {
        return None;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moment_measures_forward_and_never_below_zero() {
        let first = now().expect("this machine has a clock");
        let later = now().unwrap();
        assert!(later >= first);
        assert_eq!(first.since(later), Duration::ZERO);
        assert_eq!(since(None), Duration::ZERO);
    }
}
