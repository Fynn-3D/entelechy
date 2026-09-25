//! A gamepad as another keyboard. Buttons and the left stick land in the
//! same key codes the keyboard uses, so a scene written for arrows, Z, X,
//! and Enter plays on a pad without knowing one exists:
//!
//! ```text
//! D-pad, left stick   arrows 37 38 39 40 (left, up, right, down)
//! South (A / Cross)   Z  (90)
//! East  (B / Circle)  X  (88)
//! Start               Enter (13)
//! Select              Escape (27)
//! ```
//!
//! The analog values are in the context beside the pointer: `stick(0)` is
//! the left stick, `stick(1)` the right, `stick(2)` the two triggers, and
//! the pad's buttons are `button(8)` upwards (south, east, west, north,
//! the shoulders, the thumbs, start, select). A scene written for arrows
//! still plays, since the stick presses them too. What a pad is called
//! on each machine (gilrs's names, a browser's standard mapping) is the
//! device's to say; what it means is here.

use orla_runtime::{InputState, pad};

/// A pad's button, by where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadButton {
    /// The lowest of the four face buttons (A, Cross).
    South,
    /// The right one (B, Circle).
    East,
    /// The left one (X, Square).
    West,
    /// The top one (Y, Triangle).
    North,
    /// The shoulder button on the left.
    LeftShoulder,
    /// The shoulder button on the right.
    RightShoulder,
    /// The left stick, pressed.
    LeftThumb,
    /// The right stick, pressed.
    RightThumb,
    /// Start.
    Start,
    /// Select.
    Select,
    /// The D-pad's left.
    Left,
    /// The D-pad's up.
    Up,
    /// The D-pad's right.
    Right,
    /// The D-pad's down.
    Down,
}

/// A pad's analog axis. A stick's up is positive, as a stick's axes
/// are read on every machine that says which way is up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadAxis {
    /// The left stick across, right positive.
    LeftX,
    /// The left stick up and down, up positive.
    LeftY,
    /// The right stick across.
    RightX,
    /// The right stick up and down.
    RightY,
    /// The left trigger, 0 released to 1 pulled.
    LeftTrigger,
    /// The right trigger.
    RightTrigger,
}

/// The button number a pad button holds, if any: `orla_abi::pad`.
fn button_for(button: PadButton) -> Option<i32> {
    Some(match button {
        PadButton::South => pad::SOUTH,
        PadButton::East => pad::EAST,
        PadButton::West => pad::WEST,
        PadButton::North => pad::NORTH,
        PadButton::LeftShoulder => pad::LEFT_SHOULDER,
        PadButton::RightShoulder => pad::RIGHT_SHOULDER,
        PadButton::LeftThumb => pad::LEFT_THUMB,
        PadButton::RightThumb => pad::RIGHT_THUMB,
        PadButton::Start => pad::START,
        PadButton::Select => pad::SELECT,
        _ => return None,
    })
}

/// The key code a button presses, if any.
fn key_for(button: PadButton) -> Option<i32> {
    Some(match button {
        PadButton::Left => 37,
        PadButton::Up => 38,
        PadButton::Right => 39,
        PadButton::Down => 40,
        PadButton::South => 90,
        PadButton::East => 88,
        PadButton::Start => 13,
        PadButton::Select => 27,
        _ => return None,
    })
}

/// A stick's rest is never quite zero, so a little of it is nothing.
fn deadened(value: f32) -> f32 {
    let dead = 0.12;
    if value.abs() <= dead {
        0.0
    } else {
        ((value.abs() - dead) / (1.0 - dead)).min(1.0) * value.signum()
    }
}

/// The pads' state beyond what the input holds: which arrows the left
/// stick holds down, so a stick that drifts back releases them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pad {
    stick_keys: [bool; 4],
}

impl Pad {
    /// A button pressed or released.
    pub fn button(&mut self, input: &mut InputState, button: PadButton, down: bool) {
        if let Some(code) = key_for(button) {
            input.set_key(code, down);
        }
        if let Some(code) = button_for(button) {
            input.set_button(code, down);
        }
    }

    /// An axis moved to `value`.
    pub fn axis(&mut self, input: &mut InputState, axis: PadAxis, value: f32) {
        // The analog value as it is, less its dead middle.
        let held = deadened(value);
        match axis {
            PadAxis::LeftX => input.sticks[0][0] = held,
            PadAxis::LeftY => input.sticks[0][1] = held,
            PadAxis::RightX => input.sticks[1][0] = held,
            PadAxis::RightY => input.sticks[1][1] = held,
            PadAxis::LeftTrigger => input.sticks[2][0] = held.abs(),
            PadAxis::RightTrigger => input.sticks[2][1] = held.abs(),
        }
        // The stick presses an arrow past a threshold and releases it
        // well inside, so it never chatters.
        let (negative, positive) = match axis {
            PadAxis::LeftX => (0, 2),
            PadAxis::LeftY => (3, 1),
            _ => return,
        };
        for (slot, held) in [(negative, -value), (positive, value)] {
            let was = self.stick_keys[slot];
            let now = if was { held > 0.35 } else { held > 0.5 };
            if now != was {
                self.stick_keys[slot] = now;
                input.set_key(37 + slot as i32, now);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stick presses the arrow it leans toward past a half, holds it
    /// down to a third, and the D-pad and the face buttons press keys and
    /// buttons both.
    #[test]
    fn a_pad_is_another_keyboard() {
        let mut input = InputState::default();
        let mut pad = Pad::default();
        pad.axis(&mut input, PadAxis::LeftX, 0.6);
        assert!(input.key(39), "right past a half");
        pad.axis(&mut input, PadAxis::LeftX, 0.4);
        assert!(input.key(39), "held inside the release");
        pad.axis(&mut input, PadAxis::LeftX, 0.3);
        assert!(!input.key(39), "released well inside");
        pad.axis(&mut input, PadAxis::LeftY, 0.7);
        assert!(input.key(38), "up is up");
        assert!(input.sticks[0][1] > 0.6);
        pad.axis(&mut input, PadAxis::RightX, 0.05);
        assert_eq!(input.sticks[1][0], 0.0, "the dead middle");
        pad.button(&mut input, PadButton::South, true);
        assert!(input.key(90));
        pad.button(&mut input, PadButton::Down, true);
        assert!(input.key(40));
    }
}
