//! A gamepad as a keyboard: buttons and the left stick press the
//! keyboard's key codes.
//!
//! ```text
//! D-pad, left stick   arrows 37 38 39 40 (left, up, right, down)
//! South (A / Cross)   Z  (90)
//! East  (B / Circle)  X  (88)
//! Start               Enter (13)
//! Select              Escape (27)
//! ```
//!
//! Analog values are in the context: `stick(0)` left, `stick(1)` right,
//! `stick(2)` the triggers; pad buttons are `button(8)` upwards (south,
//! east, west, north, shoulders, thumbs, start, select). The device
//! names the pad's controls; this module gives them meaning.

use orla_kernel::{InputState, pad};

/// A pad's button, by position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadButton {
    /// Bottom face button (A, Cross).
    South,
    /// Right face button (B, Circle).
    East,
    /// Left face button (X, Square).
    West,
    /// Top face button (Y, Triangle).
    North,
    /// Left shoulder.
    LeftShoulder,
    /// Right shoulder.
    RightShoulder,
    /// Left stick pressed.
    LeftThumb,
    /// Right stick pressed.
    RightThumb,
    /// Start.
    Start,
    /// Select.
    Select,
    /// D-pad left.
    Left,
    /// D-pad up.
    Up,
    /// D-pad right.
    Right,
    /// D-pad down.
    Down,
}

/// A pad's analog axis; a stick's up and right are positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadAxis {
    /// Left stick, horizontal.
    LeftX,
    /// Left stick, vertical.
    LeftY,
    /// Right stick, horizontal.
    RightX,
    /// Right stick, vertical.
    RightY,
    /// Left trigger, 0 released to 1 pulled.
    LeftTrigger,
    /// Right trigger.
    RightTrigger,
}

/// The `pad` button number a pad button sets, if any.
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

/// Removes the stick's dead zone and rescales the rest to 0..1.
fn deadened(value: f32) -> f32 {
    let dead = 0.12;
    if value.abs() <= dead {
        0.0
    } else {
        ((value.abs() - dead) / (1.0 - dead)).min(1.0) * value.signum()
    }
}

/// Pad state beyond the input: which arrows the left stick holds.
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
        let held = deadened(value);
        match axis {
            PadAxis::LeftX => input.sticks[0][0] = held,
            PadAxis::LeftY => input.sticks[0][1] = held,
            PadAxis::RightX => input.sticks[1][0] = held,
            PadAxis::RightY => input.sticks[1][1] = held,
            PadAxis::LeftTrigger => input.sticks[2][0] = held.abs(),
            PadAxis::RightTrigger => input.sticks[2][1] = held.abs(),
        }
        // Hysteresis: an arrow presses past 0.5 and releases below 0.35.
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

    /// The stick presses arrows with hysteresis; D-pad and face buttons
    /// press keys and buttons both.
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
