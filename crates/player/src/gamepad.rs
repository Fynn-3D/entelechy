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
//! still plays, since the stick presses them too.

use gilrs::{Axis, Button, EventType, Gilrs};
use orla_runtime::InputState;

/// The connected pads, polled once per frame.
pub struct Gamepad {
    gilrs: Gilrs,
    /// Which arrow keys the stick currently holds, so a stick that drifts
    /// back releases them.
    stick_keys: [bool; 4],
}

/// The button number a pad button holds, if any: `orla_abi::pad`.
pub fn button_for(button: Button) -> Option<i32> {
    use orla_runtime::pad;
    Some(match button {
        Button::South => pad::SOUTH,
        Button::East => pad::EAST,
        Button::West => pad::WEST,
        Button::North => pad::NORTH,
        Button::LeftTrigger => pad::LEFT_SHOULDER,
        Button::RightTrigger => pad::RIGHT_SHOULDER,
        Button::LeftThumb => pad::LEFT_THUMB,
        Button::RightThumb => pad::RIGHT_THUMB,
        Button::Start => pad::START,
        Button::Select => pad::SELECT,
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

/// The key code a button maps to, if any.
pub fn key_for(button: Button) -> Option<i32> {
    Some(match button {
        Button::DPadLeft => 37,
        Button::DPadUp => 38,
        Button::DPadRight => 39,
        Button::DPadDown => 40,
        Button::South => 90,
        Button::East => 88,
        Button::Start => 13,
        Button::Select => 27,
        _ => return None,
    })
}

impl Gamepad {
    /// Opens the gamepad system; `None` when it is unavailable, which is
    /// not an error, only a keyboard.
    pub fn open() -> Option<Self> {
        let gilrs = Gilrs::new().ok()?;
        Some(Self {
            gilrs,
            stick_keys: [false; 4],
        })
    }

    /// Whether any pad is connected.
    pub fn connected(&self) -> bool {
        self.gilrs.gamepads().next().is_some()
    }

    /// Drains pending events into the input state.
    pub fn poll(&mut self, input: &mut InputState) {
        while let Some(event) = self.gilrs.next_event() {
            match event.event {
                EventType::ButtonPressed(button, _) => {
                    if let Some(code) = key_for(button) {
                        input.set_key(code, true);
                    }
                    if let Some(code) = button_for(button) {
                        input.set_button(code, true);
                    }
                }
                EventType::ButtonReleased(button, _) => {
                    if let Some(code) = key_for(button) {
                        input.set_key(code, false);
                    }
                    if let Some(code) = button_for(button) {
                        input.set_button(code, false);
                    }
                }
                EventType::AxisChanged(axis, value, _) => {
                    // The analog value as it is, less its dead middle.
                    let held = deadened(value);
                    match axis {
                        Axis::LeftStickX => input.sticks[0][0] = held,
                        Axis::LeftStickY => input.sticks[0][1] = held,
                        Axis::RightStickX => input.sticks[1][0] = held,
                        Axis::RightStickY => input.sticks[1][1] = held,
                        Axis::LeftZ => input.sticks[2][0] = held.abs(),
                        Axis::RightZ => input.sticks[2][1] = held.abs(),
                        _ => {}
                    }
                    // The stick presses an arrow past a threshold and
                    // releases it well inside, so it never chatters.
                    let (negative, positive) = match axis {
                        Axis::LeftStickX => (0, 2),
                        Axis::LeftStickY => (3, 1),
                        _ => continue,
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
                _ => {}
            }
        }
    }
}
