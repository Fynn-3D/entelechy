//! The system's gamepads through gilrs, each button and axis handed to
//! the play by name. Their meaning is `orla_session::play::pad`'s.

use gilrs::{Axis, Button, EventType, Gilrs};
use orla_session::play::{PadAxis, PadButton, Play};

/// The connected pads, polled once per frame.
pub struct Gamepad {
    gilrs: Gilrs,
}

/// A gilrs button by the play's name for it, if it has one.
pub fn button(button: Button) -> Option<PadButton> {
    Some(match button {
        Button::South => PadButton::South,
        Button::East => PadButton::East,
        Button::West => PadButton::West,
        Button::North => PadButton::North,
        Button::LeftTrigger => PadButton::LeftShoulder,
        Button::RightTrigger => PadButton::RightShoulder,
        Button::LeftThumb => PadButton::LeftThumb,
        Button::RightThumb => PadButton::RightThumb,
        Button::Start => PadButton::Start,
        Button::Select => PadButton::Select,
        Button::DPadLeft => PadButton::Left,
        Button::DPadUp => PadButton::Up,
        Button::DPadRight => PadButton::Right,
        Button::DPadDown => PadButton::Down,
        _ => return None,
    })
}

/// A gilrs axis by the play's name for it; up is positive in both.
pub fn axis(axis: Axis) -> Option<PadAxis> {
    Some(match axis {
        Axis::LeftStickX => PadAxis::LeftX,
        Axis::LeftStickY => PadAxis::LeftY,
        Axis::RightStickX => PadAxis::RightX,
        Axis::RightStickY => PadAxis::RightY,
        Axis::LeftZ => PadAxis::LeftTrigger,
        Axis::RightZ => PadAxis::RightTrigger,
        _ => return None,
    })
}

impl Gamepad {
    /// Opens the gamepad system; `None` when it is unavailable.
    pub fn open() -> Option<Self> {
        let gilrs = Gilrs::new().ok()?;
        Some(Self { gilrs })
    }

    /// Whether any pad is connected.
    pub fn connected(&self) -> bool {
        self.gilrs.gamepads().next().is_some()
    }

    /// Hands the pending events to the play.
    pub fn poll(&mut self, play: &mut Play) {
        while let Some(event) = self.gilrs.next_event() {
            match event.event {
                EventType::ButtonPressed(pressed, _) => {
                    if let Some(pressed) = button(pressed) {
                        play.pad_button(pressed, true);
                    }
                }
                EventType::ButtonReleased(released, _) => {
                    if let Some(released) = button(released) {
                        play.pad_button(released, false);
                    }
                }
                EventType::AxisChanged(moved, value, _) => {
                    if let Some(moved) = axis(moved) {
                        play.pad_axis(moved, value);
                    }
                }
                _ => {}
            }
        }
    }
}
