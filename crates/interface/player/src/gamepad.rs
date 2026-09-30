//! The system's gamepads through gilrs, each button and axis handed to
//! the play by name, and how many are connected. Their meaning is
//! `orla_session::play::pad`'s.

use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};
use orla_session::play::{PadAxis, PadButton, Play};

/// The connected pads, polled once per frame.
pub struct Gamepad {
    gilrs: Gilrs,
}

/// The stick axes the SDL game controller database gilrs carries marks
/// inverted (`~`) for each pad on this system, by its GUID. gilrs reads
/// the mark and does not apply it; the adapter applies it.
#[cfg(target_os = "macos")]
const INVERTED: &[(&str, &[Axis])] = &[
    (
        "030000008a3500000102000000010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000008a3500000202000000010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000008a3500000302000000010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    ("03000000260900008888000088020000", &[Axis::RightStickY]),
    (
        "030000006d04000019c2000000020000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000006d04000018c2000000010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    ("030000008f0e00001330000011010000", &[Axis::RightStickX]),
    ("030000005e0400002800000002010000", &[Axis::RightStickY]),
    (
        "030000007e0500006920000001010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000007e0500007320000001010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "03000000110100002014000000000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "03000000110100002014000001000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "03000000381000002014000001000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "05000000484944204465766963650000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "050000004e696d6275732b0000000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "03000000110100001714000000000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "03000000110100001714000020010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000005e0400008e02000010010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
];
#[cfg(target_os = "linux")]
const INVERTED: &[(&str, &[Axis])] = &[
    ("03000000260900008888000000010000", &[Axis::RightStickY]),
    ("030000008f0e00001330000010010000", &[Axis::RightStickX]),
    ("030000005e0400002800000000010000", &[Axis::RightStickY]),
    (
        "03000000790000004618000010010000",
        &[Axis::RightStickX, Axis::RightStickY],
    ),
    (
        "030000007e0500006920000011010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    ("050000007e0500000620000001800000", &[Axis::LeftStickY]),
    ("050000007e0500000720000001800000", &[Axis::LeftStickX]),
    (
        "030000007e0500007320000011010000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    ("03000000250900000017000010010000", &[Axis::RightStickX]),
];
#[cfg(target_os = "windows")]
const INVERTED: &[(&str, &[Axis])] = &[
    ("03000000260900008888000000000000", &[Axis::RightStickY]),
    ("030000008f0e00001330000000000000", &[Axis::RightStickX]),
    ("030000005e0400002800000000000000", &[Axis::RightStickY]),
    (
        "030000007e0500006920000000000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
    (
        "030000007e0500007320000000000000",
        &[Axis::LeftStickY, Axis::RightStickY],
    ),
];
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
const INVERTED: &[(&str, &[Axis])] = &[];

/// Whether the database marks `axis` inverted for the pad with `guid`.
pub fn inverted(guid: &str, axis: Axis) -> bool {
    INVERTED
        .iter()
        .any(|(id, axes)| *id == guid && axes.contains(&axis))
}

/// A pad's GUID as the database writes it: its sixteen bytes in hex.
fn guid(bytes: [u8; 16]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

/// A trigger a pad reports as an analog button, by the play's axis for it.
pub fn trigger(button: Button) -> Option<PadAxis> {
    Some(match button {
        Button::LeftTrigger2 => PadAxis::LeftTrigger,
        Button::RightTrigger2 => PadAxis::RightTrigger,
        _ => return None,
    })
}

impl Gamepad {
    /// Opens the gamepad system; `None` when it is unavailable.
    pub fn open() -> Option<Self> {
        let gilrs = Gilrs::new().ok()?;
        Some(Self { gilrs })
    }

    /// How many pads are connected.
    pub fn connected(&self) -> usize {
        self.gilrs.gamepads().count()
    }

    /// An axis's value from the pad `id`, the database's inversion applied.
    fn value(&self, id: GamepadId, moved: Axis, value: f32) -> f32 {
        let pad = self.gilrs.gamepad(id);
        if inverted(&guid(pad.uuid()), moved) {
            -value
        } else {
            value
        }
    }

    /// Hands the pending events to the play, and how many pads are
    /// connected.
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
                EventType::ButtonChanged(changed, value, _) => {
                    if let Some(changed) = trigger(changed) {
                        play.pad_axis(changed, value);
                    }
                }
                EventType::AxisChanged(moved, value, _) => {
                    let value = self.value(event.id, moved, value);
                    if let Some(moved) = axis(moved) {
                        play.pad_axis(moved, value);
                    }
                }
                _ => {}
            }
        }
        play.pads_connected(self.connected() as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pad the database marks inverted has those axes turned, and no
    /// other; a trigger reported as a button is the play's trigger axis.
    #[test]
    fn marked_axes_are_turned_and_triggers_are_axes() {
        let marked = INVERTED.first().map(|(id, axes)| (*id, axes[0]));
        if let Some((id, axis)) = marked {
            assert!(inverted(id, axis));
            assert!(!inverted("00000000000000000000000000000000", axis));
        }
        #[cfg(target_os = "macos")]
        {
            let xbox_360 = "030000005e0400008e02000010010000";
            assert!(inverted(xbox_360, Axis::LeftStickY) && inverted(xbox_360, Axis::RightStickY));
            assert!(!inverted(xbox_360, Axis::LeftStickX));
        }
        assert_eq!(
            guid([3, 0, 0, 0, 0x5e, 4, 0, 0, 0x8e, 2, 0, 0, 0x10, 1, 0, 0]),
            "030000005e0400008e02000010010000"
        );
        assert_eq!(trigger(Button::LeftTrigger2), Some(PadAxis::LeftTrigger));
        assert_eq!(trigger(Button::South), None);
    }
}
