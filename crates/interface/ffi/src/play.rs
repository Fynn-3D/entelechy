//! The shared play (`entelechy_session::play`) through the C ABI. A host (a
//! page, say) hands in keys, pointer, wheel and pads, asks for a frame,
//! and shows the picture and plays the sound on its own devices. Key
//! meanings, pad mapping, the frame data line and replays belong to the
//! play.

use std::ffi::{CString, c_char, c_int};
use std::path::Path;

use entelechy_session::play::{Frame, PadAxis, PadButton, View, hud};

use crate::{
    EntelechyField, EntelechyFiles, EntelechySession, OPEN_ERROR, Opening, c_string, clock, enter,
    guarded, opened, text,
};

/// A pad button by its name in the header.
fn pad_button(name: &str) -> Option<PadButton> {
    Some(match name {
        "south" => PadButton::South,
        "east" => PadButton::East,
        "west" => PadButton::West,
        "north" => PadButton::North,
        "left_shoulder" => PadButton::LeftShoulder,
        "right_shoulder" => PadButton::RightShoulder,
        "left_thumb" => PadButton::LeftThumb,
        "right_thumb" => PadButton::RightThumb,
        "start" => PadButton::Start,
        "select" => PadButton::Select,
        "left" => PadButton::Left,
        "up" => PadButton::Up,
        "right" => PadButton::Right,
        "down" => PadButton::Down,
        _ => return None,
    })
}

/// A pad axis by its name in the header.
fn pad_axis(name: &str) -> Option<PadAxis> {
    Some(match name {
        "left_x" => PadAxis::LeftX,
        "left_y" => PadAxis::LeftY,
        "right_x" => PadAxis::RightX,
        "right_y" => PadAxis::RightY,
        "left_trigger" => PadAxis::LeftTrigger,
        "right_trigger" => PadAxis::RightTrigger,
        _ => return None,
    })
}

/// Opens the scene and library the replay at `replay` (a path among
/// `files`) names, and plays it: its inputs drive the frames until it
/// ends.
///
/// # Safety
/// `files` is a pointer `entelechy_files` returned, not freed yet; `replay`
/// and `backend` are NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_open_replay(
    files: *mut EntelechyFiles,
    replay: *const c_char,
    backend: *const c_char,
) -> *mut EntelechySession {
    let (Some(files), Some(replay)) = (unsafe { files.as_ref() }, unsafe { text(replay) }) else {
        OPEN_ERROR.with(|slot| {
            *slot.borrow_mut() = c_string("entelechy_open_replay needs files and a replay");
        });
        return std::ptr::null_mut();
    };
    let store: std::sync::Arc<dyn entelechy_session::Store> = files.files.clone();
    let backend = unsafe { text(backend) }.unwrap_or("jit");
    opened(Some(store), Opening::Replay(Path::new(replay)), backend)
}

/// A key by name (`a`, `ArrowLeft`, `F5`), pressed or released; 0 taken,
/// -1 when the session or the name cannot be read.
///
/// # Safety
/// `session` is a live session; `name` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_key(
    session: *mut EntelechySession,
    name: *const c_char,
    down: c_int,
) -> c_int {
    let (Some(mut s), Some(name)) = (unsafe { enter(session) }, unsafe { text(name) }) else {
        return -1;
    };
    let name = name.to_owned();
    guarded(&mut s, |s| {
        s.play.key(&mut s.session, &name, down != 0);
        0
    })
}

/// The pointer at `(x, y)` in the canvas's pixels.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_pointer(session: *mut EntelechySession, x: f32, y: f32) {
    if let Some(mut s) = unsafe { enter(session) } {
        s.play.pointer(x, y);
    }
}

/// A pointer button, 0 left, 1 right, 2 middle.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_button(
    session: *mut EntelechySession,
    which: c_int,
    down: c_int,
) {
    if let (Some(mut s), Ok(which)) = (unsafe { enter(session) }, u8::try_from(which)) {
        s.play.button(which, down != 0);
    }
}

/// The wheel turned by `(x, y)` lines, y positive away from the user.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_wheel(session: *mut EntelechySession, x: f32, y: f32) {
    if let Some(mut s) = unsafe { enter(session) } {
        s.play.wheel(x, y);
    }
}

/// A pad button by name, pressed or released; -1 for an unknown name.
///
/// # Safety
/// `session` is a live session; `name` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_pad(
    session: *mut EntelechySession,
    name: *const c_char,
    down: c_int,
) -> c_int {
    let (Some(mut s), Some(button)) = (
        unsafe { enter(session) },
        unsafe { text(name) }.and_then(pad_button),
    ) else {
        return -1;
    };
    s.play.pad_button(button, down != 0);
    0
}

/// A pad axis by name at `value`, a stick's up positive; -1 for an
/// unknown name.
///
/// # Safety
/// `session` is a live session; `name` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_axis(
    session: *mut EntelechySession,
    name: *const c_char,
    value: f32,
) -> c_int {
    let (Some(mut s), Some(axis)) = (
        unsafe { enter(session) },
        unsafe { text(name) }.and_then(pad_axis),
    ) else {
        return -1;
    };
    s.play.pad_axis(axis, value);
    0
}

/// How many pads are connected, what `pads()` reads; a negative count
/// is none.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_pads(session: *mut EntelechySession, count: c_int) {
    if let Some(mut s) = unsafe { enter(session) } {
        s.play.pads_connected(count.max(0) as u32);
    }
}

/// One frame played on a `width` by `height` pixel window at `scale`, the
/// sound device at `rate` (zero without one). 0 shown, 1 waiting, -1
/// failed.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_frame(
    session: *mut EntelechySession,
    width: f32,
    height: f32,
    scale: f32,
    rate: f32,
) -> c_int {
    let Some(mut s) = (unsafe { enter(session) }) else {
        return -1;
    };
    let view = View {
        window: [width, height],
        scale,
        rate,
    };
    guarded(&mut s, |s| {
        s.clear();
        match s.play.frame(&mut s.session, view) {
            Frame::Wait => 1,
            Frame::Shown(shown) => {
                s.report = c_string(
                    &entelechy_protocol::report_json(&s.session, &shown.report).to_string(),
                );
                s.shown = Some((shown.picture, shown.sound));
                0
            }
            Frame::Failed => {
                let status = s.play.status().to_owned();
                s.fail(status)
            }
        }
    })
}

/// Draws the frame's picture from the GPU on the page's canvas with
/// element id `canvas`, the frame data line over it when F10 shows it.
///
/// # Safety
/// `session` is a live session; `canvas` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_present(
    session: *mut EntelechySession,
    canvas: *const c_char,
) -> c_int {
    let (Some(mut s), Some(canvas)) = (unsafe { enter(session) }, unsafe { text(canvas) }) else {
        return -1;
    };
    let canvas = canvas.to_owned();
    guarded(&mut s, |s| {
        s.clear();
        let Some((picture, _)) = s.shown else {
            return 1;
        };
        let started = clock();
        let panel = s.play.panel().cloned();
        let overlay = panel.as_ref().map(|panel| entelechy_gpu::present::Overlay {
            pixels: &panel.pixels,
            width: panel.width,
            height: panel.height,
        });
        match s.present_on(&canvas, picture, overlay) {
            Ok(Some((placement, size))) => {
                let path = if placement.is_some() {
                    "presented on the device"
                } else {
                    "window not ready"
                };
                s.play
                    .placed(placement, size, path, false, clock() - started);
                0
            }
            Ok(None) => 1,
            Err(message) => s.fail(message),
        }
    })
}

/// The frame's picture drawn on the CPU into a `width` by `height` window,
/// letterboxed, with the frame data line: `0x00RRGGBB`, row after row,
/// valid until the next call on the session. NULL with `entelechy_error()` set
/// on failure.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_pixels(
    session: *mut EntelechySession,
    width: u32,
    height: u32,
) -> *const u32 {
    let Some(mut s) = (unsafe { enter(session) }) else {
        return std::ptr::null();
    };
    let drawn = guarded(&mut s, |s| {
        s.clear();
        let started = clock();
        // The picture read back; when it cannot be, the last is shown.
        let mut downloaded = false;
        if let Some((picture, _)) = s.shown
            && let Ok(image) = s.session.picture(picture)
        {
            s.picture = (image.pixels().to_vec(), (image.width(), image.height()));
            downloaded = true;
        }
        let (pixels, (shown_width, shown_height)) = &s.picture;
        s.pixels.clear();
        s.pixels.resize(width as usize * height as usize, 0);
        let placement = entelechy_session::play::present(
            pixels,
            *shown_width,
            *shown_height,
            width,
            height,
            &mut s.pixels,
        );
        if let Some(panel) = s.play.panel() {
            hud::stamp(panel, &mut s.pixels, width, height);
        }
        let size = (*shown_width, *shown_height);
        s.play.placed(
            Some(placement),
            size,
            "downloaded and blitted",
            downloaded,
            clock() - started,
        );
        0
    });
    if drawn == 0 {
        s.pixels.as_ptr()
    } else {
        std::ptr::null()
    }
}

/// The frame's sound block (one or two floats per sample frame), read in
/// place. 0 and `field` filled when there is one, 1 when none, -1 with
/// `entelechy_error()` set on failure.
///
/// # Safety
/// `session` is a live session; `field` points to an `EntelechyField` to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_sound(
    session: *mut EntelechySession,
    field: *mut EntelechyField,
) -> c_int {
    let (Some(mut s), Some(field)) = (unsafe { enter(session) }, unsafe { field.as_mut() }) else {
        return -1;
    };
    guarded(&mut s, |s| {
        s.clear();
        let Some((_, Some(sound))) = s.shown else {
            return 1;
        };
        match s.field(sound, field) {
            Ok(()) => 0,
            Err(message) => s.fail(message),
        }
    })
}

/// Reports the sound device open, as the host describes it, for the frame
/// data line.
///
/// # Safety
/// `session` is a live session; `description` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_sound_opened(
    session: *mut EntelechySession,
    description: *const c_char,
) {
    if let (Some(mut s), Some(description)) =
        (unsafe { enter(session) }, unsafe { text(description) })
    {
        s.play.sound_opened(description);
    }
}

/// The play's messages since the last call, one per line (events,
/// failures, the preview, a replay's outcome); "" when none.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_said(session: *mut EntelechySession) -> *const c_char {
    let Some(mut s) = (unsafe { enter(session) }) else {
        return crate::EMPTY.as_ptr();
    };
    s.said = CString::new(s.play.said().join("\n").replace('\0', " ")).unwrap_or_default();
    s.said.as_ptr()
}

/// Whether a replay is playing: 1 or 0.
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_replaying(session: *const EntelechySession) -> c_int {
    unsafe { enter(session) }.is_some_and(|s| s.play.replaying()) as c_int
}

/// The Kernels still waiting their turn to prepare (a page prepares a few
/// each frame, and a frame waits until the scene's are all prepared): what
/// a page shows while it loads. Zero once they are, or where they prepare
/// on threads.
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_preparing(session: *const EntelechySession) -> c_int {
    unsafe { enter(session) }.map_or(0, |s| {
        c_int::try_from(s.session.preparing_left()).unwrap_or(c_int::MAX)
    })
}

/// A turn of loading: one sync that prepares a few of the Kernels waiting
/// (a page prepares them in turns), its messages kept for
/// `entelechy_play_said`. The Kernels still waiting, as
/// `entelechy_play_preparing` counts them; -1 for a session that is not.
///
/// # Safety
/// `session` is a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_play_turn(session: *mut EntelechySession) -> c_int {
    let Some(mut s) = (unsafe { enter(session) }) else {
        return -1;
    };
    for event in s.session.sync_a_turn() {
        s.play.say(event.to_string());
    }
    c_int::try_from(s.session.preparing_left()).unwrap_or(c_int::MAX)
}
