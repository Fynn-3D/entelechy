//! Characters typed, handed over the C ABI: a scene reads them with
//! `typed()` on the frame after, as it does in the Player.

use std::ffi::{CStr, CString};

use entelechy_ffi::EntelechyField;

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

const SCENE: &str = "node read: kernel {
    requires { field.write; input.read; }
    output result: Rows<Int>;
    domain = 4;
    compute { return typed(element()); }
}
output result <- read.result;
";

#[test]
fn typed_characters_reach_the_scene() {
    let dir = std::env::temp_dir().join(format!("entelechy-ffi-typed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scene.esn");
    std::fs::write(&path, SCENE).unwrap();
    let scene = c(path.to_str().unwrap());
    let session = unsafe {
        entelechy_ffi::entelechy_open(scene.as_ptr(), c("cpu").as_ptr(), std::ptr::null())
    };
    assert!(!session.is_null(), "{}", unsafe {
        CStr::from_ptr(entelechy_ffi::entelechy_open_error()).to_string_lossy()
    });
    // What is typed is not a key: the space of a command line, a letter
    // of another script.
    assert_eq!(
        unsafe { entelechy_ffi::entelechy_play_typed(session, c("a 月").as_ptr()) },
        0
    );
    assert_eq!(
        unsafe { entelechy_ffi::entelechy_play_typed(session, std::ptr::null()) },
        -1
    );
    assert_eq!(
        unsafe { entelechy_ffi::entelechy_play_frame(session, 64.0, 64.0, 1.0, 0.0) },
        0
    );
    let mut field = EntelechyField {
        words: std::ptr::null(),
        width: 0,
        height: 0,
        depth: 0,
        count: 0,
        stride: 0,
        shape: 0,
        pixels: 0,
        schema: std::ptr::null(),
    };
    assert_eq!(
        unsafe { entelechy_ffi::entelechy_output(session, c("result").as_ptr(), &mut field) },
        0
    );
    let words = unsafe {
        std::slice::from_raw_parts(field.words, field.count as usize * field.stride as usize)
    };
    assert_eq!(words, ['a' as u32, ' ' as u32, '月' as u32, 0]);
    unsafe { entelechy_ffi::entelechy_close(session) };
    let _ = std::fs::remove_dir_all(&dir);
}
