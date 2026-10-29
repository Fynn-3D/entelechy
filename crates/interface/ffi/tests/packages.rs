//! A scene opened through the C ABI finds a package by its name where
//! every front finds it: `ENTELECHY_PACKAGES`. Its own process, as it sets
//! the environment.

use std::ffi::{CStr, CString};

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

#[test]
fn a_scene_opened_through_the_abi_finds_a_package_by_name() {
    let dir = std::env::temp_dir().join(format!("entelechy-ffi-packages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let package = dir.join("shelf").join("tint.entpkg");
    std::fs::create_dir_all(package.join("kernels")).unwrap();
    std::fs::write(
        package.join("package.esn"),
        "package tint@1 { language = 2; ir = 1; }\n",
    )
    .unwrap();
    std::fs::write(
        package.join("kernels").join("tint.ent"),
        "kernel tint.red@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }\n",
    )
    .unwrap();
    let scene = dir.join("scene.esn");
    std::fs::write(
        &scene,
        "use \"tint.entpkg\";\nnode red: tint.red@1 { domain = vec2(2, 2); }\noutput result <- red.result;\n",
    )
    .unwrap();
    // The only test in this process: nothing else reads the environment.
    unsafe { std::env::set_var("ENTELECHY_PACKAGES", dir.join("shelf")) };
    let session = unsafe {
        entelechy_ffi::entelechy_open(
            c(scene.to_str().unwrap()).as_ptr(),
            c("cpu").as_ptr(),
            std::ptr::null(),
        )
    };
    assert!(!session.is_null(), "{}", unsafe {
        CStr::from_ptr(entelechy_ffi::entelechy_open_error()).to_string_lossy()
    });
    unsafe { entelechy_ffi::entelechy_close(session) };
}
