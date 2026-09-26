//! The C ABI driven from Rust as a C caller would: open, evaluate, read
//! a Field in place, a protocol request, a node's rows, the errors.

use std::ffi::{CStr, CString};
use std::path::Path;

use orla_ffi::{OrlaField, OrlaInput};

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

fn example(name: &str) -> CString {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples")
        .join(name);
    c(path.to_str().unwrap())
}

unsafe fn str_of(pointer: *const std::ffi::c_char) -> String {
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn pixel(field: &OrlaField, x: u32, y: u32) -> [f32; 4] {
    assert_eq!(field.pixels, 1);
    let at = ((y * field.width + x) * field.stride) as usize;
    let words =
        unsafe { std::slice::from_raw_parts(field.words, (field.count * field.stride) as usize) };
    [
        f32::from_bits(words[at]),
        f32::from_bits(words[at + 1]),
        f32::from_bits(words[at + 2]),
        f32::from_bits(words[at + 3]),
    ]
}

fn near(a: [f32; 4], rgb: [f32; 3]) -> bool {
    (0..3).all(|k| (a[k] - rgb[k]).abs() <= 0.02)
}

#[test]
fn a_scene_opens_evaluates_and_its_fields_read_in_place() {
    assert_eq!(unsafe { str_of(orla_ffi::orla_version()) }, "0.1.0");
    let scene = example("text/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(scene.as_ptr(), c("cpu").as_ptr(), std::ptr::null()) };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    assert_eq!(
        unsafe { orla_ffi::orla_evaluate(session, std::ptr::null()) },
        0,
        "{}",
        unsafe { str_of(orla_ffi::orla_error(session)) }
    );
    let report = unsafe { str_of(orla_ffi::orla_report(session)) };
    assert!(report.contains("\"outputs\""), "{report}");
    // The picture as the replay pins it: 192 by 88, the title's O at (8, 6)
    // at double size, read from the Host's words.
    let mut field = OrlaField {
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
        unsafe { orla_ffi::orla_output(session, std::ptr::null(), &mut field) },
        0
    );
    assert_eq!(
        (
            field.width,
            field.height,
            field.stride,
            field.shape,
            field.count
        ),
        (192, 88, 4, 1, 192 * 88)
    );
    assert_eq!(unsafe { str_of(field.schema) }, "f,f,f,f");
    assert!(
        near(pixel(&field, 10, 6), [1.0, 0.85, 0.45]),
        "{:?}",
        pixel(&field, 10, 6)
    );
    assert!(
        near(pixel(&field, 12, 12), [0.09, 0.1, 0.13]),
        "{:?}",
        pixel(&field, 12, 12)
    );
    // The protocol through the same entry: a recoloured copy of the
    // background published over it changes the picture.
    let response = unsafe {
        str_of(orla_ffi::orla_request(
            session,
            c("{\"op\":\"kernel\",\"kernel\":\"scene.back@1\"}").as_ptr(),
        ))
    };
    let start = response.find("\"source\":\"").unwrap() + 10;
    let end = response[start..].find("\"}").unwrap() + start;
    let source = response[start..end]
        .replace("\\n", "\n")
        .replace("\\\"", "\"");
    assert!(source.starts_with("kernel scene.back@1 {"), "{source}");
    let recoloured = source.replace("vec4(0.09, 0.1, 0.13, 1.0)", "vec4(0.3, 0.1, 0.1, 1.0)");
    assert_ne!(recoloured, source);
    let publish = format!(
        "{{\"op\":\"publish\",\"expect\":\"any\",\"source\":{}}}",
        serde_json_string(&recoloured)
    );
    let response = unsafe { str_of(orla_ffi::orla_request(session, c(&publish).as_ptr())) };
    assert!(response.contains("\"ok\""), "{response}");
    let input = OrlaInput {
        time: 0.1,
        ..OrlaInput::default()
    };
    assert_eq!(unsafe { orla_ffi::orla_evaluate(session, &input) }, 0);
    assert_eq!(
        unsafe { orla_ffi::orla_output(session, c("result").as_ptr(), &mut field) },
        0
    );
    assert!(
        near(pixel(&field, 12, 12), [0.3, 0.1, 0.1]),
        "{:?}",
        pixel(&field, 12, 12)
    );
    assert!(near(pixel(&field, 10, 6), [1.0, 0.85, 0.45]));
    // A node's rows: the specimen's character codes, pinned and read as
    // ints.
    assert_eq!(
        unsafe { orla_ffi::orla_node(session, c("codes").as_ptr(), std::ptr::null(), &mut field) },
        -1
    );
    assert!(unsafe { str_of(orla_ffi::orla_error(session)) }.contains("pin it"));
    unsafe { orla_ffi::orla_request(session, c("{\"op\":\"pin\",\"node\":\"codes\"}").as_ptr()) };
    assert_eq!(unsafe { orla_ffi::orla_evaluate(session, &input) }, 0);
    assert_eq!(
        unsafe { orla_ffi::orla_node(session, c("codes").as_ptr(), std::ptr::null(), &mut field) },
        0
    );
    assert_eq!(
        (field.count, field.stride, field.shape, field.pixels),
        (95, 1, 0, 0)
    );
    let words = unsafe { std::slice::from_raw_parts(field.words, 95) };
    assert_eq!(words[0], 32);
    assert_eq!(words[94], 126);
    assert_eq!(unsafe { str_of(orla_ffi::orla_error(session)) }, "");
    // A wrong output and a bad request line are errors.
    assert_eq!(
        unsafe { orla_ffi::orla_output(session, c("nope").as_ptr(), &mut field) },
        -1
    );
    assert!(unsafe { str_of(orla_ffi::orla_error(session)) }.contains("nope"));
    let response = unsafe { str_of(orla_ffi::orla_request(session, c("not json").as_ptr())) };
    assert!(response.contains("\"error\""), "{response}");
    unsafe { orla_ffi::orla_close(session) };
}

#[test]
fn a_scene_that_does_not_exist_or_a_backend_that_does_not_is_refused() {
    let missing = example("nowhere/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(missing.as_ptr(), std::ptr::null(), std::ptr::null()) };
    assert!(session.is_null());
    let reason = unsafe { str_of(orla_ffi::orla_open_error()) };
    assert!(!reason.is_empty(), "a reason is given");
    let scene = example("text/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(scene.as_ptr(), c("abacus").as_ptr(), std::ptr::null()) };
    assert!(session.is_null());
    assert!(unsafe { str_of(orla_ffi::orla_open_error()) }.contains("abacus"));
    // NULL sessions are inert.
    assert_eq!(
        unsafe { orla_ffi::orla_evaluate(std::ptr::null_mut(), std::ptr::null()) },
        -1
    );
    assert_eq!(
        unsafe { str_of(orla_ffi::orla_error(std::ptr::null())) },
        ""
    );
    unsafe { orla_ffi::orla_close(std::ptr::null_mut()) };
}

#[test]
fn the_input_reaches_the_kernels() {
    // The voxel editor: the pointer at the replay's spot finds the tree,
    // the left button places a voxel; the window is the input's.
    let scene = example("voxel/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(scene.as_ptr(), c("jit").as_ptr(), std::ptr::null()) };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    unsafe { orla_ffi::orla_request(session, c("{\"op\":\"pin\",\"node\":\"state\"}").as_ptr()) };
    unsafe { orla_ffi::orla_request(session, c("{\"op\":\"pin\",\"node\":\"world\"}").as_ptr()) };
    let mut input = OrlaInput {
        window: [240.0, 135.0],
        ..OrlaInput::default()
    };
    assert_eq!(
        unsafe { orla_ffi::orla_evaluate(session, &input) },
        0,
        "{}",
        unsafe { str_of(orla_ffi::orla_error(session)) }
    );
    let mut field = OrlaField {
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
        unsafe { orla_ffi::orla_output(session, std::ptr::null(), &mut field) },
        0
    );
    assert_eq!((field.width, field.height), (240, 135));
    input.pointer = [120.0, 78.0];
    input.time = 0.05;
    assert_eq!(unsafe { orla_ffi::orla_evaluate(session, &input) }, 0);
    assert_eq!(
        unsafe { orla_ffi::orla_node(session, c("state").as_ptr(), std::ptr::null(), &mut field) },
        0
    );
    let state =
        unsafe { std::slice::from_raw_parts(field.words, (field.count * field.stride) as usize) };
    assert_eq!(
        (f32::from_bits(state[6]), f32::from_bits(state[7])),
        (120.0, 78.0)
    );
    assert_eq!(state[18], 1, "the pointer finds a voxel");
    input.buttons = 1;
    input.time = 0.1;
    assert_eq!(unsafe { orla_ffi::orla_evaluate(session, &input) }, 0);
    input.buttons = 0;
    input.time = 0.15;
    assert_eq!(unsafe { orla_ffi::orla_evaluate(session, &input) }, 0);
    assert_eq!(
        unsafe { orla_ffi::orla_node(session, c("world").as_ptr(), std::ptr::null(), &mut field) },
        0
    );
    let world =
        unsafe { std::slice::from_raw_parts(field.words, (field.count * field.stride) as usize) };
    assert_eq!(world[62811], 1, "a click places a voxel");
    unsafe { orla_ffi::orla_close(session) };
}

/// A JSON string literal of `text`.
fn serde_json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A scene opened on files handed in reads no disk: its document and the
/// files a `use` reaches come from what was put, at their paths. A file
/// put again after opening is read by the next evaluate.
#[test]
fn a_scene_opens_on_files_handed_in_and_reads_them_again() {
    let put = |files: *mut orla_ffi::OrlaFiles, path: &str, bytes: &[u8]| {
        let path = c(path);
        let done =
            unsafe { orla_ffi::orla_files_put(files, path.as_ptr(), bytes.as_ptr(), bytes.len()) };
        assert_eq!(done, 0);
    };
    let back = |colour: &str| {
        format!(
            "use \"kernels/\";\nnode back: fill.flat@1 {{ domain = vec2(4, 2); }}\noutput result <- back.result;\n// {colour}\n"
        )
    };
    let kernel = |colour: &str| {
        format!(
            "kernel fill.flat@1 {{ requires {{ field.write; }} output result: Image; compute {{ return vec4({colour}, 1.0); }} }}\n"
        )
    };
    let files = orla_ffi::orla_files();
    // Paths no disk has.
    put(files, "nowhere/scene.osn", back("first").as_bytes());
    put(
        files,
        "nowhere/kernels/fill.orla",
        kernel("0.2, 0.4, 0.6").as_bytes(),
    );
    let session = unsafe {
        orla_ffi::orla_open_files(files, c("nowhere/scene.osn").as_ptr(), c("cpu").as_ptr())
    };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    let picture = |session: *mut orla_ffi::OrlaSession| {
        assert_eq!(
            unsafe { orla_ffi::orla_evaluate(session, std::ptr::null()) },
            0,
            "{}",
            unsafe { str_of(orla_ffi::orla_error(session)) }
        );
        let mut field = OrlaField {
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
            unsafe { orla_ffi::orla_output(session, std::ptr::null(), &mut field) },
            0
        );
        pixel(&field, 1, 1)
    };
    assert!(near(picture(session), [0.2, 0.4, 0.6]));
    // The caller's hold goes; the session keeps its own, and a Kernel's
    // file put again is read by the next evaluate.
    put(
        files,
        "nowhere/kernels/fill.orla",
        kernel("0.7, 0.1, 0.3").as_bytes(),
    );
    unsafe { orla_ffi::orla_files_free(files) };
    assert!(near(picture(session), [0.7, 0.1, 0.3]));
    unsafe { orla_ffi::orla_close(session) };
    // Memory allocated in Orla, as a page asks for it, and freed.
    let room = orla_ffi::orla_alloc(64);
    assert!(!room.is_null());
    unsafe { orla_ffi::orla_free(room, 64) };
}

/// A picture is presented on a page's canvas; a program reads it in place,
/// and asking to present says so.
#[test]
fn presenting_is_for_a_page() {
    let scene = example("text/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(scene.as_ptr(), c("cpu").as_ptr(), std::ptr::null()) };
    assert!(!session.is_null());
    assert_eq!(
        unsafe { orla_ffi::orla_evaluate(session, std::ptr::null()) },
        0
    );
    let presented =
        unsafe { orla_ffi::orla_present(session, c("view").as_ptr(), std::ptr::null()) };
    assert_eq!(presented, -1);
    let why = unsafe { str_of(orla_ffi::orla_error(session)) };
    assert!(why.contains("orla_output"), "{why}");
    unsafe { orla_ffi::orla_close(session) };
}

/// Play through the C ABI on a host without a GPU: a replay handed in as
/// files plays to its end and says so, and each frame draws on the CPU,
/// with the frame-data line once F10 asks.
#[test]
fn a_host_plays_a_replay_through_the_play() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/ant");
    let files = orla_ffi::orla_files();
    for relative in ["scene.osn", "kernels/ant.orla", "tests/ant.replay"] {
        let bytes = std::fs::read(root.join(relative)).unwrap();
        let path = c(&format!("ant/{relative}"));
        let put =
            unsafe { orla_ffi::orla_files_put(files, path.as_ptr(), bytes.as_ptr(), bytes.len()) };
        assert_eq!(put, 0);
    }
    let session = unsafe {
        orla_ffi::orla_open_replay(files, c("ant/tests/ant.replay").as_ptr(), c("cpu").as_ptr())
    };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    unsafe { orla_ffi::orla_files_free(files) };
    let mut said = String::new();
    let mut frames = 0;
    // The replay's frames come at its own times, by the host's clock.
    while unsafe { orla_ffi::orla_play_replaying(session) } == 1 && frames < 2000 {
        let framed = unsafe { orla_ffi::orla_play_frame(session, 400.0, 300.0, 1.0, 0.0) };
        assert!(framed >= 0, "{}", unsafe {
            str_of(orla_ffi::orla_error(session))
        });
        said.push_str(&unsafe { str_of(orla_ffi::orla_play_said(session)) });
        said.push('\n');
        if framed == 1 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        frames += 1;
    }
    assert!(said.contains("replay: ok (2 frames, 14 checks)"), "{said}");
    // F10, and a CPU frame drawn with its line.
    assert_eq!(
        unsafe { orla_ffi::orla_play_key(session, c("F10").as_ptr(), 1) },
        0
    );
    assert_eq!(
        unsafe { orla_ffi::orla_play_frame(session, 400.0, 300.0, 1.0, 0.0) },
        0
    );
    let pixels = unsafe { orla_ffi::orla_play_pixels(session, 400, 300) };
    assert!(!pixels.is_null());
    let frame = unsafe { std::slice::from_raw_parts(pixels, 400 * 300) };
    assert!(
        frame[..400 * 9].contains(&0x00E0_E0E0),
        "the line's ink in the top rows"
    );
    assert_eq!(
        unsafe { orla_ffi::orla_play_key(session, c("Escape").as_ptr(), 1) },
        1
    );
    assert_eq!(
        unsafe { orla_ffi::orla_play_pad(session, c("south").as_ptr(), 1) },
        0
    );
    assert_eq!(
        unsafe { orla_ffi::orla_play_pad(session, c("nowhere").as_ptr(), 1) },
        -1
    );
    unsafe { orla_ffi::orla_close(session) };
}

/// A change through `orla_request` is recorded as the protocol's.
#[test]
fn a_request_through_the_abi_is_recorded_as_the_protocols() {
    let scene = example("text/scene.osn");
    let session =
        unsafe { orla_ffi::orla_open(scene.as_ptr(), c("cpu").as_ptr(), std::ptr::null()) };
    assert!(!session.is_null());
    let request = |line: &str| unsafe { str_of(orla_ffi::orla_request(session, c(line).as_ptr())) };
    let published = request(
        r#"{"op":"publish","source":"kernel t.said@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }"}"#,
    );
    assert!(published.contains("\"ok\""), "{published}");
    let history = request(r#"{"op":"history","since":0}"#);
    assert!(
        history.contains(r#"{"protocol":"orla_request"}"#),
        "{history}"
    );
    unsafe { orla_ffi::orla_close(session) };
}

/// `orla_request` requests are an agent's: a composed scene reads only its
/// documents' directories and those allowed by `orla_allow`.
#[test]
fn a_request_reads_beside_the_scene_or_where_the_program_allowed() {
    let base = std::env::temp_dir().join(format!("orla-ffi-reads-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (dir, elsewhere) = (base.join("scene"), base.join("elsewhere"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(dir.join("beside.bin"), b"ab").unwrap();
    std::fs::write(elsewhere.join("secret.bin"), b"xyz").unwrap();
    let text = |path: &str| {
        format!(
            "input data: Rows<Int> = bytes \"{path}\";\n\
             node copy: kernel {{ requires {{ field.read; field.write; }} input bytes: Rows<Int> <- data; output result: Rows<Int>; domain = 1; compute {{ return at(bytes, 0); }} }}\n\
             output result <- copy.result;\n"
        )
    };
    let scene = dir.join("scene.osn");
    std::fs::write(&scene, text("beside.bin")).unwrap();
    let session = unsafe {
        orla_ffi::orla_open(
            c(scene.to_str().unwrap()).as_ptr(),
            c("cpu").as_ptr(),
            std::ptr::null(),
        )
    };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    let secret = elsewhere.join("secret.bin").display().to_string();
    let quoted = text(&secret)
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    let request = c(&format!(
        "{{\"id\":1,\"op\":\"compose\",\"text\":\"{quoted}\"}}"
    ));
    let refused = unsafe { str_of(orla_ffi::orla_request(session, request.as_ptr())) };
    assert!(
        refused.contains("outside the directories the scene's documents are in or use"),
        "{refused}"
    );
    let allow = c(&format!("file.read={}", elsewhere.display()));
    assert_eq!(unsafe { orla_ffi::orla_allow(session, allow.as_ptr()) }, 0);
    let taken = unsafe { str_of(orla_ffi::orla_request(session, request.as_ptr())) };
    assert!(taken.contains("\"scene_loaded\""), "{taken}");
    assert_eq!(
        unsafe { orla_ffi::orla_allow(session, c("file.everything=/").as_ptr()) },
        -1
    );
    assert!(unsafe { str_of(orla_ffi::orla_error(session)) }.contains("--allow"));
    unsafe { orla_ffi::orla_close(session) };
    let _ = std::fs::remove_dir_all(&base);
}

/// A C ABI session prepares replacements off the caller's thread: frames
/// go on while an edited Kernel prepares, and the next sync publishes it.
#[test]
fn a_replacement_is_prepared_off_the_callers_thread() {
    let base = std::env::temp_dir().join(format!("orla-ffi-background-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let scene = base.join("scene.osn");
    let text = |red: &str| {
        format!(
            "node wave: kernel k.wave@1 {{ requires {{ field.write; }} output result: Image; domain = vec2(4, 4); compute {{ return vec4({red}, 0.5, 0.25, 1.0); }} }}\noutput result <- wave.result;\n"
        )
    };
    std::fs::write(&scene, text("0.1")).unwrap();
    let session = unsafe {
        orla_ffi::orla_open(
            c(scene.to_str().unwrap()).as_ptr(),
            c("cpu").as_ptr(),
            std::ptr::null(),
        )
    };
    assert!(!session.is_null(), "{}", unsafe {
        str_of(orla_ffi::orla_open_error())
    });
    // A later write, with a stamp of its own.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(&scene, text("0.9")).unwrap();
    let sync = c(r#"{"op":"sync"}"#);
    let said = unsafe { str_of(orla_ffi::orla_request(session, sync.as_ptr())) };
    assert!(said.contains("\"preparing\""), "{said}");
    let mut published = said.contains("\"published\"");
    for _ in 0..1000 {
        if published {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        let said = unsafe { str_of(orla_ffi::orla_request(session, sync.as_ptr())) };
        published = said.contains("\"published\"");
    }
    assert!(published, "the next sync publishes it");
    unsafe { orla_ffi::orla_close(session) };
    let _ = std::fs::remove_dir_all(&base);
}
