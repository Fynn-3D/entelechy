//! What an interface asks of a scene's outputs, answered by the session:
//! the output a name reads, and whether the device can show a picture as
//! it lies.

use std::path::{Path, PathBuf};

use orla_kernel::Context;
use orla_session::{Limits, Session};

const KERNELS: &str = "
kernel t.paint@1 { requires { field.write; } output result: Grid<Vec4>; compute { return vec4(0.5, 0.25, 1.0, 1.0); } }
kernel t.narrow@1 { requires { field.write; } output result: Grid<Vec4 as unorm8>; compute { return vec4(0.5, 0.25, 1.0, 1.0); } }
kernel t.count@1 { requires { field.write; } output result: Rows<Int>; compute { return element(); } }
";

const SCENE: &str = "
use \"k.orla\";
node paint: t.paint@1 { domain = vec2(4, 2); }
node narrow: t.narrow@1 { domain = vec2(4, 2); }
node count: t.count@1 { domain = 3; }
output picture <- paint.result;
output kept <- narrow.result;
output rows <- count.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-outputs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.osn"), SCENE).unwrap();
    dir
}

fn open(dir: &Path, backend: &str) -> Option<Session> {
    let runtime = Session::runtime_for(backend, Limits::default()).ok()?;
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    let events = session.sync_settled();
    assert!(session.graph().is_some(), "{events:?}");
    Some(session)
}

#[test]
fn an_output_is_read_by_its_name_or_first() {
    let dir = scratch("names");
    let mut session = open(&dir, "cpu").unwrap();
    assert_eq!(
        session.output(None).unwrap_err(),
        "evaluate before reading an output"
    );
    session.evaluate(&Context::default()).unwrap();
    let first = session.output(None).unwrap();
    assert_eq!(session.output(Some("picture")).unwrap(), first);
    let rows = session.output(Some("rows")).unwrap();
    assert_ne!(rows, first);
    let wrong = session.output(Some("row")).unwrap_err();
    assert!(
        wrong.starts_with("no output named `row`") && wrong.contains("`rows`"),
        "{wrong}"
    );
}

#[test]
fn only_a_wide_picture_the_device_holds_is_shown_as_it_lies() {
    let dir = scratch("device");
    let mut cpu = open(&dir, "cpu").unwrap();
    cpu.evaluate(&Context::default()).unwrap();
    let picture = cpu.output(Some("picture")).unwrap();
    assert!(
        cpu.on_device(picture).is_none(),
        "the interpreter holds nothing on a device"
    );

    let Some(mut gpu) = open(&dir, "gpu") else {
        eprintln!("skipped: no GPU");
        return;
    };
    gpu.evaluate(&Context::default()).unwrap();
    let picture = gpu.output(Some("picture")).unwrap();
    let (_, width, height) = gpu
        .on_device(picture)
        .expect("a wide picture on the device");
    assert_eq!((width, height), (4, 2));
    for narrow in ["kept", "rows"] {
        let handle = gpu.output(Some(narrow)).unwrap();
        assert!(
            gpu.on_device(handle).is_none(),
            "`{narrow}` is brought back to be shown"
        );
    }
}

/// A Field goes to a file by its extension, saved by port or exported by
/// handle: a narrow picture is a PNG or PPM like a wide one, and a format
/// that cannot take it is refused, the file named, nothing written.
#[test]
fn a_field_is_written_as_its_file_says() {
    let dir = scratch("export");
    let mut session = open(&dir, "cpu").unwrap();
    session.allow_writes(&dir).unwrap();
    session.evaluate(&Context::default()).unwrap();
    for (output, file) in [("picture", "wide.png"), ("kept", "narrow.png")] {
        let handle = session.output(Some(output)).unwrap();
        session.export(handle, &dir.join(file)).unwrap();
        let bytes = std::fs::read(dir.join(file)).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"), "{file} is a PNG");
    }
    let kept = session.output(Some("kept")).unwrap();
    session.export(kept, &dir.join("narrow.ppm")).unwrap();
    assert!(
        std::fs::read(dir.join("narrow.ppm"))
            .unwrap()
            .starts_with(b"P6\n4 2\n255\n")
    );
    let rows = session.output(Some("rows")).unwrap();
    for file in ["rows.ppm", "rows.txt"] {
        let why = session
            .export(rows, &dir.join(file))
            .unwrap_err()
            .to_string();
        assert!(
            why.starts_with("cannot write ") && why.contains(file),
            "{why}"
        );
        assert!(!dir.join(file).exists());
    }
}
