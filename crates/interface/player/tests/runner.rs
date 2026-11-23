//! End-to-end execution of `entelechy-runner` on exported standalone bundles,
//! trailer executables, and macOS `.app` bundles.

use std::path::{Path, PathBuf};
use std::process::Command;

use entelechy_export::{native, page};
use entelechy_session::bundle::{PointerMode, Standalone, pack_executable};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn runner_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_entelechy-runner"))
}

#[test]
fn runner_executes_compiled_bundle_and_trailer_file() {
    let root = repository().canonicalize().unwrap_or_else(|_| repository());
    let scene = root.join("examples/ant/scene.esn");
    let compiled = page::compile(&scene, None).expect("ant compiles");

    let dir = std::env::temp_dir().join(format!("entelechy-runner-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let bundle_path = dir.join("ant.bundle");
    std::fs::write(&bundle_path, &compiled.bundle).expect("write bundle");

    let out = Command::new(runner_bin())
        .arg(&bundle_path)
        .args(["--backend", "jit", "--frames", "2"])
        .output()
        .expect("runner runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("ran 2 frame(s) on jit"),
        "runner failed: {}\n{}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );

    // Also run a packed PE-stub + trailer file handed to `entelechy-runner`.
    let packed = pack_executable(
        b"MZ\x90\x00stub",
        &Standalone {
            title: "Trailer Ant".into(),
            backend: "jit".into(),
            pointer: PointerMode::Free,
            window: Some((320, 180)),
            bundle: compiled.bundle,
        },
    );
    let exe_path = dir.join("Ant.exe");
    std::fs::write(&exe_path, packed).expect("write exe");
    let out_exe = Command::new(runner_bin())
        .arg(&exe_path)
        .args(["--frames", "2"])
        .output()
        .expect("runner runs packed file");
    let stdout_exe = String::from_utf8_lossy(&out_exe.stdout);
    assert!(
        out_exe.status.success() && stdout_exe.contains("Trailer Ant: ran 2 frame(s) on jit"),
        "packed trailer run failed: {}\n{}",
        stdout_exe,
        String::from_utf8_lossy(&out_exe.stderr)
    );

    #[cfg(target_os = "macos")]
    {
        let app_dir = dir.join("NativeAnt.app");
        native::macos_app(
            &scene,
            &app_dir,
            &native::Options {
                runner: Some(&runner_bin()),
                backend: Some("jit"),
                title: Some("Native Ant"),
                ..Default::default()
            },
        )
        .expect("export macOS .app");

        let verify = Command::new("/usr/bin/codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(&app_dir)
            .output()
            .expect("codesign runs");
        assert!(
            verify.status.success(),
            "exported .app failed strict codesign: {}",
            String::from_utf8_lossy(&verify.stderr)
        );

        let app_exe = app_dir.join("Contents/MacOS/NativeAnt");
        let app_run = Command::new(&app_exe)
            .args(["--frames", "3"])
            .output()
            .expect("exported .app executable launches directly");
        let app_stdout = String::from_utf8_lossy(&app_run.stdout);
        assert!(
            app_run.status.success() && app_stdout.contains("Native Ant: ran 3 frame(s) on jit"),
            "exported .app failed: {}\n{}",
            app_stdout,
            String::from_utf8_lossy(&app_run.stderr)
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
