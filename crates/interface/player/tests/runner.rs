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

        // Verify pure-Rust Mach-O bundle sealing (`codesign::seal_macho_bundle`)
        // from a pre-signed template runner (`run.entelechy.app`) without invoking
        // `/usr/bin/codesign` on the newly exported `.app`!
        let tpl_dir = dir.join("Template.app");
        native::macos_app(
            &scene,
            &tpl_dir,
            &native::Options {
                runner: Some(&runner_bin()),
                app_id: Some("run.entelechy.app"),
                title: Some("Template"),
                ..Default::default()
            },
        )
        .expect("export template .app");
        let presigned_runner = tpl_dir.join("Contents/MacOS/Template");
        assert_eq!(
            entelechy_export::codesign::bundle_signing_id(
                &std::fs::read(&presigned_runner).unwrap()
            )
            .as_deref(),
            Some("run.entelechy.app")
        );

        let voxel_scene = root.join("examples/voxel/scene.esn");
        let sealed_app = dir.join("SealedVoxel.app");
        native::macos_app(
            &voxel_scene,
            &sealed_app,
            &native::Options {
                runner: Some(&presigned_runner),
                backend: Some("jit"),
                title: Some("Sealed Voxel"),
                ..Default::default()
            },
        )
        .expect("pure-Rust seal .app from pre-signed runner");
        let strict_check = Command::new("/usr/bin/codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(&sealed_app)
            .output()
            .expect("codesign verify runs");
        assert!(
            strict_check.status.success(),
            "pure-Rust sealed .app failed strict codesign: {}",
            String::from_utf8_lossy(&strict_check.stderr)
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runner_persists_saves_across_restarts() {
    let dir = std::env::temp_dir().join(format!("entelechy-runner-save-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let scene_path = dir.join("save_app.esn");
    std::fs::write(
        &scene_path,
        "kernel s.fresh@1 { requires { field.write; } output result: Rows<Int>; compute { return 100; } }\n\
         kernel s.step@1 { requires { field.read; field.write; } input last: Rows<Int>; output result: Rows<Int>; compute { return at(last, 0) + 1; } }\n\
         kernel s.pic@1 { requires { field.read; field.write; } input v: Rows<Int>; output result: Image; compute { return vec4(0.2, 0.4, 0.8, 1.0); } }\n\
         input saved: Rows<Int> = \"saves/progress.field\" else fresh.result;\n\
         node fresh: s.fresh@1 { domain = 1; }\n\
         node step: s.step@1 { last <~ step.result from saved; domain = 1; }\n\
         node pic: s.pic@1 { v <- step.result; domain = vec2(4, 4); }\n\
         output picture <- pic.result;\n\
         output kept <- step.result to \"saves/progress.field\";\n",
    )
    .unwrap();

    let compiled = page::compile(&scene_path, None).expect("save scene compiles");
    let bundle_path = dir.join("save_app.bundle");
    std::fs::write(&bundle_path, &compiled.bundle).unwrap();
    let save_dir = dir.join("user_saves");

    // First process launch: starts from fallback 100, runs 3 frames -> writes 103.
    let run1 = Command::new(runner_bin())
        .arg(&bundle_path)
        .args(["--backend", "jit", "--frames", "3", "--save-dir"])
        .arg(&save_dir)
        .output()
        .expect("run 1");
    assert!(
        run1.status.success(),
        "run 1 failed: {}",
        String::from_utf8_lossy(&run1.stderr)
    );
    let save_file = save_dir.join("saves/progress.field");
    let data1 = entelechy_formats::file::decode(&std::fs::read(&save_file).expect("save written"))
        .expect("valid .field");
    assert_eq!(data1.values().as_ref(), &[103u32]);

    // Second process launch: preloads 103 from `save_dir`, runs 2 frames -> writes 105!
    let run2 = Command::new(runner_bin())
        .arg(&bundle_path)
        .args(["--backend", "jit", "--frames", "2", "--save-dir"])
        .arg(&save_dir)
        .output()
        .expect("run 2");
    assert!(
        run2.status.success(),
        "run 2 failed: {}",
        String::from_utf8_lossy(&run2.stderr)
    );
    let data2 = entelechy_formats::file::decode(&std::fs::read(&save_file).expect("save updated"))
        .expect("valid .field");
    assert_eq!(data2.values().as_ref(), &[105u32]);

    let _ = std::fs::remove_dir_all(&dir);
}
