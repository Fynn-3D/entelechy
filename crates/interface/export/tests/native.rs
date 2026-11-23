//! Native application exports: `.app` (macOS), `.exe` (Windows PE), `.bin`
//! (Unix ELF/Mach-O), `.ios` (Xcode project), and `.android` (NDK project).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use entelechy_session::bundle::{Bundle, PointerMode, Standalone, read_trailer};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn exporter() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_entelechy-export"))
}

fn exported_c_symbols(header: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for token in header.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if token.starts_with("entelechy_") {
            found.insert(token.to_owned());
        }
    }
    found
}

fn referenced_calls(source: &str) -> BTreeSet<String> {
    let mut calls = BTreeSet::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"entelechy_") {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if bytes.get(i) == Some(&b'(') {
                calls.insert(source[start..i].to_owned());
            }
        } else {
            i += 1;
        }
    }
    calls
}

#[test]
fn desktop_app_and_exe_exports_pack_valid_standalone_bundles_and_check_runner_headers() {
    let root = repository().canonicalize().unwrap_or_else(|_| repository());
    let scene = root.join("examples/ant/scene.esn");
    let dir = std::env::temp_dir().join(format!("entelechy-native-exp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let macho_stub = dir.join("macho-runner");
    let mut macho_bytes = vec![0xCF, 0xFA, 0xED, 0xFE];
    macho_bytes.extend_from_slice(b"macho-stub-body");
    std::fs::write(&macho_stub, &macho_bytes).expect("macho stub");

    let pe_stub = dir.join("pe-runner.exe");
    let mut pe_bytes = b"MZ\x90\x00".to_vec();
    pe_bytes.extend_from_slice(b"pe-stub-body");
    std::fs::write(&pe_stub, &pe_bytes).expect("pe stub");

    // Passing a Mach-O runner to `.exe` is refused before packing.
    let bad_exe = Command::new(exporter())
        .arg(&scene)
        .args(["--runner"])
        .arg(&macho_stub)
        .args(["--out"])
        .arg(dir.join("Wrong.exe"))
        .output()
        .expect("exporter runs");
    assert!(
        !bad_exe.status.success()
            && String::from_utf8_lossy(&bad_exe.stderr).contains("Windows .exe needs a PE"),
        "expected PE check failure: {:?}",
        bad_exe
    );

    // Exporting `.exe` with a PE runner appends a valid Standalone trailer.
    let exe_out = dir.join("AntWorld.exe");
    let ok_exe = Command::new(exporter())
        .arg(&scene)
        .args(["--runner"])
        .arg(&pe_stub)
        .args([
            "--name",
            "Ant World",
            "--window",
            "800x600",
            "--lock-pointer",
            "--backend",
            "gpu",
            "--out",
        ])
        .arg(&exe_out)
        .output()
        .expect("exporter runs");
    assert!(
        ok_exe.status.success(),
        "exe export failed: {}",
        String::from_utf8_lossy(&ok_exe.stderr)
    );
    let raw_exe = std::fs::read(&exe_out).expect("exe written");
    assert!(raw_exe.starts_with(&pe_bytes), "PE prefix preserved");
    let unpacked = read_trailer(&exe_out)
        .expect("trailer valid")
        .expect("trailer present");
    assert_eq!(unpacked.title, "Ant World");
    assert_eq!(unpacked.backend, "gpu");
    assert_eq!(unpacked.pointer, PointerMode::Lock);
    assert_eq!(unpacked.window, Some((800, 600)));
    assert!(Bundle::decode(&unpacked.bundle).is_ok());

    // Exporting `.app` leaves `Contents/MacOS/<stem>` byte-identical to the Mach-O runner.
    let app_out = dir.join("AntWorld.app");
    let ok_app = Command::new(exporter())
        .arg(&scene)
        .args(["--runner"])
        .arg(&macho_stub)
        .args([
            "--name",
            "Ant & Friends",
            "--app-id",
            "run.entelechy.ant_friends",
            "--hold-pointer",
            "--out",
        ])
        .arg(&app_out)
        .output()
        .expect("exporter runs");
    assert!(
        ok_app.status.success(),
        "app export failed: {}",
        String::from_utf8_lossy(&ok_app.stderr)
    );
    let copied_runner =
        std::fs::read(app_out.join("Contents/MacOS/AntWorld")).expect("MacOS executable");
    assert_eq!(
        copied_runner, macho_bytes,
        "Mach-O runner copied byte-for-byte so Apple Silicon linker signature holds"
    );
    let plist = std::fs::read_to_string(app_out.join("Contents/Info.plist")).expect("Info.plist");
    assert!(plist.contains("<string>Ant &amp; Friends</string>"));
    assert!(plist.contains("<string>run.entelechy.ant_friends</string>"));
    let entapp =
        std::fs::read(app_out.join("Contents/Resources/app.entapp")).expect("app.entapp written");
    let decoded_app = Standalone::decode(&entapp).expect("valid app.entapp");
    assert_eq!(decoded_app.title, "Ant & Friends");
    assert_eq!(decoded_app.pointer, PointerMode::Hold);
    assert!(Bundle::decode(&decoded_app.bundle).is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ios_and_android_exports_emit_complete_projects_matching_c_abi_header() {
    let root = repository().canonicalize().unwrap_or_else(|_| repository());
    let scene = root.join("examples/ant/scene.esn");
    let dir = std::env::temp_dir().join(format!("entelechy-mobile-exp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    // `.ios` refuses `--backend jit` because iOS forbids runtime executable pages.
    let bad_ios = Command::new(exporter())
        .arg(&scene)
        .args(["--web"])
        .arg(&root)
        .args(["--backend", "jit", "--out"])
        .arg(dir.join("Bad.ios"))
        .output()
        .expect("exporter runs");
    assert!(
        !bad_ios.status.success()
            && String::from_utf8_lossy(&bad_ios.stderr).contains("iOS forbids"),
        "expected iOS JIT refusal: {:?}",
        bad_ios
    );

    let fake_static_lib = dir.join("libentelechy_ffi.a");
    std::fs::write(&fake_static_lib, b"!<arch>\n").expect("stub .a");
    let ios_out = dir.join("JumpJump.ios");
    let ok_ios = Command::new(exporter())
        .arg(&scene)
        .args(["--web"])
        .arg(&root)
        .args(["--lib"])
        .arg(&fake_static_lib)
        .args([
            "--name",
            "Jump Jump",
            "--app-id",
            "run.entelechy.jump_jump",
            "--lock-pointer",
            "--out",
        ])
        .arg(&ios_out)
        .output()
        .expect("exporter runs");
    assert!(
        ok_ios.status.success(),
        "ios export failed: {}",
        String::from_utf8_lossy(&ok_ios.stderr)
    );
    assert!(ios_out.join("JumpJump.xcodeproj/project.pbxproj").is_file());
    assert!(ios_out.join("lib/libentelechy_ffi.a").is_file());
    let ios_bundle = std::fs::read(ios_out.join("scene.bundle")).expect("ios scene.bundle");
    assert!(Bundle::decode(&ios_bundle).is_ok());
    let ios_header = std::fs::read_to_string(ios_out.join("entelechy.h")).expect("ios header");
    let ios_source = std::fs::read_to_string(ios_out.join("App.m")).expect("App.m");
    assert!(
        !ios_source.contains("@@"),
        "all template slots filled in App.m"
    );
    let header_syms = exported_c_symbols(&ios_header);
    for call in referenced_calls(&ios_source) {
        assert!(
            header_syms.contains(&call),
            "App.m calls `{call}` not declared in entelechy.h"
        );
    }

    let fake_shared_lib = dir.join("libentelechy_ffi.so");
    std::fs::write(&fake_shared_lib, b"\x7fELFstub").expect("stub .so");
    let android_out = dir.join("JumpJump.android");
    let ok_android = Command::new(exporter())
        .arg(&scene)
        .args(["--web"])
        .arg(&root)
        .args(["--lib"])
        .arg(&fake_shared_lib)
        .args([
            "--name",
            "Jump Jump",
            "--app-id",
            "run.entelechy.jump_jump",
            "--out",
        ])
        .arg(&android_out)
        .output()
        .expect("exporter runs");
    assert!(
        ok_android.status.success(),
        "android export failed: {}",
        String::from_utf8_lossy(&ok_android.stderr)
    );
    assert!(
        android_out
            .join("app/src/main/jniLibs/arm64-v8a/libentelechy_ffi.so")
            .is_file()
    );
    let android_bundle = std::fs::read(android_out.join("app/src/main/assets/scene.bundle"))
        .expect("android bundle");
    assert!(Bundle::decode(&android_bundle).is_ok());
    let android_main =
        std::fs::read_to_string(android_out.join("app/src/main/cpp/main.c")).expect("main.c");
    assert!(
        !android_main.contains("@@"),
        "all template slots filled in main.c"
    );
    for call in referenced_calls(&android_main) {
        assert!(
            header_syms.contains(&call),
            "main.c calls `{call}` not declared in entelechy.h"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn exporter_accepts_precompiled_bundle_for_desktop_and_mobile_targets() {
    let root = repository().canonicalize().unwrap_or_else(|_| repository());
    let scene = root.join("examples/ant/scene.esn");
    let dir = std::env::temp_dir().join(format!("entelechy-bundle-in-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let bundle_path = dir.join("ant.bundle");
    let made_bundle = Command::new(exporter())
        .arg(&scene)
        .args(["--out"])
        .arg(&bundle_path)
        .output()
        .expect("compile bundle");
    assert!(made_bundle.status.success());
    let bundle_bytes = std::fs::read(&bundle_path).expect("bundle bytes");

    let pe_stub = dir.join("pe-runner.exe");
    std::fs::write(&pe_stub, b"MZ\x90\x00pe-stub").expect("pe stub");

    let exe_out = dir.join("FromBundle.exe");
    let ok_exe = Command::new(exporter())
        .arg(&bundle_path)
        .args(["--runner"])
        .arg(&pe_stub)
        .args(["--name", "From Bundle", "--out"])
        .arg(&exe_out)
        .output()
        .expect("exe from bundle");
    assert!(
        ok_exe.status.success(),
        "{}",
        String::from_utf8_lossy(&ok_exe.stderr)
    );
    let unpacked = read_trailer(&exe_out).unwrap().unwrap();
    assert_eq!(unpacked.title, "From Bundle");
    assert_eq!(unpacked.bundle, bundle_bytes);

    let entapp_out = dir.join("app.entapp");
    let ok_entapp = Command::new(exporter())
        .arg(&bundle_path)
        .args(["--name", "Envelope App", "--lock-pointer", "--out"])
        .arg(&entapp_out)
        .output()
        .expect("entapp from bundle");
    assert!(ok_entapp.status.success());
    let decoded_env =
        Standalone::decode(&std::fs::read(&entapp_out).unwrap()).expect("valid entapp");
    assert_eq!(decoded_env.title, "Envelope App");
    assert_eq!(decoded_env.pointer, PointerMode::Lock);
    assert_eq!(decoded_env.bundle, bundle_bytes);

    let ios_out = dir.join("FromBundle.ios");
    let ok_ios = Command::new(exporter())
        .arg(&bundle_path)
        .args(["--web"])
        .arg(&root)
        .args(["--name", "Bundle iOS", "--out"])
        .arg(&ios_out)
        .output()
        .expect("ios from bundle");
    assert!(ok_ios.status.success());
    assert_eq!(
        std::fs::read(ios_out.join("scene.bundle")).unwrap(),
        bundle_bytes
    );

    let _ = std::fs::remove_dir_all(&dir);
}
