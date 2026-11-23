//! Native standalone application exports: macOS `.app` bundles, Windows `.exe`
//! and Linux `.bin` self-contained executables, iOS Xcode projects (`.ios`),
//! and Android NDK projects (`.android`).
//!
//! Desktop exports pair a precompiled `entelechy-runner` binary with a
//! [`Standalone`](entelechy_session::bundle::Standalone) envelope holding the
//! compiled scene [`Bundle`](entelechy_session::bundle::Bundle), so no Rust or C
//! toolchain runs at export time. Mobile exports emit clean Xcode / Android NDK
//! project directories over `entelechy.h` and `scene.bundle`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_session::bundle::{PointerMode, Standalone, pack_executable};

use crate::page::{self, default_title, fill};

/// Options shared across native export targets.
#[derive(Clone, Debug, Default)]
pub struct Options<'a> {
    /// Prebuilt `entelechy-runner` binary for `.app`, `.exe`, or `.bin`.
    pub runner: Option<&'a Path>,
    /// Root directory holding `bridges/{ios,android}` and `entelechy.h`.
    pub bridge_root: Option<&'a Path>,
    /// Optional prebuilt `libentelechy_ffi.{a,so}` copied into `.ios` or `.android`.
    pub lib: Option<&'a Path>,
    /// Preferred runtime backend (`gpu`, `jit`, `cpu`, or `best`).
    pub backend: Option<&'a str>,
    /// Optional character whitelist for subsetting `font` nodes.
    pub chars: Option<&'a str>,
    /// Pointer capture mode on click or touch drag.
    pub pointer: PointerMode,
    /// Application display title (defaults to [`default_title`]).
    pub title: Option<&'a str>,
    /// Fixed initial logical window dimensions, when requested.
    pub window: Option<(u32, u32)>,
    /// Reverse-DNS bundle / package identifier (`run.entelechy.<slug>` by default).
    pub app_id: Option<&'a str>,
}

/// Kind of single-file packed desktop executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExeKind {
    /// Windows Portable Executable (`.exe`, `"MZ"` header).
    WindowsPe,
    /// Unix standalone binary (`.bin`, ELF or Mach-O header).
    UnixBin,
}

/// Summary of an exported desktop application (`.app`, `.exe`, or `.bin`).
pub struct MadeApp {
    pub modules: usize,
    pub inputs: usize,
    pub bundle_bytes: usize,
    pub runner_bytes: usize,
    pub total_bytes: usize,
    pub fonts: Vec<(String, usize, usize)>,
}

/// Summary of an exported mobile project directory (`.ios` or `.android`).
pub struct MadeMobile {
    pub modules: usize,
    pub inputs: usize,
    pub bundle_bytes: usize,
    pub lib_copied: bool,
    pub fonts: Vec<(String, usize, usize)>,
}

fn chosen_title(scene: &Path, title: Option<&str>) -> String {
    title
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| default_title(scene))
}

fn ascii_slug(text: &str, fallback: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !out.is_empty() && !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        return fallback.to_owned();
    }
    if out.as_bytes()[0].is_ascii_digit() {
        out.insert_str(0, "app_");
    }
    out
}

fn executable_stem(out: &Path, fallback_title: &str) -> String {
    let raw = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback_title);
    let mut clean: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if clean.is_empty() {
        clean = ascii_slug(fallback_title, "EntelechyApp");
    }
    clean
}

fn default_app_id(title: &str) -> String {
    format!("run.entelechy.{}", ascii_slug(title, "app"))
}

fn validate_app_id(id: &str) -> Result<String, String> {
    let trimmed = id.trim();
    let segments: Vec<&str> = trimmed.split('.').collect();
    if segments.len() < 2 {
        return Err(format!(
            "--app-id `{trimmed}` needs at least two dot-separated segments (e.g. run.entelechy.game)"
        ));
    }
    for seg in &segments {
        if seg.is_empty()
            || seg.as_bytes()[0].is_ascii_digit()
            || !seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(format!(
                "--app-id `{trimmed}` has invalid segment `{seg}` (use ASCII letters, digits and underscores, starting with a letter)"
            ));
        }
    }
    Ok(trimmed.to_owned())
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn is_macho(bytes: &[u8]) -> bool {
    let Some(head) = bytes.get(..4) else {
        return false;
    };
    matches!(
        u32::from_be_bytes([head[0], head[1], head[2], head[3]]),
        0xFEED_FACF | 0xCFFA_EDFE | 0xFEED_FACE | 0xCEFA_EDFE | 0xCAFE_BABE
    )
}

fn is_pe(bytes: &[u8]) -> bool {
    bytes.starts_with(b"MZ")
}

fn is_elf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF")
}

fn find_runner(explicit: Option<&Path>, bin_name: &str) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        return Err(format!("{}: runner binary not found", path.display()));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let beside = dir.join(bin_name);
        if beside.is_file() {
            return Ok(beside);
        }
    }
    Err(format!(
        "cannot find `{bin_name}` beside `entelechy-export`; build it (`cargo build --release -p entelechy-player --bin entelechy-runner`) or pass `--runner <file>`"
    ))
}

#[cfg(unix)]
fn mark_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(not(unix))]
fn mark_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Exports `scene` as a macOS `.app` application bundle at `out`.
///
/// The Mach-O `entelechy-runner` binary is copied untouched into
/// `Contents/MacOS/` so its Apple Silicon linker signature remains valid, and
/// the compiled [`Standalone`] envelope is written to
/// `Contents/Resources/app.entapp`.
pub fn macos_app(scene: &Path, out: &Path, opts: &Options<'_>) -> Result<MadeApp, String> {
    let backend = match opts.backend {
        None | Some("best") => "best",
        Some(name @ ("gpu" | "jit" | "cpu")) => name,
        Some(other) => {
            return Err(format!(
                "--backend for a .app is gpu, jit, cpu or best, not `{other}`"
            ));
        }
    };
    let runner_path = find_runner(opts.runner, "entelechy-runner")?;
    let runner_bytes =
        std::fs::read(&runner_path).map_err(|e| format!("{}: {e}", runner_path.display()))?;
    if !is_macho(&runner_bytes) {
        return Err(format!(
            "{}: a macOS .app needs a Mach-O `entelechy-runner` binary",
            runner_path.display()
        ));
    }

    let title = chosen_title(scene, opts.title);
    let app_id = match opts.app_id {
        Some(id) => validate_app_id(id)?,
        None => default_app_id(&title),
    };
    let exe_name = executable_stem(out, &title);
    let compiled = page::compile(scene, opts.chars)?;
    let app = Standalone {
        title: title.clone(),
        backend: backend.to_owned(),
        pointer: opts.pointer,
        window: opts.window,
        bundle: compiled.bundle.clone(),
    };
    let app_bytes = app.encode();

    let contents = out.join("Contents");
    let macos_dir = contents.join("MacOS");
    let resources_dir = contents.join("Resources");
    std::fs::create_dir_all(&macos_dir).map_err(|e| format!("{}: {e}", macos_dir.display()))?;
    std::fs::create_dir_all(&resources_dir)
        .map_err(|e| format!("{}: {e}", resources_dir.display()))?;

    let plist = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n\
<dict>\n\
    <key>CFBundleDevelopmentRegion</key>\n\
    <string>en</string>\n\
    <key>CFBundleDisplayName</key>\n\
    <string>{title_xml}</string>\n\
    <key>CFBundleExecutable</key>\n\
    <string>{exe_xml}</string>\n\
    <key>CFBundleIdentifier</key>\n\
    <string>{id_xml}</string>\n\
    <key>CFBundleInfoDictionaryVersion</key>\n\
    <string>6.0</string>\n\
    <key>CFBundleName</key>\n\
    <string>{title_xml}</string>\n\
    <key>CFBundlePackageType</key>\n\
    <string>APPL</string>\n\
    <key>CFBundleShortVersionString</key>\n\
    <string>0.1.0</string>\n\
    <key>CFBundleVersion</key>\n\
    <string>1</string>\n\
    <key>NSHighResolutionCapable</key>\n\
    <true/>\n\
</dict>\n\
</plist>\n",
        title_xml = escape_xml(&title),
        exe_xml = escape_xml(&exe_name),
        id_xml = escape_xml(&app_id),
    );
    let plist_path = contents.join("Info.plist");
    std::fs::write(&plist_path, plist.as_bytes())
        .map_err(|e| format!("{}: {e}", plist_path.display()))?;

    let exe_out = macos_dir.join(&exe_name);
    std::fs::write(&exe_out, &runner_bytes).map_err(|e| format!("{}: {e}", exe_out.display()))?;
    mark_executable(&exe_out)?;

    let res_out = resources_dir.join("app.entapp");
    std::fs::write(&res_out, &app_bytes).map_err(|e| format!("{}: {e}", res_out.display()))?;

    Ok(MadeApp {
        modules: compiled.modules,
        inputs: compiled.inputs,
        bundle_bytes: compiled.bundle.len(),
        runner_bytes: runner_bytes.len(),
        total_bytes: runner_bytes.len() + app_bytes.len() + plist.len(),
        fonts: compiled.fonts,
    })
}

/// Exports `scene` as a single-file desktop executable (`.exe` or `.bin`) with
/// the compiled [`Standalone`] envelope appended as a trailer.
pub fn packed_executable(
    scene: &Path,
    out: &Path,
    kind: ExeKind,
    opts: &Options<'_>,
) -> Result<MadeApp, String> {
    let backend = match opts.backend {
        None | Some("best") => "best",
        Some(name @ ("gpu" | "jit" | "cpu")) => name,
        Some(other) => {
            return Err(format!(
                "--backend for a standalone executable is gpu, jit, cpu or best, not `{other}`"
            ));
        }
    };
    let default_bin = match kind {
        ExeKind::WindowsPe => "entelechy-runner.exe",
        ExeKind::UnixBin => "entelechy-runner",
    };
    let runner_path = find_runner(opts.runner, default_bin)?;
    let runner_bytes =
        std::fs::read(&runner_path).map_err(|e| format!("{}: {e}", runner_path.display()))?;
    match kind {
        ExeKind::WindowsPe => {
            if !is_pe(&runner_bytes) {
                return Err(format!(
                    "{}: a Windows .exe needs a PE `entelechy-runner.exe` binary (pass `--runner <path-to-entelechy-runner.exe>`)",
                    runner_path.display()
                ));
            }
        }
        ExeKind::UnixBin => {
            if !is_elf(&runner_bytes) && !is_macho(&runner_bytes) {
                return Err(format!(
                    "{}: expected an ELF or Mach-O `entelechy-runner` binary",
                    runner_path.display()
                ));
            }
        }
    }

    let title = chosen_title(scene, opts.title);
    let compiled = page::compile(scene, opts.chars)?;
    let app = Standalone {
        title,
        backend: backend.to_owned(),
        pointer: opts.pointer,
        window: opts.window,
        bundle: compiled.bundle.clone(),
    };
    let packed = pack_executable(&runner_bytes, &app);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(out, &packed).map_err(|e| format!("{}: {e}", out.display()))?;
    if kind == ExeKind::UnixBin {
        mark_executable(out)?;
    }

    Ok(MadeApp {
        modules: compiled.modules,
        inputs: compiled.inputs,
        bundle_bytes: compiled.bundle.len(),
        runner_bytes: runner_bytes.len(),
        total_bytes: packed.len(),
        fonts: compiled.fonts,
    })
}

fn find_header(root: &Path) -> Result<Vec<u8>, String> {
    for rel in [
        "crates/interface/ffi/include/entelechy.h",
        "include/entelechy.h",
    ] {
        let candidate = root.join(rel);
        if candidate.is_file() {
            return std::fs::read(&candidate).map_err(|e| format!("{}: {e}", candidate.display()));
        }
    }
    Err(format!(
        "cannot find `entelechy.h` under `{}` (looked in crates/interface/ffi/include/ and include/)",
        root.display()
    ))
}

fn read_template(bridge: &Path, rel: &str) -> Result<String, String> {
    let path = bridge.join(rel);
    std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_filled(
    bridge: &Path,
    rel: &str,
    slots: &BTreeMap<&str, String>,
    dst: &Path,
) -> Result<(), String> {
    let tpl = read_template(bridge, rel)?;
    let text = fill(&tpl, bridge, slots)?;
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(dst, text.as_bytes()).map_err(|e| format!("{}: {e}", dst.display()))
}

/// Exports `scene` as an iOS Xcode project directory (`.ios`) over `bridges/ios`.
pub fn ios_project(scene: &Path, out: &Path, opts: &Options<'_>) -> Result<MadeMobile, String> {
    let backend = match opts.backend {
        None | Some("best") => "best",
        Some(name @ ("gpu" | "cpu")) => name,
        Some("jit") => {
            return Err(
                "iOS forbids mapping executable pages for a runtime JIT; use gpu, cpu or best"
                    .into(),
            );
        }
        Some(other) => {
            return Err(format!(
                "--backend for .ios is gpu, cpu or best, not `{other}`"
            ));
        }
    };
    let root = opts.bridge_root.unwrap_or(Path::new("."));
    let bridge = root.join("bridges").join("ios");
    let header = find_header(root)?;

    let title = chosen_title(scene, opts.title);
    let app_id = match opts.app_id {
        Some(id) => validate_app_id(id)?,
        None => default_app_id(&title),
    };
    let exe_name = executable_stem(out, &title);
    let compiled = page::compile(scene, opts.chars)?;

    let mut slots = BTreeMap::new();
    slots.insert("title", escape_xml(&title));
    slots.insert("executable", exe_name.clone());
    slots.insert("bundle_id", app_id);
    slots.insert("backend", backend.to_owned());
    slots.insert("pointer", opts.pointer.as_str().to_owned());

    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    write_filled(&bridge, "Info.plist", &slots, &out.join("Info.plist"))?;
    write_filled(&bridge, "App.m", &slots, &out.join("App.m"))?;
    write_filled(&bridge, "README.md", &slots, &out.join("README.md"))?;
    write_filled(
        &bridge,
        "project.pbxproj",
        &slots,
        &out.join(format!("{exe_name}.xcodeproj"))
            .join("project.pbxproj"),
    )?;

    let h_dst = out.join("entelechy.h");
    std::fs::write(&h_dst, &header).map_err(|e| format!("{}: {e}", h_dst.display()))?;
    let b_dst = out.join("scene.bundle");
    std::fs::write(&b_dst, &compiled.bundle).map_err(|e| format!("{}: {e}", b_dst.display()))?;

    let lib_dir = out.join("lib");
    std::fs::create_dir_all(&lib_dir).map_err(|e| format!("{}: {e}", lib_dir.display()))?;
    let mut lib_copied = false;
    if let Some(lib_path) = opts.lib {
        let bytes = std::fs::read(lib_path).map_err(|e| format!("{}: {e}", lib_path.display()))?;
        let dst = lib_dir.join("libentelechy_ffi.a");
        std::fs::write(&dst, bytes).map_err(|e| format!("{}: {e}", dst.display()))?;
        lib_copied = true;
    }

    Ok(MadeMobile {
        modules: compiled.modules,
        inputs: compiled.inputs,
        bundle_bytes: compiled.bundle.len(),
        lib_copied,
        fonts: compiled.fonts,
    })
}

/// Exports `scene` as an Android NDK (`NativeActivity`) project directory
/// (`.android`) over `bridges/android`.
pub fn android_project(scene: &Path, out: &Path, opts: &Options<'_>) -> Result<MadeMobile, String> {
    let backend = match opts.backend {
        None | Some("best") => "best",
        Some(name @ ("gpu" | "jit" | "cpu")) => name,
        Some(other) => {
            return Err(format!(
                "--backend for .android is gpu, jit, cpu or best, not `{other}`"
            ));
        }
    };
    let root = opts.bridge_root.unwrap_or(Path::new("."));
    let bridge = root.join("bridges").join("android");
    let header = find_header(root)?;

    let title = chosen_title(scene, opts.title);
    let package_id = match opts.app_id {
        Some(id) => validate_app_id(id)?,
        None => default_app_id(&title),
    };
    let target_name = executable_stem(out, &title);
    let lib_name = ascii_slug(&target_name, "entelechy_app");
    let compiled = page::compile(scene, opts.chars)?;

    let mut slots = BTreeMap::new();
    slots.insert("title", escape_xml(&title));
    slots.insert("target_name", target_name);
    slots.insert("package_id", package_id);
    slots.insert("lib_name", lib_name);
    slots.insert("backend", backend.to_owned());
    slots.insert("pointer", opts.pointer.as_str().to_owned());

    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for rel in [
        "settings.gradle",
        "build.gradle",
        "app/build.gradle",
        "app/src/main/AndroidManifest.xml",
        "app/src/main/cpp/CMakeLists.txt",
        "app/src/main/cpp/main.c",
        "README.md",
    ] {
        write_filled(&bridge, rel, &slots, &out.join(rel))?;
    }

    let h_dst = out.join("app/src/main/cpp/entelechy.h");
    std::fs::write(&h_dst, &header).map_err(|e| format!("{}: {e}", h_dst.display()))?;
    let assets_dir = out.join("app/src/main/assets");
    std::fs::create_dir_all(&assets_dir).map_err(|e| format!("{}: {e}", assets_dir.display()))?;
    let b_dst = assets_dir.join("scene.bundle");
    std::fs::write(&b_dst, &compiled.bundle).map_err(|e| format!("{}: {e}", b_dst.display()))?;

    let mut lib_copied = false;
    if let Some(lib_path) = opts.lib {
        let bytes = std::fs::read(lib_path).map_err(|e| format!("{}: {e}", lib_path.display()))?;
        let jni_dir = out.join("app/src/main/jniLibs/arm64-v8a");
        std::fs::create_dir_all(&jni_dir).map_err(|e| format!("{}: {e}", jni_dir.display()))?;
        let dst = jni_dir.join("libentelechy_ffi.so");
        std::fs::write(&dst, bytes).map_err(|e| format!("{}: {e}", dst.display()))?;
        lib_copied = true;
    }

    Ok(MadeMobile {
        modules: compiled.modules,
        inputs: compiled.inputs,
        bundle_bytes: compiled.bundle.len(),
        lib_copied,
        fonts: compiled.fonts,
    })
}
