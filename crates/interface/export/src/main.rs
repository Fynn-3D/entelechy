//! `entelechy-export`: the Host's files as other software's formats. Mesh
//! rows (and a PNG) become a GLB, rows of one or two floats a WAV, a
//! grid a PNG, a font's Fields a TrueType, OpenType (CFF) or WOFF font; a
//! scene becomes a bundle, one page, a desktop app (`.app`, `.exe`, `.bin`),
//! or a mobile project (`.ios`, `.android`) that plays it.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use entelechy_export::{font, gltf, native, page};
use entelechy_formats::rig::Part;
use entelechy_host::{FieldData, Shape};

const USAGE: &str = "\
usage:
  entelechy-export <file>... --out <target> [--rate <hz>] [--beats <n>]
              [--filter linear|nearest] [--joints <name>,<name>,...]
              [--double-sided] [--name <title|family>]
              [--chars <text> | --chars-file <file>]
              [--web <dir>] [--runner <file>] [--lib <file>] [--app-id <id>]
              [--window <w>x<h>] [--backend gpu|jit|cpu|best]
              [--hold-pointer | --lock-pointer]
      To .glb: a vertices .field (rows of position, normal, uv, and a
      tangent where a normal map is read along one), a triangles .field
      (rows of three ints), and PNGs: the base color, a normal map named
      <stem>.normal.png, an occlusion map named <stem>.occlusion.png and an
      ORM map (occlusion, roughness, metalness) named <stem>.orm.png;
      a rig's weights, skeleton and clip
      .fields beside them write a skin and an animation apiece, --rate naming
      the frames a second the clips were sampled at (30 by default).
      --filter says how the PNG is sampled: linear (blended, with mipmaps,
      the default) or nearest (flat texels, for pixel art and palettes).
      --joints names the skeleton's joints in its rows' order, one name a
      row; without it they are joint0, joint1 and on. --double-sided
      draws and lights both sides of each triangle (a leaf, a sheet).
      To .wav: a .field of one or two floats per row; --rate says its
      frames per second, 48000 by default (ffmpeg makes an MP3 or OGG of the
      WAV). To .png: a .field grid of pixels or floats, or a .ppm or .pgm.
      To .mid: a .field of notes as the importer reads them (start, duration,
      pitch, velocity, channel); --beats says the quarter notes a minute, 120
      by default.
      To .ttf, .otf or .woff: a font's .fields as the importer writes them
      (glyph rows, outline segments in ems, the measures); --name names the
      family, the file's stem by default; --chars (or --chars-file, a UTF-8
      file) keeps those characters alone.
      To .bundle: a scene (.esn) compiled into one standalone binary bundle
      (its validated IR modules, lowered Graph, baked Field inputs and compact
      font atlases, subsetted by --chars / --chars-file as for .html).
      To .html: a scene (.esn) or precompiled bundle (.bundle) as one page that
      plays it, opened from the disk with nothing beside it: the Player in a
      page, Entelechy's bundle runtime, and the scene's binary bundle. --web is
      where Entelechy for the web is (the directory holding bridges/web and
      target/page; the current directory by default).
      To .app, .exe, .bin or .entapp: a scene (.esn) or bundle (.bundle) packed
      with a prebuilt `entelechy-runner` binary (--runner <file>, else beside
      entelechy-export) into a standalone native desktop application: a macOS
      .app bundle (sealing the Mach-O signature and CodeResources), a
      single-file Windows .exe, a single-file Unix .bin, or a standalone
      .entapp envelope. --name sets the application title, --app-id the bundle
      identifier (`run.entelechy.<slug>` by default), and --window <w>x<h> the
      initial logical window size.
      To .ios or .android: a scene (.esn) or bundle (.bundle) exported as a
      self-contained Xcode project directory (.ios) or Android NDK
      NativeActivity project directory (.android) over `entelechy.h` and
      `scene.bundle`, using `bridges/ios` or `bridges/android` under --web
      (current directory by default); --lib copies a prebuilt
      `libentelechy_ffi.a` or `libentelechy_ffi.so` into it.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    match export(&args) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn read_field(path: &Path) -> Result<FieldData, String> {
    entelechy_formats::read(path, None)
        .map(|(data, _)| data)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_window_wh(text: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = text.split('x').collect();
    let [w, h] = parts.as_slice() else {
        return Err(format!("--window `{text}` is not <w>x<h>"));
    };
    let w = w
        .parse::<u32>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("--window width `{w}` must be positive"))?;
    let h = h
        .parse::<u32>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("--window height `{h}` must be positive"))?;
    Ok((w, h))
}

fn require_single_scene<'a>(inputs: &'a [PathBuf], what: &str) -> Result<&'a Path, String> {
    let [scene] = inputs else {
        return Err(format!("{what} is made of one scene"));
    };
    let ext = scene
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if ext != "esn" {
        return Err(format!(
            "{}: {what} is made of a scene (.esn)",
            scene.display()
        ));
    }
    Ok(scene.as_path())
}

fn require_scene_or_bundle<'a>(inputs: &'a [PathBuf], what: &str) -> Result<&'a Path, String> {
    let [scene] = inputs else {
        return Err(format!(
            "{what} is made of one scene (.esn) or bundle (.bundle)"
        ));
    };
    let ext = scene
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !matches!(ext.as_str(), "esn" | "bundle") {
        return Err(format!(
            "{}: {what} is made of a scene (.esn) or compiled bundle (.bundle)",
            scene.display()
        ));
    }
    Ok(scene.as_path())
}

fn append_fonts(said: &mut String, fonts: &[(String, usize, usize)]) {
    for (name, read, kept) in fonts {
        said.push_str(&format!("\n  {name}: {read} bytes carried as {kept}"));
    }
}

fn export(args: &[String]) -> Result<String, String> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut rate: Option<u32> = None;
    let mut beats: Option<u32> = None;
    let mut options = gltf::Options::default();
    let mut family: Option<String> = None;
    let mut chars: Option<String> = None;
    let mut web: Option<PathBuf> = None;
    let mut runner: Option<PathBuf> = None;
    let mut lib: Option<PathBuf> = None;
    let mut app_id: Option<String> = None;
    let mut window: Option<(u32, u32)> = None;
    let mut backend: Option<String> = None;
    let mut pointer = page::PointerMode::Free;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a file")?)),
            "--web" => web = Some(PathBuf::from(iter.next().ok_or("--web needs a directory")?)),
            "--runner" => runner = Some(PathBuf::from(iter.next().ok_or("--runner needs a file")?)),
            "--lib" => lib = Some(PathBuf::from(iter.next().ok_or("--lib needs a file")?)),
            "--app-id" => app_id = Some(iter.next().ok_or("--app-id needs an identifier")?.clone()),
            "--window" => {
                window = Some(parse_window_wh(
                    iter.next().ok_or("--window needs <w>x<h>")?,
                )?)
            }
            "--backend" => {
                backend = match iter.next().map(String::as_str) {
                    Some(name @ ("gpu" | "jit" | "cpu" | "best")) => Some(name.to_string()),
                    _ => return Err("--backend is gpu, jit, cpu or best".into()),
                }
            }
            "--rate" => {
                rate = Some(
                    iter.next()
                        .ok_or("--rate needs a number")?
                        .parse()
                        .map_err(|_| "--rate needs a whole number")?,
                )
            }
            "--beats" => {
                beats = Some(
                    iter.next()
                        .ok_or("--beats needs a number")?
                        .parse()
                        .map_err(|_| "--beats needs a whole number")?,
                )
            }
            "--filter" => {
                options.filter = match iter.next().map(String::as_str) {
                    Some("linear") => gltf::Filter::Linear,
                    Some("nearest") => gltf::Filter::Nearest,
                    Some(other) => {
                        return Err(format!("--filter is linear or nearest, not `{other}`"));
                    }
                    None => return Err("--filter needs linear or nearest".into()),
                }
            }
            "--joints" => {
                options.joints = iter
                    .next()
                    .ok_or("--joints needs names, split by commas")?
                    .split(',')
                    .map(|name| name.trim().to_owned())
                    .collect();
            }
            "--double-sided" => options.double_sided = true,
            "--name" => family = Some(iter.next().ok_or("--name needs a family name")?.clone()),
            "--chars" => chars
                .get_or_insert_with(String::new)
                .push_str(iter.next().ok_or("--chars needs the characters")?),
            "--chars-file" => {
                let path = iter.next().ok_or("--chars-file needs a file")?;
                chars
                    .get_or_insert_with(String::new)
                    .push_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?);
            }
            "--hold-pointer" => pointer = page::PointerMode::Hold,
            "--lock-pointer" => pointer = page::PointerMode::Lock,
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => inputs.push(PathBuf::from(other)),
        }
    }
    let out = out.ok_or("--out names the file to write")?;
    if inputs.is_empty() {
        return Err("name the file or files to export".into());
    }
    let extension = |path: &Path| {
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default()
    };
    let ext = extension(&out);
    let is_scene_app = matches!(
        ext.as_str(),
        "html" | "app" | "exe" | "bin" | "entapp" | "ios" | "android"
    );
    if family.is_some() && !matches!(ext.as_str(), "ttf" | "otf" | "woff") && !is_scene_app {
        return Err("--name is for a font, a page, or an application".into());
    }
    if chars.is_some()
        && !matches!(ext.as_str(), "ttf" | "otf" | "woff" | "bundle")
        && !is_scene_app
    {
        return Err(
            "--chars and --chars-file are for a font, a bundle, a page, or an application".into(),
        );
    }
    if web.is_some() && !matches!(ext.as_str(), "html" | "ios" | "android") {
        return Err("--web is for a page (.html) or mobile project (.ios, .android)".into());
    }
    if backend.is_some() && !is_scene_app {
        return Err("--backend is for a page or an application".into());
    }
    if pointer != page::PointerMode::Free && !is_scene_app {
        return Err("--hold-pointer and --lock-pointer are for a page or an application".into());
    }
    if runner.is_some() && !matches!(ext.as_str(), "app" | "exe" | "bin") {
        return Err("--runner is for a desktop application (.app, .exe, .bin)".into());
    }
    if lib.is_some() && !matches!(ext.as_str(), "ios" | "android") {
        return Err("--lib is for a mobile project (.ios, .android)".into());
    }
    if app_id.is_some() && !matches!(ext.as_str(), "app" | "ios" | "android") {
        return Err("--app-id is for .app, .ios, or .android".into());
    }
    if window.is_some() && !matches!(ext.as_str(), "app" | "exe" | "bin" | "entapp") {
        return Err("--window is for a desktop application (.app, .exe, .bin, .entapp)".into());
    }

    let native_opts = native::Options {
        runner: runner.as_deref(),
        bridge_root: web.as_deref(),
        lib: lib.as_deref(),
        backend: backend.as_deref(),
        chars: chars.as_deref(),
        pointer,
        title: family.as_deref(),
        window,
        app_id: app_id.as_deref(),
    };

    match ext.as_str() {
        "bundle" => {
            let scene = require_single_scene(&inputs, "a bundle")?;
            let compiled = page::compile(scene, chars.as_deref())?;
            std::fs::write(&out, &compiled.bundle)
                .map_err(|e| format!("{}: {e}", out.display()))?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s))",
                out.display(),
                compiled.bundle.len(),
                compiled.modules,
                compiled.inputs,
            );
            append_fonts(&mut said, &compiled.fonts);
            Ok(said)
        }
        "html" => {
            let scene = require_scene_or_bundle(&inputs, "a page")?;
            let made = page::page_with_options(
                scene,
                web.as_deref().unwrap_or(Path::new(".")),
                backend.as_deref(),
                chars.as_deref(),
                pointer,
                family.as_deref(),
            )?;
            std::fs::write(&out, &made.page).map_err(|e| format!("{}: {e}", out.display()))?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s), bundle {} bytes, Entelechy for a page {} bytes)",
                out.display(),
                made.page.len(),
                made.modules,
                made.inputs,
                made.bundle_bytes,
                made.runtime_bytes
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "app" => {
            let scene = require_scene_or_bundle(&inputs, "a macOS .app")?;
            let made = native::macos_app(scene, &out, &native_opts)?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s), bundle {} bytes, runner {} bytes)",
                out.display(),
                made.total_bytes,
                made.modules,
                made.inputs,
                made.bundle_bytes,
                made.runner_bytes
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "exe" => {
            let scene = require_scene_or_bundle(&inputs, "a Windows .exe")?;
            let made =
                native::packed_executable(scene, &out, native::ExeKind::WindowsPe, &native_opts)?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s), bundle {} bytes, runner {} bytes)",
                out.display(),
                made.total_bytes,
                made.modules,
                made.inputs,
                made.bundle_bytes,
                made.runner_bytes
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "bin" => {
            let scene = require_scene_or_bundle(&inputs, "a standalone binary")?;
            let made =
                native::packed_executable(scene, &out, native::ExeKind::UnixBin, &native_opts)?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s), bundle {} bytes, runner {} bytes)",
                out.display(),
                made.total_bytes,
                made.modules,
                made.inputs,
                made.bundle_bytes,
                made.runner_bytes
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "entapp" => {
            let scene = require_scene_or_bundle(&inputs, "a standalone envelope")?;
            let made = native::standalone_envelope(scene, &out, &native_opts)?;
            let mut said = format!(
                "{}: {} bytes ({} Kernel(s), {} input(s), bundle {} bytes)",
                out.display(),
                made.total_bytes,
                made.modules,
                made.inputs,
                made.bundle_bytes
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "ios" => {
            let scene = require_scene_or_bundle(&inputs, "an iOS project")?;
            let made = native::ios_project(scene, &out, &native_opts)?;
            let mut said = format!(
                "{}: Xcode project ({} Kernel(s), {} input(s), bundle {} bytes{})",
                out.display(),
                made.modules,
                made.inputs,
                made.bundle_bytes,
                if made.lib_copied {
                    ", libentelechy_ffi.a included"
                } else {
                    ""
                }
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "android" => {
            let scene = require_scene_or_bundle(&inputs, "an Android project")?;
            let made = native::android_project(scene, &out, &native_opts)?;
            let mut said = format!(
                "{}: Android NDK project ({} Kernel(s), {} input(s), bundle {} bytes{})",
                out.display(),
                made.modules,
                made.inputs,
                made.bundle_bytes,
                if made.lib_copied {
                    ", libentelechy_ffi.so included"
                } else {
                    ""
                }
            );
            append_fonts(&mut said, &made.fonts);
            Ok(said)
        }
        "glb" => {
            let mut vertices = None;
            let mut triangles = None;
            let mut textures: Vec<PathBuf> = Vec::new();
            let mut rig: Vec<PathBuf> = Vec::new();
            let mut clips: Vec<PathBuf> = Vec::new();
            for input in &inputs {
                match extension(input).as_str() {
                    "png" => textures.push(input.clone()),
                    "field" => {
                        let data = read_field(input)?;
                        let part = (data.domain().shape() == Shape::Rows)
                            .then(|| Part::of(data.schema()))
                            .flatten();
                        match part {
                            Some(Part::Weights | Part::Skeleton) => rig.push(input.clone()),
                            Some(Part::Vertices | Part::TangentVertices) => {
                                vertices = Some(input.clone())
                            }
                            Some(Part::Triangles) => triangles = Some(input.clone()),
                            Some(Part::Clip) => clips.push(input.clone()),
                            None => {
                                return Err(format!(
                                    "{}: no part of a mesh or a rig",
                                    input.display()
                                ));
                            }
                        }
                    }
                    other => {
                        return Err(format!(
                            "{}: `.{other}` has no place in a GLB",
                            input.display()
                        ));
                    }
                }
            }
            let vertices =
                vertices.ok_or("a GLB needs a vertices .field (rows of eight or twelve floats)")?;
            let triangles =
                triangles.ok_or("a GLB needs a triangles .field (rows of three ints)")?;
            let rig_paths: Vec<&Path> = rig.iter().map(PathBuf::as_path).collect();
            let clip_paths: Vec<&Path> = clips.iter().map(PathBuf::as_path).collect();
            let texture_paths: Vec<&Path> = textures.iter().map(PathBuf::as_path).collect();
            gltf::export_rig(
                &vertices,
                &triangles,
                &texture_paths,
                &rig_paths,
                &clip_paths,
                rate.unwrap_or(30) as f32,
                &out,
                &options,
            )?;
            let extra = if rig_paths.is_empty() {
                String::new()
            } else {
                format!(
                    ", with a skin and {} clip(s) at {} frames a second",
                    clip_paths.len(),
                    rate.unwrap_or(30)
                )
            };
            Ok(format!("wrote {}{extra}", out.display()))
        }
        format @ ("ttf" | "otf" | "woff") => {
            let fields = inputs
                .iter()
                .map(|input| read_field(input))
                .collect::<Result<Vec<_>, _>>()?;
            let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("Font");
            let face = font::face(&fields, family.as_deref().unwrap_or(stem), chars.as_deref())?;
            let bytes = match format {
                "ttf" => font::ttf(&face),
                "otf" => font::otf(&face),
                _ => font::woff(&face),
            };
            std::fs::write(&out, &bytes).map_err(|e| format!("{}: {e}", out.display()))?;
            Ok(format!(
                "wrote {}: {} character(s), {} glyph(s), {} bytes",
                out.display(),
                face.characters(),
                face.glyphs(),
                bytes.len()
            ))
        }
        format @ ("wav" | "mid" | "midi" | "png") => {
            if options.filter != gltf::Filter::Linear
                || !options.joints.is_empty()
                || options.double_sided
            {
                return Err("--filter, --joints and --double-sided are for a .glb".into());
            }
            let [input] = inputs.as_slice() else {
                return Err(format!("a .{format} comes from one file"));
            };
            let data = read_field(input)?;
            let rate = match format {
                "wav" => rate.or_else(|| {
                    let bytes = std::fs::read(input).ok()?;
                    entelechy_formats::wav::decode(&bytes)
                        .ok()
                        .map(|sound| sound.rate)
                }),
                "png" => None,
                _ => beats,
            };
            let bytes = entelechy_formats::encode(&data, format, rate, None)
                .map_err(|why| format!("{}: {why}", input.display()))?;
            std::fs::write(&out, bytes).map_err(|e| format!("{}: {e}", out.display()))?;
            let domain = data.domain();
            Ok(match format {
                "wav" => format!(
                    "wrote {}: {} frames, {} channel(s) at {} Hz",
                    out.display(),
                    domain.count(),
                    data.schema().value_words(),
                    rate.unwrap_or(entelechy_formats::DEFAULT_RATE)
                ),
                "png" => format!(
                    "wrote {}: {} by {}",
                    out.display(),
                    domain.width(),
                    domain.height()
                ),
                _ => {
                    let values = data.values();
                    let notes = values
                        .chunks_exact(entelechy_formats::midi::schema().value_words())
                        .map(entelechy_formats::midi::note);
                    let seconds = notes
                        .map(|note| note.start + note.duration)
                        .fold(0.0f32, f32::max);
                    format!(
                        "wrote {}: {} note(s) over {seconds:.2} s at {} beats a minute",
                        out.display(),
                        domain.count(),
                        beats.map_or(entelechy_formats::DEFAULT_BEATS, |beats| beats as f32)
                    )
                }
            })
        }
        other => Err(format!(
            "`.{other}` is not a format export knows; write .glb, .wav, .mid, .png, .ttf, .otf, .woff, .bundle, .html, .app, .exe, .bin, .ios or .android"
        )),
    }
}
