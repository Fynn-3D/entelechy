//! `orla-import`: brings a file in as the Host's files, from outside the
//! runtime. A model becomes vertex and triangle rows and a PNG; a
//! compressed sound becomes a WAV. The runtime learns no format here; a
//! scene reads what this writes as inputs.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod audio;
mod gltf;

const USAGE: &str = "\
usage:
  orla-import <file> [--out <dir>]
      A model (.gltf, .glb) becomes <stem>.vertices.field, <stem>.triangles.field
      and, when it has one in PNG, <stem>.albedo.png; a sound (.mp3, .ogg, .flac,
      .m4a, .aac, .mka, .wav) becomes <stem>.wav at its own rate. Files go to
      --out, the source's directory by default.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let result = import(&args);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn import(args: &[String]) -> Result<(), String> {
    let mut source = None;
    let mut out = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a directory")?)),
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if source.replace(PathBuf::from(other)).is_some() {
                    return Err("import takes one file".into());
                }
            }
        }
    }
    let source = source.ok_or("import needs a file")?;
    let out = out.unwrap_or_else(|| {
        source
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    let extension = source
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match extension.as_str() {
        "gltf" | "glb" => {
            let imported = gltf::import(&source, &out)?;
            for file in &imported.files {
                println!("wrote {}", file.display());
            }
            let (lo, hi) = imported.bounds;
            println!(
                "{} vertices, {} triangles, bounds ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3}){}",
                imported.vertices,
                imported.triangles,
                lo[0],
                lo[1],
                lo[2],
                hi[0],
                hi[1],
                hi[2],
                if imported.texture.is_some() {
                    ", texture in PNG"
                } else {
                    ", no texture"
                }
            );
            for note in &imported.notes {
                println!("note: {note}");
            }
        }
        "mp3" | "ogg" | "oga" | "flac" | "m4a" | "aac" | "mka" | "mkv" | "wav" => {
            let imported = audio::import(&source, &out)?;
            println!(
                "wrote {}: {} frames, {} channel(s) at {} Hz",
                imported.file.display(),
                imported.frames,
                imported.channels,
                imported.rate
            );
        }
        other => {
            return Err(format!(
                "`.{other}` is not a format import knows; models are .gltf/.glb, sounds .mp3/.ogg/.flac/.m4a/.aac/.mka/.wav"
            ));
        }
    }
    Ok(())
}
