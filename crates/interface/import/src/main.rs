//! `orla-import`: outside files as the Host's files, outside the
//! runtime. A model becomes vertex and triangle rows and a PNG; a
//! compressed sound a WAV; a score a notes Field; a drawing the segments
//! and shapes the `vector` package draws.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use orla_import::{audio, gltf, midi, svg};

const USAGE: &str = "\
usage:
  orla-import <file> [--out <dir>] [--rate <fps>]
      A model (.gltf, .glb) becomes <stem>.vertices.field, <stem>.triangles.field
      and, when it has one in PNG, <stem>.albedo.png. A model with a skin also
      becomes <stem>.weights.field and <stem>.skeleton.field, and one
      <stem>.<name>.clip.field per animation, sampled at --rate frames a second
      (30 by default), which the `skin` package poses a mesh from; a sound (.mp3, .ogg, .flac,
      .m4a, .aac, .mka, .wav) becomes <stem>.wav at its own rate; a score (.mid,
      .midi) becomes <stem>.notes.field, a row a note with its seconds, its key,
      how hard it was struck and its channel; a drawing (.svg) becomes
      <stem>.segments.field, <stem>.shapes.field, <stem>.stops.field and
      <stem>.view.field, which the `vector` package draws. Files go to --out,
      the source's directory by default.";

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
    let mut rate = 30.0f32;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a directory")?)),
            "--rate" => {
                let text = iter.next().ok_or("--rate needs frames a second")?;
                rate = text
                    .parse::<f32>()
                    .ok()
                    .filter(|v| *v > 0.0 && v.is_finite())
                    .ok_or_else(|| format!("--rate takes frames a second, found `{text}`"))?;
            }
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
            let imported = gltf::import(&source, &out, rate)?;
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
            if imported.joints > 0 {
                let clips: Vec<String> = imported
                    .clips
                    .iter()
                    .map(|(name, frames)| format!("{name} ({frames} frames)"))
                    .collect();
                println!(
                    "{} joints, {} at {} frames a second{}",
                    imported.joints,
                    if clips.is_empty() {
                        "no clips".to_owned()
                    } else {
                        clips.join(", ")
                    },
                    imported.rate,
                    if clips.is_empty() {
                        ""
                    } else {
                        "; a scene reads a clip through the `skin` package"
                    }
                );
            }
            for note in &imported.notes {
                println!("note: {note}");
            }
        }
        "mid" | "midi" => {
            let imported = midi::import(&source, &out)?;
            println!("wrote {}", imported.file.display());
            let channels: std::collections::BTreeSet<i32> =
                imported.notes.iter().map(|note| note.channel).collect();
            let lowest = imported
                .notes
                .iter()
                .map(|note| note.pitch)
                .min()
                .unwrap_or(0);
            let highest = imported
                .notes
                .iter()
                .map(|note| note.pitch)
                .max()
                .unwrap_or(0);
            println!(
                "{} note(s) over {:.2} s from {} track(s), keys {lowest} to {highest}, channel(s) {}",
                imported.notes.len(),
                imported.seconds,
                imported.tracks,
                channels
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            for note in &imported.left_out {
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
        "svg" => {
            let imported = svg::import(&source, &out)?;
            for file in &imported.files {
                println!("wrote {}", file.display());
            }
            println!(
                "{} shape(s) of {} segment(s), {} by {} units",
                imported.shapes, imported.segments, imported.size.0, imported.size.1
            );
            for note in &imported.notes {
                println!("note: {note}");
            }
        }
        other => {
            return Err(format!(
                "`.{other}` is not a format import knows; models are .gltf/.glb, sounds .mp3/.ogg/.flac/.m4a/.aac/.mka/.wav, scores .mid, drawings .svg"
            ));
        }
    }
    Ok(())
}
