//! `orla-export`: the Host's files as other software's formats. Mesh
//! rows (and a PNG) become a GLB, rows of one or two floats a WAV, a
//! grid a PNG.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use orla_export::gltf;
use orla_formats::rig::Part;
use orla_host::{FieldData, Shape};

const USAGE: &str = "\
usage:
  orla-export <file>... --out <target> [--rate <hz>] [--beats <n>]
      To .glb: a vertices .field (rows of position, normal, uv), a triangles
      .field (rows of three ints), and a .png for the base color if you have
      .field and, when it has one, a PNG; a rig's weights, skeleton and clip
      .fields beside them write a skin and an animation apiece, --rate naming
      the frames a second the clips were sampled at (30 by default).
      To .wav: a .field of one or two floats per row; --rate says its
      frames per second, 48000 by default (ffmpeg makes an MP3 or OGG of the
      WAV). To .png: a .field grid of pixels or floats, or a .ppm or .pgm.
      To .mid: a .field of notes as the importer reads them (start, duration,
      pitch, velocity, channel); --beats says the quarter notes a minute, 120
      by default.";

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
    orla_formats::read(path, None)
        .map(|(data, _)| data)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn export(args: &[String]) -> Result<String, String> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut rate: Option<u32> = None;
    let mut beats: Option<u32> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a file")?)),
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
    match extension(&out).as_str() {
        "glb" => {
            let mut vertices = None;
            let mut triangles = None;
            let mut texture = None;
            let mut rig: Vec<PathBuf> = Vec::new();
            let mut clips: Vec<PathBuf> = Vec::new();
            for input in &inputs {
                match extension(input).as_str() {
                    "png" => texture = Some(input.clone()),
                    "field" => {
                        let data = read_field(input)?;
                        // A rig's rows are told apart by layout.
                        let part = (data.domain().shape() == Shape::Rows)
                            .then(|| Part::of(data.schema()))
                            .flatten();
                        match part {
                            Some(Part::Weights | Part::Skeleton) => rig.push(input.clone()),
                            Some(Part::Vertices) => vertices = Some(input.clone()),
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
                vertices.ok_or("a GLB needs a vertices .field (rows of eight floats)")?;
            let triangles =
                triangles.ok_or("a GLB needs a triangles .field (rows of three ints)")?;
            let rig_paths: Vec<&Path> = rig.iter().map(PathBuf::as_path).collect();
            let clip_paths: Vec<&Path> = clips.iter().map(PathBuf::as_path).collect();
            gltf::export_rig(
                &vertices,
                &triangles,
                texture.as_deref(),
                &rig_paths,
                &clip_paths,
                // 30 frames a second unless given, as `orla import`.
                rate.unwrap_or(30) as f32,
                &out,
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
        format @ ("wav" | "mid" | "midi" | "png") => {
            let [input] = inputs.as_slice() else {
                return Err(format!("a .{format} comes from one file"));
            };
            let data = read_field(input)?;
            let rate = match format {
                // A WAV input brings its rate; a .field takes the given one.
                "wav" => rate.or_else(|| {
                    let bytes = std::fs::read(input).ok()?;
                    orla_formats::wav::decode(&bytes)
                        .ok()
                        .map(|sound| sound.rate)
                }),
                "png" => None,
                _ => beats,
            };
            let bytes = orla_formats::encode(&data, format, rate)
                .map_err(|why| format!("{}: {why}", input.display()))?;
            std::fs::write(&out, bytes).map_err(|e| format!("{}: {e}", out.display()))?;
            let domain = data.domain();
            Ok(match format {
                "wav" => format!(
                    "wrote {}: {} frames, {} channel(s) at {} Hz",
                    out.display(),
                    domain.count(),
                    data.schema().value_words(),
                    rate.unwrap_or(orla_formats::DEFAULT_RATE)
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
                        .chunks_exact(orla_formats::midi::schema().value_words())
                        .map(orla_formats::midi::note);
                    let seconds = notes
                        .map(|note| note.start + note.duration)
                        .fold(0.0f32, f32::max);
                    format!(
                        "wrote {}: {} note(s) over {seconds:.2} s at {} beats a minute",
                        out.display(),
                        domain.count(),
                        beats.map_or(orla_formats::DEFAULT_BEATS, |beats| beats as f32)
                    )
                }
            })
        }
        other => Err(format!(
            "`.{other}` is not a format export knows; write .glb, .wav, .mid, or .png"
        )),
    }
}
