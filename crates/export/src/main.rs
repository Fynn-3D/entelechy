//! `orla-export`: writes the Host's files as formats other software
//! opens. A mesh's vertex and triangle rows (and a PNG) become a GLB;
//! rows of one or two floats become a WAV; a grid of pixels or floats
//! becomes a PNG.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use orla_export::{gltf, midi, sound};
use orla_runtime::{FieldData, Shape};

const USAGE: &str = "\
usage:
  orla-export <file>... --out <target> [--rate <hz>]
      To .glb: a vertices .field (rows of position, normal, uv), a triangles
      .field (rows of three ints), and a .png for the base colour if you have
      one. To .wav: a .field of one or two floats per row; --rate says its
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
    orla::session::Session::load_field(path, None)
        .map(|(data, _)| data)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn export(args: &[String]) -> Result<String, String> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut rate: Option<u32> = None;
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
            for input in &inputs {
                match extension(input).as_str() {
                    "png" => texture = Some(input.clone()),
                    "field" => {
                        let data = read_field(input)?;
                        match (data.domain().shape(), data.schema().words()) {
                            (Shape::Rows, 8) => vertices = Some(input.clone()),
                            (Shape::Rows, 3) => triangles = Some(input.clone()),
                            _ => {
                                return Err(format!(
                                    "{}: neither vertex rows nor triangle rows",
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
            gltf::export(&vertices, &triangles, texture.as_deref(), &out)?;
            Ok(format!("wrote {}", out.display()))
        }
        "wav" => {
            if inputs.len() != 1 {
                return Err("a sound comes from one file".into());
            }
            let (data, detail) = orla::session::Session::load_field(&inputs[0], None)
                .map_err(|e| format!("{}: {e}", inputs[0].display()))?;
            let channels = data.schema().words();
            if data.domain().shape() != Shape::Rows || channels > 2 || data.schema().is_pixel() {
                return Err(format!(
                    "{}: not rows of one or two floats",
                    inputs[0].display()
                ));
            }
            // A WAV brings its rate; a .field is given one.
            let file_rate = detail
                .rsplit_once(" at ")
                .and_then(|(_, tail)| tail.strip_suffix(" Hz"))
                .and_then(|hz| hz.parse::<u32>().ok());
            let rate = rate.or(file_rate).unwrap_or(48_000);
            let samples: Vec<f32> = data.words().iter().map(|w| f32::from_bits(*w)).collect();
            let pcm = sound::Pcm {
                samples: &samples,
                channels,
                rate,
            };
            std::fs::write(&out, sound::wav(&pcm))
                .map_err(|e| format!("{}: {e}", out.display()))?;
            Ok(format!(
                "wrote {}: {} frames, {channels} channel(s) at {rate} Hz",
                out.display(),
                data.domain().count()
            ))
        }
        "mid" | "midi" => {
            if inputs.len() != 1 {
                return Err("a score comes from one file".into());
            }
            let data = read_field(&inputs[0])?;
            let words = data.schema().words();
            if data.domain().shape() != Shape::Rows || words != 5 {
                return Err(format!(
                    "{}: not rows of a note (start, duration, pitch, velocity, channel)",
                    inputs[0].display()
                ));
            }
            let notes: Vec<midi::Note> = data
                .words()
                .chunks_exact(words)
                .map(|note| midi::Note {
                    start: f32::from_bits(note[0]),
                    duration: f32::from_bits(note[1]),
                    pitch: note[2] as i32,
                    velocity: f32::from_bits(note[3]),
                    channel: note[4] as i32,
                })
                .collect();
            let beats = rate.map(|beats| beats as f32).unwrap_or(120.0);
            std::fs::write(&out, midi::midi(&notes, beats))
                .map_err(|e| format!("{}: {e}", out.display()))?;
            let seconds = notes
                .iter()
                .map(|note| note.start + note.duration)
                .fold(0.0f32, f32::max);
            Ok(format!(
                "wrote {}: {} note(s) over {seconds:.2} s at {beats} beats a minute",
                out.display(),
                notes.len()
            ))
        }
        "png" => {
            if inputs.len() != 1 {
                return Err("a picture comes from one file".into());
            }
            let data = read_field(&inputs[0])?;
            let domain = data.domain();
            if domain.shape() != Shape::Grid {
                return Err(format!("{}: not a grid", inputs[0].display()));
            }
            let pixels: Vec<[f32; 4]> = if data.schema().is_pixel() {
                data.words()
                    .chunks_exact(4)
                    .map(|p| {
                        [
                            f32::from_bits(p[0]),
                            f32::from_bits(p[1]),
                            f32::from_bits(p[2]),
                            f32::from_bits(p[3]),
                        ]
                    })
                    .collect()
            } else if data.schema().words() == 1 {
                data.words()
                    .iter()
                    .map(|w| {
                        let v = f32::from_bits(*w);
                        [v, v, v, 1.0]
                    })
                    .collect()
            } else {
                return Err(format!(
                    "{}: neither pixels nor a Grid<Float>",
                    inputs[0].display()
                ));
            };
            std::fs::write(
                &out,
                orla::png::encode(domain.width(), domain.height(), &pixels),
            )
            .map_err(|e| format!("{}: {e}", out.display()))?;
            Ok(format!(
                "wrote {}: {} by {}",
                out.display(),
                domain.width(),
                domain.height()
            ))
        }
        other => Err(format!(
            "`.{other}` is not a format export knows; write .glb, .wav, or .png"
        )),
    }
}
