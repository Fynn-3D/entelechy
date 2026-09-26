//! `render`: steps a scene at a fixed rate for a count of frames, each
//! picture a PNG in a directory and the sound, block after block, one
//! WAV beside them.
//!
//! The clock is frame count over rate, so a render is deterministic;
//! `async` nodes are waited for every frame. The picture is the first
//! pixel-grid output and the sound the first output of rows of one or
//! two floats, as in the Player.

use std::path::{Path, PathBuf};

use orla_kernel::{Context, Handle};

use orla_formats::{png, wav};
use orla_session::Session;

/// How to render.
#[derive(Clone, Debug)]
pub struct Options {
    /// How many frames.
    pub frames: u32,
    /// Frames per second: frame `i` is at time `i / rate`.
    pub rate: f32,
    /// The sound clock Kernels read as `rate()`, and the WAV's rate.
    pub sound_rate: u32,
    /// The directory the files go into; created if missing.
    pub out: PathBuf,
    /// The window a scene sized by `window()` fills, and its scale.
    pub window: Option<([f32; 2], f32)>,
}

/// What a render wrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Frames written as pictures.
    pub pictures: u32,
    /// The picture's size, when there was one.
    pub size: Option<(u32, u32)>,
    /// Sound frames written, over all the blocks.
    pub sound_frames: usize,
    /// The sound's channels, 1 or 2, when there was a sound.
    pub channels: usize,
}

/// The file name of frame `index`.
pub fn picture_name(index: u32) -> String {
    format!("{index:04}.png")
}

/// The sound's file name.
pub const SOUND_NAME: &str = "sound.wav";

/// Renders `options.frames` frames into `options.out`, calling
/// `progress` with each written frame's index.
pub fn render(
    session: &mut Session,
    options: &Options,
    mut progress: impl FnMut(u32),
) -> Result<Summary, String> {
    std::fs::create_dir_all(&options.out).map_err(|e| format!("{}: {e}", options.out.display()))?;
    let (size, scale) = options.window.unwrap_or(([0.0, 0.0], 1.0));
    let mut summary = Summary::default();
    let mut sound: Vec<u32> = Vec::new();
    for index in 0..options.frames {
        let context = Context {
            time: index as f32 / options.rate,
            rate: options.sound_rate as f32,
            window: size,
            scale,
            ..Context::default()
        };
        let report = session
            .evaluate(&context)
            .map_err(|e| format!("frame {index}: {e}"))?;
        let picture = session.picture_of(&report);
        let block = session.sound_of(&report);
        if picture.is_none() && block.is_none() {
            return Err("the scene has no picture output and no sound output".into());
        }
        if let Some(handle) = picture {
            let (width, height, pixels) = pixels_of(session, handle)?;
            let path = options.out.join(picture_name(index));
            std::fs::write(&path, png::encode(width, height, &pixels))
                .map_err(|e| format!("{}: {e}", path.display()))?;
            summary.pictures += 1;
            summary.size.get_or_insert((width, height));
        }
        if let Some((handle, channels)) = block {
            let words = words_of(session, handle)?;
            summary.sound_frames += words.len() / channels;
            summary.channels = channels;
            sound.extend(words);
        }
        session
            .settle()
            .map_err(|e| format!("frame {index}: {e}"))?;
        progress(index);
    }
    if summary.channels > 0 {
        let path = options.out.join(SOUND_NAME);
        std::fs::write(
            &path,
            wav::encode(&sound, summary.channels, options.sound_rate),
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(summary)
}

fn words_of(session: &mut Session, handle: Handle) -> Result<Vec<u32>, String> {
    Ok(session
        .field(handle)
        .map_err(|e| e.to_string())?
        .values()
        .to_vec())
}

fn pixels_of(session: &mut Session, handle: Handle) -> Result<(u32, u32, Vec<[f32; 4]>), String> {
    let image = session.picture(handle).map_err(|e| e.to_string())?;
    Ok((image.width(), image.height(), image.pixels().to_vec()))
}

/// The path of a directory's frame `index`, for callers that read back.
pub fn picture_path(dir: &Path, index: u32) -> PathBuf {
    dir.join(picture_name(index))
}
