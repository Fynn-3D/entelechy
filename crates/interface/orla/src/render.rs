//! Recording a session's frames, and `render`, which steps a scene at a
//! fixed rate for a count of frames and records them.
//!
//! A recording goes where it is told: a directory holds each frame's
//! picture as a PNG and the sound, block after block, as one WAV beside
//! them; a `.wav` file holds the sound alone. The picture is the first
//! pixel-grid output and the sound the first output of rows of one or
//! two floats, as in the Player. A render's clock is frame count over
//! rate, so a render is deterministic; `async` nodes are waited for
//! every frame. A replay records the frames its steps evaluate.

use std::path::{Path, PathBuf};

use orla_graph::Report;
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
    /// Where the recording goes (see [`Target::of`]).
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

/// Where a recording goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// A directory: each frame's picture, and the sound as
    /// [`SOUND_NAME`] beside them.
    Directory(PathBuf),
    /// A WAV file: the sound alone.
    Sound(PathBuf),
}

impl Target {
    /// The target a path names: a `.wav` file, or else a directory.
    pub fn of(path: &Path) -> Self {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
        {
            Target::Sound(path.to_path_buf())
        } else {
            Target::Directory(path.to_path_buf())
        }
    }

    /// Where the sound is written.
    pub fn sound_path(&self) -> PathBuf {
        match self {
            Target::Directory(dir) => dir.join(SOUND_NAME),
            Target::Sound(path) => path.clone(),
        }
    }
}

/// A recording in progress: each evaluated frame's picture written as it
/// comes, its sound gathered and written once it is finished.
pub struct Recorder {
    target: Target,
    sound_rate: u32,
    summary: Summary,
    sound: Vec<u32>,
}

impl Recorder {
    /// A recording into `target`, its sound written at `sound_rate`; a
    /// directory is created if missing.
    pub fn new(target: Target, sound_rate: u32) -> Result<Self, String> {
        if let Target::Directory(dir) = &target {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        Ok(Self {
            target,
            sound_rate,
            summary: Summary::default(),
            sound: Vec::new(),
        })
    }

    /// Records the frame `report` evaluated: its picture, into a
    /// directory, and its sound block.
    pub fn frame(&mut self, session: &mut Session, report: &Report) -> Result<(), String> {
        let block = session.sound_of(report);
        let picture = match &self.target {
            Target::Directory(dir) => session
                .picture_of(report)
                .map(|handle| (dir.clone(), handle)),
            Target::Sound(_) => None,
        };
        match &self.target {
            Target::Directory(_) if picture.is_none() && block.is_none() => {
                return Err("the scene has no picture output and no sound output".into());
            }
            Target::Sound(_) if block.is_none() => {
                return Err("the scene has no sound output".into());
            }
            _ => {}
        }
        if let Some((dir, handle)) = picture {
            let (width, height, pixels) = pixels_of(session, handle)?;
            let path = dir.join(picture_name(self.summary.pictures));
            std::fs::write(&path, png::encode(width, height, &pixels))
                .map_err(|e| format!("{}: {e}", path.display()))?;
            self.summary.pictures += 1;
            self.summary.size.get_or_insert((width, height));
        }
        if let Some((handle, channels)) = block {
            let words = words_of(session, handle)?;
            self.summary.sound_frames += words.len() / channels;
            self.summary.channels = channels;
            self.sound.extend(words);
        }
        Ok(())
    }

    /// Writes the sound gathered, and says what was recorded.
    pub fn finish(self) -> Result<Summary, String> {
        if self.summary.channels > 0 {
            let path = self.target.sound_path();
            std::fs::write(
                &path,
                wav::encode(&self.sound, self.summary.channels, self.sound_rate),
            )
            .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(self.summary)
    }
}

/// Renders `options.frames` frames into `options.out`, calling
/// `progress` with each recorded frame's index and report.
pub fn render(
    session: &mut Session,
    options: &Options,
    mut progress: impl FnMut(u32, &orla_graph::Report),
) -> Result<Summary, String> {
    let mut recorder = Recorder::new(Target::of(&options.out), options.sound_rate)?;
    let (size, scale) = options.window.unwrap_or(([0.0, 0.0], 1.0));
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
        recorder
            .frame(session, &report)
            .map_err(|e| format!("frame {index}: {e}"))?;
        session
            .settle()
            .map_err(|e| format!("frame {index}: {e}"))?;
        progress(index, &report);
    }
    recorder.finish()
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
