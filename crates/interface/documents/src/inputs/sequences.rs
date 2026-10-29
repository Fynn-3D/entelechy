//! Directory inputs: frame files in name order, the one held chosen by
//! time.

use std::path::{Path, PathBuf};

use entelechy_kernel::Host;

use super::{Inputs, kept_as, read_within, room};
use crate::store::Store;
use crate::{Diagnostic, Event};

/// A directory input: frame files in name order and the one held now.
pub(super) struct Sequence {
    pub(super) dir: PathBuf,
    ty: String,
    rate: u32,
    files: Vec<PathBuf>,
    current: Option<usize>,
}

/// The file extensions a frame may have.
const FRAME_EXTENSIONS: [&str; 5] = ["png", "ppm", "pgm", "field", "wav"];

/// The files of `dir` that can be frames, in name order.
fn frame_files(store: &dyn Store, dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = store
        .list(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| FRAME_EXTENSIONS.contains(&e))
        })
        .collect();
    files.sort();
    Ok(files)
}

impl Inputs {
    /// Relists each `frames` directory and forgets undeclared ones. A
    /// changed listing resets; the due frame loads at the next evaluation.
    pub(super) fn sync_sequences(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let wanted: Vec<(String, PathBuf, String, u32)> = files.sequences.clone();
        let stale: Vec<String> = self
            .sequences
            .keys()
            .filter(|name| !wanted.iter().any(|(wanted, ..)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.sequences.remove(&name);
            self.forget(host, &name);
        }
        for (name, dir, ty, rate) in wanted {
            let files = match readable(&dir).and_then(|()| frame_files(store, &dir)) {
                Ok(files) => files,
                Err(error) => {
                    if self.sequences.remove(&name).is_some() {
                        self.forget(host, &name);
                    }
                    events.push(Event::Rejected {
                        path: dir,
                        error: Diagnostic::new(error.to_string()),
                    });
                    continue;
                }
            };
            if self.sequences.get(&name).is_some_and(|sequence| {
                sequence.dir == dir
                    && sequence.ty == ty
                    && sequence.rate == rate
                    && sequence.files == files
            }) {
                continue;
            }
            if files.is_empty() {
                events.push(Event::Rejected {
                    path: dir.clone(),
                    error: Diagnostic::new(
                        "holds no frame: a .png, .ppm, .pgm, .field or .wav file",
                    ),
                });
            } else {
                events.push(Event::Loaded {
                    name: name.clone(),
                    path: dir.clone(),
                    detail: format!("{} frame(s) at {rate} a second", files.len()),
                });
            }
            self.sequences.insert(
                name,
                Sequence {
                    dir,
                    ty,
                    rate,
                    files,
                    current: None,
                },
            );
        }
    }

    /// Loads frame `floor(time * rate)`, wrapping, for each `frames`
    /// input. An unreadable frame is reported by the next sync and the
    /// input keeps its frame.
    pub fn select_frames(&mut self, store: &dyn Store, host: &mut Host, time: f64) {
        let names: Vec<String> = self.sequences.keys().cloned().collect();
        for name in names {
            let Some(sequence) = self.sequences.get(&name) else {
                continue;
            };
            if sequence.files.is_empty() {
                continue;
            }
            let count = sequence.files.len() as i64;
            let index =
                ((time * f64::from(sequence.rate)).floor() as i64).rem_euclid(count) as usize;
            if sequence.current == Some(index) {
                continue;
            }
            let path = sequence.files[index].clone();
            let ty = sequence.ty.clone();
            let loaded = read_within(store, &path, room(host))
                .and_then(|found| entelechy_formats::decode(&found, &path, Some(&ty), room(host)))
                .and_then(|(data, _)| {
                    let data = kept_as(data, &ty)?;
                    host.adopt(data).map_err(|e| e.to_string())
                });
            match loaded {
                Ok(handle) => {
                    self.hold(host, &name, handle);
                    if let Some(sequence) = self.sequences.get_mut(&name) {
                        sequence.current = Some(index);
                    }
                }
                Err(error) => self.frame_events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(error.to_string()),
                }),
            }
        }
    }
}
