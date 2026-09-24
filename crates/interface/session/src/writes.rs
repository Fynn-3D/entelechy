//! The files a scene writes: `output name <- source to "file";`. A
//! Kernel never writes a file. It computes a Field, and once an
//! evaluation that changed the output has committed, the session writes
//! it, into a directory the person running the session allowed and
//! nowhere else. An evaluation that fails writes nothing, and a file is
//! replaced whole or not at all.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Component, Path, PathBuf};

use orla_graph::{Graph, Report};
use orla_runtime::{Handle, Runtime};

use crate::Event;

/// The directories writes may land in, and what was last written.
#[derive(Default)]
pub(crate) struct Writes {
    /// Allowed directories, resolved.
    allowed: Vec<PathBuf>,
    /// Per file: the Field last written, and a digest of its bytes.
    written: BTreeMap<PathBuf, (Handle, u64)>,
    /// Refusals already said, so each is said once until it changes.
    said: BTreeSet<(PathBuf, String)>,
    events: Vec<Event>,
}

impl Writes {
    /// Allows writes into `dir` and anything under it. The directory
    /// must exist: an allowance names a place, it does not make one.
    pub(crate) fn allow(&mut self, dir: &Path) -> Result<(), String> {
        let resolved = dir
            .canonicalize()
            .map_err(|e| format!("cannot allow writes into {}: {e}", dir.display()))?;
        if !resolved.is_dir() {
            return Err(format!(
                "cannot allow writes into {}: not a directory",
                dir.display()
            ));
        }
        self.allowed.push(resolved);
        Ok(())
    }

    /// Writes every output of `graph` the evaluation behind `report`
    /// changed. Called once the evaluation has committed.
    pub(crate) fn after(&mut self, graph: &Graph, report: &Report, runtime: &mut Runtime) {
        for write in &graph.writes {
            let Some(handle) = report
                .outputs
                .iter()
                .find(|(name, _)| name == &write.output)
                .map(|(_, handle)| *handle)
            else {
                continue;
            };
            // The same Field as last time: nothing it holds has changed.
            if self
                .written
                .get(&write.path)
                .is_some_and(|(last, _)| *last == handle)
            {
                continue;
            }
            let outcome = self.place(&write.path).and_then(|target| {
                encode(runtime, handle, &write.path, write.bytes).map(|b| (target, b))
            });
            let (target, bytes) = match outcome {
                Ok(ready) => ready,
                Err(why) => {
                    self.refuse(&write.output, &write.path, why);
                    continue;
                }
            };
            let digest = digest(&bytes);
            // What is there already, this session's own last write or,
            // before it has written, the file as it was found.
            let there = match self.written.get(&write.path) {
                Some((_, last)) => Some(*last),
                None => std::fs::read(&target)
                    .ok()
                    .map(|found| self::digest(&found)),
            };
            if there == Some(digest) {
                self.written.insert(write.path.clone(), (handle, digest));
                continue;
            }
            match replace(&target, &bytes) {
                Ok(()) => {
                    self.written.insert(write.path.clone(), (handle, digest));
                    self.said.retain(|(path, _)| path != &write.path);
                    self.events.push(Event::Wrote {
                        output: write.output.clone(),
                        path: write.path.clone(),
                        bytes: bytes.len(),
                    });
                }
                Err(why) => self.refuse(&write.output, &write.path, why),
            }
        }
    }

    /// What was written or refused since the last call.
    pub(crate) fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Where `path` lands, once it is known to be inside an allowed
    /// directory: `..` and links are followed before the test, so no
    /// spelling reaches past one.
    pub(crate) fn place(&self, path: &Path) -> Result<PathBuf, String> {
        if self.allowed.is_empty() {
            return Err(format!(
                "writing files is not allowed; run with `--allow file.write={}`",
                path.parent().unwrap_or(Path::new(".")).display()
            ));
        }
        let target = resolve(path)?;
        if !self.allowed.iter().any(|dir| target.starts_with(dir)) {
            return Err(format!(
                "{} is outside every directory allowed with `--allow file.write`",
                target.display()
            ));
        }
        Ok(target)
    }

    fn refuse(&mut self, output: &str, path: &Path, why: String) {
        if self.said.insert((path.to_path_buf(), why.clone())) {
            self.events.push(Event::WriteRefused {
                output: output.to_owned(),
                path: path.to_path_buf(),
                why,
            });
        }
    }
}

/// The Field's bytes as the file takes them: its bytes, one to an
/// element of a `Rows<Int>`, or the format the extension names.
fn encode(
    runtime: &mut Runtime,
    handle: Handle,
    path: &Path,
    bytes: bool,
) -> Result<Vec<u8>, String> {
    runtime.materialize(handle).map_err(|e| e.to_string())?;
    let data = runtime.host().data(handle).map_err(|e| e.to_string())?;
    if bytes {
        return orla_formats::encode_bytes(data);
    }
    orla_formats::encode(data, &orla_formats::extension(path), None).map_err(|why| match why {
        orla_formats::EncodeError::Unknown => why.to_string(),
        orla_formats::EncodeError::Unfit(what) => {
            format!("the output is {what}; write it as .field")
        }
    })
}

/// A file replaced whole: written beside it, then renamed over it, so a
/// reader, the scene's own input included, never sees half of it.
fn replace(target: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = target.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let beside = parent.join(format!(".{name}.writing"));
    std::fs::write(&beside, bytes).map_err(|e| format!("{}: {e}", beside.display()))?;
    std::fs::rename(&beside, target).map_err(|e| {
        let _ = std::fs::remove_file(&beside);
        format!("{}: {e}", target.display())
    })
}

/// `path` made absolute with `.` and `..` taken away, and the longest
/// part of it that exists resolved through its links.
fn resolve(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut plain = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                plain.pop();
            }
            Component::CurDir => {}
            other => plain.push(other),
        }
    }
    let mut existing = plain.as_path();
    let mut rest = Vec::new();
    while !existing.exists() {
        let Some(parent) = existing.parent() else {
            break;
        };
        rest.push(existing.file_name().map(PathBuf::from).unwrap_or_default());
        existing = parent;
    }
    let mut resolved = existing
        .canonicalize()
        .map_err(|e| format!("{}: {e}", existing.display()))?;
    for part in rest.iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

fn digest(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
