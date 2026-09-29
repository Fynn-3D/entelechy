//! A scene's file outputs (`output name <- source to "file";`). The
//! session writes each after a committed evaluation that changed it,
//! whole, into an allowed directory.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use orla_graph::Report;
use orla_kernel::{Handle, Runtime};

use crate::Event;
use crate::store::Store;

/// The directories writes may land in, and what was last written.
#[derive(Default)]
pub(crate) struct Writes {
    /// Allowed directories, resolved.
    allowed: Vec<PathBuf>,
    /// Allowed single files, resolved (`--save`, `--out`), for the
    /// caller's own writes ([`Writes::place`]); a scene's `to` needs an
    /// allowed directory.
    files: Vec<PathBuf>,
    /// Per file: the Field last written, and a digest of its bytes.
    written: BTreeMap<PathBuf, (Handle, u64)>,
    /// Refusals reported, each once until it changes.
    said: BTreeSet<(PathBuf, String)>,
    events: Vec<Event>,
}

impl Writes {
    /// Allows writes into the existing directory `dir` and under it.
    pub(crate) fn allow(&mut self, store: &dyn Store, dir: &Path) -> Result<(), String> {
        let resolved = store
            .canonical(dir)
            .map_err(|e| format!("cannot allow writes into {}: {e}", dir.display()))?;
        if !store.is_dir(&resolved) {
            return Err(format!(
                "cannot allow writes into {}: not a directory",
                dir.display()
            ));
        }
        self.allowed.push(resolved);
        Ok(())
    }

    /// Allows a write to the one file `path`, wherever it is.
    pub(crate) fn allow_file(&mut self, store: &dyn Store, path: &Path) -> Result<(), String> {
        let resolved = resolve(store, path)?;
        self.files.push(resolved);
        Ok(())
    }

    /// Writes every output the committed evaluation behind `report`
    /// changed.
    pub(crate) fn after(
        &mut self,
        store: &dyn Store,
        writes: &[orla_scene::Write],
        report: &Report,
        runtime: &mut Runtime,
    ) {
        for write in writes {
            let Some(handle) = report
                .outputs
                .iter()
                .find(|(name, _)| name == &write.output)
                .map(|(_, handle)| *handle)
            else {
                continue;
            };
            // The same Field as last written is unchanged.
            if self
                .written
                .get(&write.path)
                .is_some_and(|(last, _)| *last == handle)
            {
                continue;
            }
            let outcome = self.place_in(store, &write.path, false).and_then(|target| {
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
            // The digest of what is there: this session's last write, or
            // the file as found.
            let there = match self.written.get(&write.path) {
                Some((_, last)) => Some(*last),
                None => store.read(&target).ok().map(|found| self::digest(&found)),
            };
            if there == Some(digest) {
                self.written.insert(write.path.clone(), (handle, digest));
                continue;
            }
            match store
                .write(&target, &bytes)
                .map_err(|e| format!("{}: {e}", target.display()))
            {
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

    /// Where a caller's write to `path` (`save`, export) lands, if it is
    /// an allowed file or inside an allowed directory. `..` and links are
    /// resolved before the test.
    pub(crate) fn place(&self, store: &dyn Store, path: &Path) -> Result<PathBuf, String> {
        self.place_in(store, path, true)
    }

    /// Where `path` lands; with `files`, a file allowed on its own counts.
    fn place_in(&self, store: &dyn Store, path: &Path, files: bool) -> Result<PathBuf, String> {
        let files = if files { &self.files[..] } else { &[] };
        if self.allowed.is_empty() && files.is_empty() {
            return Err(format!(
                "writing files is not allowed; run with `--allow file.write={}`",
                path.parent().unwrap_or(Path::new(".")).display()
            ));
        }
        let target = resolve(store, path)?;
        if files.contains(&target) {
            return Ok(target);
        }
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

/// The Field encoded for its file: raw bytes (one per `Rows<Int>`
/// element) when `bytes`, else the format the extension names.
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

/// Where `path` leads: its longest existing prefix resolved as the store
/// follows it, links and `..` included, and the rest, which does not
/// exist and so holds no link, folded onto it.
pub(crate) fn resolve(store: &dyn Store, path: &Path) -> Result<PathBuf, String> {
    let parts: Vec<std::path::Component> = path.components().collect();
    let mut split = parts.len();
    let existing = loop {
        let prefix: PathBuf = parts[..split].iter().collect();
        if split == 0 {
            break store.whole(&prefix);
        }
        if store.exists(&prefix) {
            break prefix;
        }
        split -= 1;
    };
    let resolved = store
        .canonical(&existing)
        .map_err(|e| format!("{}: {e}", existing.display()))?;
    let rest: PathBuf = parts[split..].iter().collect();
    Ok(crate::store::folded(&resolved.join(rest)))
}

fn digest(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
