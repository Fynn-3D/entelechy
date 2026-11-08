//! A scene's file outputs (`output name <- source to "file";`). The
//! session writes each once the evaluation that changed it is confirmed,
//! whole, into an allowed directory; an output `to files "dir/"` is a
//! list of operations on the files under the directory, each done once
//! when the list changes: a write where writes are allowed, a removal or
//! a rename where removals are too.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use entelechy_kernel::{Fetch, FieldData, Handle, Runtime};

use crate::Event;
use crate::store::Store;

/// The directories writes may land in, and what was last written.
#[derive(Default)]
pub struct Writes {
    /// Allowed directories, resolved.
    allowed: Vec<PathBuf>,
    /// Allowed single files, resolved (`--save`, `--out`), for the
    /// caller's own writes ([`Writes::place`]); a scene's `to` needs an
    /// allowed directory.
    files: Vec<PathBuf>,
    /// Directories files may be removed from and renamed in, resolved.
    deletes: Vec<PathBuf>,
    /// Per file: the Field last written, and a digest of its bytes.
    written: BTreeMap<PathBuf, (Handle, u64)>,
    /// Per file: the words of the Fields asked for and not yet written,
    /// oldest first ([`Writes::ask`]).
    asked: BTreeMap<PathBuf, VecDeque<(Handle, Fetch)>>,
    /// Each file an output wrote, where it landed: the output, and the
    /// file's stamp once written.
    authored: BTreeMap<PathBuf, (String, Option<crate::store::Stamp>)>,
    /// Refusals reported, each once until it changes.
    said: BTreeSet<(PathBuf, String)>,
    events: Vec<Event>,
}

impl Writes {
    /// Allows writes into the existing directory `dir` and under it.
    pub fn allow(&mut self, store: &dyn Store, dir: &Path) -> Result<(), String> {
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

    /// Allows files in the existing directory `dir` and under it to be
    /// removed and renamed.
    pub fn allow_deletes(&mut self, store: &dyn Store, dir: &Path) -> Result<(), String> {
        let resolved = store
            .canonical(dir)
            .map_err(|e| format!("cannot allow removals in {}: {e}", dir.display()))?;
        if !store.is_dir(&resolved) {
            return Err(format!(
                "cannot allow removals in {}: not a directory",
                dir.display()
            ));
        }
        self.deletes.push(resolved);
        Ok(())
    }

    /// Takes back writes, removals and renames in `dir` that
    /// [`Writes::allow`] and [`Writes::allow_deletes`] gave.
    pub fn revoke(&mut self, store: &dyn Store, dir: &Path) {
        if let Ok(resolved) = store.canonical(dir) {
            self.allowed.retain(|allowed| *allowed != resolved);
            self.deletes.retain(|allowed| *allowed != resolved);
        }
    }

    /// Allows a write to the one file `path`, wherever it is.
    pub fn allow_file(&mut self, store: &dyn Store, path: &Path) -> Result<(), String> {
        let resolved = resolve(store, path)?;
        self.files.push(resolved);
        Ok(())
    }

    /// Asks for the words of each output written to a file that the
    /// evaluation just made changed, `outputs` as it left them by name: the
    /// copy waits behind that evaluation's work alone, so once it is
    /// confirmed ([`Writes::after`]) the words are in hand and writing
    /// them waits for no frame made since.
    pub fn ask(
        &mut self,
        writes: &[entelechy_scene::Write],
        outputs: &[(String, Handle)],
        runtime: &Runtime,
    ) {
        for write in writes {
            let Some(handle) = outputs
                .iter()
                .find(|(name, _)| name == &write.output)
                .map(|(_, handle)| *handle)
            else {
                continue;
            };
            if self
                .written
                .get(&write.path)
                .is_some_and(|(last, _)| *last == handle)
            {
                continue;
            }
            let asked = self.asked.entry(write.path.clone()).or_default();
            if asked.iter().any(|(already, _)| *already == handle) {
                continue;
            }
            // Asked for and never confirmed (a frame taken back), the
            // oldest go: at most two frames are in flight.
            if asked.len() >= ASKED {
                asked.pop_front();
            }
            if let Ok(fetch) = runtime.fetch(handle) {
                asked.push_back((handle, fetch));
            }
        }
    }

    /// Writes every output a confirmed evaluation changed, `outputs` as it
    /// left them by name; `names` gives an output's element's numbers by
    /// name, which a data table names its columns by.
    pub fn after(
        &mut self,
        store: &dyn Store,
        writes: &[entelechy_scene::Write],
        outputs: &[(String, Handle)],
        runtime: &mut Runtime,
        names: &dyn Fn(&str) -> Result<Vec<String>, String>,
    ) {
        for write in writes {
            let Some(handle) = outputs
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
            if write.files {
                self.operate(store, write, handle, runtime);
                continue;
            }
            let outcome = self.place_in(store, &write.path, false).and_then(|target| {
                let data = words(&mut self.asked, &write.path, handle, runtime)?;
                encode(&data, &write.path, write.bytes, &|| names(&write.output))
                    .map(|b| (target, b))
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
                    self.authored(store, &target, &write.output);
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

    /// Does the operations output `write` holds, once for each list that
    /// differs from the last one done: each in order, one refused leaving
    /// the others to be done.
    fn operate(
        &mut self,
        store: &dyn Store,
        write: &entelechy_scene::Write,
        handle: Handle,
        runtime: &mut Runtime,
    ) {
        let ops = words(&mut self.asked, &write.path, handle, runtime).and_then(|data| {
            let words: Vec<u8> = data.values().iter().flat_map(|w| w.to_le_bytes()).collect();
            Ok((digest(&words), entelechy_formats::directory::ops(&data)?))
        });
        let (digest, ops) = match ops {
            Ok(read) => read,
            Err(why) => {
                self.written.insert(write.path.clone(), (handle, 0));
                self.refuse(&write.output, &write.path, why);
                return;
            }
        };
        let last = self.written.insert(write.path.clone(), (handle, digest));
        if last.is_some_and(|(_, done)| done == digest) {
            return;
        }
        let root = match resolve(store, &write.path) {
            Ok(root) => root,
            Err(why) => return self.refuse(&write.output, &write.path, why),
        };
        for op in ops {
            let named = write.path.join(op.path());
            if let Err(why) = self.done(store, &root, write, &op) {
                self.refuse(&write.output, &named, why);
            } else {
                self.said.retain(|(path, _)| path != &named);
            }
        }
    }

    /// Does one operation under `root`, the directory resolved.
    fn done(
        &mut self,
        store: &dyn Store,
        root: &Path,
        write: &entelechy_scene::Write,
        op: &entelechy_formats::directory::Op,
    ) -> Result<(), String> {
        use entelechy_formats::directory::Op;
        let output = write.output.clone();
        match op {
            Op::Write { path, bytes } => {
                // A file holding the bytes already is left as it is, allowed
                // or not: nothing is written.
                let named = write.path.join(path);
                if resolve(store, &named)
                    .is_ok_and(|at| store.read(&at).is_ok_and(|found| found == *bytes))
                {
                    return Ok(());
                }
                let target = self.place_in(store, &named, false)?;
                store
                    .write(&target, bytes)
                    .map_err(|e| format!("{}: {e}", target.display()))?;
                self.authored(store, &target, &output);
                self.events.push(Event::Wrote {
                    output,
                    path: write.path.join(path),
                    bytes: bytes.len(),
                });
            }
            Op::Remove { path } => {
                let target = self.removable(store, &write.path.join(path))?;
                store
                    .remove(&target)
                    .map_err(|e| format!("{}: {e}", target.display()))?;
                self.emptied(store, root, &target);
                self.events.push(Event::Removed {
                    output,
                    path: write.path.join(path),
                });
            }
            Op::Rename { from, to } => {
                let source = self.removable(store, &write.path.join(from))?;
                let target = self.place_in(store, &write.path.join(to), false)?;
                if store.exists(&target) {
                    return Err(format!("{} is there already", target.display()));
                }
                store
                    .rename(&source, &target)
                    .map_err(|e| format!("{}: {e}", source.display()))?;
                self.emptied(store, root, &source);
                self.events.push(Event::Renamed {
                    output,
                    from: write.path.join(from),
                    to: write.path.join(to),
                });
            }
        }
        Ok(())
    }

    /// Where a file to be removed or renamed is: a file, inside a directory
    /// allowed with `--allow file.delete`.
    fn removable(&self, store: &dyn Store, path: &Path) -> Result<PathBuf, String> {
        if self.deletes.is_empty() {
            return Err(format!(
                "removing and renaming files is not allowed; run with `--allow file.delete={}`",
                path.parent().unwrap_or(Path::new(".")).display()
            ));
        }
        let target = resolve(store, path)?;
        if !self.deletes.iter().any(|dir| target.starts_with(dir)) {
            return Err(format!(
                "{} is outside every directory allowed with `--allow file.delete`",
                target.display()
            ));
        }
        if store.kind(&target) != Some(crate::store::Kind::File) {
            return Err(format!("{} is not a file", target.display()));
        }
        Ok(target)
    }

    /// Removes the directories `gone` left empty, up to `root`.
    fn emptied(&self, store: &dyn Store, root: &Path, gone: &Path) {
        let mut dir = gone.parent();
        while let Some(at) = dir.filter(|at| at.starts_with(root) && *at != root) {
            if store.remove_empty(at).is_err() {
                break;
            }
            dir = at.parent();
        }
    }

    /// Notes that output `output` wrote the file at `target`, as it now
    /// stands.
    fn authored(&mut self, store: &dyn Store, target: &Path, output: &str) {
        self.authored.insert(
            target.to_path_buf(),
            (output.to_owned(), store.stamp(target)),
        );
    }

    /// The output that wrote the file at `path`, while the file is as it
    /// wrote it.
    pub fn author(&self, store: &dyn Store, path: &Path) -> Option<&str> {
        let at = store.canonical(path).ok()?;
        self.authored
            .get(&at)
            .filter(|(_, stamp)| stamp.is_some() && store.stamp(&at) == *stamp)
            .map(|(output, _)| output.as_str())
    }

    /// What was written or refused since the last call.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Where a caller's write to `path` (`save`, export) lands, if it is
    /// an allowed file or inside an allowed directory. `..` and links are
    /// resolved before the test.
    pub fn place(&self, store: &dyn Store, path: &Path) -> Result<PathBuf, String> {
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
/// The Fields asked for a file at most: the frames in flight and the one
/// confirmed.
const ASKED: usize = 3;

/// The words of `handle`, written to `path`: those asked for when it was
/// made, the older asked before them let go; else brought home now.
fn words<'r>(
    asked: &mut BTreeMap<PathBuf, VecDeque<(Handle, Fetch)>>,
    path: &Path,
    handle: Handle,
    runtime: &'r mut Runtime,
) -> Result<Cow<'r, FieldData>, String> {
    if let Some(queue) = asked.get_mut(path)
        && let Some(at) = queue.iter().position(|(made, _)| *made == handle)
    {
        queue.drain(..at);
        let (_, fetch) = queue.pop_front().expect("the match is first");
        return fetch.take().map(Cow::Owned).map_err(|e| e.to_string());
    }
    runtime.materialize(handle).map_err(|e| e.to_string())?;
    runtime
        .host()
        .data(handle)
        .map(Cow::Borrowed)
        .map_err(|e| e.to_string())
}

fn encode(
    data: &FieldData,
    path: &Path,
    bytes: bool,
    names: &dyn Fn() -> Result<Vec<String>, String>,
) -> Result<Vec<u8>, String> {
    if bytes {
        return entelechy_formats::encode_bytes(data);
    }
    let extension = entelechy_formats::extension(path);
    let names = if entelechy_formats::table::is_table(&extension) {
        Some(names()?)
    } else {
        None
    };
    entelechy_formats::encode(data, &extension, None, names.as_deref()).map_err(|why| match why {
        entelechy_formats::EncodeError::Unfit(what) => {
            format!("the output is {what}; write it as .field")
        }
        other => other.to_string(),
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
