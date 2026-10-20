//! `files` inputs: a directory's files, held packed.

use std::path::{Path, PathBuf};

use entelechy_kernel::Host;

use super::{Inputs, room};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A `files` input: its directory, and each file found there, by its
/// path under it, with its stamp when it was last read.
pub(super) struct Listed {
    pub(super) dir: PathBuf,
    files: Vec<(String, Option<Stamp>)>,
}

/// Every file under `dir` (hidden ones and those under hidden
/// directories left out), each by its path under `dir`, parted by `/`,
/// in path order.
fn files_under(store: &dyn Store, dir: &Path) -> Vec<String> {
    fn walk(store: &dyn Store, dir: &Path, prefix: &str, found: &mut Vec<String>) {
        let mut entries = store.list(dir).unwrap_or_default();
        entries.sort();
        for entry in entries {
            let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let path = format!("{prefix}{name}");
            match store.kind(&entry) {
                Some(crate::store::Kind::Dir) => walk(store, &entry, &format!("{path}/"), found),
                Some(crate::store::Kind::File) => found.push(path),
                None => {}
            }
        }
    }
    let mut found = Vec::new();
    if store.kind(dir) == Some(crate::store::Kind::Dir) {
        walk(store, dir, "", &mut found);
    }
    found.sort();
    found
}

impl Inputs {
    /// Reads each `files` input's directory again when one of its files,
    /// or which there are, changed, and holds them packed
    /// ([`entelechy_formats::directory::pack`]); forgets undeclared ones.
    pub(super) fn sync_files(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .listed
            .keys()
            .filter(|name| !files.files.iter().any(|(wanted, _)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.listed.remove(&name);
            self.forget(host, &name);
        }
        for (name, dir) in &files.files {
            let paths = files_under(store, dir);
            let stamps: Vec<(String, Option<Stamp>)> = paths
                .iter()
                .map(|path| (path.clone(), store.stamp(&dir.join(path))))
                .collect();
            if self
                .listed
                .get(name)
                .is_some_and(|listed| listed.dir == *dir && listed.files == stamps)
            {
                continue;
            }
            self.listed.insert(
                name.clone(),
                Listed {
                    dir: dir.clone(),
                    files: stamps,
                },
            );
            let read = std::iter::once(dir.clone())
                .chain(paths.iter().map(|path| dir.join(path)))
                .find_map(|path| readable(&path).err())
                .map_or_else(
                    || {
                        let mut room = room(host);
                        let mut read = Vec::with_capacity(paths.len());
                        for path in &paths {
                            let bytes = store
                                .read(&dir.join(path))
                                .map_err(|e| format!("{path}: {e}"))?;
                            room = room.checked_sub(bytes.len() + path.len() + 2).ok_or_else(
                                || {
                                    format!(
                                        "the files under {} are past the memory budget",
                                        dir.display()
                                    )
                                },
                            )?;
                            read.push((path.clone(), bytes));
                        }
                        Ok(read)
                    },
                    Err,
                )
                .and_then(|read| {
                    let size: usize = read.iter().map(|(_, bytes)| bytes.len()).sum();
                    let data = entelechy_formats::directory::pack(&read)?;
                    let handle = host.adopt(data).map_err(|error| error.to_string())?;
                    Ok((handle, size))
                });
            match read {
                Ok((handle, size)) => {
                    self.hold(host, name, handle);
                    events.push(Event::Loaded {
                        name: name.clone(),
                        path: dir.clone(),
                        detail: format!("{} file(s), {size} byte(s)", paths.len()),
                    });
                }
                Err(error) => {
                    self.forget(host, name);
                    events.push(Event::Rejected {
                        path: dir.clone(),
                        error: Diagnostic::new(error),
                    });
                }
            }
        }
    }
}
