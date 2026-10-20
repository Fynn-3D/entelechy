//! Files nodes: a directory listed, and the files a node's paths name, read
//! for the next evaluation.

use std::path::{Path, PathBuf};

use entelechy_kernel::{FieldData, Handle, Host};

use super::{Inputs, room};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A `files` node: its directory, each file it listed with its stamp,
/// the paths it was last given and each file it read of them with its
/// stamp, and the Field it last read its paths from.
pub(super) struct FileNode {
    pub(super) dir: PathBuf,
    listed: Vec<(String, Option<Stamp>)>,
    paths: Vec<String>,
    opened: Vec<(String, Option<Stamp>)>,
    seen: Option<Handle>,
}

/// Every file under `dir` a listing shows: hidden ones, those under hidden
/// directories and those a `.gitignore` on the way leaves out are not
/// ([`crate::ignore`]); each by its path under `dir`, parted by `/`, with
/// its stamp, in path order.
fn listed_under(store: &dyn Store, dir: &Path) -> Vec<(String, Option<Stamp>)> {
    fn walk(
        store: &dyn Store,
        dir: &Path,
        prefix: &str,
        ignore: &crate::ignore::Ignore,
        found: &mut Vec<(String, Option<Stamp>)>,
    ) {
        let ignore = match store.read_text(&dir.join(".gitignore")) {
            Ok(text) => ignore.with(prefix, &text),
            Err(_) => ignore.clone(),
        };
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
                Some(crate::store::Kind::Dir) if !ignore.ignores(&path, true) => {
                    walk(store, &entry, &format!("{path}/"), &ignore, found);
                }
                Some(crate::store::Kind::File) if !ignore.ignores(&path, false) => {
                    found.push((path, store.stamp(&entry)));
                }
                _ => {}
            }
        }
    }
    let mut found = Vec::new();
    if store.kind(dir) == Some(crate::store::Kind::Dir) {
        walk(
            store,
            dir,
            "",
            &crate::ignore::Ignore::default(),
            &mut found,
        );
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// The paths a files node's `paths` spell: each path's UTF-8 bytes, ended
/// by a 0 or the Field's end; an empty one is no path.
pub(crate) fn paths_spelled(words: &[u32]) -> Result<Vec<String>, String> {
    let mut paths = Vec::new();
    for part in words.split(|word| *word == 0) {
        if part.is_empty() {
            continue;
        }
        let bytes: Vec<u8> = part
            .iter()
            .map(|word| u8::try_from(*word).map_err(|_| format!("{word} is not a byte of a path")))
            .collect::<Result<_, _>>()?;
        let path = String::from_utf8(bytes).map_err(|_| "a path is not UTF-8".to_owned())?;
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
    Ok(paths)
}

impl Inputs {
    /// Lists each files node's directory when what it holds changed, and
    /// reads again the files it was given that changed; a new node holds
    /// an empty `opened` until it is given paths; forgets undeclared ones.
    pub(super) fn sync_file_nodes(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .file_nodes
            .keys()
            .filter(|node| !files.file_nodes.iter().any(|wanted| &wanted.node == *node))
            .cloned()
            .collect();
        for node in stale {
            self.file_nodes.remove(&node);
            for port in entelechy_scene::FILES_OUTPUTS {
                self.forget(host, &format!("{node}.{port}"));
            }
        }
        for node in &files.file_nodes {
            let fresh = self
                .file_nodes
                .get(&node.node)
                .is_none_or(|known| known.dir != node.dir);
            if fresh {
                self.file_nodes.insert(
                    node.node.clone(),
                    FileNode {
                        dir: node.dir.clone(),
                        listed: Vec::new(),
                        paths: Vec::new(),
                        opened: Vec::new(),
                        seen: None,
                    },
                );
            }
            // The listing, when a file came, went or changed.
            let listed = match readable(&node.dir) {
                Ok(()) => listed_under(store, &node.dir),
                Err(why) => {
                    events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(why),
                    });
                    Vec::new()
                }
            };
            let known = self.file_nodes.get(&node.node).expect("just placed");
            if fresh || known.listed != listed {
                let sized: Vec<(String, u64)> = listed
                    .iter()
                    .map(|(path, stamp)| {
                        let size = stamp.and_then(|stamp| stamp.length()).unwrap_or_else(|| {
                            store
                                .read(&node.dir.join(path))
                                .map_or(0, |bytes| bytes.len() as u64)
                        });
                        (path.clone(), size)
                    })
                    .collect();
                match self.hold_output(
                    node,
                    "listing",
                    entelechy_formats::directory::pack_listing(&sized),
                    host,
                ) {
                    Ok(()) => events.push(Event::Loaded {
                        name: format!("{}.listing", node.node),
                        path: node.dir.clone(),
                        detail: format!("{} file(s) listed", sized.len()),
                    }),
                    Err(why) => events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(why),
                    }),
                }
                self.file_nodes
                    .get_mut(&node.node)
                    .expect("just placed")
                    .listed = listed;
            }
            // What it read, when one of those files changed.
            let known = self.file_nodes.get(&node.node).expect("just placed");
            let changed = fresh
                || known
                    .opened
                    .iter()
                    .any(|(path, stamp)| store.stamp(&node.dir.join(path)) != *stamp)
                || known.paths.iter().any(|path| {
                    !known.opened.iter().any(|(read, _)| read == path)
                        && store.stamp(&node.dir.join(path)).is_some()
                });
            if changed {
                let paths = known.paths.clone();
                if let Err(why) = self.read_opened(store, readable, node, &paths, host) {
                    events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(format!("files node `{}`: {why}", node.node)),
                    });
                }
            }
        }
    }

    /// Whether files node `node` last read its paths from `handle`.
    pub(crate) fn file_node_saw(&self, node: &entelechy_scene::FilesNode, handle: Handle) -> bool {
        self.file_nodes
            .get(&node.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives files node `node` the paths `paths` spell (read from the Field
    /// `handle`): the files they name read again when they differ from the
    /// last; what fails is reported by the next sync.
    pub(crate) fn give_paths(
        &mut self,
        store: &dyn Store,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        node: &entelechy_scene::FilesNode,
        handle: Handle,
        paths: Result<Vec<String>, String>,
        host: &mut Host,
    ) {
        let Some(known) = self.file_nodes.get_mut(&node.node) else {
            return;
        };
        known.seen = Some(handle);
        let read = match paths {
            Ok(paths) if known.paths == paths => return,
            Ok(paths) => self.read_opened(store, readable, node, &paths, host),
            Err(why) => Err(why),
        };
        if let Err(why) = read {
            self.frame_events.push(Event::Rejected {
                path: node.dir.clone(),
                error: Diagnostic::new(format!("files node `{}`: {why}", node.node)),
            });
        }
    }

    /// Reads the files `paths` names under files node `node`'s directory
    /// and holds them as its `opened`, in the order named; a path naming no
    /// file is left out. Refused whole when a path is not one under the
    /// directory, is not readable, or the files are past the budget.
    fn read_opened(
        &mut self,
        store: &dyn Store,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        node: &entelechy_scene::FilesNode,
        paths: &[String],
        host: &mut Host,
    ) -> Result<(), String> {
        let mut room = room(host);
        let mut read = Vec::with_capacity(paths.len());
        let mut opened = Vec::with_capacity(paths.len());
        for path in paths {
            entelechy_formats::directory::checked_path(path)?;
            let file = node.dir.join(path);
            readable(&file)?;
            let stamp = store.stamp(&file);
            let bytes = match store.read(&file) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(format!("{path}: {error}")),
            };
            room = room
                .checked_sub(bytes.len() + path.len() + 2)
                .ok_or_else(|| {
                    format!(
                        "the files named under {} are past the memory budget",
                        node.dir.display()
                    )
                })?;
            opened.push((path.clone(), stamp));
            read.push((path.clone(), bytes));
        }
        self.hold_output(
            node,
            "opened",
            entelechy_formats::directory::pack(&read),
            host,
        )?;
        let known = self
            .file_nodes
            .get_mut(&node.node)
            .expect("a files node gives paths");
        known.paths = paths.to_vec();
        known.opened = opened;
        Ok(())
    }

    /// Holds `data` as files node `node`'s output `port`.
    fn hold_output(
        &mut self,
        node: &entelechy_scene::FilesNode,
        port: &str,
        data: Result<FieldData, String>,
        host: &mut Host,
    ) -> Result<(), String> {
        let Some((_, external)) = node.outputs.iter().find(|(name, _)| name == port) else {
            return Ok(());
        };
        let handle = host.adopt(data?).map_err(|e| e.to_string())?;
        self.hold(host, external, handle);
        Ok(())
    }
}
