//! `checks` inputs: what the language finds wrong in documents, or what
//! they define.

use std::path::{Path, PathBuf};

use entelechy_kernel::Host;

use super::Inputs;
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A `checks` or an `outline` input: the file or directory it reads,
/// whether it holds what the documents define (else what is wrong in
/// them), and each document found there with its stamp when it was last
/// read.
pub(super) struct Checks {
    pub(super) path: PathBuf,
    outline: bool,
    /// The documents, with their stamps, of the text held.
    documents: Vec<(PathBuf, Option<Stamp>)>,
    /// A check of them as they are now, made off the frames while the
    /// session prepares in the background ([`Inputs::set_background`]):
    /// taken in by the first sync after it is done.
    running: Option<Running>,
}

/// A check made off the frames: the documents, with their stamps, it
/// checks, and where its text comes back.
struct Running {
    documents: Vec<(PathBuf, Option<Stamp>)>,
    text: std::sync::mpsc::Receiver<String>,
}

/// The documents a `checks` input checks: the file, or every `.ent` and
/// `.esn` under the directory (hidden ones left out), in name order.
fn documents_under(store: &dyn Store, path: &Path) -> Vec<PathBuf> {
    match store.kind(path) {
        Some(crate::store::Kind::File) => vec![path.to_path_buf()],
        Some(crate::store::Kind::Dir) => {
            let mut found = Vec::new();
            let mut entries = store.list(path).unwrap_or_default();
            entries.sort();
            for entry in entries {
                let hidden = entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_none_or(|name| name.starts_with('.'));
                if hidden {
                    continue;
                }
                let document =
                    matches!(entelechy_formats::extension(&entry).as_str(), "ent" | "esn");
                match store.kind(&entry) {
                    Some(crate::store::Kind::Dir) => found.extend(documents_under(store, &entry)),
                    Some(crate::store::Kind::File) if document => found.push(entry),
                    _ => {}
                }
            }
            found
        }
        None => Vec::new(),
    }
}

/// What a `checks` input holds of `documents` under `root`: what the
/// language finds wrong in them, or what they define (`outline`).
fn checked_text(store: &dyn Store, root: &Path, documents: &[PathBuf], outline: bool) -> String {
    if outline {
        crate::check::outline_text(store, root, documents)
    } else {
        crate::check::problems_text(store, root, documents)
    }
}

impl Inputs {
    /// Reads each `checks` and `outline` input's documents again when one
    /// of them, or which there are, changed, and holds what was found as
    /// text ([`crate::check::problems_text`],
    /// [`crate::check::outline_text`]); forgets undeclared ones.
    pub(super) fn sync_checks(
        &mut self,
        store: &std::sync::Arc<dyn Store>,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .checks
            .keys()
            .filter(|name| {
                !files
                    .checks
                    .iter()
                    .chain(&files.outlines)
                    .any(|(wanted, _)| wanted == *name)
            })
            .cloned()
            .collect();
        for name in stale {
            self.checks.remove(&name);
            self.forget(host, &name);
        }
        let wanted = files
            .checks
            .iter()
            .map(|(name, path)| (name, path, false))
            .chain(files.outlines.iter().map(|(name, path)| (name, path, true)));
        for (name, path, outline) in wanted {
            // A check made off the frames, done, is taken in first.
            let finished = self.checks.get_mut(name).and_then(|checks| {
                let running = checks.running.as_ref()?;
                match running.text.try_recv() {
                    Ok(text) => Some(Some((text, checks.running.take()?.documents))),
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        checks.running = None;
                        Some(None)
                    }
                }
            });
            if let Some(Some((text, documents))) = finished {
                let checks = self.checks.get_mut(name).expect("a check ran for it");
                checks.documents = documents.clone();
                self.take_check(host, name, path, outline, &text, documents.len(), events);
            }
            let documents = documents_under(store.as_ref(), path);
            let stamps: Vec<(PathBuf, Option<Stamp>)> = documents
                .iter()
                .map(|document| (document.clone(), store.stamp(document)))
                .collect();
            let held = self
                .checks
                .get(name)
                .filter(|checks| checks.path == *path && checks.outline == outline);
            // As they are, or being checked as they are.
            if held.is_some_and(|checks| {
                checks.documents == stamps
                    || checks
                        .running
                        .as_ref()
                        .is_some_and(|running| running.documents == stamps)
            }) {
                continue;
            }
            if let Some(why) = std::iter::once(path.as_path())
                .chain(documents.iter().map(PathBuf::as_path))
                .find_map(|path| readable(path).err())
            {
                self.checks.insert(
                    name.clone(),
                    Checks {
                        path: path.clone(),
                        outline,
                        documents: stamps,
                        running: None,
                    },
                );
                self.forget(host, name);
                events.push(Event::Rejected {
                    path: path.clone(),
                    error: Diagnostic::new(why),
                });
                continue;
            }
            // Off the frames once the input holds a check (the first is
            // taken in at once, so the scene never reads it absent).
            if self.background && held.is_some() && self.holds(name) {
                let (answer, text) = std::sync::mpsc::channel();
                let (store, root) = (std::sync::Arc::clone(store), path.clone());
                std::thread::spawn(move || {
                    let _ = answer.send(checked_text(store.as_ref(), &root, &documents, outline));
                });
                self.checks.get_mut(name).expect("held").running = Some(Running {
                    documents: stamps,
                    text,
                });
                continue;
            }
            let text = checked_text(store.as_ref(), path, &documents, outline);
            self.checks.insert(
                name.clone(),
                Checks {
                    path: path.clone(),
                    outline,
                    documents: stamps,
                    running: None,
                },
            );
            self.take_check(host, name, path, outline, &text, documents.len(), events);
        }
    }

    /// Holds a check's text as input `name`, said loaded (or rejected).
    #[allow(clippy::too_many_arguments)]
    fn take_check(
        &mut self,
        host: &mut Host,
        name: &str,
        path: &Path,
        outline: bool,
        text: &str,
        documents: usize,
        events: &mut Vec<Event>,
    ) {
        let lines = text.lines().filter(|line| !line.is_empty()).count();
        let found = if outline {
            "definition(s)"
        } else {
            "problem(s)"
        };
        let read = entelechy_formats::decode_bytes(text.as_bytes())
            .and_then(|(data, _)| host.adopt(data).map_err(|error| error.to_string()));
        match read {
            Ok(handle) => {
                self.hold(host, name, handle);
                events.push(Event::Loaded {
                    name: name.to_owned(),
                    path: path.to_path_buf(),
                    detail: format!("{lines} {found} in {documents} document(s)"),
                });
            }
            Err(error) => events.push(Event::Rejected {
                path: path.to_path_buf(),
                error: Diagnostic::new(error),
            }),
        }
    }
}
