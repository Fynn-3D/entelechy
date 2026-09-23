//! Which document published which Kernel. The ledger owns that and
//! nothing else: the Kernels themselves are the runtime's, which the
//! session hands in, and what a document says is read by the scene. A
//! document's Kernel publishes when its text or what it inherits
//! changed; one that is running is replaced by a candidate prepared off
//! the caller's thread, and a replacement that fails leaves it running.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_language::Kernel;
use orla_runtime::{Expect, KernelId, Runtime, Ticket};

use crate::Event;
use crate::sources::located;
use crate::stamp::Stamp;

/// A Kernel a document defines: its id, its source text, and the parse.
pub(crate) struct Defined {
    pub(crate) id: KernelId,
    pub(crate) text: String,
    pub(crate) kernel: Kernel,
}

/// What one document has published.
#[derive(Default)]
struct DocumentState {
    stamp: Option<Stamp>,
    kernels: BTreeMap<KernelId, String>,
    /// The Kernels as they were parsed, which a trace compiles again
    /// without the optimizer so that it walks what the source says.
    asts: BTreeMap<KernelId, Kernel>,
    /// The shared declarations the document's Kernels were compiled
    /// with; when they change, every Kernel republishes even though its
    /// own text did not.
    declarations: String,
}

/// A replacement of a running Kernel the runtime is still preparing.
struct PendingPublish {
    ticket: Ticket,
    path: PathBuf,
    expect: Expect,
}

/// The ledger of published Kernels by the document that owns each.
#[derive(Default)]
pub(crate) struct Kernels {
    /// Kernel text by id for every document that published Kernels; the
    /// root document is one of them.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    pending: BTreeMap<KernelId, PendingPublish>,
}

impl Kernels {
    /// Every document that has published, or tried to.
    pub(crate) fn documents(&self) -> Vec<PathBuf> {
        self.documents.keys().cloned().collect()
    }

    /// Whether `path` was last read at `stamp`.
    pub(crate) fn read_at(&self, path: &Path, stamp: Option<Stamp>) -> bool {
        self.documents
            .get(path)
            .is_some_and(|state| state.stamp == stamp)
    }

    /// Notes that `path` was read at `stamp` and publishes nothing new,
    /// so an unreadable document is not read again until it changes.
    pub(crate) fn mark_read(&mut self, path: PathBuf, stamp: Option<Stamp>) {
        self.documents.entry(path).or_default().stamp = stamp;
    }

    /// A diagnostic with the file that declares what it could not find,
    /// searched among the documents this ledger knows.
    pub(crate) fn located(&self, message: String, document: &Path, root: &Path) -> String {
        located(message, document, root, &self.documents())
    }

    /// Publishes every replacement the worker has finished.
    pub(crate) fn collect_pending(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let ids: Vec<KernelId> = self.pending.keys().cloned().collect();
        for id in ids {
            let ticket = self.pending[&id].ticket;
            let Some(result) = runtime.poll(ticket) else {
                continue;
            };
            let PendingPublish { path, expect, .. } =
                self.pending.remove(&id).expect("pending entry exists");
            match result {
                Ok(candidate) => match runtime.publish(candidate, expect) {
                    Ok(version) => {
                        events.push(Event::Published {
                            path: path.clone(),
                            id: id.clone(),
                            generation: version.generation(),
                        });
                        self.owners.insert(id, path);
                    }
                    Err(error) => events.push(Event::Rejected {
                        path,
                        error: format!(" {error}"),
                    }),
                },
                Err(error) => events.push(Event::Rejected {
                    path,
                    error: error.to_string(),
                }),
            }
        }
    }

    /// Publishes the Kernels a document now defines: retires the ones that
    /// left, keeps the unchanged, republishes the changed, and reports each.
    /// `declarations` keys what the Kernels inherit; a change there
    /// republishes them all. `root` is the scene, for diagnostics.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish_document(
        &mut self,
        path: &Path,
        current: Option<Stamp>,
        defined: Vec<Defined>,
        declarations: &str,
        root: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let previous_state = self.documents.remove(path).unwrap_or_default();
        let previous = if previous_state.declarations == declarations {
            previous_state.kernels
        } else {
            BTreeMap::new()
        };
        let mut kernels = BTreeMap::new();
        let mut asts = BTreeMap::new();
        for id in previous.keys() {
            if !defined.iter().any(|d| &d.id == id) {
                self.retire(path, id, runtime, events);
            }
        }
        for Defined { id, text, kernel } in defined {
            asts.insert(id.clone(), kernel.clone());
            if let Some(owner) = self.owners.get(&id)
                && owner != path
            {
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: format!(
                        " {id} is already published by {}; kernel ids are unique",
                        owner.display()
                    ),
                });
                continue;
            }
            if previous.get(&id) == Some(&text) && runtime.kernel(&id).is_some() {
                kernels.insert(id, text);
                continue;
            }
            if let Some(running) = runtime.kernel(&id) {
                // Something is running: its replacement may take its time.
                match orla_compiler::compile_kernel(&kernel, Some(&id)) {
                    Ok(module) => {
                        if let Some(superseded) = self.pending.remove(&id) {
                            runtime.discard(superseded.ticket);
                        }
                        let ticket = runtime.prepare_later(module, text.clone());
                        self.pending.insert(
                            id.clone(),
                            PendingPublish {
                                ticket,
                                path: path.to_path_buf(),
                                expect: Expect::Generation(running.generation()),
                            },
                        );
                        if runtime.background() {
                            events.push(Event::Preparing {
                                path: path.to_path_buf(),
                                id: id.clone(),
                            });
                        }
                        self.collect_pending(runtime, events);
                    }
                    Err(error) => events.push(Event::Rejected {
                        path: path.to_path_buf(),
                        error: self.located(error.to_string(), path, root),
                    }),
                }
                kernels.insert(id, text);
                continue;
            }
            let prepared = orla_compiler::compile_kernel(&kernel, Some(&id))
                .map_err(|error| error.to_string())
                .and_then(|module| {
                    runtime
                        .prepare_module(module, &text)
                        .map_err(|error| error.to_string())
                });
            match prepared {
                Ok(candidate) => {
                    let expect = match runtime.kernel(&id) {
                        Some(version) => Expect::Generation(version.generation()),
                        None => Expect::Absent,
                    };
                    match runtime.publish(candidate, expect) {
                        Ok(version) => {
                            events.push(Event::Published {
                                path: path.to_path_buf(),
                                id: id.clone(),
                                generation: version.generation(),
                            });
                            self.owners.insert(id.clone(), path.to_path_buf());
                        }
                        Err(error) => events.push(Event::Rejected {
                            path: path.to_path_buf(),
                            error: format!(" {error}"),
                        }),
                    }
                }
                Err(error) => events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: self.located(error.to_string(), path, root),
                }),
            }
            kernels.insert(id, text);
        }
        self.documents.insert(
            path.to_path_buf(),
            DocumentState {
                stamp: current,
                kernels,
                asts,
                declarations: declarations.to_owned(),
            },
        );
    }

    /// Retires every Kernel a document published, and forgets it.
    pub(crate) fn retire_document(
        &mut self,
        path: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        if let Some(state) = self.documents.remove(path) {
            for id in state.kernels.keys() {
                self.retire(path, id, runtime, events);
            }
        }
    }

    fn retire(
        &mut self,
        path: &Path,
        id: &KernelId,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        if self.owners.get(id).is_some_and(|owner| owner == path) {
            self.owners.remove(id);
            if let Some(pending) = self.pending.remove(id) {
                runtime.discard(pending.ticket);
            }
            if runtime.retire(id).is_some() {
                events.push(Event::Retired {
                    path: path.to_path_buf(),
                    id: id.clone(),
                });
            }
        }
    }

    /// Retires a Kernel whoever published it; returns whether one was
    /// registered.
    pub(crate) fn retire_kernel(&mut self, id: &KernelId, runtime: &mut Runtime) -> bool {
        self.owners.remove(id);
        if let Some(pending) = self.pending.remove(id) {
            runtime.discard(pending.ticket);
        }
        for state in self.documents.values_mut() {
            state.kernels.remove(id);
        }
        runtime.retire(id).is_some()
    }

    /// The Kernel as its document wrote it, compiled without the
    /// optimizer; `None` for one no document owns, or one whose source
    /// no longer compiles on its own.
    pub(crate) fn unoptimized(&self, id: &KernelId) -> Option<orla_runtime::Validated> {
        let kernel = self
            .documents
            .values()
            .find_map(|state| state.asts.get(id))?;
        orla_compiler::compile_kernel_unoptimized(kernel, Some(id)).ok()
    }
}
