//! Which document published which Kernel. The ledger owns that and
//! nothing else: the Kernels themselves are the runtime's, which the
//! session hands in, and what a document says is read by the scene. A
//! document's Kernels publish when their text or what they inherit
//! changed, all of a document's changes as one: prepared (a running
//! Kernel's replacement off the caller's thread), then swapped in
//! together, or, when any of them fails, none, and every Kernel the
//! document had goes on running as it was.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_language::Kernel;
use orla_runtime::{Candidate, Expect, KernelId, Runtime, Ticket};

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
#[derive(Clone, Default)]
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

/// One document's change the runtime is still preparing: every Kernel
/// it publishes, ready or on the worker, swapped in together once all
/// are ready, or none when any fails.
struct Batch {
    items: Vec<Item>,
    /// Kernels the document no longer defines: retired when the batch
    /// publishes.
    leaving: Vec<KernelId>,
    /// What the ledger held for the document before this change: put
    /// back when the batch fails, so a fix later republishes everything
    /// the change had.
    before: DocumentState,
}

/// A Kernel of a batch and what its publication expects.
struct Item {
    id: KernelId,
    prep: Prep,
    expect: Expect,
}

/// A candidate prepared, or the ticket it is being prepared under.
enum Prep {
    Ready(Box<Candidate>),
    Waiting(Ticket),
}

impl Batch {
    /// Lets go of what is still being prepared.
    fn discard(self, runtime: &mut Runtime) {
        for item in self.items {
            if let Prep::Waiting(ticket) = item.prep {
                runtime.discard(ticket);
            }
        }
    }
}

/// The ledger of published Kernels by the document that owns each.
#[derive(Default)]
pub(crate) struct Kernels {
    /// Kernel text by id for every document that published Kernels; the
    /// root document is one of them.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    /// Each document's change still being prepared.
    pending: BTreeMap<PathBuf, Batch>,
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

    /// Publishes every document's change whose Kernels the worker has
    /// all finished, each as one; a change of which one failed publishes
    /// nothing.
    pub(crate) fn collect_pending(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let paths: Vec<PathBuf> = self.pending.keys().cloned().collect();
        for path in paths {
            let Some(mut batch) = self.pending.remove(&path) else {
                continue;
            };
            let mut failed = Vec::new();
            let mut waiting = false;
            for item in &mut batch.items {
                let Prep::Waiting(ticket) = item.prep else {
                    continue;
                };
                match runtime.poll(ticket) {
                    None => waiting = true,
                    Some(Ok(candidate)) => item.prep = Prep::Ready(Box::new(candidate)),
                    Some(Err(error)) => failed.push((item.id.clone(), error.to_string())),
                }
            }
            if !failed.is_empty() {
                let held = batch.items.len() > failed.len() || !batch.leaving.is_empty();
                for (id, error) in failed {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: format!(" {id}: {error}"),
                    });
                }
                self.hold_back(&path, batch, held, runtime, events);
                continue;
            }
            if waiting {
                self.pending.insert(path, batch);
                continue;
            }
            self.publish_batch(&path, batch, runtime, events);
        }
    }

    /// Swaps a ready batch in, all at once, and retires what left.
    fn publish_batch(
        &mut self,
        path: &Path,
        batch: Batch,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let Batch {
            items,
            leaving,
            before,
        } = batch;
        let ready: Vec<(Candidate, Expect)> = items
            .into_iter()
            .filter_map(|item| match item.prep {
                Prep::Ready(candidate) => Some((*candidate, item.expect)),
                Prep::Waiting(ticket) => {
                    runtime.discard(ticket);
                    None
                }
            })
            .collect();
        match runtime.publish_all(ready) {
            Ok(versions) => {
                for version in versions {
                    events.push(Event::Published {
                        path: path.to_path_buf(),
                        id: version.id().clone(),
                        generation: version.generation(),
                    });
                    self.owners.insert(version.id().clone(), path.to_path_buf());
                }
                for id in &leaving {
                    self.retire(path, id, runtime, events);
                }
            }
            Err(error) => {
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: format!(" {error}"),
                });
                let batch = Batch {
                    items: Vec::new(),
                    leaving,
                    before,
                };
                self.hold_back(path, batch, true, runtime, events);
            }
        }
    }

    /// A change that failed: nothing of it is published, what it still
    /// prepares is let go, and the ledger holds what it held before, read
    /// at the document's new stamp so it is read again when it changes.
    /// `held` says the failure held back something else the change had,
    /// which is then said too.
    fn hold_back(
        &mut self,
        path: &Path,
        batch: Batch,
        held: bool,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let mut before = batch.before.clone();
        if let Some(state) = self.documents.get(path) {
            before.stamp = state.stamp;
        }
        batch.discard(runtime);
        self.documents.insert(path.to_path_buf(), before);
        if held {
            events.push(Event::Rejected {
                path: path.to_path_buf(),
                error: " nothing in it was published: a document's Kernels change together, and the ones it had go on running".into(),
            });
        }
    }

    /// Publishes the Kernels a document now defines, as one change:
    /// retires the ones that left, keeps the unchanged, republishes the
    /// changed, and reports each. When any of them does not compile or
    /// prepare, none is published and none retired. `declarations` keys
    /// what the Kernels inherit; a change there republishes them all.
    /// `root` is the scene, for diagnostics.
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
        // A newer change supersedes one still being prepared.
        if let Some(superseded) = self.pending.remove(path) {
            let before = superseded.before.clone();
            superseded.discard(runtime);
            self.documents.insert(path.to_path_buf(), before);
        }
        let before = self.documents.remove(path).unwrap_or_default();
        let previous = if before.declarations == declarations {
            before.kernels.clone()
        } else {
            BTreeMap::new()
        };
        let leaving: Vec<KernelId> = before
            .kernels
            .keys()
            .filter(|id| !defined.iter().any(|d| &d.id == *id))
            .cloned()
            .collect();
        let mut kernels = BTreeMap::new();
        let mut asts = BTreeMap::new();
        let mut items = Vec::new();
        let mut failed = false;
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
                failed = true;
                continue;
            }
            kernels.insert(id.clone(), text.clone());
            if previous.get(&id) == Some(&text) && runtime.kernel(&id).is_some() {
                continue;
            }
            let module = match orla_compiler::compile_kernel(&kernel, Some(&id)) {
                Ok(module) => module,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.to_path_buf(),
                        error: self.located(error.to_string(), path, root),
                    });
                    failed = true;
                    continue;
                }
            };
            let prep = match runtime.kernel(&id) {
                // Something is running: its replacement may take its time.
                Some(running) => {
                    let ticket = runtime.prepare_later(module, text);
                    if runtime.background() {
                        events.push(Event::Preparing {
                            path: path.to_path_buf(),
                            id: id.clone(),
                        });
                    }
                    items.push(Item {
                        id,
                        prep: Prep::Waiting(ticket),
                        expect: Expect::Generation(running.generation()),
                    });
                    continue;
                }
                None => runtime.prepare_module(module, &text),
            };
            match prep {
                Ok(candidate) => items.push(Item {
                    id,
                    prep: Prep::Ready(Box::new(candidate)),
                    expect: Expect::Absent,
                }),
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.to_path_buf(),
                        error: self.located(error.to_string(), path, root),
                    });
                    failed = true;
                }
            }
        }
        let batch = Batch {
            items,
            leaving,
            before,
        };
        self.documents.insert(
            path.to_path_buf(),
            DocumentState {
                stamp: current,
                kernels,
                asts,
                declarations: declarations.to_owned(),
            },
        );
        if failed {
            let held = !batch.items.is_empty() || !batch.leaving.is_empty();
            self.hold_back(path, batch, held, runtime, events);
            return;
        }
        self.pending.insert(path.to_path_buf(), batch);
        self.collect_pending(runtime, events);
    }

    /// Retires every Kernel a document published, and forgets it.
    pub(crate) fn retire_document(
        &mut self,
        path: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        if let Some(batch) = self.pending.remove(path) {
            batch.discard(runtime);
        }
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
            self.forget_pending(id, runtime);
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
        self.forget_pending(id, runtime);
        for state in self.documents.values_mut() {
            state.kernels.remove(id);
        }
        runtime.retire(id).is_some()
    }

    /// Takes `id` out of whatever change is still preparing it.
    fn forget_pending(&mut self, id: &KernelId, runtime: &mut Runtime) {
        for batch in self.pending.values_mut() {
            batch.items.retain(|item| {
                if &item.id != id {
                    return true;
                }
                if let Prep::Waiting(ticket) = item.prep {
                    runtime.discard(ticket);
                }
                false
            });
        }
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
