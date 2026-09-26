//! Which document published which Kernel. The ledger owns that and
//! nothing else: the Kernels themselves are the runtime's, which the
//! session hands in, and what a document says is read by the scene.
//!
//! What changes, changes as one [`Change`]: the Kernels of the documents
//! it reads, prepared (a running Kernel's replacement off the caller's
//! thread), then swapped in together, or, when any of them fails, none,
//! and every Kernel they had goes on running as it was. A library
//! document is a change of its own; the scene's documents, the root and
//! the graphs it places, are one change with the graph they make, which
//! the scene puts in place when the change publishes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_kernel::{Candidate, Expect, KernelId, Runtime, Ticket};
use orla_language::Kernel;

use crate::sources::located;
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

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
pub(crate) struct Batch {
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

/// One change: its documents' Kernels, published together once every
/// one is ready, or none of them; with `scene`, it is the scene's own,
/// and the graph the scene lowered waits on it.
pub(crate) struct Change {
    pub(crate) documents: Vec<(PathBuf, Batch)>,
    pub(crate) scene: bool,
}

/// The ledger of published Kernels by the document that owns each.
#[derive(Default)]
pub(crate) struct Kernels {
    /// Kernel text by id for every document that published Kernels; the
    /// root document is one of them.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    /// The changes still being prepared, in the order they were made.
    pending: Vec<Change>,
}

/// Which document published which Kernel, kept to be put back whole
/// ([`Kernels::put_back`]).
pub(crate) struct Kept {
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
}

impl Kernels {
    /// The ledger as it is, to put back when a change is refused after
    /// it published: the runtime's own registry is put back beside it.
    pub(crate) fn keep(&self) -> Kept {
        Kept {
            documents: self.documents.clone(),
            owners: self.owners.clone(),
        }
    }

    /// Puts back the ledger [`Kernels::keep`] kept, letting go of every
    /// change still being prepared.
    pub(crate) fn put_back(&mut self, kept: Kept, runtime: &mut Runtime) {
        for change in std::mem::take(&mut self.pending) {
            for (_, batch) in change.documents {
                batch.discard(runtime);
            }
        }
        self.documents = kept.documents;
        self.owners = kept.owners;
    }

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
    pub(crate) fn located(
        &self,
        store: &dyn Store,
        error: Diagnostic,
        document: &Path,
        root: &Path,
    ) -> Diagnostic {
        located(store, error, document, root, &self.documents())
    }

    /// Makes `change`, and publishes what of it and of every change before
    /// it is ready: the outcome of the scene's change, when it had one
    /// and it came to one ([`Kernels::collect`]).
    pub(crate) fn make(
        &mut self,
        change: Change,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        self.pending.push(change);
        self.collect(runtime, events)
    }

    /// Publishes every change whose Kernels the worker has all finished,
    /// each as one; a change of which one failed publishes nothing. The
    /// outcome of the scene's change, when it came to one: `true` when it
    /// published, `false` when it was refused.
    pub(crate) fn collect(
        &mut self,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        let mut scene = None;
        for mut change in std::mem::take(&mut self.pending) {
            let mut failed = Vec::new();
            let mut waiting = false;
            for (path, batch) in &mut change.documents {
                // What the worker refused is gone from the batch; what is
                // left is what the refusal holds back.
                batch.items.retain_mut(|item| {
                    let Prep::Waiting(ticket) = item.prep else {
                        return true;
                    };
                    match runtime.poll(ticket) {
                        None => {
                            waiting = true;
                            true
                        }
                        Some(Ok(candidate)) => {
                            item.prep = Prep::Ready(Box::new(candidate));
                            true
                        }
                        Some(Err(error)) => {
                            failed.push(Event::Rejected {
                                path: path.clone(),
                                error: Diagnostic::new(format!("{}: {error}", item.id)),
                            });
                            false
                        }
                    }
                });
            }
            let outcome = if !failed.is_empty() {
                events.extend(failed);
                self.abandon(change, false, runtime, events)
            } else if waiting {
                self.pending.push(change);
                continue;
            } else {
                self.publish(change, runtime, events)
            };
            scene = scene.or(outcome);
        }
        scene
    }

    /// Waits until every change being prepared has been published or
    /// refused: what must stand before a world that depends on it runs.
    /// The outcome of the scene's change, as [`Kernels::collect`] says.
    pub(crate) fn wait(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) -> Option<bool> {
        let mut scene = None;
        while !self.pending.is_empty() {
            scene = scene.or(self.collect(runtime, events));
            if !self.pending.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        scene
    }

    /// Refuses a change before it is made: nothing of it is published,
    /// and each of its documents is held as it was.
    pub(crate) fn refuse(
        &mut self,
        change: Change,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        self.abandon(change, false, runtime, events);
    }

    /// Lets go of the scene's change still being prepared, superseded by
    /// a newer one: nothing of it is published.
    pub(crate) fn supersede_scene(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let (scene, rest): (Vec<Change>, Vec<Change>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|change| change.scene);
        self.pending = rest;
        for change in scene {
            self.abandon(change, false, runtime, events);
        }
    }

    /// Swaps a ready change in, all at once, and retires what its
    /// documents no longer define; refused by the registry, nothing of
    /// it is. The scene's outcome, when it was the scene's change.
    fn publish(
        &mut self,
        change: Change,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        let Change { documents, scene } = change;
        let mut ready = Vec::new();
        let mut from = Vec::new();
        let mut held = Vec::new();
        for (path, batch) in documents {
            let Batch {
                items,
                leaving,
                before,
            } = batch;
            for item in items {
                match item.prep {
                    Prep::Ready(candidate) => {
                        ready.push((*candidate, item.expect));
                        from.push(path.clone());
                    }
                    Prep::Waiting(ticket) => runtime.discard(ticket),
                }
            }
            held.push((path, leaving, before));
        }
        match runtime.publish_all(ready) {
            Ok(versions) => {
                for (version, path) in versions.into_iter().zip(from) {
                    events.push(Event::Published {
                        path: path.clone(),
                        id: version.id().clone(),
                        generation: version.generation(),
                    });
                    self.owners.insert(version.id().clone(), path);
                }
                for (path, leaving, _) in held {
                    for id in &leaving {
                        self.retire(&path, id, runtime, events);
                    }
                }
                scene.then_some(true)
            }
            Err(error) => {
                events.push(Event::Rejected {
                    path: held
                        .first()
                        .map(|(path, _, _)| path.clone())
                        .unwrap_or_default(),
                    error: Diagnostic::new(error.to_string()),
                });
                let documents = held
                    .into_iter()
                    .map(|(path, leaving, before)| {
                        let batch = Batch {
                            items: Vec::new(),
                            leaving,
                            before,
                        };
                        (path, batch)
                    })
                    .collect();
                self.abandon(Change { documents, scene }, true, runtime, events)
            }
        }
    }

    /// A change that failed: nothing of it is published, what it still
    /// prepares is let go, and each of its documents is held as it was,
    /// read at its new stamp so it is read again when it changes. A
    /// library document's change that held back more than what failed
    /// (or, `held`, a change the registry refused whole) says so; the
    /// scene's says it with its refusal. The scene's outcome, when it was
    /// the scene's change.
    fn abandon(
        &mut self,
        change: Change,
        held: bool,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        let Change { documents, scene } = change;
        for (path, batch) in documents {
            let held = !scene && (held || !batch.items.is_empty() || !batch.leaving.is_empty());
            let mut before = batch.before.clone();
            if let Some(state) = self.documents.get(&path) {
                before.stamp = state.stamp;
            }
            batch.discard(runtime);
            self.documents.insert(path.clone(), before);
            if held {
                events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(
                        "nothing in it was published: a document's Kernels change together, and the ones it had go on running",
                    ),
                });
            }
        }
        scene.then_some(false)
    }

    /// Publishes the Kernels a document now defines, as one change:
    /// retires the ones that left, keeps the unchanged, republishes the
    /// changed, and reports each. When any of them does not compile or
    /// prepare, none is published and none retired. `declarations` keys
    /// what the Kernels inherit; a change there republishes them all.
    /// `root` is the scene, for diagnostics. What was made publishes with
    /// what else is ready, the scene's change among it, whose outcome it
    /// answers ([`Kernels::collect`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish_document(
        &mut self,
        store: &dyn Store,
        path: &Path,
        current: Option<Stamp>,
        defined: Vec<Defined>,
        declarations: &str,
        root: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        let staged = self.stage_document(
            store,
            path,
            current,
            defined,
            declarations,
            root,
            runtime,
            events,
        );
        if let Some(batch) = staged {
            let change = Change {
                documents: vec![(path.to_path_buf(), batch)],
                scene: false,
            };
            return self.make(change, runtime, events);
        }
        None
    }

    /// A document's change compiled and handed to be prepared, nothing of
    /// it published: what [`Kernels::publish_document`] publishes alone,
    /// and a scene's documents together, as one [`Change`].
    /// `None` when any of its Kernels was refused, the document held
    /// back as it was.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage_document(
        &mut self,
        store: &dyn Store,
        path: &Path,
        current: Option<Stamp>,
        defined: Vec<Defined>,
        declarations: &str,
        root: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<Batch> {
        // A newer change of a library document supersedes one still being
        // prepared (the scene's own are superseded whole, by the scene).
        if let Some(at) = self
            .pending
            .iter()
            .position(|change| !change.scene && change.documents.iter().any(|(p, _)| p == path))
        {
            for (other, batch) in self.pending.remove(at).documents {
                let before = batch.before.clone();
                batch.discard(runtime);
                self.documents.insert(other, before);
            }
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
                    error: Diagnostic::new(format!(
                        "{id} is already published by {}; kernel ids are unique",
                        owner.display()
                    )),
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
                        error: self.located(store, (&error).into(), path, root),
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
                        error: Diagnostic::new(error.to_string()),
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
            let change = Change {
                documents: vec![(path.to_path_buf(), batch)],
                scene: false,
            };
            self.abandon(change, false, runtime, events);
            return None;
        }
        Some(batch)
    }

    /// Retires every Kernel a document published, and forgets it.
    pub(crate) fn retire_document(
        &mut self,
        path: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        for change in &mut self.pending {
            let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut change.documents)
                .into_iter()
                .partition(|(p, _)| p == path);
            change.documents = kept;
            for (_, batch) in gone {
                batch.discard(runtime);
            }
        }
        self.pending
            .retain(|change| !change.documents.is_empty() || change.scene);
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
        let batches = self
            .pending
            .iter_mut()
            .flat_map(|change| &mut change.documents);
        for (_, batch) in batches {
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
    pub(crate) fn unoptimized(&self, id: &KernelId) -> Option<orla_kernel::Validated> {
        let kernel = self
            .documents
            .values()
            .find_map(|state| state.asts.get(id))?;
        orla_compiler::compile_kernel_unoptimized(kernel, Some(id)).ok()
    }
}
