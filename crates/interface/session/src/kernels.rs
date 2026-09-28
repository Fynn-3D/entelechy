//! The ledger of which document published which Kernel. The runtime
//! owns the Kernels; the scene reads the documents.
//!
//! A [`Change`] prepares its documents' Kernels (a running Kernel's
//! replacement off the caller's thread) and swaps them in together, or,
//! if any fails, none, leaving the running ones as they were. A library
//! document is its own change; the root and its placed graphs are one
//! change with the graph they make.

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
    /// Parsed Kernels, which a trace compiles unoptimized.
    asts: BTreeMap<KernelId, Kernel>,
    /// Key of the shared declarations the Kernels compiled with; a
    /// change republishes every Kernel.
    declarations: String,
}

/// One document's Kernels being prepared, swapped in together once all
/// are ready, or none if any fails.
pub(crate) struct Batch {
    items: Vec<Item>,
    /// Kernels the document dropped, retired when the batch publishes.
    leaving: Vec<KernelId>,
    /// The document's prior ledger entry, restored if the batch fails,
    /// so a later fix republishes the whole change.
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
    /// Discards what is still being prepared.
    fn discard(self, runtime: &mut Runtime) {
        for item in self.items {
            if let Prep::Waiting(ticket) = item.prep {
                runtime.discard(ticket);
            }
        }
    }
}

/// Documents' Kernels published together once all are ready, or none;
/// with `scene`, the scene's lowered graph waits on it.
pub(crate) struct Change {
    pub(crate) documents: Vec<(PathBuf, Batch)>,
    pub(crate) scene: bool,
}

/// The ledger of published Kernels by the document that owns each.
#[derive(Default)]
pub(crate) struct Kernels {
    /// Each publishing document's state, the root's included.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    /// Changes being prepared, in order.
    pending: Vec<Change>,
}

impl Kernels {
    /// The document that publishes `id`, if a document does.
    pub(crate) fn owner(&self, id: &KernelId) -> Option<&Path> {
        self.owners.get(id).map(PathBuf::as_path)
    }
}

/// A copy of the ledger to restore whole ([`Kernels::put_back`]).
pub(crate) struct Kept {
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
}

impl Kernels {
    /// A copy to restore, beside the runtime's registry, when a
    /// published change is refused.
    pub(crate) fn keep(&self) -> Kept {
        Kept {
            documents: self.documents.clone(),
            owners: self.owners.clone(),
        }
    }

    /// Restores what [`Kernels::keep`] kept, discarding pending changes.
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

    /// Records `path` as read at `stamp` without publishing, so an
    /// unreadable document is reread only when it changes.
    pub(crate) fn mark_read(&mut self, path: PathBuf, stamp: Option<Stamp>) {
        self.documents.entry(path).or_default().stamp = stamp;
    }

    /// `error` located in the known document that declares what it
    /// could not find.
    pub(crate) fn located(
        &self,
        store: &dyn Store,
        error: Diagnostic,
        document: &Path,
        root: &Path,
    ) -> Diagnostic {
        located(store, error, document, root, &self.documents())
    }

    /// Queues `change` and publishes every ready change; returns the
    /// scene's outcome as [`Kernels::collect`] does.
    pub(crate) fn make(
        &mut self,
        change: Change,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        self.pending.push(change);
        self.collect(runtime, events)
    }

    /// Publishes each change whose Kernels are all prepared; a change
    /// with a failure publishes nothing. Returns the scene's outcome, if
    /// decided: `true` published, `false` refused.
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
                // Refused items leave the batch; the rest are held back.
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

    /// Waits until every pending change is published or refused; returns
    /// the scene's outcome as [`Kernels::collect`] does.
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

    /// Refuses an unmade change; its documents stay as they were.
    pub(crate) fn refuse(
        &mut self,
        change: Change,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        self.abandon(change, false, runtime, events);
    }

    /// Abandons the scene's pending change for a newer one.
    pub(crate) fn supersede_scene(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let (scene, rest): (Vec<Change>, Vec<Change>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|change| change.scene);
        self.pending = rest;
        for change in scene {
            self.abandon(change, false, runtime, events);
        }
    }

    /// Swaps a ready change in at once and retires dropped Kernels; if
    /// the registry refuses, nothing changes. Returns the scene's outcome.
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

    /// Drops a failed change: publishes nothing, discards pending
    /// preparation, and restores each document's state at its new stamp.
    /// A library change that held back more than the failure (or `held`:
    /// refused whole by the registry) reports it; the scene reports its
    /// own. Returns the scene's outcome.
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

    /// Publishes a document's Kernels as one change: retires dropped
    /// ones, keeps unchanged ones, republishes changed ones. If any fails
    /// to compile or prepare, nothing is published or retired.
    /// `declarations` keys what the Kernels inherit; `root` is for
    /// diagnostics. Returns the scene's outcome ([`Kernels::collect`]).
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

    /// Compiles a document's Kernels and starts preparing them, for a
    /// [`Change`] to publish. `None` if any is refused; the document then
    /// stays as it was.
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
        // A newer library change supersedes its pending one; the scene
        // supersedes its own changes whole.
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
        let previous = before.kernels.clone();
        let same_declarations = before.declarations == declarations;
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
            // A graph placed more than once defines its Kernels once.
            match kernels.get(&id) {
                Some(seen) if *seen == text => continue,
                Some(_) => {
                    events.push(Event::Rejected {
                        path: path.to_path_buf(),
                        error: Diagnostic::new(format!(
                            "{id} is defined twice, differently; kernel ids are unique"
                        )),
                    });
                    failed = true;
                    continue;
                }
                None => {}
            }
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
            let same_text = previous.get(&id) == Some(&text);
            if same_text && same_declarations && runtime.kernel(&id).is_some() {
                continue;
            }
            let module = match orla_compiler::compile_kernel(&kernel, Some(&id)) {
                Ok(module) => module,
                Err(error) => {
                    // One fault in one place is reported once, in the
                    // document it is written in.
                    let rejected = Event::Rejected {
                        path: crate::sources::written_in(store, path, &error),
                        error: self.located(store, (&error).into(), path, root),
                    };
                    if !events.contains(&rejected) {
                        events.push(rejected);
                    }
                    failed = true;
                    continue;
                }
            };
            // Its text the same and its IR too: what changed around it
            // does not reach it, and it is not prepared again.
            if same_text
                && runtime
                    .kernel(&id)
                    .is_some_and(|running| running.module() == Some(&module))
            {
                continue;
            }
            let prep = match runtime.kernel(&id) {
                // A running Kernel's replacement prepares in the background.
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

    /// Retires a document's Kernels and forgets the document.
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

    /// Retires a Kernel of any owner; returns whether it was registered.
    pub(crate) fn retire_kernel(&mut self, id: &KernelId, runtime: &mut Runtime) -> bool {
        self.owners.remove(id);
        self.forget_pending(id, runtime);
        for state in self.documents.values_mut() {
            state.kernels.remove(id);
        }
        runtime.retire(id).is_some()
    }

    /// Removes `id` from any pending change.
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

    /// The Kernel's source compiled unoptimized; `None` when no document
    /// owns it or it fails to compile alone.
    pub(crate) fn unoptimized(&self, id: &KernelId) -> Option<orla_kernel::Validated> {
        let kernel = self
            .documents
            .values()
            .find_map(|state| state.asts.get(id))?;
        orla_compiler::compile_kernel_unoptimized(kernel, Some(id)).ok()
    }
}
