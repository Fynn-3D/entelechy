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
use std::sync::Arc;

use entelechy_ir::Validated;
use entelechy_kernel::{Candidate, Expect, KernelId, KernelVersion, Runtime, Ticket};
use entelechy_language::Kernel;

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
    /// Parsed Kernels, which a trace compiles unoptimized, each holding
    /// only the declarations it reaches ([`Kernel::trimmed`]).
    asts: BTreeMap<KernelId, Kernel>,
    /// Each Kernel's key of the declarations it reaches
    /// ([`crate::sources::reached_key`]); a Kernel whose text and key
    /// are as they were is not compiled again.
    reached: BTreeMap<KernelId, String>,
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
    /// Kernels compiled, and prepared when new, ahead of staging this
    /// sync ([`Kernels::prime`]).
    primed: BTreeMap<KernelId, Primed>,
    /// What each Kernel [`Kernels::prime`] saw this sync reaches, by its
    /// id and text: staging the same Kernel finds it here.
    reaches_now: BTreeMap<(KernelId, String), String>,
    /// The Kernels a document defines that its last change could not
    /// publish, each with the first reason it was refused (`path:line:col:
    /// message`); gone once the Kernel is published.
    refused: BTreeMap<KernelId, String>,
}

/// A Kernel compiled ahead of staging, from its text and what it reaches,
/// and prepared when it is new, or replaces a running one with no worker
/// to prepare it.
struct Primed {
    text: String,
    reaches: String,
    module: Validated,
    candidate: Option<Candidate>,
}

impl Kernels {
    /// Why `id`'s document could not publish it, when it was refused.
    pub(crate) fn refused(&self, id: &KernelId) -> Option<&str> {
        self.refused.get(id).map(String::as_str)
    }

    /// The document that publishes `id`, if a document does.
    pub(crate) fn owner(&self, id: &KernelId) -> Option<&Path> {
        self.owners.get(id).map(PathBuf::as_path)
    }
}

/// What priming reads of the ledger and the runtime
/// ([`Kernels::ledger`]): the documents' states as published, the running
/// versions, and how to prepare. It is the ledger's at one moment, so it
/// primes on any thread.
pub(crate) struct Ledger {
    documents: BTreeMap<PathBuf, DocumentState>,
    running: BTreeMap<KernelId, Arc<KernelVersion>>,
    preparer: entelechy_kernel::Preparer,
    worker: bool,
    /// Whether Kernels prepare in turns (a page): then priming only
    /// compiles, and staging queues the new ones for their turn.
    turns: bool,
}

/// Kernels primed from a [`Ledger`], for the ledger to take
/// ([`Kernels::take_made`]).
pub(crate) struct Made {
    primed: Vec<(KernelId, Primed)>,
    reaches: Vec<((KernelId, String), String)>,
}

impl Ledger {
    /// Primes `documents` as [`Kernels::prime`] does.
    pub(crate) fn prime(&self, store: &dyn Store, documents: &[(&Path, &[Defined])]) -> Made {
        let mut reaches_now = Vec::new();
        let mut jobs: Vec<(&Path, &Defined, String, Option<Arc<KernelVersion>>)> = Vec::new();
        for (path, defined) in documents {
            let before = self.documents.get(*path);
            for d in defined.iter() {
                if jobs.iter().any(|(_, seen, _, _)| seen.id == d.id) {
                    continue;
                }
                let reaches = crate::sources::reached_key(store, &d.kernel);
                reaches_now.push(((d.id.clone(), d.text.clone()), reaches.clone()));
                let running = self.running.get(&d.id).cloned();
                let unchanged = before.is_some_and(|before| {
                    before.kernels.get(&d.id) == Some(&d.text)
                        && before.reached.get(&d.id) == Some(&reaches)
                });
                if running.is_some() && unchanged {
                    continue;
                }
                jobs.push((path, d, reaches, running));
            }
        }
        if jobs.len() < 2 {
            return Made {
                primed: Vec::new(),
                reaches: reaches_now,
            };
        }
        let (preparer, worker, turns) = (&self.preparer, self.worker, self.turns);
        let made = crate::spread::map(&jobs, |(path, d, reaches, running)| {
            let module = compiled(store, path, &d.id, &d.text, reaches, &d.kernel).ok()?;
            let candidate = match running {
                None if turns => None,
                None => preparer.prepare(module.clone(), &d.text).ok(),
                Some(running) if !worker => preparer
                    .prepare_replacing(module.clone(), &d.text, running)
                    .ok(),
                Some(_) => None,
            };
            Some((
                d.id.clone(),
                Primed {
                    text: d.text.clone(),
                    reaches: reaches.clone(),
                    module,
                    candidate,
                },
            ))
        });
        Made {
            primed: made.into_iter().flatten().collect(),
            reaches: reaches_now,
        }
    }
}

/// A copy of the ledger to restore whole ([`Kernels::put_back`]).
pub(crate) struct Kept {
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
}

impl Kernels {
    /// Compiles, and prepares when new or when no worker prepares
    /// replacements ([`Runtime::background`]), every Kernel of `documents`
    /// that staging would compile, on as many threads as there are cores, so
    /// staging finds each ready ([`Kernels::stage_document`]). What it
    /// finds is what it would have made: compiling and preparing a module
    /// read nothing but the module. A Kernel that fails is left for
    /// staging, which says why; what staging does not take is let go by
    /// [`Kernels::unprime`].
    pub(crate) fn prime(
        &mut self,
        store: &dyn Store,
        documents: &[(&Path, &[Defined])],
        runtime: &Runtime,
    ) {
        let made = self
            .ledger(documents.iter().map(|(path, _)| *path), runtime)
            .prime(store, documents);
        self.take_made(made);
    }

    /// What priming `paths` reads of the ledger and the runtime, to prime
    /// on another thread ([`Ledger::prime`]).
    pub(crate) fn ledger<'a>(
        &self,
        paths: impl Iterator<Item = &'a Path>,
        runtime: &Runtime,
    ) -> Ledger {
        Ledger {
            documents: paths
                .filter_map(|path| Some((path.to_path_buf(), self.documents.get(path)?.clone())))
                .collect(),
            running: runtime
                .kernels()
                .map(|version| (version.id().clone(), Arc::clone(version)))
                .collect(),
            preparer: runtime.preparer(),
            worker: runtime.background(),
            turns: runtime.prepares_in_turns(),
        }
    }

    /// Takes what [`Ledger::prime`] made, for staging to find.
    pub(crate) fn take_made(&mut self, made: Made) {
        self.reaches_now.extend(made.reaches);
        self.primed.extend(made.primed);
    }

    /// Lets go of what [`Kernels::prime`] made and staging did not take.
    pub(crate) fn unprime(&mut self) {
        self.primed.clear();
        self.reaches_now.clear();
    }

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

    /// The Kernels document `path` has published.
    pub(crate) fn ids_of(&self, path: &Path) -> Vec<KernelId> {
        self.documents
            .get(path)
            .map(|state| state.kernels.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Every document that has published, or tried to.
    pub(crate) fn documents(&self) -> Vec<PathBuf> {
        self.documents.keys().cloned().collect()
    }

    /// Every document with the stamp it was last read at.
    pub(crate) fn stamps(&self) -> impl Iterator<Item = (&Path, Option<Stamp>)> {
        self.documents
            .iter()
            .map(|(path, state)| (path.as_path(), state.stamp))
    }

    /// Whether `path` was last read at `stamp`.
    pub(crate) fn read_at(&self, path: &Path, stamp: Option<Stamp>) -> bool {
        self.documents
            .get(path)
            .is_some_and(|state| state.stamp == stamp)
    }

    /// Whether `path` was read before.
    pub(crate) fn has_read(&self, path: &Path) -> bool {
        self.documents.contains_key(path)
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
        // A scene publishes after the documents' changes queued before it,
        // whose Kernels it may bind: where those still prepare (in turns, in
        // a page), it waits with them.
        let mut before_waiting = false;
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
            } else if waiting || (change.scene && before_waiting) {
                before_waiting |= !change.scene;
                self.pending.push(change);
                continue;
            } else {
                self.publish(change, runtime, events)
            };
            scene = scene.or(outcome);
        }
        scene
    }

    /// Whether a change is still being prepared, to publish at a later sync.
    pub(crate) fn preparing(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Waits until every pending change is published or refused; returns
    /// the scene's outcome as [`Kernels::collect`] does.
    pub(crate) fn wait(&mut self, runtime: &mut Runtime, events: &mut Vec<Event>) -> Option<bool> {
        let mut scene = None;
        while !self.pending.is_empty() {
            scene = scene.or(self.collect(runtime, events));
            if !self.pending.is_empty() {
                // In turns (a page, where no thread sleeps), what is waited
                // for is prepared now.
                if runtime.prepares_in_turns() {
                    runtime.prepare_turn(usize::MAX);
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
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
                    self.refused.remove(version.id());
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
        root: &Path,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Option<bool> {
        let staged = self.stage_document(store, path, current, defined, root, runtime, events);
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
        let leaving: Vec<KernelId> = before
            .kernels
            .keys()
            .filter(|id| !defined.iter().any(|d| &d.id == *id))
            .cloned()
            .collect();
        let mut kernels = BTreeMap::new();
        let mut reached = BTreeMap::new();
        let mut asts = BTreeMap::new();
        let mut items = Vec::new();
        let mut failed = false;
        let said = events.len();
        let ids: Vec<KernelId> = defined.iter().map(|d| d.id.clone()).collect();
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
            asts.insert(id.clone(), kernel.trimmed());
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
            // What it reaches of what it inherits: a declaration it does
            // not reach changing changes nothing it compiles to.
            let reaches = match self.reaches_now.get(&(id.clone(), text.clone())) {
                Some(reaches) => reaches.clone(),
                None => crate::sources::reached_key(store, &kernel),
            };
            let same_reach = before.reached.get(&id) == Some(&reaches);
            reached.insert(id.clone(), reaches.clone());
            if same_text && same_reach && runtime.kernel(&id).is_some() {
                continue;
            }
            let primed = self
                .primed
                .remove(&id)
                .filter(|primed| primed.text == text && primed.reaches == reaches);
            let (module, candidate) = match primed {
                Some(primed) => (Ok(primed.module), primed.candidate),
                None => (compiled(store, path, &id, &text, &reaches, &kernel), None),
            };
            let module = match module {
                Ok(module) => module,
                Err(error) => {
                    // One fault in one place is reported once, in the
                    // document it is written in.
                    let at = crate::sources::written_in(
                        path,
                        entelechy_language::Held::of(&kernel),
                        &error,
                    );
                    let rejected = Event::Rejected {
                        error: self.located(
                            store,
                            crate::sources::said(entelechy_language::Held::of(&kernel), &error),
                            &at,
                            root,
                        ),
                        path: at,
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
                // A running Kernel's replacement made ahead with no worker.
                Some(running)
                    if !runtime.background()
                        && candidate
                            .as_ref()
                            .is_some_and(|candidate| candidate.module() == &module) =>
                {
                    items.push(Item {
                        id,
                        prep: Prep::Ready(Box::new(candidate.expect("made ahead"))),
                        expect: Expect::Generation(running.generation()),
                    });
                    continue;
                }
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
                // A new Kernel, where Kernels prepare in turns (a page),
                // waits its turn too: a scene's first load is never one
                // long preparation.
                None if runtime.prepares_in_turns()
                    && !candidate
                        .as_ref()
                        .is_some_and(|candidate| candidate.module() == &module) =>
                {
                    let ticket = runtime.prepare_later(module, text);
                    items.push(Item {
                        id,
                        prep: Prep::Waiting(ticket),
                        expect: Expect::Absent,
                    });
                    continue;
                }
                None => match candidate.filter(|candidate| candidate.module() == &module) {
                    Some(candidate) => Ok(candidate),
                    None => runtime.prepare_module(module, &text),
                },
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
                reached,
            },
        );
        if failed {
            // Each of its Kernels not running is refused for the first
            // reason this document was.
            let why = events[said..].iter().find_map(|event| match event {
                Event::Rejected { path, error } => Some(match error.at {
                    Some(_) => format!("{}:{error}", path.display()),
                    None => format!("{}: {error}", path.display()),
                }),
                _ => None,
            });
            if let Some(why) = why {
                for id in ids.iter().filter(|id| runtime.kernel(id).is_none()) {
                    self.refused.insert(id.clone(), why.clone());
                }
            }
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

    /// The Kernel its document holds compiled unoptimized, where each
    /// value is written, and the document it is written in, when the
    /// version running is the one the document wrote (`source`, the
    /// version's); `None` when no document owns it, a source published
    /// apart replaced it, or it fails to compile alone.
    pub(crate) fn unoptimized(&self, id: &KernelId, source: &str) -> Option<Written> {
        let (path, kernel) = self.documents.iter().find_map(|(path, state)| {
            (state.kernels.get(id).map(String::as_str) == Some(source))
                .then(|| state.asts.get(id).map(|kernel| (path, kernel)))
                .flatten()
        })?;
        Written::compiled(kernel, id, Some(path.clone()))
    }
}

/// A Kernel as its document writes it: unoptimized, with where each value
/// is written.
pub(crate) struct Written {
    pub(crate) module: entelechy_kernel::Validated,
    pub(crate) places: entelechy_compiler::SourcePlaces,
    /// The document it is written in; `None` for a source published apart,
    /// which no file holds.
    pub(crate) path: Option<PathBuf>,
    /// The Kernel, what it holds knowing where each was written.
    pub(crate) kernel: entelechy_language::Kernel,
}

impl Written {
    /// Where the failure a trace found is written: its value in the
    /// function it failed in (`None` the entry).
    /// `kernel`, published as `id`, compiled unoptimized, written in
    /// `path`.
    pub(crate) fn compiled(
        kernel: &entelechy_language::Kernel,
        id: &KernelId,
        path: Option<PathBuf>,
    ) -> Option<Written> {
        let (module, places) =
            entelechy_compiler::compile_kernel_unoptimized(kernel, Some(id)).ok()?;
        Some(Written {
            module,
            places,
            path,
            kernel: kernel.clone(),
        })
    }

    pub(crate) fn place_of(&self, fault: &crate::TraceFault) -> Option<crate::SourcePlace> {
        let value = fault.value?;
        let (places, path) = match &fault.function {
            None => (&self.places.entry, self.path.clone()?),
            Some(name) => {
                let index = self.module.functions.iter().position(|f| &f.name == name)?;
                (
                    self.places.functions.get(index)?,
                    entelechy_language::Held::of(&self.kernel)
                        .origin_of(name)
                        .map_or_else(
                            || self.path.clone(),
                            |origin| Some(PathBuf::from(&*origin.document)),
                        )?,
                )
            }
        };
        let span = places.get(&value)?;
        span.is_placed().then(|| crate::SourcePlace {
            path,
            line: span.line,
            column: span.column,
            function: fault.function.as_deref().map(|name| {
                entelechy_language::Held::of(&self.kernel)
                    .origin_of(name)
                    .map_or_else(|| name.to_owned(), |origin| origin.name.clone())
            }),
        })
    }
}

/// Kernels compiled, by the document, id, text and inherited declarations
/// ([`crate::sources::declarations_key`]) each was compiled from: the
/// same four compile to the same module, so a session reading the same
/// documents through the store as one before does not compile them again.
pub(crate) type Compiled = std::collections::HashMap<(PathBuf, String, String, String), Validated>;

/// A Kernel of the document at `path`, written in `text` and resolved
/// against its declarations, compiled as a session compiles it
/// ([`compiled`]: by its id, its own text and the declarations it
/// reaches), so a check and a session reading the same documents through
/// the store compile it once. One without an id is compiled as it is, and
/// says why.
pub(crate) fn compiled_in(
    store: &dyn Store,
    path: &Path,
    text: &str,
    kernel: &Kernel,
) -> Result<Validated, entelechy_compiler::CompileError> {
    let Ok(id) = crate::sources::named_id(kernel) else {
        return entelechy_compiler::compile_kernel(kernel, None);
    };
    let own = &text[entelechy_language::doc::with_doc(text, kernel.range.clone())];
    compiled(
        store,
        path,
        &id,
        own,
        &crate::sources::reached_key(store, kernel),
        kernel,
    )
}

/// A Kernel compiled, or the module it compiled to before from the same
/// document, id, text and declarations. A refusal is not kept.
fn compiled(
    store: &dyn Store,
    path: &Path,
    id: &KernelId,
    text: &str,
    declarations: &str,
    kernel: &entelechy_language::Kernel,
) -> Result<Validated, entelechy_compiler::CompileError> {
    let key = (
        path.to_path_buf(),
        id.to_string(),
        text.to_owned(),
        declarations.to_owned(),
    );
    let lock = |parsed: &crate::store::Parsed| {
        parsed
            .compiled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned()
    };
    if let Some(module) = store.parsed().and_then(lock) {
        return Ok(module);
    }
    let module = entelechy_compiler::compile_kernel(kernel, Some(id))?;
    if let Some(parsed) = store.parsed() {
        let mut kept = parsed
            .compiled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A long session's edits leave modules behind; they go when there
        // are many.
        if kept.len() >= 4096 {
            kept.clear();
        }
        kept.insert(key, module.clone());
    }
    Ok(module)
}
