//! The session in time: position, history, snapshot and restore. A
//! change lands whole or not at all; a refused one leaves world, laws,
//! structure and history as they were, and records the attempt.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_graph::Report;
use entelechy_kernel::{Context, Expect, FieldData, KernelId, KernelVersion};

use super::Session;
use crate::Event;

impl Session {
    /// Sets the origin of what the session is asked next (a protocol
    /// request's channel while it is answered), returning the previous
    /// one. Records carry it.
    pub fn act_for(&mut self, origin: crate::Origin) -> crate::Origin {
        std::mem::replace(&mut self.backing.docs_mut().origin, origin)
    }

    /// Records the next scene loaded as the current request's change (a
    /// `compose`, a param that resizes the graph).
    pub fn claim_scene(&mut self) {
        self.backing.docs_mut().claim = super::Claim::Asked;
    }

    /// The world's position: its revision (moved by every change but
    /// running) and the frames committed since.
    pub fn position(&self) -> (u64, u64) {
        self.backing.docs().history.position()
    }

    /// The remembered changes after revision `since`, oldest first.
    pub fn changes_since(&self, since: u64) -> Vec<crate::Record> {
        self.backing
            .docs()
            .history
            .changes(since)
            .cloned()
            .collect()
    }

    /// The attempts refused at or after revision `since`, oldest first.
    pub fn attempts_since(&self, since: u64) -> Vec<crate::Attempt> {
        self.backing
            .docs()
            .history
            .attempts(since)
            .cloned()
            .collect()
    }

    /// Why no scene is loaded, when none is: the scene document's last
    /// refusal the history keeps, or that it has not loaded.
    pub fn why_no_scene(&self) -> Option<String> {
        let docs = self.backing.docs();
        if docs.scene.graph().is_some() {
            return None;
        }
        let root = docs.scene.root().display().to_string();
        Some(
            docs.history
                .attempts(0)
                .filter(|attempt| attempt.subject == root)
                .last()
                .map_or_else(
                    || "its document has not loaded".to_owned(),
                    |a| a.why.clone(),
                ),
        )
    }

    /// A Kernel's source at `generation`, from the registry or the
    /// history.
    pub fn source_at(
        &self,
        id: &KernelId,
        generation: entelechy_kernel::Generation,
    ) -> Option<String> {
        match self.runtime.kernel(id) {
            Some(version) if version.generation() == generation => {
                Some(version.source().to_owned())
            }
            _ => self
                .backing
                .docs()
                .history
                .source_of(id, generation)
                .map(str::to_owned),
        }
    }

    /// A scene document a remembered change loaded, by its digest.
    pub fn scene_at(&self, digest: u64) -> Option<String> {
        self.backing
            .docs()
            .history
            .scene_text(digest)
            .map(str::to_owned)
    }

    /// The text the scene's document at `path` (as a compose names it,
    /// [`Session::composable`]) had at version `digest`, when the session
    /// remembers a compose that wrote it, or for the scene, a change that
    /// loaded it.
    pub fn document_at(&self, path: &Path, digest: u64) -> Result<String, String> {
        let landed = self.composable(path)?;
        let docs = self.backing.docs();
        let root = self
            .store()
            .canonical(docs.scene.root())
            .unwrap_or_else(|_| docs.scene.root().to_path_buf());
        docs.history
            .document_text(&landed, digest, landed == root)
            .map(str::to_owned)
            .ok_or_else(|| {
                format!(
                    "version {digest:016x} of `{}` is not one the session remembers",
                    landed.display()
                )
            })
    }

    /// The world as it stands, taken whole: position, published Kernels,
    /// every port's committed state (uncommitted outputs excluded), and
    /// the scene document. Device Fields are brought back first.
    pub fn snapshot(&mut self) -> Result<crate::Snapshot, String> {
        let (revision, frame) = self.backing.docs().history.position();
        let laws = self
            .runtime
            .kernels()
            .filter(|version| version.native().is_none())
            .map(|version| {
                (
                    version.id().clone(),
                    version.generation(),
                    version.source().to_owned(),
                )
            })
            .collect();
        let mut states = Vec::new();
        for (node, port, handle) in self.evaluator.committed_states() {
            self.runtime
                .materialize(handle)
                .map_err(|e| format!("{node}.{port}: {e}"))?;
            let data = self
                .runtime
                .host()
                .data(handle)
                .map_err(|e| format!("{node}.{port}: {e}"))?;
            states.push(crate::StateCopy {
                node: node.to_string(),
                port,
                schema: data.schema().clone(),
                domain: data.domain(),
                words: data.values().into_owned(),
            });
        }
        Ok(crate::Snapshot {
            revision,
            frame,
            laws,
            states,
            scene: self.backing.docs().scene.text().map(str::to_owned),
        })
    }

    /// Restores a snapshot as one change: its scene document written back
    /// ([`crate::Structure::Snapshot`]), its Kernels published together
    /// ([`crate::Laws::Snapshot`]), then its state installed and proved by
    /// one uncommitted frame under `context`
    /// ([`entelechy_graph::Evaluator::restore`]). On failure, the scene
    /// document, the exact running versions, the state and the history
    /// stay as they were, and the attempt is recorded.
    pub fn restore(
        &mut self,
        snapshot: &crate::Snapshot,
        laws: crate::Laws,
        structure: crate::Structure,
        context: &Context,
    ) -> Result<Report, String> {
        // Disk edits are part of the world a refused restore returns to.
        self.sync_settled();
        let (history, scene_kept, kernels_kept, origin) = {
            let docs = self.backing.docs();
            (
                docs.history.clone(),
                docs.scene.keep(),
                docs.kernels.keep(),
                docs.origin.clone(),
            )
        };
        let registry: Vec<(KernelId, Arc<KernelVersion>)> = self
            .runtime
            .kernels()
            .map(|version| (version.id().clone(), Arc::clone(version)))
            .collect();
        let mut scene_before = None;
        let mut published = Vec::new();
        let restored = self.restore_steps(
            snapshot,
            laws,
            structure,
            context,
            &mut scene_before,
            &mut published,
        );
        match restored {
            Ok(report) => {
                let docs = self.backing.docs_mut();
                for id in &published {
                    if let Some(version) = self.runtime.kernel(id) {
                        docs.history.published(
                            id,
                            version.generation(),
                            version.source(),
                            origin.clone(),
                        );
                    }
                }
                docs.history.changed(
                    crate::Change::Restored {
                        revision: snapshot.revision,
                        laws: laws == crate::Laws::Snapshot,
                    },
                    origin,
                );
                Ok(report)
            }
            Err(why) => {
                let mut why = why;
                let root = self.backing.docs().scene.root().to_path_buf();
                if let Some(before) = scene_before
                    && let Err(also) = self.store().write(&root, before.as_bytes())
                {
                    why =
                        format!("{why}; and the scene document could not be written back: {also}");
                }
                let stamp = self.store().stamp(&root);
                {
                    let docs = self.backing.docs_mut();
                    docs.scene.put_back(scene_kept, stamp);
                    docs.kernels.put_back(kernels_kept, &mut self.runtime);
                }
                let mut back: Vec<(KernelId, Option<Arc<KernelVersion>>)> = self
                    .runtime
                    .kernels()
                    .filter(|version| !registry.iter().any(|(id, _)| id == version.id()))
                    .map(|version| (version.id().clone(), None))
                    .collect();
                back.extend(
                    registry
                        .into_iter()
                        .map(|(id, version)| (id, Some(version))),
                );
                self.runtime.reinstate(back);
                self.sync();
                let docs = self.backing.docs_mut();
                docs.history = history;
                docs.history.refused(
                    format!("the snapshot of revision {}", snapshot.revision),
                    why.clone(),
                    None,
                    origin,
                );
                Err(why)
            }
        }
    }

    fn restore_steps(
        &mut self,
        snapshot: &crate::Snapshot,
        laws: crate::Laws,
        structure: crate::Structure,
        context: &Context,
        scene_before: &mut Option<String>,
        published: &mut Vec<KernelId>,
    ) -> Result<Report, String> {
        if structure == crate::Structure::Snapshot {
            let text = snapshot
                .scene
                .as_deref()
                .ok_or("the snapshot holds no scene to restore")?;
            let root = self.backing.docs().scene.root().to_path_buf();
            let before = self
                .store()
                .read_text(&root)
                .map_err(|e| format!("{}: {e}", root.display()))?;
            if before != text {
                self.check_scene(text)
                    .map_err(|e| format!("its scene does not load: {e}"))?;
                *scene_before = Some(before);
                self.claim_scene();
                self.replace_scene(text)?;
            }
        }
        if self.backing.docs().scene.graph().is_none() {
            return Err("no scene is loaded".into());
        }
        if laws == crate::Laws::Snapshot {
            let mut candidates = Vec::new();
            for (id, _, source) in &snapshot.laws {
                let current = self.runtime.kernel(id);
                if current
                    .as_ref()
                    .is_some_and(|version| version.source() == source)
                {
                    continue;
                }
                let candidate = self
                    .prepare_source(source)
                    .map_err(|e| format!("its {id} does not prepare: {e}"))?;
                let expect = match &current {
                    Some(version) => Expect::Generation(version.generation()),
                    None => Expect::Absent,
                };
                published.push(id.clone());
                candidates.push((candidate, expect));
            }
            self.runtime
                .publish_all(candidates)
                .map_err(|e| e.to_string())?;
        }
        let mut states = Vec::new();
        for copy in &snapshot.states {
            let adopted = FieldData::from_values(copy.schema.clone(), copy.domain, &copy.words)
                .and_then(|data| self.runtime.host_mut().adopt(data));
            match adopted {
                Ok(handle) => {
                    states.push((
                        entelechy_graph::NodeId(copy.node.clone().into()),
                        copy.port.clone(),
                        handle,
                    ));
                }
                Err(error) => {
                    for (_, _, handle) in states {
                        let _ = self.runtime.host_mut().release(handle);
                    }
                    return Err(format!("{}.{}: {error}", copy.node, copy.port));
                }
            }
        }
        let docs = self.backing.docs();
        let graph = docs.scene.graph().ok_or("no scene is loaded")?;
        self.evaluator
            .restore(
                graph,
                &mut self.runtime,
                docs.inputs.handles(),
                context,
                docs.scene.params(),
                states,
            )
            .map_err(|error| format!("the restored world does not run: {error}"))
    }

    /// Syncs and waits until the documents being compiled, the scene's
    /// Kernels being prepared, and the scene waiting for them, are
    /// published or refused.
    pub fn sync_settled(&mut self) -> Vec<Event> {
        let mut events = self.sync();
        while self.backing.docs().scene.compiling() {
            self.backing.docs_mut().scene.wait_compiled();
            events.extend(self.sync());
        }
        let mut landed = Vec::new();
        {
            let docs = self.backing.docs_mut();
            docs.scene
                .wait(&mut docs.kernels, &mut self.runtime, &mut landed);
        }
        self.remember(&landed);
        events.extend(landed);
        events
    }

    /// Replaces the scene document whole and reads it, its Kernels
    /// settled; the error is why it was refused.
    pub(super) fn replace_scene(&mut self, text: &str) -> Result<(), String> {
        let root = self.backing.docs().scene.root().to_path_buf();
        self.store()
            .write(&root, text.as_bytes())
            .map_err(|e| format!("{}: {e}", root.display()))?;
        let events = self.sync_settled();
        let refused = events.iter().find(|event| match event {
            Event::SceneRejected(_) => true,
            Event::Rejected { path, .. } => *path == root,
            _ => false,
        });
        match refused {
            Some(refused) => Err(refused.to_string()),
            None => Ok(()),
        }
    }

    /// Records what a sync's events changed or refused. Disk changes are
    /// their document's; a claimed scene ([`Session::claim_scene`]) is the
    /// request's. One sync is at most one scene change.
    pub(super) fn remember(&mut self, events: &[Event]) {
        use super::Claim;
        let waiting_now = self.scene_waiting();
        let superseded = self.backing.docs_mut().scene.take_superseded();
        let (claimant, landed) = match std::mem::take(&mut self.backing.docs_mut().claim) {
            Claim::Asked => (Some(self.backing.docs().origin.clone()), None),
            Claim::Waiting(origin, mut gathered) if !superseded => {
                gathered.extend(events.iter().cloned());
                (Some(origin), Some(gathered))
            }
            Claim::Landed(events) => (None, Some(events)),
            Claim::Waiting(..) | Claim::None => (None, None),
        };
        self.backing.docs_mut().claim = match (&claimant, landed) {
            (Some(origin), gathered) if waiting_now => {
                Claim::Waiting(origin.clone(), gathered.unwrap_or_default())
            }
            (_, Some(events)) => Claim::Landed(events),
            _ => Claim::None,
        };
        let scene_origin = claimant.unwrap_or_else(|| {
            crate::Origin::Document(self.backing.docs().scene.root().to_path_buf())
        });
        let mut scene_loaded = false;
        {
            let docs = self.backing.docs_mut();
            let store = Arc::clone(docs.scene.store());
            let writes = &docs.writes;
            let document = |path: &Path| match writes.author(store.as_ref(), path) {
                Some(output) => crate::Origin::Output(output.to_owned()),
                None => crate::Origin::Document(path.to_path_buf()),
            };
            for event in events {
                match event {
                    Event::Published {
                        path,
                        id,
                        generation,
                    } => {
                        let source = self
                            .runtime
                            .kernel(id)
                            .map(|version| version.source().to_owned())
                            .unwrap_or_default();
                        docs.history
                            .published(id, *generation, &source, document(path));
                    }
                    Event::Retired { path, id } => docs.history.retired(id, document(path)),
                    Event::SceneLoaded => scene_loaded = true,
                    Event::Rejected { path, error } => docs.history.refused(
                        path.display().to_string(),
                        error.to_string(),
                        None,
                        document(path),
                    ),
                    Event::SceneRejected(error) => {
                        let subject = docs.scene.root().display().to_string();
                        docs.history.refused(
                            subject,
                            error.to_string(),
                            None,
                            scene_origin.clone(),
                        );
                    }
                    _ => {}
                }
            }
        }
        if scene_loaded {
            let text = self
                .backing
                .docs()
                .scene
                .text()
                .unwrap_or_default()
                .to_owned();
            let loaded = self.loaded_documents();
            self.backing
                .docs_mut()
                .history
                .scene(&text, &loaded, scene_origin);
        }
    }

    fn loaded_documents(&self) -> Vec<(PathBuf, String)> {
        let store = self.store();
        let docs = self.backing.docs();
        let root = docs.scene.root();
        let mut loaded: Vec<(PathBuf, String)> = docs
            .scene
            .placed()
            .chain(docs.kernels.stamps())
            .filter(|(path, stamp)| *path != root && stamp.is_some())
            .filter_map(|(path, stamp)| {
                let landed = self.composable(&store.whole(path)).ok()?;
                let text = store.read_text(path).ok()?;
                let unchanged = store.stamp(path) == stamp;
                let parses = store.document(path).is_ok();
                (unchanged && parses).then_some((landed, text))
            })
            .collect();
        loaded.sort_by(|a, b| a.0.cmp(&b.0));
        loaded.dedup_by(|a, b| a.0 == b.0);
        loaded
    }
}
