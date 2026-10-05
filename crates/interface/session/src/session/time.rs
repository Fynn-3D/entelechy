//! The session in time: position, history, snapshot and restore. A
//! change lands whole or not at all; a refused one leaves world, laws,
//! structure and history as they were, and records the attempt.

use std::path::Path;
use std::sync::Arc;

use orla_graph::Report;
use orla_kernel::{Context, Expect, FieldData, KernelId, KernelVersion};

use super::Session;
use crate::Event;

impl Session {
    /// Sets the origin of what the session is asked next (a protocol
    /// request's channel while it is answered), returning the previous
    /// one. Records carry it.
    pub fn act_for(&mut self, origin: crate::Origin) -> crate::Origin {
        std::mem::replace(&mut self.origin, origin)
    }

    /// Records the next scene loaded as the current request's change (a
    /// `compose`, a param that resizes the graph).
    pub fn claim_scene(&mut self) {
        self.claim = super::Claim::Asked;
    }

    /// The world's position: its revision (moved by every change but
    /// running) and the frames committed since.
    pub fn position(&self) -> (u64, u64) {
        self.history.position()
    }

    /// The remembered changes after revision `since`, oldest first.
    pub fn changes_since(&self, since: u64) -> Vec<crate::Record> {
        self.history.changes(since).cloned().collect()
    }

    /// The attempts refused at or after revision `since`, oldest first.
    pub fn attempts_since(&self, since: u64) -> Vec<crate::Attempt> {
        self.history.attempts(since).cloned().collect()
    }

    /// Why no scene is loaded, when none is: the scene document's last
    /// refusal the history keeps, or that it has not loaded.
    pub fn why_no_scene(&self) -> Option<String> {
        if self.scene.graph().is_some() {
            return None;
        }
        let root = self.scene.root().display().to_string();
        Some(
            self.history
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
    pub fn source_at(&self, id: &KernelId, generation: orla_kernel::Generation) -> Option<String> {
        match self.runtime.kernel(id) {
            Some(version) if version.generation() == generation => {
                Some(version.source().to_owned())
            }
            _ => self.history.source_of(id, generation).map(str::to_owned),
        }
    }

    /// A scene document a remembered change loaded, by its digest.
    pub fn scene_at(&self, digest: u64) -> Option<String> {
        self.history.scene_text(digest).map(str::to_owned)
    }

    /// The world as it stands, taken whole: position, published Kernels,
    /// every port's committed state (uncommitted outputs excluded), and
    /// the scene document. Device Fields are brought back first.
    pub fn snapshot(&mut self) -> Result<crate::Snapshot, String> {
        let (revision, frame) = self.history.position();
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
            scene: self.scene.text().map(str::to_owned),
        })
    }

    /// Restores a snapshot as one change: its scene document written back
    /// ([`crate::Structure::Snapshot`]), its Kernels published together
    /// ([`crate::Laws::Snapshot`]), then its state installed and proved by
    /// one uncommitted frame under `context`
    /// ([`orla_graph::Evaluator::restore`]). On failure, the scene
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
        let history = self.history.clone();
        // Kept in memory: the text on disk may not load.
        let scene_kept = self.scene.keep();
        let kernels_kept = self.kernels.keep();
        // Every running version, to be put back exactly.
        let registry: Vec<(KernelId, Arc<KernelVersion>)> = self
            .runtime
            .kernels()
            .map(|version| (version.id().clone(), Arc::clone(version)))
            .collect();
        let origin = self.origin.clone();
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
                // Record the laws the restore published.
                for id in &published {
                    if let Some(version) = self.runtime.kernel(id) {
                        self.history.published(
                            id,
                            version.generation(),
                            version.source(),
                            origin.clone(),
                        );
                    }
                }
                self.history.changed(
                    crate::Change::Restored {
                        revision: snapshot.revision,
                        laws: laws == crate::Laws::Snapshot,
                    },
                    origin,
                );
                Ok(report)
            }
            Err(why) => {
                // Write the document's bytes back; put the scene, ledger
                // and running versions back from memory.
                let mut why = why;
                let root = self.scene.root().to_path_buf();
                if let Some(before) = scene_before
                    && let Err(also) = self.store().write(&root, before.as_bytes())
                {
                    why =
                        format!("{why}; and the scene document could not be written back: {also}");
                }
                let stamp = self.store().stamp(&root);
                self.scene.put_back(scene_kept, stamp);
                self.kernels.put_back(kernels_kept, &mut self.runtime);
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
                // The inputs and placements follow the scene put back.
                self.sync();
                self.history = history;
                self.history.refused(
                    format!("the snapshot of revision {}", snapshot.revision),
                    why.clone(),
                    None,
                    origin,
                );
                Err(why)
            }
        }
    }

    /// The restore's steps, in order, noting what each replaced for
    /// [`Session::restore`] to put back.
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
            // The document on disk is what a failure writes back.
            let root = self.scene.root().to_path_buf();
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
        if self.scene.graph().is_none() {
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
                        orla_graph::NodeId(copy.node.clone()),
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
        let graph = self.scene.graph().ok_or("no scene is loaded")?;
        self.evaluator
            .restore(
                graph,
                &mut self.runtime,
                self.inputs.handles(),
                context,
                self.scene.params(),
                states,
            )
            .map_err(|error| format!("the restored world does not run: {error}"))
    }

    /// Syncs and waits until the documents being compiled, the scene's
    /// Kernels being prepared, and the scene waiting for them, are
    /// published or refused.
    pub fn sync_settled(&mut self) -> Vec<Event> {
        let mut events = self.sync();
        // Documents compiled on another thread publish at the sync after.
        while self.scene.compiling() {
            self.scene.wait_compiled();
            events.extend(self.sync());
        }
        let mut landed = Vec::new();
        self.scene
            .wait(&mut self.kernels, &mut self.runtime, &mut landed);
        self.remember(&landed);
        events.extend(landed);
        events
    }

    /// Replaces the scene document whole and reads it, its Kernels
    /// settled; the error is why it was refused.
    pub(super) fn replace_scene(&mut self, text: &str) -> Result<(), String> {
        let root = self.scene.root().to_path_buf();
        self.store()
            .write(&root, text.as_bytes())
            .map_err(|e| format!("{}: {e}", root.display()))?;
        let events = self.sync_settled();
        // Refusals of the scene document itself: its lowering or its own
        // Kernels.
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
        let landing = events
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded | Event::SceneRejected(_)));
        // A scene read in the claimed one's place is its document's.
        let superseded = self.scene.take_superseded();
        let (claimant, landed) = match std::mem::take(&mut self.claim) {
            Claim::Asked => (Some(self.origin.clone()), None),
            Claim::Waiting(origin) if !superseded => {
                // A later sync's events are the request's answer.
                let landed = landing.then(|| events.to_vec());
                (Some(origin), landed)
            }
            Claim::Landed(events) => (None, Some(events)),
            Claim::Waiting(_) | Claim::None => (None, None),
        };
        self.claim = match (&claimant, landed) {
            (Some(origin), _) if self.scene.waiting() => Claim::Waiting(origin.clone()),
            (_, Some(events)) => Claim::Landed(events),
            _ => Claim::None,
        };
        let scene_origin =
            claimant.unwrap_or_else(|| crate::Origin::Document(self.scene.root().to_path_buf()));
        let document = |path: &Path| crate::Origin::Document(path.to_path_buf());
        let mut scene_loaded = false;
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
                    self.history
                        .published(id, *generation, &source, document(path));
                }
                Event::Retired { path, id } => self.history.retired(id, document(path)),
                Event::SceneLoaded => scene_loaded = true,
                Event::Rejected { path, error } => self.history.refused(
                    path.display().to_string(),
                    error.to_string(),
                    None,
                    document(path),
                ),
                Event::SceneRejected(error) => {
                    let subject = self.scene.root().display().to_string();
                    self.history
                        .refused(subject, error.to_string(), None, scene_origin.clone());
                }
                _ => {}
            }
        }
        if scene_loaded {
            let text = self.scene.text().unwrap_or_default().to_owned();
            self.history.scene(&text, scene_origin);
        }
    }
}
