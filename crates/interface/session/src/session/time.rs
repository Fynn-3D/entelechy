//! The session in time: where the world is, what changed it and where
//! that came from, the world taken whole as a snapshot, and a snapshot's
//! world restored. Everything here holds to one rule: a change is real
//! only when the whole of it is, and a refused one leaves the world, its
//! laws, its structure and its history as they were, but for the
//! attempt it records.

use std::path::Path;
use std::sync::Arc;

use orla_graph::Report;
use orla_runtime::{Context, Expect, FieldData, KernelId, KernelVersion};

use super::Session;
use crate::Event;

impl Session {
    /// Says where what the session is asked to do from here comes from,
    /// and answers where it came from before: a protocol request's
    /// channel while it is answered. What it records carries it.
    pub fn act_for(&mut self, origin: crate::Origin) -> crate::Origin {
        std::mem::replace(&mut self.origin, origin)
    }

    /// Says that the next scene the session loads is the change the
    /// request being answered made (a `compose`, a param that resizes the
    /// graph), so it is recorded as the request's; a scene found changed
    /// on disk is its document's.
    pub fn claim_scene(&mut self) {
        self.claimed = true;
    }

    /// The world's position: its revision, which moves when the world
    /// changes other than by running, and the frames committed since.
    pub fn position(&self) -> (u64, u64) {
        self.history.position()
    }

    /// The changes that made revisions after `since`, oldest first, as
    /// far back as the session remembers.
    pub fn changes_since(&self, since: u64) -> Vec<crate::Record> {
        self.history.changes(since).cloned().collect()
    }

    /// The attempts refused at or after revision `since`, oldest first.
    pub fn attempts_since(&self, since: u64) -> Vec<crate::Attempt> {
        self.history.attempts(since).cloned().collect()
    }

    /// A Kernel's source at `generation`: the registry's when that is
    /// the one running, else the remembered one, when a change the
    /// session remembers published it.
    pub fn source_at(&self, id: &KernelId, generation: orla_runtime::Generation) -> Option<String> {
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

    /// The world as it stands, taken whole for its caller to keep: the
    /// position, every published Kernel with its generation and source,
    /// every port's committed state with its layout, extent and words,
    /// and the scene document the running graph was built from. Only
    /// committed state is taken, never an output an evaluation computed
    /// and did not commit. A Field on a device comes home first.
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

    /// Restores a snapshot's world as one change: with
    /// [`crate::Structure::Snapshot`] its scene document written back
    /// first, with [`crate::Laws::Snapshot`] its Kernels published back
    /// all together, then its state installed and proved by one frame
    /// evaluated under `context` and not committed
    /// ([`orla_graph::Evaluator::restore`]). When anything of it fails,
    /// the scene document, the laws (the very versions that ran, their
    /// generations with them), the state and the history are what they
    /// were, and the attempt is recorded.
    pub fn restore(
        &mut self,
        snapshot: &crate::Snapshot,
        laws: crate::Laws,
        structure: crate::Structure,
        context: &Context,
    ) -> Result<Report, String> {
        // What stands, taken in first: an edit made on disk is part of
        // the world a refused restore returns to.
        self.sync();
        let history = self.history.clone();
        // Every version that runs, to be put back exactly, generations
        // and all, whatever the restore publishes or its scene writes.
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
                // The laws the restore published, now that they stand.
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
                // The scene document back as it was on disk, then every
                // version that ran, as it ran: what the scene's reading
                // republished is undone with the rest.
                let mut why = why;
                if let Some(before) = scene_before
                    && let Err(also) = self.replace_scene(&before)
                {
                    why = format!("{why}; and the scene could not be put back: {also}");
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

    /// The restore's steps, in order, each noting what it replaced for
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
            // The document as it is on disk, which is what goes back.
            let root = self.scene.root().to_path_buf();
            let before =
                std::fs::read_to_string(&root).map_err(|e| format!("{}: {e}", root.display()))?;
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

    /// Replaces the scene document whole and reads it again, the Kernels
    /// it writes in place before it answers; refused, it says why.
    pub(super) fn replace_scene(&mut self, text: &str) -> Result<(), String> {
        let root = self.scene.root().to_path_buf();
        let beside = root.with_extension("osn.restoring");
        std::fs::write(&beside, text)
            .and_then(|()| std::fs::rename(&beside, &root))
            .map_err(|e| {
                let _ = std::fs::remove_file(&beside);
                format!("{}: {e}", root.display())
            })?;
        let mut events = self.sync();
        // The scene's own Kernels, prepared off the thread where one ran
        // before, must stand before the world that runs in it does.
        let mut landed = Vec::new();
        self.kernels.wait_pending(&mut self.runtime, &mut landed);
        self.remember(&landed);
        events.extend(landed);
        // What the scene document itself was refused: its lowering, or a
        // Kernel it writes. An input's file is another matter.
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

    /// Records what a sync's events say changed or was refused. What was
    /// found on disk is its document's; the scene a request claimed
    /// ([`Session::claim_scene`]) is the request's. One sync that reads
    /// the scene twice (a library changed with it) is one change.
    pub(super) fn remember(&mut self, events: &[Event]) {
        let claimed = std::mem::take(&mut self.claimed);
        let scene_origin = if claimed {
            self.origin.clone()
        } else {
            crate::Origin::Document(self.scene.root().to_path_buf())
        };
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
