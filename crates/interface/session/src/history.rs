//! What changed the world, and where it came from. A position is the
//! revision, moved by every change but running (publish, retire, scene
//! read, param, state written or reset), and the frames committed since.
//! Changes and refused attempts are remembered with their origin, within
//! fixed bounds, newest kept. Kernel sources and the scene's document
//! texts are kept once by digest, so a generation the registry released
//! and a document's earlier version stay readable. Whole worlds are
//! snapshots, kept by whoever took them.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

use entelechy_kernel::{Generation, KernelId};

/// How many changes and attempts are remembered; the oldest go first.
const CHANGES: usize = 512;
const ATTEMPTS: usize = 256;

/// A text's FNV-1a digest, stable across builds and platforms. It names a
/// document's version and a Kernel's source.
pub fn digest(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// How a change arrived. The session holds no identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// A document the session reads, changed on disk.
    Document(PathBuf),
    /// A protocol request's channel: `stdin`, `socket 3`, `websocket 4`.
    Protocol(String),
    /// The program running the session: a command line, a window's keys.
    Local,
}

/// What changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// A Kernel was published.
    Published {
        /// The Kernel.
        id: KernelId,
        /// The generation it replaced; `None` when new.
        from: Option<Generation>,
        /// Its new generation.
        to: Generation,
        /// Its source, by digest ([`History::source`]).
        source: u64,
    },
    /// A Kernel left the registry.
    Retired {
        /// The Kernel.
        id: KernelId,
        /// The generation that left.
        from: Option<Generation>,
    },
    /// The scene graph was rebuilt from a document version
    /// ([`History::source`]), with the other documents of the scene it
    /// loaded at versions not remembered before.
    Scene {
        /// The document's text, by digest.
        document: u64,
        /// Each other document loaded at a new version, and that version.
        documents: Vec<(PathBuf, u64)>,
    },
    /// Documents of the scene were written by a compose, each by its one
    /// name and its text, by digest ([`History::source`]).
    Documents {
        /// Each document written and its version.
        written: Vec<(PathBuf, u64)>,
    },
    /// A graph param was set.
    Param {
        /// Its name.
        name: String,
    },
    /// A port's committed state was written from outside.
    Written {
        /// The node.
        node: String,
        /// Its port.
        port: String,
    },
    /// Committed state was forgotten.
    Reset,
    /// One port's committed state was forgotten.
    ResetPort {
        /// The node.
        node: String,
        /// Its port.
        port: String,
    },
    /// A snapshot's world was restored.
    Restored {
        /// The revision the snapshot was taken at.
        revision: u64,
        /// Whether its laws were restored with its state.
        laws: bool,
    },
}

/// A change and where the world was when it happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// The revision it made.
    pub revision: u64,
    /// How many frames the revision before it ran.
    pub after_frames: u64,
    /// Seconds since the Unix epoch.
    pub wall: u64,
    /// Where it came from.
    pub origin: Origin,
    /// What it was.
    pub change: Change,
}

/// A refused attempt, which changed nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    /// The revision it was refused at.
    pub revision: u64,
    /// The frame within it.
    pub frame: u64,
    /// Seconds since the Unix epoch.
    pub wall: u64,
    /// Where it came from.
    pub origin: Origin,
    /// The document or Kernel it was about.
    pub subject: String,
    /// Why.
    pub why: String,
    /// The source it tried, by digest, when it sent one.
    pub source: Option<u64>,
}

/// The session's memory of change.
#[derive(Clone, Default)]
pub(crate) struct History {
    revision: u64,
    frame: u64,
    changes: VecDeque<Record>,
    attempts: VecDeque<Attempt>,
    /// Sources by digest, with how many records name each.
    sources: BTreeMap<u64, (String, usize)>,
    /// Each Kernel's generation as last recorded, for a change's `from`.
    current: BTreeMap<KernelId, Generation>,
    /// Each document's version as last recorded, so a scene records only
    /// the documents it loaded at a new one.
    latest: BTreeMap<PathBuf, u64>,
}

/// Seconds since the Unix epoch; zero where the machine has no clock.
fn wall() -> u64 {
    entelechy_kernel::clock::wall().unwrap_or(0)
}

impl History {
    /// The world's position: its revision and the frames run in it.
    pub(crate) fn position(&self) -> (u64, u64) {
        (self.revision, self.frame)
    }

    /// An evaluation committed.
    pub(crate) fn ran(&mut self) {
        self.frame += 1;
    }

    /// Evaluations answered as committed were taken back.
    pub(crate) fn taken_back(&mut self, frames: u64) {
        self.frame = self.frame.saturating_sub(frames);
    }

    /// A Kernel was published with `source`.
    pub(crate) fn published(
        &mut self,
        id: &KernelId,
        to: Generation,
        source: &str,
        origin: Origin,
    ) {
        let hash = self.keep(source);
        let from = self.current.insert(id.clone(), to);
        self.record(
            origin,
            Change::Published {
                id: id.clone(),
                from,
                to,
                source: hash,
            },
        );
    }

    /// A Kernel left.
    pub(crate) fn retired(&mut self, id: &KernelId, origin: Origin) {
        let from = self.current.remove(id);
        self.record(
            origin,
            Change::Retired {
                id: id.clone(),
                from,
            },
        );
    }

    /// The scene rebuilt from `text`, with its other documents `loaded`;
    /// the texts are kept, the others' only when new to their path.
    pub(crate) fn scene(&mut self, text: &str, loaded: &[(PathBuf, String)], origin: Origin) {
        let document = self.keep(text);
        let documents = self.versions(loaded, true);
        self.record(
            origin,
            Change::Scene {
                document,
                documents,
            },
        );
    }

    /// Documents a compose wrote, whose texts are kept.
    pub(crate) fn documents(&mut self, written: &[(PathBuf, String)], origin: Origin) {
        let written = self.versions(written, false);
        self.record(origin, Change::Documents { written });
    }

    /// Each text kept and its path's latest version set; with `new_only`,
    /// a text already its path's latest is left out.
    fn versions(&mut self, texts: &[(PathBuf, String)], new_only: bool) -> Vec<(PathBuf, u64)> {
        let mut versions = Vec::new();
        for (path, text) in texts {
            if new_only && self.latest.get(path) == Some(&digest(text)) {
                continue;
            }
            let version = self.keep(text);
            self.latest.insert(path.clone(), version);
            versions.push((path.clone(), version));
        }
        versions
    }

    /// Anything else that changed the world.
    pub(crate) fn changed(&mut self, change: Change, origin: Origin) {
        self.record(origin, change);
    }

    /// An attempt refused.
    pub(crate) fn refused(
        &mut self,
        subject: String,
        why: String,
        source: Option<&str>,
        origin: Origin,
    ) {
        let source = source.map(|text| self.keep(text));
        self.attempts.push_back(Attempt {
            revision: self.revision,
            frame: self.frame,
            wall: wall(),
            origin,
            subject,
            why,
            source,
        });
        if self.attempts.len() > ATTEMPTS
            && let Some(gone) = self.attempts.pop_front()
            && let Some(hash) = gone.source
        {
            self.forget(hash);
        }
    }

    /// The changes after `since`, oldest first.
    pub(crate) fn changes(&self, since: u64) -> impl Iterator<Item = &Record> {
        self.changes.iter().filter(move |r| r.revision > since)
    }

    /// The attempts refused at or after revision `since`, oldest first.
    pub(crate) fn attempts(&self, since: u64) -> impl Iterator<Item = &Attempt> {
        self.attempts.iter().filter(move |a| a.revision >= since)
    }

    /// A source still remembered, by digest.
    pub(crate) fn source(&self, hash: u64) -> Option<&str> {
        self.sources.get(&hash).map(|(text, _)| text.as_str())
    }

    /// A scene document a remembered change loaded, by digest; Kernel
    /// sources are excluded.
    pub(crate) fn scene_text(&self, digest: u64) -> Option<&str> {
        self.changes
            .iter()
            .any(|record| matches!(record.change, Change::Scene { document, .. } if document == digest))
            .then(|| self.source(digest))
            .flatten()
    }

    /// A text the document at `path` had, by digest, when a remembered
    /// compose wrote it or a remembered scene change loaded it.
    pub(crate) fn document_text(
        &self,
        path: &std::path::Path,
        digest: u64,
        scene: bool,
    ) -> Option<&str> {
        let written = self.changes.iter().any(|record| match &record.change {
            Change::Documents { written } => written
                .iter()
                .any(|(at, version)| at == path && *version == digest),
            Change::Scene {
                document,
                documents,
            } => {
                (scene && *document == digest)
                    || documents
                        .iter()
                        .any(|(at, version)| at == path && *version == digest)
            }
            _ => false,
        });
        written.then(|| self.source(digest)).flatten()
    }

    /// The source a Kernel had at `generation`, when a remembered change
    /// published it.
    pub(crate) fn source_of(&self, id: &KernelId, generation: Generation) -> Option<&str> {
        self.changes
            .iter()
            .rev()
            .find_map(|record| match &record.change {
                Change::Published {
                    id: published,
                    to,
                    source,
                    ..
                } if published == id && *to == generation => self.source(*source),
                _ => None,
            })
    }

    fn record(&mut self, origin: Origin, change: Change) {
        let after_frames = self.frame;
        self.revision += 1;
        self.frame = 0;
        self.changes.push_back(Record {
            revision: self.revision,
            after_frames,
            wall: wall(),
            origin,
            change,
        });
        if self.changes.len() > CHANGES
            && let Some(gone) = self.changes.pop_front()
        {
            match gone.change {
                Change::Published { source, .. } => self.forget(source),
                Change::Scene {
                    document,
                    documents,
                } => {
                    self.forget(document);
                    self.forget_versions(documents);
                }
                Change::Documents { written } => self.forget_versions(written),
                _ => {}
            }
        }
    }

    /// Forgets the versions a dropped record named; a path's latest
    /// version no record names any more is recorded again when loaded.
    fn forget_versions(&mut self, versions: Vec<(PathBuf, u64)>) {
        for (path, version) in versions {
            if self.latest.get(&path) == Some(&version) {
                self.latest.remove(&path);
            }
            self.forget(version);
        }
    }

    /// Keeps a source once, counting who names it; answers its digest.
    fn keep(&mut self, text: &str) -> u64 {
        let hash = digest(text);
        self.sources
            .entry(hash)
            .and_modify(|(_, count)| *count += 1)
            .or_insert_with(|| (text.to_owned(), 1));
        hash
    }

    fn forget(&mut self, hash: u64) {
        if let Some((_, count)) = self.sources.get_mut(&hash) {
            *count -= 1;
            if *count == 0 {
                self.sources.remove(&hash);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source is kept once, and released with the last record naming it.
    #[test]
    fn sources_are_kept_once_and_let_go_with_the_last_record_that_names_them() {
        let mut history = History::default();
        let id: KernelId = "t.k@1".parse().unwrap();
        for (n, text) in ["a", "b", "a"].iter().enumerate() {
            history.published(&id, Generation(n as u64 + 1), text, Origin::Local);
        }
        assert_eq!(history.sources.len(), 2);
        assert_eq!(history.source_of(&id, Generation(2)), Some("b"));
        assert_eq!(history.position(), (3, 0));
        history.ran();
        history.ran();
        assert_eq!(history.position(), (3, 2));
        // Enough other changes push the three publishes out.
        for _ in 0..CHANGES {
            history.changed(Change::Reset, Origin::Local);
        }
        assert!(history.sources.is_empty(), "{:?}", history.sources.keys());
        assert_eq!(history.source_of(&id, Generation(2)), None);
        assert_eq!(history.changes(0).count(), CHANGES);
        // The first reset records the frames of the revision before it.
        let first = history.changes(0).next().unwrap();
        assert_eq!(first.revision, 4);
        assert_eq!(first.after_frames, 2);
    }

    /// Documents a compose wrote are one change, each text readable by its
    /// path and version until the change is let go.
    #[test]
    fn composed_documents_are_one_change_read_back_until_let_go() {
        let mut history = History::default();
        let (scene, kernel) = (
            PathBuf::from("/w/scene.esn"),
            PathBuf::from("/w/kernels/a.ent"),
        );
        history.documents(
            &[
                (kernel.clone(), "fn a() -> Float { return 1.0; }".into()),
                (scene.clone(), "use \"kernels/a.ent\";".into()),
            ],
            Origin::Local,
        );
        assert_eq!(history.position(), (1, 0));
        let a = digest("fn a() -> Float { return 1.0; }");
        assert_eq!(
            history.document_text(&kernel, a, false),
            Some("fn a() -> Float { return 1.0; }")
        );
        assert_eq!(
            history.document_text(&scene, a, true),
            None,
            "a version is its own document's"
        );
        for _ in 0..CHANGES {
            history.changed(Change::Reset, Origin::Local);
        }
        assert_eq!(history.document_text(&kernel, a, false), None);
        assert!(history.sources.is_empty(), "{:?}", history.sources.keys());
    }
}
