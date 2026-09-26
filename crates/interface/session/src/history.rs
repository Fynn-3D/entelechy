//! What changed the world, and where it came from. A session's position
//! is two coordinates: the revision, which moves when the world changes
//! other than by running (a Kernel published or retired, the scene read
//! again, a param set, state written or reset), and the frame, how many
//! evaluations have committed since. The changes are remembered with
//! their origin, and so are the attempts that were refused, which move
//! nothing. Kernel sources are kept once each by their digest, so a
//! generation the registry has let go can still be read.
//!
//! The session remembers what changed, never whole worlds: a snapshot is
//! taken when asked and kept by whoever asked. What is remembered here is
//! bounded, and a session running for days holds the latest of it.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

use orla_kernel::{Generation, KernelId};

/// Changes remembered, and attempts: the oldest are let go past these.
const CHANGES: usize = 512;
const ATTEMPTS: usize = 256;

/// A digest of a text: FNV-1a, the same in every build and on every
/// platform. It names a document's version and a Kernel's source.
pub fn digest(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Where a change came from: a fact about how it arrived, not who sent
/// it. The session knows no one's identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// A document the session reads, changed on disk.
    Document(PathBuf),
    /// A protocol request, and the channel it came on: `stdin`,
    /// `socket 3`, `websocket 4`.
    Protocol(String),
    /// The program running the session itself: a command line, a
    /// window's keys.
    Local,
}

/// What changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// A Kernel was published: its generation before (none when new) and
    /// now, and its source's digest.
    Published {
        /// The Kernel.
        id: KernelId,
        /// The generation it replaced.
        from: Option<Generation>,
        /// Its generation now.
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
    /// The scene was read again and its graph rebuilt from a version of
    /// its document, kept by digest ([`History::source`]).
    Scene {
        /// The document's text, by digest.
        document: u64,
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
    /// One port's committed state was forgotten; the rest kept theirs.
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
        /// Whether its laws came with it, or its state alone went under
        /// the laws that ran.
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
    /// Seconds since the Unix epoch, when it happened.
    pub wall: u64,
    /// Where it came from.
    pub origin: Origin,
    /// What it was.
    pub change: Change,
}

/// An attempt refused: it moved nothing, and why is kept.
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
    /// The document or the Kernel it was about, as the refusal names it.
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
}

/// Seconds since the Unix epoch; zero where the machine has no clock.
fn wall() -> u64 {
    orla_kernel::clock::wall().unwrap_or(0)
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

    /// The scene rebuilt from `text`, which is kept.
    pub(crate) fn scene(&mut self, text: &str, origin: Origin) {
        let document = self.keep(text);
        self.record(origin, Change::Scene { document });
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

    /// A scene document a remembered change loaded, by its digest: only
    /// what was loaded as the scene, never a Kernel's source that shares
    /// the store.
    pub(crate) fn scene_text(&self, digest: u64) -> Option<&str> {
        self.changes
            .iter()
            .any(|record| matches!(record.change, Change::Scene { document } if document == digest))
            .then(|| self.source(digest))
            .flatten()
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
                Change::Scene { document } => self.forget(document),
                _ => {}
            }
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

    /// A source is kept once however often it is published, and let go
    /// when no remembered change names it any more.
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
        // What the first reset recorded: the frames the revision before ran.
        let first = history.changes(0).next().unwrap();
        assert_eq!(first.revision, 4);
        assert_eq!(first.after_frames, 2);
    }
}
