//! What preparing Kernels is doing, readable from any thread while it
//! runs: each Kernel of the work at hand, from expected to compiled
//! (`prepare`) to built on its device (`finish`), and how long its
//! current step has taken; and which of them feed which in the scene. An
//! interface draws it while a scene loads.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use entelechy_ir::KernelId;

use crate::clock::{self, Moment};

/// Where one Kernel is in its preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Expected; nothing begun.
    Waiting,
    /// Being compiled and prepared on its backend.
    Compiling,
    /// Prepared; its device has not built it yet, or has nothing to build.
    Compiled,
    /// Being built on its device (a GPU's pipelines).
    Building,
    /// Ready to run.
    Built,
}

/// One Kernel of the work at hand.
#[derive(Clone, Debug)]
pub struct Seed {
    /// The Kernel.
    pub id: KernelId,
    /// Where it is.
    pub stage: Stage,
    /// How long its current step has run, while it runs; how long its last
    /// step took, after.
    pub took: Duration,
}

/// The work at hand, the Kernels in the order they were first expected.
#[derive(Clone, Debug, Default)]
pub struct Preparing {
    /// Each Kernel of it.
    pub seeds: Vec<Seed>,
    /// Seeds that feed each other in the scene, by their place in `seeds`,
    /// each pair once, the lower first.
    pub links: Vec<(usize, usize)>,
}

impl Preparing {
    /// The Kernels ready to run.
    pub fn built(&self) -> usize {
        self.seeds
            .iter()
            .filter(|seed| seed.stage == Stage::Built)
            .count()
    }

    /// Whether any Kernel is still being worked on or waits.
    pub fn busy(&self) -> bool {
        self.seeds.iter().any(|seed| seed.stage.busy())
    }

    /// The Kernel whose current step has run longest, while one runs.
    pub fn longest(&self) -> Option<&Seed> {
        self.seeds
            .iter()
            .filter(|seed| matches!(seed.stage, Stage::Compiling | Stage::Building))
            .max_by_key(|seed| seed.took)
    }
}

impl Stage {
    /// Whether it waits or runs.
    fn busy(self) -> bool {
        matches!(self, Stage::Waiting | Stage::Compiling | Stage::Building)
    }
}

#[derive(Debug)]
struct Entry {
    id: KernelId,
    stage: Stage,
    since: Option<Moment>,
    took: Duration,
}

/// The shared record; cloned handles see the same work.
#[derive(Clone, Default, Debug)]
pub struct Progress(Arc<Mutex<State>>);

#[derive(Default, Debug)]
struct State {
    entries: Vec<Entry>,
    links: Vec<(KernelId, KernelId)>,
}

impl Progress {
    /// The work at hand now.
    pub fn now(&self) -> Preparing {
        let state = self.lock();
        let place = |id: &KernelId| state.entries.iter().position(|entry| entry.id == *id);
        let mut links: Vec<(usize, usize)> = state
            .links
            .iter()
            .filter_map(|(a, b)| Some((place(a)?, place(b)?)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        links.sort_unstable();
        links.dedup();
        Preparing {
            seeds: state
                .entries
                .iter()
                .map(|entry| Seed {
                    id: entry.id.clone(),
                    stage: entry.stage,
                    took: match entry.stage {
                        Stage::Compiling | Stage::Building => clock::since(entry.since),
                        _ => entry.took,
                    },
                })
                .collect(),
            links,
        }
    }

    /// Says which Kernels feed which in the scene now, each pair a
    /// Kernel and one it reads from; what was said before goes.
    pub fn link(&self, links: impl IntoIterator<Item = (KernelId, KernelId)>) {
        self.lock().links = links.into_iter().collect();
    }

    /// Says `ids` are to be prepared or built. Where nothing waits or runs,
    /// this is new work: the Kernels of the last that it does not name are
    /// let go.
    pub(crate) fn expect<'a>(&self, ids: impl IntoIterator<Item = &'a KernelId>) {
        let ids: Vec<&KernelId> = ids.into_iter().collect();
        let mut state = self.lock();
        let entries = &mut state.entries;
        if !entries.iter().any(|entry| entry.stage.busy()) {
            entries.retain(|entry| ids.contains(&&entry.id));
        }
        for id in ids {
            match entries.iter_mut().find(|entry| entry.id == *id) {
                Some(entry) if entry.stage == Stage::Built => entry.stage = Stage::Waiting,
                Some(_) => {}
                None => entries.push(Entry {
                    id: id.clone(),
                    stage: Stage::Waiting,
                    since: None,
                    took: Duration::ZERO,
                }),
            }
        }
    }

    /// Runs `step` as `id`'s compiling or building, recorded around it.
    pub(crate) fn step<T>(&self, id: &KernelId, building: bool, step: impl FnOnce() -> T) -> T {
        let (running, after) = if building {
            (Stage::Building, Stage::Built)
        } else {
            (Stage::Compiling, Stage::Compiled)
        };
        let started = clock::now();
        self.set(id, running, started, Duration::ZERO);
        let made = step();
        self.set(id, after, None, clock::since(started));
        made
    }

    fn set(&self, id: &KernelId, stage: Stage, since: Option<Moment>, took: Duration) {
        let mut state = self.lock();
        let entries = &mut state.entries;
        match entries.iter_mut().find(|entry| entry.id == *id) {
            Some(entry) => {
                entry.stage = stage;
                entry.since = since;
                entry.took = took;
            }
            None => entries.push(Entry {
                id: id.clone(),
                stage,
                since,
                took,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: &str) -> KernelId {
        KernelId::new(name, 1).expect("an id")
    }

    #[test]
    fn work_is_seen_from_expected_to_built_and_new_work_lets_the_last_go() {
        let progress = Progress::default();
        let (a, b) = (id("t.a"), id("t.b"));
        progress.expect([&a, &b]);
        assert_eq!(progress.now().seeds.len(), 2);
        assert!(progress.now().busy());
        progress.step(&a, false, || ());
        progress.step(&b, false, || ());
        assert!(!progress.now().busy(), "compiled, nothing waits");
        // Building the one used keeps it, and lets the unused one go.
        progress.expect([&a]);
        let seen = progress.step(&a, true, || progress.now());
        assert_eq!(seen.seeds.len(), 1);
        assert_eq!(seen.seeds[0].stage, Stage::Building);
        assert_eq!(seen.longest().map(|seed| &seed.id), Some(&a));
        assert_eq!(progress.now().built(), 1);
        // Work expected while other work waits joins it.
        progress.expect([&b]);
        progress.expect([&a]);
        assert_eq!(progress.now().seeds.len(), 2);
        // Links name seeds by place, once, and only seeds there are.
        progress.link([
            (b.clone(), a.clone()),
            (a.clone(), b.clone()),
            (a, id("t.z")),
        ]);
        assert_eq!(progress.now().links, vec![(0, 1)]);
    }
}
