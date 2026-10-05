//! Runs remembered frame by frame. A run named with
//! [`Evaluator::recall_as`] keeps, for each frame it evaluates, the cache
//! as that frame left it; the next run of that name, frame by frame, takes
//! a node's outputs from the same frame of the last run when its key is
//! the one it had there (the same Kernel generation, parameters, inputs,
//! domain, clock and input), as it takes them from the frame before. A
//! node that runs gives way to the remembered outputs when its fresh ones
//! equal them word for word, so what reads it finds the key it found the
//! last time. A run that changes nothing of what a frame reads computes
//! nothing on it; a changed Kernel whose outputs are what they were leaves
//! everything after it as it was; once outputs part from the remembered
//! ones, what reads them runs, as on any frame.
//!
//! What a run remembers holds its Fields until the frame is remembered
//! again or the runs are forgotten, up to a budget of bytes; frames past
//! the budget are not remembered, and a frame short of room in the Host
//! forgets every run before it lets go of its own outputs. A node's
//! outputs larger than
//! [`RECALL_HOLD`] are remembered by their key and epoch alone: on a
//! backend that repeats exactly ([`orla_kernel::Backend::repeats_exactly`])
//! a node whose key is the one it had there computes what it computed
//! there, and runs again under the epoch it had, its readers finding their
//! keys; elsewhere it runs under a new epoch and its readers run. A changed graph forgets every run.

use super::*;

/// The bytes runs may hold, all told, unless set otherwise.
pub const RECALL_BUDGET: usize = 1 << 30;

/// The most bytes of one node's outputs a run holds; larger ones are
/// remembered by key and epoch alone.
pub const RECALL_HOLD: usize = 1 << 20;

/// A node's entry on a remembered frame: its key, its outputs' names and
/// epoch, and its outputs when held.
#[derive(Clone)]
pub(super) struct Remembered {
    key: CacheKey,
    output_names: Vec<String>,
    epoch: u64,
    outputs: Option<Vec<Handle>>,
}

/// What a remembered frame offers a node whose key is the one it had.
pub(super) enum Offer {
    /// Its outputs, taken into the cache.
    Taken,
    /// Its epoch: the node runs, and what it computes is what it computed.
    Rerun(u64),
}

/// Every remembered run, by name, and the Fields they hold.
#[derive(Default)]
pub(super) struct Recalled {
    /// Each run's frames, each the cache entries its nodes left.
    runs: BTreeMap<String, Vec<BTreeMap<NodeId, Remembered>>>,
    /// How many remembered entries hold each handle, and its bytes.
    held: BTreeMap<Handle, (usize, usize)>,
    /// The bytes the held handles take.
    bytes: usize,
    /// The bytes runs may hold.
    budget: usize,
    /// The run being evaluated and its next frame.
    current: Option<(String, usize)>,
    /// The graph the runs were remembered for.
    graph: Option<Graph>,
}

impl Recalled {
    pub(super) fn new() -> Self {
        Self {
            budget: RECALL_BUDGET,
            ..Self::default()
        }
    }

    /// Whether a remembered run holds `handle`.
    pub(super) fn holds(&self, handle: Handle) -> bool {
        self.held.contains_key(&handle)
    }

    /// The entry `id` left on the current frame of the last run.
    pub(super) fn entry(&self, id: &NodeId) -> Option<&Remembered> {
        let (name, frame) = self.current.as_ref()?;
        self.runs.get(name)?.get(*frame)?.get(id)
    }

    fn hold(&mut self, handle: Handle, runtime: &Runtime) {
        if let Some((count, _)) = self.held.get_mut(&handle) {
            *count += 1;
            return;
        }
        let host = runtime.host();
        let bytes = match (host.domain(handle), host.schema(handle)) {
            (Ok(domain), Ok(schema)) => domain.count() * schema.element_bytes(),
            _ => 0,
        };
        self.bytes += bytes;
        self.held.insert(handle, (1, bytes));
    }

    /// Lets go of one hold on `handle`; answers whether nothing remembered
    /// holds it any more.
    fn let_go(&mut self, handle: Handle) -> bool {
        let std::collections::btree_map::Entry::Occupied(mut at) = self.held.entry(handle) else {
            return false;
        };
        at.get_mut().0 -= 1;
        if at.get().0 > 0 {
            return false;
        }
        let (_, bytes) = at.remove();
        self.bytes -= bytes;
        true
    }
}

impl Evaluator {
    /// Names the run the next evaluations are, from its first frame: each
    /// frame takes from the same frame of the last run of `name` what it
    /// left unchanged, and is remembered for the next. `None` remembers
    /// nothing and takes nothing.
    pub fn recall_as(&mut self, name: Option<String>) {
        self.recalled.current = name.map(|name| (name, 0));
    }

    /// The bytes remembered runs may hold, all told; frames past it are not
    /// remembered.
    pub fn set_recall_budget(&mut self, bytes: usize) {
        self.recalled.budget = bytes;
    }

    /// The bytes remembered runs hold.
    pub fn recalled_bytes(&self) -> usize {
        self.recalled.bytes
    }

    /// Forgets every remembered run, releasing what only they held.
    pub fn forget_runs(&mut self, runtime: &mut Runtime) {
        let runs = std::mem::take(&mut self.recalled.runs);
        for frames in runs.into_values() {
            for frame in frames {
                for remembered in frame.into_values() {
                    self.forget_held(remembered.outputs.unwrap_or_default(), runtime);
                }
            }
        }
        self.recalled.graph = None;
    }

    /// Forgets every remembered run to make room in the Host: what a run
    /// remembers gives way to what a frame computes. Answers how many
    /// Fields only remembered runs held.
    pub(super) fn forget_runs_for_room(&mut self, runtime: &mut Runtime) -> usize {
        let held = self.recalled.held.len();
        self.forget_runs(runtime);
        held
    }

    /// Forgets every remembered run when `graph` is not the one they were
    /// remembered for.
    pub(super) fn recall_for(&mut self, graph: &Graph, runtime: &mut Runtime) {
        if self.recalled.current.is_none() && self.recalled.runs.is_empty() {
            return;
        }
        if self.recalled.graph.as_ref() != Some(graph) {
            self.forget_runs(runtime);
            self.recalled.graph = Some(graph.clone());
        }
    }

    /// Remembers what `nodes` left in the cache as the current run's frame,
    /// in place of what the last run left there, and moves to its next
    /// frame. A frame past the budget is not remembered.
    pub(super) fn remember_frame<'a>(
        &mut self,
        nodes: impl Iterator<Item = &'a NodeId>,
        runtime: &mut Runtime,
    ) {
        let Some((name, frame)) = self.recalled.current.clone() else {
            return;
        };
        self.recalled.current = Some((name.clone(), frame + 1));
        let mut kept = BTreeMap::new();
        for id in nodes {
            if let Some(cached) = self.ledger.entry(id) {
                let host = runtime.host();
                let bytes: usize = cached
                    .outputs
                    .iter()
                    .map(
                        |handle| match (host.domain(*handle), host.schema(*handle)) {
                            (Ok(domain), Ok(schema)) => domain.count() * schema.element_bytes(),
                            _ => usize::MAX / 4,
                        },
                    )
                    .sum();
                kept.insert(
                    id.clone(),
                    Remembered {
                        key: cached.key.clone(),
                        output_names: cached.output_names.clone(),
                        epoch: cached.epoch,
                        outputs: (bytes <= RECALL_HOLD).then(|| cached.outputs.clone()),
                    },
                );
            }
        }
        // What the last run left on this frame, let go first, so a frame
        // remembered again takes no more of the budget than it gives back.
        let old = self
            .recalled
            .runs
            .get_mut(&name)
            .and_then(|frames| frames.get_mut(frame))
            .map(std::mem::take);
        for remembered in old.into_iter().flat_map(BTreeMap::into_values) {
            self.forget_held(remembered.outputs.unwrap_or_default(), runtime);
        }
        {
            let frames = self.recalled.runs.entry(name.clone()).or_default();
            if frames.len() <= frame {
                frames.resize_with(frame + 1, BTreeMap::new);
            }
        }
        if self.recalled.bytes > self.recalled.budget {
            return;
        }
        for remembered in kept.values() {
            for output in remembered.outputs.iter().flatten() {
                self.recalled.hold(*output, runtime);
            }
        }
        if let Some(frames) = self.recalled.runs.get_mut(&name) {
            frames[frame] = kept;
        }
    }

    /// Lets go of a remembered entry's holds, releasing a Field nothing
    /// else holds.
    fn forget_held(&mut self, outputs: Vec<Handle>, runtime: &mut Runtime) {
        for output in outputs {
            if self.recalled.let_go(output)
                && !self.ledger.is_cached(output)
                && !self.is_kept(output)
            {
                release(runtime, output);
            }
        }
    }

    /// What this frame of the last run offers `id` when its key was `key`
    /// there: its outputs, taken into the cache, or its epoch to run under.
    pub(super) fn recall(
        &mut self,
        id: &NodeId,
        key: &CacheKey,
        runtime: &mut Runtime,
    ) -> Option<Offer> {
        let recalled = self.recalled.entry(id).filter(|r| &r.key == key)?.clone();
        let Some(outputs) = recalled.outputs else {
            return Some(Offer::Rerun(recalled.epoch));
        };
        let taken = Cached {
            key: recalled.key,
            outputs,
            output_names: recalled.output_names,
            epoch: recalled.epoch,
        };
        if let Some(previous) = self.ledger.cache_put(id.clone(), taken) {
            for output in previous.outputs {
                if !self.ledger.is_cached(output) && !self.is_kept(output) {
                    release(runtime, output);
                }
            }
        }
        Some(Offer::Taken)
    }

    /// Caches what a node ran again under a remembered epoch computed:
    /// what it computed on this frame of the last run.
    pub(super) fn store_as(
        &mut self,
        id: &NodeId,
        key: CacheKey,
        outputs: Vec<Handle>,
        output_names: Vec<String>,
        epoch: u64,
        runtime: &mut Runtime,
    ) {
        let fresh = Cached {
            key,
            outputs,
            output_names,
            epoch,
        };
        if let Some(previous) = self.ledger.cache_put(id.clone(), fresh) {
            for output in previous.outputs {
                if !self.ledger.is_cached(output) && !self.is_kept(output) {
                    release(runtime, output);
                }
            }
        }
    }

    /// The outputs `id` left on this frame of the last run, with their
    /// epoch, when `fresh` equals them word for word under `output_names`.
    pub(super) fn recalled_alike(
        &self,
        id: &NodeId,
        output_names: &[String],
        fresh: &[Handle],
        runtime: &Runtime,
    ) -> Option<(Vec<Handle>, u64)> {
        let recalled = self.recalled.entry(id)?;
        let outputs = recalled.outputs.as_ref()?;
        (recalled.output_names == output_names && same_words(runtime, outputs, fresh, true))
            .then(|| (outputs.clone(), recalled.epoch))
    }
}
