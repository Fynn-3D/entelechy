//! What the evaluator holds and how long: the cache of node outputs and
//! the committed state, each with a count per handle, so a Field is
//! released only when nothing refers to it any more.

use super::*;

impl Evaluator {
    /// Every port's committed state, by producing node and port: the
    /// world as the last successful evaluation left it, never a cached
    /// output a failed evaluation computed after it.
    pub fn committed_states(&self) -> Vec<(NodeId, String, Handle)> {
        self.ledger
            .committed_ports()
            .map(|((node, port), handle)| (node.clone(), port.clone(), handle))
            .collect()
    }

    /// Waits on every check the nodes' runs answered with, in the order
    /// they ran: the first failure fails the evaluation as a failure on
    /// the frame does, naming its node, and nothing commits. What that
    /// node and every node after it left in the cache was computed from
    /// a failed run, and goes, so the next evaluation runs them again.
    pub(super) fn wait_for_checks(
        &mut self,
        checks: Vec<(NodeId, Vec<Box<dyn orla_kernel::Check>>)>,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        if checks.iter().all(|(_, checks)| checks.is_empty()) {
            return Ok(());
        }
        let mut failed: Option<(usize, NodeId, orla_kernel::ExecutionError)> = None;
        let mut order = Vec::with_capacity(checks.len());
        for (at, (node, node_checks)) in checks.into_iter().enumerate() {
            order.push(node.clone());
            for check in node_checks {
                if let Err(error) = check.wait()
                    && failed.is_none()
                {
                    failed = Some((at, node.clone(), error));
                }
            }
        }
        let Some((at, node, error)) = failed else {
            return Ok(());
        };
        for stale in &order[at..] {
            if let Some(gone) = self.ledger.cache_take(stale) {
                for output in gone.outputs {
                    if !self.is_kept(output) {
                        release(runtime, output);
                    }
                }
            }
        }
        Err(EvaluateError::Execution { node, error })
    }

    /// Whether `handle` is state: committed, or replaced by an evaluation
    /// that may yet put it back. What is kept is never released.
    pub(super) fn is_kept(&self, handle: Handle) -> bool {
        self.ledger.is_committed(handle) || self.replaced.iter().any(|r| r.previous == Some(handle))
    }

    pub(super) fn drop_cache(&mut self, id: &NodeId, runtime: &mut Runtime) {
        if let Some(cached) = self.ledger.cache_take(id) {
            for output in cached.outputs {
                if !self.is_kept(output) {
                    release(runtime, output);
                }
            }
        }
    }

    /// Releases cache entries and state of nodes that left the graph.
    pub(super) fn prune(&mut self, graph: &Graph, runtime: &mut Runtime) {
        // One index for the three sweeps below; each used to walk the
        // whole node list once a cached entry.
        let index = graph.index();
        let gone_from = |id: &NodeId| !index.contains_key(id);
        let stale: Vec<NodeId> = self
            .ledger
            .cached_nodes()
            .filter(|id| gone_from(id))
            .cloned()
            .collect();
        for id in stale {
            self.drop_cache(&id, runtime);
        }
        let gone: Vec<NodeId> = self
            .running
            .keys()
            .filter(|id| gone_from(id))
            .cloned()
            .collect();
        for id in gone {
            if let Some(running) = self.running.remove(&id) {
                runtime.discard_run(running.run);
            }
        }
        let stale_state: Vec<(NodeId, String)> = self
            .ledger
            .committed_ports()
            .map(|(key, _)| key)
            .filter(|(node, _)| gone_from(node))
            .cloned()
            .collect();
        for key in stale_state {
            if let Some(handle) = self.ledger.commit_take(&key)
                && !self.ledger.is_cached(handle)
            {
                release(runtime, handle);
            }
        }
        self.pinned.retain(|node| graph.node(node).is_some());
    }
}

/// Every node's cached outputs and every port's committed state, each
/// with a count per handle. The counts are kept by the only methods that
/// change the maps, and nothing outside this module reaches the maps, so
/// a Field is released only when neither the cache nor the state refers
/// to it any more, and never twice.
#[derive(Default)]
pub(super) struct Ledger {
    cache: BTreeMap<NodeId, Cached>,
    /// How many cached entries hold each handle: asking whether a handle
    /// is still cached would otherwise walk every entry's outputs, once a
    /// binding a frame.
    cached_handles: BTreeMap<Handle, usize>,
    /// State as of the last successful evaluation, by producing node and
    /// port. A committed handle may also be a node's current cached
    /// output.
    committed: BTreeMap<(NodeId, String), Handle>,
    /// How many committed ports hold each handle.
    committed_handles: BTreeMap<Handle, usize>,
}

impl Ledger {
    /// A node's cache entry.
    pub(super) fn entry(&self, id: &NodeId) -> Option<&Cached> {
        self.cache.get(id)
    }

    /// The nodes the cache holds an entry for.
    pub(super) fn cached_nodes(&self) -> impl Iterator<Item = &NodeId> {
        self.cache.keys()
    }

    /// A port's committed state.
    pub(super) fn committed(&self, key: &(NodeId, String)) -> Option<Handle> {
        self.committed.get(key).copied()
    }

    /// Every committed port and its state.
    pub(super) fn committed_ports(&self) -> impl Iterator<Item = (&(NodeId, String), Handle)> {
        self.committed.iter().map(|(key, handle)| (key, *handle))
    }

    /// How many cache entries and committed ports there are.
    pub(super) fn sizes(&self) -> (usize, usize) {
        (self.cache.len(), self.committed.len())
    }

    /// Everything held, taken out: the cache and the committed state,
    /// their counts cleared with them.
    pub(super) fn take_all(
        &mut self,
    ) -> (BTreeMap<NodeId, Cached>, BTreeMap<(NodeId, String), Handle>) {
        self.cached_handles.clear();
        self.committed_handles.clear();
        (
            std::mem::take(&mut self.cache),
            std::mem::take(&mut self.committed),
        )
    }

    /// Puts a node's outputs in the cache, keeping the count of what
    /// the cache holds; answers what was there.
    pub(super) fn cache_put(&mut self, id: NodeId, fresh: Cached) -> Option<Cached> {
        for output in &fresh.outputs {
            *self.cached_handles.entry(*output).or_default() += 1;
        }
        let gone = self.cache.insert(id, fresh);
        if let Some(gone) = &gone {
            forget(&mut self.cached_handles, gone.outputs.iter().copied());
        }
        gone
    }

    /// Takes a node's entry out of the cache, keeping the count.
    pub(super) fn cache_take(&mut self, id: &NodeId) -> Option<Cached> {
        let gone = self.cache.remove(id);
        if let Some(gone) = &gone {
            forget(&mut self.cached_handles, gone.outputs.iter().copied());
        }
        gone
    }

    /// Commits a port's state, keeping the count; answers what was there.
    pub(super) fn commit_put(&mut self, key: (NodeId, String), handle: Handle) -> Option<Handle> {
        *self.committed_handles.entry(handle).or_default() += 1;
        let gone = self.committed.insert(key, handle);
        if let Some(gone) = gone {
            forget(&mut self.committed_handles, std::iter::once(gone));
        }
        gone
    }

    /// Takes a port's committed state, keeping the count.
    pub(super) fn commit_take(&mut self, key: &(NodeId, String)) -> Option<Handle> {
        let gone = self.committed.remove(key);
        if let Some(gone) = gone {
            forget(&mut self.committed_handles, std::iter::once(gone));
        }
        gone
    }

    pub(super) fn is_cached(&self, handle: Handle) -> bool {
        let held = self.cached_handles.contains_key(&handle);
        // The count is the answer; the walk is what it stands for. A
        // count that drifted would leak a Field or release one twice,
        // so every test run checks the two agree.
        debug_assert_eq!(
            held,
            self.cache
                .values()
                .any(|cached| cached.outputs.contains(&handle)),
            "the cached-handle count disagrees with the cache"
        );
        held
    }

    pub(super) fn is_committed(&self, handle: Handle) -> bool {
        let held = self.committed_handles.contains_key(&handle);
        debug_assert_eq!(
            held,
            self.committed
                .values()
                .any(|committed| *committed == handle),
            "the committed-handle count disagrees with the committed state"
        );
        held
    }
}

/// Drops a count for every handle given, and the entry when it
/// reaches nothing.
fn forget(counts: &mut BTreeMap<Handle, usize>, handles: impl Iterator<Item = Handle>) {
    for handle in handles {
        if let std::collections::btree_map::Entry::Occupied(mut at) = counts.entry(handle) {
            *at.get_mut() -= 1;
            if *at.get() == 0 {
                at.remove();
            }
        }
    }
}
