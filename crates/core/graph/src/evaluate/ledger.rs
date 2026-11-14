//! The evaluator's holdings: cached node outputs and committed state, each
//! counted per handle, so a Field is released when nothing refers to it;
//! and the outputs let go cold, of which the key, epoch and layout stay
//! (see [`Cold`]).

use super::*;
use crate::hash::QuickMap;

impl Evaluator {
    /// Every port's committed state, by producing node and port, as the
    /// last successful evaluation left it.
    pub fn committed_states(&self) -> Vec<(NodeId, String, Handle)> {
        self.ledger
            .committed_ports()
            .map(|((node, port), handle)| (node.clone(), port.clone(), handle))
            .collect()
    }

    /// Waits on the runs' checks in run order. The first failure fails the
    /// evaluation, naming its node; that node and every later one leave
    /// the cache, so the next evaluation reruns them.
    pub(super) fn wait_for_checks(
        &mut self,
        checks: Vec<(NodeId, Vec<Box<dyn entelechy_kernel::Check>>)>,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        if checks.iter().all(|(_, checks)| checks.is_empty()) {
            return Ok(());
        }
        let mut failed: Option<(usize, NodeId, entelechy_kernel::ExecutionError)> = None;
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
    /// that may put it back, or held by a remembered run; or an output a
    /// caller awaits. None of these is released.
    pub(super) fn is_kept(&self, handle: Handle) -> bool {
        self.ledger.is_committed(handle)
            || self.replaced.iter().any(|r| r.previous == Some(handle))
            || self.recalled.holds(handle)
            || self.held_back(handle)
            || self.awaits(handle)
    }

    /// Lets go of a node's cached outputs, keeping what a reader binds by:
    /// its key and epoch, and each output's extent and layout. A frame
    /// that finds the key unchanged passes the node by unrun; a reader
    /// that runs has it made again first ([`Evaluator::warm`]). Answers
    /// whether it had outputs to let go.
    pub(super) fn chill(&mut self, id: &NodeId, runtime: &mut Runtime) -> bool {
        let Some(cached) = self.ledger.cache_take(id) else {
            return false;
        };
        let host = runtime.host();
        let laid: Option<Vec<(Domain, entelechy_abi::Schema)>> = cached
            .outputs
            .iter()
            .map(|handle| {
                Some((
                    host.domain(*handle).ok()?,
                    host.schema(*handle).ok()?.clone(),
                ))
            })
            .collect();
        for output in &cached.outputs {
            if !self.ledger.is_cached(*output) && !self.is_kept(*output) {
                release(runtime, *output);
            }
        }
        // A handle the Host no longer describes leaves nothing to bind by.
        if let Some(laid) = laid {
            self.ledger.cold_put(
                id.clone(),
                Cold {
                    key: cached.key,
                    output_names: cached.output_names,
                    epoch: cached.epoch,
                    laid,
                },
            );
        }
        true
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
        // One index for the three sweeps below.
        let index = graph.index();
        let gone_from = |id: &NodeId| !index.contains_key(id);
        let stale: Vec<NodeId> = self
            .ledger
            .cached_nodes()
            .chain(self.ledger.cold_nodes())
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
                && !self.recalled.holds(handle)
            {
                release(runtime, handle);
            }
        }
        self.pinned.retain(|node| graph.node(node).is_some());
    }
}

/// Every node's cached outputs and every port's committed state, each
/// counted per handle. Only this module's methods change the maps, so a
/// Field is released once, when neither refers to it.
#[derive(Default)]
pub(super) struct Ledger {
    /// Looked up for every node of every frame, so hashed; walked in order
    /// of node ([`Ledger::cached_nodes`]).
    cache: QuickMap<NodeId, Cached>,
    /// Nodes whose outputs were let go cold; none of them is in `cache`.
    cold: QuickMap<NodeId, Cold>,
    /// How many cached entries hold each handle.
    cached_handles: QuickMap<Handle, usize>,
    /// State as of the last successful evaluation, by producing node and
    /// port; a handle may also be a cached output.
    committed: QuickMap<(NodeId, String), Handle>,
    /// How many committed ports hold each handle.
    committed_handles: QuickMap<Handle, usize>,
    /// Moved by every change to the cache, its cold outputs or the
    /// committed state.
    version: u64,
}

/// What stays of outputs let go cold: the key they were made under, their
/// names and epoch, and each one's extent and layout, which is all a
/// reader's key and binding read of them before it runs.
#[derive(Clone, Debug)]
pub(super) struct Cold {
    pub(super) key: CacheKey,
    pub(super) output_names: Arc<Vec<String>>,
    pub(super) epoch: u64,
    pub(super) laid: Vec<(Domain, entelechy_abi::Schema)>,
}

impl Ledger {
    /// A node's outputs let go cold.
    pub(super) fn cold(&self, id: &NodeId) -> Option<&Cold> {
        self.cold.get(id)
    }

    /// The nodes whose outputs are cold, in order.
    pub(super) fn cold_nodes(&self) -> impl Iterator<Item = &NodeId> {
        in_order(self.cold.keys())
    }

    fn cold_put(&mut self, id: NodeId, cold: Cold) {
        self.version += 1;
        self.cache.remove(&id);
        self.cold.insert(id, cold);
    }

    /// A node's cache entry.
    pub(super) fn entry(&self, id: &NodeId) -> Option<&Cached> {
        self.cache.get(id)
    }

    /// The nodes the cache holds an entry for, in order.
    pub(super) fn cached_nodes(&self) -> impl Iterator<Item = &NodeId> {
        in_order(self.cache.keys())
    }

    /// A port's committed state.
    pub(super) fn committed(&self, key: &(NodeId, String)) -> Option<Handle> {
        self.committed.get(key).copied()
    }

    /// Every committed port and its state, in port order.
    pub(super) fn committed_ports(&self) -> std::vec::IntoIter<(&(NodeId, String), Handle)> {
        let mut ports: Vec<(&(NodeId, String), Handle)> = self
            .committed
            .iter()
            .map(|(key, handle)| (key, *handle))
            .collect();
        ports.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        ports.into_iter()
    }

    /// What every change to the ledger moves.
    pub(super) fn version(&self) -> u64 {
        self.version
    }

    /// How many cache entries and committed ports there are.
    pub(super) fn sizes(&self) -> (usize, usize) {
        (self.cache.len(), self.committed.len())
    }

    /// Takes out the cache and the committed state, clearing the counts.
    pub(super) fn take_all(
        &mut self,
    ) -> (BTreeMap<NodeId, Cached>, BTreeMap<(NodeId, String), Handle>) {
        self.version += 1;
        self.cached_handles.clear();
        self.committed_handles.clear();
        self.cold.clear();
        (
            std::mem::take(&mut self.cache).into_iter().collect(),
            std::mem::take(&mut self.committed).into_iter().collect(),
        )
    }

    /// Puts a node's outputs in the cache, keeping the count; answers what
    /// was there. Outputs it had let go cold are forgotten.
    pub(super) fn cache_put(&mut self, id: &NodeId, fresh: Cached) -> Option<Cached> {
        self.version += 1;
        self.cold.remove(id);
        for output in &fresh.outputs {
            *self.cached_handles.entry(*output).or_default() += 1;
        }
        // An entry replaced keeps its name: the name is copied only for a
        // node new to the cache.
        let gone = match self.cache.get_mut(id) {
            Some(slot) => Some(std::mem::replace(slot, fresh)),
            None => {
                self.cache.insert(id.clone(), fresh);
                None
            }
        };
        if let Some(gone) = &gone {
            forget(&mut self.cached_handles, gone.outputs.iter().copied());
        }
        gone
    }

    /// Takes a node's entry out of the cache, keeping the count, and
    /// forgets any outputs it let go cold.
    pub(super) fn cache_take(&mut self, id: &NodeId) -> Option<Cached> {
        self.version += 1;
        self.cold.remove(id);
        let gone = self.cache.remove(id);
        if let Some(gone) = &gone {
            forget(&mut self.cached_handles, gone.outputs.iter().copied());
        }
        gone
    }

    /// Commits a port's state, keeping the count; answers what was there.
    pub(super) fn commit_put(&mut self, key: (NodeId, String), handle: Handle) -> Option<Handle> {
        self.version += 1;
        *self.committed_handles.entry(handle).or_default() += 1;
        let gone = self.committed.insert(key, handle);
        if let Some(gone) = gone {
            forget(&mut self.committed_handles, std::iter::once(gone));
        }
        gone
    }

    /// Takes a port's committed state, keeping the count.
    pub(super) fn commit_take(&mut self, key: &(NodeId, String)) -> Option<Handle> {
        self.version += 1;
        let gone = self.committed.remove(key);
        if let Some(gone) = gone {
            forget(&mut self.committed_handles, std::iter::once(gone));
        }
        gone
    }

    pub(super) fn is_cached(&self, handle: Handle) -> bool {
        let held = self.cached_handles.contains_key(&handle);
        // Debug builds check the count against the cache.
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

/// `ids` in order.
fn in_order<'a>(ids: impl Iterator<Item = &'a NodeId>) -> std::vec::IntoIter<&'a NodeId> {
    let mut ids: Vec<&NodeId> = ids.collect();
    ids.sort_unstable();
    ids.into_iter()
}

/// Decrements each handle's count, removing it at zero.
fn forget(counts: &mut QuickMap<Handle, usize>, handles: impl Iterator<Item = Handle>) {
    for handle in handles {
        if let std::collections::hash_map::Entry::Occupied(mut at) = counts.entry(handle) {
            *at.get_mut() -= 1;
            if *at.get() == 0 {
                at.remove();
            }
        }
    }
}

/// Where a frame is, for making room: the group running, where each node's
/// outputs are last read, who reads each, and what is never let go.
pub(super) struct Room<'a> {
    pub(super) at: usize,
    pub(super) running: &'a NodeId,
    /// The group each node runs in.
    pub(super) runs_at: &'a BTreeMap<NodeId, usize>,
    pub(super) last_read: &'a BTreeMap<NodeId, usize>,
    pub(super) readers: &'a BTreeMap<NodeId, Vec<NodeId>>,
    pub(super) held: &'a BTreeSet<NodeId>,
}

impl Evaluator {
    /// Lets go cold ([`Evaluator::chill`]) cached outputs no later run of
    /// this frame reads, to make room in the Host: those read last before
    /// the group running, and those of nodes still to be reached
    /// downstream of one run this frame (`ran`), which run again when
    /// reached, and the running node's own, which it replaces. A node
    /// passed without running, reused or an `async` node's last output, is
    /// read as it is. What `held` names stays. Answers how many nodes let
    /// go; a later frame passes one by while its key is unchanged, and
    /// makes it again for a reader that runs.
    /// [`Evaluator::make_room`], the nodes let go added to `count`.
    pub(super) fn make_room_counted<'n>(
        &mut self,
        count: &mut usize,
        room: Room<'_>,
        ran: impl Iterator<Item = &'n NodeId>,
        runtime: &mut Runtime,
    ) -> usize {
        let done = self.make_room(room, ran, runtime);
        *count += done;
        done
    }

    pub(super) fn make_room<'n>(
        &mut self,
        room: Room<'_>,
        ran: impl Iterator<Item = &'n NodeId>,
        runtime: &mut Runtime,
    ) -> usize {
        let mut ran: BTreeSet<NodeId> = ran.cloned().collect();
        ran.insert(room.running.clone());
        let mut stale: BTreeSet<NodeId> = BTreeSet::new();
        let mut frontier: Vec<NodeId> = ran.iter().cloned().collect();
        while let Some(id) = frontier.pop() {
            for reader in room.readers.get(&id).into_iter().flatten() {
                if !ran.contains(reader) && stale.insert(reader.clone()) {
                    frontier.push(reader.clone());
                }
            }
        }
        let done: Vec<NodeId> = self
            .ledger
            .cached_nodes()
            .filter(|id| {
                !room.held.contains(*id)
                    && ((stale.contains(*id)
                        && room.runs_at.get(*id).is_some_and(|runs| *runs > room.at))
                        || *id == room.running
                        || room.last_read.get(*id).is_some_and(|read| *read < room.at))
            })
            .cloned()
            .collect();
        // Let go cold: a later frame that finds a key unchanged passes the
        // node by, and makes it again only for a reader that runs.
        for id in &done {
            self.chill(id, runtime);
        }
        done.len()
    }
}
