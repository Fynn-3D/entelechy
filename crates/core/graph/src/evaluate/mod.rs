//! Cached, dirty-tracked evaluation of a graph, with state committed
//! between evaluations and pointwise chains fused.
//!
//! A node's outputs are reused while its Kernel generation, parameters,
//! domain, context and upstream results are unchanged. `<~` bindings read
//! the state committed by the previous successful evaluation; an
//! evaluation commits atomically at its end, and a failed one commits
//! nothing.
//!
//! Fusion inlines a producer into its one consumer when its single output
//! is read only through `sample`, over the same domain, and it is not a
//! graph output, state source or pinned node. The group executes as one
//! Kernel and one cache unit; pin a node to observe it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use orla_abi::{Capability, Context, Domain, FieldData, Handle, Invocation, Type, Value};
use orla_kernel::{
    ExecutionError, Generation, HostError, KernelId, KernelVersion, PrepareError, Run, Runtime,
    Ticket,
};

use crate::domain::VALUE_MAX_ELEMENTS;
use crate::{Binding, DomainExpr, DomainSpec, Graph, GraphError, Node, NodeId, ParamValue, Source};

mod bind;
mod error;
mod ledger;
mod plan;
mod state;

pub use error::EvaluateError;
use plan::{graph_param, suggest};

/// What one evaluation did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// The graph output image, owned by the evaluator's cache.
    pub output: Handle,
    /// Nodes that executed, in evaluation order.
    pub executed: Vec<NodeId>,
    /// Nodes whose cached outputs were reused.
    pub reused: Vec<NodeId>,
    /// Sources whose state advanced in this evaluation's commit.
    pub committed: Vec<(NodeId, String)>,
    /// Every declared output by name.
    pub outputs: Vec<(String, Handle)>,
    /// Groups that ran as one Kernel: the executing node and the nodes it
    /// absorbed.
    pub fused: Vec<(NodeId, Vec<NodeId>)>,
    /// Each executed node's Kernel time, in evaluation order: the execute
    /// call (a GPU submission), or the device clock under a measured
    /// profile.
    pub timings: Vec<(NodeId, std::time::Duration)>,
    /// Where each executed node ran, in evaluation order.
    pub placements: Vec<Placed>,
    /// State ports carried into a new layout by their migration Kernel.
    pub migrated: Migrated,
    /// State ports whose layout changed with no migration, restarted from
    /// the initial source, with why.
    pub reset: Reset,
    /// `async` nodes with a run in flight; their readers took the last
    /// output.
    pub pending: Vec<NodeId>,
    /// `async` nodes whose run landed before this evaluation's nodes ran.
    pub landed: Vec<NodeId>,
}

/// Where one node ran; see [`Report::placements`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The node.
    pub node: NodeId,
    /// The backend that ran it.
    pub backend: &'static str,
    /// Why the main backend did not run it, if so.
    pub routed: Option<String>,
    /// Bytes moved from Host memory to a device.
    pub uploaded: u64,
    /// Bytes moved from a device to Host memory.
    pub downloaded: u64,
}

/// State ports carried into a new layout: the reading node, its port,
/// and the Kernel that carried it.
pub type Migrated = Vec<(NodeId, String, KernelId)>;

/// State ports restarted from their initial source: the reading node,
/// its port, and why.
pub type Reset = Vec<(NodeId, String, String)>;

/// A state port an evaluation changed before its nodes ran, with what was
/// committed, kept until the commit so a failure can put it back.
struct Replaced {
    key: (NodeId, String),
    /// What was committed there; `None` when a restore installed the port.
    previous: Option<Handle>,
}

#[derive(Clone, Debug, PartialEq)]
enum InputKey {
    External(Handle),
    Node(NodeId, usize, u64),
    /// A committed value or an initial value, identified by handle.
    State(Handle),
}

#[derive(Clone, Debug, PartialEq)]
struct CacheKey {
    generation: Generation,
    params: Vec<Value>,
    inputs: Vec<InputKey>,
    domain: Domain,
    /// Present only for Kernels that read the clock.
    time: Option<u32>,
    /// Present only for Kernels that read input.
    input: Option<u64>,
}

#[derive(Debug)]
struct Cached {
    key: CacheKey,
    outputs: Vec<Handle>,
    output_names: Vec<String>,
    epoch: u64,
}

/// A fused version kept while its constituents are unchanged.
struct Fused {
    parts: Vec<(NodeId, Generation)>,
    version: Arc<KernelVersion>,
}

/// A fused version the runtime is still preparing; until it arrives the
/// group's nodes execute separately.
struct Pending {
    parts: Vec<(NodeId, Generation)>,
    ticket: Ticket,
}

/// An `async` node's run off the frame, with the cache key it lands under.
struct Running {
    run: Run,
    key: CacheKey,
    output_names: Vec<String>,
}

/// One executing unit: a node, a node with producers inlined, or a run
/// of the same node placed several times over its own output.
struct Group {
    node: Node,
    version: Arc<KernelVersion>,
    parts: Vec<(NodeId, Generation)>,
    absorbed: Vec<NodeId>,
    /// How many times the Kernel runs, each feeding `chain` from the one
    /// before: 1, or N for `times = N` over the node's own output.
    repeat: u32,
    /// The `carry port <- output` pair a repeated run feeds itself by.
    chain: Option<(usize, usize)>,
}

/// One executing unit as the planner sees it, for inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanEntry {
    /// The node that executes.
    pub node: NodeId,
    /// The Kernel it runs; a fused group runs a detached fused version.
    pub kernel: KernelId,
    /// Producers inlined into it.
    pub absorbed: Vec<NodeId>,
    /// The bindings the group executes with, absorbed producers' inputs
    /// renamed `producer/port`.
    pub inputs: BTreeMap<String, Binding>,
    /// The explicit or planned domain, when known before execution.
    pub domain: Option<Domain>,
}

/// Evaluates graphs and owns the cached node outputs and committed state.
#[derive(Default)]
pub struct Evaluator {
    /// Cached outputs, committed state and their release counts.
    ledger: ledger::Ledger,
    epoch: u64,
    fusion: bool,
    pinned: BTreeSet<NodeId>,
    fused: BTreeMap<NodeId, Fused>,
    /// Groups fusion declined, by consumer, at their generations; retried
    /// when one changes.
    declined: BTreeMap<NodeId, Vec<(NodeId, Generation)>>,
    pending: BTreeMap<NodeId, Pending>,
    running: BTreeMap<NodeId, Running>,
    /// State an evaluation replaced before its nodes ran (a migration, a
    /// restart), held until the commit or put back on failure.
    replaced: Vec<Replaced>,
    /// Whether the running evaluation is a restore's trial, which commits
    /// nothing ([`Evaluator::restore`]).
    trial: bool,
}

impl fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Evaluator")
            .field("cached", &self.ledger.sizes().0)
            .field("committed", &self.ledger.sizes().1)
            .field("fusion", &self.fusion)
            .finish()
    }
}

impl Evaluator {
    /// An evaluator with an empty cache, no state, and fusion enabled.
    pub fn new() -> Self {
        Self {
            fusion: true,
            ..Self::default()
        }
    }

    /// Enables or disables fusion for later evaluations.
    pub fn set_fusion(&mut self, enabled: bool) {
        self.fusion = enabled;
    }

    /// Whether fusion is enabled.
    pub fn fusion(&self) -> bool {
        self.fusion
    }

    /// Keeps a node materialized so its image can be observed.
    pub fn pin(&mut self, node: NodeId) {
        self.pinned.insert(node);
    }

    /// The nodes kept materialized.
    pub fn pins(&self) -> &BTreeSet<NodeId> {
        &self.pinned
    }

    /// The node `node` runs fused into, when the plan has folded it into
    /// its reader: it then keeps no output of its own until pinned.
    pub fn fused_into(&self, node: &NodeId) -> Option<&NodeId> {
        self.fused
            .iter()
            .find(|(consumer, fused)| {
                *consumer != node && fused.parts.iter().any(|(part, _)| part == node)
            })
            .map(|(consumer, _)| consumer)
    }

    /// Lets fusion have `node` back.
    pub fn unpin(&mut self, node: &NodeId) {
        self.pinned.remove(node);
    }

    /// Releases every cached image and all committed state.
    pub fn clear(&mut self, runtime: &mut Runtime) {
        let (cache, committed) = self.ledger.take_all();
        self.fused.clear();
        for (_, pending) in std::mem::take(&mut self.pending) {
            runtime.discard(pending.ticket);
        }
        for (_, running) in std::mem::take(&mut self.running) {
            runtime.discard_run(running.run);
        }
        let mut released = BTreeSet::new();
        for cached in cache.into_values() {
            for output in cached.outputs {
                if released.insert(output) {
                    release(runtime, output);
                }
            }
        }
        for handle in committed.into_values() {
            if released.insert(handle) {
                release(runtime, handle);
            }
        }
    }

    /// Forgets committed state so `<~` bindings read their initial sources
    /// next evaluation. Cached outputs stay; runs off the frame are
    /// dropped, since they began from the forgotten state.
    pub fn reset_state(&mut self, runtime: &mut Runtime) {
        self.recover(runtime);
        for (_, running) in std::mem::take(&mut self.running) {
            runtime.discard_run(running.run);
        }
        let ports: Vec<(NodeId, String)> = self
            .ledger
            .committed_ports()
            .map(|(key, _)| key.clone())
            .collect();
        for key in ports {
            self.forget_state(&key, runtime);
        }
    }

    /// Whether any state has been committed.
    pub fn has_state(&self) -> bool {
        self.ledger.sizes().1 > 0
    }

    /// The cached image for a node's output port, if it was evaluated and
    /// materialized. Absorbed nodes have none; pin them to observe them.
    pub fn cached(&self, node: &NodeId, port: &str) -> Option<Handle> {
        let cached = self.ledger.entry(node)?;
        let index = cached.output_names.iter().position(|name| name == port)?;
        Some(cached.outputs[index])
    }

    /// What would execute for `graph` now, in evaluation order. Prepares
    /// fused versions; executes nothing.
    pub fn plan_summary(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<Vec<PlanEntry>, EvaluateError> {
        let order = graph.order()?;
        let groups = self.plan(
            graph,
            &order,
            runtime,
            externals,
            overrides,
            &Context::default(),
        )?;
        Ok(groups
            .into_iter()
            .map(|group| PlanEntry {
                node: group.node.id.clone(),
                kernel: group.version.id().clone(),
                absorbed: group.absorbed,
                inputs: group.node.inputs,
                domain: group.node.domain.as_ref().and_then(DomainSpec::literal),
            })
            .collect())
    }

    /// Evaluates `graph` under `context`, executing only nodes whose inputs
    /// changed, then commits state. `params` overrides graph parameters by
    /// name; unlisted ones use their defaults.
    pub fn evaluate(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        self.evaluate_with(graph, runtime, externals, context, params, None)
    }

    /// Evaluates as [`Evaluator::evaluate`] does and commits nothing: `<~`
    /// readers keep the state last committed, and the results stay cached
    /// for reading; runs off the frame are not started.
    pub fn evaluate_trial(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        self.trial = true;
        let tried = self.evaluate_with(graph, runtime, externals, context, params, None);
        self.trial = false;
        tried
    }

    /// Evaluates only what `demand` needs: the demanded nodes, what they
    /// read through `<-`, every `<~` source, and any output never
    /// produced. Everything else keeps its last value.
    pub fn evaluate_for(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: &BTreeSet<NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        self.evaluate_with(graph, runtime, externals, context, params, Some(demand))
    }

    /// The unfused invocation `node` would execute with now, resolved as
    /// an evaluation resolves it; valid after an evaluation produced
    /// what the node reads.
    #[allow(clippy::too_many_arguments)]
    pub fn invocation(
        &self,
        graph: &Graph,
        node: &Node,
        version: &Arc<KernelVersion>,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
    ) -> Result<Invocation, EvaluateError> {
        self.bind(graph, node, version, runtime, externals, context, params)
            .map(|(invocation, _)| invocation)
    }

    /// The nodes an evaluation under `demand` must run: see
    /// [`Evaluator::evaluate_for`].
    fn needed(&self, graph: &Graph, demand: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
        let at = graph.index();
        let mut needed: BTreeSet<NodeId> = demand.clone();
        // State: its producer must run, and its initial source is read
        // until the first commit.
        for node in &graph.nodes {
            for binding in node.inputs.values() {
                if let Binding::Previous {
                    source, initial, ..
                } = binding
                {
                    for source in [source, initial] {
                        if let Source::Node { node, .. } = source {
                            needed.insert(node.clone());
                        }
                    }
                }
            }
        }
        for (_, source) in &graph.outputs {
            if let Source::Node { node, port } = source
                && self.cached(node, port).is_none()
            {
                needed.insert(node.clone());
            }
        }
        let mut frontier: Vec<NodeId> = needed.iter().cloned().collect();
        while let Some(id) = frontier.pop() {
            let Some(node) = at.get(&id).map(|at| &graph.nodes[*at]) else {
                continue;
            };
            for binding in node.inputs.values() {
                if let Binding::Now(Source::Node { node: producer, .. }) = binding
                    && needed.insert(producer.clone())
                {
                    frontier.push(producer.clone());
                }
            }
        }
        needed
    }

    fn evaluate_with(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: Option<&BTreeSet<NodeId>>,
    ) -> Result<Report, EvaluateError> {
        // An external the caller supplies nothing for reads its fallback,
        // written into a copy of the graph only when one is in use.
        let substituted = substitute_fallbacks(graph, externals);
        let graph = substituted.as_ref().unwrap_or(graph);
        let order = graph.order()?;
        self.prune(graph, runtime);
        let landed = self.land(runtime)?;
        let needed = demand.map(|demand| self.needed(graph, demand));
        let groups = self.plan(graph, &order, runtime, externals, params, context)?;
        let (migrated, reset) = self.reconcile_state(&groups, runtime, context)?;
        // From here to the commit, a failure leaves state as it was: a
        // migration's replaced state is put back.
        let evaluated = (|| -> Result<Report, EvaluateError> {
            // Absorbed nodes release their cached outputs.
            let absorbed: BTreeSet<&NodeId> =
                groups.iter().flat_map(|g| g.absorbed.iter()).collect();
            let stale: Vec<NodeId> = self
                .ledger
                .cached_nodes()
                .filter(|id| absorbed.contains(id))
                .cloned()
                .collect();
            for id in stale {
                self.drop_cache(&id, runtime);
            }
            let mut executed = Vec::new();
            let mut reused = Vec::new();
            let mut fused = Vec::new();
            let mut timings = Vec::new();
            let mut placements = Vec::new();
            // Every node run on the frame, in order, with its device
            // status, checked before anything commits.
            let mut checks: Vec<(NodeId, Vec<Box<dyn orla_kernel::Check>>)> = Vec::new();
            runtime.take_checks();
            let mut pending = Vec::new();
            for group in &groups {
                let node = &group.node;
                if needed
                    .as_ref()
                    .is_some_and(|needed| !needed.contains(&node.id))
                {
                    continue;
                }
                let (invocation, key) = self.bind(
                    graph,
                    node,
                    &group.version,
                    runtime,
                    externals,
                    context,
                    params,
                )?;
                if !group.absorbed.is_empty() {
                    fused.push((node.id.clone(), group.absorbed.clone()));
                }
                if self
                    .ledger
                    .entry(&node.id)
                    .is_some_and(|cached| cached.key == key)
                {
                    reused.push(node.id.clone());
                    continue;
                }
                let output_names: Vec<String> = group
                    .version
                    .manifest()
                    .outputs
                    .iter()
                    .map(|port| port.name.clone())
                    .collect();
                // A restore's trial runs every node on the frame, leaving
                // no run in flight.
                if node.is_async && !self.trial && self.ledger.entry(&node.id).is_some() {
                    // One run at a time; the frame carries the last output.
                    if !self.running.contains_key(&node.id) {
                        let run = runtime.execute_later(&group.version, &invocation).map_err(
                            |error| EvaluateError::Execution {
                                node: node.id.clone(),
                                error,
                            },
                        )?;
                        self.running.insert(
                            node.id.clone(),
                            Running {
                                run,
                                key,
                                output_names,
                            },
                        );
                    }
                    pending.push(node.id.clone());
                    continue;
                }
                // A first run, or one with nothing cached, is on the frame
                // and drops any run in flight.
                if let Some(running) = self.running.remove(&node.id) {
                    runtime.discard_run(running.run);
                }
                let placement = runtime.placement(&group.version);
                runtime.take_moved();
                let started = orla_kernel::clock::now();
                let outputs = match group.chain {
                    Some(chain) => {
                        runtime.execute_repeated(&group.version, &invocation, group.repeat, chain)
                    }
                    None => runtime.execute(&group.version, &invocation),
                }
                .map_err(|error| EvaluateError::Execution {
                    node: node.id.clone(),
                    error,
                })?;
                // What the device says it took, when it measures; the call's
                // own time otherwise.
                let took = runtime
                    .last_measured()
                    .unwrap_or_else(|| orla_kernel::clock::since(started));
                timings.push((node.id.clone(), took));
                checks.push((node.id.clone(), runtime.take_checks()));
                let moved = runtime.take_moved();
                placements.push(Placed {
                    node: node.id.clone(),
                    backend: placement.backend,
                    routed: placement.routed,
                    uploaded: moved.uploaded,
                    downloaded: moved.downloaded,
                });
                self.store(&node.id, key, outputs, output_names, runtime);
                executed.push(node.id.clone());
            }
            let mut outputs = Vec::with_capacity(graph.outputs.len());
            for (name, source) in &graph.outputs {
                let handle = match source {
                    Source::External(external) => {
                        *externals
                            .get(external)
                            .ok_or_else(|| EvaluateError::MissingExternal {
                                node: NodeId(format!("output {name}")),
                                name: external.clone(),
                            })?
                    }
                    Source::Node { node, port } => {
                        self.cached(node, port).ok_or_else(|| {
                            // Its real outputs, for the suggestion.
                            let near = self.ledger.entry(node).and_then(|cached| {
                                orla_abi::names::nearest(
                                    port,
                                    cached.output_names.iter().map(String::as_str),
                                )
                                .map(str::to_owned)
                            });
                            EvaluateError::UnknownOutput {
                                node: node.clone(),
                                port: port.clone(),
                                near,
                            }
                        })?
                    }
                };
                outputs.push((name.clone(), handle));
            }
            let output = outputs
                .first()
                .map(|(_, handle)| *handle)
                .ok_or(EvaluateError::Graph(GraphError::NoOutput))?;
            self.wait_for_checks(checks, runtime)?;
            // A trial commits nothing.
            let committed = if self.trial {
                Vec::new()
            } else {
                self.commit(graph, runtime)?
            };
            Ok(Report {
                output,
                executed,
                timings,
                placements,
                reused,
                committed,
                outputs,
                fused,
                migrated,
                reset,
                pending,
                landed,
            })
        })();
        match evaluated {
            Ok(report) => {
                self.release_replaced(runtime);
                Ok(report)
            }
            Err(error) => {
                self.restore_replaced(runtime);
                Err(error)
            }
        }
    }

    /// Caches a node's fresh outputs and lets go of the ones they replace,
    /// unless they are committed state.
    fn store(
        &mut self,
        id: &NodeId,
        key: CacheKey,
        outputs: Vec<Handle>,
        output_names: Vec<String>,
        runtime: &mut Runtime,
    ) {
        self.epoch += 1;
        let fresh = Cached {
            key,
            outputs,
            output_names,
            epoch: self.epoch,
        };
        if let Some(previous) = self.ledger.cache_put(id.clone(), fresh) {
            for output in previous.outputs {
                if !self.is_kept(output) {
                    release(runtime, output);
                }
            }
        }
    }

    /// Takes every landed run into the cache; a failed run fails the
    /// evaluation.
    fn land(&mut self, runtime: &mut Runtime) -> Result<Vec<NodeId>, EvaluateError> {
        let mut landed = Vec::new();
        let ids: Vec<NodeId> = self.running.keys().cloned().collect();
        for id in ids {
            let Some(outcome) = runtime.poll_run(self.running[&id].run) else {
                continue;
            };
            let running = self.running.remove(&id).expect("listed");
            self.take_landed(&id, running, outcome, runtime)?;
            landed.push(id);
        }
        Ok(landed)
    }

    /// Waits for every run off the frame, takes each into the cache, and
    /// returns the nodes that landed.
    pub fn settle(&mut self, runtime: &mut Runtime) -> Result<Vec<NodeId>, EvaluateError> {
        let mut landed = Vec::new();
        while let Some(id) = self.running.keys().next().cloned() {
            let running = self.running.remove(&id).expect("listed");
            let outcome = runtime.wait_run(running.run);
            self.take_landed(&id, running, outcome, runtime)?;
            landed.push(id);
        }
        Ok(landed)
    }

    /// Whether any node's run is still going.
    pub fn has_running(&self) -> bool {
        !self.running.is_empty()
    }

    fn take_landed(
        &mut self,
        id: &NodeId,
        running: Running,
        outcome: Result<Vec<Handle>, ExecutionError>,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        let outputs = outcome.map_err(|error| EvaluateError::Execution {
            node: id.clone(),
            error,
        })?;
        self.store(id, running.key, outputs, running.output_names, runtime);
        Ok(())
    }
}

fn release(runtime: &mut Runtime, handle: Handle) {
    // The evaluator owns these handles; a failure means the interface
    // already released one, so the error is ignored.
    let _ = runtime.host_mut().release(handle);
}

/// The context a domain expression reads: the window's size, while one
/// is open, and its scale.
fn window_context(context: &Context) -> impl Fn(&str) -> Option<Value> + '_ {
    move |name: &str| match name {
        "window" if context.has_window() => Some(Value::Vec2(context.window)),
        "scale" => Some(Value::Float(context.scale)),
        _ => None,
    }
}

/// The graph with each unsupplied external that has a fallback read from
/// it; `None` when nothing is substituted.
fn substitute_fallbacks(graph: &Graph, externals: &BTreeMap<String, Handle>) -> Option<Graph> {
    let missing: BTreeMap<&str, &Source> = graph
        .fallbacks
        .iter()
        .filter(|(name, _)| !externals.contains_key(name))
        .map(|(name, source)| (name.as_str(), source))
        .collect();
    if missing.is_empty() {
        return None;
    }
    let swap = |source: &mut Source| {
        if let Source::External(name) = source
            && let Some(fallback) = missing.get(name.as_str())
        {
            *source = (*fallback).clone();
        }
    };
    let mut graph = graph.clone();
    for node in &mut graph.nodes {
        for binding in node.inputs.values_mut() {
            match binding {
                Binding::Now(source) => swap(source),
                Binding::Previous { initial, .. } => swap(initial),
            }
        }
    }
    for (_, source) in &mut graph.outputs {
        swap(source);
    }
    Some(graph)
}
