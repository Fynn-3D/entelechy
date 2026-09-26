//! Cached, dirty-tracked evaluation of a graph against a runtime, with
//! state committed between evaluations and pointwise chains fused.
//!
//! Every node's outputs are cached and reused while its Kernel generation,
//! parameters, domain, context, and upstream results are unchanged. State
//! lives in `<~` bindings: a node reads a source's value as it was
//! committed by the previous successful evaluation. All nodes of one
//! evaluation read the same committed state; the evaluation commits
//! atomically at its end, and a failed evaluation commits nothing, so state
//! never advances past a failure.
//!
//! Fusion: a producer whose single output is read by exactly one consumer,
//! only through `sample`, over the same domain, and which is neither a graph
//! output, a state source, nor pinned, is inlined into that consumer. The
//! fused group executes as one Kernel and is one cache unit; the
//! producer's image is never materialized. Pin a node to observe it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use orla_abi::{Capability, Context, Domain, FieldData, Handle, Invocation, Type, Value};
use orla_runtime::{
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
    /// absorbed, whose images were never materialized.
    pub fused: Vec<(NodeId, Vec<NodeId>)>,
    /// How long each executed node's Kernel took, in evaluation order:
    /// the backend's execute call, a GPU dispatch not waited for counting
    /// as its submission; under a measured profile, what the device's own
    /// clock says the work took.
    pub timings: Vec<(NodeId, std::time::Duration)>,
    /// Where each executed node ran, in evaluation order: the backend,
    /// why it was not the main one when it was not, and the bytes its
    /// run carried between Host memory and a device.
    pub placements: Vec<Placed>,
    /// State ports whose committed value was carried into a new layout
    /// by the binding's migration Kernel before the nodes ran.
    pub migrated: Migrated,
    /// State ports whose committed value no longer had the port's layout
    /// and, with no migration named, restarted from the initial source;
    /// with why.
    pub reset: Reset,
    /// `async` nodes whose run off the frame was going on, started by this
    /// evaluation or an earlier one; their readers took their last
    /// output.
    pub pending: Vec<NodeId>,
    /// `async` nodes whose run landed before this evaluation's nodes ran,
    /// so their readers saw the new output.
    pub landed: Vec<NodeId>,
}

/// Where one node ran; see [`Report::placements`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The node.
    pub node: NodeId,
    /// The backend that ran it.
    pub backend: &'static str,
    /// Why the main backend did not, when it did not.
    pub routed: Option<String>,
    /// Bytes its run brought to a device from Host memory.
    pub uploaded: u64,
    /// Bytes its run brought back to Host memory from a device.
    pub downloaded: u64,
}

/// State ports carried into a new layout: the reading node, its port,
/// and the Kernel that carried it.
pub type Migrated = Vec<(NodeId, String, KernelId)>;

/// State ports restarted from their initial source: the reading node,
/// its port, and why.
pub type Reset = Vec<(NodeId, String, String)>;

/// A state port an evaluation changed before its nodes ran: what was
/// committed, kept alive until the evaluation commits, so a failure can
/// put it back.
struct Replaced {
    key: (NodeId, String),
    /// What was committed there, or nothing: a restore that installs a
    /// port the world had not committed yet.
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
    /// Present only for Kernels that read the clock, so pure nodes stay cached
    /// across frames.
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

/// A fused version the runtime is still preparing. Until it arrives the
/// group's nodes execute separately, which gives the same images.
struct Pending {
    parts: Vec<(NodeId, Generation)>,
    ticket: Ticket,
}

/// An `async` node's run off the frame: the key it was started under,
/// which the landed outputs are cached with.
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
    /// How many times the Kernel runs, feeding `chain` from what the
    /// time before gave. One for an ordinary node; `times = N` on a
    /// node whose only changing input is its own output is N.
    repeat: u32,
    /// The input this run feeds from the last output, and which output
    /// it takes: the `carry port <- output` of the node that was
    /// placed several times.
    chain: Option<(usize, usize)>,
}

/// One executing unit as the planner sees it, for inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanEntry {
    /// The node that executes.
    pub node: NodeId,
    /// The Kernel it runs; a fused group runs a detached fused version.
    pub kernel: KernelId,
    /// Producers inlined into it, whose images are never written.
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
    /// The cached node outputs and the committed state, and the counts
    /// that say when a Field may be released ([`ledger::Ledger`]).
    ledger: ledger::Ledger,
    epoch: u64,
    fusion: bool,
    pinned: BTreeSet<NodeId>,
    fused: BTreeMap<NodeId, Fused>,
    /// The groups fusion declined, by consumer, at the generations it
    /// declined them: not tried again until one of them changes.
    declined: BTreeMap<NodeId, Vec<(NodeId, Generation)>>,
    pending: BTreeMap<NodeId, Pending>,
    running: BTreeMap<NodeId, Running>,
    /// State an evaluation replaced before its nodes ran (a migration, a
    /// restart), kept until it commits or put back when it fails; while
    /// here it is held as committed state is.
    replaced: Vec<Replaced>,
    /// Whether the evaluation running is a restore's trial: it runs as
    /// any does and commits nothing ([`Evaluator::restore`]).
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

    /// Lets a node be absorbed again.
    pub fn pins(&self) -> &BTreeSet<NodeId> {
        &self.pinned
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
    /// again on the next evaluation. Cached outputs stay. Runs off the
    /// frame are let go: they began from the state forgotten, and one
    /// that landed after would commit it again.
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

    /// What would execute for `graph` right now: the groups in evaluation
    /// order with what each absorbed. Prepares fused versions as evaluation
    /// would, but executes nothing.
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

    /// Evaluates only what `demand` needs: the demanded nodes, whatever
    /// they read through `<-`, every `<~` source (state must advance), and
    /// any graph output that has never been produced. The rest keeps its
    /// last value, so an output not demanded reads as of its last
    /// evaluation. This is how a test that asserts on a state row skips
    /// the picture.
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

    /// The invocation `node` would execute with now, its Kernel `version`
    /// unfused: the inputs from the cache and the committed state, the
    /// params and the domain resolved as an evaluation resolves them. For
    /// a trace, after an evaluation that produced what the node reads.
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
        // The walk below reaches every node the demand depends on, and
        // finding each by name was a walk of the whole list — which a
        // replay and the protocol pay every frame, since they are the
        // two that evaluate for a demand.
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
        // Everything from here to the commit either commits or leaves the
        // state as it was before this evaluation began: a node that fails
        // after a migration puts the state it replaced back.
        let evaluated = (|| -> Result<Report, EvaluateError> {
            // Nodes absorbed into a group are no longer materialized.
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
            // Every node run on the frame, in order, with what its runs
            // answered besides their outputs: whether a device's run failed,
            // known once the device is waited for, before anything commits.
            let mut checks: Vec<(NodeId, Vec<Box<dyn orla_runtime::Check>>)> = Vec::new();
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
                // A restore's trial runs every node on the frame: what it
                // proves is the world restored, whole, and it leaves no run
                // going that would land a world it may refuse.
                if node.is_async && !self.trial && self.ledger.entry(&node.id).is_some() {
                    // One run at a time; while it goes, the frame carries the
                    // last output and the state this node feeds stays.
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
                // A first run, or one with nothing cached to carry, is on the
                // frame; a run still going for it is no longer wanted.
                if let Some(running) = self.running.remove(&node.id) {
                    runtime.discard_run(running.run);
                }
                let placement = runtime.placement(&group.version);
                runtime.take_moved();
                let started = orla_runtime::clock::now();
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
                    .unwrap_or_else(|| orla_runtime::clock::since(started));
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
                            // What that node does write, so a port named
                            // slightly wrong is named.
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
            // A trial proves the world can run; the frame it ran is not
            // the world's, and commits nothing.
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

    /// Takes every run that has landed into the cache, as an execution on
    /// the frame would have gone in. A run that failed fails the
    /// evaluation the way a failure on the frame does.
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

    /// Waits for every run off the frame and takes each into the cache;
    /// the next evaluation reads them, and commits the state they feed.
    /// Returns the nodes that landed.
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
    // Handles here were created by executions this evaluator owns; a failure
    // means the interface released them behind our back, which is the only
    // case where ignoring the error is correct.
    let _ = runtime.host_mut().release(handle);
}

/// What a domain expression may take from the context: the window's size
/// while one is open, and its scale.
fn window_context(context: &Context) -> impl Fn(&str) -> Option<Value> + '_ {
    move |name: &str| match name {
        "window" if context.has_window() => Some(Value::Vec2(context.window)),
        "scale" => Some(Value::Float(context.scale)),
        _ => None,
    }
}

/// The graph with every external the caller did not supply, and that has
/// a fallback, read from its fallback instead; `None` when there is
/// nothing to substitute, which is every graph whose files are all there.
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
