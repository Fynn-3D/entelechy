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

use crate::{Binding, DomainSpec, Graph, GraphError, Node, NodeId, ParamValue, Source};

/// Why evaluation stopped. Nodes evaluated before the failure keep their new
/// results; the failing node keeps its last successful result, if any; no
/// state is committed.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluateError {
    /// The graph is malformed.
    Graph(GraphError),
    /// A node binds a Kernel that is not published.
    UnknownKernel {
        /// The node.
        node: NodeId,
        /// The missing Kernel.
        kernel: KernelId,
    },
    /// The caller did not supply a declared external image.
    MissingExternal {
        /// The node that needs it.
        node: NodeId,
        /// The external name.
        name: String,
    },
    /// A manifest input has no binding.
    UnboundInput {
        /// The node.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A binding or parameter names something the manifest does not have.
    UnknownPort {
        /// The node.
        node: NodeId,
        /// The port or parameter.
        port: String,
    },
    /// A parameter override has the wrong type.
    ParamType {
        /// The node.
        node: NodeId,
        /// The parameter.
        name: String,
        /// Declared type.
        expected: Type,
        /// Supplied type.
        actual: Type,
    },
    /// A source refers to an output port the producer does not declare.
    UnknownOutput {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A node has no image inputs and no explicit domain.
    NoDomain(NodeId),
    /// The Kernel failed while executing.
    Execution {
        /// The node.
        node: NodeId,
        /// The failure.
        error: ExecutionError,
    },
    /// A reduction node was given a `domain`; a reduction produces one
    /// element and maps over its first input.
    ReduceDomain(NodeId),
    /// A scatter node wrote a `domain`; it takes the extent of its target.
    ScatterDomain(NodeId),
    /// A node's `domain` expression could not be resolved.
    Domain {
        /// The node.
        node: NodeId,
        /// What is wrong.
        message: String,
    },
    /// A write named an output no `<~` binding reads, so it holds no
    /// state to replace.
    NotState {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
    },
    /// A write supplied the wrong number of words for the port's layout
    /// and extent.
    WordCount {
        /// The producer.
        node: NodeId,
        /// The port.
        port: String,
        /// Words the port holds.
        expected: usize,
        /// Words supplied.
        found: usize,
    },
    /// A port is bound to a resource of the other kind, or to a Field with
    /// a different element layout.
    PortKind {
        /// The node.
        node: NodeId,
        /// The port.
        port: String,
        /// What is wrong.
        message: String,
    },
    /// A `<~` port's committed state no longer has the port's layout and
    /// the binding's migration Kernel could not carry it over.
    Migration {
        /// The node reading the state.
        node: NodeId,
        /// The port.
        port: String,
        /// The migration Kernel.
        kernel: KernelId,
        /// What went wrong.
        message: String,
    },
    /// A fused group could not be prepared by the backend.
    Fusion {
        /// The consuming node.
        node: NodeId,
        /// The failure.
        message: String,
    },
    /// The Host rejected a resource operation.
    Host(HostError),
}

impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "{error}"),
            Self::UnknownKernel { node, kernel } => {
                write!(f, "node `{node}` binds `{kernel}`, which is not published")
            }
            Self::MissingExternal { node, name } => write!(
                f,
                "node `{node}` needs external image `{name}`, which was not supplied"
            ),
            Self::UnboundInput { node, port } => {
                write!(f, "node `{node}` does not bind input `{port}`")
            }
            Self::UnknownPort { node, port } => write!(
                f,
                "node `{node}` refers to `{port}`, which its kernel does not declare"
            ),
            Self::ParamType {
                node,
                name,
                expected,
                actual,
            } => write!(
                f,
                "node `{node}` sets `{name}` to {actual}, but the kernel expects {expected}"
            ),
            Self::UnknownOutput { node, port } => {
                write!(f, "node `{node}` has no output `{port}`")
            }
            Self::NoDomain(node) => write!(
                f,
                "node `{node}` has no image inputs; add `domain = vec2(width, height);`"
            ),
            Self::Execution { node, error } => write!(f, "node `{node}` failed: {error}"),
            Self::Fusion { node, message } => {
                write!(f, "fused group at `{node}` was rejected: {message}")
            }
            Self::Domain { node, message } => write!(f, "node `{node}`: {message}"),
            Self::NotState { node, port } => write!(
                f,
                "`{node}.{port}` is not state: no `<~` binding reads it, so there is nothing to replace"
            ),
            Self::WordCount {
                node,
                port,
                expected,
                found,
            } => write!(
                f,
                "`{node}.{port}` holds {expected} words; {found} were written"
            ),
            Self::ReduceDomain(node) => write!(
                f,
                "node `{node}` is a reduction, which produces one element; remove its `domain`"
            ),
            Self::ScatterDomain(node) => write!(
                f,
                "node `{node}` is a scatter, which writes the extent of the input it scatters into; remove its `domain`"
            ),
            Self::PortKind {
                node,
                port,
                message,
            } => write!(f, "node `{node}` port `{port}`: {message}"),
            Self::Migration {
                node,
                port,
                kernel,
                message,
            } => write!(
                f,
                "node `{node}` port `{port}`: its state migration `{kernel}` {message}; the state stays as it was"
            ),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EvaluateError {}

impl From<GraphError> for EvaluateError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<HostError> for EvaluateError {
    fn from(error: HostError) -> Self {
        Self::Host(error)
    }
}

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
    /// as its submission.
    pub timings: Vec<(NodeId, std::time::Duration)>,
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

/// State ports carried into a new layout: the reading node, its port,
/// and the Kernel that carried it.
pub type Migrated = Vec<(NodeId, String, KernelId)>;

/// State ports restarted from their initial source: the reading node,
/// its port, and why.
pub type Reset = Vec<(NodeId, String, String)>;

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

/// A graph parameter's value: the caller's override, or its default.
fn graph_param(graph: &Graph, name: &str, overrides: &BTreeMap<String, Value>) -> Option<Value> {
    overrides.get(name).copied().or_else(|| {
        graph
            .params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default)
    })
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

/// One executing unit: a node, or a node with producers inlined.
struct Group {
    node: Node,
    version: Arc<KernelVersion>,
    parts: Vec<(NodeId, Generation)>,
    absorbed: Vec<NodeId>,
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
    cache: BTreeMap<NodeId, Cached>,
    /// State as of the last successful evaluation, by producing node and
    /// port. A committed handle may also be a node's current cached output;
    /// it is released only when neither refers to it any more.
    committed: BTreeMap<(NodeId, String), Handle>,
    epoch: u64,
    fusion: bool,
    pinned: BTreeSet<NodeId>,
    fused: BTreeMap<NodeId, Fused>,
    pending: BTreeMap<NodeId, Pending>,
    running: BTreeMap<NodeId, Running>,
}

impl fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Evaluator")
            .field("cached", &self.cache.len())
            .field("committed", &self.committed.len())
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
        let cache = std::mem::take(&mut self.cache);
        let committed = std::mem::take(&mut self.committed);
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
    /// again on the next evaluation. Cached outputs stay.
    pub fn reset_state(&mut self, runtime: &mut Runtime) {
        for (_, handle) in std::mem::take(&mut self.committed) {
            if !self.is_cached(handle) {
                release(runtime, handle);
            }
        }
    }

    /// Whether any state has been committed.
    pub fn has_state(&self) -> bool {
        !self.committed.is_empty()
    }

    /// The cached image for a node's output port, if it was evaluated and
    /// materialized. Absorbed nodes have none; pin them to observe them.
    pub fn cached(&self, node: &NodeId, port: &str) -> Option<Handle> {
        let cached = self.cache.get(node)?;
        let index = cached.output_names.iter().position(|name| name == port)?;
        Some(cached.outputs[index])
    }

    fn is_cached(&self, handle: Handle) -> bool {
        self.cache
            .values()
            .any(|cached| cached.outputs.contains(&handle))
    }

    fn is_committed(&self, handle: Handle) -> bool {
        self.committed
            .values()
            .any(|committed| *committed == handle)
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
        runtime: &Runtime,
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
            let Some(node) = graph.node(&id) else {
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
        let order = graph.order()?;
        self.prune(graph, runtime);
        let landed = self.land(runtime)?;
        let needed = demand.map(|demand| self.needed(graph, demand));
        let groups = self.plan(graph, &order, runtime, externals, params, context)?;
        let (migrated, reset) = self.reconcile_state(&groups, runtime, context)?;
        // Nodes absorbed into a group are no longer materialized.
        let absorbed: BTreeSet<&NodeId> = groups.iter().flat_map(|g| g.absorbed.iter()).collect();
        let stale: Vec<NodeId> = self
            .cache
            .keys()
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
                .cache
                .get(&node.id)
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
            if node.is_async && self.cache.contains_key(&node.id) {
                // One run at a time; while it goes, the frame carries the
                // last output and the state this node feeds stays.
                if !self.running.contains_key(&node.id) {
                    let run =
                        runtime
                            .execute_later(&group.version, &invocation)
                            .map_err(|error| EvaluateError::Execution {
                                node: node.id.clone(),
                                error,
                            })?;
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
            let started = std::time::Instant::now();
            let outputs = runtime
                .execute(&group.version, &invocation)
                .map_err(|error| EvaluateError::Execution {
                    node: node.id.clone(),
                    error,
                })?;
            timings.push((node.id.clone(), started.elapsed()));
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
                    self.cached(node, port)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: node.clone(),
                            port: port.clone(),
                        })?
                }
            };
            outputs.push((name.clone(), handle));
        }
        let output = outputs
            .first()
            .map(|(_, handle)| *handle)
            .ok_or(EvaluateError::Graph(GraphError::NoOutput))?;
        let committed = self.commit(graph, runtime)?;
        Ok(Report {
            output,
            executed,
            timings,
            reused,
            committed,
            outputs,
            fused,
            migrated,
            reset,
            pending,
            landed,
        })
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
        if let Some(previous) = self.cache.insert(id.clone(), fresh) {
            for output in previous.outputs {
                if !self.is_committed(output) {
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

    /// Before the nodes run: every `<~` port whose committed state no
    /// longer has the port's shape and layout (a republished Kernel
    /// changed what it carries) is carried over by the binding's
    /// migration Kernel, a map from the old elements to the new run once
    /// over the state; without one, the port restarts from its initial
    /// source. A migration that fails, or gives the wrong layout, fails
    /// the evaluation and leaves the state as it was.
    fn reconcile_state(
        &mut self,
        groups: &[Group],
        runtime: &mut Runtime,
        context: &Context,
    ) -> Result<(Migrated, Reset), EvaluateError> {
        let mut migrated = Vec::new();
        let mut reset = Vec::new();
        for group in groups {
            let manifest = group.version.manifest();
            for (name, binding) in &group.node.inputs {
                let Binding::Previous {
                    source:
                        Source::Node {
                            node: producer,
                            port,
                        },
                    migration,
                    ..
                } = binding
                else {
                    continue;
                };
                let Some(input) = manifest.inputs.iter().find(|input| &input.name == name) else {
                    continue;
                };
                let key = (producer.clone(), port.clone());
                let Some(current) = self.committed.get(&key).copied() else {
                    continue;
                };
                let shape = runtime.host().domain(current)?.shape();
                let schema = runtime.host().schema(current)?.clone();
                if shape == input.ty.shape && schema == input.ty.schema {
                    continue;
                }
                let mismatch = if shape != input.ty.shape {
                    format!("expects {} but the state is a {shape}", input.ty.shape)
                } else {
                    format!(
                        "expects elements laid out as {} but the state is {schema}",
                        input.ty.schema
                    )
                };
                let carried = match migration {
                    Some(kernel) => {
                        let fail = |message: String| EvaluateError::Migration {
                            node: group.node.id.clone(),
                            port: name.clone(),
                            kernel: kernel.clone(),
                            message,
                        };
                        let version = runtime
                            .kernel(kernel)
                            .ok_or_else(|| fail("is not published".into()))?;
                        let carried = migrate(runtime, &version, current, context).map_err(fail)?;
                        let carried_shape = runtime.host().domain(carried)?.shape();
                        let carried_schema = runtime.host().schema(carried)?.clone();
                        if carried_shape != input.ty.shape || carried_schema != input.ty.schema {
                            release(runtime, carried);
                            return Err(fail(format!(
                                "gave a {carried_shape} of {carried_schema}, and the port {mismatch}"
                            )));
                        }
                        migrated.push((group.node.id.clone(), name.clone(), kernel.clone()));
                        Some(carried)
                    }
                    None => {
                        reset.push((group.node.id.clone(), name.clone(), mismatch));
                        None
                    }
                };
                self.committed.remove(&key);
                if !self.is_cached(current) && !self.is_committed(current) {
                    release(runtime, current);
                }
                if let Some(carried) = carried {
                    self.committed.insert(key, carried);
                }
            }
        }
        Ok((migrated, reset))
    }

    /// Decides what executes: every node as its own group, then producers
    /// folded into their sole pointwise consumers when fusion is on.
    fn plan(
        &mut self,
        graph: &Graph,
        order: &[usize],
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Result<Vec<Group>, EvaluateError> {
        let mut groups: Vec<Group> = Vec::with_capacity(order.len());
        for index in order {
            let node = &graph.nodes[*index];
            let version =
                runtime
                    .kernel(&node.kernel)
                    .ok_or_else(|| EvaluateError::UnknownKernel {
                        node: node.id.clone(),
                        kernel: node.kernel.clone(),
                    })?;
            let parts = vec![(node.id.clone(), version.generation())];
            groups.push(Group {
                node: node.clone(),
                version,
                parts,
                absorbed: Vec::new(),
            });
        }
        if !self.fusion {
            self.fused.clear();
            self.forget_pending(runtime, |_| false);
            return Ok(groups);
        }
        // How many readers each output has: bindings and graph outputs.
        let mut readers: BTreeMap<(NodeId, String), usize> = BTreeMap::new();
        let mut count = |source: &Source| {
            if let Source::Node { node, port } = source {
                *readers.entry((node.clone(), port.clone())).or_default() += 1;
            }
        };
        for node in &graph.nodes {
            for binding in node.inputs.values() {
                match binding {
                    Binding::Now(source) => count(source),
                    Binding::Previous {
                        source, initial, ..
                    } => {
                        count(source);
                        count(initial);
                        // State sources must be materialized; count twice.
                        count(source);
                    }
                }
            }
        }
        for (_, source) in &graph.outputs {
            count(source);
        }
        let mut domains: BTreeMap<NodeId, Domain> = BTreeMap::new();
        let mut absorbed_into: BTreeMap<NodeId, usize> = BTreeMap::new();
        for position in 0..groups.len() {
            // Planned domain of this group, for the equality check.
            let planned = self.planned_domain(
                graph,
                &groups[position],
                runtime,
                externals,
                &domains,
                overrides,
                context,
            );
            let ports: Vec<(String, Binding)> = groups[position]
                .node
                .inputs
                .iter()
                .map(|(port, binding)| (port.clone(), binding.clone()))
                .collect();
            for (port, binding) in ports {
                let Binding::Now(Source::Node {
                    node: producer_id,
                    port: producer_port,
                }) = binding
                else {
                    continue;
                };
                if readers
                    .get(&(producer_id.clone(), producer_port.clone()))
                    .copied()
                    != Some(1)
                    || self.pinned.contains(&producer_id)
                    || absorbed_into.contains_key(&producer_id)
                    || groups[position].node.is_async
                    || graph.node(&producer_id).is_some_and(|node| node.is_async)
                {
                    continue;
                }
                let Some(producer_position) = groups.iter().position(|g| g.node.id == producer_id)
                else {
                    continue;
                };
                let producer_domain = domains.get(&producer_id).copied();
                if producer_domain.is_none() || producer_domain != planned {
                    continue;
                }
                let consumer = &groups[position];
                let Some(input_index) = consumer
                    .version
                    .manifest()
                    .inputs
                    .iter()
                    .position(|input| input.name == port)
                else {
                    continue;
                };
                // Native operations have no IR to inline or be inlined into,
                // and a reduction neither maps over its output nor yields
                // one element per consumer element.
                let (Some(consumer_module), Some(producer_module)) = (
                    consumer.version.module(),
                    groups[producer_position].version.module(),
                ) else {
                    continue;
                };
                if consumer_module.manifest.mode != orla_ir::Mode::Map
                    || producer_module.manifest.mode != orla_ir::Mode::Map
                {
                    continue;
                }
                if !orla_ir::samples_only(consumer_module, input_index as u32)
                    || groups[producer_position].version.manifest().outputs.len() != 1
                {
                    continue;
                }
                let producer = &groups[producer_position];
                let mut parts = consumer.parts.clone();
                parts.extend(producer.parts.iter().cloned());
                let fuse = || {
                    orla_ir::fuse(
                        consumer_module,
                        input_index as u32,
                        producer_module,
                        &producer.node.id.0,
                    )
                };
                let Some(version) =
                    self.fused_version(runtime, &consumer.node.id, parts.clone(), fuse)?
                else {
                    continue;
                };
                // Rewire: the fused input disappears; the producer's inputs
                // and params join under `producer/name`.
                let producer = &groups[producer_position];
                let prefix = producer.node.id.0.clone();
                let mut inputs = consumer.node.inputs.clone();
                inputs.remove(&port);
                for (name, binding) in &producer.node.inputs {
                    inputs.insert(format!("{prefix}/{name}"), binding.clone());
                }
                let mut params = consumer.node.params.clone();
                for (name, value) in &producer.node.params {
                    params.insert(format!("{prefix}/{name}"), value.clone());
                }
                let mut absorbed = consumer.absorbed.clone();
                absorbed.push(producer_id.clone());
                absorbed.extend(producer.absorbed.iter().cloned());
                let group = &mut groups[position];
                group.version = version;
                group.parts = parts;
                group.absorbed = absorbed;
                group.node.inputs = inputs;
                group.node.params = params;
                if group.node.domain.is_none() {
                    group.node.domain = planned.map(DomainSpec::Literal);
                }
                absorbed_into.insert(producer_id, position);
            }
            if let Some(domain) = planned {
                domains.insert(groups[position].node.id.clone(), domain);
                // The plan reports what an expression resolved to.
                if matches!(groups[position].node.domain, Some(DomainSpec::Expr(_))) {
                    groups[position].node.domain = Some(DomainSpec::Literal(domain));
                }
            }
        }
        let executing: Vec<Group> = groups
            .into_iter()
            .filter(|group| !absorbed_into.contains_key(&group.node.id))
            .collect();
        // Keep every group's version while its node is in the graph: a
        // chain is planned from its head each time, so the inner groups
        // (gray with source, under inv) are needed again next time.
        let present: BTreeSet<&NodeId> = graph.nodes.iter().map(|node| &node.id).collect();
        self.fused.retain(|node, _| present.contains(node));
        self.forget_pending(runtime, |node| present.contains(node));
        Ok(executing)
    }

    /// The fused version for `parts`, if it is ready: kept from an earlier
    /// evaluation, prepared inline, or collected from the runtime's worker.
    /// `None` means it is still being prepared and the nodes run separately
    /// this time.
    fn fused_version(
        &mut self,
        runtime: &mut Runtime,
        consumer: &NodeId,
        parts: Vec<(NodeId, Generation)>,
        fuse: impl FnOnce() -> Result<orla_ir::Validated, orla_ir::FuseError>,
    ) -> Result<Option<Arc<KernelVersion>>, EvaluateError> {
        if let Some(existing) = self.fused.get(consumer)
            && existing.parts == parts
        {
            return Ok(Some(Arc::clone(&existing.version)));
        }
        let fusion = |message: String| EvaluateError::Fusion {
            node: consumer.clone(),
            message,
        };
        let ticket = match self.pending.get(consumer) {
            Some(pending) if pending.parts == parts => pending.ticket,
            other => {
                if let Some(superseded) = other {
                    runtime.discard(superseded.ticket);
                }
                let module = fuse().map_err(|error| fusion(error.to_string()))?;
                let source = format!(
                    "// fused: {}\n",
                    parts
                        .iter()
                        .map(|(node, generation)| format!("{node}@{generation}"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                );
                let ticket = runtime.prepare_later(module, source);
                self.pending.insert(
                    consumer.clone(),
                    Pending {
                        parts: parts.clone(),
                        ticket,
                    },
                );
                ticket
            }
        };
        let Some(result) = runtime.poll(ticket) else {
            return Ok(None);
        };
        self.pending.remove(consumer);
        let candidate = result.map_err(|error: PrepareError| fusion(error.to_string()))?;
        let version = runtime.detach(candidate, fused_id(consumer));
        self.fused.insert(
            consumer.clone(),
            Fused {
                parts,
                version: Arc::clone(&version),
            },
        );
        Ok(Some(version))
    }

    /// Drops pending preparations for nodes `keep` rejects.
    fn forget_pending(&mut self, runtime: &mut Runtime, keep: impl Fn(&NodeId) -> bool) {
        let dropped: Vec<NodeId> = self
            .pending
            .keys()
            .filter(|node| !keep(node))
            .cloned()
            .collect();
        for node in dropped {
            if let Some(pending) = self.pending.remove(&node) {
                runtime.discard(pending.ticket);
            }
        }
    }

    /// The domain a group will execute over, when it can be known before
    /// execution: explicit, or the size of its first bound image.
    #[allow(clippy::too_many_arguments)]
    fn planned_domain(
        &self,
        graph: &Graph,
        group: &Group,
        runtime: &Runtime,
        externals: &BTreeMap<String, Handle>,
        domains: &BTreeMap<NodeId, Domain>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Option<Domain> {
        let manifest = group.version.manifest();
        if manifest.mode == orla_ir::Mode::Reduce {
            return Some(Domain::Rows(1));
        }
        let size = |handle: Handle| runtime.host().domain(handle).ok();
        let plain = |source: &Source| match source {
            Source::External(name) => externals.get(name).and_then(|h| size(*h)),
            Source::Node { node, .. } => domains.get(node).copied(),
        };
        // A memory port of a node has the memory's extent, not the node's
        // domain (a scan runs over its ticks, its grid memory is a grid):
        // what it committed last, else what its initial value is sized.
        let of_source = |source: &Source| match source {
            Source::Node { node, port } => {
                match graph.node(node).and_then(|n| n.inputs.get(port)) {
                    Some(Binding::Previous { initial, .. }) => self
                        .committed
                        .get(&(node.clone(), port.clone()))
                        .and_then(|h| size(*h))
                        .or_else(|| plain(initial)),
                    _ => plain(source),
                }
            }
            external => plain(external),
        };
        let of_binding = |binding: &Binding| match binding {
            Binding::Now(source) => of_source(source),
            Binding::Previous {
                source: Source::Node { node, port },
                initial,
                ..
            } => self
                .committed
                .get(&(node.clone(), port.clone()))
                .and_then(|h| size(*h))
                .or_else(|| of_source(initial)),
            Binding::Previous { .. } => None,
        };
        if let orla_ir::Mode::Scatter { base, .. } = manifest.mode {
            let target = manifest.inputs.get(base as usize)?;
            return of_binding(group.node.inputs.get(&target.name)?);
        }
        match &group.node.domain {
            Some(DomainSpec::Literal(domain)) => Some(*domain),
            Some(DomainSpec::Expr(expr)) => {
                let port = |name: &str| group.node.inputs.get(name).and_then(of_binding);
                let param = |name: &str| {
                    let manifest_default = manifest
                        .params
                        .iter()
                        .find(|p| p.name == name)
                        .map(|p| p.default);
                    match group.node.params.get(name) {
                        Some(ParamValue::Literal(value)) => Some(*value),
                        Some(ParamValue::Graph(name)) => graph_param(graph, name, overrides),
                        None => manifest_default.or_else(|| graph_param(graph, name, overrides)),
                    }
                };
                expr.resolve(&port, &param, &window_context(context)).ok()
            }
            None => {
                let first = manifest.inputs.first()?;
                of_binding(group.node.inputs.get(&first.name)?)
            }
        }
    }

    fn drop_cache(&mut self, id: &NodeId, runtime: &mut Runtime) {
        if let Some(cached) = self.cache.remove(id) {
            for output in cached.outputs {
                if !self.is_committed(output) {
                    release(runtime, output);
                }
            }
        }
    }

    /// Releases cache entries and state of nodes that left the graph.
    fn prune(&mut self, graph: &Graph, runtime: &mut Runtime) {
        let stale: Vec<NodeId> = self
            .cache
            .keys()
            .filter(|id| graph.node(id).is_none())
            .cloned()
            .collect();
        for id in stale {
            self.drop_cache(&id, runtime);
        }
        let gone: Vec<NodeId> = self
            .running
            .keys()
            .filter(|id| graph.node(id).is_none())
            .cloned()
            .collect();
        for id in gone {
            if let Some(running) = self.running.remove(&id) {
                runtime.discard_run(running.run);
            }
        }
        let stale_state: Vec<(NodeId, String)> = self
            .committed
            .keys()
            .filter(|(node, _)| graph.node(node).is_none())
            .cloned()
            .collect();
        for key in stale_state {
            if let Some(handle) = self.committed.remove(&key)
                && !self.is_cached(handle)
            {
                release(runtime, handle);
            }
        }
        self.pinned.retain(|node| graph.node(node).is_some());
    }

    /// Replaces the committed state of `node.port` with `words`, laid out
    /// as the port's current value is, so the next evaluation's `<~`
    /// readers start from it: how a Field comes in from outside, a saved
    /// file for instance. The port must be state, read by some `<~`
    /// binding, and must have been evaluated once, so its layout and
    /// extent are known.
    pub fn write(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        node: &NodeId,
        port: &str,
        words: Vec<u32>,
    ) -> Result<(), EvaluateError> {
        let is_state = graph.nodes.iter().any(|reader| {
            reader.inputs.values().any(|binding| {
                matches!(binding, Binding::Previous { source: Source::Node { node: producer, port: read }, .. }
                    if producer == node && read == port)
            })
        });
        if !is_state {
            return Err(EvaluateError::NotState {
                node: node.clone(),
                port: port.to_owned(),
            });
        }
        let key = (node.clone(), port.to_owned());
        let current = self
            .committed
            .get(&key)
            .copied()
            .or_else(|| self.cached(node, port))
            .ok_or_else(|| EvaluateError::UnknownOutput {
                node: node.clone(),
                port: port.to_owned(),
            })?;
        let domain = runtime.host().domain(current)?;
        let schema = runtime.host().schema(current)?.clone();
        let expected = domain.count() * schema.words();
        if words.len() != expected {
            return Err(EvaluateError::WordCount {
                node: node.clone(),
                port: port.to_owned(),
                expected,
                found: words.len(),
            });
        }
        let data = FieldData::from_words(schema, domain, words)?;
        let handle = runtime.host_mut().adopt(data)?;
        if let Some(previous) = self.committed.insert(key, handle)
            && !self.is_cached(previous)
            && !self.is_committed(previous)
        {
            release(runtime, previous);
        }
        Ok(())
    }

    /// Makes every `<~` source's current output the committed state.
    fn commit(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
    ) -> Result<Vec<(NodeId, String)>, EvaluateError> {
        let mut advanced = Vec::new();
        for node in &graph.nodes {
            for binding in node.inputs.values() {
                let Binding::Previous {
                    source: Source::Node { node, port },
                    ..
                } = binding
                else {
                    continue;
                };
                let key = (node.clone(), port.clone());
                // A run off the frame has not given this evaluation its
                // value: the state stays until it lands.
                if advanced.contains(&key) || self.running.contains_key(node) {
                    continue;
                }
                let current =
                    self.cached(node, port)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: node.clone(),
                            port: port.clone(),
                        })?;
                let previous = self.committed.insert(key.clone(), current);
                if previous != Some(current) {
                    advanced.push(key);
                    if let Some(previous) = previous
                        && !self.is_cached(previous)
                        && !self.is_committed(previous)
                    {
                        release(runtime, previous);
                    }
                }
            }
        }
        Ok(advanced)
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(
        &self,
        graph: &Graph,
        node: &Node,
        version: &Arc<KernelVersion>,
        runtime: &Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<(Invocation, CacheKey), EvaluateError> {
        let manifest = version.manifest();
        for port in node.inputs.keys().chain(node.params.keys()) {
            let declared = manifest.inputs.iter().any(|input| &input.name == port)
                || manifest.params.iter().any(|param| &param.name == port);
            if !declared {
                return Err(EvaluateError::UnknownPort {
                    node: node.id.clone(),
                    port: port.clone(),
                });
            }
        }
        let mut inputs = Vec::with_capacity(manifest.inputs.len());
        let mut input_keys = Vec::with_capacity(manifest.inputs.len());
        for input in &manifest.inputs {
            let binding =
                node.inputs
                    .get(&input.name)
                    .ok_or_else(|| EvaluateError::UnboundInput {
                        node: node.id.clone(),
                        port: input.name.clone(),
                    })?;
            let (handle, key) = match binding {
                Binding::Now(source) => self.resolve(node, source, externals)?,
                Binding::Previous {
                    source:
                        Source::Node {
                            node: producer,
                            port,
                        },
                    initial,
                    ..
                } => match self.committed.get(&(producer.clone(), port.clone())) {
                    Some(handle) => (*handle, InputKey::State(*handle)),
                    None => {
                        let (handle, _) = self.resolve(node, initial, externals)?;
                        (handle, InputKey::State(handle))
                    }
                },
                Binding::Previous { .. } => {
                    return Err(EvaluateError::Graph(GraphError::FeedbackNeedsNode {
                        node: node.id.clone(),
                        port: input.name.clone(),
                    }));
                }
            };
            // The bound resource must have the shape and layout the port
            // declares; the spelling of either side does not matter.
            let bound_shape = runtime.host().domain(handle)?.shape();
            let bound_schema = runtime.host().schema(handle)?;
            let message = if input.ty.shape != bound_shape {
                Some(format!(
                    "expects {} but is bound to a {bound_shape}",
                    input.ty.shape
                ))
            } else if input.ty.schema != *bound_schema {
                Some(format!(
                    "expects elements laid out as {} but is bound to {bound_schema}",
                    input.ty.schema
                ))
            } else {
                None
            };
            if let Some(message) = message {
                return Err(EvaluateError::PortKind {
                    node: node.id.clone(),
                    port: input.name.clone(),
                    message,
                });
            }
            inputs.push(handle);
            input_keys.push(key);
        }
        let mut params = Vec::with_capacity(manifest.params.len());
        for param in &manifest.params {
            let given = match node.params.get(&param.name) {
                None => None,
                Some(ParamValue::Literal(value)) => Some(*value),
                Some(ParamValue::Graph(name)) => Some(
                    overrides
                        .get(name)
                        .copied()
                        .or_else(|| {
                            graph
                                .params
                                .iter()
                                .find(|p| &p.name == name)
                                .map(|p| p.default)
                        })
                        .ok_or_else(|| EvaluateError::UnknownPort {
                            node: node.id.clone(),
                            port: name.clone(),
                        })?,
                ),
            };
            let value = match given {
                None => param.default,
                Some(Value::Int(v)) if param.default.ty() == Type::Float => Value::Float(v as f32),
                Some(value) if value.ty() == param.default.ty() => value,
                Some(value) => {
                    return Err(EvaluateError::ParamType {
                        node: node.id.clone(),
                        name: param.name.clone(),
                        expected: param.default.ty(),
                        actual: value.ty(),
                    });
                }
            };
            params.push(value);
        }
        let domain = if manifest.mode == orla_ir::Mode::Reduce {
            if node.domain.is_some() {
                return Err(EvaluateError::ReduceDomain(node.id.clone()));
            }
            Domain::Rows(1)
        } else if let orla_ir::Mode::Scatter { base, .. } = manifest.mode {
            if node.domain.is_some() {
                return Err(EvaluateError::ScatterDomain(node.id.clone()));
            }
            runtime.host().domain(inputs[base as usize])?
        } else {
            match (&node.domain, inputs.first()) {
                (Some(DomainSpec::Literal(domain)), _) => *domain,
                (Some(DomainSpec::Expr(expr)), _) => {
                    let port = |name: &str| {
                        manifest
                            .inputs
                            .iter()
                            .position(|input| input.name == name)
                            .and_then(|index| runtime.host().domain(inputs[index]).ok())
                    };
                    let param = |name: &str| {
                        manifest
                            .params
                            .iter()
                            .position(|p| p.name == name)
                            .map(|index| params[index])
                            .or_else(|| graph_param(graph, name, overrides))
                    };
                    expr.resolve(&port, &param, &window_context(context))
                        .map_err(|e| EvaluateError::Domain {
                            node: node.id.clone(),
                            message: e.message,
                        })?
                }
                (None, Some(first)) => runtime.host().domain(*first)?,
                (None, None) => return Err(EvaluateError::NoDomain(node.id.clone())),
            }
        };
        let reads_clock = manifest.requires.contains(Capability::TimeRead);
        let reads_input = manifest.requires.contains(Capability::InputRead);
        let key = CacheKey {
            generation: version.generation(),
            params: params.clone(),
            inputs: input_keys,
            domain,
            time: reads_clock
                .then_some(context.time.to_bits() ^ context.rate.to_bits().rotate_left(7)),
            input: reads_input.then_some(context.input_fingerprint()),
        };
        Ok((
            Invocation {
                inputs,
                params,
                domain,
                context: *context,
            },
            key,
        ))
    }

    /// Resolves a `Now` source to a handle and its cache key.
    fn resolve(
        &self,
        node: &Node,
        source: &Source,
        externals: &BTreeMap<String, Handle>,
    ) -> Result<(Handle, InputKey), EvaluateError> {
        match source {
            Source::External(name) => {
                let handle =
                    *externals
                        .get(name)
                        .ok_or_else(|| EvaluateError::MissingExternal {
                            node: node.id.clone(),
                            name: name.clone(),
                        })?;
                Ok((handle, InputKey::External(handle)))
            }
            Source::Node {
                node: producer,
                port,
            } => {
                // The producer precedes this node in topological order, so a
                // missing entry means it never produced this port.
                let cached =
                    self.cache
                        .get(producer)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: producer.clone(),
                            port: port.clone(),
                        })?;
                let index = cached
                    .output_names
                    .iter()
                    .position(|name| name == port)
                    .ok_or_else(|| EvaluateError::UnknownOutput {
                        node: producer.clone(),
                        port: port.clone(),
                    })?;
                Ok((
                    cached.outputs[index],
                    InputKey::Node(producer.clone(), index, cached.epoch),
                ))
            }
        }
    }
}

/// The id a fused group executes under: `fused.<node>@1` with the node
/// name made id-safe.
fn fused_id(node: &NodeId) -> KernelId {
    let segment: String = node
        .0
        .chars()
        .map(|c| match c.to_ascii_lowercase() {
            c @ ('a'..='z' | '0'..='9' | '_') => c,
            _ => '_',
        })
        .collect();
    let segment = if segment.starts_with(|c: char| c.is_ascii_digit()) || segment.is_empty() {
        format!("n{segment}")
    } else {
        segment
    };
    KernelId::new(&format!("fused.{segment}"), 1).expect("sanitized segment forms a valid id")
}

/// Runs a migration Kernel once over a state Field: a map with that one
/// input, its parameters at their defaults, over the state's extent.
/// Returns the new state, or why the Kernel cannot be one.
fn migrate(
    runtime: &mut Runtime,
    version: &Arc<KernelVersion>,
    state: Handle,
    context: &Context,
) -> Result<Handle, String> {
    let manifest = version.manifest();
    if manifest.mode != orla_ir::Mode::Map || manifest.inputs.len() != 1 {
        return Err("must be a map over one input, the old state, giving the new".to_owned());
    }
    let domain = runtime
        .host()
        .domain(state)
        .map_err(|error| error.to_string())?;
    // Its input must take the state as it is: the layout it was
    // committed with, not the one the port wants now.
    let schema = runtime
        .host()
        .schema(state)
        .map_err(|error| error.to_string())?;
    let takes = &manifest.inputs[0].ty;
    if takes.shape != domain.shape() || takes.schema != *schema {
        return Err(format!(
            "takes {} laid out as {}, but the state is a {} laid out as {schema}",
            takes.shape,
            takes.schema,
            domain.shape()
        ));
    }
    let invocation = Invocation {
        inputs: vec![state],
        params: manifest.params.iter().map(|param| param.default).collect(),
        domain,
        context: *context,
    };
    let mut outputs = runtime
        .execute(version, &invocation)
        .map_err(|error| format!("failed: {error}"))?;
    let carried = outputs.remove(0);
    for extra in outputs {
        release(runtime, extra);
    }
    Ok(carried)
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
