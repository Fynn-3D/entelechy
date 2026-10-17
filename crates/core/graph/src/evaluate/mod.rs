//! Cached, dirty-tracked evaluation of a graph, with state committed
//! between evaluations and pointwise chains fused.
//!
//! A frame runs what something reads ([`Graph::read_through`]): the
//! outputs, the state, the pinned nodes and what they read; a trial runs
//! every node.
//!
//! A node's outputs are reused while its Kernel generation, parameters,
//! domain, context and upstream results are unchanged. A node that runs
//! again and gives outputs equal word for word to its last keeps its
//! last, so what reads them is reused too, and state at a fixed point does
//! not advance: Fields in Host memory are compared there, and Fields on a
//! device on it when the node ran for its state alone. `<~` bindings read
//! the state committed by the previous successful evaluation; an
//! evaluation commits atomically at its end, and a failed one commits
//! nothing.
//!
//! Outputs made once, a settled node's read only by settled nodes, are let
//! go cold once their readers hold what they made of them: their key and
//! epoch stay, a frame passes them by while the key is unchanged, and a
//! reader that runs has them made again first.
//!
//! Fusion inlines a producer into its one consumer when its single output
//! is read only through `sample`, over the same domain, and it is not a
//! graph output, state source or pinned node. The group executes as one
//! Kernel and one cache unit; pin a node to observe it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use entelechy_abi::{Capability, Context, Domain, FieldData, Handle, Invocation, Type, Value};
use entelechy_kernel::{
    ExecutionError, Generation, HostError, KernelId, KernelVersion, PrepareError, Run, Runtime,
    Ticket,
};

use crate::domain::VALUE_MAX_ELEMENTS;
use crate::{
    AsGraph, Binding, DomainExpr, DomainSpec, Graph, GraphError, Node, NodeId, ParamValue, Source,
};

mod bind;
mod error;
mod frames;
mod ledger;
mod plan;
mod recall;
mod state;

use bind::Bind;
pub use bind::{Bound, binding_fault, borrowed_domain, type_fault};
pub use error::EvaluateError;
use ledger::Room;
use plan::{graph_param, suggest};

/// Which nodes an evaluation runs.
#[derive(Clone, Copy)]
enum Demand<'a> {
    /// Every node: a trial, proving every node of a scene before it lands.
    Every,
    /// A frame: what something reads ([`Graph::read_through`]), with the
    /// nodes pinned to be read.
    Frame,
    /// These, what they read, state producers and outputs never produced.
    For(&'a BTreeSet<NodeId>),
    /// Only what these read, for a trial reporting no output.
    Read(&'a BTreeSet<NodeId>),
}

/// The report of an evaluation that reports one: every one but a
/// [`Demand::Read`].
fn reported(report: Option<Report>) -> Result<Report, EvaluateError> {
    report.ok_or(EvaluateError::Graph(GraphError::NoOutput))
}

/// The nodes an evaluation reused: those of its plan it reached, in order,
/// but the ones it ran or left running off the frame. The plan's names are
/// shared, so a frame that reuses most of a large graph lists them without
/// copying them.
#[derive(Clone)]
pub struct Reused {
    names: Arc<Vec<NodeId>>,
    /// Places in `names` not reused, in order.
    absent: Vec<u32>,
}

impl Reused {
    fn new(names: Arc<Vec<NodeId>>, absent: Vec<u32>) -> Self {
        Self { names, absent }
    }

    /// How many nodes were reused.
    pub fn len(&self) -> usize {
        self.names.len() - self.absent.len()
    }

    /// Whether none was.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The nodes reused, in evaluation order.
    pub fn iter(&self) -> impl Iterator<Item = &NodeId> + '_ {
        let mut absent = self.absent.iter().copied().peekable();
        self.names.iter().enumerate().filter_map(move |(at, id)| {
            if absent.peek() == Some(&(at as u32)) {
                absent.next();
                None
            } else {
                Some(id)
            }
        })
    }

    /// Whether `id` was reused.
    pub fn contains(&self, id: &NodeId) -> bool {
        self.iter().any(|node| node == id)
    }

    /// The nodes reused, as a list of their own.
    pub fn to_vec(&self) -> Vec<NodeId> {
        self.iter().cloned().collect()
    }
}

impl fmt::Debug for Reused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for Reused {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for Reused {}

impl PartialEq<Vec<NodeId>> for Reused {
    fn eq(&self, other: &Vec<NodeId>) -> bool {
        self.iter().eq(other.iter())
    }
}

/// The groups that ran as one Kernel, each the executing node and the nodes
/// it absorbed: the plan's, shared, read as a slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fusions(Arc<Vec<(NodeId, Vec<NodeId>)>>);

impl std::ops::Deref for Fusions {
    type Target = [(NodeId, Vec<NodeId>)];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq<Vec<(NodeId, Vec<NodeId>)>> for Fusions {
    fn eq(&self, other: &Vec<(NodeId, Vec<NodeId>)>) -> bool {
        *self.0 == *other
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
    pub reused: Reused,
    /// Sources whose state advanced in this evaluation's commit.
    pub committed: Vec<(NodeId, String)>,
    /// Every declared output by name.
    pub outputs: Vec<(String, Handle)>,
    /// Groups that ran as one Kernel: the executing node and the nodes it
    /// absorbed.
    pub fused: Fusions,
    /// Each executed node's Kernel time, in evaluation order: the execute
    /// call (a GPU submission), or the device clock under a measured
    /// profile.
    pub timings: Vec<(NodeId, std::time::Duration)>,
    /// Each executed node's runs in the clocked frame, in evaluation order:
    /// those the device was given between its marks
    /// ([`entelechy_kernel::Runtime::clock_mark`]), which the frame's clock
    /// times once the device has done them.
    pub clocked: Vec<(NodeId, std::ops::Range<usize>)>,
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
    /// Groups a backend refused to run fused, found since the last
    /// report: they run apart until one of their Kernels changes.
    pub refused: Vec<Refused>,
    /// Nodes whose cached outputs were let go to make room in the Host:
    /// a later frame that asks for one runs it again. Above zero frame
    /// after frame, the scene holds more than the Host's budget.
    pub let_go: usize,
}

/// A group a backend took Kernel by Kernel but refused fused: the node
/// it would have run as, the nodes in it, and the backend's reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    /// The node the group would have run as.
    pub node: NodeId,
    /// The group's nodes, the node it would have run as first.
    pub members: Vec<NodeId>,
    /// The backend's reason.
    pub reason: String,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let members = self
            .members
            .iter()
            .map(|node| format!("`{node}`"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "fusing {members} into `{}` was refused by the backend, so they run apart until one of their Kernels changes: {}",
            self.node, self.reason
        )
    }
}

/// Where one node ran; see [`Report::placements`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The node.
    pub node: NodeId,
    /// The backend that ran it.
    pub backend: &'static str,
    /// Why the main backend did not run it, if so.
    pub routed: Option<entelechy_kernel::Routed>,
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
    /// Present only for Kernels that read the clock: the time's whole
    /// bits with the rate's.
    time: Option<u64>,
    /// Present only for Kernels that read input.
    input: Option<u64>,
}

impl CacheKey {
    /// Whether this key differs from `last` in the state it reads alone:
    /// the node runs again for nothing but committed state.
    fn moved_by_state_alone(&self, last: &CacheKey) -> bool {
        self.generation == last.generation
            && self.params == last.params
            && self.domain == last.domain
            && self.time == last.time
            && self.input == last.input
            && self.inputs.len() == last.inputs.len()
            && self.inputs.iter().zip(&last.inputs).all(|(now, was)| {
                now == was || matches!((now, was), (InputKey::State(_), InputKey::State(_)))
            })
    }
}

#[derive(Clone, Debug)]
struct Cached {
    key: CacheKey,
    outputs: Vec<Handle>,
    output_names: Arc<Vec<String>>,
    epoch: u64,
}

/// What a fused version is made of: the nodes inlined into it, at their
/// generations, and the producers whose extent it reads from params.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Members {
    parts: Vec<(NodeId, Generation)>,
    counted: Vec<NodeId>,
}

/// A fused version kept while its constituents are unchanged.
struct Fused {
    members: Members,
    version: Arc<KernelVersion>,
}

/// A fused version the runtime is still preparing; until it arrives the
/// group's nodes execute separately.
struct Pending {
    members: Members,
    ticket: Ticket,
}

/// An `async` node's run off the frame, with the cache key it lands under.
struct Running {
    run: Run,
    key: CacheKey,
    output_names: Arc<Vec<String>>,
}

/// A node whose outputs are made once, and its readers.
type MadeOnce = (NodeId, Vec<NodeId>);

/// The nodes of a plan whose outputs are made once, each with its readers:
/// a settled node's (one that reads no clock, input, window or state,
/// runs in its turn, and reads only settled nodes), read only by settled
/// nodes.
fn made_once(groups: &[Group]) -> Vec<MadeOnce> {
    let at: BTreeMap<&NodeId, usize> = groups
        .iter()
        .enumerate()
        .map(|(at, group)| (&group.node.id, at))
        .collect();
    // Groups come producers first.
    let mut moving = vec![false; groups.len()];
    let mut readers: BTreeMap<&NodeId, Vec<usize>> = BTreeMap::new();
    for (index, group) in groups.iter().enumerate() {
        let requires = &group.manifest().requires;
        let mut moves = group.node.is_async
            || requires.contains(Capability::TimeRead)
            || requires.contains(Capability::InputRead)
            || matches!(&group.node.domain, Some(DomainSpec::Expr(expr)) if expr.reads_context());
        for binding in group.node.inputs.values() {
            match binding {
                Binding::Previous { .. } => moves = true,
                Binding::Now(Source::Node { node, .. }) => {
                    moves |= at.get(node).is_some_and(|producer| moving[*producer]);
                    readers.entry(node).or_default().push(index);
                }
                Binding::Now(Source::External(_)) => {}
            }
        }
        moving[index] = moves;
    }
    groups
        .iter()
        .enumerate()
        .filter(|(index, _)| !moving[*index])
        .filter_map(|(_, group)| {
            let readers = readers.get(&group.node.id)?;
            readers.iter().all(|reader| !moving[*reader]).then(|| {
                (
                    group.node.id.clone(),
                    readers
                        .iter()
                        .map(|reader| groups[*reader].node.id.clone())
                        .collect(),
                )
            })
        })
        .collect()
}

/// The smallest outputs let go cold once made ([`Evaluator::chill_made_once`]):
/// smaller ones cost more to make again than they hold.
const COLD_LEAST: usize = 1 << 20;

/// What a frame's runs gather, and where it is, for making room.
struct Frame {
    executed: Vec<NodeId>,
    timings: Vec<(NodeId, std::time::Duration)>,
    clocked: Vec<(NodeId, std::ops::Range<usize>)>,
    /// Every node run on the frame, in order, with its device status,
    /// checked before anything commits.
    checks: Vec<(NodeId, Vec<Box<dyn entelechy_kernel::Check>>)>,
    placements: Vec<Placed>,
    let_go: usize,
    /// Where each node runs and is read in the plan, what it holds, and
    /// which of its groups this frame runs.
    layout: Arc<Layout>,
    /// Held besides the layout's: what is asked for by name.
    asked: BTreeSet<NodeId>,
}

impl Frame {
    fn new(layout: Arc<Layout>, asked: BTreeSet<NodeId>) -> Self {
        Self {
            executed: Vec::new(),
            timings: Vec::new(),
            clocked: Vec::new(),
            checks: Vec::new(),
            placements: Vec::new(),
            let_go: 0,
            layout,
            asked,
        }
    }

    /// Whether `id`'s outputs are never let go on this frame.
    fn holds(&self, id: &NodeId) -> bool {
        self.layout.held.contains(id) || self.asked.contains(id)
    }

    /// Everything never let go on this frame.
    fn held(&self) -> BTreeSet<NodeId> {
        self.layout
            .held
            .iter()
            .chain(&self.asked)
            .cloned()
            .collect()
    }
}

/// What a plan's frames share, found once a plan: the group each node runs
/// in, where each node's outputs are last read and who reads each, what is
/// never let go (the scene's outputs, the state's sources and the pinned),
/// and, for the nodes a demand reaches, which groups run.
struct Layout {
    runs_at: BTreeMap<NodeId, usize>,
    last_read: BTreeMap<NodeId, usize>,
    readers: BTreeMap<NodeId, Vec<NodeId>>,
    held: BTreeSet<NodeId>,
    /// Whether each group runs; every group when `None`.
    runs: Option<Vec<bool>>,
    /// Who reads each node `<-` in the graph, fused or not, and the nodes
    /// a scene's output or a state reads ([`Evaluator::weigh_alike`]).
    graph_readers: BTreeMap<NodeId, Vec<NodeId>>,
    kept: BTreeSet<NodeId>,
    /// The nodes fused into others, each once, in order.
    absorbed: Vec<NodeId>,
    /// The nodes of the groups a frame runs, in order, and each group's
    /// place among them.
    names: Arc<Vec<NodeId>>,
    place: Vec<u32>,
    /// Each running group that fused others, with them.
    fused: Arc<Vec<(NodeId, Vec<NodeId>)>>,
    /// The groups that read each group `<-`, by place.
    readers_at: Vec<Vec<usize>>,
    /// Whether a group's key may change with nothing it reads `<-`
    /// changing: it reads the clock, input, state or a caller's Field, runs
    /// off the frame, takes its domain from an expression, or reads a node
    /// the plan does not run.
    moving: Vec<bool>,
}

impl Layout {
    fn new(
        graph: &Graph,
        groups: &[Group],
        pinned: &BTreeSet<NodeId>,
        needed: Option<&BTreeSet<NodeId>>,
    ) -> Self {
        let mut last_read: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut readers: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        let mut runs_at: BTreeMap<NodeId, usize> = BTreeMap::new();
        for (at, group) in groups.iter().enumerate() {
            runs_at.insert(group.node.id.clone(), at);
            for binding in group.node.inputs.values() {
                if let Binding::Now(Source::Node { node, .. }) = binding {
                    last_read.insert(node.clone(), at);
                    readers
                        .entry(node.clone())
                        .or_default()
                        .push(group.node.id.clone());
                }
            }
        }
        let runs = needed.map(|needed| {
            groups
                .iter()
                .map(|group| needed.contains(&group.node.id))
                .collect()
        });
        let mut graph_readers: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        let mut kept: BTreeSet<NodeId> = BTreeSet::new();
        for (_, source) in &graph.outputs {
            if let Source::Node { node, .. } = source {
                kept.insert(node.clone());
            }
        }
        for reader in &graph.nodes {
            for binding in reader.inputs.values() {
                match binding {
                    Binding::Now(Source::Node { node, .. }) => graph_readers
                        .entry(node.clone())
                        .or_default()
                        .push(reader.id.clone()),
                    Binding::Previous {
                        source: Source::Node { node, .. },
                        ..
                    } => {
                        kept.insert(node.clone());
                    }
                    _ => {}
                }
            }
        }
        let mut deps = Vec::with_capacity(groups.len());
        let mut moving = Vec::with_capacity(groups.len());
        for group in groups {
            let requires = &group.manifest().requires;
            let mut moves = group.node.is_async
                || requires.contains(Capability::TimeRead)
                || requires.contains(Capability::InputRead)
                || matches!(&group.node.domain, Some(DomainSpec::Expr(_)));
            let mut reads = Vec::new();
            for binding in group.node.inputs.values() {
                match binding {
                    Binding::Now(Source::Node { node, .. }) => match runs_at.get(node) {
                        Some(at) => reads.push(*at),
                        None => moves = true,
                    },
                    Binding::Now(Source::External(_)) | Binding::Previous { .. } => moves = true,
                }
            }
            deps.push(reads);
            moving.push(moves);
        }
        let absorbed: BTreeSet<&NodeId> = groups.iter().flat_map(|g| g.absorbed.iter()).collect();
        let running = |at: usize| runs.as_ref().is_none_or(|runs: &Vec<bool>| runs[at]);
        let mut names = Vec::new();
        let mut place = vec![u32::MAX; groups.len()];
        let mut fused = Vec::new();
        for (at, group) in groups.iter().enumerate() {
            if running(at) {
                place[at] = names.len() as u32;
                names.push(group.node.id.clone());
                if !group.absorbed.is_empty() {
                    fused.push((group.node.id.clone(), group.absorbed.clone()));
                }
            }
        }
        let mut readers_at = vec![Vec::new(); groups.len()];
        for (at, reads) in deps.iter().enumerate() {
            for read in reads {
                readers_at[*read].push(at);
            }
        }
        Self {
            runs_at,
            last_read,
            readers,
            held: held_nodes(graph, pinned),
            runs,
            graph_readers,
            kept,
            absorbed: absorbed.into_iter().cloned().collect(),
            names: Arc::new(names),
            place,
            fused: Arc::new(fused),
            readers_at,
            moving,
        }
    }

    /// Whether group `at` runs.
    fn runs(&self, at: usize) -> bool {
        self.runs.as_ref().is_none_or(|runs| runs[at])
    }

    /// Marks the groups reading group `at` to bind: its outputs changed.
    fn touch_readers(&self, at: usize, dirty: &mut [bool]) {
        for reader in &self.readers_at[at] {
            dirty[*reader] = true;
        }
    }
}

impl Evaluator {
    /// The epoch of `id`'s outputs, cached or let go cold.
    fn output_epoch(&self, id: &NodeId) -> Option<u64> {
        self.ledger
            .entry(id)
            .map(|cached| cached.epoch)
            .or_else(|| self.ledger.cold(id).map(|cold| cold.epoch))
    }
}

/// What a frame of the graph left that the next may lean on: its plan's
/// layout, the ledger as the frame left it, the parameters and Fields it was
/// given, and which groups' entries are as their keys were found. A group
/// that does not move ([`Layout::moving`]) and none of whose reads changed
/// is passed by unbound while all of it holds.
struct Quiet {
    layout: Arc<Layout>,
    ledger: u64,
    params: BTreeMap<String, Value>,
    externals: BTreeMap<String, Handle>,
    steady: Vec<bool>,
}

/// A plan's layout, with the plan and the reach it was found for.
struct LayoutFor {
    groups: Arc<Vec<Group>>,
    needed: Option<Arc<BTreeSet<NodeId>>>,
    layout: Arc<Layout>,
}

impl Evaluator {
    /// The layout of plan `groups` for the nodes `needed` reaches, kept
    /// while both are the ones it was found for.
    fn layout_of(
        &mut self,
        graph: &Graph,
        groups: &Arc<Vec<Group>>,
        needed: Option<&Arc<BTreeSet<NodeId>>>,
    ) -> Arc<Layout> {
        if let Some(known) = &self.layout
            && Arc::ptr_eq(&known.groups, groups)
            && match (&known.needed, needed) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
        {
            return Arc::clone(&known.layout);
        }
        let layout = Arc::new(Layout::new(
            graph,
            groups,
            &self.pinned,
            needed.map(|needed| needed.as_ref()),
        ));
        self.layout = Some(LayoutFor {
            groups: Arc::clone(groups),
            needed: needed.cloned(),
            layout: Arc::clone(&layout),
        });
        layout
    }
}

/// What a frame evaluates: the graph, its plan, and what it is given.
struct Scene<'a> {
    graph: &'a Graph,
    groups: &'a [Group],
    externals: &'a BTreeMap<String, Handle>,
    context: &'a Context,
    params: &'a BTreeMap<String, Value>,
}

/// The names of what a group's Kernel outputs.
fn output_names(group: &Group) -> Arc<Vec<String>> {
    Arc::clone(group.names.get_or_init(|| {
        Arc::new(
            group
                .version
                .manifest()
                .outputs
                .iter()
                .map(|port| port.name.clone())
                .collect(),
        )
    }))
}

/// One executing unit: a node, a node with producers inlined, or a run
/// of the same node placed several times over its own output.
#[derive(Clone)]
struct Group {
    node: Node,
    version: Arc<KernelVersion>,
    parts: Vec<(NodeId, Generation)>,
    absorbed: Vec<NodeId>,
    /// The producers no longer stored for it, whose count or size it reads
    /// from params instead.
    counted: Vec<NodeId>,
    /// How many times the Kernel runs, each feeding `chain` from the one
    /// before: 1, or N for `times = N` over the node's own output.
    repeat: u32,
    /// The `carry port <- output` pair a repeated run feeds itself by.
    chain: Option<(usize, usize)>,
    /// While the plan is made, the fused module of the producers inlined
    /// so far; prepared once the group is known to execute.
    fused: Option<Arc<entelechy_ir::Validated>>,
    /// Its outputs' names, found when first asked, once the plan is made.
    names: std::sync::OnceLock<Arc<Vec<String>>>,
}

impl Group {
    /// The manifest of what the group computes: its fused module's, else
    /// its version's.
    fn manifest(&self) -> &entelechy_kernel::Manifest {
        self.fused
            .as_ref()
            .map_or_else(|| self.version.manifest(), |module| &module.manifest)
    }

    /// The IR of what the group computes, when it has one.
    fn module(&self) -> Option<&entelechy_ir::Validated> {
        self.fused.as_deref().or_else(|| self.version.module())
    }

    /// What its fused version is made of.
    fn members(&self) -> Members {
        Members {
            parts: self.parts.clone(),
            counted: self.counted.clone(),
        }
    }
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
    /// Producers fused elsewhere whose count or size it reads from the
    /// extent the plan knows, as they are not stored for it.
    pub counted: Vec<NodeId>,
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
    declined: BTreeMap<NodeId, Members>,
    /// Groups a backend refused fused, kept until a report carries them.
    refused: Vec<Refused>,
    pending: BTreeMap<NodeId, Pending>,
    /// The last graph planned and its order. A plan is kept for this
    /// graph only: a changed graph forgets it.
    shape: Option<(Arc<Graph>, Arc<Vec<usize>>)>,
    /// The last plan and what it was made from, besides the graph.
    last_plan: Option<(plan::PlanKey, Arc<Vec<Group>>)>,
    /// The last plan and its inputs besides the graph ([`plan::PlanInputs`]).
    planned_from: Option<(plan::PlanInputs, Arc<Vec<Group>>)>,
    /// The ports holding state in the graph in `shape`, with the nodes and
    /// ports they hold.
    state_ports: Option<Vec<((NodeId, String), String)>>,
    /// The ports holding state in the graph in `shape`, in binding order.
    state_keys: Option<Vec<(NodeId, String)>>,
    /// The state reads of the last plan reconciled.
    state_reads: Option<(Arc<Vec<Group>>, Arc<state::StateReads>)>,
    /// What a frame of the graph in `shape` reads, and the pins it was
    /// found under.
    frame_needed: Option<(BTreeSet<NodeId>, Arc<BTreeSet<NodeId>>)>,
    /// The layout of the last plan run, with the plan and the reach it
    /// was found for.
    layout: Option<LayoutFor>,
    /// What the last frame left for the next to lean on.
    quiet: Option<Quiet>,
    /// Whether what left the graph in `shape` is let go: nodes leave only
    /// with a changed graph, and a pin, a restore or a reset may leave
    /// something to let go.
    pruned: bool,
    running: BTreeMap<NodeId, Running>,
    /// State an evaluation replaced before its nodes ran (a migration, a
    /// restart), held until the commit or put back on failure.
    replaced: Vec<Replaced>,
    /// Whether the running evaluation is a restore's trial, which commits
    /// nothing ([`Evaluator::restore`]).
    trial: bool,
    /// Nodes running for their state alone whose Fields on a device were
    /// found changed; see [`Rest`].
    rest: crate::hash::QuickMap<NodeId, Rest>,
    /// Runs remembered frame by frame; see [`recall`].
    recalled: recall::Recalled,
    /// Whether a frame commits without waiting for its device; see
    /// [`frames`].
    overlap: bool,
    /// The frame committed and not yet confirmed.
    unconfirmed: Option<frames::Unconfirmed>,
    /// Frames answered as committed and taken back since last asked; see
    /// [`Evaluator::take_taken_back`].
    taken_back: u64,
    /// Small nodes this evaluation found unchanged; see
    /// [`Evaluator::weigh_alike`].
    alike: Vec<NodeId>,
    /// The plan last asked which of its nodes' outputs are made once, and
    /// those nodes with their readers ([`made_once`]).
    made_once: Option<(Arc<Vec<Group>>, Vec<MadeOnce>)>,
}

/// When a node that runs for its state alone, or whose outputs are small
/// ([`SMALL_COMPARED`]), has its Fields on a device compared with its
/// last: comparing waits for the device, so each time they are found
/// changed, or found unchanged while sparing nothing
/// ([`Evaluator::weigh_alike`]), the next comparison waits twice as many
/// such runs, up to 2^[`REST_WAIT_MOST`]. State that settles is found at
/// rest at once; state that keeps changing waits for the device seldom.
#[derive(Debug, Default)]
struct Rest {
    since: u32,
    changed: u32,
}

const REST_WAIT_MOST: u32 = 10;

/// The most bytes of a node's outputs compared on the device when fresh
/// inputs moved it: a choice or a count others are made from, which costs
/// one short wait to find unchanged and spares all that reads it.
const SMALL_COMPARED: usize = 4096;

/// The bytes `outputs` hold; a handle the Host does not know counts as
/// past any bound.
fn outputs_bytes(runtime: &Runtime, outputs: &[Handle]) -> usize {
    let host = runtime.host();
    outputs.iter().fold(0usize, |sum, handle| {
        match (host.domain(*handle), host.schema(*handle)) {
            (Ok(domain), Ok(schema)) => {
                sum.saturating_add(schema.storage_words(domain.count()) * 4)
            }
            _ => usize::MAX,
        }
    })
}

impl Rest {
    /// Counts a run for state alone; whether this one is compared.
    fn due(&mut self) -> bool {
        self.since += 1;
        self.since >= 1 << self.changed.min(REST_WAIT_MOST)
    }

    /// A comparison found the Fields changed.
    fn found_changed(&mut self) {
        self.changed += 1;
        self.since = 0;
    }
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
            recalled: recall::Recalled::new(),
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
        self.pruned = false;
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
                *consumer != node && fused.members.parts.iter().any(|(part, _)| part == node)
            })
            .map(|(consumer, _)| consumer)
    }

    /// Lets fusion have `node` back.
    pub fn unpin(&mut self, node: &NodeId) {
        self.pruned = false;
        self.pinned.remove(node);
    }

    /// Releases every cached image and all committed state.
    pub fn clear(&mut self, runtime: &mut Runtime) {
        self.pruned = false;
        // What only remembered runs hold goes first; the rest below.
        self.forget_runs(runtime);
        let mut released = BTreeSet::new();
        self.drop_in_flight(&mut released, runtime);
        let (cache, committed) = self.ledger.take_all();
        self.fused.clear();
        for (_, pending) in std::mem::take(&mut self.pending) {
            runtime.discard(pending.ticket);
        }
        for (_, running) in std::mem::take(&mut self.running) {
            runtime.discard_run(running.run);
        }
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
        self.pruned = false;
        self.recover(runtime);
        // A frame in flight that failed leaves the state it began from,
        // which is forgotten here all the same.
        let _ = self.confirm(runtime);
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
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<Vec<PlanEntry>, EvaluateError> {
        let (graph, shared) = (graph.graph(), graph.shared());
        let order = self.order_of(graph, shared)?;
        let groups = self.plan(
            graph,
            &order,
            runtime,
            externals,
            overrides,
            &Context::default(),
        )?;
        Ok(groups
            .iter()
            .map(|group| PlanEntry {
                node: group.node.id.clone(),
                kernel: group.version.id().clone(),
                absorbed: group.absorbed.clone(),
                counted: group.counted.clone(),
                inputs: group.node.inputs.clone(),
                domain: group.node.domain.as_ref().and_then(DomainSpec::literal),
            })
            .collect())
    }

    /// Evaluates `graph` under `context`, executing only nodes whose inputs
    /// changed, then commits state. `params` overrides graph parameters by
    /// name; unlisted ones use their defaults.
    pub fn evaluate(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        if !self.overlap {
            self.confirm(runtime)?;
        }
        self.evaluate_with(graph, runtime, externals, context, params, Demand::Frame)
            .and_then(reported)
    }

    /// Evaluates as [`Evaluator::evaluate`] does, every node of the graph,
    /// and commits nothing: `<~`
    /// readers keep the state last committed, and the results stay cached
    /// for reading; runs off the frame are not started.
    pub fn evaluate_trial(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        self.confirm(runtime)?;
        self.trial = true;
        let tried = self
            .evaluate_with(graph, runtime, externals, context, params, Demand::Every)
            .and_then(reported);
        self.trial = false;
        tried
    }

    /// A trial of only what `readers` read: the nodes they read through
    /// `<-`, and the initial sources of their `<~` reads, not the readers
    /// themselves. Nothing is committed and no output is reported; the
    /// results stay cached for reading.
    pub fn evaluate_trial_reading(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        readers: &BTreeSet<NodeId>,
    ) -> Result<(), EvaluateError> {
        self.recover(runtime);
        self.confirm(runtime)?;
        self.trial = true;
        let tried = self
            .evaluate_with(
                graph,
                runtime,
                externals,
                context,
                params,
                Demand::Read(readers),
            )
            .map(|_| ());
        self.trial = false;
        tried
    }

    /// Evaluates only what `demand` needs: the demanded nodes, what they
    /// read through `<-`, every `<~` source, and any output never
    /// produced. Everything else keeps its last value.
    pub fn evaluate_for(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: &BTreeSet<NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.recover(runtime);
        self.confirm(runtime)?;
        self.evaluate_with(
            graph,
            runtime,
            externals,
            context,
            params,
            Demand::For(demand),
        )
        .and_then(reported)
    }

    /// Whether `node`'s outputs were let go cold: its key and epoch kept,
    /// its Fields made again when read ([`Evaluator::warm_for_reading`]).
    pub fn is_cold(&self, node: &NodeId) -> bool {
        self.ledger.cold(node).is_some()
    }

    /// The nodes whose outputs were let go cold.
    pub fn cold_nodes(&self) -> Vec<NodeId> {
        self.ledger.cold_nodes().cloned().collect()
    }

    /// Makes `node`'s outputs again when they were let go cold, for
    /// reading: a trial of it and what it reads, which commits nothing.
    pub fn warm_for_reading(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        node: &NodeId,
    ) -> Result<(), EvaluateError> {
        if !self.is_cold(node) {
            return Ok(());
        }
        self.recover(runtime);
        self.confirm(runtime)?;
        self.trial = true;
        let demand = BTreeSet::from([node.clone()]);
        let warmed = self
            .evaluate_with(
                graph,
                runtime,
                externals,
                context,
                params,
                Demand::For(&demand),
            )
            .map(|_| ());
        self.trial = false;
        warmed
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
        match self.bind(graph, node, version, runtime, externals, context, params)? {
            Bind::Ready(invocation, _) => Ok(invocation),
            Bind::Keyed(_, cold) | Bind::Unkeyed(cold) => Err(EvaluateError::UnknownOutput {
                node: cold[0].clone(),
                port: "result".to_owned(),
                near: None,
            }),
        }
    }

    /// The nodes an evaluation under `demand` must run: see
    /// [`Evaluator::evaluate_for`] and [`Evaluator::evaluate_trial_reading`];
    /// `None` every node.
    fn needed(&mut self, graph: &Graph, demand: Demand<'_>) -> Option<Arc<BTreeSet<NodeId>>> {
        let demand = match demand {
            Demand::Every => return None,
            // A frame reads what the graph's outputs, state and pins reach:
            // found once for the graph in `shape` and its pins.
            Demand::Frame => {
                if let Some((pins, read)) = &self.frame_needed
                    && *pins == self.pinned
                {
                    return Some(Arc::clone(read));
                }
                let read = Arc::new(graph.read_through(self.pinned.iter().cloned()));
                self.frame_needed = Some((self.pinned.clone(), Arc::clone(&read)));
                return Some(read);
            }
            Demand::For(demand) => demand,
            Demand::Read(readers) => {
                let at = graph.index();
                let mut needed = BTreeSet::new();
                let mut frontier: Vec<NodeId> = readers.iter().cloned().collect();
                while let Some(id) = frontier.pop() {
                    let Some(node) = at.get(&id).map(|at| &graph.nodes[*at]) else {
                        continue;
                    };
                    for binding in node.inputs.values() {
                        let read = match binding {
                            Binding::Now(source) => source,
                            Binding::Previous { initial, .. } => initial,
                        };
                        if let Source::Node { node: producer, .. } = read
                            && needed.insert(producer.clone())
                        {
                            frontier.push(producer.clone());
                        }
                    }
                }
                return Some(Arc::new(needed));
            }
        };
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
        Some(Arc::new(needed))
    }

    /// Evaluates what `demand` asks; the report is `None` only for a
    /// [`Demand::Read`].
    fn evaluate_with(
        &mut self,
        graph: &dyn AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: Demand<'_>,
    ) -> Result<Option<Report>, EvaluateError> {
        // An external the caller supplies nothing for reads its fallback,
        // written into a copy of the graph only when one is in use.
        let substituted = substitute_fallbacks(graph.graph(), externals);
        // A graph with fallbacks written in is a new one, not the caller's.
        let shared = if substituted.is_some() {
            None
        } else {
            graph.shared()
        };
        let graph = substituted.as_ref().unwrap_or(graph.graph());
        // What the parameters must hold, before anything runs.
        for requirement in &graph.requirements {
            requirement
                .check(
                    &|name| graph_param(graph, name, params),
                    &window_context(context),
                )
                .map_err(|error| EvaluateError::Requirement(error.message))?;
        }
        let order = self.order_of(graph, shared)?;
        self.recall_for(graph, runtime);
        if !self.pruned {
            self.prune(graph, runtime);
            self.pruned = true;
        }
        let landed = self.land(runtime)?;
        let needed = self.needed(graph, demand);
        let groups = self.plan(graph, &order, runtime, externals, params, context)?;
        let (migrated, reset) = self.reconcile_state(&groups, runtime, context)?;
        // From here to the commit, a failure leaves state as it was: a
        // migration's replaced state is put back.
        let evaluated = (|| -> Result<Option<Report>, EvaluateError> {
            let layout = self.layout_of(graph, &groups, needed.as_ref());
            // Absorbed nodes release their cached outputs.
            let stale: Vec<NodeId> = layout
                .absorbed
                .iter()
                .filter(|id| self.ledger.entry(id).is_some())
                .cloned()
                .collect();
            for id in stale {
                self.drop_cache(&id, runtime);
            }
            // Where in the plan's names a group ran or was left running:
            // every other group it reached was reused.
            let mut absent: Vec<u32> = Vec::new();
            runtime.take_checks();
            runtime.begin_evaluation();
            let mut pending = Vec::new();
            // What is asked for by name is made, though let go cold.
            let asked = match demand {
                Demand::For(demand) => demand.clone(),
                _ => BTreeSet::new(),
            };
            // A frame leans on the last when nothing it was made from moved
            // but the clock and input: the same plan, ledger, parameters and
            // Fields. Then a group that does not move, none of whose reads
            // changed, keeps its entry unbound.
            let leans = matches!(demand, Demand::Frame) && !self.trial && !self.recalling();
            let last = if leans { self.quiet.take() } else { None };
            let ledger_now = self.ledger.version();
            let mut lean = last.as_ref().is_some_and(|last| {
                Arc::ptr_eq(&last.layout, &layout)
                    && last.ledger == ledger_now
                    && last.params == *params
                    && last.externals == *externals
            });
            let mut steady = match last {
                Some(last) if lean => last.steady,
                _ => vec![false; groups.len()],
            };
            // A group binds when it moves, was not left as its key was
            // found, or reads a group whose outputs changed on this frame.
            let mut dirty: Vec<bool> = if lean {
                (0..groups.len())
                    .map(|at| layout.moving[at] || !steady[at])
                    .collect()
            } else {
                vec![true; groups.len()]
            };
            let mut frame = Frame::new(layout, asked);
            let scene = Scene {
                graph,
                groups: &groups,
                externals,
                context,
                params,
            };
            for (at, group) in groups.iter().enumerate() {
                let node = &group.node;
                if !frame.layout.runs(at) {
                    continue;
                }
                if lean && !dirty[at] {
                    continue;
                }
                let before = self.output_epoch(&node.id);
                let mut bound = self.bind(
                    graph,
                    node,
                    &group.version,
                    runtime,
                    externals,
                    context,
                    params,
                )?;
                // A domain read from outputs let go cold waits for them.
                while let Bind::Unkeyed(cold) = &bound {
                    // What is made again may take a new epoch: the rest of
                    // the frame binds every group.
                    lean = false;
                    for producer in cold.clone() {
                        self.warm(&producer, at, &scene, &mut frame, runtime)?;
                    }
                    bound = self.bind(
                        graph,
                        node,
                        &group.version,
                        runtime,
                        externals,
                        context,
                        params,
                    )?;
                }
                let key = match &bound {
                    Bind::Ready(_, key) | Bind::Keyed(key, _) => key,
                    Bind::Unkeyed(_) => unreachable!("waited for above"),
                };
                if self
                    .ledger
                    .entry(&node.id)
                    .is_some_and(|cached| cached.key == *key)
                {
                    steady[at] = true;
                    continue;
                }
                // Let go cold under the same key: passed by unrun, unless
                // held, which is made again.
                let cold_epoch = self
                    .ledger
                    .cold(&node.id)
                    .filter(|cold| cold.key == *key)
                    .map(|cold| cold.epoch);
                if cold_epoch.is_some() && !frame.holds(&node.id) {
                    steady[at] = true;
                    continue;
                }
                // What this frame of the remembered run left under the
                // same key: its outputs, or the epoch to run again under.
                let rerun = match (!node.is_async)
                    .then(|| self.recall(&node.id, key, runtime))
                    .flatten()
                {
                    Some(recall::Offer::Taken) => {
                        frame.layout.touch_readers(at, &mut dirty);
                        continue;
                    }
                    Some(recall::Offer::Rerun(epoch)) => Some(epoch),
                    None => cold_epoch,
                };
                // What it reads let go cold is made again before it runs.
                let (invocation, key) = match bound {
                    Bind::Ready(invocation, key) => (invocation, key),
                    _ => {
                        lean = false;
                        self.ready(at, at, &scene, &mut frame, runtime)?
                    }
                };
                // A restore's trial runs every node on the frame, leaving
                // no run in flight.
                if node.is_async && !self.trial && self.ledger.entry(&node.id).is_some() {
                    // One run at a time; the frame carries the last output.
                    if !self.running.contains_key(&node.id) {
                        let output_names = output_names(group);
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
                    frame.layout.touch_readers(at, &mut dirty);
                    absent.push(frame.layout.place[at]);
                    pending.push(node.id.clone());
                    continue;
                }
                self.execute_group(
                    at,
                    at,
                    invocation,
                    key,
                    rerun,
                    &[],
                    &scene,
                    &mut frame,
                    runtime,
                )?;
                // Outputs equal to the last, word for word, keep its epoch:
                // what reads them finds nothing changed.
                if self.output_epoch(&node.id) != before {
                    frame.layout.touch_readers(at, &mut dirty);
                }
                absent.push(frame.layout.place[at]);
                steady[at] = true;
            }
            let Frame {
                executed,
                timings,
                clocked,
                checks,
                placements,
                let_go,
                layout,
                asked,
                ..
            } = frame;
            let reused = Reused::new(Arc::clone(&layout.names), absent);
            let fused = Fusions(Arc::clone(&layout.fused));
            let held: std::borrow::Cow<'_, BTreeSet<NodeId>> = if asked.is_empty() {
                std::borrow::Cow::Borrowed(&layout.held)
            } else {
                std::borrow::Cow::Owned(layout.held.iter().chain(&asked).cloned().collect())
            };
            self.weigh_alike(&layout, &executed);
            if matches!(demand, Demand::Read(_)) {
                self.wait_for_checks(checks, runtime)?;
                return Ok(None);
            }
            let mut outputs = Vec::with_capacity(graph.outputs.len());
            for (name, source) in &graph.outputs {
                let handle = match source {
                    Source::External(external) => {
                        *externals
                            .get(external)
                            .ok_or_else(|| EvaluateError::MissingExternal {
                                node: NodeId(format!("output {name}").into()),
                                name: external.clone(),
                            })?
                    }
                    Source::Node { node, port } => {
                        self.cached(node, port).ok_or_else(|| {
                            // Its real outputs, for the suggestion.
                            let near = self.ledger.entry(node).and_then(|cached| {
                                entelechy_abi::names::nearest(
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
            // A trial commits nothing, and is not remembered. Overlapping,
            // the frame commits now and its device is waited for once the
            // next frame's work has gone to it.
            let committed = if self.trial {
                self.wait_for_checks(checks, runtime)?;
                Vec::new()
            } else if self.overlap && matches!(demand, Demand::Every | Demand::Frame) {
                let committed = self.commit_in_flight(graph, runtime, checks, executed.clone())?;
                self.remember_frame(executed.iter().chain(reused.iter()), runtime);
                self.chill_made_once(&groups, &held, runtime);
                committed
            } else {
                self.wait_for_checks(checks, runtime)?;
                let committed = self.commit(graph, runtime)?;
                self.remember_frame(executed.iter().chain(reused.iter()), runtime);
                self.chill_made_once(&groups, &held, runtime);
                committed
            };
            if leans {
                self.quiet = Some(Quiet {
                    layout: Arc::clone(&layout),
                    ledger: self.ledger.version(),
                    params: params.clone(),
                    externals: externals.clone(),
                    steady,
                });
            }
            Ok(Some(Report {
                output,
                executed,
                timings,
                clocked,
                placements,
                reused,
                committed,
                outputs,
                fused,
                migrated,
                reset,
                pending,
                landed,
                refused: std::mem::take(&mut self.refused),
                let_go,
            }))
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

    /// Binds the group at `at` ready to run: each producer it reads let go
    /// cold is made again first ([`Evaluator::warm`]), for a reader whose
    /// turn is `turn`.
    fn ready(
        &mut self,
        at: usize,
        turn: usize,
        scene: &Scene<'_>,
        frame: &mut Frame,
        runtime: &mut Runtime,
    ) -> Result<(Invocation, CacheKey), EvaluateError> {
        let group = &scene.groups[at];
        // Each round makes at least one producer again; a producer let go
        // again at once, for want of room, ends it.
        for _ in 0..=group.node.inputs.len() {
            match self.bind(
                scene.graph,
                &group.node,
                &group.version,
                runtime,
                scene.externals,
                scene.context,
                scene.params,
            )? {
                Bind::Ready(invocation, key) => return Ok((invocation, key)),
                Bind::Keyed(_, cold) | Bind::Unkeyed(cold) => {
                    for producer in cold {
                        self.warm(&producer, turn, scene, frame, runtime)?;
                    }
                }
            }
        }
        Err(EvaluateError::NoRoomToMakeAgain(group.node.id.clone()))
    }

    /// Makes again a producer let go cold, for a reader that runs at
    /// `turn`: its own producers let go cold first. Made under the key it
    /// was let go under, on a backend that repeats exactly, it keeps its
    /// epoch, and what else reads it stays as it was.
    fn warm(
        &mut self,
        id: &NodeId,
        turn: usize,
        scene: &Scene<'_>,
        frame: &mut Frame,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        if self.ledger.entry(id).is_some() {
            return Ok(());
        }
        let missing = || EvaluateError::UnknownOutput {
            node: id.clone(),
            port: "result".to_owned(),
            near: None,
        };
        let cold = self.ledger.cold(id).cloned().ok_or_else(missing)?;
        let at = *frame.layout.runs_at.get(id).ok_or_else(missing)?;
        let (invocation, key) = self.ready(at, turn, scene, frame, runtime)?;
        let rerun = (cold.key == key).then_some(cold.epoch);
        // What it reads stays while it runs.
        let reads: Vec<NodeId> = scene.groups[at]
            .node
            .inputs
            .values()
            .filter_map(|binding| match binding {
                Binding::Now(Source::Node { node, .. }) => Some(node.clone()),
                _ => None,
            })
            .collect();
        self.execute_group(
            at, turn, invocation, key, rerun, &reads, scene, frame, runtime,
        )
    }

    /// Runs the group at `at` with `invocation`, on the frame at `turn`
    /// (its own place, or the reader's it is made again for), and caches
    /// what it gives: under `rerun`'s epoch when made again under the key
    /// that epoch was made under, on a backend that repeats exactly.
    /// Short of room in the Host, remembered runs and the outputs the
    /// frame is done with are let go and it runs again, `keep` staying.
    #[allow(clippy::too_many_arguments)]
    fn execute_group(
        &mut self,
        at: usize,
        turn: usize,
        invocation: Invocation,
        key: CacheKey,
        rerun: Option<u64>,
        keep: &[NodeId],
        scene: &Scene<'_>,
        frame: &mut Frame,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        let group = &scene.groups[at];
        let node = &group.node;
        // A first run, or one with nothing cached, is on the frame and
        // drops any run in flight.
        if let Some(running) = self.running.remove(&node.id) {
            runtime.discard_run(running.run);
        }
        runtime.take_moved();
        let started = entelechy_kernel::clock::now();
        let first_run = runtime.clock_mark();
        let run = |runtime: &mut Runtime| match group.chain {
            Some(chain) => {
                runtime.execute_repeated(&group.version, &invocation, group.repeat, chain)
            }
            None => runtime.execute(&group.version, &invocation),
        };
        // A run the Host has no room for runs again once remembered runs
        // and the outputs this frame has done with are let go; they are
        // made again when a later frame asks.
        let outputs = match run(runtime) {
            Err(error) if short_of_room(&error) => {
                let mut held = frame.held();
                held.extend(keep.iter().cloned());
                let made = self.forget_runs_for_room(runtime)
                    + self.make_room_counted(
                        &mut frame.let_go,
                        Room {
                            at: turn,
                            running: &node.id,
                            runs_at: &frame.layout.runs_at,
                            last_read: &frame.layout.last_read,
                            readers: &frame.layout.readers,
                            held: &held,
                        },
                        frame.executed.iter(),
                        runtime,
                    );
                if made > 0 { run(runtime) } else { Err(error) }
            }
            ran => ran,
        }
        .map_err(|error| EvaluateError::Execution {
            node: node.id.clone(),
            error,
        })?;
        // What the device says it took, when it measures; the call's own
        // time otherwise.
        let took = runtime
            .last_measured()
            .unwrap_or_else(|| entelechy_kernel::clock::since(started));
        frame.timings.push((node.id.clone(), took));
        frame
            .clocked
            .push((node.id.clone(), first_run..runtime.clock_mark()));
        frame.checks.push((node.id.clone(), runtime.take_checks()));
        // Where it ran, which a run's size may decide.
        let placement = runtime
            .last_placement()
            .cloned()
            .unwrap_or_else(|| runtime.placement(&group.version));
        let moved = runtime.take_moved();
        frame.placements.push(Placed {
            node: node.id.clone(),
            backend: placement.backend,
            routed: placement.routed,
            uploaded: moved.uploaded,
            downloaded: moved.downloaded,
        });
        // A run made again under a remembered key is the run remembered
        // only on a backend that repeats exactly; else its readers find a
        // new epoch and run.
        let output_names = output_names(group);
        match rerun.filter(|_| runtime.last_repeats_exactly()) {
            Some(epoch) => self.store_as(&node.id, key, outputs, output_names, epoch, runtime),
            None => self.store(&node.id, key, outputs, output_names, runtime),
        }
        frame.executed.push(node.id.clone());
        Ok(())
    }

    /// Lets go cold, once the frame is done, the outputs made once
    /// ([`made_once`]) whose readers all hold what they made of them,
    /// larger than [`COLD_LEAST`]. Nothing reads them again until
    /// something upstream, or a reader's Kernel or parameters, change; then
    /// they are made again ([`Evaluator::warm`]). What `held` names stays.
    fn chill_made_once(
        &mut self,
        groups: &Arc<Vec<Group>>,
        held: &BTreeSet<NodeId>,
        runtime: &mut Runtime,
    ) {
        // Which they are is the plan's, found once a plan.
        if !self
            .made_once
            .as_ref()
            .is_some_and(|(plan, _)| Arc::ptr_eq(plan, groups))
        {
            self.made_once = Some((Arc::clone(groups), made_once(groups)));
        }
        let Some((_, made)) = &self.made_once else {
            return;
        };
        let has = |id: &NodeId| self.ledger.entry(id).is_some() || self.ledger.cold(id).is_some();
        let chilled: Vec<NodeId> =
            made.iter()
                .filter(|(id, readers)| {
                    !held.contains(id)
                        && self.ledger.entry(id).is_some_and(|cached| {
                            outputs_bytes(runtime, &cached.outputs) >= COLD_LEAST
                        })
                        && readers.iter().all(has)
                })
                .map(|(id, _)| id.clone())
                .collect();
        for id in chilled {
            self.chill(&id, runtime);
        }
    }

    /// Caches a node's fresh outputs and lets go of the ones they replace,
    /// unless they are committed state. Fresh outputs equal word for word
    /// to the ones they replace give way to those, which keep their epoch:
    /// what reads them is not run again.
    fn store(
        &mut self,
        id: &NodeId,
        key: CacheKey,
        outputs: Vec<Handle>,
        output_names: Arc<Vec<String>>,
        runtime: &mut Runtime,
    ) {
        let by_state = self
            .ledger
            .entry(id)
            .is_some_and(|previous| key.moved_by_state_alone(&previous.key));
        // Small outputs are compared on the device whatever moved the
        // node: what reads them, however costly, runs again only when
        // they changed, wherever the node was placed.
        let compared = by_state || outputs_bytes(runtime, &outputs) <= SMALL_COMPARED;
        let on_device = compared && self.rest.entry(id.clone()).or_default().due();
        if !compared {
            self.rest.remove(id);
        }
        // Equal to what this frame of the remembered run left: those
        // outputs, and their epoch, so what reads them finds its key.
        if let Some((kept, epoch)) = self.recalled_alike(id, &output_names, &outputs, runtime) {
            for output in &outputs {
                if !kept.contains(output)
                    && !self.ledger.is_cached(*output)
                    && !self.is_kept(*output)
                {
                    release(runtime, *output);
                }
            }
            let fresh = Cached {
                key,
                outputs: kept,
                output_names,
                epoch,
            };
            if let Some(previous) = self.ledger.cache_put(id, fresh) {
                for output in previous.outputs {
                    if !self.ledger.is_cached(output) && !self.is_kept(output) {
                        release(runtime, output);
                    }
                }
            }
            return;
        }
        if let Some(previous) = self.ledger.entry(id)
            && previous.output_names == output_names
            && same_words(runtime, &previous.outputs, &outputs, on_device)
        {
            if on_device && !by_state {
                self.alike.push(id.clone());
            } else {
                self.rest.remove(id);
            }
            let kept = previous.outputs.clone();
            let epoch = previous.epoch;
            for output in &outputs {
                if !kept.contains(output)
                    && !self.ledger.is_cached(*output)
                    && !self.is_kept(*output)
                {
                    release(runtime, *output);
                }
            }
            self.ledger.cache_put(
                id,
                Cached {
                    key,
                    outputs: kept,
                    output_names,
                    epoch,
                },
            );
            return;
        }
        if on_device && let Some(rest) = self.rest.get_mut(id) {
            rest.found_changed();
        }
        self.epoch += 1;
        let fresh = Cached {
            key,
            outputs,
            output_names,
            epoch: self.epoch,
        };
        if let Some(previous) = self.ledger.cache_put(id, fresh) {
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

    /// Weighs each small node found unchanged this evaluation: it spared
    /// what reads it only if a reader did not run. Where every reader ran
    /// all the same, for what else it reads, the comparison waited for the
    /// device for nothing and rests as one that found a change does
    /// ([`Rest`]). A node read as state, or as a graph's output, is kept
    /// compared: what it spares shows after this evaluation.
    fn weigh_alike(&mut self, layout: &Layout, executed: &[NodeId]) {
        if self.alike.is_empty() {
            return;
        }
        let ran: std::collections::HashSet<&NodeId> = executed.iter().collect();
        for id in std::mem::take(&mut self.alike) {
            let readers = layout.graph_readers.get(&id).map_or(&[][..], Vec::as_slice);
            let held = layout.kept.contains(&id);
            if !held && !readers.is_empty() && readers.iter().all(|reader| ran.contains(reader)) {
                self.rest.entry(id).or_default().found_changed();
            } else {
                self.rest.remove(&id);
            }
        }
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

/// Whether two runs' outputs hold the same Fields word for word. Fields
/// in Host memory are compared there, one made from the other on the
/// pages it changed alone (`Host::same`); Fields held on a
/// device alone are compared on it only `on_device`, when the node ran
/// for nothing but the state it reads, or its outputs are no larger than
/// [`SMALL_COMPARED`], and its [`Rest`] is due: a node with larger
/// outputs that runs for fresh inputs waits for no device.
fn same_words(runtime: &Runtime, previous: &[Handle], fresh: &[Handle], on_device: bool) -> bool {
    let host = runtime.host();
    previous.len() == fresh.len()
        && previous.iter().zip(fresh).all(|(old, new)| {
            old == new
                || match host.same(*old, *new) {
                    Some(same) => same,
                    None => on_device && runtime.same_words(*old, *new),
                }
        })
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

/// The nodes whose outputs are never let go to make room: the scene's
/// outputs, the sources and first values of its state, and the pinned.
fn held_nodes(graph: &Graph, pinned: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    let mut held = pinned.clone();
    for (_, source) in &graph.outputs {
        if let Source::Node { node, .. } = source {
            held.insert(node.clone());
        }
    }
    for node in &graph.nodes {
        for binding in node.inputs.values() {
            if let Binding::Previous {
                source, initial, ..
            } = binding
            {
                for source in [source, initial] {
                    if let Source::Node { node, .. } = source {
                        held.insert(node.clone());
                    }
                }
            }
        }
    }
    held
}

/// Whether a run failed for want of room in the Host, which letting go of
/// what the frame is done with may give it.
fn short_of_room(error: &entelechy_kernel::ExecutionError) -> bool {
    matches!(
        error,
        entelechy_kernel::ExecutionError::OverRoom { .. }
            | entelechy_kernel::ExecutionError::Abi(entelechy_abi::AbiError::Host(
                entelechy_abi::HostError::BudgetExceeded { .. }
            ))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_that_keeps_changing_is_compared_seldom() {
        // Compared at once; then each change found doubles the runs before
        // the next comparison, up to the most.
        let mut rest = Rest::default();
        let mut compared = Vec::new();
        for run in 1..=2100 {
            if rest.due() {
                compared.push(run);
                rest.found_changed();
            }
        }
        assert_eq!(compared, [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]);
    }
}
