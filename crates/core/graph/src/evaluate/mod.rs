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
    AsGraph, Awake, Binding, DomainExpr, DomainSpec, Graph, GraphError, Node, NodeId, ParamValue,
    Source,
};

mod bind;
mod compiled;
mod error;
mod frames;
mod ledger;
mod plan;
mod recall;
mod state;

use bind::Bind;
pub use bind::{Bound, binding_fault, borrowed_domain, type_fault};
use compiled::{CompiledPlan, CompiledSource, GroupBind};
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
    /// State producers, `demand`, and what they read, skipping unasked
    /// graph outputs (`warp`).
    State(&'a BTreeSet<NodeId>),
    /// These, what they read, state producers and outputs never produced.
    For(&'a BTreeSet<NodeId>),
    /// Only what these read, for a trial reporting no output.
    Read(&'a BTreeSet<NodeId>),
}

enum Evaluated {
    None,
    Frame(Box<Report>),
    State(StateStep),
}

/// The report of an evaluation that reports a full frame.
fn reported(outcome: Evaluated) -> Result<Report, EvaluateError> {
    match outcome {
        Evaluated::Frame(report) => Ok(*report),
        Evaluated::None | Evaluated::State(_) => Err(EvaluateError::Graph(GraphError::NoOutput)),
    }
}

/// The report of a headless causal state step ([`Evaluator::evaluate_state`]).
fn state_reported(outcome: Evaluated) -> Result<StateStep, EvaluateError> {
    match outcome {
        Evaluated::State(report) => Ok(report),
        Evaluated::None | Evaluated::Frame(_) => Err(EvaluateError::Graph(GraphError::NoOutput)),
    }
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
    /// How many groups the evaluation bound to find their keys: every group
    /// it reached, or, on a frame that leans on the last, those that move
    /// and those that read an output changed on this frame.
    pub bound: usize,
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
    /// Each node that kept the frame's thread waiting for the device, and
    /// for how long, in evaluation order: binding it (a `value` it reads)
    /// or running it (an input brought from the device, a producer made
    /// again). What waits besides, at the frame's end, is no node's.
    pub waits: Vec<(NodeId, std::time::Duration)>,
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

/// Summary of one headless causal state step ([`Evaluator::evaluate_state`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateStep {
    /// Nodes that executed, in evaluation order.
    pub executed: Vec<NodeId>,
    /// Nodes whose cached outputs were reused.
    pub reused: Reused,
    /// Sources whose state advanced in this step's commit.
    pub committed: Vec<(NodeId, String)>,
    /// `async` nodes with a run in flight; their readers took the last output.
    pub pending: Vec<NodeId>,
    /// Nodes whose cached outputs were let go to make room in the Host.
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

/// A Field the evaluator holds ([`Evaluator::holdings`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holding {
    /// The node whose output or state port holds it.
    pub node: NodeId,
    /// The port.
    pub port: String,
    /// The Field.
    pub handle: Handle,
    /// What it is held as.
    pub held: Held,
}

/// What a Field is held as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Held {
    /// A node's cached output.
    Output,
    /// A port's committed state.
    State,
    /// State the frame in flight replaced, held until it is confirmed.
    Back,
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
    links: Vec<Link>,
}

/// One fusion in a group: the producer inlined, the pairs of the
/// consumer's input it fed and the producer's output that fed it, and the
/// stencil halo radius (`0` for pointwise / indexed, `1..=2` for tiled stencil).
type Link = (NodeId, Vec<(u32, u32)>, u32);

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
/// runs in its turn, is never asleep, and reads only settled nodes), read
/// only by settled nodes.
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
        // (a node that may sleep keeps what it gave for its readers: never
        // let go as made once)
        let mut moves = group.node.is_async
            || !group.node.awake.is_empty()
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
    /// Whether each group of the plan ran, by its place.
    ran: Vec<bool>,
    timings: Vec<(NodeId, std::time::Duration)>,
    /// The nodes that waited for the device ([`Report::waits`]).
    waits: Vec<(NodeId, std::time::Duration)>,
    /// The node being bound and run, and the device's wait count as it
    /// began ([`Frame::waiting_for`]).
    waiting: Option<(NodeId, std::time::Duration)>,
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
            ran: vec![false; layout.place.len()],
            timings: Vec::new(),
            waits: Vec::new(),
            waiting: None,
            clocked: Vec::new(),
            checks: Vec::new(),
            placements: Vec::new(),
            let_go: 0,
            layout,
            asked,
        }
    }

    /// Counts what the device was waited for since the node before began
    /// as that node's, `waited` being the device's wait count now, and
    /// begins counting `node`'s.
    fn waiting_for(&mut self, node: Option<&NodeId>, waited: std::time::Duration) {
        if let Some((id, since)) = self.waiting.take() {
            let spent = waited.saturating_sub(since);
            if !spent.is_zero() {
                self.waits.push((id, spent));
            }
        }
        self.waiting = node.map(|node| (node.clone(), waited));
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
    /// The group each fused node runs in, by its place: a node fused into
    /// another has no run of its own, and ran when that group did.
    fused_at: BTreeMap<NodeId, usize>,
    /// The groups that read each group `<-`, by place.
    readers_at: Vec<Vec<usize>>,
    /// Whether a group's key may change with nothing it reads `<-`
    /// changing while the parameters and the caller's Fields stay as they
    /// were: it reads the clock, input or state, runs off the frame, takes
    /// its domain from the window or its scale, or reads a node the plan
    /// does not run.
    moving: Vec<bool>,
    /// The nodes whose outputs group `at` reads last, which a frame lets go
    /// cold once it has run when they are new frame after frame
    /// ([`CHILL_STREAK`]). What is never let go, and what an `async` node
    /// reads, is not among them.
    chill_after: Vec<Vec<NodeId>>,
    /// Output slots of each group that the Host reads directly (`awake`,
    /// `value` in a domain, an `async` reader's inputs, or `meshes`/`instances`
    /// of `native.trace@2` on hardware ray-query backends).
    host_outputs: Vec<Vec<usize>>,
    /// Groups producing `meshes`/`instances` for `native.trace@2` on a backend
    /// whose TLAS build reads them in Host memory: modest pure host-input runs
    /// prefer the ordered (CPU/JIT) backend to avoid a GPU flush and download.
    prefer_host: Vec<bool>,
    /// `(output_slot, reader)` pairs for each group's executing readers,
    /// prefetched when `reader` runs routed off the main device.
    slot_readers: Vec<Vec<(usize, NodeId)>>,
    /// Pre-compiled slot bindings for every group and graph output.
    compiled: CompiledPlan,
}

impl Layout {
    fn new(
        graph: &Graph,
        groups: &[Group],
        pinned: &BTreeSet<NodeId>,
        needed: Option<&BTreeSet<NodeId>>,
        runtime: &Runtime,
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
            // A domain written from counts, values and parameters is as its
            // inputs and the parameters are; one that reads the window or
            // its scale moves with the evaluation's context.
            let mut moves = group.node.is_async
                || requires.contains(Capability::TimeRead)
                || requires.contains(Capability::InputRead)
                || matches!(&group.node.domain, Some(DomainSpec::Expr(expr)) if expr.reads_context());
            let mut reads = Vec::new();
            for binding in group.node.inputs.values() {
                match binding {
                    Binding::Now(Source::Node { node, .. }) => match runs_at.get(node) {
                        Some(at) => reads.push(*at),
                        None => moves = true,
                    },
                    // A caller's Field is keyed by its handle, and a frame
                    // leans only under the same Fields ([`Quiet`]).
                    Binding::Now(Source::External(_)) => {}
                    Binding::Previous { .. } => moves = true,
                }
            }
            deps.push(reads);
            moving.push(moves);
        }
        let held = held_nodes(graph, pinned);
        let mut chill_after = vec![Vec::new(); groups.len()];
        for (at, group) in groups.iter().enumerate() {
            let id = &group.node.id;
            let read_by_async = groups
                .iter()
                .enumerate()
                .any(|(reader, g)| g.node.is_async && deps[reader].contains(&at));
            if !held.contains(id)
                && !kept.contains(id)
                && !read_by_async
                && let Some(last) = last_read.get(id)
            {
                chill_after[*last].push(id.clone());
            }
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
        let fused_at: BTreeMap<NodeId, usize> = fused
            .iter()
            .filter_map(|(group, absorbed)| Some((runs_at.get(group)?, absorbed)))
            .flat_map(|(at, absorbed)| absorbed.iter().map(|node| (node.clone(), *at)))
            .collect();
        let output_slot = |producer: &NodeId, port: &str| -> Option<(usize, usize)> {
            let at = *runs_at.get(producer)?;
            let slot = groups[at]
                .manifest()
                .outputs
                .iter()
                .position(|output| output.name == port)?;
            Some((at, slot))
        };
        let mut host_outputs = vec![Vec::new(); groups.len()];
        let mut prefer_host = vec![false; groups.len()];
        let mut slot_readers: Vec<Vec<(usize, NodeId)>> = vec![Vec::new(); groups.len()];
        let mark_host = |host_outputs: &mut [Vec<usize>], at: usize, slot: usize| {
            if !host_outputs[at].contains(&slot) {
                host_outputs[at].push(slot);
            }
        };
        for node in &graph.nodes {
            for awake in &node.awake {
                if let Source::Node {
                    node: producer,
                    port,
                } = &awake.source
                    && let Some((at, slot)) = output_slot(producer, port)
                {
                    mark_host(&mut host_outputs, at, slot);
                }
            }
        }
        for group in groups {
            if let Some(DomainSpec::Expr(expr)) = &group.node.domain {
                let indirect_port =
                    matches!(group.manifest().mode, entelechy_ir::Mode::Scatter { .. })
                        .then(|| {
                            if group.node.is_async
                                || !runtime.supports_indirect_scatter(&group.version)
                            {
                                return None;
                            }
                            let (port_name, ..) = expr.indirect_rows_spec()?;
                            let slot = group
                                .manifest()
                                .inputs
                                .iter()
                                .position(|i| i.name == port_name)?;
                            (group.manifest().inputs[slot].element() == Type::Int)
                                .then_some(port_name)
                        })
                        .flatten();
                for (in_port, _) in expr.values_read() {
                    if indirect_port.as_deref() == Some(in_port.as_str()) {
                        continue;
                    }
                    if let Some(binding) = group.node.inputs.get(&in_port) {
                        let source = match binding {
                            Binding::Now(source) | Binding::Previous { source, .. } => source,
                        };
                        if let Source::Node {
                            node: producer,
                            port,
                        } = source
                            && let Some((at, slot)) = output_slot(producer, port)
                        {
                            mark_host(&mut host_outputs, at, slot);
                        }
                    }
                }
            }
            if runtime.needs_host_trace_instances()
                && group.version.native() == Some(entelechy_kernel::NativeOp::TraceInstances)
            {
                for port_name in ["meshes", "instances"] {
                    if let Some(binding) = group.node.inputs.get(port_name) {
                        let source = match binding {
                            Binding::Now(source) | Binding::Previous { source, .. } => source,
                        };
                        if let Source::Node {
                            node: producer,
                            port,
                        } = source
                            && let Some((at, slot)) = output_slot(producer, port)
                        {
                            mark_host(&mut host_outputs, at, slot);
                            prefer_host[at] = true;
                        }
                    }
                }
            }
            for binding in group.node.inputs.values() {
                let source = match binding {
                    Binding::Now(source) | Binding::Previous { source, .. } => source,
                };
                if let Source::Node {
                    node: producer,
                    port,
                } = source
                    && let Some((at, slot)) = output_slot(producer, port)
                {
                    if group.node.is_async {
                        mark_host(&mut host_outputs, at, slot);
                    } else {
                        let pair = (slot, group.node.id.clone());
                        if !slot_readers[at].contains(&pair) {
                            slot_readers[at].push(pair);
                        }
                    }
                }
            }
        }
        let compiled = CompiledPlan::new(graph, groups, &runs_at);
        Self {
            runs_at,
            last_read,
            readers,
            held,
            runs,
            graph_readers,
            kept,
            absorbed: absorbed.into_iter().cloned().collect(),
            names: Arc::new(names),
            place,
            fused_at,
            fused: Arc::new(fused),
            readers_at,
            moving,
            chill_after,
            host_outputs,
            prefer_host,
            slot_readers,
            compiled,
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
/// is passed by unbound while all of it holds: its key is made of its
/// Kernel's version, its parameters, the Fields and outputs it reads and
/// the extent they give it, none of which has changed.
struct Quiet {
    layout: Arc<Layout>,
    ledger: u64,
    params: BTreeMap<String, Value>,
    externals: BTreeMap<String, Handle>,
    steady: Vec<bool>,
}

/// What a frame reads, with the pins and the nodes asleep it was found
/// under.
struct FrameReach {
    pins: BTreeSet<NodeId>,
    asleep: BTreeSet<NodeId>,
    read: Arc<BTreeSet<NodeId>>,
}

/// What a headless state step reads, with the demanded nodes and asleep
/// nodes it was found under.
struct StateReach {
    demand: BTreeSet<NodeId>,
    asleep: BTreeSet<NodeId>,
    read: Arc<BTreeSet<NodeId>>,
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
        runtime: &Runtime,
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
            runtime,
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
    external_handles: Vec<Option<Handle>>,
    context: &'a Context,
    params: &'a BTreeMap<String, Value>,
    overridden_params: Vec<Option<&'a Value>>,
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
    /// The producers it alone reads that run on their own, and why.
    apart: Vec<(NodeId, String)>,
    /// Each producer inlined and the ports it fed, in the order fused.
    links: Vec<Link>,
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
            links: self.links.clone(),
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
    /// The producers only this group reads that run on their own, each
    /// with why fusion left it apart.
    pub apart: Vec<(NodeId, String)>,
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
    /// Groups fusion declined, by consumer, at their generations, with
    /// why; retried when one changes.
    declined: BTreeMap<NodeId, (Members, String)>,
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
    /// Indices of nodes in the graph in `shape` that declare `awake` conditions.
    awake_nodes: Option<Vec<usize>>,
    /// The state reads of the last plan reconciled.
    state_reads: Option<(Arc<Vec<Group>>, Arc<state::StateReads>)>,
    /// What a frame of the graph in `shape` reads, and the pins and the
    /// nodes asleep it was found under.
    frame_needed: Option<FrameReach>,
    /// What a headless state step of the graph in `shape` reads, kept so
    /// repeated `warp` ticks share one [`Layout`] and [`Quiet`] state.
    state_needed: Option<StateReach>,
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
    /// The outputs a caller reads once their frame is confirmed; see
    /// [`Evaluator::await_outputs`].
    awaited: Vec<String>,
    /// Those outputs as the frame last confirmed left them.
    confirmed: Vec<(String, Handle)>,
    /// Small nodes this evaluation found unchanged; see
    /// [`Evaluator::weigh_alike`].
    alike: Vec<NodeId>,
    /// The plan last asked which of its nodes' outputs are made once, and
    /// those nodes with their readers ([`made_once`]).
    made_once: Option<(Arc<Vec<Group>>, Vec<MadeOnce>)>,
    /// How many frames in a row each node's outputs came out new.
    changing: crate::hash::QuickMap<NodeId, u32>,
    /// Executing nodes whose last run was routed off the main device.
    routed: crate::hash::QuickSet<NodeId>,
    /// The last shared graph that read a fallback, the externals it was
    /// given none for, and the graph with their fallbacks written in
    /// ([`Evaluator::with_fallbacks`]).
    fallen: Option<Fallen>,
}

/// A shared graph, the names of the externals whose fallbacks are in use,
/// and the graph with them written in.
type Fallen = (Arc<Graph>, Vec<String>, Arc<Graph>);

/// The frames in a row a node's outputs must come out new for a frame to
/// let them go cold once their last reader has run. The next frame makes
/// them anew whatever it finds, so holding them from frame to frame keeps
/// memory nothing reads again; let go as soon as read, their memory serves
/// what runs after them. A node whose outputs come out the same on some
/// frames keeps them, for they spare what reads them then (and a frame
/// at the same time as the last runs nothing).
const CHILL_STREAK: u32 = 8;

/// The frames between two that keep a node's new outputs to compare the
/// next ones with ([`lets_go`]).
const CHILL_CHECK: u32 = 32;

/// Whether node `id`, whose outputs came out new `run` frames in a row,
/// lets them go cold once their last reader has run. Let go, the next
/// frame has nothing to compare its outputs with and finds them new, so
/// the run would never end: one frame in [`CHILL_CHECK`] keeps them, and
/// the next finds whether they settled. Outputs that settle are found so
/// within that many frames, and kept from then on. Which frame is the
/// node's own, so nodes that began changing together keep theirs on
/// different frames, and no frame holds them all at once.
fn lets_go(id: &NodeId, run: u32) -> bool {
    let own =
        id.0.bytes()
            .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(u32::from(b)));
    run >= CHILL_STREAK && run.wrapping_add(own) % CHILL_CHECK != 0
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

/// The most doublings of a [`Rest`]'s wait: a node compared at least
/// once in 1024 of its runs.
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
            recalled: recall::Recalled::default(),
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

    /// Keeps a node materialized so its image can be observed. Pinning a
    /// node already pinned changes nothing, and asks for no sweep of the
    /// ledger: a session pins the same nodes before every frame.
    pub fn pin(&mut self, node: NodeId) {
        if self.pinned.insert(node) {
            self.pruned = false;
        }
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

    /// The nodes run fused into `consumer`, itself left out, in their
    /// order: a fault said of `consumer` may be any of theirs.
    pub fn fused_parts(&self, consumer: &NodeId) -> Vec<NodeId> {
        self.fused
            .iter()
            .filter(|(fused_into, _)| *fused_into == consumer)
            .flat_map(|(_, fused)| fused.members.parts.iter().map(|(part, _)| part))
            .filter(|part| *part != consumer)
            .cloned()
            .collect()
    }

    /// Lets fusion have `node` back.
    pub fn unpin(&mut self, node: &NodeId) {
        if self.pinned.remove(node) {
            self.pruned = false;
        }
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
        self.routed.clear();
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

    /// Every Field the evaluator holds and for what: a node's cached
    /// output, a port's committed state, or the state the frame in flight
    /// replaced, held to be put back. A Field may be held more than once.
    pub fn holdings(&self) -> Vec<Holding> {
        let mut holdings = Vec::new();
        for node in self.ledger.cached_nodes() {
            if let Some(cached) = self.ledger.entry(node) {
                for (port, handle) in cached.output_names.iter().zip(&cached.outputs) {
                    holdings.push(Holding {
                        node: node.clone(),
                        port: port.clone(),
                        handle: *handle,
                        held: Held::Output,
                    });
                }
            }
        }
        for ((node, port), handle) in self.ledger.committed_ports() {
            holdings.push(Holding {
                node: node.clone(),
                port: port.clone(),
                handle,
                held: Held::State,
            });
        }
        for ((node, port), handle) in self.held_back_ports() {
            holdings.push(Holding {
                node,
                port,
                handle,
                held: Held::Back,
            });
        }
        holdings
    }

    /// The cached image for a node's output port, if it was evaluated and
    /// materialized. Absorbed nodes have none; pin them to observe them.
    pub fn cached(&self, node: &NodeId, port: &str) -> Option<Handle> {
        let cached = self.ledger.entry(node)?;
        let index = cached.output_names.iter().position(|name| name == port)?;
        Some(cached.outputs[index])
    }

    /// The nodes the runtime will give its ordered backend however the
    /// graph is placed, as far as is known before anything runs: each runs
    /// over [`entelechy_kernel::FEW_ELEMENTS`] or fewer elements, and what
    /// it reads now is the caller's, state, or such a node's, nothing a
    /// device made in the evaluation. None where the runtime keeps every run
    /// on its main backend. A Kernel only these run needs nothing built on
    /// the main backend's device.
    pub fn runs_few(
        &mut self,
        graph: &impl AsGraph,
        runtime: &Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> BTreeSet<NodeId> {
        let mut few = BTreeSet::new();
        let (graph, shared) = (graph.graph(), graph.shared());
        if !runtime.routes_few() {
            return few;
        }
        let Ok(order) = self.order_of(graph, shared) else {
            return few;
        };
        let nodes: BTreeMap<&NodeId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(at, node)| (&node.id, at))
            .collect();
        let mut domains = BTreeMap::new();
        let mut off_device = BTreeSet::new();
        for &at in order.iter() {
            let node = &graph.nodes[at];
            let Some(version) = runtime.kernel(&node.kernel) else {
                continue;
            };
            let planned = self.planned(
                graph, &nodes, node, &version, runtime, externals, &domains, overrides, context,
            );
            if let Some(extent) = planned.extent {
                domains.insert(node.id.clone(), extent);
            }
            if runtime.placement(&version).routed.is_some() {
                off_device.insert(node.id.clone());
                continue;
            }
            // A reduction, a prefix and a scatter run over more than their
            // output says: only a map's and a scan's runs are counted here.
            let counted = matches!(
                version.manifest().mode,
                entelechy_ir::Mode::Map | entelechy_ir::Mode::Scan
            );
            let reads_device = node.inputs.values().any(|binding| match binding {
                Binding::Now(Source::Node { node, .. }) => !off_device.contains(node),
                _ => false,
            });
            if counted
                && !reads_device
                && planned
                    .runs
                    .is_some_and(|runs| runs.count() as u64 <= entelechy_kernel::FEW_ELEMENTS)
            {
                off_device.insert(node.id.clone());
                few.insert(node.id.clone());
            }
        }
        few
    }

    /// What would execute for `graph` now under `context` (its window and
    /// scale size what reads them), in evaluation order. Prepares fused
    /// versions; executes nothing.
    pub fn plan_summary(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Result<Vec<PlanEntry>, EvaluateError> {
        let (graph, shared) = (graph.graph(), graph.shared());
        let order = self.order_of(graph, shared)?;
        let groups = self.plan(graph, &order, runtime, externals, overrides, context)?;
        Ok(groups
            .iter()
            .map(|group| PlanEntry {
                node: group.node.id.clone(),
                kernel: group.version.id().clone(),
                absorbed: group.absorbed.clone(),
                counted: group.counted.clone(),
                apart: group.apart.clone(),
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

    /// Evaluates one headless causal state step: every `<~` state producer,
    /// `demand`, and what they read while awake, skipping unasked graph
    /// outputs (`picture`, `sound`).
    pub fn evaluate_state(
        &mut self,
        graph: &impl AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: &BTreeSet<NodeId>,
    ) -> Result<StateStep, EvaluateError> {
        self.recover(runtime);
        self.confirm(runtime)?;
        self.evaluate_with(
            graph,
            runtime,
            externals,
            context,
            params,
            Demand::State(demand),
        )
        .and_then(state_reported)
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
            Bind::Ready(invocation) => Ok(invocation),
            Bind::Cold(cold) => Err(EvaluateError::UnknownOutput {
                node: cold[0].clone(),
                port: "result".to_owned(),
                near: None,
            }),
        }
    }

    /// The nodes an evaluation under `demand` must run: see
    /// [`Evaluator::evaluate_for`] and [`Evaluator::evaluate_trial_reading`];
    /// `None` every node.
    fn needed(
        &mut self,
        graph: &Graph,
        demand: Demand<'_>,
        asleep: &BTreeSet<NodeId>,
    ) -> Option<Arc<BTreeSet<NodeId>>> {
        let demand = match demand {
            Demand::Every => return None,
            // A frame reads what the graph's outputs, state and pins reach
            // through what is awake: found once for the graph in `shape`,
            // its pins and the nodes asleep.
            Demand::Frame => {
                if let Some(known) = &self.frame_needed
                    && known.pins == self.pinned
                    && known.asleep == *asleep
                {
                    return Some(Arc::clone(&known.read));
                }
                let read = Arc::new(graph.read_awake(self.pinned.iter().cloned(), asleep));
                self.frame_needed = Some(FrameReach {
                    pins: self.pinned.clone(),
                    asleep: asleep.clone(),
                    read: Arc::clone(&read),
                });
                return Some(read);
            }
            Demand::State(demand) => {
                if let Some(known) = &self.state_needed
                    && known.demand == *demand
                    && known.asleep == *asleep
                {
                    return Some(Arc::clone(&known.read));
                }
                let read = Arc::new(graph.read_state_awake(demand.iter().cloned(), asleep));
                self.state_needed = Some(StateReach {
                    demand: demand.clone(),
                    asleep: asleep.clone(),
                    read: Arc::clone(&read),
                });
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

    /// The nodes asleep on this frame: those with an `awake` whose row the
    /// last frame left at 0, and with outputs of their own to give (one
    /// never run runs). A row is read where it is; one on a device is
    /// brought to the Host, a wait.
    fn asleep(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
    ) -> Result<BTreeSet<NodeId>, EvaluateError> {
        let indices = self.awake_nodes.get_or_insert_with(|| {
            graph
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(at, node)| (!node.awake.is_empty()).then_some(at))
                .collect()
        });
        if indices.is_empty() {
            return Ok(BTreeSet::new());
        }
        // Take temporarily so `self.row_is_zero` can borrow `&self`.
        let indices = self.awake_nodes.take().expect("initialized");
        let result = (|| -> Result<BTreeSet<NodeId>, EvaluateError> {
            let mut asleep = BTreeSet::new();
            // A row read once a frame, however many nodes read it.
            let mut rows: Vec<(&Awake, bool)> = Vec::new();
            for &at in &indices {
                let node = &graph.nodes[at];
                if self.ledger.entry(&node.id).is_none() {
                    continue;
                }
                for awake in &node.awake {
                    let zero = match rows.iter().find(|(read, _)| *read == awake) {
                        Some((_, zero)) => *zero,
                        None => {
                            let zero = self.row_is_zero(node, awake, runtime, externals)?;
                            rows.push((awake, zero));
                            zero
                        }
                    };
                    if zero {
                        asleep.insert(node.id.clone());
                        break;
                    }
                }
            }
            Ok(asleep)
        })();
        self.awake_nodes = Some(indices);
        result
    }

    /// Whether `awake`'s row, as the last frame left it, is 0; nothing yet
    /// made is not 0.
    fn row_is_zero(
        &self,
        node: &Node,
        awake: &Awake,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
    ) -> Result<bool, EvaluateError> {
        let handle = match &awake.source {
            Source::External(name) => externals.get(name).copied(),
            Source::Node { node, port } => self.cached(node, port),
        };
        let Some(handle) = handle else {
            return Ok(false);
        };
        let fail = |message: String| EvaluateError::Domain {
            node: node.id.clone(),
            message: format!(
                "`awake = value({}, {})`: {message}",
                awake.source, awake.row
            ),
        };
        let count = runtime.host().domain(handle)?.count();
        if count > VALUE_MAX_ELEMENTS {
            return Err(fail(format!(
                "a row is brought to the Host to be read, so it holds counts, not data; this one has {count} elements and the most is {VALUE_MAX_ELEMENTS}"
            )));
        }
        if awake.row as usize >= count {
            return Err(fail(format!("it has {count} element(s)")));
        }
        runtime
            .materialize(handle)
            .map_err(|e| fail(e.to_string()))?;
        Ok(runtime.host().element(handle, awake.row)?[0] == 0)
    }

    /// The graph with each unsupplied external that has a fallback read
    /// from it; `None` when nothing is substituted. For a shared graph it
    /// is made once and kept while the graph is the one before and the
    /// same externals are missing, so the frames after find the plan by
    /// the copy's identity instead of comparing a new copy whole.
    fn with_fallbacks(
        &mut self,
        graph: &dyn AsGraph,
        externals: &BTreeMap<String, Handle>,
    ) -> Option<Arc<Graph>> {
        let missing = |name: &&(String, Source)| !externals.contains_key(&name.0);
        if let Some((from, names, made)) = &self.fallen
            && graph
                .shared()
                .is_some_and(|shared| Arc::ptr_eq(shared, from))
            && from
                .fallbacks
                .iter()
                .filter(missing)
                .map(|(name, _)| name)
                .eq(names)
        {
            return Some(Arc::clone(made));
        }
        self.fallen = None;
        let names: Vec<String> = graph
            .graph()
            .fallbacks
            .iter()
            .filter(missing)
            .map(|(name, _)| name.clone())
            .collect();
        if names.is_empty() {
            return None;
        }
        let made = Arc::new(substitute_fallbacks(graph.graph(), &names));
        if let Some(shared) = graph.shared() {
            self.fallen = Some((Arc::clone(shared), names, Arc::clone(&made)));
        }
        Some(made)
    }

    /// Evaluates what `demand` asks; the report is [`Evaluated::None`] only
    /// for a [`Demand::Read`].
    fn evaluate_with(
        &mut self,
        graph: &dyn AsGraph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        demand: Demand<'_>,
    ) -> Result<Evaluated, EvaluateError> {
        // An external the caller supplies nothing for reads its fallback,
        // written into a copy of the graph only when one is in use.
        let substituted = self.with_fallbacks(graph, externals);
        // A graph with fallbacks written in is a new one, not the caller's.
        let shared = match &substituted {
            Some(substituted) => Some(substituted),
            None => graph.shared(),
        };
        let graph = substituted.as_deref().unwrap_or(graph.graph());
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
        // Who sleeps on a frame or state step, by what the last left: none on a trial.
        let asleep = if matches!(demand, Demand::Frame | Demand::State(_)) && !self.trial {
            self.asleep(graph, runtime, externals)?
        } else {
            BTreeSet::new()
        };
        let needed = self.needed(graph, demand, &asleep);
        let groups = self.plan(graph, &order, runtime, externals, params, context)?;
        let (migrated, reset) = self.reconcile_state(&groups, runtime, context)?;
        // From here to the commit, a failure leaves state as it was: a
        // migration's replaced state is put back.
        let evaluated = (|| -> Result<Evaluated, EvaluateError> {
            let layout = self.layout_of(graph, &groups, needed.as_ref(), runtime);
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
                Demand::For(demand) | Demand::State(demand) => demand.clone(),
                _ => BTreeSet::new(),
            };
            // A frame or state step leans on the last when nothing it was
            // made from moved but the clock and input: the same plan,
            // ledger, parameters and Fields. Then a group that does not
            // move, none of whose reads changed, keeps its entry unbound.
            let leans = matches!(demand, Demand::Frame | Demand::State(_))
                && !self.trial
                && !self.recalling();
            let last = if leans { self.quiet.take() } else { None };
            let ledger_now = self.ledger.version();
            let mut lean = last.as_ref().is_some_and(|last| {
                Arc::ptr_eq(&last.layout, &layout)
                    && last.ledger == ledger_now
                    && last.params == *params
                    && last.externals == *externals
            });
            let (mut steady, kept_quiet) = match last {
                Some(last) if lean => (last.steady, Some((last.params, last.externals))),
                _ => (vec![false; groups.len()], None),
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
            let external_handles: Vec<Option<Handle>> = layout
                .compiled
                .external_names
                .iter()
                .map(|name| externals.get(name).copied())
                .collect();
            let overridden_params: Vec<Option<&Value>> = if params.is_empty() {
                vec![None; graph.params.len()]
            } else {
                graph.params.iter().map(|p| params.get(&p.name)).collect()
            };
            let mut frame = Frame::new(layout, asked);
            let mut bound_groups = 0usize;
            let scene = Scene {
                graph,
                groups: &groups,
                external_handles,
                context,
                params,
                overridden_params,
            };
            // What every-frame outputs a frame lets go as their last reader
            // has run ([`Layout::chill_after`]): a frame that commits only.
            let chills = !self.trial && matches!(demand, Demand::Every | Demand::Frame);
            // A node's run of new outputs ends unless it runs and makes new
            // ones again on this frame: the runs as the last frame left
            // them are taken, and a node that does puts its own back, one
            // longer. What is left of them when the frame is done ended.
            let mut streaks = if chills {
                std::mem::take(&mut self.changing)
            } else {
                crate::hash::QuickMap::default()
            };
            for (at, group) in groups.iter().enumerate() {
                if chills && at > 0 {
                    for id in &frame.layout.chill_after[at - 1] {
                        if self.changing.get(id).is_some_and(|run| lets_go(id, *run)) {
                            self.chill(id, runtime);
                        }
                    }
                }
                // Passed by before anything of it is looked up: on a frame
                // that leans, most groups are.
                if lean && !dirty[at] {
                    continue;
                }
                let node = &group.node;
                if !frame.layout.runs(at) {
                    continue;
                }
                // Asleep: its readers take what it last gave.
                if !asleep.is_empty() && asleep.contains(&node.id) {
                    continue;
                }
                frame.waiting_for(Some(&node.id), runtime.device_waited());
                let before = self.output_epoch(&node.id);
                bound_groups += 1;
                let mut bound = self.bind_group(at, &frame.layout, &scene, runtime)?;
                // A domain read from outputs let go cold waits for them.
                while let GroupBind::Unkeyed(cold) = &bound {
                    // What is made again may take a new epoch: the rest of
                    // the frame binds every group.
                    lean = false;
                    for producer in cold.clone() {
                        self.warm(&producer, at, &scene, &mut frame, runtime)?;
                    }
                    bound = self.bind_group(at, &frame.layout, &scene, runtime)?;
                }
                let key = match &bound {
                    GroupBind::Ready { key, .. } | GroupBind::Keyed(key, _) => key,
                    GroupBind::Unkeyed(_) => unreachable!("waited for above"),
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
                let (invocation, key, indirect) = match bound {
                    GroupBind::Ready {
                        handles,
                        key,
                        indirect,
                    } => (
                        Invocation {
                            inputs: handles,
                            params: key.params.clone(),
                            domain: key.domain,
                            context: *context,
                        },
                        key,
                        indirect,
                    ),
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
                    indirect,
                    rerun,
                    &[],
                    &scene,
                    &mut frame,
                    runtime,
                )?;
                // Outputs equal to the last, word for word, keep its epoch:
                // what reads them finds nothing changed.
                let new = self.output_epoch(&node.id) != before;
                if new {
                    frame.layout.touch_readers(at, &mut dirty);
                }
                if chills && new {
                    let streak = streaks.remove(&node.id).unwrap_or(0);
                    self.changing
                        .insert(node.id.clone(), streak.saturating_add(1));
                }
                absent.push(frame.layout.place[at]);
                steady[at] = true;
            }
            if chills && let Some(last) = frame.layout.chill_after.last() {
                for id in last {
                    if self.changing.get(id).is_some_and(|run| lets_go(id, *run)) {
                        self.chill(id, runtime);
                    }
                }
            }
            frame.waiting_for(None, runtime.device_waited());
            let Frame {
                executed,
                ran,
                timings,
                waits,
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
            self.weigh_alike(&layout, &ran);
            if matches!(demand, Demand::Read(_)) {
                self.wait_for_checks(checks, runtime)?;
                return Ok(Evaluated::None);
            }
            let state_only = matches!(demand, Demand::State(_));
            let mut outputs = Vec::with_capacity(layout.compiled.outputs.len());
            for (name, source) in &layout.compiled.outputs {
                let found = match source {
                    CompiledSource::External {
                        slot,
                        name: external,
                    } => match scene.external_handles[*slot] {
                        Some(handle) => Some(handle),
                        None if state_only => None,
                        None => {
                            return Err(EvaluateError::MissingExternal {
                                node: NodeId(format!("output {name}").into()),
                                name: external.clone(),
                            });
                        }
                    },
                    CompiledSource::Node {
                        producer,
                        port,
                        known,
                    } => {
                        let cached = self.ledger.entry(producer).and_then(|cached| match known {
                            Some((names, slot, _)) if Arc::ptr_eq(&cached.output_names, names) => {
                                Some(cached.outputs[*slot])
                            }
                            _ => cached
                                .output_names
                                .iter()
                                .position(|n| n == port)
                                .map(|idx| cached.outputs[idx]),
                        });
                        match cached {
                            Some(handle) => Some(handle),
                            None if state_only => None,
                            None => {
                                let near = self.ledger.entry(producer).and_then(|cached| {
                                    entelechy_abi::names::nearest(
                                        port,
                                        cached.output_names.iter().map(String::as_str),
                                    )
                                    .map(str::to_owned)
                                });
                                return Err(EvaluateError::UnknownOutput {
                                    node: producer.clone(),
                                    port: port.clone(),
                                    near,
                                });
                            }
                        }
                    }
                };
                if let Some(handle) = found {
                    outputs.push((name.clone(), handle));
                }
            }
            let first_output = if state_only {
                None
            } else {
                Some(
                    outputs
                        .first()
                        .map(|(_, handle)| *handle)
                        .ok_or(EvaluateError::Graph(GraphError::NoOutput))?,
                )
            };
            // A trial commits nothing, and is not remembered. Overlapping,
            // the frame commits now and its device is waited for once the
            // next frame's work has gone to it.
            let committed = if self.trial {
                self.wait_for_checks(checks, runtime)?;
                Vec::new()
            } else if self.overlap && matches!(demand, Demand::Every | Demand::Frame) {
                let awaited = self.awaited_of(graph, &outputs);
                let committed =
                    self.commit_in_flight(graph, runtime, checks, executed.clone(), awaited)?;
                self.remember_frame(executed.iter().chain(reused.iter()), runtime);
                self.chill_made_once(&groups, &held, runtime);
                committed
            } else {
                self.wait_for_checks(checks, runtime)?;
                let committed = self.commit(graph, runtime)?;
                let awaited = self.awaited_of(graph, &outputs);
                self.confirm_outputs(awaited, runtime);
                self.remember_frame(executed.iter().chain(reused.iter()), runtime);
                self.chill_made_once(&groups, &held, runtime);
                committed
            };
            if leans {
                let (saved_params, saved_externals) =
                    kept_quiet.unwrap_or_else(|| (params.clone(), externals.clone()));
                self.quiet = Some(Quiet {
                    layout: Arc::clone(&layout),
                    ledger: self.ledger.version(),
                    params: saved_params,
                    externals: saved_externals,
                    steady,
                });
            }
            let Some(output) = first_output else {
                self.refused.clear();
                return Ok(Evaluated::State(StateStep {
                    executed,
                    reused,
                    committed,
                    pending,
                    let_go,
                }));
            };
            Ok(Evaluated::Frame(Box::new(Report {
                output,
                executed,
                timings,
                waits,
                clocked,
                placements,
                reused,
                bound: bound_groups,
                committed,
                outputs,
                fused,
                migrated,
                reset,
                pending,
                landed,
                refused: std::mem::take(&mut self.refused),
                let_go,
            })))
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
    ) -> Result<
        (
            Invocation,
            CacheKey,
            Option<entelechy_kernel::IndirectCount>,
        ),
        EvaluateError,
    > {
        let group = &scene.groups[at];
        // Each round makes at least one producer again; a producer let go
        // again at once, for want of room, ends it.
        for _ in 0..=group.node.inputs.len() {
            match self.bind_group(at, &frame.layout, scene, runtime)? {
                GroupBind::Ready {
                    handles,
                    key,
                    indirect,
                } => {
                    return Ok((
                        Invocation {
                            inputs: handles,
                            params: key.params.clone(),
                            domain: key.domain,
                            context: *scene.context,
                        },
                        key,
                        indirect,
                    ));
                }
                GroupBind::Keyed(_, cold) | GroupBind::Unkeyed(cold) => {
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
        let (invocation, key, indirect) = self.ready(at, turn, scene, frame, runtime)?;
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
            at, turn, invocation, key, indirect, rerun, &reads, scene, frame, runtime,
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
        indirect: Option<entelechy_kernel::IndirectCount>,
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
        let run = |runtime: &mut Runtime| match (group.chain, indirect) {
            (Some(chain), _) => {
                runtime.execute_repeated(&group.version, &invocation, group.repeat, chain)
            }
            (None, Some(ic)) => {
                if let Some(res) = runtime.execute_indirect_scatter(&group.version, &invocation, ic)
                {
                    res
                } else {
                    // Backend declined indirect dispatch at runtime: fall back
                    // by materializing the count once on the Host.
                    let mut fallback = invocation.clone();
                    let h = fallback.inputs[ic.slot];
                    runtime.materialize(h)?;
                    let raw = runtime
                        .host()
                        .element(h, ic.index)
                        .map_err(|e| ExecutionError::Device(e.to_string()))?[0]
                        as i32;
                    let val = i64::from(raw) * i64::from(ic.scale) + i64::from(ic.offset);
                    let clamped = if ic.min_val > 0 {
                        val.max(i64::from(ic.min_val))
                    } else {
                        val
                    };
                    let n = u32::try_from(clamped)
                        .ok()
                        .filter(|&n| n > 0)
                        .ok_or_else(|| {
                            ExecutionError::Device(
                                "indirect domain resolved to a non-positive row count".into(),
                            )
                        })?;
                    fallback.domain = Domain::Rows(n);
                    runtime.execute(&group.version, &fallback)
                }
            }
            (None, None) => {
                if frame.layout.prefer_host[at] {
                    runtime.execute_for_host(&group.version, &invocation)
                } else {
                    runtime.execute(&group.version, &invocation)
                }
            }
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
        let on_main = placement.routed.is_none();
        if on_main {
            self.routed.remove(&node.id);
        } else {
            self.routed.insert(node.id.clone());
        }
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
        if on_main && !self.trial && runtime.has_device() {
            let host_slots = &frame.layout.host_outputs[at];
            let readers = &frame.layout.slot_readers[at];
            if (!host_slots.is_empty() || !readers.is_empty())
                && let Some(cached) = self.ledger.entry(&node.id)
            {
                let mut to_prefetch: Vec<Handle> = Vec::new();
                for &slot in host_slots {
                    if let Some(&handle) = cached.outputs.get(slot)
                        && !to_prefetch.contains(&handle)
                    {
                        to_prefetch.push(handle);
                    }
                }
                for (slot, reader) in readers {
                    if self.routed.contains(reader)
                        && let Some(&handle) = cached.outputs.get(*slot)
                        && !to_prefetch.contains(&handle)
                    {
                        to_prefetch.push(handle);
                    }
                }
                for handle in to_prefetch {
                    runtime.prefetch(handle);
                }
            }
        }
        frame.executed.push(node.id.clone());
        frame.ran[at] = true;
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
        if self.running.is_empty() {
            return Ok(Vec::new());
        }
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
    fn weigh_alike(&mut self, layout: &Layout, ran: &[bool]) {
        if self.alike.is_empty() {
            return;
        }
        // A reader ran when its group did, asked by the group's place: a
        // reader fused into another node has no run of its own, and its
        // group is that node's. Only the readers of what was found alike
        // are asked of, however many nodes the frame ran.
        let ran = |reader: &NodeId| {
            layout
                .runs_at
                .get(reader)
                .or_else(|| layout.fused_at.get(reader))
                .is_some_and(|at| ran[*at])
        };
        for id in std::mem::take(&mut self.alike) {
            let readers = layout.graph_readers.get(&id).map_or(&[][..], Vec::as_slice);
            let held = layout.kept.contains(&id);
            if !held && !readers.is_empty() && readers.iter().all(ran) {
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

    /// Whether any fused group is still preparing off the frame.
    pub fn has_pending_fusions(&self) -> bool {
        !self.pending.is_empty()
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
    let _ = runtime.release(handle);
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

/// The graph with each of the externals `missing` names read from its
/// fallback.
fn substitute_fallbacks(graph: &Graph, missing: &[String]) -> Graph {
    let missing: BTreeMap<&str, &Source> = graph
        .fallbacks
        .iter()
        .filter(|(name, _)| missing.contains(name))
        .map(|(name, source)| (name.as_str(), source))
        .collect();
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
    graph
}

/// The nodes whose outputs are never let go to make room: the scene's
/// outputs, the sources and first values of its state, the nodes that may
/// sleep, and the pinned.
fn held_nodes(graph: &Graph, pinned: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    let mut held = pinned.clone();
    // What sleeps is what its readers take while it does.
    for node in &graph.nodes {
        if !node.awake.is_empty() {
            held.insert(node.id.clone());
        }
    }
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
