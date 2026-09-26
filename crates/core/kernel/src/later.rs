//! Execution off the frame: a version run on its own thread over copies
//! of its inputs, redeemed by a poll.
//!
//! [`Runtime::execute_later`] brings the inputs into Host memory, copies
//! them into the thread's own Host and starts the backend there.
//! [`Runtime::poll_run`] takes what landed as Fields of the runtime's
//! Host; [`Runtime::wait_run`] blocks for one run. The copies cost a pass
//! over the inputs, plus a download and upload on a resident backend.
//!
//! A run's room is set aside when it starts, on the caller's thread and
//! in the caller's order: its input copies and its outputs twice over (a
//! backend may alternate two buffers). The Host counts that room until
//! the run is redeemed or discarded, and the run's Host holds no more, so
//! whether a run fits depends only on what it was given.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use orla_abi::{FieldData, Handle, Host, Invocation, Scope};
use orla_ir::Validated;
use orla_rules::ExecutionError;

use crate::{Backend, Body, KernelVersion, Runtime};

/// An execution handed to a thread; redeemed with [`Runtime::poll_run`]
/// or [`Runtime::wait_run`], once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Run(u64);

impl fmt::Display for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// What a run gives back: one Field per manifest output, as words.
type Outcome = Result<Vec<FieldData>, ExecutionError>;

struct Landed {
    run: Run,
    outcome: Outcome,
}

/// The runs in flight, and the ones back and waiting to be redeemed.
pub(crate) struct Runs {
    next: u64,
    landed: Sender<Landed>,
    arrivals: Receiver<Landed>,
    in_flight: BTreeSet<Run>,
    ready: BTreeMap<Run, Outcome>,
    discarded: BTreeSet<Run>,
    /// Budget words set aside for each run not yet redeemed or discarded.
    reserved: BTreeMap<Run, usize>,
}

impl Runs {
    pub(crate) fn new() -> Self {
        let (landed, arrivals) = channel();
        Self {
            next: 1,
            landed,
            arrivals,
            in_flight: BTreeSet::new(),
            ready: BTreeMap::new(),
            discarded: BTreeSet::new(),
            reserved: BTreeMap::new(),
        }
    }

    fn issue(&mut self) -> Run {
        let run = Run(self.next);
        self.next += 1;
        run
    }

    fn file(&mut self, landed: Landed) {
        self.in_flight.remove(&landed.run);
        if !self.discarded.remove(&landed.run) {
            self.ready.insert(landed.run, landed.outcome);
        }
    }

    fn collect(&mut self) {
        let arrived: Vec<Landed> = self.arrivals.try_iter().collect();
        for landed in arrived {
            self.file(landed);
        }
    }
}

impl Runtime {
    /// Starts executing a pinned version on its own thread and returns at
    /// once. Inputs are copied first, so the caller may release or overwrite
    /// them. A native operation is device work and runs now; its run is
    /// redeemed like any other.
    pub fn execute_later(
        &mut self,
        version: &Arc<KernelVersion>,
        invocation: &Invocation,
    ) -> Result<Run, ExecutionError> {
        let run = self.runs.issue();
        let Body::Kernel { module, .. } = &version.body else {
            let outcome = self
                .execute(version, invocation)
                .and_then(|outputs| self.take_words(outputs));
            self.runs.ready.insert(run, outcome);
            return Ok(run);
        };
        let room = self.reserve_for(module, invocation)?;
        self.runs.reserved.insert(run, room);
        let mut scratch = Host::new(room);
        scratch.set_work(self.host.work());
        let mut inputs = Vec::with_capacity(invocation.inputs.len());
        for handle in &invocation.inputs {
            let copied = self.materialize(*handle).and_then(|()| {
                let data = self.host.data(*handle).map_err(abi)?.clone();
                scratch.adopt(data).map_err(abi)
            });
            match copied {
                Ok(input) => inputs.push(input),
                Err(error) => {
                    self.give_back(run);
                    return Err(error);
                }
            }
        }
        let invocation = Invocation {
            inputs,
            ..invocation.clone()
        };
        let backend = self.shared_backend_for(module);
        let granted = self.scope_grants(version);
        let version = Arc::clone(version);
        let work = move || {
            run_on(
                backend.as_ref(),
                &version,
                &invocation,
                granted,
                &mut scratch,
            )
        };
        // The work is shared with the thread; where no thread can start
        // (WebAssembly) it is taken back and done here, landing before
        // this returns.
        let shared = std::sync::Arc::new(std::sync::Mutex::new(Some(work)));
        let theirs = std::sync::Arc::clone(&shared);
        let landed = self.runs.landed.clone();
        let started = std::thread::Builder::new()
            .name(format!("orla run {run}"))
            .spawn(move || {
                let work = theirs.lock().ok().and_then(|mut work| work.take());
                if let Some(work) = work {
                    let _ = landed.send(Landed {
                        run,
                        outcome: work(),
                    });
                }
            });
        if started.is_err() {
            let work = shared.lock().ok().and_then(|mut work| work.take());
            let outcome = work.map_or_else(
                || Err(ExecutionError::Device("the run was lost".into())),
                |work| work(),
            );
            self.runs.ready.insert(run, outcome);
            return Ok(run);
        }
        self.runs.in_flight.insert(run);
        Ok(run)
    }

    /// Takes a landed run's outputs as Fields of this Host, or `None`
    /// while it is still going. A run is redeemed once.
    pub fn poll_run(&mut self, run: Run) -> Option<Result<Vec<Handle>, ExecutionError>> {
        self.runs.collect();
        let outcome = self.runs.ready.remove(&run)?;
        self.give_back(run);
        Some(outcome.and_then(|fields| self.adopt_all(fields)))
    }

    /// Blocks until `run` lands and takes its outputs; other runs that
    /// land meanwhile wait for their own poll.
    pub fn wait_run(&mut self, run: Run) -> Result<Vec<Handle>, ExecutionError> {
        loop {
            self.runs.collect();
            if let Some(outcome) = self.runs.ready.remove(&run) {
                self.give_back(run);
                return outcome.and_then(|fields| self.adopt_all(fields));
            }
            if !self.runs.in_flight.contains(&run) {
                return Err(ExecutionError::Device(format!(
                    "run {run} is not in flight"
                )));
            }
            match self.runs.arrivals.recv() {
                Ok(landed) => self.runs.file(landed),
                Err(_) => {
                    return Err(ExecutionError::Device(format!("run {run} was lost")));
                }
            }
        }
    }

    /// Forgets a run whose outputs are unwanted; what it lands is dropped.
    pub fn discard_run(&mut self, run: Run) {
        self.give_back(run);
        self.runs.collect();
        if self.runs.ready.remove(&run).is_none() && self.runs.in_flight.contains(&run) {
            self.runs.discarded.insert(run);
        }
    }

    /// How many runs are still going.
    pub fn runs_in_flight(&mut self) -> usize {
        self.runs.collect();
        self.runs.in_flight.len()
    }

    /// Sets aside in the Host what a run of `module` over `invocation` may
    /// hold: its copies of the inputs and its outputs twice over.
    fn reserve_for(
        &mut self,
        module: &Validated,
        invocation: &Invocation,
    ) -> Result<usize, ExecutionError> {
        let mut fields = Vec::new();
        let mut sizes = Vec::with_capacity(invocation.inputs.len());
        for handle in &invocation.inputs {
            let domain = self.host.domain(*handle).map_err(abi)?;
            fields.push((self.host.schema(*handle).map_err(abi)?.clone(), domain));
            sizes.push(domain);
        }
        let domain = orla_rules::output_domain(module.manifest.mode, invocation.domain, &sizes);
        for port in &module.manifest.outputs {
            fields.push((port.ty.schema.clone(), domain));
            fields.push((port.ty.schema.clone(), domain));
        }
        let fields: Vec<_> = fields
            .iter()
            .map(|(schema, domain)| (schema, *domain))
            .collect();
        self.host.reserve(&fields).map_err(abi)
    }

    /// Gives back the room a run was set aside, once.
    fn give_back(&mut self, run: Run) {
        if let Some(room) = self.runs.reserved.remove(&run) {
            self.host.unreserve(room);
        }
    }

    /// The backend a module executes on, shared with a thread.
    fn shared_backend_for(&self, module: &Validated) -> Arc<dyn Backend> {
        match &self.ordered {
            Some(ordered) if module.manifest.mode == orla_ir::Mode::Scan => Arc::clone(ordered),
            _ => Arc::clone(&self.backend),
        }
    }

    /// The words of Fields of this Host, which are released.
    fn take_words(&mut self, outputs: Vec<Handle>) -> Outcome {
        let mut fields = Vec::with_capacity(outputs.len());
        let mut failed = None;
        for handle in &outputs {
            if failed.is_none() {
                match self
                    .materialize(*handle)
                    .and_then(|()| self.host.data(*handle).cloned().map_err(abi))
                {
                    Ok(data) => fields.push(data),
                    Err(error) => failed = Some(error),
                }
            }
            let _ = self.host.release(*handle);
        }
        match failed {
            Some(error) => Err(error),
            None => Ok(fields),
        }
    }

    /// Fields of this Host from landed words; nothing is kept on failure.
    fn adopt_all(&mut self, fields: Vec<FieldData>) -> Result<Vec<Handle>, ExecutionError> {
        let mut handles = Vec::with_capacity(fields.len());
        for data in fields {
            match self.host.adopt(data) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    for handle in handles {
                        let _ = self.host.release(handle);
                    }
                    return Err(abi(error));
                }
            }
        }
        Ok(handles)
    }
}

/// On the run's thread: executes over the scratch Host and brings every
/// output to words, downloading resident ones.
fn run_on(
    backend: &dyn Backend,
    version: &KernelVersion,
    invocation: &Invocation,
    granted: orla_abi::Capabilities,
    host: &mut Host,
) -> Outcome {
    let Body::Kernel { run, prepared, .. } = &version.body else {
        return Err(ExecutionError::Device(
            "a native operation runs on the frame".into(),
        ));
    };
    let outputs = {
        let mut scope = Scope::new(host, granted);
        backend.execute(prepared.as_ref(), run, invocation, &mut scope)
    };
    let outputs = crate::held_to(version.manifest(), host, outputs)?;
    outputs
        .into_iter()
        .map(|handle| {
            if let Some(device) = host.resident(handle).map_err(abi)? {
                let count = host.domain(handle).map_err(abi)?.count();
                let byte_len = host.schema(handle).map_err(abi)?.storage_words(count) * 4;
                let bytes = backend
                    .download(device, byte_len)
                    .map_err(ExecutionError::Device)?;
                let words = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                host.materialize(handle, words).map_err(abi)?;
            }
            host.data(handle).cloned().map_err(abi)
        })
        .collect()
}

fn abi(error: orla_abi::HostError) -> ExecutionError {
    ExecutionError::Abi(error.into())
}
