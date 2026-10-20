//! Preparing Kernels while frames continue: on worker threads, one core
//! left to the frames, or, where no thread can start (a page), queued and
//! prepared a few at a time between frames ([`Preparer::turn`]), so one
//! long preparation never holds the page, or its device, at once. Each
//! preparation's result is collected once by its ticket.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use entelechy_ir::Validated;
use entelechy_rules::Backend;

use crate::{Ahead, Candidate, PrepareError, Progress, Ticket, prepare_finished};

/// A module handed to the worker.
struct Job {
    ticket: Ticket,
    module: Validated,
    source: String,
    /// Where it is made ready before it is ready.
    ahead: Ahead,
}

/// What the worker made of one.
struct Done {
    ticket: Ticket,
    result: Result<Candidate, PrepareError>,
}

/// The threads that prepare modules while frames keep going, each taking
/// the next module handed to them.
struct Worker {
    jobs: Sender<Job>,
    done: Receiver<Done>,
    handles: Vec<JoinHandle<()>>,
}

/// Where no thread can start: the backends and the jobs waiting their
/// turn, oldest first.
struct Turns {
    backend: Arc<dyn Backend>,
    ordered: Option<Arc<dyn Backend>>,
    progress: Progress,
    jobs: VecDeque<Job>,
}

/// The worker, if any (or the turns where no thread can start),
/// uncollected results, and tickets whose results are unwanted.
#[derive(Default)]
pub(crate) struct Preparer {
    worker: Option<Worker>,
    turns: Option<Turns>,
    next_ticket: u64,
    ready: BTreeMap<Ticket, Result<Candidate, PrepareError>>,
    discarded: BTreeSet<Ticket>,
}

impl Preparer {
    /// Starts the workers, which prepare on `backend`, or on `ordered` what
    /// `backend` declines: as many as there are cores but one, which the
    /// frames keep. Where no thread can start (WebAssembly), the jobs are
    /// queued and prepared in turns ([`Preparer::turn`]).
    pub(crate) fn start(
        &mut self,
        backend: Arc<dyn Backend>,
        ordered: Option<Arc<dyn Backend>>,
        progress: Progress,
    ) {
        let (jobs, inbox) = channel::<Job>();
        let (outbox, done) = channel::<Done>();
        let inbox = Arc::new(Mutex::new(inbox));
        let mut handles = Vec::new();
        for k in 0..entelechy_rules::cores().saturating_sub(1).max(1) {
            let inbox = Arc::clone(&inbox);
            let outbox = outbox.clone();
            let backend = Arc::clone(&backend);
            let ordered = ordered.clone();
            let progress = progress.clone();
            let Ok(handle) = std::thread::Builder::new()
                .name(format!("entelechy-prepare-{k}"))
                // Room for the language's nesting bound
                // (entelechy_language::DEEP_STACK), which backends recurse through.
                .stack_size(256 << 20)
                .spawn(move || {
                    loop {
                        // The lock is held only to take the next job.
                        let job = inbox
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv();
                        let Ok(job) = job else {
                            break;
                        };
                        let result = prepare_finished(
                            backend.as_ref(),
                            ordered.as_deref(),
                            job.module,
                            job.source,
                            job.ahead,
                            &progress,
                        );
                        if outbox
                            .send(Done {
                                ticket: job.ticket,
                                result,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
            else {
                break;
            };
            handles.push(handle);
        }
        if handles.is_empty() {
            self.turns = Some(Turns {
                backend,
                ordered,
                progress,
                jobs: VecDeque::new(),
            });
            return;
        }
        self.worker = Some(Worker {
            jobs,
            done,
            handles,
        });
    }

    /// Whether the workers are running, or the turns are kept.
    pub(crate) fn running(&self) -> bool {
        self.worker.is_some() || self.turns.is_some()
    }

    /// Keeps turns instead of workers, as where no thread can start: what a
    /// page does, for a test to run elsewhere.
    pub(crate) fn start_turns(
        &mut self,
        backend: Arc<dyn Backend>,
        ordered: Option<Arc<dyn Backend>>,
        progress: Progress,
    ) {
        self.stop();
        self.turns = Some(Turns {
            backend,
            ordered,
            progress,
            jobs: VecDeque::new(),
        });
    }

    /// Prepares at most `most` queued jobs, oldest first, where no thread
    /// can start; their results are collected as a worker's are. The jobs
    /// still waiting.
    pub(crate) fn turn(&mut self, most: usize) -> usize {
        let Some(turns) = &mut self.turns else {
            return 0;
        };
        let mut made = Vec::new();
        for _ in 0..most {
            let Some(job) = turns.jobs.pop_front() else {
                break;
            };
            let result = prepare_finished(
                turns.backend.as_ref(),
                turns.ordered.as_deref(),
                job.module,
                job.source,
                job.ahead,
                &turns.progress,
            );
            made.push(Done {
                ticket: job.ticket,
                result,
            });
        }
        let left = turns.jobs.len();
        for done in made {
            self.file(done);
        }
        left
    }

    /// Whether it keeps turns rather than workers.
    pub(crate) fn in_turns(&self) -> bool {
        self.turns.is_some()
    }

    /// The jobs waiting their turn.
    pub(crate) fn queued(&self) -> usize {
        self.turns.as_ref().map_or(0, |turns| turns.jobs.len())
    }

    /// Stops the workers after the jobs in flight, keeping their results;
    /// the turns prepare what still waits.
    pub(crate) fn stop(&mut self) {
        while self.turn(usize::MAX) > 0 {}
        self.turns = None;
        if let Some(Worker {
            jobs,
            done,
            handles,
        }) = self.worker.take()
        {
            drop(jobs);
            for handle in handles {
                let _ = handle.join();
            }
            for finished in done.try_iter() {
                self.file(finished);
            }
        }
    }

    /// A new ticket.
    pub(crate) fn ticket(&mut self) -> Ticket {
        self.next_ticket += 1;
        Ticket(self.next_ticket)
    }

    /// Hands a module to the worker, or gives it back when there is none,
    /// for the caller to prepare inline and [`Preparer::finish`].
    pub(crate) fn send(
        &mut self,
        ticket: Ticket,
        module: Validated,
        source: String,
        ahead: Ahead,
    ) -> Option<(Validated, String)> {
        if let Some(turns) = &mut self.turns {
            turns.jobs.push_back(Job {
                ticket,
                module,
                source,
                ahead,
            });
            return None;
        }
        let Some(worker) = &self.worker else {
            return Some((module, source));
        };
        match worker.jobs.send(Job {
            ticket,
            module,
            source,
            ahead,
        }) {
            Ok(()) => None,
            Err(unsent) => Some((unsent.0.module, unsent.0.source)),
        }
    }

    /// Keeps the result of a preparation done inline.
    pub(crate) fn finish(&mut self, ticket: Ticket, result: Result<Candidate, PrepareError>) {
        self.ready.insert(ticket, result);
    }

    /// Takes a finished preparation, or `None` while it is still running.
    pub(crate) fn poll(&mut self, ticket: Ticket) -> Option<Result<Candidate, PrepareError>> {
        self.collect();
        self.ready.remove(&ticket)
    }

    /// Uncollected finished or forgotten preparations.
    pub(crate) fn uncollected(&self) -> usize {
        self.ready.len() + self.discarded.len()
    }

    /// Forgets a preparation whose result is unwanted.
    pub(crate) fn discard(&mut self, ticket: Ticket) {
        if let Some(turns) = &mut self.turns
            && let Some(at) = turns.jobs.iter().position(|job| job.ticket == ticket)
        {
            turns.jobs.remove(at);
            return;
        }
        self.collect();
        if self.ready.remove(&ticket).is_none() {
            self.discarded.insert(ticket);
        }
    }

    fn file(&mut self, finished: Done) {
        if !self.discarded.remove(&finished.ticket) {
            self.ready.insert(finished.ticket, finished.result);
        }
    }

    fn collect(&mut self) {
        let finished: Vec<Done> = match &self.worker {
            Some(worker) => worker.done.try_iter().collect(),
            None => Vec::new(),
        };
        for done in finished {
            self.file(done);
        }
    }
}

impl Drop for Preparer {
    fn drop(&mut self) {
        // What waits its turn is wanted by no one now: let go, not made.
        self.turns = None;
        self.stop();
    }
}
