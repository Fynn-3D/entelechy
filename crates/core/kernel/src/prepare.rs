//! Preparing Kernels while frames continue: on a worker thread, or inline
//! where no thread can start. Each preparation's result is collected once
//! by its ticket.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use orla_ir::Validated;
use orla_rules::Backend;

use crate::{Candidate, PrepareError, Ticket, prepare_on};

/// A module handed to the worker.
struct Job {
    ticket: Ticket,
    module: Validated,
    source: String,
}

/// What the worker made of one.
struct Done {
    ticket: Ticket,
    result: Result<Candidate, PrepareError>,
}

/// The thread that prepares modules while frames keep going.
struct Worker {
    jobs: Sender<Job>,
    done: Receiver<Done>,
    handle: JoinHandle<()>,
}

/// The worker, if any, uncollected results, and tickets whose results
/// are unwanted.
#[derive(Default)]
pub(crate) struct Preparer {
    worker: Option<Worker>,
    next_ticket: u64,
    ready: BTreeMap<Ticket, Result<Candidate, PrepareError>>,
    discarded: BTreeSet<Ticket>,
}

impl Preparer {
    /// Starts the worker, which prepares on `backend`, or on `ordered` what
    /// `backend` declines. Where no thread can start (WebAssembly), every
    /// preparation is inline.
    pub(crate) fn start(&mut self, backend: Arc<dyn Backend>, ordered: Option<Arc<dyn Backend>>) {
        let (jobs, inbox) = channel::<Job>();
        let (outbox, done) = channel::<Done>();
        let Ok(handle) = std::thread::Builder::new()
            .name("orla-prepare".into())
            // Room for the language's nesting bound
            // (orla_language::DEEP_STACK), which backends recurse through.
            .stack_size(256 << 20)
            .spawn(move || {
                for job in inbox {
                    let result =
                        prepare_on(backend.as_ref(), ordered.as_deref(), job.module, job.source);
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
            return;
        };
        self.worker = Some(Worker { jobs, done, handle });
    }

    /// Whether a worker is running.
    pub(crate) fn running(&self) -> bool {
        self.worker.is_some()
    }

    /// Stops the worker after the jobs in flight, keeping their results.
    pub(crate) fn stop(&mut self) {
        if let Some(Worker { jobs, done, handle }) = self.worker.take() {
            drop(jobs);
            let _ = handle.join();
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
    ) -> Option<(Validated, String)> {
        let Some(worker) = &self.worker else {
            return Some((module, source));
        };
        match worker.jobs.send(Job {
            ticket,
            module,
            source,
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
        self.stop();
    }
}
