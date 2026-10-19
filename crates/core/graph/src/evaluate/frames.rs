//! Frames in flight. An evaluator that overlaps
//! ([`Evaluator::set_overlap`]) commits a frame without waiting for its
//! device: the frame's checks, and a mark of its work, are waited on once
//! the next frame's work has gone to the device, or before anything reads
//! what the frame left ([`Evaluator::confirm`]). The processor makes the
//! next frame while the device draws this one, and at most two frames are
//! in flight. A frame found failed takes its commit back, and the commit of
//! the frame made from it: the state is what the frame before them left,
//! as though neither had run, and the failure is the answer of the call
//! that found it.

use super::*;

/// A frame committed and not yet confirmed.
pub(super) struct Unconfirmed {
    /// Its runs' checks, in run order.
    checks: Vec<(NodeId, Vec<Box<dyn entelechy_kernel::Check>>)>,
    /// Its work, all of it, finished on the device.
    mark: Option<Box<dyn entelechy_kernel::Check>>,
    /// What its commit replaced, port by port, held to be put back.
    replaced: Vec<((NodeId, String), Option<Handle>)>,
    /// The nodes it ran, whose outputs leave the cache if it failed.
    ran: Vec<NodeId>,
}

impl Evaluator {
    /// Whether later evaluations overlap: each commits without waiting for
    /// its device, and is confirmed once the next has gone to the device or
    /// before anything reads it. Turning it off confirms the frame in
    /// flight first.
    pub fn set_overlap(&mut self, on: bool, runtime: &mut Runtime) -> Result<(), EvaluateError> {
        if !on {
            self.confirm(runtime)?;
        }
        self.overlap = on;
        Ok(())
    }

    /// The state the frame in flight replaced, held to be put back should
    /// it fail: each port and the Field it held.
    pub(super) fn held_back_ports(&self) -> Vec<((NodeId, String), Handle)> {
        self.unconfirmed.as_ref().map_or_else(Vec::new, |frame| {
            frame
                .replaced
                .iter()
                .filter_map(|(port, held)| held.map(|held| (port.clone(), held)))
                .collect()
        })
    }

    /// Whether evaluations overlap ([`Evaluator::set_overlap`]).
    pub fn overlaps(&self) -> bool {
        self.overlap
    }

    /// Waits for the frame committed and not yet confirmed, if any. A
    /// failed one has its commit taken back, its outputs leave the cache,
    /// and its failure is the answer.
    pub fn confirm(&mut self, runtime: &mut Runtime) -> Result<(), EvaluateError> {
        match self.unconfirmed.take() {
            Some(frame) => self.settle_frame(frame, runtime),
            None => Ok(()),
        }
    }

    /// How many frames answered as committed were taken back since last
    /// asked: a frame found failed is one, the frame made from it had its
    /// failure as its answer.
    pub fn take_taken_back(&mut self) -> u64 {
        std::mem::take(&mut self.taken_back)
    }

    /// Whether `handle` is state the frame in flight replaced, held to be
    /// put back.
    pub(super) fn held_back(&self, handle: Handle) -> bool {
        self.unconfirmed
            .as_ref()
            .is_some_and(|frame| frame.replaced.iter().any(|(_, held)| *held == Some(handle)))
    }

    /// Commits the frame just made without waiting for its device, holding
    /// what it replaced; then confirms the frame before it, whose work went
    /// to the device a frame ago. When that one failed, both commits are
    /// taken back and its failure is the answer.
    pub(super) fn commit_in_flight(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        checks: Vec<(NodeId, Vec<Box<dyn entelechy_kernel::Check>>)>,
        ran: Vec<NodeId>,
    ) -> Result<Vec<(NodeId, String)>, EvaluateError> {
        let (advanced, replaced) = self.commit_holding(graph)?;
        let frame = Unconfirmed {
            checks,
            mark: runtime.submitted(),
            replaced,
            ran,
        };
        match self.unconfirmed.replace(frame) {
            Some(before) => self.settle_frame(before, runtime).map(|()| advanced),
            None => Ok(advanced),
        }
    }

    /// Waits on `frame`'s checks in run order, then on its work. Confirmed,
    /// the state it replaced is let go; failed, the frame in flight after it
    /// (made from it) and then it are taken back, and the first failure is
    /// the answer.
    fn settle_frame(
        &mut self,
        frame: Unconfirmed,
        runtime: &mut Runtime,
    ) -> Result<(), EvaluateError> {
        let Unconfirmed {
            checks,
            mark,
            replaced,
            ran,
        } = frame;
        let mut failed: Option<(NodeId, entelechy_kernel::ExecutionError)> = None;
        for (node, node_checks) in checks {
            for check in node_checks {
                if let Err(error) = check.wait()
                    && failed.is_none()
                {
                    failed = Some((node.clone(), error));
                }
            }
        }
        if let Some(mark) = mark
            && let Err(error) = mark.wait()
            && failed.is_none()
        {
            failed = ran.last().cloned().map(|node| (node, error));
        }
        let Some((node, error)) = failed else {
            for (_, held) in replaced {
                if let Some(held) = held
                    && !self.ledger.is_cached(held)
                    && !self.is_kept(held)
                {
                    release(runtime, held);
                }
            }
            return Ok(());
        };
        // The frame made from it goes first, then it: the state is put
        // back as each commit found it, newest first.
        let mut gone = ran;
        if let Some(after) = self.unconfirmed.take() {
            self.put_back(after.replaced, runtime);
            gone.extend(after.ran);
        }
        self.put_back(replaced, runtime);
        for id in gone {
            self.drop_cache(&id, runtime);
        }
        // Remembered runs may hold what the taken-back frames made.
        self.forget_runs(runtime);
        self.taken_back += 1;
        Err(EvaluateError::Execution { node, error })
    }

    /// Puts back what a commit replaced, last port first, letting go of
    /// what it had committed.
    fn put_back(
        &mut self,
        replaced: Vec<((NodeId, String), Option<Handle>)>,
        runtime: &mut Runtime,
    ) {
        for (key, previous) in replaced.into_iter().rev() {
            let committed = match previous {
                Some(previous) => self.ledger.commit_put(key, previous),
                None => self.ledger.commit_take(&key),
            };
            if let Some(committed) = committed
                && Some(committed) != previous
                && !self.ledger.is_cached(committed)
                && !self.is_kept(committed)
            {
                release(runtime, committed);
            }
        }
    }

    /// Lets go of the frame in flight without waiting for it, its held
    /// state with it: for [`Evaluator::clear`], which releases everything.
    pub(super) fn drop_in_flight(
        &mut self,
        released: &mut BTreeSet<Handle>,
        runtime: &mut Runtime,
    ) {
        if let Some(frame) = self.unconfirmed.take() {
            for (_, held) in frame.replaced {
                if let Some(held) = held
                    && released.insert(held)
                {
                    release(runtime, held);
                }
            }
        }
    }
}
