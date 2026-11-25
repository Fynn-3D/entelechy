//! What the device is waited for later: every run submitted so far, and
//! downloads queued behind them.

use super::*;

/// A download queued behind the work before it:
/// [`Backend::download_later`].
pub(crate) struct Coming {
    pub(crate) wait: Wait,
    pub(crate) spent: Spent,
    pub(crate) device: wgpu::Device,
    pub(crate) staging: Option<wgpu::Buffer>,
    pub(crate) byte_len: usize,
    /// The copy's submission: taking waits for it alone.
    pub(crate) after: wgpu::SubmissionIndex,
    /// The copy's readback, asked for as it was submitted ([`Wait::ask_map`]).
    pub(crate) asked: Option<crate::wait::Asked<Result<(), wgpu::BufferAsyncError>>>,
    /// What the device said by itself ([`crate::told`]).
    pub(crate) heard: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    pub(crate) page_queue: crate::PageQueue,
}

impl Drop for Coming {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            crate::wait::discard(staging);
        }
    }
}

impl entelechy_rules::Downloading for Coming {
    fn take(mut self: Box<Self>) -> Result<Vec<u32>, String> {
        let asked = self.asked.take().expect("asked on submission");
        let staging = self.staging.take().expect("staging buffer held");
        let slice = staging.slice(..);
        self.spent
            .time(|| {
                self.wait
                    .answer_after(&self.device, &self.after, asked)?
                    .map_err(|error| error.to_string())
            })
            .map_err(|error| crate::told(&self.heard, error))?;
        self.page_queue.clear();
        let words = crate::words_of(
            &slice
                .get_mapped_range()
                .map_err(|error| error.to_string())?[..self.byte_len],
        );
        staging.unmap();
        crate::wait::discard(staging);
        Ok(words)
    }
}

/// Every run submitted before it, finished: [`Backend::submitted`].
pub(crate) struct Submitted {
    pub(crate) wait: Wait,
    pub(crate) spent: Spent,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) after: wgpu::SubmissionIndex,
    /// The end of the work, asked for as it was marked where the device
    /// answers only for everything before ([`Wait::ask_done`]).
    pub(crate) asked: Option<crate::wait::Asked<()>>,
    pub(crate) page_queue: crate::PageQueue,
    pub(crate) ns_per_mop: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub(crate) pending: Vec<Box<dyn Check>>,
}

impl Check for Submitted {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let asked = self.asked.take();
        let waited = self
            .spent
            .time(|| match asked {
                Some(asked) => self.wait.answer(&self.device, asked),
                None => self.wait.done_after(&self.device, &self.queue, &self.after),
            })
            .map_err(ExecutionError::Device);
        if self.wait.parks_with_host() {
            let (queued_ops, elapsed_ns) = self.page_queue.take_timing();
            crate::record_page_rate(&self.ns_per_mop, queued_ops, elapsed_ns);
        } else {
            self.page_queue.clear();
        }
        let mut first = waited;
        for check in self.pending {
            let answer = check.wait();
            if first.is_ok() {
                first = answer;
            }
        }
        first
    }
}

/// A deferred error already surfaced when flushing an open batch at frame end.
pub(crate) struct ImmediateError(pub(crate) ExecutionError);

impl Check for ImmediateError {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        Err(self.0)
    }
}
