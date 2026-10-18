//! What the device is waited for later: every run submitted so far, and
//! downloads queued behind them.

use super::*;

/// A download queued behind the work before it:
/// [`Backend::download_later`].
pub(crate) struct Coming {
    pub(crate) wait: Wait,
    pub(crate) spent: Spent,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) staging: wgpu::Buffer,
    pub(crate) byte_len: usize,
    /// The copy's submission: taking waits for it alone.
    pub(crate) after: wgpu::SubmissionIndex,
    /// The copy's readback, asked for as it was submitted where the device
    /// answers only for everything before ([`Wait::ask_map`]).
    pub(crate) asked: Option<crate::wait::Asked<Result<(), wgpu::BufferAsyncError>>>,
    /// What the device said by itself ([`crate::told`]).
    pub(crate) heard: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl entelechy_rules::Downloading for Coming {
    fn take(mut self: Box<Self>) -> Result<Vec<u8>, String> {
        let asked = self.asked.take();
        let slice = self.staging.slice(..);
        self.spent
            .time(|| match asked {
                Some(asked) => self
                    .wait
                    .answer(&self.device, asked)?
                    .map_err(|error| error.to_string()),
                None => self
                    .wait
                    .map_read_after(&self.device, &self.queue, slice, &self.after),
            })
            .map_err(|error| crate::told(&self.heard, error))?;
        let bytes = slice
            .get_mapped_range()
            .map_err(|error| error.to_string())?[..self.byte_len]
            .to_vec();
        self.staging.unmap();
        Ok(bytes)
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
}

impl Check for Submitted {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let asked = self.asked.take();
        self.spent
            .time(|| match asked {
                Some(asked) => self.wait.answer(&self.device, asked),
                None => self.wait.done_after(&self.device, &self.queue, &self.after),
            })
            .map_err(ExecutionError::Device)
    }
}
