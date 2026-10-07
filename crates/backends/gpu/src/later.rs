//! What the device is waited for later: every run submitted so far, and
//! downloads queued behind them.

use super::*;

/// A download queued behind the work before it:
/// [`Backend::download_later`].
pub(crate) struct Coming {
    pub(crate) wait: Wait,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) staging: wgpu::Buffer,
    pub(crate) byte_len: usize,
    /// The copy's submission: taking waits for it alone.
    pub(crate) after: wgpu::SubmissionIndex,
}

impl orla_rules::Downloading for Coming {
    fn take(self: Box<Self>) -> Result<Vec<u8>, String> {
        let slice = self.staging.slice(..);
        self.wait
            .map_read_after(&self.device, &self.queue, slice, &self.after)?;
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
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) after: wgpu::SubmissionIndex,
}

impl Check for Submitted {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        self.wait
            .done_after(&self.device, &self.queue, &self.after)
            .map_err(ExecutionError::Device)
    }
}
