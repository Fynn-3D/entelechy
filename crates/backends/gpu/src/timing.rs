//! Device time: the device's own clock where it answers, else the call and
//! its wait, and the floor below which a time says nothing.

use super::*;

/// What asking the device's own clock has been worth; see
/// [`GpuBackend::device_clock_answered`].
pub(crate) mod clock {
    /// Nobody has asked yet.
    pub const UNTRIED: u8 = 0;
    /// The clock answered: a profile prints the device's time.
    pub const ANSWERED: u8 = 1;
    /// The driver resolved nothing: a profile prints the call and its wait,
    /// and nothing asks again.
    pub const SILENT: u8 = 2;
}

impl GpuBackend {
    /// Whether executions wait for the device; see
    /// [`Backend::set_measured`].
    pub fn measured(&self) -> bool {
        self.measured.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The device time of the last measured execution, by its own clock;
    /// `None` when nothing was measured or the adapter has no timestamps.
    /// Taken once.
    pub fn last_device_time(&self) -> Option<std::time::Duration> {
        let nanos = self
            .last_nanos
            .swap(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        (nanos != u64::MAX).then(|| std::time::Duration::from_nanos(nanos))
    }

    /// The query set a measured call writes its passes' edges into; `None`
    /// when nothing is measured or the adapter has no timestamps. Made before
    /// the encoder so the opening timestamp is its first command.
    pub(crate) fn timing_set(&self, count: u32) -> Option<wgpu::QuerySet> {
        let worth_asking = self.clock.load(std::sync::atomic::Ordering::Relaxed) != clock::SILENT;
        (self.measured() && self.timestamps && worth_asking).then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("orla timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            })
        })
    }

    /// Copies a call's timestamps to Host-readable memory, in the same
    /// encoder, with no submission of its own.
    pub(crate) fn resolve_timing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        set: Option<&wgpu::QuerySet>,
        count: u32,
    ) -> Option<wgpu::Buffer> {
        set.map(|set| {
            let bytes = u64::from(count) * 8;
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orla timing read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(set, 0..count, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, bytes);
            read
        })
    }

    /// Waits for a measured call and records its device time, first timestamp
    /// to last. An unmeasured run returns at once: one queue runs in order,
    /// so a later node's submission needs no wait.
    pub(crate) fn settle_measured(
        &self,
        submission: wgpu::SubmissionIndex,
        timing: Option<wgpu::Buffer>,
    ) {
        let _ = submission;
        if !self.measured() {
            let _ = self.device.poll(wgpu::PollType::Poll);
            return;
        }
        let _ = self.wait.drain(&self.device, &self.queue);
        let Some(read) = timing else { return };
        if self
            .wait
            .map_read(&self.device, &self.queue, read.slice(..))
            .is_ok()
            && let Ok(bytes) = read.slice(..).get_mapped_range()
        {
            let ticks =
                |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
            let (begin, end) = (ticks(0), ticks(bytes.len() - 8));
            let period = self.queue.get_timestamp_period() as f64;
            drop(bytes);
            read.unmap();
            // A driver may take the queries and write nothing: then the wait is
            // the answer.
            let (answer, kind) = if end > begin {
                (((end - begin) as f64 * period) as u64, clock::ANSWERED)
            } else {
                (u64::MAX, clock::SILENT)
            };
            if answer != u64::MAX {
                self.last_nanos
                    .store(answer, std::sync::atomic::Ordering::Relaxed);
            }
            self.clock.store(kind, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The cost of a measured execution's wait with nothing to wait for: one
    /// empty submission polled to completion, the least of sixteen. Every
    /// node a profile times carries it before its own work.
    pub fn wait_floor(&self) -> std::time::Duration {
        let mut least = std::time::Duration::MAX;
        for _ in 0..16 {
            let encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("orla floor"),
                });
            let started = std::time::Instant::now();
            self.queue.submit(Some(encoder.finish()));
            let _ = self.wait.drain(&self.device, &self.queue);
            least = least.min(started.elapsed());
        }
        least
    }

    /// Whether the device's own clock has answered a measured execution: set
    /// the first time a resolve returns two different numbers, since a driver
    /// may take the queries and resolve zeros. Until then a measured run's
    /// numbers are the call and its wait, and whatever prints them says so.
    pub fn device_clock_answered(&self) -> bool {
        self.clock.load(std::sync::atomic::Ordering::Relaxed) == clock::ANSWERED
    }
}
