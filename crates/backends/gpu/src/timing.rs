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

/// Frames clocked as they run: each run's submission and the moment the
/// device tells it done, kept by frame; nothing waits for them. The device
/// runs its queue in order, so a run kept it busy from its submission, or
/// the run before's end if later, to its own end: a frame's busy time is
/// its runs' summed. A waiter on its own thread hears the device as soon as
/// it is done, so the ends are told when they happen, not when next asked.
pub(crate) struct Frames {
    on: Arc<std::sync::atomic::AtomicBool>,
    /// Runs told submitted and not yet done, which the waiter waits on.
    pending: Arc<std::sync::atomic::AtomicUsize>,
    /// Whether a waiter runs.
    waiting: Arc<std::sync::atomic::AtomicBool>,
    /// The runs since the last frame ended: each its submission and end.
    open: Mutex<Vec<Run>>,
    /// Ended frames whose runs are not all done, oldest first, by number.
    ended: Mutex<std::collections::VecDeque<(u64, Vec<Run>)>>,
    /// The frames ended since clocking began.
    frames: std::sync::atomic::AtomicU64,
    /// The end of the last run counted, which a run after it starts from
    /// at the earliest.
    last_end: Mutex<Option<std::time::Instant>>,
    /// The latest frame done.
    last: Mutex<Option<orla_rules::ClockedFrame>>,
}

type Run = Arc<Mutex<(std::time::Instant, Option<std::time::Instant>)>>;

impl Default for Frames {
    fn default() -> Self {
        Self {
            on: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            waiting: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            open: Mutex::new(Vec::new()),
            ended: Mutex::new(std::collections::VecDeque::new()),
            frames: std::sync::atomic::AtomicU64::new(0),
            last_end: Mutex::new(None),
            last: Mutex::new(None),
        }
    }
}

/// At most this many ended frames wait for their runs; an older one is
/// let go uncounted.
const FRAMES_WAITING: usize = 8;

impl Frames {
    pub(crate) fn clocked(&self) -> bool {
        self.on.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clocks frames from now, or no longer; on, a waiter hears the device.
    pub(crate) fn set_clocked(&self, on: bool, device: &wgpu::Device) {
        self.on.store(on, std::sync::atomic::Ordering::Relaxed);
        if !on {
            lock(&self.open).clear();
            lock(&self.ended).clear();
            *lock(&self.last_end) = None;
            *lock(&self.last) = None;
            return;
        }
        if cfg!(target_arch = "wasm32")
            || self
                .waiting
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let device = device.clone();
        let on = Arc::clone(&self.on);
        let pending = Arc::clone(&self.pending);
        let waiting = Arc::clone(&self.waiting);
        let _ = std::thread::Builder::new()
            .name("orla device clock".into())
            .spawn(move || {
                while on.load(std::sync::atomic::Ordering::Relaxed) {
                    if pending.load(std::sync::atomic::Ordering::Relaxed) > 0 {
                        let _ = device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(std::time::Duration::from_millis(5)),
                        });
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(500));
                    }
                }
                waiting.store(false, std::sync::atomic::Ordering::Relaxed);
            });
    }

    /// A run just submitted, work or copy, for the open frame: its end is
    /// told when the device has done it.
    pub(crate) fn submitted(&self, queue: &wgpu::Queue) {
        if !self.clocked() {
            return;
        }
        let run: Run = Arc::new(Mutex::new((std::time::Instant::now(), None)));
        lock(&self.open).push(Arc::clone(&run));
        let pending = Arc::clone(&self.pending);
        pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        queue.on_submitted_work_done(move || {
            lock(&run).1 = Some(std::time::Instant::now());
            pending.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        });
    }

    /// How many runs the open frame has been given.
    pub(crate) fn mark(&self) -> usize {
        lock(&self.open).len()
    }

    /// Ends the open frame, answering its number; `None` unclocked.
    pub(crate) fn end_frame(&self) -> Option<u64> {
        if !self.clocked() {
            return None;
        }
        let number = self
            .frames
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let runs = std::mem::take(&mut *lock(&self.open));
        let mut ended = lock(&self.ended);
        ended.push_back((number, runs));
        while ended.len() > FRAMES_WAITING {
            ended.pop_front();
        }
        Some(number)
    }

    /// The latest frame whose runs are all done, each run's busy time.
    pub(crate) fn done(&self) -> Option<orla_rules::ClockedFrame> {
        let mut ended = lock(&self.ended);
        let mut last_end = lock(&self.last_end);
        while let Some((number, runs)) = ended.front() {
            let told: Option<Vec<(std::time::Instant, std::time::Instant)>> = runs
                .iter()
                .map(|run| {
                    let r = lock(run);
                    r.1.map(|end| (r.0, end))
                })
                .collect();
            let Some(told) = told else { break };
            // The device runs its queue in order: each run kept it busy
            // from its submission, or the run before's end if later.
            let mut busy = Vec::with_capacity(told.len());
            for (submitted, end) in told {
                let from = last_end.map_or(submitted, |before| submitted.max(before));
                busy.push(end.saturating_duration_since(from));
                *last_end = Some(end.max(last_end.unwrap_or(end)));
            }
            *lock(&self.last) = Some(orla_rules::ClockedFrame {
                frame: *number,
                runs: busy,
            });
            ended.pop_front();
        }
        lock(&self.last).clone()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
