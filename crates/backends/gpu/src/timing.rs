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

/// Where a run writes the device's clock at its passes' edges: a query set
/// and the run's first stamp in it; index 0 opens the run's first pass, the
/// last index closes its last.
pub(crate) struct Timing {
    set: wgpu::QuerySet,
    base: u32,
}

impl Timing {
    /// A compute pass's stamps, at the run's indices `begin` and `end`.
    pub(crate) fn compute(
        &self,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: begin.map(|at| self.base + at),
            end_of_pass_write_index: end.map(|at| self.base + at),
        }
    }

    /// A render pass's stamps, as [`Timing::compute`].
    pub(crate) fn render(
        &self,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: begin.map(|at| self.base + at),
            end_of_pass_write_index: end.map(|at| self.base + at),
        }
    }
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
    /// A page's clocked frame times each run by such a set too ([`Frames`]).
    pub(crate) fn timing_set(&self, count: u32) -> Option<Timing> {
        if !self.measured() && self.frames.clocked() && self.frames.stamps() {
            return self.frames.run_timing(&self.device);
        }
        let worth_asking = self.clock.load(std::sync::atomic::Ordering::Relaxed) != clock::SILENT;
        (self.measured() && self.timestamps && worth_asking).then(|| Timing {
            set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("entelechy timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            base: 0,
        })
    }

    /// Copies a call's timestamps to Host-readable memory, in the same
    /// encoder, with no submission of its own.
    pub(crate) fn resolve_timing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timing: Option<&Timing>,
        count: u32,
    ) -> Option<wgpu::Buffer> {
        if !self.measured() && self.frames.clocked() && self.frames.stamps() {
            if timing.is_some() {
                self.frames.stamped_run(count);
            }
            return None;
        }
        timing.map(|Timing { set, .. }| {
            let bytes = u64::from(count) * 8;
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy timing resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy timing read"),
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
                    label: Some("entelechy floor"),
                });
            let started = entelechy_rules::clock::now();
            self.queue.submit(Some(encoder.finish()));
            let _ = self.wait.drain(&self.device, &self.queue);
            least = least.min(entelechy_rules::clock::since(started));
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

/// Frames clocked as they run: each run's busy time, kept by frame; nothing
/// waits for them.
///
/// Where the device has timestamps its own clock times each run, by the
/// stamps a measured run writes at its passes' edges
/// ([`GpuBackend::timing_set`]), resolved in the run and read back once the
/// device has done the frame: a run's work, apart from the device starting
/// again after a pause, which the Host's clock would count as the first
/// run's after it. A run with no pass (a copy) counts as none.
///
/// Elsewhere the Host's clock times a run from its submission, or the run
/// before's end if later, to the moment the device tells it done: the
/// device runs its queue in order, so that is how long it kept the device
/// busy. The backend's waiter hears each run done as it is, within a look
/// ([`crate::wait::Waiter`]), so the ends are told when they happen, not
/// when next asked. A page has no thread to
/// hear the device on, so a device of a page's without timestamps clocks
/// no frame.
pub(crate) struct Frames {
    on: Arc<std::sync::atomic::AtomicBool>,
    /// Whether the device's own stamps time the runs ([`Frames::stamps`]).
    stamps: std::sync::atomic::AtomicBool,
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
    last: Mutex<Option<entelechy_rules::ClockedFrame>>,
    /// A page's frames, timed by the device's stamps.
    stamped: Mutex<Stamped>,
}

type Run = Arc<Mutex<(std::time::Instant, Option<std::time::Instant>)>>;

/// The runs a page's frame stamps; later runs in it are counted as none.
const STAMPED_RUNS: u32 = 512;

impl Default for Frames {
    fn default() -> Self {
        Self {
            on: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stamps: std::sync::atomic::AtomicBool::new(false),
            open: Mutex::new(Vec::new()),
            ended: Mutex::new(std::collections::VecDeque::new()),
            frames: std::sync::atomic::AtomicU64::new(0),
            last_end: Mutex::new(None),
            last: Mutex::new(None),
            stamped: Mutex::new(Stamped::default()),
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

    /// Whether the runs are timed by the stamps the device writes at their
    /// passes' edges: the device's own clock, which tells a run's work
    /// apart from the device starting again after a pause. The Host's
    /// clock hears only when a run is done.
    pub(crate) fn stamps(&self) -> bool {
        self.stamps.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clocks frames from now, or no longer: by the device's stamps where
    /// it has `timestamps`, else by when the backend's waiter hears it tell
    /// each run done ([`crate::wait::Waiter`]), which a page cannot.
    pub(crate) fn set_clocked(&self, on: bool, timestamps: bool) {
        let on = on && (!crate::PAGE || timestamps);
        self.on.store(on, std::sync::atomic::Ordering::Relaxed);
        self.stamps
            .store(timestamps, std::sync::atomic::Ordering::Relaxed);
        if !on {
            lock(&self.open).clear();
            lock(&self.ended).clear();
            *lock(&self.last_end) = None;
            *lock(&self.last) = None;
            *lock(&self.stamped) = Stamped::default();
        }
    }

    /// Gives the device a run, work or copy, counted for the open frame.
    pub(crate) fn submit(
        &self,
        queue: &wgpu::Queue,
        commands: wgpu::CommandBuffer,
    ) -> wgpu::SubmissionIndex {
        if !self.clocked() {
            return queue.submit(Some(commands));
        }
        if self.stamps() {
            return lock(&self.stamped).submit(queue, commands);
        }
        let submission = queue.submit(Some(commands));
        let run: Run = Arc::new(Mutex::new((std::time::Instant::now(), None)));
        lock(&self.open).push(Arc::clone(&run));
        queue.on_submitted_work_done(move || {
            lock(&run).1 = Some(std::time::Instant::now());
        });
        submission
    }

    /// How many runs the open frame has been given.
    pub(crate) fn mark(&self) -> usize {
        if self.stamps() {
            return lock(&self.stamped).runs();
        }
        lock(&self.open).len()
    }

    /// Ends the open frame, answering its number; `None` unclocked.
    pub(crate) fn end_frame(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<u64> {
        if !self.clocked() {
            return None;
        }
        let number = self
            .frames
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.stamps() {
            lock(&self.stamped).end_frame(number, device, queue);
            return Some(number);
        }
        let runs = std::mem::take(&mut *lock(&self.open));
        let mut ended = lock(&self.ended);
        ended.push_back((number, runs));
        while ended.len() > FRAMES_WAITING {
            ended.pop_front();
        }
        Some(number)
    }

    /// The latest frame whose runs are all done, each run's busy time.
    pub(crate) fn done(&self) -> Option<entelechy_rules::ClockedFrame> {
        if self.stamps() {
            if let Some(frame) = lock(&self.stamped).done() {
                *lock(&self.last) = Some(frame);
            }
            return lock(&self.last).clone();
        }
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
            *lock(&self.last) = Some(entelechy_rules::ClockedFrame {
                frame: *number,
                runs: busy,
            });
            ended.pop_front();
        }
        lock(&self.last).clone()
    }
}

/// Frames each run of which is timed by the stamps at its passes' edges,
/// all in one query set (a device keeps few): a page's, and a machine's
/// whose device writes them.
#[derive(Default)]
struct Stamped {
    /// The set every run of a frame writes its stamps into, from its base.
    set: Option<wgpu::QuerySet>,
    /// The open frame's runs: each its stamps' count, none for a run
    /// unstamped.
    counts: Vec<u32>,
    /// The stamps of the run being made, once resolved.
    resolving: Option<u32>,
    /// Ended frames being read back, oldest first: each its number, runs'
    /// counts and stamps; a frame that stamped nothing has none.
    ended: std::collections::VecDeque<(u64, Vec<u32>, Option<Stamps>)>,
    /// Stamps read and free for a later frame.
    spare: Vec<Stamps>,
    /// Nanoseconds a tick of the device's clock lasts.
    period: f64,
}

/// One frame's stamps, resolved and copied to be read.
struct Stamps {
    resolved: wgpu::Buffer,
    read: wgpu::Buffer,
    /// The read's mapping: `WAITING`, `MAPPED` or `FAILED`.
    state: Arc<std::sync::atomic::AtomicU8>,
}

const WAITING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// The stamps a run writes at most (a native operation's two passes).
const RUN_STAMPS: u32 = 4;

impl Stamps {
    fn new(device: &wgpu::Device) -> Self {
        let bytes = u64::from(STAMPED_RUNS * RUN_STAMPS) * 8;
        Self {
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame stamps resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entelechy frame stamps read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: Arc::new(std::sync::atomic::AtomicU8::new(WAITING)),
        }
    }
}

impl Frames {
    /// Where the open frame's next run writes its stamps; `None` past the
    /// runs a frame stamps.
    pub(crate) fn run_timing(&self, device: &wgpu::Device) -> Option<Timing> {
        let mut stamped = lock(&self.stamped);
        let at = stamped.counts.len() as u32;
        if at >= STAMPED_RUNS {
            return None;
        }
        let set = stamped.set.get_or_insert_with(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("entelechy frame stamps"),
                ty: wgpu::QueryType::Timestamp,
                count: STAMPED_RUNS * RUN_STAMPS,
            })
        });
        Some(Timing {
            set: set.clone(),
            base: at * RUN_STAMPS,
        })
    }

    /// The run being made wrote `count` stamps from its base.
    pub(crate) fn stamped_run(&self, count: u32) {
        lock(&self.stamped).resolving = Some(count.min(RUN_STAMPS));
    }
}

impl Stamped {
    fn submit(
        &mut self,
        queue: &wgpu::Queue,
        commands: wgpu::CommandBuffer,
    ) -> wgpu::SubmissionIndex {
        let count = self.resolving.take().unwrap_or(0);
        self.counts.push(count);
        queue.submit(Some(commands))
    }

    fn runs(&self) -> usize {
        self.counts.len()
    }

    fn end_frame(&mut self, number: u64, device: &wgpu::Device, queue: &wgpu::Queue) {
        let counts = std::mem::take(&mut self.counts);
        let stamped = (counts.len() as u32).min(STAMPED_RUNS) * RUN_STAMPS;
        let (Some(set), true) = (&self.set, counts.iter().any(|&count| count > 0)) else {
            self.ended.push_back((number, counts, None));
            return;
        };
        let stamps = self.spare.pop().unwrap_or_else(|| Stamps::new(device));
        let bytes = u64::from(stamped) * 8;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("entelechy frame stamps"),
        });
        encoder.resolve_query_set(set, 0..stamped, &stamps.resolved, 0);
        encoder.copy_buffer_to_buffer(&stamps.resolved, 0, &stamps.read, 0, bytes);
        queue.submit(Some(encoder.finish()));
        self.period = f64::from(queue.get_timestamp_period());
        stamps
            .state
            .store(WAITING, std::sync::atomic::Ordering::Relaxed);
        let state = Arc::clone(&stamps.state);
        stamps
            .read
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |mapped| {
                let to = if mapped.is_ok() { MAPPED } else { FAILED };
                state.store(to, std::sync::atomic::Ordering::Relaxed);
            });
        self.ended.push_back((number, counts, Some(stamps)));
        // A frame left waiting too long is let go uncounted.
        while self.ended.len() > FRAMES_WAITING {
            self.ended.pop_front();
        }
    }

    /// The latest frame read back, if one has been since last asked: each
    /// run its first stamp to its last.
    fn done(&mut self) -> Option<entelechy_rules::ClockedFrame> {
        let mut latest = None;
        while let Some((_, _, stamps)) = self.ended.front() {
            let state = stamps.as_ref().map_or(MAPPED, |s| {
                s.state.load(std::sync::atomic::Ordering::Relaxed)
            });
            if state == WAITING {
                break;
            }
            let Some((number, counts, stamps)) = self.ended.pop_front() else {
                break;
            };
            if state == FAILED {
                continue;
            }
            let mut busy = vec![std::time::Duration::ZERO; counts.len()];
            // Whether every stamp its runs wrote reads back: a device may
            // leave a frame's unwritten (the first it clocks, say), and a
            // frame some of whose runs say nothing is not told.
            let mut whole = true;
            if let Some(stamps) = stamps {
                let stamped = counts.len().min(STAMPED_RUNS as usize);
                let bytes = (stamped * RUN_STAMPS as usize * 8) as u64;
                if let Ok(read) = stamps.read.slice(..bytes).get_mapped_range() {
                    let tick = |at: usize| {
                        u64::from_le_bytes(read[at * 8..at * 8 + 8].try_into().unwrap_or([0; 8]))
                    };
                    for (run, (took, &count)) in
                        busy.iter_mut().zip(&counts).take(stamped).enumerate()
                    {
                        if count < 2 {
                            continue;
                        }
                        let base = run * RUN_STAMPS as usize;
                        let (first, last) = (tick(base), tick(base + count as usize - 1));
                        // A stamp left unwritten reads zero.
                        if first == 0 || last < first {
                            whole = false;
                            continue;
                        }
                        *took = std::time::Duration::from_nanos(
                            ((last - first) as f64 * self.period) as u64,
                        );
                    }
                }
                stamps.read.unmap();
                self.spare.push(stamps);
            }
            if whole {
                latest = Some(entelechy_rules::ClockedFrame {
                    frame: number,
                    runs: busy,
                });
            }
        }
        latest
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
