//! Waiting for the device. On a machine the thread blocks. In a page the
//! browser answers only when the page's script has a turn, so waiting
//! parks Entelechy ([`Wait::parks`]): the host suspends it (JavaScript Promise
//! Integration) until the answer's callback, or a future's waker, calls
//! the host's wake. Neither way spins.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

/// The time spent blocked waiting for the device's work, summed over
/// every wait that counts it: [`Backend::waited`] reads it. Shared by the
/// backend and what it hands out to wait later (marks, checks, downloads).
///
/// [`Backend::waited`]: entelechy_rules::Backend::waited
#[derive(Clone, Debug, Default)]
pub(crate) struct Spent(Arc<std::sync::atomic::AtomicU64>);

impl Spent {
    /// `wait`'s answer, its time counted.
    pub(crate) fn time<T>(&self, wait: impl FnOnce() -> T) -> T {
        let started = entelechy_rules::clock::now();
        let answer = wait();
        let nanos = entelechy_rules::clock::since(started)
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        self.0
            .fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
        answer
    }

    /// All the time counted so far.
    pub(crate) fn total(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// A thread looking at the device for as long as the backend lives: what
/// the device finishes is heard as it finishes (a map, a frame's clocked
/// end), and a device that drops its clock while nothing keeps up with its
/// work (Apple's GPUs) keeps pace with the frames. It looks and does not
/// wait: a wait holds the device's lock, which presenting a frame takes,
/// and one given a timeout can be told the queue is empty while it holds a
/// finished index read before another thread's poll or submission emptied
/// it, which wgpu takes for a broken promise and panics. A page has no
/// thread: it hears the device as it gets its turns.
pub(crate) struct Waiter {
    on: Arc<std::sync::atomic::AtomicBool>,
    #[cfg(test)]
    thread: std::thread::JoinHandle<()>,
}

/// How long the waiter sleeps between looks at the device.
const LOOK: std::time::Duration = std::time::Duration::from_micros(100);

impl Waiter {
    /// Starts looking at `device`'s work; nothing where there is no thread
    /// to look on.
    pub(crate) fn start(device: &wgpu::Device) -> Option<Self> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let on = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let waiting = Arc::clone(&on);
        let device = device.clone();
        let thread = std::thread::Builder::new()
            .name("entelechy device waiter".into())
            .spawn(move || {
                while waiting.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = device.poll(wgpu::PollType::Poll);
                    std::thread::sleep(LOOK);
                }
            })
            .ok()?;
        #[cfg(not(test))]
        drop(thread);
        Some(Self {
            on,
            #[cfg(test)]
            thread,
        })
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        self.on.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Lets go of a buffer the device is done with once the work submitted
/// has run. On a machine dropping it is all: the waiter frees it then. A
/// page has no waiter and frees what is destroyed.
pub(crate) fn discard(buffer: wgpu::Buffer) {
    if crate::PAGE {
        buffer.destroy();
    }
}

/// Lets go of a texture as [`discard`] does a buffer.
pub(crate) fn discard_texture(texture: wgpu::Texture) {
    if crate::PAGE {
        texture.destroy();
    }
}

/// How to wait: block, or park with the host until it is woken.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wait(Option<Parking>);

/// The host's two halves of parking: `park` suspends until `wake` has
/// been called (or at once when it was called since the last park).
#[derive(Clone, Copy, Debug)]
struct Parking {
    park: fn(),
    wake: fn(),
}

/// A waker that wakes the host.
struct HostWaker(fn());

impl Wake for HostWaker {
    fn wake(self: Arc<Self>) {
        (self.0)();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        (self.0)();
    }
}

impl Wait {
    /// Blocks the thread: a machine's own device.
    pub fn blocking() -> Self {
        Self(None)
    }

    /// Parks with the host until woken: a page. `park` returns once
    /// `wake` has been called since it last returned.
    pub fn parks(park: fn(), wake: fn()) -> Self {
        Self(Some(Parking { park, wake }))
    }

    /// Whether this waits by parking with the host.
    pub fn parks_with_host(self) -> bool {
        self.0.is_some()
    }

    /// The value `future` comes to.
    pub fn block_on<F: Future>(self, future: F) -> F::Output {
        let Some(parking) = self.0 else {
            return pollster::block_on(future);
        };
        let mut future = std::pin::pin!(future);
        let waker = Waker::from(Arc::new(HostWaker(parking.wake)));
        let mut context = Context::from_waker(&waker);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
            (parking.park)();
        }
    }

    /// Maps `slice` for reading and waits until it is, after everything the
    /// queue was given before.
    pub fn map_read(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slice: wgpu::BufferSlice<'_>,
    ) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let wake = self.wake();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
            wake();
        });
        // A browser may answer a mapping only when it next looks at the device
        // (Firefox: on a submission, else a tenth of a second later); an empty
        // submission makes it look now.
        if self.parks_with_host() {
            queue.submit(std::iter::empty());
        }
        self.until(device, || receiver.try_recv().ok())?
            .map_err(|error| error.to_string())
    }

    /// Maps `slice` for reading as [`Wait::map_read`] does, waiting only
    /// for the work up to `after`: what was submitted since runs on.
    pub fn map_read_after(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slice: wgpu::BufferSlice<'_>,
        after: &wgpu::SubmissionIndex,
    ) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let wake = self.wake();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
            wake();
        });
        if self.parks_with_host() {
            queue.submit(std::iter::empty());
        }
        self.until_after(device, after, || receiver.try_recv().ok())?
            .map_err(|error| error.to_string())
    }

    /// Waits until the work up to `after` has run, not what followed it.
    pub fn done_after(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        after: &wgpu::SubmissionIndex,
    ) -> Result<(), String> {
        match self.0 {
            // A page answers a submission's end only as everything's.
            Some(_) => self.drain(device, queue),
            None => device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(after.clone()),
                    timeout: None,
                })
                .map(drop)
                .map_err(|error| error.to_string()),
        }
    }

    /// Waits until the queue has run everything it was given.
    pub fn drain(self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let wake = self.wake();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
            wake();
        });
        self.until(device, || receiver.try_recv().ok())
    }

    /// Asks for `slice` mapped for reading now, to be answered later
    /// ([`Wait::answer`]): `None` where the thread blocks, which asks when
    /// it waits. A page answers a map only once everything submitted before
    /// the asking is done, so a run's readback is asked for as the run is
    /// submitted, not when it is wanted: by then the next frame's work is
    /// queued, and asking would wait for it too.
    pub(crate) fn ask_map(
        self,
        queue: &wgpu::Queue,
        slice: wgpu::BufferSlice<'_>,
    ) -> Option<Asked<Result<(), wgpu::BufferAsyncError>>> {
        self.parks_with_host()
            .then(|| self.ask_map_now(queue, slice))
    }

    /// Asks for `slice` mapped for reading now on every target, to be
    /// answered by [`Wait::answer_after`]: on a machine [`Waiter`] hears it
    /// finish while later work is submitted, and in a page an empty
    /// submission makes browsers that poll on submissions look now.
    pub(crate) fn ask_map_now(
        self,
        queue: &wgpu::Queue,
        slice: wgpu::BufferSlice<'_>,
    ) -> Asked<Result<(), wgpu::BufferAsyncError>> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let wake = self.wake();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
            wake();
        });
        if self.parks_with_host() {
            queue.submit(std::iter::empty());
        }
        Asked(receiver)
    }

    /// Asks to be told when everything submitted so far is done, as
    /// [`Wait::ask_map`] asks for a map.
    pub(crate) fn ask_done(self, queue: &wgpu::Queue) -> Option<Asked<()>> {
        self.parks_with_host().then(|| {
            let (sender, receiver) = std::sync::mpsc::channel();
            let wake = self.wake();
            queue.on_submitted_work_done(move || {
                let _ = sender.send(());
                wake();
            });
            Asked(receiver)
        })
    }

    /// The answer to what was asked, once it comes.
    pub(crate) fn answer<T>(self, device: &wgpu::Device, asked: Asked<T>) -> Result<T, String> {
        self.until(device, || asked.0.try_recv().ok())
    }

    /// The answer to what was asked at `after`, polling only up to `after`.
    pub(crate) fn answer_after<T>(
        self,
        device: &wgpu::Device,
        after: &wgpu::SubmissionIndex,
        asked: Asked<T>,
    ) -> Result<T, String> {
        self.until_after(device, after, || asked.0.try_recv().ok())
    }

    /// What wakes the host, or nothing when the thread blocks.
    fn wake(self) -> fn() {
        fn nothing() {}
        self.0.map_or(nothing, |parking| parking.wake)
    }

    /// As [`Wait::until`], polling only for the work up to `after`.
    fn until_after<T>(
        self,
        device: &wgpu::Device,
        after: &wgpu::SubmissionIndex,
        mut answer: impl FnMut() -> Option<T>,
    ) -> Result<T, String> {
        loop {
            if let Some(value) = answer() {
                return Ok(value);
            }
            match self.0 {
                Some(parking) => (parking.park)(),
                None => {
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(after.clone()),
                            timeout: None,
                        })
                        .map_err(|error| error.to_string())?;
                }
            }
        }
    }

    /// Polls the device, or parks, until `answer` has something.
    fn until<T>(
        self,
        device: &wgpu::Device,
        mut answer: impl FnMut() -> Option<T>,
    ) -> Result<T, String> {
        loop {
            if let Some(value) = answer() {
                return Ok(value);
            }
            match self.0 {
                Some(parking) => (parking.park)(),
                None => {
                    device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .map_err(|error| error.to_string())?;
                }
            }
        }
    }
}

/// What the device was asked for and has yet to answer: [`Wait::ask_map`].
pub(crate) struct Asked<T>(std::sync::mpsc::Receiver<T>);

#[cfg(test)]
mod tests {
    use super::*;

    /// Other threads polling and submitting as the waiter looks: it lives
    /// through them, however the queue empties under it.
    #[test]
    fn the_waiter_lives_through_other_threads_polling() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(&Default::default()))
        else {
            return;
        };
        let waiter = Waiter::start(&device).expect("a thread to look on");
        let began = std::time::Instant::now();
        while began.elapsed() < std::time::Duration::from_secs(2) {
            let run = queue.submit(std::iter::empty());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(run),
                    timeout: None,
                })
                .unwrap();
        }
        assert!(!waiter.thread.is_finished(), "the waiter died");
    }
}
