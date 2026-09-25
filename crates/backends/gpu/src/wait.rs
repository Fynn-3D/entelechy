//! Waiting for the device. On a machine of its own the thread blocks
//! until the device answers. In a page it cannot: the browser answers
//! only when the page's script gets a turn, so there waiting is parking
//! Orla ([`Wait::parks`]) until the answer wakes it: the host suspends
//! Orla (JavaScript Promise Integration) and the answer's callback, or a
//! future's waker, calls the host's wake. Nothing spins while it waits.
//! Everything that waits here waits the same way either way.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

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

    /// Maps `slice` for reading and waits until it is: after whatever
    /// the queue was given before.
    pub fn map_read(
        self,
        device: &wgpu::Device,
        slice: wgpu::BufferSlice<'_>,
    ) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let wake = self.wake();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
            wake();
        });
        self.until(device, || receiver.try_recv().ok())?
            .map_err(|error| error.to_string())
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

    /// What wakes the host, or nothing when the thread blocks.
    fn wake(self) -> fn() {
        fn nothing() {}
        self.0.map_or(nothing, |parking| parking.wake)
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
