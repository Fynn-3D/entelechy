//! Waiting for the device. On a machine of its own the thread blocks
//! until the device answers. In a page it cannot: the browser answers
//! only when the page's script gets a turn, so there waiting is giving
//! the host turns ([`Wait::turns`]) until the answer is in; the host
//! (the page, through JavaScript Promise Integration) suspends Orla for
//! each. Everything that waits here waits the same way either way.

use std::future::Future;
use std::task::{Context, Poll, Waker};

/// How to wait: block, or give the host turns with this function.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wait(Option<fn()>);

impl Wait {
    /// Blocks the thread: a machine's own device.
    pub fn blocking() -> Self {
        Self(None)
    }

    /// Gives the host a turn with `turn` until the answer is in: a page.
    pub fn turns(turn: fn()) -> Self {
        Self(Some(turn))
    }

    /// Whether this waits by giving the host turns.
    pub fn gives_turns(self) -> bool {
        self.0.is_some()
    }

    /// The value `future` comes to.
    pub fn block_on<F: Future>(self, future: F) -> F::Output {
        let Some(turn) = self.0 else {
            return pollster::block_on(future);
        };
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
            turn();
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
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.until(device, || receiver.try_recv().ok())?
            .map_err(|error| error.to_string())
    }

    /// Waits until the queue has run everything it was given.
    pub fn drain(self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.until(device, || receiver.try_recv().ok())
    }

    /// Polls the device until `answer` has something.
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
                Some(turn) => turn(),
                None => {
                    device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .map_err(|error| error.to_string())?;
                }
            }
        }
    }
}
