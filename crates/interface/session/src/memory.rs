//! Where a session's memory is: every Field the Host holds, by what holds
//! it (an input's file, a node's output, a port's state, or the state the
//! frame in flight replaced), and what the backends keep of their own.

use entelechy_graph::Held;
use entelechy_kernel::{Census, FieldBytes, Handle};

use crate::Session;

/// A session's memory, at one moment ([`Session::memory_map`]).
#[derive(Clone, Debug, Default)]
pub struct MemoryMap {
    /// The Host's own count of where its bytes are.
    pub census: Census,
    /// Every Field something holds, the largest first; a Field held twice
    /// is counted once, for the first of input, state, output and held
    /// back that holds it.
    pub held: Vec<HeldBytes>,
    /// The bytes of live Fields nothing above holds: a run's scratch, a
    /// read in hand.
    pub loose: FieldBytes,
    /// What the backends keep of their own, each part by name.
    pub backends: Vec<(String, u64)>,
}

/// One Field and what holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldBytes {
    /// An input's name, or `node.port`.
    pub owner: String,
    /// `input`, `state`, `output` or `back`.
    pub kind: &'static str,
    /// Where its bytes are.
    pub bytes: FieldBytes,
}

impl HeldBytes {
    /// Its bytes wherever they are.
    pub fn total(&self) -> u64 {
        self.bytes.host + self.bytes.device
    }
}

/// The fewest bytes of an input's Host copy that are let go once the
/// input is on a device ([`Session::let_go_of_input_copies`]).
pub const COPY_LEAST: u64 = 1 << 20;

impl Session {
    /// Lets go of the Host copies of the inputs on a device, each once and
    /// each of [`COPY_LEAST`] bytes or more: an input does not change from
    /// frame to frame, so the device's words serve every frame, and what
    /// reads it on the processor brings them back, once, and keeps them.
    pub(crate) fn let_go_of_input_copies(&mut self) {
        let handles: Vec<Handle> = self.input_handles().values().copied().collect();
        self.copies_let_go.retain(|handle| handles.contains(handle));
        for handle in handles {
            if self.copies_let_go.contains(&handle) {
                continue;
            }
            let host = self.host_mut();
            let on_device = host.resident(handle).is_ok_and(|device| device.is_some());
            if on_device
                && host
                    .field_bytes(handle)
                    .is_ok_and(|bytes| bytes.host >= COPY_LEAST)
                && host.let_go_of_copy(handle).is_ok()
            {
                self.copies_let_go.insert(handle);
            }
        }
    }

    /// Where the session's memory is now.
    pub fn memory_map(&self) -> MemoryMap {
        let host = self.runtime().host();
        let census = host.census();
        let mut seen = std::collections::BTreeSet::<Handle>::new();
        let mut held = Vec::new();
        let mut claim = |owner: String, kind: &'static str, handle: Handle| {
            if seen.insert(handle)
                && let Ok(bytes) = host.field_bytes(handle)
            {
                held.push(HeldBytes { owner, kind, bytes });
            }
        };
        for (name, handle) in self.input_handles() {
            claim(name.clone(), "input", *handle);
        }
        let mut holdings = self.holdings();
        holdings.sort_by_key(|holding| match holding.held {
            Held::State => 0,
            Held::Output => 1,
            Held::Back => 2,
        });
        for holding in holdings {
            let kind = match holding.held {
                Held::State => "state",
                Held::Output => "output",
                Held::Back => "back",
            };
            claim(
                format!("{}.{}", holding.node, holding.port),
                kind,
                holding.handle,
            );
        }
        held.sort_by_key(|entry| std::cmp::Reverse(entry.total()));
        let (host_held, device_held) = held.iter().fold((0, 0), |(h, d), entry| {
            (h + entry.bytes.host, d + entry.bytes.device)
        });
        MemoryMap {
            census,
            held,
            loose: FieldBytes {
                host: (census.host + census.kept).saturating_sub(host_held),
                device: census.device.saturating_sub(device_held),
            },
            backends: self.runtime().backend_memory(),
        }
    }
}
