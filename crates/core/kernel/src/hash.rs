//! A quick deterministic hasher for the runtime's placement maps. Their keys
//! are handles and run descriptors the runtime holds, never chosen to collide,
//! so it trades a keyed hash's resistance to chosen collisions for speed; with
//! no random state, a map's order is the same on every run.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// Builds [`Quick`] hashers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QuickState;

impl BuildHasher for QuickState {
    type Hasher = Quick;
    fn build_hasher(&self) -> Quick {
        Quick(0)
    }
}

/// Multiplies each word in, a word at a time.
pub(crate) struct Quick(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Quick {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for Quick {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
    }
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
    fn finish(&self) -> u64 {
        let h = self.0;
        (h ^ (h >> 29)).wrapping_mul(SEED) ^ (h >> 32)
    }
}

/// A map hashed with [`Quick`].
pub(crate) type QuickMap<K, V> = HashMap<K, V, QuickState>;

/// A set hashed with [`Quick`].
pub(crate) type QuickSet<T> = HashSet<T, QuickState>;
