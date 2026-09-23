//! The sound card as a device adapter. It claims a graph output that is
//! rows of one or two floats, takes each evaluation's block into a ring,
//! and feeds the device from that ring in its real-time callback. The
//! callback never waits on the Graph, the GPU, or an allocation: when the
//! ring is empty it plays silence and counts an underrun.
//!
//! The adapter provides device access, not a signal model: sample rate
//! and block size are facts it reports; oscillators, envelopes, and mixing
//! are Kernels.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// A single-producer, single-consumer ring of interleaved samples. The
/// producer is the frame thread, the consumer the device callback. Slots
/// hold float bits in atomics, so the ring needs no lock and no unsafe
/// code; the indices are published with release/acquire ordering.
pub struct Ring {
    samples: Box<[AtomicU32]>,
    /// Next sample to read; advanced by the consumer.
    head: AtomicUsize,
    /// Next sample to write; advanced by the producer.
    tail: AtomicUsize,
    underruns: AtomicUsize,
}

impl Ring {
    /// A ring holding `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: (0..capacity.max(2))
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            underruns: AtomicUsize::new(0),
        }
    }

    /// Samples waiting to be played.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail + self.samples.len() - head) % self.samples.len()
    }

    /// Whether nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Samples that can still be written.
    pub fn room(&self) -> usize {
        self.samples.len() - 1 - self.len()
    }

    /// Writes as many of `block` as fit and returns how many; a block that
    /// does not fit is written partially, so the producer should check
    /// [`Ring::room`] first when it wants all-or-nothing.
    pub fn push(&self, block: &[f32]) -> usize {
        let capacity = self.samples.len();
        let count = block.len().min(self.room());
        let tail = self.tail.load(Ordering::Relaxed);
        for (offset, sample) in block[..count].iter().enumerate() {
            self.samples[(tail + offset) % capacity].store(sample.to_bits(), Ordering::Relaxed);
        }
        self.tail
            .store((tail + count) % capacity, Ordering::Release);
        count
    }

    /// Fills `out` from the ring, silence where it runs dry; returns how
    /// many samples were real.
    pub fn pop(&self, out: &mut [f32]) -> usize {
        let capacity = self.samples.len();
        let count = out.len().min(self.len());
        let head = self.head.load(Ordering::Relaxed);
        for (offset, slot) in out[..count].iter_mut().enumerate() {
            *slot =
                f32::from_bits(self.samples[(head + offset) % capacity].load(Ordering::Relaxed));
        }
        for slot in &mut out[count..] {
            *slot = 0.0;
        }
        self.head
            .store((head + count) % capacity, Ordering::Release);
        if count < out.len() {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        count
    }

    /// Callbacks that found the ring short.
    pub fn underruns(&self) -> usize {
        self.underruns.load(Ordering::Relaxed)
    }
}

/// How many blocks the ring holds ahead of the device.
const BLOCKS_AHEAD: usize = 4;

/// An open output device fed from a ring.
pub struct Audio {
    _stream: cpal::Stream,
    ring: Arc<Ring>,
    rate: u32,
    channels: usize,
    /// Samples per channel of the last block, sizing the ring.
    block: usize,
    dropped: usize,
}

impl Audio {
    /// Opens the default output device, preferring 48 kHz stereo floats
    /// and taking what the device offers otherwise. `block` is the number
    /// of frames one evaluation produces, which sizes the ring.
    pub fn open(block: usize) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no output device".to_string())?;
        let default = device
            .default_output_config()
            .map_err(|e| format!("no output config: {e}"))?;
        let mut rate = default.sample_rate();
        let mut channels = default.channels().max(1);
        if let Ok(configs) = device.supported_output_configs() {
            for range in configs {
                if range.sample_format() == cpal::SampleFormat::F32
                    && range.channels() >= 2
                    && (range.min_sample_rate()..=range.max_sample_rate()).contains(&48_000)
                {
                    rate = 48_000;
                    channels = range.channels();
                    break;
                }
            }
        }
        let channels = channels.min(2) as usize;
        let config = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let ring = Arc::new(Ring::new(block.max(64) * channels * BLOCKS_AHEAD));
        // One block of silence ahead, so the device's first callbacks do
        // not count as underruns while the first block is computed.
        ring.push(&vec![0.0; block.max(64) * channels]);
        let consumer = Arc::clone(&ring);
        let stream = device
            .build_output_stream(
                config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    consumer.pop(out);
                },
                |error| eprintln!("audio stream error: {error}"),
                None,
            )
            .map_err(|e| format!("cannot open the output stream: {e}"))?;
        stream
            .play()
            .map_err(|e| format!("cannot start audio: {e}"))?;
        Ok(Self {
            _stream: stream,
            ring,
            rate,
            channels,
            block,
            dropped: 0,
        })
    }

    /// The device's sample rate.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Whether the ring can take another block of the last block's size:
    /// when it cannot, the device has not yet consumed what it was given,
    /// and evaluating again would only drop the result.
    pub fn has_room(&self) -> bool {
        self.ring.room() >= self.block.max(1) * self.channels
    }

    /// Queues one block of frames laid out as `words_per_frame` floats
    /// each (one for mono, two for stereo), spreading mono to every
    /// channel. A block that does not fit is dropped whole, so a frame
    /// boundary never lands mid-block.
    pub fn push(&mut self, words: &[u32], words_per_frame: usize) -> bool {
        if words_per_frame == 0 {
            return false;
        }
        let frames = words.len() / words_per_frame;
        let mut interleaved = Vec::with_capacity(frames * self.channels);
        for frame in words.chunks_exact(words_per_frame) {
            for channel in 0..self.channels {
                let word = frame[channel.min(words_per_frame - 1)];
                interleaved.push(f32::from_bits(word).clamp(-1.0, 1.0));
            }
        }
        if self.ring.room() < interleaved.len() {
            self.dropped += 1;
            return false;
        }
        self.block = frames;
        self.ring.push(&interleaved);
        true
    }

    /// A short description for the title.
    pub fn describe(&self) -> String {
        format!(
            "audio {} Hz {} ch, block {}, {} underruns, {} dropped",
            self.rate,
            self.channels,
            self.block,
            self.ring.underruns(),
            self.dropped
        )
    }
}

impl Drop for Audio {
    fn drop(&mut self) {
        eprintln!("audio closed: {}", self.describe());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_plays_what_was_pushed_and_silence_when_dry() {
        let ring = Ring::new(8);
        assert_eq!(ring.room(), 7);
        assert_eq!(ring.push(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(ring.len(), 3);
        let mut out = [9.0; 5];
        assert_eq!(ring.pop(&mut out), 3);
        assert_eq!(out, [1.0, 2.0, 3.0, 0.0, 0.0]);
        assert_eq!(ring.underruns(), 1);
        // Wraps around the end.
        assert_eq!(ring.push(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 6);
        assert_eq!(ring.push(&[10.0, 11.0]), 1, "only one slot was left");
        let mut out = [0.0; 7];
        assert_eq!(ring.pop(&mut out), 7);
        assert_eq!(out, [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert!(ring.is_empty());
    }
}
