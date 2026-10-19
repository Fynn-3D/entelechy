//! The sound card as a device adapter: each evaluation's block of rows
//! of one or two floats goes into a ring the real-time callback drains.
//! The callback never waits on the Graph, the GPU, or an allocation; on
//! an empty ring it plays silence and counts an underrun. Sample rate
//! and block size are reported; signal models are Kernels.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// A lock-free single-producer (frame thread), single-consumer (device
/// callback) ring of interleaved samples: float bits in atomic slots,
/// indices published with release/acquire ordering.
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

    /// Writes as many of `block` as fit and returns how many; check
    /// [`Ring::room`] first for all-or-nothing.
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

    /// Fills `out` from the ring, silence where it runs dry; returns the
    /// number of real samples.
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

/// The block a ring is sized by: the first, and at least two frames' worth
/// at 60 Hz, so a first block that came short (the time since a frame
/// that was quick) leaves room for the frames after it.
fn sized(first: usize, rate: u32) -> usize {
    first.max(rate as usize / 30).max(64)
}

/// An open output device fed from a ring.
pub struct Audio {
    _stream: cpal::Stream,
    ring: Arc<Ring>,
    rate: u32,
    channels: usize,
    /// Samples per channel of the first block, which sized the ring.
    first: usize,
    dropped: usize,
}

impl Audio {
    /// Opens the default output device, preferring 48 kHz stereo floats.
    /// `block`, the frames one evaluation produces, sizes the ring.
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
        let first = sized(block, rate);
        let ring = Arc::new(Ring::new(first * channels * BLOCKS_AHEAD));
        // One block of silence ahead covers the first block's computation.
        ring.push(&vec![0.0; first * channels]);
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
            first,
            dropped: 0,
        })
    }

    /// The device's sample rate.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The samples per channel the device holds and has not yet played.
    pub fn queued(&self) -> u32 {
        (self.ring.len() / self.channels.max(1)) as u32
    }

    /// Whether the ring has room for a frame's worth of sound at 60 Hz, so
    /// a frame may make another block. A frame's worth, not room for a
    /// block the size of the last: a sound whose blocks are as long as the
    /// time since the frame before (`synth.paced`) would otherwise keep,
    /// after a stall, the long blocks and slow frames it left. Frames that
    /// each make one block of a fixed size are held to the device's pace.
    pub fn has_room(&self) -> bool {
        self.ring.room() >= (self.rate as usize / 60).max(1) * self.channels
    }

    /// Queues one block of frames of `words_per_frame` floats (1 mono,
    /// 2 stereo), spreading mono to every channel. A block that does not
    /// fit is dropped whole.
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
        self.ring.push(&interleaved);
        true
    }

    /// A short description for the title.
    pub fn describe(&self) -> String {
        format!(
            "audio {} Hz {} ch, block {}, {} underruns, {} dropped",
            self.rate,
            self.channels,
            self.first,
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
