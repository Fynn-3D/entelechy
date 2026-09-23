//! A sound as WAV, from rows of one or two floats. WAV is the lossless
//! way out; a compressed copy is a job for ffmpeg, not for the runtime's
//! neighbours.

/// Interleaved frames of `channels` floats in `-1..=1`.
pub struct Pcm<'a> {
    /// Samples, frame by frame.
    pub samples: &'a [f32],
    /// 1 or 2.
    pub channels: usize,
    /// Frames per second.
    pub rate: u32,
}

/// 16-bit PCM WAV.
pub fn wav(pcm: &Pcm<'_>) -> Vec<u8> {
    let words: Vec<u32> = pcm.samples.iter().map(|v| v.to_bits()).collect();
    orla_session::wav::encode(&words, pcm.channels, pcm.rate)
}
