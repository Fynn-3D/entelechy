//! WAV files as rows: a sound decodes to `Rows<Float>` (mono) or
//! `Rows<Vec2>` (stereo), samples in `-1..=1`, and rows of one or two
//! floats encode as 16-bit PCM. The sample rate is a fact about the file,
//! not the Field: the reader reports it, and a Kernel that plays the
//! rows at the device's `rate()` resamples with `at_linear`.

use std::fmt;

use orla_host::{Domain, FieldData, HostError, Scalar, Schema};

/// Why bytes are not a WAV the Host can take.
#[derive(Debug)]
pub enum WavError {
    /// The file is malformed or uses a format not supported.
    Format(String),
    /// The Host refused the Field.
    Host(HostError),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(message) => write!(f, "not a WAV the Host can read: {message}"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for WavError {}

/// A decoded sound: frames of one or two channels, and the file's rate.
pub struct Decoded {
    /// The Field: `Rows` of one float per frame or two.
    pub field: FieldData,
    /// Frames per second as the file says.
    pub rate: u32,
    /// Channels kept: 1 or 2.
    pub channels: usize,
}

/// Decodes a RIFF WAVE file of PCM (8, 16, 24, or 32-bit) or 32-bit float
/// samples; more than two channels keep the first two.
pub fn decode(bytes: &[u8]) -> Result<Decoded, WavError> {
    let format = |message: &str| WavError::Format(message.to_owned());
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format("missing RIFF/WAVE header"));
    }
    let mut position = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size = u32::from_le_bytes([
            bytes[position + 4],
            bytes[position + 5],
            bytes[position + 6],
            bytes[position + 7],
        ]) as usize;
        let body = bytes
            .get(position + 8..(position + 8 + size).min(bytes.len()))
            .ok_or_else(|| format("chunk ends early"))?;
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(format("fmt chunk too short"));
                }
                let mut tag = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                if tag == 0xfffe && body.len() >= 26 {
                    tag = u16::from_le_bytes([body[24], body[25]]);
                }
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        position += 8 + size + (size & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or_else(|| format("no fmt chunk"))?;
    let data = data.ok_or_else(|| format("no data chunk"))?;
    if channels == 0 {
        return Err(format("no channels"));
    }
    let sample = |frame: &[u8]| -> Result<f32, WavError> {
        Ok(match (tag, bits) {
            (1, 8) => frame[0] as f32 / 127.5 - 1.0,
            (1, 16) => i16::from_le_bytes([frame[0], frame[1]]) as f32 / 32768.0,
            (1, 24) => {
                (i32::from_le_bytes([0, frame[0], frame[1], frame[2]]) >> 8) as f32 / 8_388_608.0
            }
            (1, 32) => {
                i32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as f32
                    / 2_147_483_648.0
            }
            (3, 32) => f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]),
            _ => {
                return Err(format(
                    "sample format; use PCM 8/16/24/32-bit or 32-bit float",
                ));
            }
        })
    };
    let bytes_per_sample = bits as usize / 8;
    if bytes_per_sample == 0 {
        return Err(format("bits per sample"));
    }
    let frame_bytes = bytes_per_sample * channels as usize;
    let frames = data.len() / frame_bytes;
    if frames == 0 {
        return Err(format("no frames"));
    }
    let kept = (channels as usize).min(2);
    let mut words = Vec::with_capacity(frames * kept);
    for frame in data.chunks_exact(frame_bytes).take(frames) {
        for channel in 0..kept {
            let start = channel * bytes_per_sample;
            words.push(sample(&frame[start..start + bytes_per_sample])?.to_bits());
        }
    }
    let schema = Schema::new(vec![Scalar::Float; kept]);
    let field = FieldData::from_storage(schema, Domain::Rows(frames as u32), words)
        .map_err(WavError::Host)?;
    Ok(Decoded {
        field,
        rate,
        channels: kept,
    })
}

/// Encodes rows of `channels` floats (1 or 2) as 16-bit PCM at `rate`.
pub fn encode(words: &[u32], channels: usize, rate: u32) -> Vec<u8> {
    let frames = words.len() / channels.max(1);
    let data_len = (frames * channels * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&((channels * 2) as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for word in &words[..frames * channels] {
        let value = (f32::from_bits(*word).clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_and_mono_round_trip_and_the_rate_is_reported() {
        let words: Vec<u32> = [0.0f32, 0.5, -0.5, 1.0, 0.25, -1.0]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let stereo = decode(&encode(&words, 2, 44100)).unwrap();
        assert_eq!((stereo.rate, stereo.channels), (44100, 2));
        assert_eq!(stereo.field.domain(), Domain::Rows(3));
        let back: Vec<f32> = stereo
            .field
            .storage()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();
        assert!((back[1] - 0.5).abs() < 1e-4 && (back[5] + 1.0).abs() < 1e-4);
        let mono = decode(&encode(&words, 1, 8000)).unwrap();
        assert_eq!((mono.channels, mono.field.domain()), (1, Domain::Rows(6)));
        assert!(decode(b"RIFF....WAVEjunk").is_err());
    }
}
