//! Compressed sound as a WAV the Host reads: MP3, OGG Vorbis, FLAC, AAC in
//! MP4, and more, decoded by symphonia and written as 16-bit PCM at the
//! file's own rate, one or two channels. A WAV is what `input` takes, so
//! nothing in the runtime learns a lossy format.

use std::path::{Path, PathBuf};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// What an audio import produced.
pub struct Imported {
    /// The WAV written.
    pub file: PathBuf,
    /// Frames per second, as the source says.
    pub rate: u32,
    /// Channels kept: 1 or 2.
    pub channels: usize,
    /// Frames written.
    pub frames: usize,
}

/// Decodes `path` and writes `<stem>.wav` into `out_dir`.
pub fn import(path: &Path, out_dir: &Path) -> Result<Imported, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| format!("{}: no audio track", path.display()))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rate = 0;
    let mut source_channels = 0;
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                rate = spec.rate;
                source_channels = spec.channels.count();
                let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buffer.copy_interleaved_ref(decoded);
                interleaved.extend_from_slice(buffer.samples());
            }
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    if source_channels == 0 || interleaved.is_empty() {
        return Err(format!("{}: nothing decoded", path.display()));
    }
    let channels = source_channels.min(2);
    let frames = interleaved.len() / source_channels;
    let mut words = Vec::with_capacity(frames * channels);
    for frame in interleaved.chunks_exact(source_channels) {
        for sample in &frame[..channels] {
            words.push(sample.to_bits());
        }
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("sound");
    let file = out_dir.join(format!("{stem}.wav"));
    std::fs::write(&file, orla_formats::wav::encode(&words, channels, rate))
        .map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Imported {
        file,
        rate,
        channels,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are a 440 Hz tone at half amplitude, 0.3 s at 22050
    /// Hz, made with ffmpeg (the OGG in stereo, as its encoder insists);
    /// the decoders must give back the rate, the channels, the loudness,
    /// and the pitch, counted in zero crossings.
    fn check(name: &str, channels: usize) {
        let dir = std::env::temp_dir().join(format!("orla-audio-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let imported = import(&source, &dir).unwrap();
        assert_eq!(
            (imported.rate, imported.channels),
            (22050, channels),
            "{name}"
        );
        assert!(
            imported.frames >= 6615 && imported.frames < 9000,
            "{name}: {} frames",
            imported.frames
        );
        let (field, detail) = orla_session::Session::load_field(&imported.file, None).unwrap();
        assert!(detail.ends_with("at 22050 Hz"), "{detail}");
        assert_eq!(field.schema().value_words(), channels);
        let left: Vec<f32> = field
            .values()
            .iter()
            .step_by(channels)
            .map(|w| f32::from_bits(*w))
            .collect();
        let peak = left.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.3 && peak < 0.6, "{name}: peak {peak}");
        // Zero crossings over 0.1 s in the middle: 440 Hz gives 88.
        let start = imported.frames / 2;
        let window = &left[start..start + 2205];
        let crossings = window
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        assert!(
            (84..=92).contains(&crossings),
            "{name}: {crossings} crossings"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compressed_sounds_come_in_as_wav() {
        check("tone.mp3", 1);
        check("tone.ogg", 2);
        check("tone.flac", 1);
        check("tone.m4a", 1);
    }
}
