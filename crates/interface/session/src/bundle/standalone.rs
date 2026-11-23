//! Standalone application envelope and executable trailer for native exports.
//!
//! A shipped native app carries a [`Standalone`] envelope: its window title,
//! preferred backend, pointer capture mode, optional logical window size, and
//! compiled [`Bundle`](super::Bundle) bytes. On macOS it sits in
//! `Contents/Resources/app.entapp` beside the untouched Mach-O runner so the
//! linker signature holds; on Windows and Linux it is appended to the runner
//! executable with a 24-byte footer ([`pack_executable`], [`read_trailer`]).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const APP_MAGIC: &[u8; 16] = b"ENTELECHYAPP\0\x01\0\0";
const TRAILER_MAGIC: &[u8; 16] = b"ENTELECHYTRLR\x01\0\0";
const FOOTER_LEN: usize = 8 + TRAILER_MAGIC.len();
const MAX_TEXT: usize = 1024;
const MAX_PAYLOAD_BYTES: u64 = 1 << 30;

/// How a window or exported page captures the system pointer on click.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointerMode {
    /// Free system cursor mapped into the picture.
    #[default]
    Free,
    /// Captured on click, stepped by relative motion and clamped to the picture's edges.
    Hold,
    /// Captured on click, stepped by unbounded relative motion for 360-degree camera aim.
    Lock,
}

impl PointerMode {
    /// Name used in templates and diagnostics (`free`, `hold`, `lock`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Hold => "hold",
            Self::Lock => "lock",
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::Free => 0,
            Self::Hold => 1,
            Self::Lock => 2,
        }
    }

    fn from_byte(byte: u8) -> Result<Self, String> {
        match byte {
            0 => Ok(Self::Free),
            1 => Ok(Self::Hold),
            2 => Ok(Self::Lock),
            other => Err(format!("unknown standalone pointer mode {other}")),
        }
    }
}

/// Metadata and compiled [`Bundle`](super::Bundle) carried by a standalone native app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Standalone {
    /// Window and application title.
    pub title: String,
    /// Preferred backend (`best`, `gpu`, `jit`, or `cpu`).
    pub backend: String,
    /// Pointer capture mode when the window is clicked.
    pub pointer: PointerMode,
    /// Initial logical window size, when fixed at export time.
    pub window: Option<(u32, u32)>,
    /// Encoded [`Bundle`](super::Bundle) bytes (`ENTELECHYBUNDLE`).
    pub bundle: Vec<u8>,
}

impl Standalone {
    /// Whether `bytes` begins with the standalone envelope magic.
    pub fn has_magic(bytes: &[u8]) -> bool {
        bytes.starts_with(APP_MAGIC)
    }

    /// Encodes this standalone envelope into bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            APP_MAGIC.len() + 24 + self.title.len() + self.backend.len() + self.bundle.len(),
        );
        out.extend_from_slice(APP_MAGIC);
        put_str(&mut out, &self.title);
        put_str(&mut out, &self.backend);
        out.push(self.pointer.to_byte());
        let (w, h) = self.window.unwrap_or((0, 0));
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        out.extend_from_slice(&(self.bundle.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.bundle);
        out
    }

    /// Decodes a standalone envelope from `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if !Self::has_magic(bytes) {
            return Err("not an Entelechy standalone application envelope".into());
        }
        let mut at = APP_MAGIC.len();
        let title = take_str(bytes, &mut at, "title")?;
        let backend = take_str(bytes, &mut at, "backend")?;
        let pointer_byte = *bytes
            .get(at)
            .ok_or_else(|| "truncated standalone pointer mode".to_owned())?;
        at += 1;
        let pointer = PointerMode::from_byte(pointer_byte)?;
        let w = take_u32(bytes, &mut at, "window width")?;
        let h = take_u32(bytes, &mut at, "window height")?;
        let window = (w > 0 && h > 0).then_some((w, h));
        let bundle_len = take_u64(bytes, &mut at, "bundle length")?;
        if bundle_len > MAX_PAYLOAD_BYTES {
            return Err(format!(
                "standalone bundle length {bundle_len} exceeds limit"
            ));
        }
        let bundle_len = usize::try_from(bundle_len)
            .map_err(|_| "standalone bundle length overflows address space".to_owned())?;
        let end = at
            .checked_add(bundle_len)
            .ok_or_else(|| "standalone bundle length overflow".to_owned())?;
        let slice = bytes
            .get(at..end)
            .ok_or_else(|| "truncated standalone bundle bytes".to_owned())?;
        if end != bytes.len() {
            return Err("trailing bytes after standalone envelope".into());
        }
        Ok(Self {
            title,
            backend,
            pointer,
            window,
            bundle: slice.to_vec(),
        })
    }
}

/// Appends `app` and a 24-byte trailer footer to a prebuilt `runner` executable.
pub fn pack_executable(runner: &[u8], app: &Standalone) -> Vec<u8> {
    let payload = app.encode();
    let mut out = Vec::with_capacity(runner.len() + payload.len() + FOOTER_LEN);
    out.extend_from_slice(runner);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(TRAILER_MAGIC);
    out
}

/// Extracts an appended [`Standalone`] envelope from `executable`, or `Ok(None)`
/// when the binary carries no trailer.
pub fn unpack_trailer(executable: &[u8]) -> Result<Option<Standalone>, String> {
    if executable.len() < FOOTER_LEN {
        return Ok(None);
    }
    let footer = &executable[executable.len() - FOOTER_LEN..];
    if &footer[8..] != TRAILER_MAGIC {
        return Ok(None);
    }
    let len = u64::from_le_bytes(footer[..8].try_into().expect("8 bytes"));
    if len > MAX_PAYLOAD_BYTES {
        return Err(format!("standalone trailer length {len} exceeds limit"));
    }
    let len = usize::try_from(len)
        .map_err(|_| "standalone trailer length overflows address space".to_owned())?;
    let body_end = executable.len() - FOOTER_LEN;
    let Some(start) = body_end.checked_sub(len) else {
        return Err("standalone trailer length exceeds executable size".into());
    };
    Standalone::decode(&executable[start..body_end]).map(Some)
}

/// Seeks to the end of `path` and reads an appended [`Standalone`] trailer when
/// present, without reading the runner image itself into memory.
pub fn read_trailer(path: &Path) -> Result<Option<Standalone>, String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let total = file
        .seek(SeekFrom::End(0))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if total < FOOTER_LEN as u64 {
        return Ok(None);
    }
    file.seek(SeekFrom::End(-(FOOTER_LEN as i64)))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut footer = [0u8; FOOTER_LEN];
    file.read_exact(&mut footer)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if &footer[8..] != TRAILER_MAGIC {
        return Ok(None);
    }
    let len = u64::from_le_bytes(footer[..8].try_into().expect("8 bytes"));
    if len > MAX_PAYLOAD_BYTES {
        return Err(format!(
            "{}: standalone trailer length {len} exceeds limit",
            path.display()
        ));
    }
    let body_end = total - FOOTER_LEN as u64;
    let Some(start) = body_end.checked_sub(len) else {
        return Err(format!(
            "{}: standalone trailer length exceeds executable size",
            path.display()
        ));
    };
    let cap = usize::try_from(len)
        .map_err(|_| format!("{}: standalone trailer overflows memory", path.display()))?;
    file.seek(SeekFrom::Start(start))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut payload = vec![0u8; cap];
    file.read_exact(&mut payload)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Standalone::decode(&payload)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn take_u32(bytes: &[u8], at: &mut usize, what: &str) -> Result<u32, String> {
    let end = at
        .checked_add(4)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    let slice = bytes
        .get(*at..end)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    *at = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

fn take_u64(bytes: &[u8], at: &mut usize, what: &str) -> Result<u64, String> {
    let end = at
        .checked_add(8)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    let slice = bytes
        .get(*at..end)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    *at = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

fn take_str(bytes: &[u8], at: &mut usize, what: &str) -> Result<String, String> {
    let len = take_u32(bytes, at, what)? as usize;
    if len > MAX_TEXT {
        return Err(format!("standalone {what} length {len} exceeds {MAX_TEXT}"));
    }
    let end = at
        .checked_add(len)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    let slice = bytes
        .get(*at..end)
        .ok_or_else(|| format!("truncated standalone {what}"))?;
    *at = end;
    std::str::from_utf8(slice)
        .map(str::to_owned)
        .map_err(|_| format!("standalone {what} is not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Standalone {
        Standalone {
            title: "Jump Jump".into(),
            backend: "best".into(),
            pointer: PointerMode::Lock,
            window: Some((960, 540)),
            bundle: b"ENTELECHYBUNDLE\0test-bytes".to_vec(),
        }
    }

    #[test]
    fn standalone_envelope_round_trips() {
        let app = sample();
        let bytes = app.encode();
        assert!(Standalone::has_magic(&bytes));
        assert_eq!(Standalone::decode(&bytes).unwrap(), app);
    }

    #[test]
    fn executable_trailer_round_trips_and_ignores_plain_binaries() {
        let runner = b"MZ\x90\x00stub-runner-image";
        assert_eq!(unpack_trailer(runner).unwrap(), None);

        let app = sample();
        let packed = pack_executable(runner, &app);
        assert!(packed.starts_with(runner));
        assert_eq!(unpack_trailer(&packed).unwrap(), Some(app.clone()));

        let dir = std::env::temp_dir().join(format!("entelechy-trailer-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let plain_path = dir.join("plain.exe");
        let packed_path = dir.join("packed.exe");
        std::fs::write(&plain_path, runner).unwrap();
        std::fs::write(&packed_path, &packed).unwrap();
        assert_eq!(read_trailer(&plain_path).unwrap(), None);
        assert_eq!(read_trailer(&packed_path).unwrap(), Some(app));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
