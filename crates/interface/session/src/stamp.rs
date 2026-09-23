//! What a file looked like when it was last read, so a sync reads again
//! only what changed on disk.

use std::path::Path;
use std::time::SystemTime;

/// A file's modification time and length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

/// The file's stamp now; `None` when it cannot be read.
pub(crate) fn stamp(path: &Path) -> Option<Stamp> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(Stamp {
        modified: metadata.modified().ok(),
        len: metadata.len(),
    })
}
