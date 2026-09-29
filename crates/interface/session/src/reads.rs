//! What a session may read. Where reads are held (an agent composes the
//! scene), every read the session makes goes through [`Held`]: the path is
//! resolved as the store follows it, links and `..` included, and read
//! only where it lands inside a readable directory. Documents, placed
//! graphs, inputs and frames are all read there, so no reader checks
//! apart from the others.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::store::{Kind, Parsed, Stamp, Store};

/// The directories a session may read under.
#[derive(Debug, Default)]
pub(crate) struct Reads {
    /// Whether reads are held (an agent composes the scene).
    restricted: bool,
    /// Resolved directories the person allowed reads from.
    allowed: Vec<PathBuf>,
    /// Resolved directories the scene's documents are in or use, those of
    /// the files its inputs read, and the package directories.
    /// Where reads are held, these are fixed at the opened scene's first
    /// load: an agent composing the scene cannot grant itself reads by
    /// what it writes.
    beside: Vec<PathBuf>,
    /// Whether `beside` is fixed.
    settled: bool,
    /// Whether a sync has read the scene, fixing whether reads are held.
    synced: bool,
}

impl Reads {
    /// Holds reads to the documents' directories, the package directories
    /// and those [`Reads::allow`] allowed. Callable only before the first
    /// sync, so the readable set is the first-loaded scene's directories.
    pub(crate) fn restrict(&mut self) -> Result<(), String> {
        if self.synced && !self.restricted {
            return Err("reads are held from before the scene is first read, or not at all".into());
        }
        self.restricted = true;
        Ok(())
    }

    /// Allows files in `dir` and under it to be read.
    pub(crate) fn allow(&mut self, store: &dyn Store, dir: &Path) -> Result<(), String> {
        let resolved = store
            .canonical(dir)
            .map_err(|e| format!("cannot allow reads from {}: {e}", dir.display()))?;
        if !store.is_dir(&resolved) {
            return Err(format!(
                "cannot allow reads from {}: not a directory",
                dir.display()
            ));
        }
        self.allowed.push(resolved);
        Ok(())
    }

    /// Takes the directories the scene's documents are in or use and its
    /// inputs read from, once a sync has read it; where reads are held,
    /// only the first time.
    pub(crate) fn settle(&mut self, store: &dyn Store, read_from: &[PathBuf]) {
        if self.restricted && self.settled {
            return;
        }
        self.beside = read_from
            .iter()
            .chain(store.packages())
            .filter_map(|dir| store.canonical(dir).ok())
            .collect();
        self.settled = self.restricted;
        self.synced = true;
    }

    /// Whether `path` may be read: where it resolves to is inside a
    /// directory the first-loaded scene's documents are in or use or its
    /// inputs read from, a package directory, or an allowed directory.
    pub(crate) fn check(&self, store: &dyn Store, path: &Path) -> Result<(), String> {
        // The opened scene's first load reads freely; later reads are held.
        if !self.restricted || !self.settled {
            return Ok(());
        }
        let target = crate::writes::resolve(store, path)?;
        if self
            .beside
            .iter()
            .chain(&self.allowed)
            .any(|dir| target.starts_with(dir))
        {
            return Ok(());
        }
        Err(format!(
            "{} is outside the directories the scene's documents are in or use or its inputs read from, every package directory, and every directory allowed with `--allow file.read`",
            target.display()
        ))
    }
}

/// A store whose reads are held to [`Reads`].
pub(crate) struct Held {
    inner: Arc<dyn Store>,
    reads: Arc<Mutex<Reads>>,
}

impl Held {
    /// `inner`, read where `reads` allows.
    pub(crate) fn new(inner: Arc<dyn Store>, reads: Arc<Mutex<Reads>>) -> Self {
        Self { inner, reads }
    }

    fn check(&self, path: &Path) -> io::Result<()> {
        lock(&self.reads)
            .check(self.inner.as_ref(), path)
            .map_err(|why| io::Error::new(io::ErrorKind::PermissionDenied, why))
    }
}

/// The reads, whatever a panicking holder left.
pub(crate) fn lock(reads: &Mutex<Reads>) -> MutexGuard<'_, Reads> {
    reads.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Store for Held {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.check(path)?;
        self.inner.read(path)
    }
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.inner.write(path, bytes)
    }
    fn kind(&self, path: &Path) -> Option<Kind> {
        self.inner.kind(path)
    }
    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        self.check(dir)?;
        self.inner.list(dir)
    }
    /// No stamp where the file may not be read, so no document parsed
    /// before reads were held is taken from the parsed documents.
    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.check(path).ok()?;
        self.inner.stamp(path)
    }
    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        self.inner.canonical(path)
    }
    fn whole(&self, path: &Path) -> PathBuf {
        self.inner.whole(path)
    }
    fn packages(&self) -> &[PathBuf] {
        self.inner.packages()
    }
    fn parsed(&self) -> Option<&Parsed> {
        self.inner.parsed()
    }
    fn stamps_directories(&self) -> bool {
        self.inner.stamps_directories()
    }
}
