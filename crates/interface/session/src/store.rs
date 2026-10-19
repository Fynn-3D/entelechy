//! Where a session's documents are. Every session read and write goes
//! through a [`Store`]: the [`Disk`], or [`Memory`], files the caller
//! handed in (a page). The caller chooses; the session behaves the same.

use std::collections::BTreeMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

/// What is at a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A file.
    File,
    /// A directory.
    Dir,
}

/// A file's version as read; it changes whenever the file does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp(Seen);

impl Stamp {
    /// The file's length in bytes, where the stamp holds it (a file on
    /// disk's does).
    pub(crate) fn length(&self) -> Option<u64> {
        match self.0 {
            Seen::Disk(_, length) => Some(length),
            Seen::Memory(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seen {
    /// A file on disk: its modification time and length.
    Disk(Option<std::time::SystemTime>, u64),
    /// A file handed in: the number of the write that made it.
    Memory(u64),
}

/// Files by path, read and written whole.
pub trait Store: Send + Sync {
    /// The file's bytes.
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    /// The file's first `bytes` bytes, or all of it when shorter: what
    /// its header says, before the rest is read.
    fn read_start(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        let mut all = self.read(path)?;
        all.truncate(bytes);
        Ok(all)
    }
    /// Replaces the file atomically, making the directories it needs.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;
    /// Removes the file.
    fn remove(&self, path: &Path) -> io::Result<()>;
    /// Renames the file `from` to `to`, making the directories `to` needs.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// Removes the directory when it holds nothing; leaves it otherwise.
    fn remove_empty(&self, dir: &Path) -> io::Result<()>;
    /// What is at `path`, if anything.
    fn kind(&self, path: &Path) -> Option<Kind>;
    /// What a directory holds, unordered.
    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>>;
    /// The file's stamp now, if it exists.
    fn stamp(&self, path: &Path) -> Option<Stamp>;
    /// The file's one name under any spelling; an error when absent.
    fn canonical(&self, path: &Path) -> io::Result<PathBuf>;
    /// `path` made whole: absolute where the store has a current
    /// directory, `.` and `..` folded, links unread.
    fn whole(&self, path: &Path) -> PathBuf;
    /// The directories a `use` by name looks for a package in, in order;
    /// none unless the store was given some.
    fn packages(&self) -> &[PathBuf] {
        &[]
    }
    /// Where the store keeps documents it has parsed, if it keeps them.
    fn parsed(&self) -> Option<&Parsed> {
        None
    }
    /// Whether a directory's stamp changes whenever what it holds does
    /// (an entry added, removed or renamed).
    fn stamps_directories(&self) -> bool {
        false
    }
    /// A sync begins: until it ends, the store may answer a path's kind,
    /// stamp and resolved path as it first found them, for a sync walks
    /// the same files from every document. A file changed meanwhile is
    /// seen by the next sync.
    fn sync_begins(&self) {}
    /// The sync ends: every path is looked at afresh.
    fn sync_ends(&self) {}
}

/// Documents parsed once per stamp: a document read again unchanged is
/// not parsed again; and what is worked out of them once, by content,
/// for every session reading through the store.
#[derive(Clone, Debug, Default)]
pub struct Parsed {
    documents: std::sync::Arc<Mutex<BTreeMap<PathBuf, ParsedAt>>>,
    /// Each function body's hash, while the body is held
    /// ([`crate::sources::declarations_key`]).
    pub(crate) bodies: std::sync::Arc<Mutex<crate::sources::BodyHashes>>,
    /// Kernels compiled, by what they were compiled from
    /// ([`crate::kernels::Compiled`]).
    pub(crate) compiled: std::sync::Arc<Mutex<crate::kernels::Compiled>>,
    /// What the language named of a document's reach, by what it was
    /// named from ([`crate::sources::declarations_within`]).
    pub(crate) scoped: std::sync::Arc<
        Mutex<std::collections::HashMap<String, Result<entelechy_language::Declarations, String>>>,
    >,
    /// What the compiler found wrong in a document's functions, by the
    /// document, its stamp and the declarations it inherits
    /// ([`crate::check`]): a check of many documents sharing one checks it
    /// once.
    pub(crate) functions_checked: std::sync::Arc<
        Mutex<std::collections::HashMap<String, Vec<entelechy_compiler::CompileError>>>,
    >,
    /// What checking a document a checked document uses found, by the
    /// document, its stamp, its reach's key
    /// ([`crate::sources::declarations_keyed`]) and how it is said
    /// ([`crate::check`]): each document used by many is checked once a
    /// sync, and again only when it or what it reaches changes.
    pub(crate) used_checked:
        std::sync::Arc<Mutex<std::collections::HashMap<String, crate::check::UsedChecked>>>,
}

/// A document as parsed, and the stamp of the text it was parsed from.
type ParsedAt = (Stamp, std::sync::Arc<entelechy_language::Document>);

impl dyn Store + '_ {
    /// The document at `path`, parsed; an error names the path. A store
    /// that keeps parsed documents parses a file once per stamp.
    pub fn document(
        &self,
        path: &Path,
    ) -> Result<std::sync::Arc<entelechy_language::Document>, String> {
        let stamp = self.stamp(path);
        if let (Some(parsed), Some(stamp)) = (self.parsed(), stamp)
            && let Some((seen, document)) = parsed
                .documents
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(path)
            && *seen == stamp
        {
            return Ok(std::sync::Arc::clone(document));
        }
        let text = self
            .read_text(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let document = std::sync::Arc::new(
            entelechy_language::parse_document(&text)
                .map_err(|e| format!("{}:{e}", path.display()))?,
        );
        if let (Some(parsed), Some(stamp)) = (self.parsed(), stamp) {
            parsed
                .documents
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(
                    path.to_path_buf(),
                    (stamp, std::sync::Arc::clone(&document)),
                );
        }
        Ok(document)
    }

    /// The file's text.
    pub fn read_text(&self, path: &Path) -> io::Result<String> {
        String::from_utf8(self.read(path)?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "not UTF-8 text"))
    }

    /// Whether anything is at `path`.
    pub fn exists(&self, path: &Path) -> bool {
        self.kind(path).is_some()
    }

    /// Whether a file is at `path`.
    pub fn is_file(&self, path: &Path) -> bool {
        self.kind(path) == Some(Kind::File)
    }

    /// Whether a directory is at `path`.
    pub fn is_dir(&self, path: &Path) -> bool {
        self.kind(path) == Some(Kind::Dir)
    }
}

/// `path` with its `.` and `..` folded away; a `..` past the start is
/// kept.
pub(crate) fn folded(path: &Path) -> PathBuf {
    let mut plain = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                let up = matches!(
                    plain.components().next_back(),
                    None | Some(Component::ParentDir)
                );
                if up || !plain.pop() {
                    plain.push("..");
                }
            }
            Component::CurDir => {}
            other => plain.push(other),
        }
    }
    plain
}

/// The process's file system, and the directories packages are found in
/// by name, as whoever opened the session gave them.
#[derive(Clone, Debug, Default)]
pub struct Disk {
    packages: Vec<PathBuf>,
    parsed: Parsed,
    /// What a sync found of each path, while one runs.
    found: std::sync::Arc<Mutex<Option<Found>>>,
}

/// What a sync found of the file system, by path.
#[derive(Debug, Default)]
struct Found {
    lists: std::collections::HashMap<PathBuf, Vec<PathBuf>>,
    kinds: std::collections::HashMap<PathBuf, Option<Kind>>,
    stamps: std::collections::HashMap<PathBuf, Option<Stamp>>,
    resolved: std::collections::HashMap<PathBuf, PathBuf>,
}

impl Disk {
    /// Forgets what the running sync found: the file system is changing.
    fn forget_found(&self) {
        if let Some(found) = self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            *found = Found::default();
        }
    }

    /// `answer` for `path`, from what the running sync found, or asked of
    /// the file system and kept until the sync ends.
    fn found<T: Clone>(
        &self,
        path: &Path,
        table: fn(&mut Found) -> &mut std::collections::HashMap<PathBuf, T>,
        answer: impl FnOnce() -> T,
    ) -> T {
        let mut found = self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match found.as_mut() {
            Some(found) => table(found)
                .entry(path.to_path_buf())
                .or_insert_with(answer)
                .clone(),
            None => answer(),
        }
    }
}

impl Disk {
    /// The file system, finding packages by name in `packages`.
    pub fn with_packages(packages: Vec<PathBuf>) -> Self {
        Self {
            packages,
            parsed: Parsed::default(),
            found: Default::default(),
        }
    }
}

impl Store for Disk {
    fn packages(&self) -> &[PathBuf] {
        &self.packages
    }

    /// A directory's modification time changes with its entries.
    fn stamps_directories(&self) -> bool {
        true
    }

    fn parsed(&self) -> Option<&Parsed> {
        Some(&self.parsed)
    }

    fn sync_begins(&self) {
        *self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Found::default());
    }

    fn sync_ends(&self) {
        *self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn read_start(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        use std::io::Read;
        let mut start = Vec::with_capacity(bytes);
        std::fs::File::open(path)?
            .take(bytes as u64)
            .read_to_end(&mut start)?;
        Ok(start)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let beside = parent.join(format!(".{name}.writing"));
        // Whatever lies at the working name, a link included, is removed,
        // and the file is made new there, never written through a link.
        match std::fs::remove_file(&beside) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&beside)
            .and_then(|mut file| std::io::Write::write_all(&mut file, bytes))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&beside);
            })?;
        self.forget_found();
        std::fs::rename(&beside, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&beside);
        })
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        self.forget_found();
        std::fs::remove_file(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        if let Some(dir) = to.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        self.forget_found();
        std::fs::rename(from, to)
    }

    fn remove_empty(&self, dir: &Path) -> io::Result<()> {
        if std::fs::read_dir(dir)?.next().is_some() {
            return Ok(());
        }
        self.forget_found();
        std::fs::remove_dir(dir)
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        self.found(
            path,
            |found| &mut found.kinds,
            || {
                let metadata = std::fs::metadata(path).ok()?;
                Some(if metadata.is_dir() {
                    Kind::Dir
                } else {
                    Kind::File
                })
            },
        )
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        // What a directory holds, as its kind and stamp, is looked at once
        // a sync; a failed read is not kept.
        let lock = || {
            self.found
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if let Some(listed) = lock()
            .as_ref()
            .and_then(|found| found.lists.get(dir).cloned())
        {
            return Ok(listed);
        }
        let listed: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        if let Some(found) = lock().as_mut() {
            found.lists.insert(dir.to_path_buf(), listed.clone());
        }
        Ok(listed)
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.found(
            path,
            |found| &mut found.stamps,
            || {
                let metadata = std::fs::metadata(path).ok()?;
                Some(Stamp(Seen::Disk(metadata.modified().ok(), metadata.len())))
            },
        )
    }

    /// The platform's resolved path; under WASI, whose sandbox keeps links
    /// inside its directories, the path made whole.
    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        let known = self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|found| found.resolved.get(path).cloned());
        if let Some(resolved) = known {
            return Ok(resolved);
        }
        let resolved = self.resolved(path)?;
        if let Some(found) = self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            found.resolved.insert(path.to_path_buf(), resolved.clone());
        }
        Ok(resolved)
    }

    fn whole(&self, path: &Path) -> PathBuf {
        match std::env::current_dir() {
            Ok(dir) if path.is_relative() => folded(&dir.join(path)),
            _ => folded(path),
        }
    }
}

impl Disk {
    fn resolved(&self, path: &Path) -> io::Result<PathBuf> {
        match path.canonicalize() {
            Err(error) if error.kind() == io::ErrorKind::Unsupported => {
                if !path.exists() {
                    return Err(io::ErrorKind::NotFound.into());
                }
                Ok(self.whole(path))
            }
            resolved => resolved,
        }
    }
}

/// Files handed in by the caller, kept in memory. A path is its name with
/// `.` and `..` folded; a directory is any path a file is under; every
/// file has one name.
#[derive(Debug, Default)]
pub struct Memory {
    held: Mutex<Held>,
    parsed: Parsed,
    /// Where the files kept elsewhere are read from.
    fetch: Option<Fetch>,
}

/// Reads a file kept elsewhere: its first `bytes` bytes (all of them when
/// it has no more), or an error.
pub type FetchFn = dyn Fn(&Path, usize) -> io::Result<Vec<u8>> + Send + Sync;

/// A [`FetchFn`], shown by name alone.
struct Fetch(Arc<FetchFn>);

impl std::fmt::Debug for Fetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Fetch")
    }
}

/// What a file holds: its bytes, or the count of them, kept elsewhere
/// and read through the store's [`FetchFn`] each time.
#[derive(Debug)]
enum Content {
    Bytes(Vec<u8>),
    Elsewhere(usize),
}

/// The files, each with the number of the write that made it (its stamp).
#[derive(Debug, Default)]
struct Held {
    files: BTreeMap<PathBuf, (Content, u64)>,
    writes: u64,
}

impl Memory {
    /// A store whose files may be kept elsewhere, read through `fetch`.
    pub fn with_fetch(fetch: Arc<FetchFn>) -> Self {
        Self {
            fetch: Some(Fetch(fetch)),
            ..Self::default()
        }
    }

    /// Puts a file at `path`; the next sync reads it as a changed file.
    pub fn put(&self, path: &Path, bytes: Vec<u8>) {
        self.insert(path, Content::Bytes(bytes));
    }

    /// Puts a file of `bytes` bytes at `path` whose bytes stay where they
    /// are kept, read through the store's fetch whenever the file is; the
    /// next sync reads it as a changed file.
    pub fn put_elsewhere(&self, path: &Path, bytes: usize) {
        self.insert(path, Content::Elsewhere(bytes));
    }

    fn insert(&self, path: &Path, content: Content) {
        let mut held = self.held();
        held.writes += 1;
        let version = held.writes;
        held.files.insert(folded(path), (content, version));
    }

    /// The first `bytes` bytes of the file (or all of it), wherever kept.
    fn read_up_to(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        let elsewhere = match self.held().files.get(&folded(path)) {
            Some((Content::Bytes(held), _)) => return Ok(held[..held.len().min(bytes)].to_vec()),
            Some((Content::Elsewhere(count), _)) => (*count).min(bytes),
            None => return Err(io::ErrorKind::NotFound.into()),
        };
        match &self.fetch {
            Some(Fetch(fetch)) => fetch(&folded(path), elsewhere),
            None => Err(io::Error::other(
                "the file is kept elsewhere and the store reads nothing there",
            )),
        }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A store seen through texts not yet written: each of `texts` read as
/// if it stood at its path (new files and the directories they would
/// make included), everything else as `base` has it. What a compose of
/// several documents is checked against before any reaches `base`;
/// nothing is written through it.
pub(crate) struct Overlay<'a> {
    base: &'a dyn Store,
    /// Each text by where its path lands ([`Overlay::landed`]).
    texts: BTreeMap<PathBuf, (Vec<u8>, u64)>,
}

impl<'a> Overlay<'a> {
    pub(crate) fn new(base: &'a dyn Store, texts: &[(PathBuf, String)]) -> Self {
        let mut overlay = Self {
            base,
            texts: BTreeMap::new(),
        };
        for (path, text) in texts {
            let at = overlay.landed(path);
            overlay
                .texts
                .insert(at, (text.as_bytes().to_vec(), crate::history::digest(text)));
        }
        overlay
    }

    /// Where `path` lands in `base`: its nearest directory that exists by
    /// its one name, and the rest of the path after it, so every spelling
    /// of a file not yet written names it alike.
    fn landed(&self, path: &Path) -> PathBuf {
        let mut existing = self.base.whole(path);
        let mut rest = Vec::new();
        while self.base.kind(&existing).is_none() {
            let (Some(up), Some(name)) = (existing.parent(), existing.file_name()) else {
                break;
            };
            rest.push(name.to_owned());
            existing = up.to_path_buf();
        }
        let mut at = self.base.canonical(&existing).unwrap_or(existing);
        for name in rest.iter().rev() {
            at.push(name);
        }
        at
    }

    fn held(&self, path: &Path) -> Option<&(Vec<u8>, u64)> {
        self.texts.get(&self.landed(path))
    }
}

impl Store for Overlay<'_> {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        match self.held(path) {
            Some((bytes, _)) => Ok(bytes.clone()),
            None => self.base.read(path),
        }
    }

    fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("texts being checked are not written"))
    }

    fn remove(&self, _: &Path) -> io::Result<()> {
        Err(io::Error::other("texts being checked are not removed"))
    }

    fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(io::Error::other("texts being checked are not renamed"))
    }

    fn remove_empty(&self, _: &Path) -> io::Result<()> {
        Ok(())
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        if self.held(path).is_some() {
            return Some(Kind::File);
        }
        self.base.kind(path).or_else(|| {
            let at = self.landed(path);
            self.texts
                .keys()
                .any(|file| file.starts_with(&at))
                .then_some(Kind::Dir)
        })
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        let at = self.landed(dir);
        let mut found = self.base.list(dir).unwrap_or_default();
        for file in self.texts.keys() {
            if let Ok(rest) = file.strip_prefix(&at)
                && let Some(first) = rest.components().next()
            {
                let entry = dir.join(first);
                if !found.contains(&entry) {
                    found.push(entry);
                }
            }
        }
        if found.is_empty() && self.base.kind(dir).is_none() {
            return Err(io::ErrorKind::NotFound.into());
        }
        Ok(found)
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        match self.held(path) {
            Some((_, version)) => Some(Stamp(Seen::Memory(*version))),
            None => self.base.stamp(path),
        }
    }

    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        match self.base.kind(path) {
            Some(_) => self.base.canonical(path),
            None if self.kind(path).is_some() => Ok(self.landed(path)),
            None => Err(io::ErrorKind::NotFound.into()),
        }
    }

    fn whole(&self, path: &Path) -> PathBuf {
        self.base.whole(path)
    }

    fn packages(&self) -> &[PathBuf] {
        self.base.packages()
    }
}

impl Store for Memory {
    fn parsed(&self) -> Option<&Parsed> {
        Some(&self.parsed)
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.read_up_to(path, usize::MAX)
    }

    fn read_start(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        self.read_up_to(path, bytes)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.insert(path, Content::Bytes(bytes.to_vec()));
        Ok(())
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        self.held()
            .files
            .remove(&folded(path))
            .map(drop)
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut held = self.held();
        let (bytes, _) = held
            .files
            .remove(&folded(from))
            .ok_or(io::ErrorKind::NotFound)?;
        held.writes += 1;
        let version = held.writes;
        held.files.insert(folded(to), (bytes, version));
        Ok(())
    }

    fn remove_empty(&self, _: &Path) -> io::Result<()> {
        // A directory is there while a file under it is.
        Ok(())
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        let path = folded(path);
        let held = self.held();
        if held.files.contains_key(&path) {
            return Some(Kind::File);
        }
        let empty = path.as_os_str().is_empty();
        held.files
            .keys()
            .any(|file| empty || file.starts_with(&path))
            .then_some(Kind::Dir)
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        let dir = folded(dir);
        let held = self.held();
        let mut found: Vec<PathBuf> = held
            .files
            .keys()
            .filter_map(|file| {
                let rest = file.strip_prefix(&dir).ok()?;
                let first = rest.components().next()?;
                Some(dir.join(first))
            })
            .collect();
        found.dedup();
        if found.is_empty() {
            return Err(io::ErrorKind::NotFound.into());
        }
        Ok(found)
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.held()
            .files
            .get(&folded(path))
            .map(|(_, version)| Stamp(Seen::Memory(*version)))
    }

    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        match self.kind(path) {
            Some(_) => Ok(folded(path)),
            None => Err(io::ErrorKind::NotFound.into()),
        }
    }

    fn whole(&self, path: &Path) -> PathBuf {
        folded(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_kept_elsewhere_is_read_through_the_fetch_each_time() {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&asked);
        let files = Memory::with_fetch(Arc::new(move |path: &Path, bytes: usize| {
            seen.lock().unwrap().push((path.to_path_buf(), bytes));
            Ok(b"0123456789"[..bytes.min(10)].to_vec())
        }));
        files.put_elsewhere(Path::new("a/../big.field"), 10);
        assert_eq!(files.kind(Path::new("big.field")), Some(Kind::File));
        assert_eq!(files.read(Path::new("big.field")).unwrap(), b"0123456789");
        assert_eq!(
            files.read_start(Path::new("big.field"), 4).unwrap(),
            b"0123"
        );
        assert_eq!(
            *asked.lock().unwrap(),
            vec![
                (PathBuf::from("big.field"), 10),
                (PathBuf::from("big.field"), 4)
            ]
        );
        let before = files.stamp(Path::new("big.field"));
        files.put_elsewhere(Path::new("big.field"), 10);
        assert_ne!(
            files.stamp(Path::new("big.field")),
            before,
            "a new put is a change"
        );
        let plain = Memory::default();
        plain.put_elsewhere(Path::new("big.field"), 10);
        assert!(
            plain.read(Path::new("big.field")).is_err(),
            "nothing reads it there"
        );
    }

    #[test]
    fn a_files_start_is_read_alone() {
        let dir =
            std::env::temp_dir().join(format!("entelechy-store-start-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f.txt");
        std::fs::write(&file, "0123456789").unwrap();
        assert_eq!(Disk::default().read_start(&file, 4).unwrap(), b"0123");
        assert_eq!(
            Disk::default().read_start(&file, 64).unwrap(),
            b"0123456789"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_at_the_working_name_is_not_written_through() {
        let dir = std::env::temp_dir().join(format!("entelechy-store-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let elsewhere = dir.join("elsewhere.txt");
        std::fs::write(&elsewhere, "kept").unwrap();
        std::os::unix::fs::symlink(&elsewhere, dir.join(".out.txt.writing")).unwrap();
        Disk::default()
            .write(&dir.join("out.txt"), b"written")
            .unwrap();
        assert_eq!(std::fs::read_to_string(&elsewhere).unwrap(), "kept");
        assert_eq!(
            std::fs::read_to_string(dir.join("out.txt")).unwrap(),
            "written"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_spellings_fold_to_one_name() {
        assert_eq!(
            folded(Path::new("/a/b/../c/./d/../e")),
            PathBuf::from("/a/c/e")
        );
        assert_eq!(folded(Path::new("a/../../b")), PathBuf::from("../b"));
        assert_eq!(folded(Path::new("../../b")), PathBuf::from("../../b"));
    }

    /// Files are found by any spelling, imply their directories, and get a
    /// new stamp when put again.
    #[test]
    fn memory_holds_files_and_the_directories_they_imply() {
        let memory = Memory::default();
        memory.put(Path::new("scene/main.esn"), b"a".to_vec());
        memory.put(Path::new("scene/kernels/k.ent"), b"k".to_vec());
        let store: &dyn Store = &memory;
        assert_eq!(
            store
                .read_text(Path::new("scene/kernels/../main.esn"))
                .unwrap(),
            "a"
        );
        assert!(store.is_dir(Path::new("scene/kernels")));
        assert!(store.is_file(Path::new("./scene/main.esn")));
        assert!(!store.exists(Path::new("scene/other")));
        let mut listed = store.list(Path::new("scene")).unwrap();
        listed.sort();
        assert_eq!(
            listed,
            [
                PathBuf::from("scene/kernels"),
                PathBuf::from("scene/main.esn")
            ]
        );
        let before = store.stamp(Path::new("scene/main.esn"));
        memory.put(Path::new("scene/main.esn"), b"b".to_vec());
        assert_ne!(store.stamp(Path::new("scene/main.esn")), before);
    }
}

#[cfg(test)]
mod parsed_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Files held in memory, counting how often each is read.
    #[derive(Default)]
    struct Counting {
        files: Memory,
        reads: AtomicUsize,
        parsed: Parsed,
    }

    impl Store for Counting {
        fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.files.read(path)
        }
        fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
            self.files.write(path, bytes)
        }
        fn remove(&self, path: &Path) -> io::Result<()> {
            self.files.remove(path)
        }
        fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            self.files.rename(from, to)
        }
        fn remove_empty(&self, dir: &Path) -> io::Result<()> {
            self.files.remove_empty(dir)
        }
        fn kind(&self, path: &Path) -> Option<Kind> {
            self.files.kind(path)
        }
        fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
            self.files.list(dir)
        }
        fn stamp(&self, path: &Path) -> Option<Stamp> {
            self.files.stamp(path)
        }
        fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
            self.files.canonical(path)
        }
        fn whole(&self, path: &Path) -> PathBuf {
            self.files.whole(path)
        }
        fn parsed(&self) -> Option<&Parsed> {
            Some(&self.parsed)
        }
    }

    /// A document is parsed once while its file is unchanged, and again
    /// once it changes.
    #[test]
    fn a_document_is_parsed_once_per_stamp() {
        let store = Counting::default();
        let path = Path::new("/s/lib.ent");
        store
            .files
            .put(path, b"fn f(x: Float) -> Float { return x; }".to_vec());
        let store: &dyn Store = &store;
        let first = store.document(path).unwrap();
        let again = store.document(path).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        store
            .write(path, b"fn g(x: Float) -> Float { return x; }")
            .unwrap();
        let changed = store.document(path).unwrap();
        assert_eq!(changed.functions[0].name.value, "g");
    }
}
