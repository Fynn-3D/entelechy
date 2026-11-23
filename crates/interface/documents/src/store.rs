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
    #[cfg(feature = "documents")]
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

/// The bytes a file read as words is read in at a time
/// ([`Store::read_words`]).
const WORDS_PIECE: usize = 1 << 20;

/// Appends little-endian words of `bytes`, the last padded with zeros.
fn extend_words(words: &mut Vec<u32>, bytes: &[u8]) {
    words.extend(bytes.chunks(4).map(|w| {
        let mut word = [0u8; 4];
        word[..w.len()].copy_from_slice(w);
        u32::from_le_bytes(word)
    }));
}

/// A file being read off the caller's thread ([`Store::read_later`]).
pub trait Later: Send {
    /// The file once it is in, or why it is not; `None` while it is
    /// still on its way. Answers once.
    fn poll(&mut self) -> Option<io::Result<Box<dyn Arrived>>>;
}

/// A file read whole, held where it was read into, its bytes taken as
/// they are needed.
pub trait Arrived: Send {
    /// Its length in bytes.
    fn len(&self) -> usize;
    /// Whether it holds no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Its first `bytes` bytes, or all of it when it is shorter.
    fn start(&self, bytes: usize) -> Vec<u8>;
    /// Its bytes from `from` on as little-endian words, the last padded
    /// with zeros, and how many bytes they are.
    fn words(&self, from: usize) -> (Vec<u32>, usize);
}

/// A file's bytes in hand, as an [`Arrived`].
struct Bytes(Vec<u8>);

impl Arrived for Bytes {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn start(&self, bytes: usize) -> Vec<u8> {
        self.0[..bytes.min(self.0.len())].to_vec()
    }

    fn words(&self, from: usize) -> (Vec<u32>, usize) {
        let body = self.0.get(from..).unwrap_or_default();
        let mut words = Vec::with_capacity(body.len().div_ceil(4));
        extend_words(&mut words, body);
        (words, body.len())
    }
}

/// A file read already: a [`Later`] that answers at once.
struct Now(Option<io::Result<Box<dyn Arrived>>>);

impl Later for Now {
    fn poll(&mut self) -> Option<io::Result<Box<dyn Arrived>>> {
        self.0.take()
    }
}

/// A [`Later`] of a read finished now.
pub(crate) fn read_now(read: io::Result<Vec<u8>>) -> Box<dyn Later> {
    Box::new(Now(Some(
        read.map(|bytes| Box::new(Bytes(bytes)) as Box<dyn Arrived>),
    )))
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
    /// The file's bytes from `from` on as little-endian words, the last
    /// padded with zeros, and how many bytes they are: a body of words
    /// read as words, never held as bytes beside them where the store
    /// can read it a piece at a time.
    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        let all = self.read(path)?;
        let body = all.get(from..).unwrap_or_default();
        let mut words = Vec::with_capacity(body.len().div_ceil(4));
        extend_words(&mut words, body);
        Ok((words, body.len()))
    }
    /// Reads the file off the caller's thread where the store can, else
    /// now: what the caller asks after says when it is in.
    fn read_later(&self, path: &Path) -> Box<dyn Later> {
        read_now(self.read(path))
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
    #[cfg(feature = "documents")]
    fn parsed(&self) -> Option<&Parsed> {
        None
    }
    /// Whether a directory's stamp changes whenever what it holds does
    /// (an entry added, removed or renamed).
    fn stamps_directories(&self) -> bool {
        false
    }
    /// A sync begins, or another run that walks the same files from every
    /// document (a check): until it ends, the store may answer a path's
    /// kind, stamp and resolved path as it first found them. A file
    /// changed meanwhile is seen by the next run. Runs nest; the last to
    /// end ends them.
    fn sync_begins(&self) {}
    /// The run ends: every path is looked at afresh.
    fn sync_ends(&self) {}
}

/// Documents parsed once per stamp: a document read again unchanged is
/// not parsed again; and what is worked out of them once, by content,
/// for every session reading through the store.
#[cfg(feature = "documents")]
#[derive(Clone, Debug, Default)]
pub struct Parsed {
    documents: std::sync::Arc<Mutex<BTreeMap<PathBuf, ParsedAt>>>,
    /// Each function body's hash, while the body is held
    /// ([`crate::sources::declarations_key`]).
    pub(crate) bodies: std::sync::Arc<Mutex<crate::sources::BodyHashes>>,
    /// Kernels compiled, by what they were compiled from
    /// ([`crate::kernels::Compiled`]).
    pub(crate) compiled: std::sync::Arc<Mutex<crate::kernels::Compiled>>,
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
#[cfg(feature = "documents")]
type ParsedAt = (Stamp, std::sync::Arc<entelechy_language::Document>);

impl dyn Store + '_ {
    /// The document at `path`, parsed; an error names the path. A store
    /// that keeps parsed documents parses a file once per stamp.
    #[cfg(feature = "documents")]
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
    #[cfg(feature = "documents")]
    parsed: Parsed,
    /// What a sync found of each path, while one runs.
    found: std::sync::Arc<Mutex<Option<Found>>>,
    /// The runs open ([`Store::sync_begins`]).
    runs: std::sync::Arc<std::sync::atomic::AtomicUsize>,
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
            #[cfg(feature = "documents")]
            parsed: Parsed::default(),
            found: Default::default(),
            runs: Default::default(),
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

    #[cfg(feature = "documents")]
    fn parsed(&self) -> Option<&Parsed> {
        Some(&self.parsed)
    }

    fn sync_begins(&self) {
        if self.runs.fetch_add(1, std::sync::atomic::Ordering::AcqRel) == 0 {
            *self
                .found
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Found::default());
        }
    }

    fn sync_ends(&self) {
        let ended = self.runs.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |runs| runs.checked_sub(1),
        );
        if ended == Ok(1) {
            *self
                .found
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        }
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

    #[cfg(not(target_arch = "wasm32"))]
    fn read_later(&self, path: &Path) -> Box<dyn Later> {
        /// A read on a thread of its own.
        struct Reading(std::sync::mpsc::Receiver<io::Result<Vec<u8>>>);
        impl Later for Reading {
            fn poll(&mut self) -> Option<io::Result<Box<dyn Arrived>>> {
                match self.0.try_recv() {
                    Ok(read) => read_now(read).poll(),
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        Some(Err(io::Error::other("the read stopped")))
                    }
                }
            }
        }
        let (sent, read) = std::sync::mpsc::channel();
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let _ = sent.send(std::fs::read(path));
        });
        Box::new(Reading(read))
    }

    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        use std::io::{Read, Seek};
        let mut file = std::fs::File::open(path)?;
        let length = file.metadata()?.len() as usize;
        let body = length.saturating_sub(from);
        file.seek(io::SeekFrom::Start(from as u64))?;
        let mut words = Vec::with_capacity(body.div_ceil(4));
        let mut piece = vec![0u8; body.min(WORDS_PIECE)];
        let mut read = 0;
        while read < body {
            let want = (body - read).min(WORDS_PIECE);
            file.read_exact(&mut piece[..want])?;
            extend_words(&mut words, &piece[..want]);
            read += want;
        }
        Ok((words, body))
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
/// file has one name. A package named alone is found in `packages/`.
#[derive(Debug, Default)]
pub struct Memory {
    held: Mutex<Held>,
    #[cfg(feature = "documents")]
    parsed: Parsed,
    /// Where the files kept elsewhere are read from.
    fetch: Option<Fetch>,
    /// Where they are read from off the caller's thread, if anywhere.
    fetch_later: Option<FetchLater>,
}

/// Reads a file kept elsewhere into `out` from byte `from` on, answering
/// the bytes read (fewer where the file ends first), or an error. A read
/// from the start reads the file anew; one further on may be answered
/// from what the last read of that file found, as such a file does not
/// change.
pub type FetchFn = dyn Fn(&Path, usize, &mut [u8]) -> io::Result<usize> + Send + Sync;

/// Starts reading a file kept elsewhere off the caller's thread
/// ([`Store::read_later`]).
pub type FetchLaterFn = dyn Fn(&Path) -> Box<dyn Later> + Send + Sync;

/// A [`FetchFn`], shown by name alone.
struct Fetch(Arc<FetchFn>);

/// A [`FetchLaterFn`], shown by name alone.
struct FetchLater(Arc<FetchLaterFn>);

impl std::fmt::Debug for FetchLater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FetchLater")
    }
}

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
    /// By [`key`]: a string compares far faster than a path, and the
    /// files under a directory lie together in its order.
    files: BTreeMap<String, (Content, u64)>,
    /// Each directory some file is under, with how many files are: what
    /// `kind` answers without walking the files.
    dirs: std::collections::HashMap<String, usize>,
    writes: u64,
}

/// A path as the store keys it: folded, its parts joined by `/`.
fn key(path: &Path) -> String {
    folded(path).to_string_lossy().replace('\\', "/")
}

/// The directories `key` lies under, innermost first.
fn dirs_of(key: &str) -> impl Iterator<Item = &str> {
    key.match_indices('/')
        .rev()
        .map(|(at, _)| &key[..at])
        .filter(|dir| !dir.is_empty())
}

impl Held {
    /// Puts `content` at `key` as a new write.
    fn put(&mut self, key: String, content: Content) {
        self.writes += 1;
        let version = self.writes;
        for dir in dirs_of(&key) {
            if !self.files.contains_key(&key) {
                *self.dirs.entry(dir.to_owned()).or_default() += 1;
            }
        }
        self.files.insert(key, (content, version));
    }

    /// Takes the file at `key` out.
    fn take(&mut self, key: &str) -> Option<Content> {
        let (content, _) = self.files.remove(key)?;
        for dir in dirs_of(key) {
            if let Some(count) = self.dirs.get_mut(dir) {
                *count -= 1;
                if *count == 0 {
                    self.dirs.remove(dir);
                }
            }
        }
        Some(content)
    }
}

impl Memory {
    /// A store whose files may be kept elsewhere, read through `fetch`.
    pub fn with_fetch(fetch: Arc<FetchFn>) -> Self {
        Self {
            fetch: Some(Fetch(fetch)),
            ..Self::default()
        }
    }

    /// This store reading its files kept elsewhere off the caller's thread
    /// through `later` ([`Store::read_later`]).
    pub fn fetching_later(mut self, later: Arc<FetchLaterFn>) -> Self {
        self.fetch_later = Some(FetchLater(later));
        self
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
        self.held().put(key(path), content);
    }

    /// The first `bytes` bytes of the file (or all of it), wherever kept.
    fn read_up_to(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        let key = key(path);
        let elsewhere = match self.held().files.get(&key) {
            Some((Content::Bytes(held), _)) => return Ok(held[..held.len().min(bytes)].to_vec()),
            Some((Content::Elsewhere(count), _)) => (*count).min(bytes),
            None => return Err(io::ErrorKind::NotFound.into()),
        };
        match &self.fetch {
            Some(Fetch(fetch)) => {
                let mut out = vec![0u8; elsewhere];
                let got = fetch(Path::new(&key), 0, &mut out)?;
                out.truncate(got);
                Ok(out)
            }
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

/// Where `path` lands in `base`: its nearest directory that exists by its
/// one name, and the rest of the path after it, so every spelling of a
/// file not yet written names it alike.
fn landed(base: &dyn Store, path: &Path) -> PathBuf {
    let mut existing = base.whole(path);
    let mut rest = Vec::new();
    while base.kind(&existing).is_none() {
        let (Some(up), Some(name)) = (existing.parent(), existing.file_name()) else {
            break;
        };
        rest.push(name.to_owned());
        existing = up.to_path_buf();
    }
    let mut at = base.canonical(&existing).unwrap_or(existing);
    for name in rest.iter().rev() {
        at.push(name);
    }
    at
}

/// What a [`Kept`] store holds, by where each path lands: a file written,
/// with the number of the write, or `None` for one removed.
type KeptFiles = BTreeMap<PathBuf, Option<(Vec<u8>, u64)>>;

/// A store whose writes are kept beside `base` and never reach it: each
/// file written, removed or renamed through it is held here, everything
/// else read as `base` has it. What a replay that allows writes runs
/// over, so a run changes nothing on disk and every run starts from the
/// same files.
pub struct Kept {
    base: Arc<dyn Store>,
    /// By where each path lands ([`landed`]).
    held: Mutex<KeptFiles>,
    writes: std::sync::atomic::AtomicU64,
}

impl Kept {
    /// `base`, its writes to be kept.
    pub fn new(base: Arc<dyn Store>) -> Self {
        Self {
            base,
            held: Mutex::default(),
            // Above every number a store hands out itself, so a file
            // written here never takes a stamp it had in `base`.
            writes: std::sync::atomic::AtomicU64::new(1 << 63),
        }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, KeptFiles> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn landed(&self, path: &Path) -> PathBuf {
        landed(self.base.as_ref(), path)
    }

    /// What is held for `path`: `Some(Some(..))` a file written here,
    /// `Some(None)` one removed here, `None` none, read from `base`.
    fn found(&self, path: &Path) -> Option<Option<(Vec<u8>, u64)>> {
        // Where the path lands is asked of `base` before what is held is
        // locked: asked under the lock, every reader of the store waits
        // for every other's look at the file system.
        let at = self.landed(path);
        self.held().get(&at).cloned()
    }

    fn put(&self, path: &Path, bytes: Option<Vec<u8>>) {
        let write = self
            .writes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let at = self.landed(path);
        self.held().insert(at, bytes.map(|bytes| (bytes, write)));
    }
}

impl Store for Kept {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        match self.found(path) {
            Some(Some((bytes, _))) => Ok(bytes),
            Some(None) => Err(io::ErrorKind::NotFound.into()),
            None => self.base.read(path),
        }
    }

    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        match self.found(path) {
            Some(Some((bytes, _))) => {
                let body = bytes.get(from..).unwrap_or_default();
                let mut words = Vec::with_capacity(body.len().div_ceil(4));
                extend_words(&mut words, body);
                Ok((words, body.len()))
            }
            Some(None) => Err(io::ErrorKind::NotFound.into()),
            None => self.base.read_words(path, from),
        }
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.put(path, Some(bytes.to_vec()));
        Ok(())
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        if self.kind(path) != Some(Kind::File) {
            return Err(io::ErrorKind::NotFound.into());
        }
        self.put(path, None);
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let bytes = self.read(from)?;
        self.put(to, Some(bytes));
        self.put(from, None);
        Ok(())
    }

    fn remove_empty(&self, _: &Path) -> io::Result<()> {
        Ok(())
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        match self.found(path) {
            Some(Some(_)) => return Some(Kind::File),
            Some(None) => return None,
            None => {}
        }
        self.base.kind(path).or_else(|| {
            let at = self.landed(path);
            self.held()
                .iter()
                .any(|(file, held)| held.is_some() && file.starts_with(&at))
                .then_some(Kind::Dir)
        })
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        let at = self.landed(dir);
        let mut found = self.base.list(dir).unwrap_or_default();
        found.retain(|entry| !matches!(self.found(entry), Some(None)));
        for (file, held) in self.held().iter() {
            if held.is_none() {
                continue;
            }
            if let Ok(rest) = file.strip_prefix(&at)
                && let Some(first) = rest.components().next()
            {
                let entry = dir.join(first);
                if !found.contains(&entry) {
                    found.push(entry);
                }
            }
        }
        if found.is_empty() && self.kind(dir).is_none() {
            return Err(io::ErrorKind::NotFound.into());
        }
        Ok(found)
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        match self.found(path) {
            Some(Some((_, write))) => Some(Stamp(Seen::Memory(write))),
            Some(None) => None,
            None => self.base.stamp(path),
        }
    }

    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        match self.found(path) {
            Some(Some(_)) => Ok(self.landed(path)),
            Some(None) => Err(io::ErrorKind::NotFound.into()),
            None => self.base.canonical(path),
        }
    }

    fn whole(&self, path: &Path) -> PathBuf {
        self.base.whole(path)
    }

    fn packages(&self) -> &[PathBuf] {
        self.base.packages()
    }

    #[cfg(feature = "documents")]
    fn parsed(&self) -> Option<&Parsed> {
        self.base.parsed()
    }

    fn stamps_directories(&self) -> bool {
        // A file written here changes no directory's stamp in `base`.
        false
    }

    fn sync_begins(&self) {
        self.base.sync_begins();
    }

    fn sync_ends(&self) {
        self.base.sync_ends();
    }
}

/// A store seen through texts not yet written: each of `texts` read as
/// if it stood at its path (new files and the directories they would
/// make included), everything else as `base` has it. What a compose of
/// several documents is checked against before any reaches `base`;
/// nothing is written through it.
#[cfg(feature = "documents")]
pub struct Overlay<'a> {
    base: &'a dyn Store,
    /// Each text by where its path lands ([`Overlay::landed`]).
    texts: BTreeMap<PathBuf, (Vec<u8>, u64)>,
}

#[cfg(feature = "documents")]
impl<'a> Overlay<'a> {
    /// `base` seen through `texts`.
    pub fn new(base: &'a dyn Store, texts: &[(PathBuf, String)]) -> Self {
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

    fn landed(&self, path: &Path) -> PathBuf {
        landed(self.base, path)
    }

    fn held(&self, path: &Path) -> Option<&(Vec<u8>, u64)> {
        self.texts.get(&self.landed(path))
    }
}

#[cfg(feature = "documents")]
impl Store for Overlay<'_> {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        match self.held(path) {
            Some((bytes, _)) => Ok(bytes.clone()),
            None => self.base.read(path),
        }
    }

    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        match self.held(path) {
            Some((bytes, _)) => {
                let body = bytes.get(from..).unwrap_or_default();
                let mut words = Vec::with_capacity(body.len().div_ceil(4));
                extend_words(&mut words, body);
                Ok((words, body.len()))
            }
            None => self.base.read_words(path, from),
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
    #[cfg(feature = "documents")]
    fn parsed(&self) -> Option<&Parsed> {
        Some(&self.parsed)
    }

    /// Files handed in have no program to be beside: a package named
    /// alone is looked for in `packages/` among them.
    fn packages(&self) -> &[PathBuf] {
        static AMONG: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
        AMONG.get_or_init(|| vec![PathBuf::from("packages")])
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.read_up_to(path, usize::MAX)
    }

    fn read_start(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        self.read_up_to(path, bytes)
    }

    fn read_later(&self, path: &Path) -> Box<dyn Later> {
        let key = key(path);
        let elsewhere = matches!(
            self.held().files.get(&key),
            Some((Content::Elsewhere(_), _))
        );
        match &self.fetch_later {
            Some(FetchLater(later)) if elsewhere => later(Path::new(&key)),
            _ => read_now(self.read(path)),
        }
    }

    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        let key = key(path);
        let length = match self.held().files.get(&key) {
            Some((Content::Bytes(held), _)) => {
                let body = held.get(from..).unwrap_or_default();
                let mut words = Vec::with_capacity(body.len().div_ceil(4));
                extend_words(&mut words, body);
                return Ok((words, body.len()));
            }
            Some((Content::Elsewhere(count), _)) => *count,
            None => return Err(io::ErrorKind::NotFound.into()),
        };
        let Some(Fetch(fetch)) = &self.fetch else {
            return Err(io::Error::other(
                "the file is kept elsewhere and the store reads nothing there",
            ));
        };
        // A piece at a time, each into the words as it comes.
        let body = length.saturating_sub(from);
        let mut words = Vec::with_capacity(body.div_ceil(4));
        let mut piece = vec![0u8; body.min(WORDS_PIECE)];
        let mut at = from;
        while at < length {
            let want = (length - at).min(WORDS_PIECE);
            let got = fetch(Path::new(&key), at, &mut piece[..want])?;
            if got < want {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            extend_words(&mut words, &piece[..want]);
            at += want;
        }
        Ok((words, body))
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.insert(path, Content::Bytes(bytes.to_vec()));
        Ok(())
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        self.held()
            .take(&key(path))
            .map(drop)
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut held = self.held();
        let content = held.take(&key(from)).ok_or(io::ErrorKind::NotFound)?;
        held.put(key(to), content);
        Ok(())
    }

    fn remove_empty(&self, _: &Path) -> io::Result<()> {
        // A directory is there while a file under it is.
        Ok(())
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        let key = key(path);
        let held = self.held();
        if held.files.contains_key(&key) {
            return Some(Kind::File);
        }
        let dir = if key.is_empty() {
            !held.files.is_empty()
        } else {
            held.dirs.contains_key(&key)
        };
        dir.then_some(Kind::Dir)
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        let dir = key(dir);
        let prefix = if dir.is_empty() {
            dir
        } else {
            format!("{dir}/")
        };
        let held = self.held();
        // The files under `dir` lie together in the map's order.
        let mut found: Vec<PathBuf> = held
            .files
            .range::<str, _>((
                std::ops::Bound::Included(prefix.as_str()),
                std::ops::Bound::Unbounded,
            ))
            .map(|(file, _)| file)
            .take_while(|file| file.starts_with(&prefix))
            .filter_map(|file| {
                let first = file[prefix.len()..].split('/').next()?;
                Some(PathBuf::from(format!("{prefix}{first}")))
            })
            .collect();
        // In a path's order, which a string's differs from.
        found.sort();
        found.dedup();
        if found.is_empty() {
            return Err(io::ErrorKind::NotFound.into());
        }
        Ok(found)
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.held()
            .files
            .get(&key(path))
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
    fn memory_knows_its_directories_as_files_come_and_go() {
        let files = Memory::default();
        files.put(Path::new("a/b/one.ent"), b"1".to_vec());
        files.put(Path::new("a/b-x/two.ent"), b"2".to_vec());
        files.put(Path::new("a/three.ent"), b"3".to_vec());
        assert_eq!(files.kind(Path::new("a")), Some(Kind::Dir));
        assert_eq!(files.kind(Path::new("a/b")), Some(Kind::Dir));
        assert_eq!(files.kind(Path::new("a/b/one.ent")), Some(Kind::File));
        assert_eq!(files.kind(Path::new("a/c")), None);
        assert_eq!(files.kind(Path::new("")), Some(Kind::Dir));
        assert_eq!(
            files.list(Path::new("a")).unwrap(),
            vec![
                PathBuf::from("a/b"),
                PathBuf::from("a/b-x"),
                PathBuf::from("a/three.ent")
            ]
        );
        assert_eq!(
            files.list(Path::new("a/b")).unwrap(),
            vec![PathBuf::from("a/b/one.ent")]
        );
        assert!(files.list(Path::new("a/three.ent")).is_err());
        files
            .rename(Path::new("a/b/one.ent"), Path::new("c/one.ent"))
            .unwrap();
        assert_eq!(
            files.kind(Path::new("a/b")),
            None,
            "nothing is under it now"
        );
        assert_eq!(files.kind(Path::new("c")), Some(Kind::Dir));
        files.remove(Path::new("a/b-x/two.ent")).unwrap();
        files.remove(Path::new("a/three.ent")).unwrap();
        assert_eq!(files.kind(Path::new("a")), None);
        files.put(Path::new("c/one.ent"), b"again".to_vec());
        files.remove(Path::new("c/one.ent")).unwrap();
        assert_eq!(
            files.kind(Path::new("c")),
            None,
            "a file put twice is counted once"
        );
        assert_eq!(files.kind(Path::new("")), None);
    }

    #[test]
    fn a_file_kept_elsewhere_is_read_through_the_fetch_each_time() {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&asked);
        let files =
            Memory::with_fetch(Arc::new(move |path: &Path, from: usize, out: &mut [u8]| {
                seen.lock()
                    .unwrap()
                    .push((path.to_path_buf(), from, out.len()));
                let got = &b"0123456789"[from.min(10)..(from + out.len()).min(10)];
                out[..got.len()].copy_from_slice(got);
                Ok(got.len())
            }));
        files.put_elsewhere(Path::new("a/../big.field"), 10);
        assert_eq!(files.kind(Path::new("big.field")), Some(Kind::File));
        assert_eq!(files.read(Path::new("big.field")).unwrap(), b"0123456789");
        assert_eq!(
            files.read_start(Path::new("big.field"), 4).unwrap(),
            b"0123"
        );
        // Read as words from a byte on, the last padded with zeros.
        let (words, bytes) = files.read_words(Path::new("big.field"), 3).unwrap();
        assert_eq!(bytes, 7);
        assert_eq!(
            words,
            [u32::from_le_bytes(*b"3456"), u32::from_le_bytes(*b"789\0")]
        );
        assert_eq!(
            *asked.lock().unwrap(),
            vec![
                (PathBuf::from("big.field"), 0, 10),
                (PathBuf::from("big.field"), 0, 4),
                (PathBuf::from("big.field"), 3, 7)
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

#[cfg(all(test, feature = "documents"))]
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
