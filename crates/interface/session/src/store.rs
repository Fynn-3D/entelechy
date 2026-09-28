//! Where a session's documents are. Every session read and write goes
//! through a [`Store`]: the [`Disk`], or [`Memory`], files the caller
//! handed in (a page). The caller chooses; the session behaves the same.

use std::collections::BTreeMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

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
    /// Replaces the file atomically, making the directories it needs.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;
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
}

/// Documents parsed once per stamp: a document read again unchanged is
/// not parsed again.
#[derive(Clone, Debug, Default)]
pub struct Parsed(std::sync::Arc<Mutex<BTreeMap<PathBuf, ParsedAt>>>);

/// A document as parsed, and the stamp of the text it was parsed from.
type ParsedAt = (Stamp, std::sync::Arc<orla_language::Document>);

impl dyn Store + '_ {
    /// The document at `path`, parsed; an error names the path. A store
    /// that keeps parsed documents parses a file once per stamp.
    pub fn document(&self, path: &Path) -> Result<std::sync::Arc<orla_language::Document>, String> {
        let stamp = self.stamp(path);
        if let (Some(parsed), Some(stamp)) = (self.parsed(), stamp)
            && let Some((seen, document)) = parsed
                .0
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
            orla_language::parse_document(&text).map_err(|e| format!("{}:{e}", path.display()))?,
        );
        if let (Some(parsed), Some(stamp)) = (self.parsed(), stamp) {
            parsed
                .0
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
fn folded(path: &Path) -> PathBuf {
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
}

impl Disk {
    /// The file system, finding packages by name in `packages`.
    pub fn with_packages(packages: Vec<PathBuf>) -> Self {
        Self {
            packages,
            parsed: Parsed::default(),
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

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
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
        std::fs::write(&beside, bytes)?;
        std::fs::rename(&beside, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&beside);
        })
    }

    fn kind(&self, path: &Path) -> Option<Kind> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(if metadata.is_dir() {
            Kind::Dir
        } else {
            Kind::File
        })
    }

    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        Ok(std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect())
    }

    fn stamp(&self, path: &Path) -> Option<Stamp> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(Stamp(Seen::Disk(metadata.modified().ok(), metadata.len())))
    }

    /// The platform's resolved path; under WASI, whose sandbox keeps links
    /// inside its directories, the path made whole.
    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
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

    fn whole(&self, path: &Path) -> PathBuf {
        match std::env::current_dir() {
            Ok(dir) if path.is_relative() => folded(&dir.join(path)),
            _ => folded(path),
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
}

/// The files, each with the number of the write that made it (its stamp).
#[derive(Debug, Default)]
struct Held {
    files: BTreeMap<PathBuf, (Vec<u8>, u64)>,
    writes: u64,
}

impl Memory {
    /// Puts a file at `path`; the next sync reads it as a changed file.
    pub fn put(&self, path: &Path, bytes: Vec<u8>) {
        let _ = self.write(path, &bytes);
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Store for Memory {
    fn parsed(&self) -> Option<&Parsed> {
        Some(&self.parsed)
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.held()
            .files
            .get(&folded(path))
            .map(|(bytes, _)| bytes.clone())
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let mut held = self.held();
        held.writes += 1;
        let version = held.writes;
        held.files.insert(folded(path), (bytes.to_vec(), version));
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
        memory.put(Path::new("scene/main.osn"), b"a".to_vec());
        memory.put(Path::new("scene/kernels/k.orla"), b"k".to_vec());
        let store: &dyn Store = &memory;
        assert_eq!(
            store
                .read_text(Path::new("scene/kernels/../main.osn"))
                .unwrap(),
            "a"
        );
        assert!(store.is_dir(Path::new("scene/kernels")));
        assert!(store.is_file(Path::new("./scene/main.osn")));
        assert!(!store.exists(Path::new("scene/other")));
        let mut listed = store.list(Path::new("scene")).unwrap();
        listed.sort();
        assert_eq!(
            listed,
            [
                PathBuf::from("scene/kernels"),
                PathBuf::from("scene/main.osn")
            ]
        );
        let before = store.stamp(Path::new("scene/main.osn"));
        memory.put(Path::new("scene/main.osn"), b"b".to_vec());
        assert_ne!(store.stamp(Path::new("scene/main.osn")), before);
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
        let path = Path::new("/s/lib.orla");
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
