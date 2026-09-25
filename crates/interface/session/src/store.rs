//! Where a session's documents are. The session reads its scene, the
//! documents the scene uses, the files its inputs name, and writes what
//! its outputs and its requests write, all through a [`Store`]: the
//! disk, where a process has one, or [`Memory`], the files its caller
//! handed in, where it has none (a page) or wants none. Which one it is
//! is the caller's choice; nothing the session does depends on it.

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

/// What a file looked like when it was read: whenever the file changes,
/// so does its stamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp(Seen);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seen {
    /// A file on disk: its modification time and length.
    Disk(Option<std::time::SystemTime>, u64),
    /// A file handed in: the how-manieth it was.
    Memory(u64),
}

/// Files by path, read and written whole.
pub trait Store: Send + Sync {
    /// The file's bytes.
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    /// Replaces the file whole, making the directories it needs: a
    /// reader never sees half of it.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;
    /// What is at `path`, if anything.
    fn kind(&self, path: &Path) -> Option<Kind>;
    /// What a directory holds, in no order.
    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>>;
    /// The file's stamp now; `None` when there is none.
    fn stamp(&self, path: &Path) -> Option<Stamp>;
    /// The file's one name, whatever the spelling that reached it; an
    /// error when nothing is there.
    fn canonical(&self, path: &Path) -> io::Result<PathBuf>;
    /// `path` made whole: absolute where the store has a current
    /// directory, `.` and `..` folded, links unread.
    fn whole(&self, path: &Path) -> PathBuf;
}

impl dyn Store + '_ {
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

/// The process's file system.
#[derive(Clone, Copy, Debug, Default)]
pub struct Disk;

impl Store for Disk {
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

    /// Where the platform resolves links, the path it resolves; where it
    /// does not (WebAssembly under WASI, whose sandbox lets no link lead
    /// out of the directories it was given), the path made whole.
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

/// Files handed in by the caller, kept in memory: a page's, which has no
/// disk, or a program's that carries its scenes inside itself. A path is
/// its name with `.` and `..` folded; a directory is any path some file
/// is under. There are no links, so every file has one name.
#[derive(Debug, Default)]
pub struct Memory {
    held: Mutex<Held>,
}

/// The files, each with the write that made it, and how many writes
/// there have been: a file's stamp is its write's number.
#[derive(Debug, Default)]
struct Held {
    files: BTreeMap<PathBuf, (Vec<u8>, u64)>,
    writes: u64,
}

impl Memory {
    /// Puts a file, replacing what was at `path`; the session reads it
    /// on its next sync, as it would a file changed on disk.
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

    /// Files handed in are found by any spelling, their directories are
    /// what is under them, and a file put again has a new stamp.
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
