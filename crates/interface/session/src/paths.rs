//! A file's one name. Where the platform resolves links, it is the path
//! the platform resolves; where it does not (WebAssembly under WASI,
//! whose sandbox lets no link lead out of the directories it was given),
//! it is the path made whole and its `.` and `..` folded away. A file
//! reached by two spellings is one file either way.

use std::io;
use std::path::{Component, Path, PathBuf};

/// `path`'s one name; an error when nothing is there.
pub(crate) fn canonical(path: &Path) -> io::Result<PathBuf> {
    match path.canonicalize() {
        Err(error) if error.kind() == io::ErrorKind::Unsupported => {
            if !path.exists() {
                return Err(io::ErrorKind::NotFound.into());
            }
            Ok(folded(path))
        }
        resolved => resolved,
    }
}

/// `path` made whole and its `.` and `..` folded away, links unread.
fn folded(path: &Path) -> PathBuf {
    let whole = match std::env::current_dir() {
        Ok(dir) if path.is_relative() => dir.join(path),
        _ => path.to_path_buf(),
    };
    let mut plain = PathBuf::new();
    for component in whole.components() {
        match component {
            Component::ParentDir => {
                if !plain.pop() {
                    plain.push("..");
                }
            }
            Component::CurDir => {}
            other => plain.push(other),
        }
    }
    plain
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
        assert_eq!(folded(Path::new("../x")), {
            let mut up = std::env::current_dir().unwrap();
            up.pop();
            up.join("x")
        });
    }
}
