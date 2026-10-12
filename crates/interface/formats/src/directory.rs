//! A directory's files as one `Rows<Int>`, and the operations a scene
//! asks on them, as one too: what `input x: Rows<Int> = files "dir/";`
//! holds and what `output y <- z to files "dir/";` carries.
//!
//! Files, in path order: `[n, (path_len, size, path.., bytes..)*]`, a word
//! a count, a path's UTF-8 byte or a file's byte. Operations:
//! `[n, (kind, path_len, size, path.., data..)*]`, kind 0 writing `data`
//! as the file (made when absent), 1 removing the file (`size` 0), 2
//! renaming it to the path `data` spells; words after the `n`th are
//! not read. A path is relative to the directory, its parts parted by
//! `/`, none empty, `.`, `..` or hidden (beginning with `.`).

use entelechy_host::{Domain, FieldData, Scalar, Schema, Shape};

/// The operation writing a file, by its number.
pub const WRITE: i32 = 0;
/// The operation removing a file.
pub const REMOVE: i32 = 1;
/// The operation renaming a file.
pub const RENAME: i32 = 2;

/// One operation on a directory's files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// The file `path` holds `bytes`, made when absent.
    Write {
        /// The file, under the directory.
        path: String,
        /// What it holds.
        bytes: Vec<u8>,
    },
    /// The file `path` is removed.
    Remove {
        /// The file, under the directory.
        path: String,
    },
    /// The file `from` is renamed `to`.
    Rename {
        /// The file, under the directory.
        from: String,
        /// Its new path, under the directory.
        to: String,
    },
}

impl Op {
    /// The path the operation acts on first.
    pub fn path(&self) -> &str {
        match self {
            Op::Write { path, .. } | Op::Remove { path } => path,
            Op::Rename { from, .. } => from,
        }
    }
}

/// Whether `path` names a file under a directory as the operations and
/// the files' listing spell it; why not.
pub fn checked_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("a path is empty".into());
    }
    if path.contains('\\') || path.contains('\0') {
        return Err(format!("`{path}` holds a backslash or a nul"));
    }
    for part in path.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.starts_with('.') {
            return Err(format!(
                "`{path}` is not a path under the directory: its parts are parted by `/`, none empty, `.`, `..` or hidden"
            ));
        }
    }
    Ok(())
}

/// The files, each its path and bytes, packed in the order given.
pub fn pack(files: &[(String, Vec<u8>)]) -> Result<FieldData, String> {
    let mut words: Vec<u32> = Vec::with_capacity(
        1 + files
            .iter()
            .map(|(path, bytes)| 2 + path.len() + bytes.len())
            .sum::<usize>(),
    );
    words.push(len(files.len())?);
    for (path, bytes) in files {
        words.push(len(path.len())?);
        words.push(len(bytes.len())?);
        words.extend(path.bytes().map(u32::from));
        words.extend(bytes.iter().map(|byte| u32::from(*byte)));
    }
    FieldData::from_storage(
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(len(words.len())?),
        words,
    )
    .map_err(|e| e.to_string())
}

/// The operations a Field holds, in order; refused whole when any is
/// malformed, so none of them is done.
pub fn ops(data: &FieldData) -> Result<Vec<Op>, String> {
    if data.schema().scalars() != [Scalar::Int] || data.domain().shape() != Shape::Rows {
        return Err(format!(
            "operations on files are a Rows<Int>, not {} {}",
            data.domain().shape(),
            data.schema()
        ));
    }
    let words = data.values();
    let mut at = 0usize;
    let mut word = |what: &str| -> Result<i32, String> {
        let found = words
            .get(at)
            .map(|w| *w as i32)
            .ok_or_else(|| format!("the operations end at row {at}, before their {what}"))?;
        at += 1;
        Ok(found)
    };
    let n = word("count")?;
    if n < 0 {
        return Err(format!("the operations' count is {n}"));
    }
    let mut found = Vec::new();
    for k in 0..n {
        let kind = word("kind")?;
        let path_len = word("path's length")?;
        let size = word("size")?;
        if path_len < 0 || size < 0 {
            return Err(format!("operation {k} has a length below zero"));
        }
        let mut text = |count: i32, what: &str| -> Result<Vec<u8>, String> {
            (0..count)
                .map(|_| {
                    let value = word(what)?;
                    u8::try_from(value)
                        .map_err(|_| format!("operation {k}'s {what} holds {value}, not a byte"))
                })
                .collect()
        };
        let path = utf8(text(path_len, "path")?, k)?;
        checked_path(&path).map_err(|why| format!("operation {k}: {why}"))?;
        let data = text(size, "data")?;
        found.push(match kind {
            WRITE => Op::Write { path, bytes: data },
            REMOVE => Op::Remove { path },
            RENAME => {
                let to = utf8(data, k)?;
                checked_path(&to).map_err(|why| format!("operation {k}: {why}"))?;
                Op::Rename { from: path, to }
            }
            _ => {
                return Err(format!(
                    "operation {k} is of kind {kind}: 0 writes, 1 removes, 2 renames"
                ));
            }
        });
    }
    Ok(found)
}

fn utf8(bytes: Vec<u8>, k: i32) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|_| format!("operation {k}'s path is not UTF-8"))
}

fn len(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} is past what a word counts"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(data: &FieldData) -> Vec<u32> {
        data.values().to_vec()
    }

    fn field(words: Vec<u32>) -> FieldData {
        FieldData::from_storage(
            Schema::new(vec![Scalar::Int]),
            Domain::Rows(words.len() as u32),
            words,
        )
        .unwrap()
    }

    #[test]
    fn files_pack_as_counts_paths_and_bytes() {
        let packed = pack(&[("a".into(), b"hi".to_vec()), ("d/b".into(), Vec::new())]).unwrap();
        assert_eq!(words(&packed), [2, 1, 2, 97, 104, 105, 3, 0, 100, 47, 98]);
        assert_eq!(words(&pack(&[]).unwrap()), [0]);
    }

    #[test]
    fn operations_read_in_order_and_ignore_what_follows() {
        let ops = ops(&field(vec![
            3, 0, 1, 2, 97, 104, 105, 1, 1, 0, 98, 2, 1, 1, 99, 100, 9, 9, 9,
        ]))
        .unwrap();
        assert_eq!(
            ops,
            [
                Op::Write {
                    path: "a".into(),
                    bytes: b"hi".to_vec()
                },
                Op::Remove { path: "b".into() },
                Op::Rename {
                    from: "c".into(),
                    to: "d".into()
                },
            ]
        );
    }

    #[test]
    fn malformed_operations_are_refused_whole() {
        // A path out of the directory, a kind unknown, a byte past 255,
        // and operations ending early.
        for (words, says) in [
            (vec![1, 1, 2, 0, 46, 46], "not a path under"),
            (vec![1, 7, 1, 0, 97], "kind 7"),
            (vec![1, 0, 1, 1, 97, 300], "not a byte"),
            (vec![2, 1, 1, 0, 97], "before their kind"),
            (vec![1, 2, 1, 2, 97, 46, 120], "not a path under"),
        ] {
            let why = ops(&field(words)).unwrap_err();
            assert!(why.contains(says), "{why}");
        }
    }
}
