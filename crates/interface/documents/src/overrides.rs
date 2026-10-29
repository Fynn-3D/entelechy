//! Where a scene's file-backed inputs read and its outputs write, as the
//! person running it says (`--input name=path`, `--output name=path`, a
//! folder chosen in a player): the path a document writes for an input or
//! an output replaced by the caller's, the source's kind kept. The scene's
//! own text is unchanged; the caller resolves the source.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Paths given for a scene's inputs and outputs, by name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Overrides {
    /// Inputs: a file-backed input reads the path given; an external the
    /// scene leaves to its caller is bound to the file
    /// (`Session::bind_external`).
    pub inputs: BTreeMap<String, PathBuf>,
    /// Outputs written to files write to the path given.
    pub outputs: BTreeMap<String, PathBuf>,
}

/// What a source reads or writes: one file, or a directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A file (`= "a.field"`, `bytes "a.txt"`, `to "a.png"`).
    File,
    /// A directory (`files`, `checks` and `outline` on a directory,
    /// frames, `to files`).
    Directory,
}

impl Overrides {
    /// Whether nothing is given.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.outputs.is_empty()
    }

    /// `files` with each input and output given here reading and writing
    /// the path given; a `files` node is given its directory by its id.
    /// An input given that no file-backed input is named
    /// must be one of `externals` (the caller binds it); returned so.
    /// Any other name given is refused, naming the scene's file-backed
    /// inputs and outputs.
    pub fn apply(
        &self,
        files: &mut entelechy_scene::Files,
        externals: &[String],
    ) -> Result<Vec<(String, PathBuf)>, String> {
        let mut bound = Vec::new();
        for (name, path) in &self.inputs {
            let path = path.clone();
            if let Some(asset) = files.assets.iter_mut().find(|a| &a.name == name) {
                asset.path = path;
            } else if let Some(sequence) = files.sequences.iter_mut().find(|s| &s.0 == name) {
                sequence.1 = path;
            } else if let Some(entry) = files
                .checks
                .iter_mut()
                .chain(files.files.iter_mut())
                .chain(files.outlines.iter_mut())
                .find(|(named, _)| named == name)
            {
                entry.1 = path;
            } else if let Some(node) = files.file_nodes.iter_mut().find(|n| &n.node == name) {
                node.dir = path;
            } else if let Some(node) = files.field_nodes.iter_mut().find(|n| &n.node == name) {
                node.dir = path;
            } else if externals.iter().any(|external| external == name) {
                bound.push((name.clone(), path));
            } else {
                return Err(format!(
                    "--input {name}: the scene has no input of that name; {}",
                    inputs_said(files)
                ));
            }
        }
        for (name, path) in &self.outputs {
            let mut found = false;
            for write in files.writes.iter_mut().filter(|w| &w.output == name) {
                write.path = path.clone();
                found = true;
            }
            if !found {
                return Err(format!(
                    "--output {name}: the scene writes no output of that name to a file; {}",
                    outputs_said(files)
                ));
            }
        }
        Ok(bound)
    }
}

/// What the input or output `name` of `files` reads or writes: a file or
/// a directory; `None` where none is named so.
pub fn kind_of(files: &entelechy_scene::Files, name: &str) -> Option<Kind> {
    if files.assets.iter().any(|a| a.name == name) {
        return Some(Kind::File);
    }
    if files.sequences.iter().any(|s| s.0 == name)
        || files.files.iter().any(|(n, _)| n == name)
        || files.file_nodes.iter().any(|n| n.node == name)
        || files.field_nodes.iter().any(|n| n.node == name)
    {
        return Some(Kind::Directory);
    }
    if let Some((_, path)) = files
        .checks
        .iter()
        .chain(&files.outlines)
        .find(|(n, _)| n == name)
    {
        return Some(if is_dir_path(path) {
            Kind::Directory
        } else {
            Kind::File
        });
    }
    let write = files
        .writes
        .iter()
        .find(|w| w.output == name && !w.clipboard)?;
    Some(if write.files {
        Kind::Directory
    } else {
        Kind::File
    })
}

/// Whether a source's path names a directory as written (its trailing
/// slash), or is one on disk.
fn is_dir_path(path: &Path) -> bool {
    path.as_os_str().to_string_lossy().ends_with('/') || path.is_dir()
}

fn inputs_said(files: &entelechy_scene::Files) -> String {
    let names: Vec<&str> = files
        .assets
        .iter()
        .map(|a| a.name.as_str())
        .chain(files.sequences.iter().map(|s| s.0.as_str()))
        .chain(
            files
                .checks
                .iter()
                .chain(&files.files)
                .chain(&files.outlines)
                .map(|(n, _)| n.as_str()),
        )
        .chain(files.file_nodes.iter().map(|n| n.node.as_str()))
        .chain(files.field_nodes.iter().map(|n| n.node.as_str()))
        .collect();
    if names.is_empty() {
        "it reads no input from a file".into()
    } else {
        format!("its inputs read from files: {}", names.join(", "))
    }
}

fn outputs_said(files: &entelechy_scene::Files) -> String {
    let names: Vec<&str> = files.writes.iter().map(|w| w.output.as_str()).collect();
    if names.is_empty() {
        "it writes no output to a file".into()
    } else {
        format!("its outputs written to files: {}", names.join(", "))
    }
}

/// `name=path`, as `--input` and `--output` take it.
pub fn parse_binding(flag: &str, binding: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = binding
        .split_once('=')
        .filter(|(name, path)| !name.is_empty() && !path.is_empty())
        .ok_or_else(|| format!("{flag} needs <name>=<path>"))?;
    Ok((name.to_owned(), PathBuf::from(path)))
}
