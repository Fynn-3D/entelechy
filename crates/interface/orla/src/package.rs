//! Packages: a directory with a `package.osn` manifest, its Kernels,
//! graph presets and assets, and its `.orlapkg` archive: a magic line,
//! the format version, then every file as its path and bytes,
//! uncompressed.

use std::path::{Path, PathBuf};

use orla_session::{PACKAGE_VERSION, package_manifest};

const MAGIC: &[u8] = b"ORLAPKG\n";

/// Every file under `dir`, as paths relative to it, in path order;
/// nothing hidden.
fn files_of(dir: &Path) -> Result<Vec<PathBuf>, String> {
    fn walk(dir: &Path, prefix: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                !path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            })
            .collect();
        entries.sort();
        for path in entries {
            let relative = prefix.join(path.file_name().unwrap_or_default());
            if path.is_dir() {
                walk(&path, &relative, out)?;
            } else {
                out.push(relative);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, Path::new(""), &mut out)?;
    Ok(out)
}

/// Writes a package directory as one `.orlapkg` file after checking its
/// manifest.
pub fn pack(dir: &Path, out: &Path) -> Result<usize, String> {
    let declared = package_manifest(&orla_session::Disk::default(), &dir.join("package.osn"))?;
    let files = files_of(dir)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&PACKAGE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(files.len() as u32).to_le_bytes());
    for relative in &files {
        let path = relative.to_string_lossy().replace('\\', "/");
        let data = std::fs::read(dir.join(relative))
            .map_err(|e| format!("{}: {e}", dir.join(relative).display()))?;
        bytes.extend_from_slice(&(path.len() as u32).to_le_bytes());
        bytes.extend_from_slice(path.as_bytes());
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&data);
    }
    std::fs::write(out, bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    let _ = declared;
    Ok(files.len())
}

/// Unpacks an `.orlapkg` file into a new directory and checks its
/// manifest.
pub fn unpack(file: &Path, dir: &Path) -> Result<usize, String> {
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut at = 0;
    let take = |at: &mut usize, n: usize| -> Result<&[u8], String> {
        let slice = bytes
            .get(*at..*at + n)
            .ok_or_else(|| format!("{}: the archive ends early", file.display()))?;
        *at += n;
        Ok(slice)
    };
    if take(&mut at, MAGIC.len())? != MAGIC {
        return Err(format!("{}: not an .orlapkg archive", file.display()));
    }
    let version = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap_or([0; 4]));
    if version > PACKAGE_VERSION {
        return Err(format!(
            "{}: archive format {version}, and this Orla reads format {PACKAGE_VERSION}",
            file.display()
        ));
    }
    let count = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap_or([0; 4])) as usize;
    if dir.exists() {
        return Err(format!(
            "{} exists; unpack into a new directory",
            dir.display()
        ));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // A refused archive leaves nothing: the directory is this call's own.
    let unpacked = (|| {
        unpack_into(file, dir, count, &bytes[at..])?;
        package_manifest(&orla_session::Disk::default(), &dir.join("package.osn"))
    })();
    if let Err(error) = unpacked {
        let _ = std::fs::remove_dir_all(dir);
        return Err(error);
    }
    Ok(count)
}

/// Writes the `count` files `bytes` holds into `dir`.
fn unpack_into(file: &Path, dir: &Path, count: usize, bytes: &[u8]) -> Result<(), String> {
    let mut at = 0;
    let take = |at: &mut usize, n: usize| -> Result<&[u8], String> {
        let slice = bytes
            .get(*at..*at + n)
            .ok_or_else(|| format!("{}: the archive ends early", file.display()))?;
        *at += n;
        Ok(slice)
    };
    for _ in 0..count {
        let len = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap_or([0; 4])) as usize;
        let path = std::str::from_utf8(take(&mut at, len)?)
            .map_err(|_| format!("{}: a path is not text", file.display()))?
            .to_owned();
        let size = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap_or([0; 8])) as usize;
        let data = take(&mut at, size)?;
        if path.starts_with('/') || path.split('/').any(|part| part == "..") {
            return Err(format!(
                "{}: `{path}` would leave the directory",
                file.display()
            ));
        }
        let target = dir.join(&path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&target, data).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    Ok(())
}

/// A package's id, needs, and counts of Kernel files, graph presets and
/// other files, once its manifest's Kernels compile and its presets read,
/// each checked as `orla check` checks it.
pub fn describe(dir: &Path) -> Result<String, String> {
    let declared = package_manifest(&orla_session::Disk::default(), &dir.join("package.osn"))?;
    let files = files_of(dir)?;
    let kernels = files
        .iter()
        .filter(|f| f.extension().is_some_and(|e| e == "orla"))
        .count();
    let graphs = files
        .iter()
        .filter(|f| {
            f.extension().is_some_and(|e| e == "osn")
                && f.file_name().is_some_and(|n| n != "package.osn")
        })
        .count();
    let others = files.len() - kernels - graphs - 1;
    let store = orla_session::Disk::with_packages(orla_session::environment_packages());
    let checked = std::iter::once(dir.join("package.osn")).chain(
        files
            .iter()
            .filter(|f| {
                f.extension().is_some_and(|e| e == "osn")
                    && f.file_name().is_some_and(|n| n != "package.osn")
            })
            .map(|f| dir.join(f)),
    );
    let problems: Vec<String> = checked
        .flat_map(|path| orla_session::check_document(&store, &path, false).lines)
        .filter(|(problem, _)| *problem)
        .map(|(_, line)| line)
        .collect();
    if !problems.is_empty() {
        return Err(format!(
            "{}\n{} problem(s) found",
            problems.join("\n"),
            problems.len()
        ));
    }
    Ok(format!(
        "package {} (needs language {}, ir {}): {kernels} kernel file(s), {graphs} graph preset(s), {others} other file(s)",
        declared.id.value, declared.language, declared.ir
    ))
}
