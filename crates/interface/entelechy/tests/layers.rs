//! The workspace's layers, held by its manifests: every crate sits in
//! `crates/<layer>/<crate>`, depends only in the directions AGENTS.md
//! allows, and takes an external dependency only where AGENTS.md names
//! one. Tests depend in the same directions; a backend's tests may also
//! reach the compiler, the interpreter and the conformance matrix. No
//! crate reaches itself, its tests counted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The layers, in the order a crate may look down them.
const LAYERS: [&str; 4] = ["core", "frontend", "backends", "interface"];

/// The external dependencies a non-interface crate may take: a backend's
/// machine, and for the device in a page the crates wgpu reaches the
/// browser with.
const MACHINES: [(&str, &[&str]); 2] = [
    (
        "entelechy-gpu",
        &[
            "wgpu",
            "pollster",
            "js-sys",
            "wasm-bindgen",
            "wasm-bindgen-futures",
        ],
    ),
    (
        "entelechy-jit",
        &[
            "cranelift-codegen",
            "cranelift-frontend",
            "cranelift-jit",
            "cranelift-module",
            "cranelift-native",
        ],
    ),
];

/// What a backend's tests may reach beyond its layers.
const YARDSTICKS: [&str; 3] = [
    "entelechy-compiler",
    "entelechy-cpu",
    "entelechy-conformance",
];

/// Interface crates that take no external dependency.
const PLAIN_INTERFACES: [&str; 2] = ["entelechy-formats", "entelechy-documents"];

/// The crates allowed `unsafe`: the JIT runs its code, the FFI reads a C
/// caller's pointers, and the GPU opts into native `wgpu::ExperimentalFeatures`
/// for hardware ray queries (`#![deny(unsafe_code)]` stays active crate-wide).
/// The workspace's lints forbid it elsewhere.
const UNSAFE: [&str; 3] = ["entelechy-jit", "entelechy-ffi", "entelechy-gpu"];

struct Crate {
    layer: &'static str,
    dir: PathBuf,
    /// Normal dependencies: workspace crates by package name, and the rest.
    inside: Vec<String>,
    outside: Vec<String>,
    /// Development dependencies inside the workspace, by package name.
    tests: Vec<String>,
    /// Whether it takes the workspace's lints, which forbid `unsafe`.
    workspace_lints: bool,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// What a manifest says of a crate.
struct Manifest {
    name: String,
    /// Dependencies of the crate itself, from any table: workspace crates
    /// by package name, and the rest.
    inside: Vec<String>,
    outside: Vec<String>,
    /// Workspace development dependencies of any target; external ones
    /// count as the crate's own.
    tests: Vec<String>,
    workspace_lints: bool,
}

/// A manifest read as lines: a key precedes `=`, a workspace crate has a
/// `path`, and a dependency table's name ends in `dependencies]` or is a
/// dotted one naming a single dependency.
fn read(manifest: &Path) -> Manifest {
    let text = std::fs::read_to_string(manifest).unwrap();
    let mut name = None;
    let mut table = String::new();
    // A `[dependencies.x]` table is one dependency, `x`.
    let mut dotted: Option<(String, bool)> = None;
    let (mut inside, mut outside, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    let mut workspace_lints = false;
    let flush = |dotted: &mut Option<(String, bool)>,
                 inside: &mut Vec<String>,
                 outside: &mut Vec<String>| {
        if let Some((key, path)) = dotted.take() {
            if path {
                inside.push(key)
            } else {
                outside.push(key)
            }
        }
    };
    for line in text.lines().map(str::trim) {
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            flush(&mut dotted, &mut inside, &mut outside);
            table = line.to_owned();
            if let Some(key) = table
                .strip_prefix("[dependencies.")
                .or_else(|| table.strip_prefix("[build-dependencies."))
                .and_then(|rest| rest.strip_suffix(']'))
            {
                dotted = Some((key.to_owned(), false));
            }
            assert!(
                !table.contains("dev-dependencies."),
                "{}: write a development dependency inline",
                manifest.display()
            );
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if table == "[package]" && key == "name" {
            name = Some(value.trim().trim_matches('"').to_owned());
        }
        if table == "[lints]" && key == "workspace" && value.trim() == "true" {
            workspace_lints = true;
        }
        if let Some((_, path)) = &mut dotted {
            *path |= key == "path";
            continue;
        }
        if table.ends_with("dev-dependencies]") {
            if value.contains("path") {
                tests.push(key.to_owned());
            } else {
                outside.push(key.to_owned());
            }
        } else if table.ends_with("dependencies]") {
            if value.contains("path") {
                inside.push(key.to_owned());
            } else {
                outside.push(key.to_owned());
            }
        }
    }
    flush(&mut dotted, &mut inside, &mut outside);
    Manifest {
        name: name.unwrap_or_else(|| panic!("{} names no package", manifest.display())),
        inside,
        outside,
        tests,
        workspace_lints,
    }
}

fn crates() -> BTreeMap<String, Crate> {
    let crates_dir = root().join("crates");
    let mut found = BTreeMap::new();
    for entry in std::fs::read_dir(&crates_dir).unwrap() {
        let dir = entry.unwrap().path();
        if !dir.is_dir() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let layer = LAYERS
            .iter()
            .find(|layer| **layer == name)
            .unwrap_or_else(|| panic!("`crates/{name}` is not a layer; use one of {LAYERS:?}"));
        for entry in std::fs::read_dir(&dir).unwrap() {
            let dir = entry.unwrap().path();
            if !dir.is_dir() {
                continue;
            }
            let manifest = dir.join("Cargo.toml");
            assert!(
                manifest.is_file(),
                "{} is not a crate; a layer holds crates only",
                dir.display()
            );
            let Manifest {
                name: package,
                inside,
                outside,
                tests,
                workspace_lints,
            } = read(&manifest);
            let dir = Path::new("crates")
                .join(layer)
                .join(dir.file_name().unwrap());
            found.insert(
                package,
                Crate {
                    layer,
                    dir,
                    inside,
                    outside,
                    tests,
                    workspace_lints,
                },
            );
        }
    }
    found
}

/// Whether a crate of `from` may depend on one of `to`.
fn down(from: &str, to: &str) -> bool {
    match from {
        "core" => to == "core",
        "frontend" => matches!(to, "core" | "frontend"),
        // A backend reaches no other backend.
        "backends" => to == "core",
        _ => true,
    }
}

#[test]
fn every_crate_depends_only_down_its_layers() {
    let crates = crates();
    assert!(crates.len() > 10, "found only {} crates", crates.len());
    let mut wrong = Vec::new();
    for (name, krate) in &crates {
        for dependency in &krate.inside {
            let Some(target) = crates.get(dependency) else {
                wrong.push(format!(
                    "{name} depends on `{dependency}`, which is no crate here"
                ));
                continue;
            };
            if !down(krate.layer, target.layer) {
                wrong.push(format!(
                    "{name} ({}) depends on {dependency} ({})",
                    krate.layer, target.layer
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "layers crossed:\n{}", wrong.join("\n"));
}

#[test]
fn every_crate_is_tested_down_its_layers() {
    let crates = crates();
    let mut wrong = Vec::new();
    for (name, krate) in &crates {
        for dependency in &krate.tests {
            let Some(target) = crates.get(dependency) else {
                wrong.push(format!(
                    "{name}'s tests depend on `{dependency}`, which is no crate here"
                ));
                continue;
            };
            let yardstick = krate.layer == "backends"
                && YARDSTICKS.contains(&dependency.as_str())
                && dependency != name;
            if !down(krate.layer, target.layer) && !yardstick {
                wrong.push(format!(
                    "{name}'s tests ({}) depend on {dependency} ({}); a test that needs it lives where it is below",
                    krate.layer, target.layer
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "layers crossed by tests:\n{}",
        wrong.join("\n")
    );
}

/// No crate reaches itself through what it and its tests depend on.
/// Cargo refuses a cycle of dependencies, not one a test closes, and the
/// layers let crates of one layer depend on each other.
#[test]
fn no_crate_reaches_itself_through_its_dependencies_and_tests() {
    let crates = crates();
    let next = |name: &str| -> Vec<&str> {
        let krate = &crates[name];
        let mut next: Vec<&str> = krate
            .inside
            .iter()
            .chain(&krate.tests)
            .map(String::as_str)
            .filter(|dependency| crates.contains_key(*dependency))
            .collect();
        next.sort_unstable();
        next.dedup();
        next
    };
    // Depth first; a crate met again on the path it is on closes a cycle.
    fn walk<'a>(
        name: &'a str,
        next: &dyn Fn(&str) -> Vec<&'a str>,
        path: &mut Vec<&'a str>,
        done: &mut std::collections::BTreeSet<&'a str>,
    ) -> Option<String> {
        if let Some(at) = path.iter().position(|on| *on == name) {
            let mut cycle = path[at..].to_vec();
            cycle.push(name);
            return Some(cycle.join(" -> "));
        }
        if !done.insert(name) {
            return None;
        }
        path.push(name);
        for dependency in next(name) {
            if let Some(cycle) = walk(dependency, next, path, done) {
                return Some(cycle);
            }
        }
        path.pop();
        None
    }
    let mut done = std::collections::BTreeSet::new();
    for name in crates.keys() {
        let mut path = Vec::new();
        if let Some(cycle) = walk(name, &next, &mut path, &mut done) {
            panic!("a cycle of dependencies: {cycle}");
        }
    }
}

#[test]
fn only_the_interface_and_two_machines_take_outside_dependencies() {
    let mut wrong = Vec::new();
    for (name, krate) in &crates() {
        let allowed: &[&str] = match MACHINES.iter().find(|(machine, _)| machine == name) {
            Some((_, allowed)) => allowed,
            None if krate.layer == "interface" && !PLAIN_INTERFACES.contains(&name.as_str()) => {
                continue;
            }
            None => &[],
        };
        for dependency in &krate.outside {
            if !allowed.contains(&dependency.as_str()) {
                wrong.push(format!(
                    "{name} ({}) takes `{dependency}` from outside the workspace",
                    krate.dir.display()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The documents a session runs are read, published and lowered on the
/// runtime the session gives them: `entelechy-documents` reaches no
/// backend, through any crate it takes.
#[test]
fn the_documents_run_on_the_runtime_they_are_given() {
    let crates = crates();
    let mut reached = vec!["entelechy-documents".to_string()];
    let mut next = 0;
    while next < reached.len() {
        for dependency in &crates[&reached[next]].inside {
            if !reached.contains(dependency) {
                reached.push(dependency.clone());
            }
        }
        next += 1;
    }
    let backends: Vec<&String> = reached
        .iter()
        .filter(|name| crates[*name].layer == "backends")
        .collect();
    assert!(
        backends.is_empty(),
        "entelechy-documents reaches {backends:?}"
    );
}

/// Only the two named crates set their own lints, and the workspace's
/// still forbid `unsafe`.
#[test]
fn only_the_jit_and_the_ffi_may_be_unsafe() {
    let workspace = std::fs::read_to_string(root().join("Cargo.toml")).unwrap();
    assert!(
        workspace.contains("[workspace.lints.rust]\nunsafe_code = \"forbid\""),
        "the workspace's lints forbid unsafe code"
    );
    let mut wrong = Vec::new();
    for (name, krate) in &crates() {
        let allowed = UNSAFE.contains(&name.as_str());
        if krate.workspace_lints == allowed {
            wrong.push(if allowed {
                format!("{name} takes the workspace's lints, yet it is one allowed `unsafe`")
            } else {
                format!(
                    "{name} ({}) does not take the workspace's lints (`[lints] workspace = true`), which forbid `unsafe`",
                    krate.dir.display()
                )
            });
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Nothing under `crates/core` names a path or the filesystem outside its
/// tests: scene files travel beside the Graph (`entelechy_scene::Files`) and
/// the session carries them out.
#[test]
fn the_core_knows_no_file() {
    fn walk(dir: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                walk(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                // A module's own tests may make files to read back.
                let text = text.split("#[cfg(test)]").next().unwrap_or_default();
                for word in ["std::fs", "std::path", "PathBuf"] {
                    if text.contains(word) {
                        found.push(format!("{} names `{word}`", path.display()));
                    }
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(&root().join("crates/core"), &mut found);
    assert!(found.is_empty(), "{}", found.join("\n"));
}
