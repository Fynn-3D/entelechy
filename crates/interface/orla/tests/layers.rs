//! The workspace's layers, held by its manifests: every crate sits in
//! `crates/<layer>/<crate>`, depends only in the directions AGENTS.md
//! allows, and takes a dependency from outside the workspace only where
//! AGENTS.md names one. A crate's tests depend in the same directions,
//! so a lower crate is tested without building a higher one; a test that
//! needs a higher crate lives where that crate is below it. A backend's
//! tests alone may also reach what a backend is held to: the compiler,
//! to write its Kernels as source, the interpreter it answers to, and
//! the conformance matrix.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The layers, in the order a crate may look down them.
const LAYERS: [&str; 4] = ["core", "frontend", "backends", "interface"];

/// The external dependencies a crate outside the interface layer may
/// take, by crate: a backend's machine and nothing else.
const MACHINES: [(&str, &[&str]); 2] = [
    ("orla-gpu", &["wgpu", "pollster"]),
    (
        "orla-jit",
        &[
            "cranelift-codegen",
            "cranelift-frontend",
            "cranelift-jit",
            "cranelift-module",
            "cranelift-native",
        ],
    ),
];

/// What a backend's tests may reach beyond its layers: the compiler, the
/// reference interpreter, and the conformance matrix.
const YARDSTICKS: [&str; 3] = ["orla-compiler", "orla-cpu", "orla-conformance"];

/// Interface crates that take nothing from outside all the same.
const PLAIN_INTERFACES: [&str; 1] = ["orla-formats"];

/// The crates allowed `unsafe`: the JIT calls the code it produces, the
/// FFI reads what a C caller hands it. Every other crate takes the
/// workspace's lints, which forbid it.
const UNSAFE: [&str; 2] = ["orla-jit", "orla-ffi"];

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
    /// Dependencies of the crate itself, of any table (`[dependencies]`,
    /// a target's, `[build-dependencies]`): workspace crates by package
    /// name, and the rest.
    inside: Vec<String>,
    outside: Vec<String>,
    /// Development dependencies (of any target) inside the workspace;
    /// those from outside count as the crate's own.
    tests: Vec<String>,
    workspace_lints: bool,
}

/// A manifest read as lines: a key is what comes before `=`, a workspace
/// crate is one given by `path`, and a dependency table is any table
/// whose name ends in `dependencies]`, a target's and a build's included,
/// or a dotted one that names a single dependency.
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
        // A backend is a peer of the others: it reaches none of them.
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

/// The session's scene, Kernel ledger, history, inputs and writes run on
/// the runtime the session is given; which backend that is, and what
/// only a backend has, belong to the session's opening, its trace and its
/// re-exports.
#[test]
fn the_session_runs_on_the_runtime_it_is_given() {
    let src = root().join("crates/interface/session/src");
    let parts = [
        "scene.rs",
        "kernels.rs",
        "sources.rs",
        "store.rs",
        "history.rs",
        "inputs.rs",
        "writes.rs",
        "check.rs",
        "session/time.rs",
    ];
    let mut wrong = Vec::new();
    for file in parts {
        let text = std::fs::read_to_string(src.join(file)).unwrap();
        for backend in ["orla_cpu", "orla_gpu", "orla_jit"] {
            if text.contains(backend) {
                wrong.push(format!("session/src/{file} names {backend}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `unsafe` is forbidden by the workspace's lints; a crate that does not
/// take them could use it unseen. Only the two named crates set their
/// own, and the workspace's still forbid it.
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

/// The core composes and computes; files are the interface's. Nothing
/// under `crates/core` names a path or the filesystem outside its tests:
/// what a scene reads and writes travels beside the Graph
/// (`orla_scene::Files`), carried out by the session.
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
