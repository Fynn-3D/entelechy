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

struct Crate {
    layer: &'static str,
    dir: PathBuf,
    /// Normal dependencies: workspace crates by package name, and the rest.
    inside: Vec<String>,
    outside: Vec<String>,
    /// Development dependencies inside the workspace, by package name.
    tests: Vec<String>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A manifest's package name, its `[dependencies]` inside and outside the
/// workspace, and its `[dev-dependencies]` inside, read as lines: a key
/// is what comes before `=`, and a workspace crate is one given by `path`.
fn read(manifest: &Path) -> (String, Vec<String>, Vec<String>, Vec<String>) {
    let text = std::fs::read_to_string(manifest).unwrap();
    let mut name = None;
    let mut table = "";
    let (mut inside, mut outside, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            table = line;
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if table == "[package]" && key == "name" {
            name = Some(value.trim().trim_matches('"').to_owned());
        }
        if table == "[dependencies]" && !key.starts_with('#') {
            if value.contains("path") {
                inside.push(key.to_owned());
            } else {
                outside.push(key.to_owned());
            }
        }
        if table == "[dev-dependencies]" && !key.starts_with('#') && value.contains("path") {
            tests.push(key.to_owned());
        }
    }
    (
        name.unwrap_or_else(|| panic!("{} names no package", manifest.display())),
        inside,
        outside,
        tests,
    )
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
            let (package, inside, outside, tests) = read(&manifest);
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
