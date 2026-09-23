//! The workspace's layers, held by its manifests: every crate sits in
//! `crates/<layer>/<crate>`, depends only in the directions AGENTS.md
//! allows, and takes a dependency from outside the workspace only where
//! AGENTS.md names one. Development dependencies are free: a test may
//! hold one backend to another.

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

/// Interface crates that take nothing from outside all the same.
const PLAIN_INTERFACES: [&str; 1] = ["orla-formats"];

struct Crate {
    layer: &'static str,
    dir: PathBuf,
    /// Normal dependencies: workspace crates by package name, and the rest.
    inside: Vec<String>,
    outside: Vec<String>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A manifest's package name and its `[dependencies]` table, read as
/// lines: a key is what comes before `=`, and a workspace crate is one
/// given by `path`.
fn read(manifest: &Path) -> (String, Vec<String>, Vec<String>) {
    let text = std::fs::read_to_string(manifest).unwrap();
    let mut name = None;
    let mut table = "";
    let (mut inside, mut outside) = (Vec::new(), Vec::new());
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
    }
    (
        name.unwrap_or_else(|| panic!("{} names no package", manifest.display())),
        inside,
        outside,
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
            let (package, inside, outside) = read(&manifest);
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
                },
            );
        }
    }
    found
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
            let allowed = match krate.layer {
                "core" => target.layer == "core",
                "frontend" => matches!(target.layer, "core" | "frontend"),
                // A backend is a peer of the others: it reaches none of them.
                "backends" => target.layer == "core",
                _ => true,
            };
            if !allowed {
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
