//! The repository's packages held to their names and their layers: a
//! package is a domain named by one word in lower case (`random`, `image`,
//! `render`), its manifest declares the name its directory carries, it has
//! a README, its files are named in snake_case for their topic and never
//! for the package, its Kernels' ids begin with its name, and no function,
//! struct or type is declared by two of its documents or two packages. A
//! package sits in one layer of [`LAYERS`] and uses packages of its own
//! layer or those below, never one above, and no two use each other.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// The layers, lowest first, and the packages each holds.
pub(crate) const LAYERS: &[(&str, &[&str])] = &[
    (
        "foundation",
        &["random", "geometry", "parallel", "signal", "navigation"],
    ),
    (
        "graphics",
        &[
            "image",
            "raster",
            "voxel",
            "vector",
            "mesh",
            "terrain",
            "texture",
            "render",
            "particles",
            "scatter",
            "effects",
        ],
    ),
    ("media", &["audio", "music", "sound"]),
    ("simulation", &["physics", "animation", "genesis"]),
    ("interaction", &["text", "ui", "controls"]),
];

/// The layer `package` sits in, by its place in [`LAYERS`].
fn layer_of(package: &str) -> Option<usize> {
    LAYERS.iter().position(|(_, held)| held.contains(&package))
}

/// Each fault found among the packages in `dir`, said with the files it is in.
pub(crate) fn faults(dir: &Path) -> Vec<String> {
    let mut faults = Vec::new();
    let mut packages: Vec<(String, Vec<(String, String)>)> = Vec::new();
    let mut uses: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return vec![format!("{}: not read", dir.display())];
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        let Some(stem) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".entpkg"))
        else {
            continue;
        };
        let manifest = fs::read_to_string(path.join("package.esn")).unwrap_or_default();
        faults.extend(named(stem, &manifest));
        let mut files = Vec::new();
        kernel_files(&path, &path, &mut files);
        faults.extend(owned_ids(stem, &files));
        faults.extend(laid_out(stem, &path));
        faults.extend(tested(stem, &path, &files));
        uses.insert(stem.to_owned(), used_packages(&path));
        packages.push((stem.to_owned(), files));
    }
    faults.extend(twice(&packages));
    faults.extend(layered(&uses));
    faults
}

/// What is wrong with how a package is laid out: no README, or a file not
/// named in snake_case for its topic, or named for the package.
fn laid_out(package: &str, path: &Path) -> Vec<String> {
    let mut faults = Vec::new();
    if !path.join("README.md").is_file() {
        faults.push(format!("{package}.entpkg: no README.md"));
    }
    for dir in ["kernels", "graphs"] {
        let Ok(entries) = fs::read_dir(path.join(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(stem) = name
                .strip_suffix(".ent")
                .or_else(|| name.strip_suffix(".esn"))
            else {
                continue;
            };
            let snake = !stem.is_empty()
                && stem
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                && !stem.starts_with('_')
                && !stem.ends_with('_');
            if !snake {
                faults.push(format!(
                    "{package}.entpkg/{dir}/{name}: a file is named in snake_case for its topic"
                ));
            }
            if stem == package {
                faults.push(format!(
                    "{package}.entpkg/{dir}/{name}: a file is named for its topic, not for its package"
                ));
            }
        }
    }
    faults
}

/// Each Kernel of a package no test of its own reaches: a package holds
/// `tests/`, scenes and their replays, and every Kernel it declares is
/// bound by a test scene or by a graph preset of the package a test scene
/// places, however deep.
fn tested(package: &str, path: &Path, files: &[(String, String)]) -> Vec<String> {
    let tests = path.join("tests");
    let scenes: Vec<String> = fs::read_dir(&tests)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "esn"))
                .map(|p| fs::read_to_string(p).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    let replays = fs::read_dir(&tests).map_or(0, |entries| {
        entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "replay"))
            .count()
    });
    if scenes.is_empty() || replays == 0 {
        return vec![format!(
            "{package}.entpkg: no tests/ of scenes and their replays"
        )];
    }
    // What the test scenes reach: their own text and every preset of the
    // package they place, and the presets those place.
    let mut reached = scenes.join("\n");
    let mut placed = BTreeSet::new();
    loop {
        let mut more = Vec::new();
        for (at, _) in reached.match_indices("graphs/") {
            let name: String = reached[at + 7..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && placed.insert(name.clone()) {
                more.push(name);
            }
        }
        for (at, _) in reached.match_indices("graph \"") {
            let name: String = reached[at + 7..]
                .chars()
                .take_while(|c| *c != '"')
                .collect();
            if let Some(stem) = name.rsplit('/').next().and_then(|n| n.strip_suffix(".esn"))
                && !name.contains(".entpkg/")
                && placed.insert(stem.to_owned())
            {
                more.push(stem.to_owned());
            }
        }
        if more.is_empty() {
            break;
        }
        for name in more {
            if let Ok(text) = fs::read_to_string(path.join("graphs").join(format!("{name}.esn"))) {
                reached.push('\n');
                reached.push_str(&text);
            }
        }
    }
    let mut faults = Vec::new();
    for (file, text) in files {
        let Ok(document) = entelechy_language::parse_document(text) else {
            continue;
        };
        for kernel in &document.kernels {
            if let Some(id) = &kernel.id
                && !reached.contains(&id.value)
            {
                faults.push(format!(
                    "{package}.entpkg/{file}: `{}` is reached by no test of the package",
                    id.value
                ));
            }
        }
    }
    faults
}

/// The packages the documents under `path` use by name.
fn used_packages(path: &Path) -> BTreeSet<String> {
    let mut texts = Vec::new();
    kernel_files(path, path, &mut texts);
    let mut graphs = Vec::new();
    if let Ok(entries) = fs::read_dir(path.join("graphs")) {
        for entry in entries.flatten() {
            graphs.push(fs::read_to_string(entry.path()).unwrap_or_default());
        }
    }
    let mut used = BTreeSet::new();
    for text in texts.iter().map(|(_, t)| t).chain(&graphs) {
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("use \"")
                && let Some(name) = rest
                    .split('"')
                    .next()
                    .and_then(|p| p.strip_suffix(".entpkg"))
            {
                used.insert(name.rsplit('/').next().unwrap_or(name).to_owned());
            }
        }
    }
    used
}

/// Each package outside the layers, using one above its own, or using
/// one that uses it in turn.
fn layered(uses: &BTreeMap<String, BTreeSet<String>>) -> Vec<String> {
    let mut faults = Vec::new();
    for (package, used) in uses {
        let Some(own) = layer_of(package) else {
            faults.push(format!("{package}.entpkg: in no layer; name it in `LAYERS` in crates/interface/xtask/src/packages.rs"));
            continue;
        };
        for other in used {
            match layer_of(other) {
                Some(theirs) if theirs > own => faults.push(format!(
                    "{package}.entpkg ({}) uses {other}.entpkg ({}), a layer above it",
                    LAYERS[own].0, LAYERS[theirs].0
                )),
                _ => {}
            }
            if uses.get(other).is_some_and(|back| back.contains(package)) && package < other {
                faults.push(format!(
                    "{package}.entpkg and {other}.entpkg use each other"
                ));
            }
        }
    }
    faults
}

/// The `.ent` files under `dir`, each by its path inside `package`, with its text.
fn kernel_files(package: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            // A package's tests hold their own scenes and helpers, not its interface.
            if path.parent() == Some(package) && path.file_name().is_some_and(|n| n == "tests") {
                continue;
            }
            kernel_files(package, &path, out);
        } else if path.extension().is_some_and(|e| e == "ent") {
            let inside = path.strip_prefix(package).unwrap_or(&path);
            out.push((
                inside.display().to_string(),
                fs::read_to_string(&path).unwrap_or_default(),
            ));
        }
    }
}

/// What is wrong with a package's name: not one word in lower case, or not
/// the name its manifest declares.
fn named(stem: &str, manifest: &str) -> Vec<String> {
    let mut faults = Vec::new();
    if stem.is_empty() || !stem.chars().all(|c| c.is_ascii_lowercase()) {
        faults.push(format!(
            "{stem}.entpkg: a package is a domain named by one word in lower case, as `random`"
        ));
    }
    let declared = manifest
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("package "))
        .and_then(|rest| rest.split('@').next())
        .map(str::trim);
    if declared != Some(stem) {
        faults.push(format!(
            "{stem}.entpkg/package.esn: declares {}, not `{stem}`",
            declared.map_or("no package".to_owned(), |d| format!("`{d}`"))
        ));
    }
    faults
}

/// Each Kernel whose id does not begin with its package's name.
fn owned_ids(package: &str, files: &[(String, String)]) -> Vec<String> {
    let mut faults = Vec::new();
    for (file, text) in files {
        let Ok(document) = entelechy_language::parse_document(text) else {
            continue;
        };
        for kernel in &document.kernels {
            if let Some(id) = &kernel.id
                && !id.value.starts_with(&format!("{package}."))
            {
                faults.push(format!(
                    "{package}.entpkg/{file}: `{}` is not named by its package, `{package}.`",
                    id.value
                ));
            }
        }
    }
    faults
}

/// Each function, struct or type two documents of the packages declare,
/// naming where.
fn twice(packages: &[(String, Vec<(String, String)>)]) -> Vec<String> {
    let mut declared: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut faults = Vec::new();
    for (package, files) in packages {
        for (file, text) in files {
            let document = match entelechy_language::parse_document(text) {
                Ok(document) => document,
                Err(error) => {
                    faults.push(format!("{package}.entpkg/{file}: {error}"));
                    continue;
                }
            };
            let names = document
                .functions
                .iter()
                .map(|d| &d.name.value)
                .chain(document.structs.iter().map(|d| &d.name.value))
                .chain(document.types.iter().map(|d| &d.name.value));
            for name in names {
                declared
                    .entry(name.clone())
                    .or_default()
                    .push(format!("{package}.entpkg/{file}"));
            }
        }
    }
    for (name, places) in declared {
        if places.len() > 1 {
            faults.push(format!(
                "`{name}` is declared by {}: a name is declared once among the packages",
                places.join(" and ")
            ));
        }
    }
    faults
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_package_is_named_by_one_word_its_manifest_declares() {
        assert!(named("math", "package math@1 { language = 1; ir = 1; }").is_empty());
        assert_eq!(
            named(
                "hash_noise",
                "package hash_noise@1 { language = 1; ir = 1; }"
            )
            .len(),
            1
        );
        assert_eq!(
            named("math", "package noise@1 { language = 1; ir = 1; }").len(),
            1
        );
    }

    #[test]
    fn a_package_uses_its_own_layer_or_those_below() {
        let uses: BTreeMap<String, BTreeSet<String>> = [
            ("random", vec![]),
            ("image", vec!["random"]),
            ("raster", vec!["text"]),
            ("text", vec!["ui"]),
            ("ui", vec!["text"]),
        ]
        .into_iter()
        .map(|(p, u)| (p.to_owned(), u.into_iter().map(str::to_owned).collect()))
        .collect();
        let faults = layered(&uses);
        assert_eq!(faults.len(), 2, "{faults:?}");
        assert!(
            faults[0].contains("raster.entpkg (graphics) uses text.entpkg (interaction)"),
            "{faults:?}"
        );
        assert!(
            faults[1].contains("text.entpkg and ui.entpkg use each other"),
            "{faults:?}"
        );
    }

    #[test]
    fn a_kernel_is_named_by_its_package() {
        let files = vec![(
            "kernels/a.ent".to_owned(),
            "kernel math.one@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }\nkernel noise.two@1 { requires { field.write; } output result: Rows<Int>; compute { return 2; } }".to_owned(),
        )];
        let faults = owned_ids("math", &files);
        assert_eq!(faults.len(), 1, "{faults:?}");
        assert!(faults[0].contains("`noise.two@1`"), "{faults:?}");
    }

    #[test]
    fn a_name_two_documents_declare_is_said_with_both() {
        let file = |name: &str, text: &str| (name.to_owned(), text.to_owned());
        let packages = vec![
            (
                "one".to_owned(),
                vec![
                    file(
                        "kernels/a.ent",
                        "fn clear() -> Int { return 1; }\nstruct Mark { a: Int; }",
                    ),
                    file("kernels/b.ent", "struct Mark { a: Int; }"),
                ],
            ),
            (
                "two".to_owned(),
                vec![file(
                    "kernels/a.ent",
                    "fn clear() -> Int { return 2; }\nfn own() -> Int { return 3; }",
                )],
            ),
        ];
        let faults = twice(&packages);
        assert_eq!(faults.len(), 2, "{faults:?}");
        assert!(faults.iter().any(|f| f.contains("`clear`")
            && f.contains("one.entpkg")
            && f.contains("two.entpkg")));
        assert!(
            faults
                .iter()
                .any(|f| f.contains("`Mark`") && f.contains("one.entpkg/kernels/b.ent"))
        );
    }
}
