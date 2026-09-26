//! The capabilities a session reaches, and a search over them: the
//! Kernels it has published, and every Kernel and preset graph of each
//! package in [`crate::sources::package_dirs`] or beside a package the
//! scene uses. Each says what it is for (its `///` and `//!` docs), what
//! it takes and gives, what it requires and whether this session grants
//! that, and how a scene reaches it. Matching is by text, by type and by
//! capability, in a fixed order; nothing is guessed.

use std::path::{Path, PathBuf};

use orla_language::doc;

use crate::Session;
use crate::sources::{in_a_package, package_dirs, package_files, package_manifest};
use crate::store::Store;

/// What a capability is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A Kernel a node runs.
    Kernel,
    /// A graph a node places: `node n: graph "path" { ... }`.
    Preset,
}

impl Kind {
    /// Its name on the wire.
    pub fn name(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Preset => "preset",
        }
    }
}

/// One port or param of a capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    /// Its name.
    pub name: String,
    /// Its type as written; `None` where a preset's output is not traced
    /// to a Kernel.
    pub ty: Option<String>,
    /// Its `///` doc.
    pub doc: Option<String>,
}

/// One capability a session reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    /// A Kernel or a preset.
    pub kind: Kind,
    /// A Kernel's id; a preset's path as a scene beside the session's
    /// root names it.
    pub name: String,
    /// What it is for.
    pub doc: Option<String>,
    /// The package it is in, `name@version`.
    pub package: Option<String>,
    /// What that package is for, its manifest's `//!` doc.
    pub package_doc: Option<String>,
    /// What a scene writes to reach it; `None` for a Kernel the session
    /// has published.
    pub reach: Option<String>,
    /// Its inputs, params and outputs.
    pub inputs: Vec<Port>,
    /// Its params.
    pub params: Vec<Port>,
    /// Its outputs.
    pub outputs: Vec<Port>,
    /// The capabilities a Kernel requires.
    pub requires: Vec<String>,
    /// Whether this session grants every one of them.
    pub granted: bool,
    /// Whether the session has it published now.
    pub published: bool,
}

/// What to look for; every given part must match.
#[derive(Clone, Debug, Default)]
pub struct Query {
    /// Words in its name, package, doc or port names, any case.
    pub text: Option<String>,
    /// A type one of its inputs takes, as written: `Rows<Int>`.
    pub takes: Option<String>,
    /// A type one of its outputs gives.
    pub gives: Option<String>,
    /// A capability it requires.
    pub requires: Option<String>,
}

fn same_type(a: &str, b: &str) -> bool {
    let plain = |t: &str| t.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    plain(a).eq_ignore_ascii_case(&plain(b))
}

impl Query {
    /// How well `capability` matches: `None` when it does not; higher is
    /// nearer: each word it matches counts, a word of its name most, of
    /// its own doc and ports next, of its package's least.
    pub fn rank(&self, capability: &Capability) -> Option<u32> {
        if let Some(takes) = &self.takes
            && !capability
                .inputs
                .iter()
                .any(|p| p.ty.as_deref().is_some_and(|t| same_type(t, takes)))
        {
            return None;
        }
        if let Some(gives) = &self.gives
            && !capability
                .outputs
                .iter()
                .any(|p| p.ty.as_deref().is_some_and(|t| same_type(t, gives)))
        {
            return None;
        }
        if let Some(requires) = &self.requires
            && !capability.requires.iter().any(|r| r == requires)
        {
            return None;
        }
        let Some(text) = &self.text else {
            return Some(1);
        };
        // Each word counts where it matches: a word of the name most, then
        // of the capability's own doc and ports, then of its package's; a
        // word matches a word it begins, or that begins it, of four letters
        // or more.
        let tokens = |text: &str| -> Vec<String> {
            text.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|word| !word.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let near = |a: &str, b: &str| {
            a == b || (a.len().min(b.len()) >= 4 && (a.starts_with(b) || b.starts_with(a)))
        };
        let name = tokens(&capability.name);
        let ports = capability
            .inputs
            .iter()
            .chain(&capability.params)
            .chain(&capability.outputs);
        let mut own = Vec::new();
        for part in capability
            .doc
            .iter()
            .chain(ports.clone().map(|p| &p.name))
            .chain(ports.filter_map(|p| p.doc.as_ref()))
        {
            own.extend(tokens(part));
        }
        let mut package = Vec::new();
        for part in capability
            .package
            .iter()
            .chain(capability.package_doc.iter())
        {
            package.extend(tokens(part));
        }
        let mut score: u32 = 0;
        for word in tokens(text) {
            let hit = |words: &[String]| words.iter().any(|w| near(w, &word));
            score += if hit(&name) {
                10
            } else if hit(&own) {
                3
            } else if hit(&package) {
                1
            } else {
                0
            };
        }
        (score > 0).then_some(score)
    }
}

/// `to`, as a path written in a document in `from`.
fn relative(to: &Path, from: &Path) -> String {
    let to: Vec<_> = to.components().collect();
    let from: Vec<_> = from.components().collect();
    let common = to.iter().zip(&from).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..from.len() {
        out.push("..");
    }
    for part in &to[common..] {
        out.push(part.as_os_str());
    }
    out.to_string_lossy().replace('\\', "/")
}

/// The package directory holding `path`, if any.
fn package_of(path: &Path) -> Option<&Path> {
    path.ancestors()
        .skip(1)
        .find(|dir| dir.extension().is_some_and(|ext| ext == "orlapkg"))
}

fn port(name: &orla_language::Spanned<String>, ty: Option<String>, text: &str) -> Port {
    Port {
        name: name.value.clone(),
        ty,
        doc: doc::doc_above(text, name.span.line),
    }
}

/// Every Kernel and preset of the package at `dir`, reached from `base`.
fn in_package(
    store: &dyn Store,
    dir: &Path,
    base: &Path,
    grants: &dyn Fn(&str) -> bool,
) -> Vec<Capability> {
    let Ok(declared) = package_manifest(store, &dir.join("package.osn")) else {
        return Vec::new();
    };
    let package = declared.id.value.clone();
    let package_doc = store
        .read_text(&dir.join("package.osn"))
        .ok()
        .and_then(|text| doc::doc_of_document(&text));
    let reach_package = format!("use \"{}\";", relative(dir, base));
    let mut out = Vec::new();
    // Every Kernel a package's manifest uses, by id, with its ports as
    // written; what its presets' outputs are traced to.
    let mut kernels: Vec<(String, Vec<Port>)> = Vec::new();
    for file in package_files(store, dir).unwrap_or_default() {
        let Ok(text) = store.read_text(&file) else {
            continue;
        };
        let Ok(document) = orla_language::parse_document(&text) else {
            continue;
        };
        for kernel in &document.kernels {
            let Some(id) = &kernel.id else { continue };
            let line = text[..kernel.range.start].matches('\n').count() as u32 + 1;
            let ports = |list: &[orla_language::PortDecl]| {
                list.iter()
                    .map(|p| port(&p.name, Some(p.ty.value.clone()), &text))
                    .collect::<Vec<_>>()
            };
            let requires: Vec<String> = kernel.requires.iter().map(|r| r.value.clone()).collect();
            let outputs = ports(&kernel.outputs);
            kernels.push((id.value.clone(), outputs.clone()));
            out.push(Capability {
                kind: Kind::Kernel,
                name: id.value.clone(),
                doc: doc::doc_above(&text, line),
                package: Some(package.clone()),
                package_doc: package_doc.clone(),
                reach: Some(reach_package.clone()),
                inputs: ports(&kernel.inputs),
                params: kernel
                    .params
                    .iter()
                    .map(|p| port(&p.name, Some(p.ty.value.clone()), &text))
                    .collect(),
                outputs,
                granted: requires.iter().all(|r| grants(r)),
                requires,
                published: false,
            });
        }
    }
    let mut presets: Vec<PathBuf> = store
        .list(&dir.join("graphs"))
        .unwrap_or_default()
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "osn"))
        .collect();
    presets.sort();
    for path in presets {
        let Ok(text) = store.read_text(&path) else {
            continue;
        };
        let Ok(document) = orla_language::parse_document(&text) else {
            continue;
        };
        let node_output = |source: &orla_language::SourceRef| -> Option<String> {
            let node = source.node.as_ref()?;
            let decl = document.nodes.iter().find(|n| n.name.value == *node)?;
            let orla_language::NodeKind::Reference(id) = &decl.kind else {
                return None;
            };
            let (_, outputs) = kernels.iter().find(|(k, _)| *k == id.value)?;
            outputs.iter().find(|p| p.name == source.name)?.ty.clone()
        };
        let path_text = relative(&path, base);
        out.push(Capability {
            kind: Kind::Preset,
            name: path_text.clone(),
            doc: doc::doc_of_document(&text),
            package: Some(package.clone()),
            package_doc: package_doc.clone(),
            reach: Some(format!("node name: graph \"{path_text}\" {{ ... }}")),
            inputs: document
                .inputs
                .iter()
                .map(|p| port(&p.name, Some(p.ty.value.clone()), &text))
                .collect(),
            params: document
                .params
                .iter()
                .map(|p| port(&p.name, Some(p.ty.value.clone()), &text))
                .collect(),
            outputs: document
                .outputs
                .iter()
                .map(|(name, source)| port(name, node_output(source), &text))
                .collect(),
            requires: Vec::new(),
            granted: true,
            published: false,
        });
    }
    out
}

/// Every Kernel and preset of the packages in [`package_dirs`] and in
/// `beside`, each reached from a document in `base`, in package and file
/// order; `grants` says whether a capability is granted.
pub fn in_packages(
    store: &dyn Store,
    base: &Path,
    beside: &[PathBuf],
    grants: &dyn Fn(&str) -> bool,
) -> Vec<Capability> {
    let base = store.canonical(base).unwrap_or_else(|_| base.to_path_buf());
    let mut packages: Vec<PathBuf> = Vec::new();
    for dir in package_dirs().into_iter().chain(beside.iter().cloned()) {
        let dir = store.canonical(&dir).unwrap_or(dir);
        for entry in store.list(&dir).unwrap_or_default() {
            let entry = store.canonical(&entry).unwrap_or(entry);
            if entry.extension().is_some_and(|ext| ext == "orlapkg")
                && store.is_file(&entry.join("package.osn"))
                && !packages.contains(&entry)
            {
                packages.push(entry);
            }
        }
    }
    packages.sort();
    packages
        .iter()
        .flat_map(|dir| in_package(store, dir, &base, grants))
        .collect()
}

/// The capabilities in `capabilities` matching `query`, nearest first, then
/// by name.
pub fn matching(capabilities: Vec<Capability>, query: &Query) -> Vec<Capability> {
    let mut ranked: Vec<(u32, Capability)> = capabilities
        .into_iter()
        .filter_map(|c| query.rank(&c).map(|rank| (rank, c)))
        .collect();
    ranked.sort_by(|(a, x), (b, y)| b.cmp(a).then_with(|| x.name.cmp(&y.name)));
    ranked.into_iter().map(|(_, c)| c).collect()
}

impl Session {
    /// Every capability the session reaches, in a fixed order: what it has
    /// published, then each package's by package and file.
    pub fn capabilities(&self) -> Vec<Capability> {
        let runtime = self.runtime();
        let granted = |name: &str| runtime.grants().iter().any(|c| c.to_string() == name);
        let mut out: Vec<Capability> = Vec::new();
        let mut published: Vec<_> = runtime.kernels().collect();
        published.sort_by_key(|version| version.id().to_string());
        for version in published {
            let manifest = version.manifest();
            let structs = version.module().map(|m| &m.structs[..]).unwrap_or(&[]);
            let docs = doc::kernel_docs(version.source());
            let doc_of = |name: &str| {
                docs.ports
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, d)| d.clone())
            };
            let port = |p: &orla_ir::Port| Port {
                name: p.name.clone(),
                ty: Some(orla_ir::port_type_name(&p.ty, structs)),
                doc: doc_of(&p.name),
            };
            let requires: Vec<String> = manifest.requires.iter().map(|c| c.to_string()).collect();
            out.push(Capability {
                kind: Kind::Kernel,
                name: version.id().to_string(),
                doc: docs.doc.clone(),
                package: None,
                package_doc: None,
                reach: None,
                inputs: manifest.inputs.iter().map(port).collect(),
                params: manifest
                    .params
                    .iter()
                    .map(|p| Port {
                        name: p.name.clone(),
                        ty: Some(orla_ir::type_name(p.ty, structs)),
                        doc: doc_of(&p.name),
                    })
                    .collect(),
                outputs: manifest.outputs.iter().map(port).collect(),
                granted: requires.iter().all(|r| granted(r)),
                requires,
                published: true,
            });
        }
        let base = self
            .root()
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        // Beside the named directories, every one holding a package the
        // scene reads from.
        let beside: Vec<PathBuf> = self
            .watched()
            .into_iter()
            .filter(|document| in_a_package(self.store(), document))
            .filter_map(|document| {
                package_of(&document)
                    .and_then(Path::parent)
                    .map(Path::to_path_buf)
            })
            .collect();
        for capability in in_packages(self.store(), &base, &beside, &granted) {
            // A Kernel the session publishes is answered as published.
            if !out.iter().any(|c| c.published && c.name == capability.name) {
                out.push(capability);
            }
        }
        out
    }

    /// The capabilities matching `query`, nearest first, then by name.
    pub fn search(&self, query: &Query) -> Vec<Capability> {
        matching(self.capabilities(), query)
    }
}
