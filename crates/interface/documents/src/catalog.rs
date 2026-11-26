//! Capability catalog: indexes available packages, Kernels, and preset
//! graphs from a [`Store`] without loading them into a running scene.
//!
//! Designed for both human browsing (`entelechy catalog`) and L7 AI
//! capability search (`op: "catalog"`): search by free-text tokens,
//! package name, execution mode, capability requirements, and Field port
//! types (`Image` matches `Grid<Vec4>` transparently).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use entelechy_language::doc::{kernel_docs, with_doc};

use crate::sources::{package_files, package_manifest};
use crate::store::{Kind, Store};

/// Whether a catalog entry is a Kernel or a reusable preset `.esn` graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CatalogKind {
    /// A `.ent` Kernel (`domain.name@version`).
    Kernel,
    /// A `.esn` preset graph (`pkg.entpkg/graphs/name.esn`).
    Graph,
}

impl CatalogKind {
    /// Lowercase wire name (`"kernel"` or `"graph"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Graph => "graph",
        }
    }
}

/// An input or output port in a catalog entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortEntry {
    /// Port name.
    pub name: String,
    /// Declared Field type as written (e.g. `Image`, `Rows<Vec3>`).
    pub ty: String,
    /// Leading `///` doc comment, if any.
    pub doc: Option<String>,
}

/// A parameter in a catalog entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamEntry {
    /// Parameter name.
    pub name: String,
    /// Declared parameter type as written.
    pub ty: String,
    /// Default literal expression as written.
    pub default: String,
    /// Leading `///` doc comment, if any.
    pub doc: Option<String>,
}

/// One discovered package summary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageSummary {
    /// Short package name (e.g. `image`).
    pub name: String,
    /// Full manifest id (e.g. `image@1`).
    pub id: String,
    /// Top-of-file `//!` documentation from `package.esn`, if any.
    pub doc: Option<String>,
    /// Exact `use` path (e.g. `image.entpkg`).
    pub use_path: String,
    /// Directory holding the package.
    pub path: PathBuf,
    /// Number of Kernels indexed from this package.
    pub kernels: usize,
    /// Number of preset graphs indexed from this package.
    pub graphs: usize,
}

/// One discovered Kernel or preset graph capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Kernel or preset graph.
    pub kind: CatalogKind,
    /// Kernel id (`image.blur@1`) or preset path (`image.entpkg/graphs/blur.esn`).
    pub id: String,
    /// Owning package short name, or `None` for local scene documents.
    pub package: Option<String>,
    /// Ready-to-paste `use` or `graph` reference snippet.
    pub use_stmt: String,
    /// Document path where the capability is declared.
    pub path: PathBuf,
    /// `///` or `//!` doc comment, if any.
    pub doc: Option<String>,
    /// Execution mode (`map`, `reduce`, `scan`, `scatter`, `prefix`, `group`) for Kernels.
    pub mode: Option<String>,
    /// Declared capability requirements (`field.read`, `time.read`, ...).
    pub requires: Vec<String>,
    /// Input ports in declaration order.
    pub inputs: Vec<PortEntry>,
    /// Parameters in declaration order.
    pub params: Vec<ParamEntry>,
    /// Output ports in declaration order.
    pub outputs: Vec<PortEntry>,
}

/// Filters for [`query`]. All supplied filters combine with logical AND.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatalogQuery {
    /// Free-text tokens matched against id, package, doc, port names, and types.
    pub query: Option<String>,
    /// Filter to a single package name (with or without `.entpkg`) or `"local"`.
    pub package: Option<String>,
    /// Filter to kernels or preset graphs only.
    pub kind: Option<CatalogKind>,
    /// Filter Kernels by execution mode (`map`, `reduce`, `scan`, `scatter`, `prefix`, `group`).
    pub mode: Option<String>,
    /// Match entries having at least one input port of this type (`Image` <-> `Grid<Vec4>`).
    pub input: Option<String>,
    /// Match entries having at least one output port of this type (`Image` <-> `Grid<Vec4>`).
    pub output: Option<String>,
    /// Match entries declaring this capability (or `"pure"` for no `time.read`/`input.read`).
    pub requires: Option<String>,
    /// Maximum entries returned (`None` returns all matches).
    pub limit: Option<usize>,
}

/// Result of [`query`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatalogResult {
    /// All packages discovered in search precedence order.
    pub packages: Vec<PackageSummary>,
    /// Total matching entries before `limit`.
    pub total: usize,
    /// Returned entries, ranked by relevance then deterministic order.
    pub entries: Vec<CatalogEntry>,
}

/// Discovers and filters every reachable package, Kernel, and preset graph
/// from `store` and optional `scene_dir`.
pub fn query(store: &dyn Store, scene_dir: Option<&Path>, filter: &CatalogQuery) -> CatalogResult {
    let (packages, all_entries) = scan(store, scene_dir);
    let mut scored: Vec<(u32, CatalogEntry)> = all_entries
        .into_iter()
        .filter_map(|entry| {
            let score = match_entry(&entry, filter)?;
            Some((score, entry))
        })
        .collect();
    scored.sort_by(|(sa, a), (sb, b)| {
        sb.cmp(sa)
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.package.cmp(&b.package))
            .then_with(|| a.id.cmp(&b.id))
    });
    let total = scored.len();
    let limit = filter.limit.unwrap_or(total);
    let entries = scored
        .into_iter()
        .take(limit)
        .map(|(_, entry)| entry)
        .collect();
    CatalogResult {
        packages,
        total,
        entries,
    }
}

/// Scans reachable package directories and local scene `.ent` documents.
pub fn scan(
    store: &dyn Store,
    scene_dir: Option<&Path>,
) -> (Vec<PackageSummary>, Vec<CatalogEntry>) {
    let mut packages = Vec::new();
    let mut entries: Vec<CatalogEntry> = Vec::new();
    let mut seen_packages = BTreeSet::new();
    let mut seen_files = BTreeSet::new();

    // 1. Discover packages in the exact resolution order used by `sources.rs`:
    //    beside the scene first, then every directory in `store.packages()`.
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = scene_dir {
        roots.push(dir.to_path_buf());
    }
    roots.extend(store.packages().iter().cloned());

    for root in roots {
        let Ok(children) = store.list(&root) else {
            continue;
        };
        for candidate in children {
            if store.kind(&candidate) != Some(Kind::Dir) {
                continue;
            }
            let Some(name) = candidate
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
            else {
                continue;
            };
            let Some(short) = name.strip_suffix(".entpkg") else {
                continue;
            };
            if !seen_packages.insert(short.to_owned()) {
                continue;
            }
            let manifest = candidate.join("package.esn");
            if !store.is_file(&manifest) {
                continue;
            }
            let Ok(declared) = package_manifest(store, &manifest) else {
                continue;
            };
            let pkg_doc = store
                .read_text(&manifest)
                .ok()
                .and_then(|text| top_doc(&text));
            let use_path = format!("{short}.entpkg");
            let use_stmt = format!("use \"{use_path}\";");

            let before_k = entries
                .iter()
                .filter(|e| e.kind == CatalogKind::Kernel)
                .count();
            let before_g = entries
                .iter()
                .filter(|e| e.kind == CatalogKind::Graph)
                .count();

            // Index Kernels exported by `package.esn`.
            let kernel_files = package_files(store, &candidate).unwrap_or_default();
            for file in kernel_files {
                let key = store.canonical(&file).unwrap_or_else(|_| file.clone());
                if !seen_files.insert(key) {
                    continue;
                }
                index_kernel_file(
                    store,
                    &file,
                    Some(short.to_owned()),
                    &use_stmt,
                    &mut entries,
                );
            }

            // Index preset graphs under `<pkg>.entpkg/graphs/**/*.esn`.
            let graphs_dir = candidate.join("graphs");
            if store.kind(&graphs_dir) == Some(Kind::Dir) {
                let mut graph_files = Vec::new();
                collect_esn(store, &graphs_dir, &mut graph_files);
                for file in graph_files {
                    let rel = file
                        .strip_prefix(&candidate)
                        .unwrap_or(&file)
                        .to_string_lossy()
                        .replace('\\', "/");
                    let preset_id = format!("{short}.entpkg/{rel}");
                    index_graph_file(
                        store,
                        &file,
                        &preset_id,
                        Some(short.to_owned()),
                        &mut entries,
                    );
                }
            }

            let kernels = entries
                .iter()
                .filter(|e| e.kind == CatalogKind::Kernel)
                .count()
                - before_k;
            let graphs = entries
                .iter()
                .filter(|e| e.kind == CatalogKind::Graph)
                .count()
                - before_g;

            packages.push(PackageSummary {
                name: short.to_owned(),
                id: declared.id.value,
                doc: pkg_doc,
                use_path,
                path: candidate,
                kernels,
                graphs,
            });
        }
    }

    // 2. Index local `.ent` libraries beside the scene (`scene_dir/*.ent` and `scene_dir/kernels/*.ent`).
    if let Some(dir) = scene_dir {
        let mut local_files = Vec::new();
        collect_local_ent(store, dir, &mut local_files);
        for file in local_files {
            let key = store.canonical(&file).unwrap_or_else(|_| file.clone());
            if !seen_files.insert(key) {
                continue;
            }
            let rel = file
                .strip_prefix(dir)
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            let use_stmt = format!("use \"{rel}\";");
            index_kernel_file(store, &file, None, &use_stmt, &mut entries);
        }
    }

    packages.sort_by(|a, b| a.name.cmp(&b.name));
    (packages, entries)
}

fn collect_esn(store: &dyn Store, dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(children) = store.list(dir) else {
        return;
    };
    for path in children {
        match store.kind(&path) {
            Some(Kind::Dir) => collect_esn(store, &path, out),
            Some(Kind::File) if path.extension().is_some_and(|e| e == "esn") => out.push(path),
            _ => {}
        }
    }
}

fn collect_local_ent(store: &dyn Store, dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(children) = store.list(dir) else {
        return;
    };
    for path in children {
        let hidden = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'));
        if hidden {
            continue;
        }
        match store.kind(&path) {
            Some(Kind::File) if path.extension().is_some_and(|e| e == "ent") => out.push(path),
            Some(Kind::Dir) => {
                // Skip package directories (already scanned above) and tests.
                if path.extension().is_some_and(|e| e == "entpkg")
                    || path.file_name().is_some_and(|n| n == "tests")
                {
                    continue;
                }
                if let Ok(inner) = store.list(&path) {
                    for sub in inner {
                        if store.kind(&sub) == Some(Kind::File)
                            && sub.extension().is_some_and(|e| e == "ent")
                        {
                            out.push(sub);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn index_kernel_file(
    store: &dyn Store,
    path: &Path,
    package: Option<String>,
    use_stmt: &str,
    out: &mut Vec<CatalogEntry>,
) {
    let Ok(text) = store.read_text(path) else {
        return;
    };
    let Ok(document) = store.document(path) else {
        return;
    };
    for kernel in &document.kernels {
        let Some(id) = &kernel.id else {
            continue;
        };
        let slice = text
            .get(with_doc(&text, kernel.range.clone()))
            .unwrap_or("");
        let docs = kernel_docs(slice);
        let port_docs: BTreeMap<&str, &str> = docs
            .ports
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let inputs = kernel
            .inputs
            .iter()
            .map(|p| PortEntry {
                name: p.name.value.clone(),
                ty: p.ty.value.clone(),
                doc: port_docs
                    .get(p.name.value.as_str())
                    .map(|s| (*s).to_owned()),
            })
            .collect();
        let params = kernel
            .params
            .iter()
            .map(|p| ParamEntry {
                name: p.name.value.clone(),
                ty: p.ty.value.clone(),
                default: text
                    .get(p.default_range.clone())
                    .unwrap_or("")
                    .trim()
                    .to_owned(),
                doc: port_docs
                    .get(p.name.value.as_str())
                    .map(|s| (*s).to_owned()),
            })
            .collect();
        let outputs = kernel
            .outputs
            .iter()
            .map(|p| PortEntry {
                name: p.name.value.clone(),
                ty: p.ty.value.clone(),
                doc: port_docs
                    .get(p.name.value.as_str())
                    .map(|s| (*s).to_owned()),
            })
            .collect();

        out.push(CatalogEntry {
            kind: CatalogKind::Kernel,
            id: id.value.clone(),
            package: package.clone(),
            use_stmt: use_stmt.to_owned(),
            path: path.to_path_buf(),
            doc: docs.doc,
            mode: Some(kernel_mode(kernel).to_owned()),
            requires: kernel.requires.iter().map(|r| r.value.clone()).collect(),
            inputs,
            params,
            outputs,
        });
    }
}

fn index_graph_file(
    store: &dyn Store,
    path: &Path,
    preset_id: &str,
    package: Option<String>,
    out: &mut Vec<CatalogEntry>,
) {
    let Ok(text) = store.read_text(path) else {
        return;
    };
    let Ok(document) = store.document(path) else {
        return;
    };
    let inputs = document
        .inputs
        .iter()
        .map(|p| PortEntry {
            name: p.name.value.clone(),
            ty: p.ty.value.clone(),
            doc: None,
        })
        .collect();
    let params = document
        .params
        .iter()
        .map(|p| ParamEntry {
            name: p.name.value.clone(),
            ty: p.ty.value.clone(),
            default: text
                .get(p.default_range.clone())
                .unwrap_or("")
                .trim()
                .to_owned(),
            doc: None,
        })
        .collect();
    let outputs = document
        .outputs
        .iter()
        .map(|(name, _)| PortEntry {
            name: name.value.clone(),
            ty: "Field".to_owned(),
            doc: None,
        })
        .collect();

    out.push(CatalogEntry {
        kind: CatalogKind::Graph,
        id: preset_id.to_owned(),
        package,
        use_stmt: format!("node fx: graph \"{preset_id}\" {{ ... }}"),
        path: path.to_path_buf(),
        doc: top_doc(&text),
        mode: None,
        requires: Vec::new(),
        inputs,
        params,
        outputs,
    });
}

fn kernel_mode(kernel: &entelechy_language::Kernel) -> &'static str {
    if kernel.merge.is_some() {
        "reduce"
    } else if kernel.carry.is_some() {
        "scan"
    } else if kernel.scatter.is_some() {
        "scatter"
    } else if kernel.prefix.is_some() {
        "prefix"
    } else if kernel.group.is_some() {
        "group"
    } else {
        "map"
    }
}

/// Extracts leading `//!` documentation lines at the top of a `.esn` file.
fn top_doc(text: &str) -> Option<String> {
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("//!") {
            lines.push(rest.strip_prefix(' ').unwrap_or(rest).trim_end());
        } else if trimmed.is_empty() && lines.is_empty() {
            continue;
        } else {
            break;
        }
    }
    let joined = lines.join("\n").trim().to_owned();
    (!joined.is_empty()).then_some(joined)
}

fn normalize_type(ty: &str) -> String {
    let compact: String = ty
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    if compact == "image" {
        "grid<vec4>".to_owned()
    } else {
        compact
    }
}

fn type_matches(port_ty: &str, wanted: &str) -> bool {
    let a = normalize_type(port_ty);
    let b = normalize_type(wanted);
    a == b || a.contains(&b)
}

fn match_entry(entry: &CatalogEntry, filter: &CatalogQuery) -> Option<u32> {
    if let Some(kind) = filter.kind
        && entry.kind != kind
    {
        return None;
    }
    if let Some(pkg) = &filter.package {
        let wanted = pkg.strip_suffix(".entpkg").unwrap_or(pkg);
        if wanted.eq_ignore_ascii_case("local") {
            if entry.package.is_some() {
                return None;
            }
        } else if entry
            .package
            .as_deref()
            .is_none_or(|p| !p.eq_ignore_ascii_case(wanted))
        {
            return None;
        }
    }
    if let Some(mode) = &filter.mode
        && entry
            .mode
            .as_deref()
            .is_none_or(|m| !m.eq_ignore_ascii_case(mode))
    {
        return None;
    }
    if let Some(wanted) = &filter.input
        && !entry.inputs.iter().any(|p| type_matches(&p.ty, wanted))
    {
        return None;
    }
    if let Some(wanted) = &filter.output
        && !entry.outputs.iter().any(|p| type_matches(&p.ty, wanted))
    {
        return None;
    }
    if let Some(req) = &filter.requires {
        if req.eq_ignore_ascii_case("pure") {
            if entry
                .requires
                .iter()
                .any(|r| r == "time.read" || r == "input.read")
            {
                return None;
            }
        } else if !entry.requires.iter().any(|r| r.eq_ignore_ascii_case(req)) {
            return None;
        }
    }

    let Some(raw_q) = &filter.query else {
        return Some(1);
    };
    let tokens: Vec<String> = raw_q
        .split_whitespace()
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return Some(1);
    }

    let id_low = entry.id.to_ascii_lowercase();
    let pkg_low = entry
        .package
        .as_deref()
        .unwrap_or("local")
        .to_ascii_lowercase();
    let doc_low = entry.doc.as_deref().unwrap_or("").to_ascii_lowercase();
    let ports_low: String = entry
        .inputs
        .iter()
        .chain(&entry.outputs)
        .map(|p| {
            format!(
                "{} {} {} ",
                p.name.to_ascii_lowercase(),
                p.ty.to_ascii_lowercase(),
                p.doc.as_deref().unwrap_or("").to_ascii_lowercase()
            )
        })
        .chain(entry.params.iter().map(|p| {
            format!(
                "{} {} {} ",
                p.name.to_ascii_lowercase(),
                p.ty.to_ascii_lowercase(),
                p.doc.as_deref().unwrap_or("").to_ascii_lowercase()
            )
        }))
        .collect();

    let mut score = 0u32;
    for tok in &tokens {
        let mut tok_score = 0u32;
        if id_low == *tok {
            tok_score += 100;
        } else if id_low.contains(tok) {
            tok_score += 40;
        }
        if pkg_low == *tok {
            tok_score += 30;
        } else if pkg_low.contains(tok) {
            tok_score += 15;
        }
        if doc_low.contains(tok) {
            tok_score += 12;
        }
        if ports_low.contains(tok) {
            tok_score += 8;
        }
        if tok_score == 0 {
            return None;
        }
        score += tok_score;
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Memory;

    fn sample_store() -> Memory {
        let mem = Memory::default();
        mem.put(
            Path::new("packages/demo.entpkg/package.esn"),
            b"//! Demo image utilities.\npackage demo@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n".to_vec(),
        );
        mem.put(
            Path::new("packages/demo.entpkg/kernels/ops.ent"),
            b"/// Softens an image.\nkernel demo.blur@1 {\n    requires { field.read; field.write; }\n    /// Source pixels.\n    input source: Image;\n    /// Blur radius.\n    param radius: Float = 2.0;\n    output result: Grid<Vec4>;\n    compute { return sample(source); }\n}\n\n/// Sums rows.\nkernel demo.sum@1 {\n    requires { field.read; field.write; }\n    input values: Rows<Float>;\n    output result: Rows<Float>;\n    compute over values { return sample(values); }\n    merge(a, b) { return a + b; }\n}\n".to_vec(),
        );
        mem.put(
            Path::new("packages/demo.entpkg/graphs/bloom.esn"),
            b"//! Two-pass glow.\nuse \"../package.esn\";\ninput source: Image;\nparam strength: Float = 0.5;\noutput result <- source;\n".to_vec(),
        );
        mem
    }

    #[test]
    fn discovers_packages_kernels_and_preset_graphs() {
        let store = sample_store();
        let res = query(
            &store,
            Some(Path::new("packages")),
            &CatalogQuery::default(),
        );
        assert_eq!(res.packages.len(), 1);
        assert_eq!(res.packages[0].name, "demo");
        assert_eq!(res.packages[0].kernels, 2);
        assert_eq!(res.packages[0].graphs, 1);
        assert_eq!(
            res.packages[0].doc.as_deref(),
            Some("Demo image utilities.")
        );
        assert_eq!(res.total, 3);
    }

    #[test]
    fn filters_by_type_alias_mode_and_text_query() {
        let store = sample_store();
        // `Grid<Vec4>` matches `Image` input transparently.
        let by_type = query(
            &store,
            Some(Path::new("packages")),
            &CatalogQuery {
                input: Some("Grid<Vec4>".into()),
                kind: Some(CatalogKind::Kernel),
                ..Default::default()
            },
        );
        assert_eq!(by_type.entries.len(), 1);
        assert_eq!(by_type.entries[0].id, "demo.blur@1");
        assert_eq!(by_type.entries[0].params[0].default, "2.0");
        assert_eq!(
            by_type.entries[0].inputs[0].doc.as_deref(),
            Some("Source pixels.")
        );

        let by_mode = query(
            &store,
            Some(Path::new("packages")),
            &CatalogQuery {
                mode: Some("reduce".into()),
                ..Default::default()
            },
        );
        assert_eq!(by_mode.entries.len(), 1);
        assert_eq!(by_mode.entries[0].id, "demo.sum@1");

        let by_query = query(
            &store,
            Some(Path::new("packages")),
            &CatalogQuery {
                query: Some("glow".into()),
                ..Default::default()
            },
        );
        assert_eq!(by_query.entries.len(), 1);
        assert_eq!(by_query.entries[0].kind, CatalogKind::Graph);
        assert_eq!(by_query.entries[0].id, "demo.entpkg/graphs/bloom.esn");
    }
}
