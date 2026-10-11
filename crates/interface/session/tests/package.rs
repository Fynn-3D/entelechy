//! A package is what [`entelechy_session::package::LAYOUT`] says: each part of
//! the table is proved here on a package built to it, and a part the table
//! gains without a proof fails.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_session::Session;

const MANIFEST: &str = "//! Counting things.
package tally@1 { language = 1; ir = 1; }
use \"kernels/\";
";

const KERNELS: &str = "/// Each count doubled.
kernel tally.twice@1 {
    requires { field.read; field.write; }
    input counts: Rows<Int>;
    output result: Rows<Int>;
    compute { return sample(counts) * 2; }
}
";

const PRESET: &str = "//! Doubles twice.
use \"../kernels/tally.ent\";
input counts: Rows<Int>;
node once: tally.twice@1 { counts <- counts; }
node again: tally.twice@1 { counts <- once.result; }
output result <- again.result;
";

/// A scene in `app/` that uses the package by `reach` and reads `asset`.
fn scene(reach: &str, asset: &str) -> String {
    format!(
        "use \"{reach}\";
input picture: Image = \"{asset}\";
node start: Rows<Int> {{ domain = 3; value = 1; }}
node twice: tally.twice@1 {{ counts <- start.result; }}
node quad: graph \"{reach}/graphs/quad.esn\" {{ counts <- start.result; }}
node shown: kernel {{
    requires {{ field.read; field.write; }}
    input p: Image <- picture;
    output result: Image;
    compute {{ return sample(p); }}
}}
output result <- twice.result;
output four <- quad.result;
output view <- shown.result;
"
    )
}

/// A directory holding `packages/tally.entpkg`, built to the table, and
/// an empty `app/`.
fn built(name: &str, language: u32) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-package-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let package = dir.join("packages/tally.entpkg");
    std::fs::create_dir_all(package.join("kernels")).unwrap();
    std::fs::create_dir_all(package.join("graphs")).unwrap();
    std::fs::create_dir_all(dir.join("app")).unwrap();
    std::fs::write(
        package.join("package.esn"),
        MANIFEST.replace("language = 1", &format!("language = {language}")),
    )
    .unwrap();
    std::fs::write(package.join("kernels/tally.ent"), KERNELS).unwrap();
    std::fs::write(package.join("graphs/quad.esn"), PRESET).unwrap();
    // A 1x1 picture, as a P6 file.
    std::fs::write(package.join("dot.ppm"), b"P6\n1 1\n255\n\xff\x80\x00").unwrap();
    dir
}

/// A session of `app/scene.esn`, its packages found in `packages/` by
/// name, its reads held to its own directories.
fn served(dir: &Path, reach: &str, asset: &str) -> Session {
    std::fs::write(dir.join("app/scene.esn"), scene(reach, asset)).unwrap();
    let store = Arc::new(entelechy_session::Disk::with_packages(vec![
        dir.join("packages"),
    ]));
    let runtime = Session::runtime_for("cpu", entelechy_session::Limits::default()).unwrap();
    let mut session = Session::open_in(store, &dir.join("app/scene.esn"), None, runtime);
    session.restrict_reads().unwrap();
    session
}

#[test]
fn a_package_is_what_its_layout_says() {
    let dir = built("layout", 1);
    let mut session = served(&dir, "tally.entpkg", "../packages/tally.entpkg/dot.ppm");
    let events = session.sync();
    let loaded = session.graph().is_some();
    for part in &entelechy_session::package::LAYOUT {
        match part.name {
            // The manifest: a language this Entelechy lacks refuses the
            // package, naming both.
            "manifest" => {
                let newer = built("manifest", 99);
                let mut refused =
                    served(&newer, "tally.entpkg", "../packages/tally.entpkg/dot.ppm");
                let events = refused.sync();
                assert!(refused.graph().is_none(), "{events:?}");
                assert!(
                    format!("{events:?}").contains("needs language 99"),
                    "{events:?}"
                );
                let _ = std::fs::remove_dir_all(&newer);
            }
            // The Kernels its manifest uses are published; a scene
            // outside it uses the package whole, never a file inside.
            "kernels" => {
                assert!(loaded, "{events:?}");
                let mut inside = served(
                    &dir,
                    "tally.entpkg/kernels/tally.ent",
                    "../packages/tally.entpkg/dot.ppm",
                );
                let refused = inside.sync();
                assert!(inside.graph().is_none(), "{refused:?}");
                assert!(
                    format!("{refused:?}").contains("use the package whole"),
                    "{refused:?}"
                );
                assert!(
                    session
                        .runtime()
                        .kernel(&"tally.twice@1".parse().unwrap())
                        .is_some()
                );
            }
            // A preset in `graphs/` is placed by the package's name.
            "graphs" => {
                let graph = session.graph().unwrap_or_else(|| panic!("{events:?}"));
                assert!(
                    graph.nodes.iter().any(|node| node.id.0 == "quad.again"),
                    "{:?}",
                    graph.nodes
                );
            }
            // The docs are in the source, and travel with the Kernel.
            "docs" => {
                let kernel = session
                    .runtime()
                    .kernel(&"tally.twice@1".parse().unwrap())
                    .unwrap();
                assert_eq!(
                    entelechy_language::doc::kernel_docs(kernel.source())
                        .doc
                        .as_deref(),
                    Some("Each count doubled.")
                );
            }
            // A file of the package is read where reads are held, by its
            // path or through the package's name.
            "assets" => {
                assert!(
                    events.iter().any(|e| matches!(
                        e,
                        entelechy_session::Event::Loaded { name, .. } if name == "picture"
                    )),
                    "the picture in the package is read: {events:?}"
                );
                let mut named = served(&dir, "tally.entpkg", "tally.entpkg/dot.ppm");
                let events = named.sync();
                assert!(
                    events.iter().any(|e| matches!(
                        e,
                        entelechy_session::Event::Loaded { name, path, .. }
                            if name == "picture" && path.ends_with("packages/tally.entpkg/dot.ppm")
                    )),
                    "the picture read through the package's name: {events:?}"
                );
            }
            // Found by its name, and reached by it.
            "by name" => assert!(loaded, "{events:?}"),
            other => panic!("the layout's `{other}` has no proof here"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
