//! A package is what [`orla_session::package::LAYOUT`] says: each part of
//! the table is proved here on a package built to it, and a part the table
//! gains without a proof fails.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_session::Session;
use orla_session::search::Kind;

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
use \"../kernels/tally.orla\";
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
node shown: kernel {{
    requires {{ field.read; field.write; }}
    input p: Image <- picture;
    output result: Image;
    compute {{ return sample(p); }}
}}
output result <- twice.result;
output view <- shown.result;
"
    )
}

/// A directory holding `packages/tally.orlapkg`, built to the table, and
/// an empty `app/`.
fn built(name: &str, language: u32) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-package-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let package = dir.join("packages/tally.orlapkg");
    std::fs::create_dir_all(package.join("kernels")).unwrap();
    std::fs::create_dir_all(package.join("graphs")).unwrap();
    std::fs::create_dir_all(dir.join("app")).unwrap();
    std::fs::write(
        package.join("package.osn"),
        MANIFEST.replace("language = 1", &format!("language = {language}")),
    )
    .unwrap();
    std::fs::write(package.join("kernels/tally.orla"), KERNELS).unwrap();
    std::fs::write(package.join("graphs/quad.osn"), PRESET).unwrap();
    // A 1x1 picture, as a P6 file.
    std::fs::write(package.join("dot.ppm"), b"P6\n1 1\n255\n\xff\x80\x00").unwrap();
    dir
}

/// A session of `app/scene.osn`, its packages found in `packages/` by
/// name, its reads held to its own directories.
fn served(dir: &Path, reach: &str, asset: &str) -> Session {
    std::fs::write(dir.join("app/scene.osn"), scene(reach, asset)).unwrap();
    let store = Arc::new(orla_session::Disk::with_packages(vec![
        dir.join("packages"),
    ]));
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let mut session = Session::open_in(store, &dir.join("app/scene.osn"), None, runtime);
    session.restrict_reads().unwrap();
    session
}

#[test]
fn a_package_is_what_its_layout_says() {
    let dir = built("layout", 1);
    let mut session = served(&dir, "tally.orlapkg", "../packages/tally.orlapkg/dot.ppm");
    let events = session.sync();
    let loaded = session.graph().is_some();
    let capabilities = session.capabilities();
    for part in &orla_session::package::LAYOUT {
        match part.name {
            // The manifest: a language this Orla lacks refuses the
            // package, naming both.
            "manifest" => {
                let newer = built("manifest", 99);
                let mut refused =
                    served(&newer, "tally.orlapkg", "../packages/tally.orlapkg/dot.ppm");
                let events = refused.sync();
                assert!(refused.graph().is_none(), "{events:?}");
                assert!(
                    format!("{events:?}").contains("needs language 99"),
                    "{events:?}"
                );
                let _ = std::fs::remove_dir_all(&newer);
            }
            // The Kernels its manifest uses are published.
            "kernels" => {
                assert!(loaded, "{events:?}");
                assert!(
                    session
                        .runtime()
                        .kernel(&"tally.twice@1".parse().unwrap())
                        .is_some()
                );
            }
            // A preset in `graphs/` is listed, reached by the package's
            // name.
            "graphs" => {
                let preset = capabilities
                    .iter()
                    .find(|c| c.kind == Kind::Preset)
                    .unwrap_or_else(|| panic!("no preset in {capabilities:?}"));
                assert_eq!(preset.name, "tally.orlapkg/graphs/quad.osn");
            }
            // The docs are what search answers.
            "docs" => {
                let named = |name: &str| capabilities.iter().find(|c| c.name == name);
                assert_eq!(
                    named("tally.twice@1").and_then(|c| c.doc.as_deref()),
                    Some("Each count doubled.")
                );
                assert_eq!(
                    named("tally.orlapkg/graphs/quad.osn").and_then(|c| c.doc.as_deref()),
                    Some("Doubles twice.")
                );
            }
            // A file of the package is read where reads are held.
            "assets" => {
                assert!(
                    events.iter().any(|e| matches!(
                        e,
                        orla_session::Event::Loaded { name, .. } if name == "picture"
                    )),
                    "the picture in the package is read: {events:?}"
                );
            }
            // Found by its name, and reached by it.
            "by name" => {
                assert!(loaded, "{events:?}");
                let twice = capabilities.iter().find(|c| c.name == "tally.twice@1");
                assert_eq!(
                    twice.and_then(|c| c.reach.as_deref()),
                    Some("use \"tally.orlapkg\";")
                );
            }
            other => panic!("the layout's `{other}` has no proof here"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
