//! What a session reaches and finds: the Kernels it publishes, and every
//! Kernel and preset graph of each package beside one it uses, each with its
//! docs, its types, what it requires and how a scene reaches it.

use std::path::PathBuf;

use orla_session::Session;
use orla_session::search::{Kind, Query};

const PACKAGE: &str = "//! Counting things.
package tally@1 { language = 1; ir = 1; }
use \"kernels/\";
";

const KERNELS: &str = "// A file of Kernels.

/// Each count doubled.
kernel tally.twice@1 {
    requires { field.read; field.write; }
    /// The counts.
    input counts: Rows<Int>;
    /// How many times over.
    param by: Int = 2;
    output result: Rows<Int>;
    compute { return sample(counts) * by; }
}
";

const EXTRA: &str = "package extra@1 { language = 1; ir = 1; }
use \"kernels/\";
";

const CLOCK: &str = "/// The time, as a count.
kernel extra.clock@1 {
    requires { field.write; time.read; }
    output result: Rows<Int>;
    compute { return int(time()); }
}
";

const PRESET: &str = "//! Doubles twice.
use \"../kernels/tally.orla\";
/// What to double.
input counts: Rows<Int>;
node once: tally.twice@1 { counts <- counts; }
node again: tally.twice@1 { counts <- once.result; }
/// Four times each.
output result <- again.result;
";

const LOCAL: &str = "/// The scene's own start.
kernel app.start@1 {
    requires { field.write; }
    output result: Rows<Int>;
    compute { return 1; }
}
";

const SCENE: &str = "use \"start.orla\";
use \"../packages/tally.orlapkg\";
node start: app.start@1 { domain = 3; }
node twice: tally.twice@1 { counts <- start.result; }
output result <- twice.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-search-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let package = dir.join("packages/tally.orlapkg");
    std::fs::create_dir_all(package.join("kernels")).unwrap();
    std::fs::create_dir_all(package.join("graphs")).unwrap();
    std::fs::create_dir_all(dir.join("app")).unwrap();
    std::fs::write(package.join("package.osn"), PACKAGE).unwrap();
    std::fs::write(package.join("kernels/tally.orla"), KERNELS).unwrap();
    std::fs::write(package.join("graphs/quad.osn"), PRESET).unwrap();
    // A package the scene does not use, beside one it does.
    let extra = dir.join("packages/extra.orlapkg");
    std::fs::create_dir_all(extra.join("kernels")).unwrap();
    std::fs::write(extra.join("package.osn"), EXTRA).unwrap();
    std::fs::write(extra.join("kernels/clock.orla"), CLOCK).unwrap();
    std::fs::write(dir.join("app/start.orla"), LOCAL).unwrap();
    std::fs::write(dir.join("app/scene.osn"), SCENE).unwrap();
    dir
}

#[test]
fn a_session_finds_what_it_publishes_and_what_its_packages_hold() {
    let dir = scratch("beside");
    let mut session = Session::open(&dir.join("app/scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");

    let all = session.capabilities();
    let named = |name: &str| {
        all.iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("{name} in {all:?}"))
    };
    let start = named("app.start@1");
    assert!(start.published && start.reach.is_none());
    assert_eq!(start.doc.as_deref(), Some("The scene's own start."));
    let twice = named("tally.twice@1");
    assert!(
        twice.published,
        "a published package Kernel is answered once, as published"
    );
    assert_eq!(twice.package.as_deref(), Some("tally@1"));
    assert_eq!(
        twice.reach.as_deref(),
        Some("use \"../packages/tally.orlapkg\";"),
        "with the line that reaches it"
    );
    assert_eq!(twice.inputs[0].doc.as_deref(), Some("The counts."));
    assert_eq!(twice.params[0].doc.as_deref(), Some("How many times over."));
    let clock = named("extra.clock@1");
    assert!(!clock.published);
    assert_eq!(clock.package.as_deref(), Some("extra@1"));
    assert_eq!(
        clock.reach.as_deref(),
        Some("use \"../packages/extra.orlapkg\";")
    );
    assert_eq!(clock.requires, ["field.write", "time.read"]);
    assert!(clock.granted);
    let quad = named("../packages/tally.orlapkg/graphs/quad.osn");
    assert_eq!(quad.kind, Kind::Preset);
    assert_eq!(quad.doc.as_deref(), Some("Doubles twice."));
    assert_eq!(quad.inputs[0].ty.as_deref(), Some("Rows<Int>"));
    assert_eq!(quad.inputs[0].doc.as_deref(), Some("What to double."));
    assert_eq!(
        quad.outputs[0].ty.as_deref(),
        Some("Rows<Int>"),
        "traced to tally.twice@1"
    );
    assert_eq!(quad.outputs[0].doc.as_deref(), Some("Four times each."));
    assert_eq!(all, session.capabilities(), "the same every time");

    let names = |query: Query| -> Vec<String> {
        session.search(&query).into_iter().map(|c| c.name).collect()
    };
    let found = names(Query {
        text: Some("twice".into()),
        ..Query::default()
    });
    assert_eq!(
        found[0], "tally.twice@1",
        "a name holding the word first: {found:?}"
    );
    assert!(
        found.contains(&"../packages/tally.orlapkg/graphs/quad.osn".to_owned()),
        "a doc holding it: {found:?}"
    );
    assert_eq!(
        names(Query {
            takes: Some("Rows<Int>".into()),
            ..Query::default()
        }),
        ["../packages/tally.orlapkg/graphs/quad.osn", "tally.twice@1"]
    );
    assert_eq!(
        names(Query {
            requires: Some("time.read".into()),
            ..Query::default()
        }),
        ["extra.clock@1"]
    );
    assert!(
        names(Query {
            text: Some("nothing here".into()),
            ..Query::default()
        })
        .is_empty()
    );
    // Any word counts, a word of the name most; a word matches one it
    // begins or that begins it: `counting` finds `count`.
    let found = names(Query {
        text: Some("counting clock".into()),
        ..Query::default()
    });
    assert_eq!(
        found[0], "extra.clock@1",
        "the name holds `clock`: {found:?}"
    );
    assert!(
        found.contains(&"tally.twice@1".to_owned()),
        "a doc holds `count`: {found:?}"
    );

    // A capability the session does not grant is found, and says so.
    let mut grants = orla_kernel::Capabilities::none();
    grants.insert(orla_kernel::Capability::FieldRead);
    grants.insert(orla_kernel::Capability::FieldWrite);
    let mut runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    runtime.set_grants(grants);
    let mut narrow = Session::open_with(&dir.join("app/scene.osn"), None, runtime);
    narrow.sync();
    let clock = narrow
        .capabilities()
        .into_iter()
        .find(|c| c.name == "extra.clock@1")
        .unwrap();
    assert!(!clock.granted);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package installed where the store finds packages is reached by its
/// name, and the line search gives loads; a package directory is readable
/// where reads are held, by the path to it as well.
#[test]
fn an_installed_package_is_reached_by_its_name() {
    let dir = scratch("installed");
    let store = std::sync::Arc::new(orla_session::Disk::with_packages(vec![
        dir.join("packages"),
    ]));
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let mut session = Session::open_in(store, &dir.join("app/scene.osn"), None, runtime);
    session.restrict_reads().unwrap();
    session.sync();
    let all = session.capabilities();
    let reach = |name: &str| {
        all.iter()
            .find(|c| c.name == name)
            .and_then(|c| c.reach.clone())
    };
    assert_eq!(
        reach("extra.clock@1").as_deref(),
        Some("use \"extra.orlapkg\";")
    );
    assert!(
        reach("tally.orlapkg/graphs/quad.osn").is_some(),
        "a preset by its package's name: {all:?}"
    );
    for line in [
        "use \"extra.orlapkg\";",
        "use \"../packages/extra.orlapkg\";",
    ] {
        let text = format!(
            "use \"start.orla\";\n{line}\nnode start: app.start@1 {{ domain = 3; }}\nnode clock: extra.clock@1 {{ domain = 3; }}\noutput result <- clock.result;\n"
        );
        session.write_scene(&text).unwrap().unwrap();
        let events = session.sync();
        assert!(
            session
                .graph()
                .is_some_and(|graph| graph.nodes.iter().any(|node| node.id.0 == "clock")),
            "{line}: {events:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
