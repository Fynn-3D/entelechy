//! Declarations shared through `use`: one declaration reached from two
//! files is one, wherever written; two different ones of a name are each
//! their own document's, and a document reaching both refuses the name.

use std::path::PathBuf;

use orla_session::declarations_for;

fn scratch(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-sources-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, text) in files {
        std::fs::write(dir.join(file), text).unwrap();
    }
    dir
}

fn uses(text: &str) -> Vec<orla_language::Use> {
    orla_language::parse_document(text).unwrap().uses
}

const NOTHING: orla_language::Own<'static> = orla_language::Own {
    functions: &[],
    structs: &[],
    types: &[],
};

#[test]
fn one_declaration_written_in_two_files_is_one() {
    let dir = scratch(
        "same",
        &[
            ("a.orla", "struct Surface { depth: Float; id: Int; }\n"),
            // The same fields, written further down its file.
            (
                "b.orla",
                "// another library\n\n\nstruct Surface { depth: Float; id: Int; }\n",
            ),
        ],
    );
    let (structs, _, _) = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";\nuse \"b.orla\";"),
        NOTHING,
    )
    .unwrap();
    assert_eq!(structs.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two libraries each declare `Surface` their own way: each keeps its
/// own, and what uses both sees each under its document's name.
#[test]
fn two_declarations_of_one_name_are_each_their_documents() {
    let dir = scratch(
        "different",
        &[
            (
                "a.orla",
                "struct Surface { depth: Float; id: Int; }\ntype Surfaces = Rows<Surface>;\nfn deep(s: Surface) -> Float { return s.depth; }\n",
            ),
            ("b.orla", "struct Surface { depth: Float; }\n"),
        ],
    );
    let (structs, types, functions) = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";\nuse \"b.orla\";"),
        NOTHING,
    )
    .unwrap();
    let a = format!(
        "Surface@{}",
        dir.canonicalize().unwrap().join("a.orla").display()
    );
    let b = format!(
        "Surface@{}",
        dir.canonicalize().unwrap().join("b.orla").display()
    );
    let names: Vec<&str> = structs.iter().map(|d| d.name.value.as_str()).collect();
    assert_eq!(names, [a.as_str(), b.as_str()]);
    assert_eq!(types[0].target.value, format!("Rows<{a}>"));
    assert_eq!(functions[0].args[0].1.value, a);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A document that uses both and names `Surface` is refused, naming
/// both; one that declares `Surface` itself has its own.
#[test]
fn a_name_two_used_documents_declare_is_refused_where_named() {
    let dir = scratch(
        "named",
        &[
            ("a.orla", "struct Surface { depth: Float; id: Int; }\n"),
            ("b.orla", "struct Surface { depth: Float; }\n"),
            (
                "c.orla",
                "use \"a.orla\";\nuse \"b.orla\";\nfn flat(s: Surface) -> Float { return s.depth; }\n",
            ),
        ],
    );
    let error = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"c.orla\";"),
        NOTHING,
    )
    .unwrap_err();
    assert!(
        error.contains("`Surface` is declared differently by"),
        "{error}"
    );
    assert!(
        error.contains("a.orla") && error.contains("b.orla"),
        "{error}"
    );

    let own = orla_language::parse_document("struct Surface { id: Int; }\n").unwrap();
    let (structs, _, _) = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";"),
        orla_language::Own::of(&own),
    )
    .unwrap();
    assert_eq!(
        structs[0].name.value,
        format!(
            "Surface@{}",
            dir.canonicalize().unwrap().join("a.orla").display()
        )
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file added to a directory a scene uses is published by the next
/// sync, and one taken away is retired.
#[test]
fn a_used_directory_gains_and_loses_files_between_syncs() {
    let dir = scratch(
        "gains",
        &[(
            "scene.osn",
            "use \"kernels/\";\nnode x: lib.a@1 { domain = 4; }\noutput result <- x.result;\n",
        )],
    );
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    let kernel = |name: &str| {
        format!(
            "kernel lib.{name}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 1.0; }} }}\n"
        )
    };
    std::fs::write(dir.join("kernels/a.orla"), kernel("a")).unwrap();
    let runtime =
        orla_session::Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let mut session = orla_session::Session::open_in(
        std::sync::Arc::new(orla_session::Disk::default()),
        &dir.join("scene.osn"),
        None,
        runtime,
    );
    let published = |session: &orla_session::Session, name: &str| {
        session
            .runtime()
            .kernel(&format!("lib.{name}@1").parse().unwrap())
            .is_some()
    };
    let events = session.sync();
    assert!(published(&session, "a"), "{events:?}");
    assert!(!published(&session, "b"));
    // Idle syncs in between keep what they reached.
    session.sync();
    assert!(published(&session, "a"));
    std::fs::write(dir.join("kernels/b.orla"), kernel("b")).unwrap();
    let events = session.sync();
    assert!(published(&session, "b"), "{events:?}");
    std::fs::remove_file(dir.join("kernels/b.orla")).unwrap();
    let events = session.sync();
    assert!(!published(&session, "b"), "{events:?}");
    assert!(published(&session, "a"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Files handed in, counting the stamps asked of each path.
struct Counting {
    files: orla_session::Memory,
    stamps: std::sync::Mutex<std::collections::BTreeMap<PathBuf, usize>>,
}

impl orla_session::Store for Counting {
    fn read(&self, path: &std::path::Path) -> std::io::Result<Vec<u8>> {
        self.files.read(path)
    }
    fn write(&self, path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        self.files.write(path, bytes)
    }
    fn kind(&self, path: &std::path::Path) -> Option<orla_session::store::Kind> {
        self.files.kind(path)
    }
    fn list(&self, dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
        self.files.list(dir)
    }
    fn stamp(&self, path: &std::path::Path) -> Option<orla_session::store::Stamp> {
        *self
            .stamps
            .lock()
            .unwrap()
            .entry(path.to_path_buf())
            .or_default() += 1;
        self.files.stamp(path)
    }
    fn canonical(&self, path: &std::path::Path) -> std::io::Result<PathBuf> {
        self.files.canonical(path)
    }
    fn whole(&self, path: &std::path::Path) -> PathBuf {
        self.files.whole(path)
    }
}

/// An idle sync asks each document's stamp once, however many documents
/// use it: libraries each using every one before them cost as many stamps
/// as there are libraries, not as there are uses.
#[test]
fn an_idle_sync_stamps_each_document_once() {
    let files = orla_session::Memory::default();
    let put =
        |name: &str, text: String| files.put(&PathBuf::from("/s").join(name), text.into_bytes());
    for i in 0..6 {
        let uses: String = (0..i).map(|j| format!("use \"l{j}.orla\";\n")).collect();
        put(
            &format!("lib/l{i}.orla"),
            format!(
                "{uses}kernel lib.k{i}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {i}.0; }} }}\n"
            ),
        );
    }
    put(
        "scene.osn",
        "use \"lib/\";\nnode n: lib.k5@1 { domain = 1; }\noutput result <- n.result;\n".into(),
    );
    let store = std::sync::Arc::new(Counting {
        files,
        stamps: Default::default(),
    });
    let runtime =
        orla_session::Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let mut session = orla_session::Session::open_in(
        std::sync::Arc::clone(&store) as std::sync::Arc<dyn orla_session::Store>,
        std::path::Path::new("/s/scene.osn"),
        None,
        runtime,
    );
    session.sync();
    store.stamps.lock().unwrap().clear();
    let events = session.sync();
    assert!(events.is_empty(), "{events:?}");
    let stamps = store.stamps.lock().unwrap();
    // At most once for the libraries, and once more for the one the scene
    // itself uses.
    assert!(stamps.values().all(|count| *count <= 2), "{stamps:?}");
}
