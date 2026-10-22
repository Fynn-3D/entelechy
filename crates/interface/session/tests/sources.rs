//! Declarations shared through `use`: one declaration reached from two
//! files is one, wherever written; two different ones of a name are each
//! their own document's, and a document reaching both refuses the name.

use std::path::PathBuf;

use entelechy_language::Declarations;
use entelechy_session::declarations_for;

fn scratch(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-sources-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, text) in files {
        std::fs::write(dir.join(file), text).unwrap();
    }
    dir
}

fn uses(text: &str) -> Vec<entelechy_language::Use> {
    entelechy_language::parse_document(text).unwrap().uses
}

const NOTHING: entelechy_language::Own<'static> = entelechy_language::Own {
    functions: &[],
    structs: &[],
    types: &[],
    document: None,
};

#[test]
fn one_declaration_written_in_two_files_is_one() {
    let dir = scratch(
        "same",
        &[
            ("a.ent", "struct Surface { depth: Float; id: Int; }\n"),
            // The same fields, written further down its file.
            (
                "b.ent",
                "// another library\n\n\nstruct Surface { depth: Float; id: Int; }\n",
            ),
        ],
    );
    let Declarations { structs, .. } = declarations_for(
        &entelechy_session::Disk::default(),
        &dir,
        &uses("use \"a.ent\";\nuse \"b.ent\";"),
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
                "a.ent",
                "struct Surface { depth: Float; id: Int; }\ntype Surfaces = Rows<Surface>;\nfn deep(s: Surface) -> Float { return s.depth; }\n",
            ),
            ("b.ent", "struct Surface { depth: Float; }\n"),
        ],
    );
    let Declarations {
        structs,
        types,
        functions,
        ..
    } = declarations_for(
        &entelechy_session::Disk::default(),
        &dir,
        &uses("use \"a.ent\";\nuse \"b.ent\";"),
        NOTHING,
    )
    .unwrap();
    let a = format!(
        "Surface@{}",
        dir.canonicalize().unwrap().join("a.ent").display()
    );
    let b = format!(
        "Surface@{}",
        dir.canonicalize().unwrap().join("b.ent").display()
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
            ("a.ent", "struct Surface { depth: Float; id: Int; }\n"),
            ("b.ent", "struct Surface { depth: Float; }\n"),
            (
                "c.ent",
                "use \"a.ent\";\nuse \"b.ent\";\nfn flat(s: Surface) -> Float { return s.depth; }\n",
            ),
        ],
    );
    let error = declarations_for(
        &entelechy_session::Disk::default(),
        &dir,
        &uses("use \"c.ent\";"),
        NOTHING,
    )
    .unwrap_err();
    assert!(
        error.contains("`Surface` is declared differently by"),
        "{error}"
    );
    assert!(
        error.contains("a.ent") && error.contains("b.ent"),
        "{error}"
    );

    let own = entelechy_language::parse_document("struct Surface { id: Int; }\n").unwrap();
    let Declarations { structs, .. } = declarations_for(
        &entelechy_session::Disk::default(),
        &dir,
        &uses("use \"a.ent\";"),
        entelechy_language::Own::of(&own),
    )
    .unwrap();
    assert_eq!(
        structs[0].name.value,
        format!(
            "Surface@{}",
            dir.canonicalize().unwrap().join("a.ent").display()
        )
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A declaration the using document makes alike itself is its own and
/// the used one's: it is not gathered again, and the used document's
/// functions name it as written.
#[test]
fn a_declaration_made_alike_by_the_user_is_one() {
    let dir = scratch(
        "alike",
        &[(
            "a.ent",
            "type Cells = Volume<Int>;\nfn corner(v: Cells) -> Int { return at(v, ivec3(0, 0, 0)); }\n",
        )],
    );
    let own = entelechy_language::parse_document("type Cells = Volume<Int>;\n").unwrap();
    let Declarations {
        types, functions, ..
    } = declarations_for(
        &entelechy_session::Disk::default(),
        &dir,
        &uses("use \"a.ent\";"),
        entelechy_language::Own::of(&own),
    )
    .unwrap();
    assert!(types.is_empty(), "{types:?}");
    assert_eq!(functions[0].args[0].1.value, "Cells");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file added to a directory a scene uses is published by the next
/// sync, and one taken away is retired.
#[test]
fn a_used_directory_gains_and_loses_files_between_syncs() {
    let dir = scratch(
        "gains",
        &[(
            "scene.esn",
            "use \"kernels/\";\nnode x: lib.a@1 { domain = 4; }\noutput result <- x.result;\n",
        )],
    );
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    let kernel = |name: &str| {
        format!(
            "kernel lib.{name}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 1.0; }} }}\n"
        )
    };
    std::fs::write(dir.join("kernels/a.ent"), kernel("a")).unwrap();
    let runtime =
        entelechy_session::Session::runtime_for("cpu", entelechy_session::Limits::default())
            .unwrap();
    let mut session = entelechy_session::Session::open_in(
        std::sync::Arc::new(entelechy_session::Disk::default()),
        &dir.join("scene.esn"),
        None,
        runtime,
    );
    let published = |session: &entelechy_session::Session, name: &str| {
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
    std::fs::write(dir.join("kernels/b.ent"), kernel("b")).unwrap();
    let events = session.sync();
    assert!(published(&session, "b"), "{events:?}");
    std::fs::remove_file(dir.join("kernels/b.ent")).unwrap();
    let events = session.sync();
    assert!(!published(&session, "b"), "{events:?}");
    assert!(published(&session, "a"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Files handed in, counting the stamps asked of each path.
struct Counting {
    files: entelechy_session::Memory,
    stamps: std::sync::Mutex<std::collections::BTreeMap<PathBuf, usize>>,
}

impl entelechy_session::Store for Counting {
    fn read(&self, path: &std::path::Path) -> std::io::Result<Vec<u8>> {
        self.files.read(path)
    }
    fn write(&self, path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        self.files.write(path, bytes)
    }
    fn remove(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.files.remove(path)
    }
    fn rename(&self, from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
        self.files.rename(from, to)
    }
    fn remove_empty(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.files.remove_empty(dir)
    }
    fn kind(&self, path: &std::path::Path) -> Option<entelechy_session::store::Kind> {
        self.files.kind(path)
    }
    fn list(&self, dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
        self.files.list(dir)
    }
    fn stamp(&self, path: &std::path::Path) -> Option<entelechy_session::store::Stamp> {
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
    let files = entelechy_session::Memory::default();
    let put =
        |name: &str, text: String| files.put(&PathBuf::from("/s").join(name), text.into_bytes());
    for i in 0..6 {
        let uses: String = (0..i).map(|j| format!("use \"l{j}.ent\";\n")).collect();
        put(
            &format!("lib/l{i}.ent"),
            format!(
                "{uses}kernel lib.k{i}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {i}.0; }} }}\n"
            ),
        );
    }
    put(
        "scene.esn",
        "use \"lib/\";\nnode n: lib.k5@1 { domain = 1; }\noutput result <- n.result;\n".into(),
    );
    let store = std::sync::Arc::new(Counting {
        files,
        stamps: Default::default(),
    });
    let runtime =
        entelechy_session::Session::runtime_for("cpu", entelechy_session::Limits::default())
            .unwrap();
    let mut session = entelechy_session::Session::open_in(
        std::sync::Arc::clone(&store) as std::sync::Arc<dyn entelechy_session::Store>,
        std::path::Path::new("/s/scene.esn"),
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

/// A scene's struct may hold an enum a used document declares beside the
/// scene's own params: a param's default is read with the structs in
/// reach, their fields through the aliases in reach.
#[test]
fn a_scene_struct_holds_a_used_enum_beside_its_params() {
    let dir = scratch(
        "enum-beside-param",
        &[
            ("lib.ent", "type Mode = enum { Idle, Walk };\n"),
            (
                "scene.esn",
                "use \"lib.ent\";\nparam speed: Float = 1.0;\nstruct Pose { mode: Mode; modes: [Mode; 2]; }\nnode n: kernel { requires { field.write; } output result: Rows<Pose>; domain = 1; compute { return Pose { mode: Mode.Walk, modes: [Mode.Idle, Mode.Walk] }; } }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    let (_, words) = session
        .element_field(
            &entelechy_graph::NodeId("n".into()),
            "result",
            0,
            &["mode".into()],
        )
        .unwrap();
    assert_eq!(words, vec![1]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Kernel's own function, struct and type are the innermost scope:
/// its code reaches its own, and a used library's code keeps reaching
/// the library's own of the same names.
#[test]
fn a_kernels_own_declarations_shadow_a_used_librarys() {
    let dir = scratch(
        "kernel-own",
        &[
            (
                "lib.ent",
                "struct Pair { a: Int; b: Int; }\ntype Mode = enum { Idle, Walk };\nfn over(a: Int, b: Int) -> Int { return a + b; }\nfn total(x: Int) -> Int { let p = Pair { a: x, b: x * 2 }; return over(p.a, p.b) + Mode.Walk; }\n",
            ),
            (
                "scene.esn",
                "use \"lib.ent\";\nnode n: kernel { requires { field.write; } output result: Rows<Int>; domain = 1;\n    struct Pair { left: Int; right: Int; }\n    type Mode = enum { Off = 10, On = 20 };\n    fn over(a: Int, b: Int) -> Int { return a * b; }\n    compute { let p = Pair { left: 1, right: Mode.On }; return over(3, 4) + p.left + p.right + total(5); } }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    // Its own: 3 * 4, and 1 + 20; the library's: 5 + 10, and 1.
    let (_, words) = session
        .element_field(&entelechy_graph::NodeId("n".into()), "result", 0, &[])
        .unwrap();
    assert_eq!(words, vec![12 + 21 + 16]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A fault in a declaration a Kernel inherits from a used document, a
/// struct's field, a type's shape or a function's signature, is placed
/// in the used document: its line and column are that document's.
#[test]
fn a_fault_in_an_inherited_declaration_is_placed_in_its_document() {
    let declarations = [
        "struct Pair { a: Nope; }\n",
        "type Cells = Rows<Nope>;\n",
        "fn twice(x: Nope) -> Int { return 2; }\n",
    ];
    for (k, declaration) in declarations.iter().enumerate() {
        let dir = scratch(
            &format!("inherited-fault-{k}"),
            &[
                ("lib.ent", declaration),
                (
                    "k.ent",
                    "use \"lib.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }\n",
                ),
                (
                    "scene.esn",
                    "use \"k.ent\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
                ),
            ],
        );
        let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
        let events = session.sync();
        let faults: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                entelechy_session::Event::Rejected { path, error }
                    if error.to_string().contains("Nope") =>
                {
                    Some(path.clone())
                }
                _ => None,
            })
            .collect();
        assert!(!faults.is_empty(), "{declaration}{events:?}");
        for path in faults {
            assert_eq!(
                path.file_name().unwrap(),
                "lib.ent",
                "{declaration}{events:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A fault in a Kernel's own declaration, of a name a used document
/// declares too, is placed in the Kernel's document; one in the used
/// document's, which the Kernel's own shadows, in the used document.
#[test]
fn a_fault_in_a_shadowing_declaration_is_placed_where_it_is_written() {
    let kernel = |own: &str| {
        format!(
            "use \"lib.ent\";\nkernel t.k@1 {{ requires {{ field.write; }} output result: Rows<Int>;\n    {own}\n    compute {{ return 1; }} }}\n"
        )
    };
    let cases = [
        (
            "struct Pair { a: Int; }\n",
            kernel("struct Pair { b: Nope; }"),
            "k.ent",
        ),
        (
            "struct Pair { a: Nope; }\n",
            kernel("struct Pair { b: Int; }"),
            "lib.ent",
        ),
        (
            "fn twice(x: Int) -> Int { return 2; }\n",
            kernel("fn twice(x: Nope) -> Int { return 2; }"),
            "k.ent",
        ),
        (
            "fn twice(x: Nope) -> Int { return 2; }\n",
            kernel("fn twice(x: Int) -> Int { return 2; }"),
            "lib.ent",
        ),
    ];
    for (k, (library, kernel, written)) in cases.iter().enumerate() {
        let dir = scratch(
            &format!("shadowing-fault-{k}"),
            &[
                ("lib.ent", library),
                ("k.ent", kernel),
                (
                    "scene.esn",
                    "use \"k.ent\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
                ),
            ],
        );
        let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
        let events = session.sync();
        let faults: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                entelechy_session::Event::Rejected { path, error }
                    if error.to_string().contains("Nope") =>
                {
                    Some(path.clone())
                }
                _ => None,
            })
            .collect();
        assert!(!faults.is_empty(), "{kernel}{events:?}");
        for path in faults {
            assert_eq!(
                path.file_name().unwrap(),
                *written,
                "{library}{kernel}{events:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A package names a package it uses by its name, and finds it beside
/// itself wherever the two are kept together: its Kernels' `use` and
/// its presets' `graph` alike.
#[test]
fn a_package_finds_the_packages_it_names_beside_it() {
    let manifest =
        |name: &str| format!("package {name}@1 {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n");
    let dir = scratch("packages-beside", &[]);
    for sub in [
        "kept/a.entpkg/kernels",
        "kept/a.entpkg/graphs",
        "kept/b.entpkg/kernels",
        "kept/b.entpkg/graphs",
        "scene",
    ] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    for (file, text) in [
        ("kept/a.entpkg/package.esn", manifest("a")),
        (
            "kept/a.entpkg/kernels/a.ent",
            "fn a_one() -> Int { return 1; }\nkernel a.twice@1 { requires { field.read; field.write; } input counts: Rows<Int>; output result: Rows<Int>; compute { return sample(counts) * 2; } }\n".into(),
        ),
        (
            "kept/a.entpkg/graphs/twice.esn",
            "use \"../kernels/a.ent\";\ninput counts: Rows<Int>;\nnode once: a.twice@1 { counts <- counts; }\noutput result <- once.result;\n".into(),
        ),
        ("kept/b.entpkg/package.esn", manifest("b")),
        (
            "kept/b.entpkg/kernels/b.ent",
            "use \"a.entpkg\";\nkernel b.start@1 { requires { field.write; } output result: Rows<Int>; compute { return element() + a_one(); } }\n".into(),
        ),
        (
            "kept/b.entpkg/graphs/start.esn",
            "use \"../kernels/b.ent\";\nnode s: b.start@1 { domain = 3; }\nnode t: graph \"a.entpkg/graphs/twice.esn\" { counts <- s.result; }\noutput result <- t.result;\n".into(),
        ),
        (
            "scene/scene.esn",
            "use \"../kept/b.entpkg\";\nnode p: graph \"../kept/b.entpkg/graphs/start.esn\" { }\noutput shown <- p.result;\n".into(),
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let mut session = entelechy_session::Session::open(&dir.join("scene/scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    let (_, words) = session
        .element_field(&entelechy_graph::NodeId("p".into()), "result", 2, &[])
        .unwrap();
    assert_eq!(words, vec![6]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A name is one package: a directory is named for the package its
/// manifest declares, and a scene reaching two packages of one name, as
/// two editions of it, is refused, naming both.
#[test]
fn a_name_is_one_package() {
    let package = |dir: &std::path::Path, at: &str, id: &str| {
        let root = dir.join(at);
        std::fs::create_dir_all(root.join("kernels")).unwrap();
        std::fs::write(
            root.join("package.esn"),
            format!("package {id} {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n"),
        )
        .unwrap();
        std::fs::write(
            root.join("kernels/k.ent"),
            "fn one() -> Int { return 1; }\n",
        )
        .unwrap();
    };
    let scene = "node s: kernel { requires { field.write; } output result: Rows<Int>; domain = 1; compute { return one(); } }\noutput result <- s.result;\n";
    let refusal = |dir: &std::path::Path| {
        let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
        let events = session.sync();
        (session.graph().is_some(), format!("{events:?}"))
    };

    let dir = scratch("one-package-editions", &[]);
    package(&dir, "one/p.entpkg", "p@1");
    package(&dir, "two/p.entpkg", "p@2");
    std::fs::write(
        dir.join("scene.esn"),
        format!("use \"one/p.entpkg\";\nuse \"two/p.entpkg\";\n{scene}"),
    )
    .unwrap();
    let (_, said) = refusal(&dir);
    assert!(
        said.contains("two packages named `p` are reached")
            && said.contains("(p@1)")
            && said.contains("(p@2)"),
        "{said}"
    );
    // One package reached by two spellings is one.
    std::fs::write(
        dir.join("scene.esn"),
        format!("use \"one/p.entpkg\";\nuse \"one/../one/p.entpkg\";\n{scene}"),
    )
    .unwrap();
    let (loaded, said) = refusal(&dir);
    assert!(loaded, "{said}");
    let _ = std::fs::remove_dir_all(&dir);

    let dir = scratch("one-package-misnamed", &[]);
    package(&dir, "q.entpkg", "p@1");
    std::fs::write(dir.join("scene.esn"), format!("use \"q.entpkg\";\n{scene}")).unwrap();
    let (_, said) = refusal(&dir);
    assert!(
        said.contains("the directory `q.entpkg` declares the package `p@1`"),
        "{said}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A function of a builtin's name stands for it in its own document
/// only: a document using it keeps the builtin, and the library's own
/// calls keep reaching its function. A call in the declaring document
/// that fits the builtin and not the function says the function hides
/// it, and where it is declared.
#[test]
fn a_function_hides_a_builtin_in_its_own_document_only() {
    let dir = scratch(
        "hiding-builtin",
        &[
            (
                "lib.ent",
                "fn dot(a: Int) -> Int { return a; }\nfn uses_own(a: Int) -> Int { return dot(a) + 1; }\n",
            ),
            (
                "scene.esn",
                "use \"lib.ent\";\nnode n: kernel { requires { field.write; } output result: Rows<Float>; domain = 1; compute { return dot(vec2(2.0, 0.0), vec2(1.5, 0.0)) + float(uses_own(3)); } }\noutput result <- n.result;\n",
            ),
            (
                "own.ent",
                "fn dot(a: Int) -> Int { return a; }\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return dot(vec2(1.0, 0.0), vec2(1.0, 0.0)); } }\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    // The builtin: 2 * 1.5; the library's own: 3 + 1.
    let (_, words) = session
        .element_field(&entelechy_graph::NodeId("n".into()), "result", 0, &[])
        .unwrap();
    assert_eq!(f32::from_bits(words[0]), 7.0);
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("own.ent"),
        false,
    );
    assert!(
        checked.lines.iter().any(|(problem, line)| *problem
            && line.contains("hides the builtin `dot`, declared in own.ent")),
        "{checked:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Kernel's own function and struct are its own, over its document's
/// of the same name, while the document's own code keeps reaching the
/// document's.
#[test]
fn a_kernels_own_declarations_shadow_its_documents() {
    let dir = scratch(
        "kernel-shadows-document",
        &[
            (
                "app.ent",
                "struct S { x: Float; }\nfn f() -> Float { return 1.0; }\nfn g() -> Float { return f() + S { x: 4.0 }.x; }\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; struct S { y: Float; } fn f() -> Float { return 2.0; } compute { return f() * 10.0 + g() + S { y: 100.0 }.y; } }\n",
            ),
            (
                "scene.esn",
                "use \"app.ent\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    // The Kernel's own: 2 * 10 and 100; the document's, through `g`: 1 + 4.
    let (_, words) = session
        .element_field(&entelechy_graph::NodeId("n".into()), "result", 0, &[])
        .unwrap();
    assert_eq!(f32::from_bits(words[0]), 125.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Kernel id two documents the scene publishes both write is refused
/// by the check from the second, as a session refuses it.
#[test]
fn check_refuses_a_kernel_id_two_documents_publish() {
    let kernel = "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }\n";
    let dir = scratch(
        "kernel-id-twice",
        &[
            ("a.ent", kernel),
            ("b.ent", kernel),
            (
                "scene.esn",
                "use \"a.ent\";\nuse \"b.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
            ),
        ],
    );
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert!(
        checked.lines.iter().any(|(problem, line)| *problem
            && line.contains("b.ent: t.k@1 is already published by")
            && line.contains("a.ent; kernel ids are unique")),
        "{checked:?}"
    );
    // One document reached by two paths is one.
    std::fs::write(
        dir.join("scene.esn"),
        "use \"a.ent\";\nuse \"./a.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A node bound to a Kernel written in a document the scene reaches only
/// through another is said, once, with the `use` that publishes it; the
/// check says the same.
#[test]
fn a_kernel_reached_only_through_another_document_is_said_with_its_use() {
    let dir = scratch("unpublished", &[]);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for (file, text) in [
        (
            "kernels/atlas.ent",
            "use \"palettes.ent\";\nkernel t.atlas@1 { requires { field.write; } output result: Rows<Int>; compute { return twice(1); } }\n",
        ),
        (
            "kernels/palettes.ent",
            "fn twice(x: Int) -> Int { return 2 * x; }\nkernel t.greens@1 { requires { field.write; } output result: Rows<Int>; compute { return 4; } }\n",
        ),
        (
            "scene.esn",
            "use \"kernels/atlas.ent\";\nnode a: t.atlas@1 { domain = 1; }\nnode g: t.greens@1 { domain = 1; }\noutput result <- g.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let wants = "`t.greens@1` is written in kernels/palettes.ent, which the scene reaches only through kernels/atlas.ent";
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let first = format!("{:?}{:?}", session.sync(), session.sync());
    assert_eq!(first.matches(wants).count(), 1, "{first}");
    assert!(
        first.contains("wants `use \\\"kernels/palettes.ent\\\";`"),
        "{first}"
    );
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert!(
        checked
            .lines
            .iter()
            .any(|(problem, line)| *problem && line.contains(wants)),
        "{checked:?}"
    );
    // Used itself, it is published, and nothing is said.
    std::fs::write(dir.join("scene.esn"), "use \"kernels/atlas.ent\";\nuse \"kernels/palettes.ent\";\nnode a: t.atlas@1 { domain = 1; }\nnode g: t.greens@1 { domain = 1; }\noutput result <- g.result;\n").unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");

    // A Kernel written in a package the scene reaches through a document
    // wants the package used whole, never a file inside it.
    std::fs::create_dir_all(dir.join("lib/tones.entpkg/kernels")).unwrap();
    for (file, text) in [
        (
            "lib/tones.entpkg/package.esn",
            "package tones@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
        ),
        (
            "lib/tones.entpkg/kernels/tones.ent",
            "kernel tones.start@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\n",
        ),
        (
            "kernels/music.ent",
            "use \"../lib/tones.entpkg\";\nkernel t.music@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }\n",
        ),
        (
            "scene.esn",
            "use \"kernels/music.ent\";\nnode m: t.music@1 { domain = 1; }\nnode s: tones.start@1 { domain = 1; }\noutput result <- s.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert!(
        checked.lines.iter().any(|(problem, line)| *problem
            && line.contains("which the scene reaches only through kernels/music.ent")
            && line.contains("wants `use \"lib/tones.entpkg\";`")),
        "{checked:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `entelechy check` compiles the Kernels a scene publishes, as a session
/// does: a Kernel of a document the scene reaches only through another
/// is not the scene's, while that document's functions, which Kernels
/// inherit, are checked.
#[test]
fn check_compiles_the_kernels_a_scene_publishes() {
    let dir = scratch("check-published", &[]);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for (file, text) in [
        (
            "kernels/atlas.ent",
            "use \"deep.ent\";\nkernel t.atlas@1 { requires { field.write; } output result: Rows<Int>; compute { return twice(1); } }\n",
        ),
        (
            "kernels/deep.ent",
            "fn twice(x: Int) -> Int { return 2 * x; }\nkernel t.deep@1 { requires { field.write; } output result: Rows<Int>; compute { return nowhere; } }\n",
        ),
        (
            "scene.esn",
            "use \"kernels/atlas.ent\";\nnode a: t.atlas@1 { domain = 1; }\noutput result <- a.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
    std::fs::write(dir.join("kernels/deep.ent"), "fn twice(x: Int) -> Int { return 2 * nothing; }\nkernel t.deep@1 { requires { field.write; } output result: Rows<Int>; compute { return nowhere; } }\n").unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert!(
        checked
            .lines
            .iter()
            .any(|(problem, line)| *problem && line.contains("`nothing`")),
        "{checked:?}"
    );
    assert!(
        !checked
            .lines
            .iter()
            .any(|(_, line)| line.contains("`nowhere`")),
        "{checked:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A fault says the names as they were written: an inherited function the
/// Kernel's own of its name moves out of the way is still called by its
/// own name in what a fault in it says, and the fault is placed where it
/// is written.
#[test]
fn a_fault_says_names_as_they_were_written() {
    let dir = scratch(
        "written-names",
        &[
            (
                "lib.ent",
                "fn twice(x: Int) -> Int { return x * 2; }\nfn four() -> Int { return twice(1, 2); }\n",
            ),
            (
                "k.ent",
                "use \"lib.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Int>;\n    fn twice(x: Int) -> Int { return x + x; }\n    compute { return twice(1) + four(); } }\n",
            ),
            (
                "scene.esn",
                "use \"k.ent\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let faults: Vec<_> = session
        .sync()
        .into_iter()
        .filter_map(|event| match event {
            entelechy_session::Event::Rejected { path, error }
                if error.to_string().contains("argument") =>
            {
                Some((path, error.to_string()))
            }
            _ => None,
        })
        .collect();
    assert!(!faults.is_empty());
    for (path, said) in faults {
        assert_eq!(path.file_name().unwrap(), "lib.ent", "{said}");
        assert!(
            said.contains("`twice` takes 1 argument(s), found 2") && !said.contains('@'),
            "{said}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A document sees what it uses itself: a name declared only by a
/// document another one uses is not its, and is said with the `use`
/// that brings it; the document that does use it still calls it.
#[test]
fn a_document_sees_what_it_uses_itself() {
    let dir = scratch(
        "sees-what-it-uses",
        &[
            (
                "c.ent",
                "struct Deep { v: Float; }\nfn deep() -> Deep { return Deep { v: 2.0 }; }\n",
            ),
            (
                "b.ent",
                "use \"c.ent\";\nfn mid() -> Float { return deep().v + 1.0; }\n",
            ),
            (
                "a.ent",
                "use \"b.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return deep().v; } }\n",
            ),
            (
                "scene.esn",
                "use \"a.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
            ),
        ],
    );
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let said = format!("{checked:?}");
    assert!(
        said.contains(
            "`deep` is declared in c.ent, which this document reaches only through another's `use`"
        ) && said.contains("use \\\"c.ent\\\";"),
        "{said}"
    );
    // Through a function of the document it uses, it is reached.
    std::fs::write(
        dir.join("a.ent"),
        "use \"b.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return mid(); } }\n",
    )
    .unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A function taking a Field is compiled where a Kernel binds it, and
/// still sees what its own document uses alone: a function or struct of
/// a document it does not use is refused at its call, though the
/// Kernel's document uses that document itself.
#[test]
fn a_function_taking_a_field_sees_what_its_document_uses() {
    let kernel = "use \"mid.ent\";\nuse \"lib.ent\";\nkernel t.k@1 { requires { field.read; field.write; } input xs: Rows<Float>; output result: Rows<Float>; compute { return total(xs); } }\nkernel t.s@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }\n";
    let dir = scratch(
        "field-function-sees",
        &[
            (
                "lib.ent",
                "struct Deep { v: Float; }\nfn deep() -> Float { return 2.0; }\n",
            ),
            (
                "mid.ent",
                "fn total(xs: Rows<Float>) -> Float { return at(xs, 0) + deep(); }\n",
            ),
            ("a.ent", kernel),
            (
                "scene.esn",
                "use \"a.ent\";\nnode s: t.s@1 { domain = 1; }\nnode k: t.k@1 { xs <- s.result; domain = 1; }\noutput result <- k.result;\n",
            ),
        ],
    );
    let check = || {
        entelechy_session::check_document(
            &entelechy_session::Disk::default(),
            &dir.join("scene.esn"),
            false,
        )
    };
    let said = format!("{:?}", check());
    assert!(
        said.contains("mid.ent:1:")
            && said.contains("`deep` is not a builtin")
            && said.contains("use \\\"lib.ent\\\";"),
        "{said}"
    );
    std::fs::write(
        dir.join("mid.ent"),
        "fn total(xs: Rows<Float>) -> Float { return Deep { v: at(xs, 0) }.v; }\n",
    )
    .unwrap();
    let said = format!("{:?}", check());
    assert!(
        said.contains("`Deep` is not a struct") && !said.contains("Deep@"),
        "{said}"
    );
    // Used itself, both are seen.
    std::fs::write(
        dir.join("mid.ent"),
        "use \"lib.ent\";\nfn total(xs: Rows<Float>) -> Float { return Deep { v: at(xs, 0) }.v + deep(); }\n",
    )
    .unwrap();
    let checked = check();
    assert_eq!(checked.problems(), 0, "{checked:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A scene's param reads its default over the structs its type reaches:
/// one another document declares, faulty, is said with that document and
/// its place there, at the param; a struct the type does not reach says
/// nothing at the param.
#[test]
fn a_param_says_a_faulty_struct_with_its_document() {
    let dir = scratch(
        "param-struct-elsewhere",
        &[
            (
                "lib.ent",
                "struct Fine { x: Float; }\nstruct Bad { x: Float; q: Nope; }\n",
            ),
            (
                "k.ent",
                "kernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }\n",
            ),
            (
                "scene.esn",
                "use \"lib.ent\";\nuse \"k.ent\";\nparam p: Bad = Bad { x: 1.0, q: 2 };\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
            ),
        ],
    );
    let check = || {
        entelechy_session::check_document(
            &entelechy_session::Disk::default(),
            &dir.join("scene.esn"),
            false,
        )
    };
    let said = format!("{:?}", check());
    assert!(
        said.contains("struct `Bad` (")
            && said.contains("lib.ent:2:")
            && said.contains("unknown type `Nope`"),
        "{said}"
    );
    std::fs::write(
        dir.join("scene.esn"),
        "use \"lib.ent\";\nuse \"k.ent\";\nparam p: Fine = Fine { x: 1.0 };\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let said = format!("{:?}", check());
    assert!(!said.contains("Nope"), "{said}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A check says every fault a document's functions hold, each once, in
/// one run: a fault in a function others call is said where it is.
#[test]
fn a_check_says_every_fault_of_a_document() {
    let dir = scratch(
        "every-fault",
        &[
            (
                "lib.ent",
                "fn a(x: Float) -> Float { return x + nope1; }\nfn b(x: Float) -> Float { return a(x) * 2.0; }\nfn c(x: Float) -> Float { return x + nope2(x); }\nfn d(x: Int) -> Float { return x.w; }\nfn fine(x: Float) -> Float { return x; }\n",
            ),
            (
                "k.ent",
                "use \"lib.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return fine(1.0); } }\n",
            ),
            (
                "scene.esn",
                "use \"k.ent\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
            ),
        ],
    );
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let problems: Vec<&String> = checked
        .lines
        .iter()
        .filter(|(p, _)| *p)
        .map(|(_, l)| l)
        .collect();
    assert_eq!(problems.len(), 3, "{problems:?}");
    for (place, says) in [
        ("lib.ent:1:", "nope1"),
        ("lib.ent:3:", "nope2"),
        ("lib.ent:4:", "`.w`"),
    ] {
        assert!(
            problems
                .iter()
                .any(|l| l.contains(place) && l.contains(says)),
            "{place} {says}: {problems:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A library declares a name a document it uses declares too, as a
/// game's widgets beside a package's: a scene using the library sees the
/// library's alone, so nothing clashes; the library's own functions call
/// what they see.
#[test]
fn names_a_used_library_uses_are_not_the_documents() {
    let dir = scratch(
        "not-the-documents",
        &[
            (
                "package.ent",
                "fn shade() -> Float { return 1.0; }\nfn lit() -> Float { return shade() + 2.0; }\n",
            ),
            (
                "widgets.ent",
                "use \"package.ent\";\nfn shade() -> Float { return 10.0; }\nfn both() -> Float { return shade() + lit(); }\n",
            ),
            (
                "scene.esn",
                "use \"widgets.ent\";\nnode k: kernel { requires { field.write; } output result: Rows<Float>; domain = 1; compute { return shade() * 100.0 + both(); } }\noutput result <- k.result;\n",
            ),
        ],
    );
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    let handle = session.output(Some("result")).unwrap();
    let value = f32::from_bits(session.field(handle).unwrap().values()[0]);
    // The scene's shade is the widgets' (10); both() is its 10 and the
    // package's lit(), the package's shade (1) and 2.
    assert_eq!(value, 1013.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A name declared in a package the store finds by name is said with
/// the package's name to use, not a path to it.
#[test]
fn a_name_in_a_package_found_by_name_is_said_with_the_packages_name() {
    let dir = scratch("package-by-name", &[]);
    std::fs::create_dir_all(dir.join("packages/deep.entpkg/kernels")).unwrap();
    for (file, text) in [
        (
            "packages/deep.entpkg/package.esn",
            "package deep@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
        ),
        (
            "packages/deep.entpkg/kernels/deep.ent",
            "fn deep() -> Float { return 2.0; }\n",
        ),
        (
            "b.ent",
            "use \"deep.entpkg\";\nfn mid() -> Float { return deep(); }\n",
        ),
        (
            "a.ent",
            "use \"b.ent\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return deep(); } }\n",
        ),
        (
            "scene.esn",
            "use \"a.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let store = entelechy_session::Disk::with_packages(vec![dir.join("packages")]);
    let checked = entelechy_session::check_document(&store, &dir.join("scene.esn"), false);
    let said = format!("{checked:?}");
    assert!(said.contains("use \\\"deep.entpkg\\\";"), "{said}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package no directory holds is said by its name, with where it was
/// looked for and how a directory joins the search.
#[test]
fn a_package_not_found_is_said_with_where_it_was_looked_for() {
    let dir = scratch("package-not-found", &[]);
    std::fs::create_dir_all(dir.join("packages")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"absent.entpkg\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let store = entelechy_session::Disk::with_packages(vec![dir.join("packages")]);
    let checked = entelechy_session::check_document(&store, &dir.join("scene.esn"), false);
    let said = format!("{checked:?}");
    assert!(
        said.contains("no package `absent.entpkg` is found"),
        "{said}"
    );
    assert!(
        said.contains(&format!(
            "nor in the package directories `{}`",
            dir.join("packages").display()
        )),
        "{said}"
    );
    assert!(said.contains("to `ENTELECHY_PACKAGES`"), "{said}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two packages declaring one name: a document using both names each by
/// its package, `cast_fx::clear` and `ui_panel::clear`, a struct as a
/// function; each package keeps calling its own, and names the other's by
/// its package; a document's or Kernel's own declaration of a name does
/// not hide the one a qualified name reaches; a qualified name the
/// document's uses do not reach is refused, saying why.
#[test]
fn a_name_qualified_by_its_package_reaches_that_packages() {
    let manifest =
        |name: &str| format!("package {name}@1 {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n");
    let dir = scratch("qualified", &[]);
    for sub in ["cast_fx.entpkg/kernels", "ui_panel.entpkg/kernels"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    let scene = |body: &str| {
        format!(
            "use \"cast_fx.entpkg\";\nuse \"ui_panel.entpkg\";\nfn made() -> cast_fx::Mark {{ return cast_fx::Mark {{ a: 7 }}; }}\nfn cast_twice() -> Int {{ return 4; }}\nkernel t.both@1 {{ requires {{ field.write; }} output result: Rows<Int>; struct Mark {{ z: Int; }} fn panel_mark() -> Int {{ return 6; }} compute {{ {body} }} }}\nnode a: t.both@1 {{ domain = 1; }}\noutput result <- a.result;\n"
        )
    };
    for (file, text) in [
        ("cast_fx.entpkg/package.esn", manifest("cast_fx")),
        (
            "cast_fx.entpkg/kernels/cast_fx.ent",
            "struct Mark { a: Int; }\nfn clear() -> Int { return 1; }\nfn cast_twice() -> Int { return clear() * 2; }\n".into(),
        ),
        ("ui_panel.entpkg/package.esn", manifest("ui_panel")),
        (
            "ui_panel.entpkg/kernels/ui_panel.ent",
            "use \"cast_fx.entpkg\";\nstruct Mark { b: Int; c: Int; }\nfn clear() -> Int { return 3; }\nfn panel_mark() -> Mark { return Mark { b: clear() + cast_fx::clear() * 10, c: 0 }; }\n".into(),
        ),
        (
            "scene.esn",
            scene("let m = ui_panel::Mark { b: 5, c: 0 }; let n = made(); let z = Mark { z: 0 }; return cast_fx::clear() * 1000 + ui_panel::clear() * 100 + cast_fx::cast_twice() * 10 + m.b + n.a + ui_panel::panel_mark().b + cast_twice() * 100000 + panel_mark() * 10000 + z.z;"),
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let mut session = entelechy_session::Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    let (_, words) = session
        .element_field(&entelechy_graph::NodeId("a".into()), "result", 0, &[])
        .unwrap();
    assert_eq!(words, vec![1000 + 300 + 20 + 5 + 7 + 13 + 400000 + 60000]);
    let checked = |body: &str| {
        std::fs::write(dir.join("scene.esn"), scene(body)).unwrap();
        entelechy_session::check_document(
            &entelechy_session::Disk::default(),
            &dir.join("scene.esn"),
            false,
        )
    };
    let said = |checked: &entelechy_session::Checked, words: &str| {
        checked
            .lines
            .iter()
            .any(|(problem, line)| *problem && line.contains(words))
    };
    let unqualified = checked("return clear();");
    assert!(said(&unqualified, "cast_fx::clear"), "{unqualified:?}");
    let unused = checked("return hash_noise::fbm2();");
    assert!(said(&unused, "use \"hash_noise.entpkg\""), "{unused:?}");
    let undeclared = checked("return ui_panel::clean();");
    assert!(said(&undeclared, "declares no `clean`"), "{undeclared:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A scene inside a package, opened by its file name from its own
/// directory, finds the packages beside its package by name, as one
/// opened by a longer path does.
#[test]
fn a_scene_opened_where_it_lies_finds_the_packages_beside_its_own() {
    let manifest =
        |name: &str| format!("package {name}@1 {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n");
    let dir = scratch("opened-where-it-lies", &[]);
    for sub in [
        "home.entpkg/kernels",
        "home.entpkg/tests",
        "beside.entpkg/kernels",
    ] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    for (file, text) in [
        ("home.entpkg/package.esn", manifest("home")),
        ("home.entpkg/kernels/home.ent", "fn home_one() -> Int { return 1; }\n".into()),
        ("beside.entpkg/package.esn", manifest("beside")),
        ("beside.entpkg/kernels/beside.ent", "fn beside_two() -> Int { return 2; }\n".into()),
        (
            "home.entpkg/tests/scene.esn",
            "use \"../kernels/\";\nuse \"beside.entpkg\";\nnode n: kernel { requires { field.write; } output result: Rows<Int>; domain = 1; compute { return home_one() + beside_two(); } }\noutput result <- n.result;\n".into(),
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let was = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.join("home.entpkg/tests")).unwrap();
    let mut session =
        entelechy_session::Session::open(std::path::Path::new("scene.esn"), None, 1 << 20);
    let events = session.sync();
    std::env::set_current_dir(was).unwrap();
    assert!(session.graph().is_some(), "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
