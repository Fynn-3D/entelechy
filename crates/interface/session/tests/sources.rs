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
    document: None,
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

/// A declaration the using document makes alike itself is its own and
/// the used one's: it is not gathered again, and the used document's
/// functions name it as written.
#[test]
fn a_declaration_made_alike_by_the_user_is_one() {
    let dir = scratch(
        "alike",
        &[(
            "a.orla",
            "type Cells = Volume<Int>;\nfn corner(v: Cells) -> Int { return at(v, ivec3(0, 0, 0)); }\n",
        )],
    );
    let own = orla_language::parse_document("type Cells = Volume<Int>;\n").unwrap();
    let (_, types, functions) = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";"),
        orla_language::Own::of(&own),
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

/// A scene's struct may hold an enum a used document declares beside the
/// scene's own params: a param's default is read with the structs in
/// reach, their fields through the aliases in reach.
#[test]
fn a_scene_struct_holds_a_used_enum_beside_its_params() {
    let dir = scratch(
        "enum-beside-param",
        &[
            ("lib.orla", "type Mode = enum { Idle, Walk };\n"),
            (
                "scene.osn",
                "use \"lib.orla\";\nparam speed: Float = 1.0;\nstruct Pose { mode: Mode; modes: [Mode; 2]; }\nnode n: kernel { requires { field.write; } output result: Rows<Pose>; domain = 1; compute { return Pose { mode: Mode.Walk, modes: [Mode.Idle, Mode.Walk] }; } }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&orla_kernel::Context::default()).unwrap();
    let (_, words) = session
        .element_field(
            &orla_graph::NodeId("n".into()),
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
                "lib.orla",
                "struct Pair { a: Int; b: Int; }\ntype Mode = enum { Idle, Walk };\nfn over(a: Int, b: Int) -> Int { return a + b; }\nfn total(x: Int) -> Int { let p = Pair { a: x, b: x * 2 }; return over(p.a, p.b) + Mode.Walk; }\n",
            ),
            (
                "scene.osn",
                "use \"lib.orla\";\nnode n: kernel { requires { field.write; } output result: Rows<Int>; domain = 1;\n    struct Pair { left: Int; right: Int; }\n    type Mode = enum { Off = 10, On = 20 };\n    fn over(a: Int, b: Int) -> Int { return a * b; }\n    compute { let p = Pair { left: 1, right: Mode.On }; return over(3, 4) + p.left + p.right + total(5); } }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&orla_kernel::Context::default()).unwrap();
    // Its own: 3 * 4, and 1 + 20; the library's: 5 + 10, and 1.
    let (_, words) = session
        .element_field(&orla_graph::NodeId("n".into()), "result", 0, &[])
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
                ("lib.orla", declaration),
                (
                    "k.orla",
                    "use \"lib.orla\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }\n",
                ),
                (
                    "scene.osn",
                    "use \"k.orla\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
                ),
            ],
        );
        let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
        let events = session.sync();
        let faults: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                orla_session::Event::Rejected { path, error }
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
                "lib.orla",
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
            "use \"lib.orla\";\nkernel t.k@1 {{ requires {{ field.write; }} output result: Rows<Int>;\n    {own}\n    compute {{ return 1; }} }}\n"
        )
    };
    let cases = [
        (
            "struct Pair { a: Int; }\n",
            kernel("struct Pair { b: Nope; }"),
            "k.orla",
        ),
        (
            "struct Pair { a: Nope; }\n",
            kernel("struct Pair { b: Int; }"),
            "lib.orla",
        ),
        (
            "fn twice(x: Int) -> Int { return 2; }\n",
            kernel("fn twice(x: Nope) -> Int { return 2; }"),
            "k.orla",
        ),
        (
            "fn twice(x: Nope) -> Int { return 2; }\n",
            kernel("fn twice(x: Int) -> Int { return 2; }"),
            "lib.orla",
        ),
    ];
    for (k, (library, kernel, written)) in cases.iter().enumerate() {
        let dir = scratch(
            &format!("shadowing-fault-{k}"),
            &[
                ("lib.orla", library),
                ("k.orla", kernel),
                (
                    "scene.osn",
                    "use \"k.orla\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
                ),
            ],
        );
        let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
        let events = session.sync();
        let faults: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                orla_session::Event::Rejected { path, error }
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
        "kept/a.orlapkg/kernels",
        "kept/a.orlapkg/graphs",
        "kept/b.orlapkg/kernels",
        "kept/b.orlapkg/graphs",
        "scene",
    ] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    for (file, text) in [
        ("kept/a.orlapkg/package.osn", manifest("a")),
        (
            "kept/a.orlapkg/kernels/a.orla",
            "fn a_one() -> Int { return 1; }\nkernel a.twice@1 { requires { field.read; field.write; } input counts: Rows<Int>; output result: Rows<Int>; compute { return sample(counts) * 2; } }\n".into(),
        ),
        (
            "kept/a.orlapkg/graphs/twice.osn",
            "use \"../kernels/a.orla\";\ninput counts: Rows<Int>;\nnode once: a.twice@1 { counts <- counts; }\noutput result <- once.result;\n".into(),
        ),
        ("kept/b.orlapkg/package.osn", manifest("b")),
        (
            "kept/b.orlapkg/kernels/b.orla",
            "use \"a.orlapkg\";\nkernel b.start@1 { requires { field.write; } output result: Rows<Int>; compute { return element() + a_one(); } }\n".into(),
        ),
        (
            "kept/b.orlapkg/graphs/start.osn",
            "use \"../kernels/b.orla\";\nnode s: b.start@1 { domain = 3; }\nnode t: graph \"a.orlapkg/graphs/twice.osn\" { counts <- s.result; }\noutput result <- t.result;\n".into(),
        ),
        (
            "scene/scene.osn",
            "use \"../kept/b.orlapkg\";\nnode p: graph \"../kept/b.orlapkg/graphs/start.osn\" { }\noutput shown <- p.result;\n".into(),
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let mut session = orla_session::Session::open(&dir.join("scene/scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&orla_kernel::Context::default()).unwrap();
    let (_, words) = session
        .element_field(&orla_graph::NodeId("p".into()), "result", 2, &[])
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
            root.join("package.osn"),
            format!("package {id} {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n"),
        )
        .unwrap();
        std::fs::write(
            root.join("kernels/k.orla"),
            "fn one() -> Int { return 1; }\n",
        )
        .unwrap();
    };
    let scene = "node s: kernel { requires { field.write; } output result: Rows<Int>; domain = 1; compute { return one(); } }\noutput result <- s.result;\n";
    let refusal = |dir: &std::path::Path| {
        let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
        let events = session.sync();
        (session.graph().is_some(), format!("{events:?}"))
    };

    let dir = scratch("one-package-editions", &[]);
    package(&dir, "one/p.orlapkg", "p@1");
    package(&dir, "two/p.orlapkg", "p@2");
    std::fs::write(
        dir.join("scene.osn"),
        format!("use \"one/p.orlapkg\";\nuse \"two/p.orlapkg\";\n{scene}"),
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
        dir.join("scene.osn"),
        format!("use \"one/p.orlapkg\";\nuse \"one/../one/p.orlapkg\";\n{scene}"),
    )
    .unwrap();
    let (loaded, said) = refusal(&dir);
    assert!(loaded, "{said}");
    let _ = std::fs::remove_dir_all(&dir);

    let dir = scratch("one-package-misnamed", &[]);
    package(&dir, "q.orlapkg", "p@1");
    std::fs::write(
        dir.join("scene.osn"),
        format!("use \"q.orlapkg\";\n{scene}"),
    )
    .unwrap();
    let (_, said) = refusal(&dir);
    assert!(
        said.contains("the directory `q.orlapkg` declares the package `p@1`"),
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
                "lib.orla",
                "fn dot(a: Int) -> Int { return a; }\nfn uses_own(a: Int) -> Int { return dot(a) + 1; }\n",
            ),
            (
                "scene.osn",
                "use \"lib.orla\";\nnode n: kernel { requires { field.write; } output result: Rows<Float>; domain = 1; compute { return dot(vec2(2.0, 0.0), vec2(1.5, 0.0)) + float(uses_own(3)); } }\noutput result <- n.result;\n",
            ),
            (
                "own.orla",
                "fn dot(a: Int) -> Int { return a; }\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return dot(vec2(1.0, 0.0), vec2(1.0, 0.0)); } }\n",
            ),
        ],
    );
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&orla_kernel::Context::default()).unwrap();
    // The builtin: 2 * 1.5; the library's own: 3 + 1.
    let (_, words) = session
        .element_field(&orla_graph::NodeId("n".into()), "result", 0, &[])
        .unwrap();
    assert_eq!(f32::from_bits(words[0]), 7.0);
    let checked =
        orla_session::check_document(&orla_session::Disk::default(), &dir.join("own.orla"), false);
    assert!(
        checked.lines.iter().any(|(problem, line)| *problem
            && line.contains("hides the builtin `dot`, declared in own.orla")),
        "{checked:?}"
    );
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
            "kernels/atlas.orla",
            "use \"palettes.orla\";\nkernel t.atlas@1 { requires { field.write; } output result: Rows<Int>; compute { return twice(1); } }\n",
        ),
        (
            "kernels/palettes.orla",
            "fn twice(x: Int) -> Int { return 2 * x; }\nkernel t.greens@1 { requires { field.write; } output result: Rows<Int>; compute { return 4; } }\n",
        ),
        (
            "scene.osn",
            "use \"kernels/atlas.orla\";\nnode a: t.atlas@1 { domain = 1; }\nnode g: t.greens@1 { domain = 1; }\noutput result <- g.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let wants = "`t.greens@1` is written in kernels/palettes.orla, which the scene reaches only through kernels/atlas.orla";
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let first = format!("{:?}{:?}", session.sync(), session.sync());
    assert_eq!(first.matches(wants).count(), 1, "{first}");
    assert!(
        first.contains("wants `use \\\"kernels/palettes.orla\\\";`"),
        "{first}"
    );
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
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
    std::fs::write(dir.join("scene.osn"), "use \"kernels/atlas.orla\";\nuse \"kernels/palettes.orla\";\nnode a: t.atlas@1 { domain = 1; }\nnode g: t.greens@1 { domain = 1; }\noutput result <- g.result;\n").unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `orla check` compiles the Kernels a scene publishes, as a session
/// does: a Kernel of a document the scene reaches only through another
/// is not the scene's, while that document's functions, which Kernels
/// inherit, are checked.
#[test]
fn check_compiles_the_kernels_a_scene_publishes() {
    let dir = scratch("check-published", &[]);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for (file, text) in [
        (
            "kernels/atlas.orla",
            "use \"deep.orla\";\nkernel t.atlas@1 { requires { field.write; } output result: Rows<Int>; compute { return twice(1); } }\n",
        ),
        (
            "kernels/deep.orla",
            "fn twice(x: Int) -> Int { return 2 * x; }\nkernel t.deep@1 { requires { field.write; } output result: Rows<Int>; compute { return nowhere; } }\n",
        ),
        (
            "scene.osn",
            "use \"kernels/atlas.orla\";\nnode a: t.atlas@1 { domain = 1; }\noutput result <- a.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
    std::fs::write(dir.join("kernels/deep.orla"), "fn twice(x: Int) -> Int { return 2 * nothing; }\nkernel t.deep@1 { requires { field.write; } output result: Rows<Int>; compute { return nowhere; } }\n").unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
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
                "lib.orla",
                "fn twice(x: Int) -> Int { return x * 2; }\nfn four() -> Int { return twice(1, 2); }\n",
            ),
            (
                "k.orla",
                "use \"lib.orla\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Int>;\n    fn twice(x: Int) -> Int { return x + x; }\n    compute { return twice(1) + four(); } }\n",
            ),
            (
                "scene.osn",
                "use \"k.orla\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
            ),
        ],
    );
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let faults: Vec<_> = session
        .sync()
        .into_iter()
        .filter_map(|event| match event {
            orla_session::Event::Rejected { path, error }
                if error.to_string().contains("argument") =>
            {
                Some((path, error.to_string()))
            }
            _ => None,
        })
        .collect();
    assert!(!faults.is_empty());
    for (path, said) in faults {
        assert_eq!(path.file_name().unwrap(), "lib.orla", "{said}");
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
                "c.orla",
                "struct Deep { v: Float; }\nfn deep() -> Deep { return Deep { v: 2.0 }; }\n",
            ),
            (
                "b.orla",
                "use \"c.orla\";\nfn mid() -> Float { return deep().v + 1.0; }\n",
            ),
            (
                "a.orla",
                "use \"b.orla\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return deep().v; } }\n",
            ),
            (
                "scene.osn",
                "use \"a.orla\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
            ),
        ],
    );
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    let said = format!("{checked:?}");
    assert!(
        said.contains(
            "`deep` is declared in c.orla, which this document reaches only through another's `use`"
        ) && said.contains("use \\\"c.orla\\\";"),
        "{said}"
    );
    // Through a function of the document it uses, it is reached.
    std::fs::write(
        dir.join("a.orla"),
        "use \"b.orla\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return mid(); } }\n",
    )
    .unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{checked:?}");
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
                "package.orla",
                "fn shade() -> Float { return 1.0; }\nfn lit() -> Float { return shade() + 2.0; }\n",
            ),
            (
                "widgets.orla",
                "use \"package.orla\";\nfn shade() -> Float { return 10.0; }\nfn both() -> Float { return shade() + lit(); }\n",
            ),
            (
                "scene.osn",
                "use \"widgets.orla\";\nnode k: kernel { requires { field.write; } output result: Rows<Float>; domain = 1; compute { return shade() * 100.0 + both(); } }\noutput result <- k.result;\n",
            ),
        ],
    );
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&orla_kernel::Context::default()).unwrap();
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
    std::fs::create_dir_all(dir.join("packages/deep.orlapkg/kernels")).unwrap();
    for (file, text) in [
        (
            "packages/deep.orlapkg/package.osn",
            "package deep@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
        ),
        (
            "packages/deep.orlapkg/kernels/deep.orla",
            "fn deep() -> Float { return 2.0; }\n",
        ),
        (
            "b.orla",
            "use \"deep.orlapkg\";\nfn mid() -> Float { return deep(); }\n",
        ),
        (
            "a.orla",
            "use \"b.orla\";\nkernel t.k@1 { requires { field.write; } output result: Rows<Float>; compute { return deep(); } }\n",
        ),
        (
            "scene.osn",
            "use \"a.orla\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
        ),
    ] {
        std::fs::write(dir.join(file), text).unwrap();
    }
    let store = orla_session::Disk::with_packages(vec![dir.join("packages")]);
    let checked = orla_session::check_document(&store, &dir.join("scene.osn"), false);
    let said = format!("{checked:?}");
    assert!(said.contains("use \\\"deep.orlapkg\\\";"), "{said}");
    let _ = std::fs::remove_dir_all(&dir);
}
