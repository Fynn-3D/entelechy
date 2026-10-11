//! The names a document sees, case by case, as `docs/DOCUMENTS.md`
//! (Declarations, `use` and functions; Packages) says them: what a
//! document sees through `use`, whose declaration a name reaches, what
//! two used documents may share, and what is refused, and why.
//!
//! Each case is a few documents and a scene placing `t.k@1` as `k`: it
//! checks clean and evaluates `k` to a value, or the check refuses it
//! with a problem saying the given words.

use std::path::PathBuf;

/// What a case comes to.
enum Want {
    /// Checks clean, and `k`'s first element is this.
    Runs(f32),
    /// Refused by the check, a problem saying this.
    Refused(&'static str),
}
use Want::{Refused, Runs};

/// The scene every case shares: `t.k@1` from `app.ent`.
const SCENE: &str = "use \"app.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n";

/// A package's manifest.
fn manifest(name: &str) -> String {
    format!("package {name}@1 {{ language = 1; ir = 1; }}\nuse \"kernels/\";\n")
}

/// `t.k@1` computing `body`, with `local` declared inside it.
fn kernel(local: &str, body: &str) -> String {
    format!(
        "kernel t.k@1 {{ requires {{ field.read; field.write; }} output result: Rows<Float>; {local} compute {{ {body} }} }}\n"
    )
}

fn app(uses: &[&str], top: &str, local: &str, body: &str) -> String {
    let uses: String = uses.iter().map(|u| format!("use \"{u}\";\n")).collect();
    format!("{uses}{top}{}", kernel(local, body))
}

/// A case: what it is, its documents by path, what it comes to.
type Case = (&'static str, Vec<(&'static str, String)>, Want);

fn cases() -> Vec<Case> {
    let s = |text: &str| text.to_string();
    vec![
        // A document sees what it uses itself.
        (
            "a used document's function calls what that document uses",
            vec![
                ("c.ent", s("fn deep() -> Float { return 3.0; }\n")),
                (
                    "b.ent",
                    s("use \"c.ent\";\nfn mid() -> Float { return deep(); }\n"),
                ),
                ("app.ent", app(&["b.ent"], "", "", "return mid();")),
            ],
            Runs(3.0),
        ),
        (
            "a function reached only through another's use is not seen",
            vec![
                ("c.ent", s("fn deep() -> Float { return 3.0; }\n")),
                (
                    "b.ent",
                    s("use \"c.ent\";\nfn mid() -> Float { return deep(); }\n"),
                ),
                ("app.ent", app(&["b.ent"], "", "", "return deep();")),
            ],
            Refused(
                "`deep` is declared in c.ent, which this document reaches only through another's `use`",
            ),
        ),
        (
            "a struct reached only through another's use is not named",
            vec![
                ("c.ent", s("struct Deep { x: Float; }\n")),
                (
                    "b.ent",
                    s("use \"c.ent\";\nfn mk() -> Deep { return Deep { x: 1.0 }; }\n"),
                ),
                (
                    "app.ent",
                    app(&["b.ent"], "", "", "return Deep { x: 2.0 }.x;"),
                ),
            ],
            Refused(
                "struct `Deep` is declared in c.ent, which this document reaches only through another's `use`",
            ),
        ),
        (
            "a value of a type not seen by name is used",
            vec![
                ("c.ent", s("struct Deep { x: Float; }\n")),
                (
                    "b.ent",
                    s("use \"c.ent\";\nfn mk() -> Deep { return Deep { x: 1.5 }; }\n"),
                ),
                ("app.ent", app(&["b.ent"], "", "", "return mk().x;")),
            ],
            Runs(1.5),
        ),
        (
            "two documents that use each other",
            vec![
                (
                    "a.ent",
                    s("use \"b.ent\";\nfn fa() -> Float { return 1.0; }\n"),
                ),
                (
                    "b.ent",
                    s("use \"a.ent\";\nfn fb() -> Float { return 2.0; }\n"),
                ),
                ("app.ent", app(&["a.ent"], "", "", "return fa();")),
            ],
            Refused("`a.ent` and `b.ent` use each other"),
        ),
        // A name reaches the Kernel's own, then the document's, then the
        // one a used document declares; each document's code keeps its own.
        (
            "the document's function over a used one's, the library keeping its own",
            vec![
                (
                    "lib.ent",
                    s("fn f() -> Float { return 1.0; }\nfn g() -> Float { return f(); }\n"),
                ),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "fn f() -> Float { return 10.0; }\n",
                        "",
                        "return f() + g();",
                    ),
                ),
            ],
            Runs(11.0),
        ),
        (
            "the Kernel's function over a used one's, the library keeping its own",
            vec![
                (
                    "lib.ent",
                    s("fn f() -> Float { return 1.0; }\nfn g() -> Float { return f(); }\n"),
                ),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "",
                        "fn f() -> Float { return 2.0; }",
                        "return f() * 10.0 + g();",
                    ),
                ),
            ],
            Runs(21.0),
        ),
        (
            "the Kernel's function over its document's, the document keeping its own",
            vec![(
                "app.ent",
                app(
                    &[],
                    "fn f() -> Float { return 1.0; }\nfn g() -> Float { return f(); }\n",
                    "fn f() -> Float { return 2.0; }",
                    "return f() * 10.0 + g();",
                ),
            )],
            Runs(21.0),
        ),
        (
            "the Kernel's struct over its document's",
            vec![(
                "app.ent",
                app(
                    &[],
                    "struct S { x: Float; }\n",
                    "struct S { y: Float; }",
                    "return S { y: 2.0 }.y;",
                ),
            )],
            Runs(2.0),
        ),
        (
            "the Kernel's struct over a used one's",
            vec![
                ("lib.ent", s("struct S { x: Float; }\n")),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "",
                        "struct S { y: Float; }",
                        "return S { y: 2.0 }.y;",
                    ),
                ),
            ],
            Runs(2.0),
        ),
        (
            "a library's function of a builtin's name stands for it there only",
            vec![
                (
                    "lib.ent",
                    s(
                        "fn dot(a: Vec2, b: Vec2) -> Float { return 99.0; }\nfn ld() -> Float { return dot(vec2(1.0, 0.0), vec2(1.0, 0.0)); }\n",
                    ),
                ),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "",
                        "",
                        "return dot(vec2(1.0, 0.0), vec2(1.0, 0.0)) + ld();",
                    ),
                ),
            ],
            Runs(100.0),
        ),
        (
            "a library's generic function over the document's struct",
            vec![
                ("lib.ent", s("fn same<T>(a: T) -> T { return a; }\n")),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "struct Q { v: Float; }\n",
                        "",
                        "return same(Q { v: 2.0 }).v;",
                    ),
                ),
            ],
            Runs(2.0),
        ),
        // Two used documents answering a name differently: refused where
        // it is named, and only there.
        (
            "two libraries' functions of one name, named",
            vec![
                ("a.ent", s("fn f() -> Float { return 1.0; }\n")),
                ("b.ent", s("fn f() -> Float { return 2.0; }\n")),
                ("app.ent", app(&["a.ent", "b.ent"], "", "", "return f();")),
            ],
            Refused("fn `f` is declared by a.ent and b.ent, which this document uses"),
        ),
        (
            "two libraries' functions of one name, each called by its own",
            vec![
                (
                    "a.ent",
                    s("fn f() -> Float { return 1.0; }\nfn fa() -> Float { return f(); }\n"),
                ),
                (
                    "b.ent",
                    s("fn f() -> Float { return 2.0; }\nfn fb() -> Float { return f(); }\n"),
                ),
                (
                    "app.ent",
                    app(&["a.ent", "b.ent"], "", "", "return fa() + fb();"),
                ),
            ],
            Runs(3.0),
        ),
        (
            "two libraries' functions written alike are still two",
            vec![
                ("a.ent", s("fn f() -> Float { return 1.0; }\n")),
                ("b.ent", s("fn f() -> Float { return 1.0; }\n")),
                ("app.ent", app(&["a.ent", "b.ent"], "", "", "return f();")),
            ],
            Refused("fn `f` is declared by a.ent and b.ent"),
        ),
        (
            "two files of a used directory, functions of one name",
            vec![
                ("lib/a.ent", s("fn f() -> Float { return 1.0; }\n")),
                ("lib/b.ent", s("fn f() -> Float { return 2.0; }\n")),
                ("app.ent", app(&["lib/"], "", "", "return f();")),
            ],
            Refused("fn `f` is declared by lib/a.ent and lib/b.ent"),
        ),
        (
            "two libraries' structs of one name, different",
            vec![
                ("a.ent", s("struct S { x: Float; }\n")),
                ("b.ent", s("struct S { y: Float; }\n")),
                (
                    "app.ent",
                    app(&["a.ent", "b.ent"], "", "", "return S { x: 1.0 }.x;"),
                ),
            ],
            Refused("struct `S` is declared by a.ent and b.ent"),
        ),
        (
            "two libraries' enums of one name, different",
            vec![
                ("a.ent", s("type Mode = enum { Idle, Walk };\n")),
                ("b.ent", s("type Mode = enum { Run, Fly };\n")),
                (
                    "app.ent",
                    s(
                        "use \"a.ent\";\nuse \"b.ent\";\nkernel t.k@1 { requires { field.write; } param m: Mode = 0; output result: Rows<Float>; compute { return float(m); } }\n",
                    ),
                ),
            ],
            Refused("type `Mode` is declared by a.ent and b.ent"),
        ),
        // Written alike, one; alike only where the names in them reach alike.
        (
            "a struct written alike in two libraries is one",
            vec![
                ("a.ent", s("struct S { x: Float; }\n")),
                ("b.ent", s("struct S { x: Float; }\n")),
                (
                    "app.ent",
                    app(&["a.ent", "b.ent"], "", "", "return S { x: 1.0 }.x;"),
                ),
            ],
            Runs(1.0),
        ),
        (
            "an enum written alike in two libraries is one",
            vec![
                ("a.ent", s("type Mode = enum { Idle, Walk };\n")),
                ("b.ent", s("type Mode = enum { Idle, Walk };\n")),
                (
                    "app.ent",
                    app(&["a.ent", "b.ent"], "", "", "return float(Mode.Walk);"),
                ),
            ],
            Runs(1.0),
        ),
        (
            "two libraries' enums of one name, a variant named",
            vec![
                ("a.ent", s("type Mode = enum { Idle, Walk };\n")),
                ("b.ent", s("type Mode = enum { Run, Fly };\n")),
                (
                    "app.ent",
                    app(&["a.ent", "b.ent"], "", "", "return float(Mode.Idle);"),
                ),
            ],
            Refused(
                "`Mode` is not an enum or a value; `Mode.Idle` names a variant of an enum this document sees; type `Mode` is declared by a.ent and b.ent",
            ),
        ),
        (
            "an enum reached only through another's use, a variant named",
            vec![
                ("c.ent", s("type Mode = enum { Idle, Walk };\n")),
                ("b.ent", s("use \"c.ent\";\n")),
                (
                    "app.ent",
                    app(&["b.ent"], "", "", "return float(Mode.Walk);"),
                ),
            ],
            Refused(
                "type `Mode` is declared in c.ent, which this document reaches only through another's `use`",
            ),
        ),
        (
            "a value misspelled before a member keeps its suggestion",
            vec![(
                "app.ent",
                app(&[], "", "", "let place = vec2(1.0, 2.0); return plcae.x;"),
            )],
            Refused("`plcae` is not defined; did you mean `place`?"),
        ),
        (
            "an enum of the same variants at other values is another",
            vec![
                ("a.ent", s("type Mode = enum { Idle, Walk };\n")),
                ("b.ent", s("type Mode = enum { Idle, Walk = 10 };\n")),
                (
                    "app.ent",
                    s(
                        "use \"a.ent\";\nuse \"b.ent\";\nkernel t.k@1 { requires { field.write; } param m: Mode = 0; output result: Rows<Float>; compute { return float(m); } }\n",
                    ),
                ),
            ],
            Refused("type `Mode` is declared by a.ent and b.ent"),
        ),
        (
            "a struct written alike over different inner structs is two",
            vec![
                ("a.ent", s("struct In { x: Float; }\nstruct S { i: In; }\n")),
                ("b.ent", s("struct In { y: Int; }\nstruct S { i: In; }\n")),
                (
                    "app.ent",
                    app(
                        &["a.ent", "b.ent"],
                        "",
                        "",
                        "return S { i: In { x: 1.0 } }.i.x;",
                    ),
                ),
            ],
            Refused("struct `S` is declared by a.ent and b.ent"),
        ),
        (
            "a type written alike over different structs is two",
            vec![
                (
                    "a.ent",
                    s("struct St { x: Float; }\ntype Sts = Rows<St>;\n"),
                ),
                ("b.ent", s("struct St { y: Int; }\ntype Sts = Rows<St>;\n")),
                (
                    "app.ent",
                    s(
                        "use \"a.ent\";\nuse \"b.ent\";\nkernel t.k@1 { requires { field.read; field.write; } input s: Sts; output result: Rows<Float>; compute { return 0.0; } }\n",
                    ),
                ),
            ],
            Refused("type `Sts` is declared by a.ent and b.ent"),
        ),
        (
            "alike structs over different inner ones, each built by its own library",
            vec![
                (
                    "a.ent",
                    s(
                        "struct In { x: Float; }\nstruct S { i: In; }\nfn sa() -> S { return S { i: In { x: 1.0 } }; }\n",
                    ),
                ),
                (
                    "b.ent",
                    s(
                        "struct In { y: Int; }\nstruct S { i: In; }\nfn sb() -> S { return S { i: In { y: 2 } }; }\n",
                    ),
                ),
                (
                    "app.ent",
                    app(
                        &["a.ent", "b.ent"],
                        "",
                        "",
                        "return sa().i.x + float(sb().i.y);",
                    ),
                ),
            ],
            Runs(3.0),
        ),
        // A value matches by layout: the document's own alike struct is
        // the library's, a different one is refused where it is passed.
        (
            "the document's alike struct passed to a library's function",
            vec![
                (
                    "lib.ent",
                    s("struct P { x: Float; }\nfn px(p: P) -> Float { return p.x; }\n"),
                ),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "struct P { x: Float; }\n",
                        "",
                        "return px(P { x: 1.0 });",
                    ),
                ),
            ],
            Runs(1.0),
        ),
        (
            "the document's different struct passed to a library's function",
            vec![
                (
                    "lib.ent",
                    s("struct P { x: Float; }\nfn px(p: P) -> Float { return p.x; }\n"),
                ),
                (
                    "app.ent",
                    app(
                        &["lib.ent"],
                        "struct P { y: Int; z: Int; }\n",
                        "",
                        "return px(P { y: 1, z: 2 });",
                    ),
                ),
            ],
            Refused("`px` expects P (of lib.ent) here, found P"),
        ),
        (
            "the document's type alike over its own struct passed to a library's function",
            vec![
                (
                    "lib.ent",
                    s(
                        "struct St { x: Float; }\ntype Sts = Rows<St>;\nfn first_x(s: Sts) -> Float { return at(s, 0).x; }\n",
                    ),
                ),
                (
                    "app.ent",
                    s(
                        "use \"lib.ent\";\nstruct St { q: Int; }\ntype Sts = Rows<St>;\nkernel t.k@1 { requires { field.read; field.write; } input s: Sts; output result: Rows<Float>; compute { return first_x(s); } }\n",
                    ),
                ),
            ],
            Refused("`first_x` expects Sts (of lib.ent) for `s`, but `s` is Rows<St>"),
        ),
        // A scene's inline Kernels see what the scene uses, as a document's.
        (
            "a scene's inline Kernel naming two libraries' function",
            vec![
                ("a.ent", s("fn f() -> Float { return 1.0; }\n")),
                ("b.ent", s("fn f() -> Float { return 2.0; }\n")),
                (
                    "scene.esn",
                    s(
                        "use \"a.ent\";\nuse \"b.ent\";\nnode k: kernel { requires { field.write; } output result: Rows<Float>; compute { return f(); } }\noutput result <- k.result;\n",
                    ),
                ),
            ],
            Refused("fn `f` is declared by a.ent and b.ent"),
        ),
        // A Kernel id is one Kernel.
        (
            "a Kernel id written in two documents the scene publishes",
            vec![
                ("a.ent", kernel("", "return 1.0;")),
                ("b.ent", kernel("", "return 2.0;")),
                (
                    "scene.esn",
                    s(
                        "use \"a.ent\";\nuse \"b.ent\";\nnode k: t.k@1 { domain = 1; }\noutput result <- k.result;\n",
                    ),
                ),
            ],
            Refused("t.k@1 is already published by"),
        ),
        // A package is used whole; it finds the packages it names beside
        // it; what it uses is its own; a name is one package.
        (
            "a package used whole",
            vec![
                ("p.entpkg/package.esn", manifest("p")),
                (
                    "p.entpkg/kernels/p.ent",
                    s("fn pf() -> Float { return 5.0; }\n"),
                ),
                ("app.ent", app(&["p.entpkg"], "", "", "return pf();")),
            ],
            Runs(5.0),
        ),
        (
            "a file inside a package",
            vec![
                ("p.entpkg/package.esn", manifest("p")),
                (
                    "p.entpkg/kernels/p.ent",
                    s("fn pf() -> Float { return 5.0; }\n"),
                ),
                (
                    "app.ent",
                    app(&["p.entpkg/kernels/p.ent"], "", "", "return pf();"),
                ),
            ],
            Refused("is inside the package p.entpkg; use the package whole"),
        ),
        (
            "a package naming the package beside it",
            vec![
                ("lib/q.entpkg/package.esn", manifest("q")),
                (
                    "lib/q.entpkg/kernels/q.ent",
                    s("fn qf() -> Float { return 2.0; }\n"),
                ),
                ("lib/p.entpkg/package.esn", manifest("p")),
                (
                    "lib/p.entpkg/kernels/p.ent",
                    s("use \"q.entpkg\";\nfn pf() -> Float { return qf() * 3.0; }\n"),
                ),
                ("app.ent", app(&["lib/p.entpkg"], "", "", "return pf();")),
            ],
            Runs(6.0),
        ),
        (
            "what a package uses is not seen by its user",
            vec![
                ("lib/q.entpkg/package.esn", manifest("q")),
                (
                    "lib/q.entpkg/kernels/q.ent",
                    s("fn qf() -> Float { return 2.0; }\n"),
                ),
                ("lib/p.entpkg/package.esn", manifest("p")),
                (
                    "lib/p.entpkg/kernels/p.ent",
                    s("use \"q.entpkg\";\nfn pf() -> Float { return qf() * 3.0; }\n"),
                ),
                ("app.ent", app(&["lib/p.entpkg"], "", "", "return qf();")),
            ],
            Refused(
                "which this document reaches only through another's `use`: a document sees what it uses itself, `use \"lib/q.entpkg\";`",
            ),
        ),
        (
            "two editions of one package",
            vec![
                ("x/p.entpkg/package.esn", manifest("p")),
                (
                    "x/p.entpkg/kernels/p.ent",
                    s("fn pf() -> Float { return 5.0; }\n"),
                ),
                ("y/p.entpkg/package.esn", manifest("p")),
                (
                    "y/p.entpkg/kernels/p.ent",
                    s("fn pg() -> Float { return 6.0; }\n"),
                ),
                (
                    "app.ent",
                    app(&["x/p.entpkg", "y/p.entpkg"], "", "", "return pf();"),
                ),
            ],
            Refused("two packages named `p` are reached"),
        ),
    ]
}

/// The case's documents written into a directory of their own.
fn written(index: usize, files: &[(&str, String)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-scope-{index}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for (file, text) in files {
        let path = dir.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    if !files.iter().any(|(file, _)| *file == "scene.esn") {
        std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    }
    dir
}

#[test]
fn the_names_a_document_sees() {
    let mut wrong = Vec::new();
    for (index, (name, files, want)) in cases().into_iter().enumerate() {
        let dir = written(index, &files);
        let scene = dir.join("scene.esn");
        let checked =
            entelechy_session::check_document(&entelechy_session::Disk::default(), &scene, false);
        let problems: Vec<&str> = checked
            .lines
            .iter()
            .filter(|(problem, _)| *problem)
            .map(|(_, line)| line.as_str())
            .collect();
        match want {
            Refused(words) => {
                if !problems.iter().any(|line| line.contains(words)) {
                    wrong.push(format!(
                        "{name}: wanted refused saying {words:?}, the check said {problems:?}"
                    ));
                }
            }
            Runs(value) => {
                if !problems.is_empty() {
                    wrong.push(format!(
                        "{name}: wanted {value}, the check said {problems:?}"
                    ));
                } else {
                    let mut session = entelechy_session::Session::open(&scene, None, 1 << 20);
                    let events = session.sync();
                    let got = session
                        .evaluate(&entelechy_kernel::Context::default())
                        .ok()
                        .and_then(|_| {
                            session
                                .element_field(
                                    &entelechy_graph::NodeId("k".into()),
                                    "result",
                                    0,
                                    &[],
                                )
                                .ok()
                        })
                        .map(|(_, words)| f32::from_bits(words[0]));
                    if got != Some(value) {
                        wrong.push(format!("{name}: wanted {value}, got {got:?} ({events:?})"));
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
