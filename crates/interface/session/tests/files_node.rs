//! A `files` node lists a directory, leaving out what its `.gitignore`
//! files and hidden names leave out, each file's path and size; and gives
//! the files its `paths` named when last evaluated, their bytes, in the
//! order named, a path naming no file left out; each read again when it
//! changes.

use std::path::{Path, PathBuf};

use entelechy_kernel::{Context, Value};
use entelechy_session::Session;

/// A Kernel's array literal of `paths`, each path's bytes ended by a 0,
/// padded with 0 to `n`.
fn spelled(paths: &[&str], n: usize) -> String {
    let mut words: Vec<String> = paths
        .iter()
        .flat_map(|path| {
            path.bytes()
                .map(|b| b.to_string())
                .chain(std::iter::once("0".into()))
        })
        .collect();
    words.resize(n, "0".into());
    format!("[{}]", words.join(", "))
}

fn scene(first: &[&str], then: &[&str]) -> String {
    let n = 64;
    format!(
        "kernel t.paths@1 {{ requires {{ field.write; }} param which: Int = 0; output result: Rows<Int>;
    compute {{ return if which == 0 {{ {}[element()] }} else {{ {}[element()] }}; }} }}
kernel t.copy@1 {{ requires {{ field.read; field.write; }} input words: Rows<Int>; output result: Rows<Int>; compute {{ return sample(words); }} }}
param which: Int = 0;
node paths: t.paths@1 {{ which = which; domain = {n}; }}
node d: files \"proj/\" {{ paths <- paths.result; }}
node listing: t.copy@1 {{ words <- d.listing; }}
node opened: t.copy@1 {{ words <- d.opened; }}
output listing <- listing.result;
output opened <- opened.result;
",
        spelled(first, n),
        spelled(then, n)
    )
}

fn scratch(name: &str, first: &[&str], then: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "entelechy-files-node-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let write = |path: &str, text: &str| {
        let path = dir.join("proj").join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write(".gitignore", "target/\n*.log\n!keep.log\n");
    write(".hidden", "not listed");
    write("notes.md", "# notes\n");
    write("src/a.ent", "fn a() -> Int { return 1; }\n");
    write("src/b.ent", "fn b() -> Int { return 2; }\n");
    write("target/out.bin", "built");
    write("build.log", "a log");
    write("keep.log", "kept");
    write("sub/.gitignore", "local.txt\n");
    write("sub/local.txt", "mine");
    write("sub/kept.txt", "ours");
    std::fs::write(dir.join("scene.esn"), scene(first, then)).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

fn words(session: &mut Session, output: &str) -> Vec<u32> {
    let handle = session.output(Some(output)).unwrap();
    session.field(handle).unwrap().values().to_vec()
}

/// The listing: each file's path and size.
fn listing(session: &mut Session) -> Vec<(String, u32)> {
    session.evaluate(&Context::default()).unwrap();
    let w = words(session, "listing");
    let mut at = 1;
    let mut out = Vec::new();
    for _ in 0..w[0] {
        let (len, size) = (w[at] as usize, w[at + 1]);
        let path: Vec<u8> = w[at + 2..at + 2 + len].iter().map(|b| *b as u8).collect();
        out.push((String::from_utf8(path).unwrap(), size));
        at += 2 + len;
    }
    out
}

/// What was read: each file's path and text.
fn opened(session: &mut Session) -> Vec<(String, String)> {
    session.evaluate(&Context::default()).unwrap();
    let w = words(session, "opened");
    let mut at = 1;
    let mut out = Vec::new();
    for _ in 0..w[0] {
        let (len, size) = (w[at] as usize, w[at + 1] as usize);
        let bytes = |from: usize, n: usize| -> String {
            String::from_utf8(w[from..from + n].iter().map(|b| *b as u8).collect()).unwrap()
        };
        out.push((bytes(at + 2, len), bytes(at + 2 + len, size)));
        at += 2 + len + size;
    }
    out
}

#[test]
fn the_listing_leaves_out_what_the_gitignore_files_and_hidden_names_leave_out() {
    let dir = scratch("listing", &[], &[]);
    let mut session = open(&dir);
    assert_eq!(
        listing(&mut session),
        vec![
            ("keep.log".to_owned(), 4),
            ("notes.md".to_owned(), 8),
            ("src/a.ent".to_owned(), 28),
            ("src/b.ent".to_owned(), 28),
            ("sub/kept.txt".to_owned(), 4),
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_files_named_are_read_a_frame_later_in_the_order_named() {
    let dir = scratch(
        "named",
        &["src/a.ent", "missing.txt", "notes.md"],
        &["src/b.ent", "target/out.bin"],
    );
    let mut session = open(&dir);
    // Before any paths are evaluated, nothing is read.
    assert_eq!(opened(&mut session), vec![]);
    // Then the files named, one naming nothing left out.
    assert_eq!(
        opened(&mut session),
        vec![
            (
                "src/a.ent".to_owned(),
                "fn a() -> Int { return 1; }\n".to_owned()
            ),
            ("notes.md".to_owned(), "# notes\n".to_owned()),
        ]
    );
    // Other paths: other files, a frame later; one the listing leaves out
    // is read all the same when named.
    session.set_param("which", Value::Int(1)).unwrap();
    opened(&mut session);
    assert_eq!(
        opened(&mut session),
        vec![
            (
                "src/b.ent".to_owned(),
                "fn b() -> Int { return 2; }\n".to_owned()
            ),
            ("target/out.bin".to_owned(), "built".to_owned()),
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_named_is_read_again_when_it_changes_and_the_listing_follows_the_directory() {
    let dir = scratch("changes", &["src/a.ent", "fresh.md"], &[]);
    let mut session = open(&dir);
    opened(&mut session);
    assert_eq!(opened(&mut session).len(), 1);
    // A stamp is a modification time and a length: a write lands later.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.join("proj/src/a.ent"), "fn a() -> Int { return 10; }\n").unwrap();
    std::fs::write(dir.join("proj/fresh.md"), "new").unwrap();
    session.sync();
    assert_eq!(
        opened(&mut session),
        vec![
            (
                "src/a.ent".to_owned(),
                "fn a() -> Int { return 10; }\n".to_owned()
            ),
            ("fresh.md".to_owned(), "new".to_owned()),
        ]
    );
    assert!(listing(&mut session).contains(&("fresh.md".to_owned(), 3)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_path_out_of_the_directory_is_refused_and_what_was_read_stays() {
    let dir = scratch("out", &["notes.md"], &["../scene.esn"]);
    let mut session = open(&dir);
    opened(&mut session);
    assert_eq!(opened(&mut session).len(), 1);
    session.set_param("which", Value::Int(1)).unwrap();
    opened(&mut session);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(
        said.iter()
            .any(|line| line.contains("files node `d`") && line.contains("../scene.esn")),
        "{said:?}"
    );
    assert_eq!(opened(&mut session).len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_files_node_is_said_wrong_where_it_is_written() {
    let dir = scratch("wrong", &[], &[]);
    let said = |text: &str| -> Vec<String> {
        std::fs::write(dir.join("scene.esn"), text).unwrap();
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
        session.sync().iter().map(ToString::to_string).collect()
    };
    let none = said("node d: files \"proj/\" { }\noutput o <- d.listing;\n");
    assert!(
        none.iter()
            .any(|line| line.contains("a files node needs `paths <-")),
        "{none:?}"
    );
    let sized = said(
        "kernel t.p@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\nnode p: t.p@1 { domain = 1; }\nnode d: files \"proj/\" { paths <- p.result; domain = 4; }\noutput o <- d.listing;\n",
    );
    assert!(
        sized
            .iter()
            .any(|line| line.contains("a files node takes no `domain`")),
        "{sized:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_files_node_lists_the_directory_given_for_it_by_name() {
    let dir = scratch("given", &[], &[]);
    std::fs::create_dir_all(dir.join("other")).unwrap();
    std::fs::write(dir.join("other/only.md"), "x").unwrap();
    let mut session = open(&dir);
    let mut given = entelechy_session::overrides::Overrides::default();
    given.inputs.insert("d".into(), dir.join("other/"));
    session.place(given).unwrap();
    session.sync();
    assert_eq!(listing(&mut session), vec![("only.md".to_owned(), 1)]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_session_started_over_reads_no_file_until_it_is_given_paths_again() {
    let dir = scratch("over", &["notes.md"], &[]);
    let mut session = open(&dir);
    opened(&mut session);
    assert_eq!(opened(&mut session).len(), 1);
    session.start_over();
    assert_eq!(opened(&mut session), vec![]);
    assert_eq!(opened(&mut session).len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_named_through_a_package_is_listed_and_read_inside_that_package() {
    // The directory only in a package kept elsewhere, the scene naming it by
    // the package's name, as an input's file is.
    let dir = scratch("package", &["notes.md"], &[]);
    let packages = dir.join("packages");
    std::fs::create_dir_all(packages.join("notes.entpkg")).unwrap();
    std::fs::write(
        packages.join("notes.entpkg/package.esn"),
        "package notes@1 { language = 1; ir = 1; }\n",
    )
    .unwrap();
    std::fs::rename(dir.join("proj"), packages.join("notes.entpkg/proj")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        scene(&["notes.md"], &[]).replace("files \"proj/\"", "files \"notes.entpkg/proj/\""),
    )
    .unwrap();
    let store = std::sync::Arc::new(entelechy_session::Disk::with_packages(vec![packages]));
    let mut session = Session::open_in(
        store,
        &dir.join("scene.esn"),
        None,
        entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(entelechy_cpu::CpuBackend)),
    );
    // Reads held to the scene's own directories and its packages, as the
    // Player's are.
    session.restrict_reads().unwrap();
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    assert!(
        listing(&mut session)
            .iter()
            .any(|(path, _)| path == "notes.md")
    );
    assert_eq!(
        opened(&mut session),
        vec![("notes.md".to_owned(), "# notes\n".to_owned())]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
