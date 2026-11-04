//! A `font` node gives the glyphs of the characters its `codes` held when
//! last evaluated: a row a code, in order, a code the font lacks `-1`;
//! read again only when the codes or the font change.

use std::path::PathBuf;

use entelechy_kernel::{Context, Value};
use entelechy_session::Session;

const SCENE: &str = "
struct Letterform { code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2; }
kernel t.codes@1 { requires { field.write; } param which: Int = 0; output result: Rows<Int>;
    compute { return if which == 0 { [111, 20013, 73][element()] } else { [73, 73, 0][element()] }; } }
kernel t.copy@1 { requires { field.read; field.write; } input glyphs: Rows<Letterform>; output result: Rows<Letterform>; compute { return sample(glyphs); } }
struct Measures { units: Int; ascent: Float; descent: Float; gap: Float; }
kernel t.measured@1 { requires { field.read; field.write; } input metrics: Rows<Measures>; output result: Rows<Measures>; compute { return sample(metrics); } }
kernel t.ends@1 { requires { field.read; field.write; } input outlines: Rows<Vec4>; output result: Rows<Vec4>; compute { return sample(outlines); } }
param which: Int = 0;
node codes: t.codes@1 { which = which; domain = 3; }
node g: font \"fonts/sample.ttf\" { codes <- codes.result; }
node copy: t.copy@1 { glyphs <- g.glyphs; }
node ends: t.ends@1 { outlines <- g.outlines; }
node measured: t.measured@1 { metrics <- g.metrics; }
output result <- copy.result;
output outlines <- ends.result;
output metrics <- measured.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-font-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("fonts")).unwrap();
    std::fs::write(
        dir.join("fonts/sample.ttf"),
        entelechy_import::font::sample::latin(),
    )
    .unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn open(dir: &std::path::Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// The codes of the glyph rows the scene's `result` holds.
fn codes(session: &mut Session) -> Vec<i32> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let words = session.field(handle).unwrap().values().to_vec();
    words.chunks(8).map(|row| row[0] as i32).collect()
}

#[test]
fn the_glyphs_follow_the_codes_a_frame_later() {
    let dir = scratch("follow");
    let mut session = open(&dir);
    // Before any codes are evaluated, the missing glyph alone.
    assert_eq!(codes(&mut session), vec![-1]);
    // Then the three asked, the one the font lacks -1.
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    let metrics = session.output(Some("metrics")).unwrap();
    assert_eq!(session.field(metrics).unwrap().values()[0], 1000);
    let outlines = session.output(Some("outlines")).unwrap();
    assert!(session.field(outlines).unwrap().values().len() > 8);
    // Other codes: other glyphs, a frame later; `I` drawn once for both.
    session.set_param("which", Value::Int(1)).unwrap();
    codes(&mut session);
    assert_eq!(codes(&mut session), vec![73, 73, -1]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_same_codes_leave_the_glyphs_as_they_were() {
    let dir = scratch("same");
    let mut session = open(&dir);
    codes(&mut session);
    codes(&mut session);
    let report = session.evaluate(&Context::default()).unwrap();
    assert!(
        report.reused.iter().any(|node| node.0 == "copy"),
        "{:?}",
        report.executed
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_font_node_names_its_one_input() {
    let dir = scratch("input");
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("{ codes <- codes.result; }", "{ chars <- codes.result; }"),
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(said.iter().any(|line| line.contains("`codes`")), "{said:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_font_by_absolute_path_elsewhere_gives_fields_from_the_first_frame() {
    let dir = scratch("absolute");
    let elsewhere = scratch("absolute-font");
    let font = elsewhere.join("fonts/sample.ttf");
    std::fs::remove_dir_all(dir.join("fonts")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("fonts/sample.ttf", font.to_str().unwrap()),
    )
    .unwrap();
    let mut session = open(&dir);
    // The first evaluate after opening, as `serve` and the Player run it.
    assert_eq!(codes(&mut session), vec![-1]);
    let metrics = session.output(Some("metrics")).unwrap();
    assert_eq!(session.field(metrics).unwrap().values()[0], 1000);
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&elsewhere);
}

#[test]
fn a_font_that_cannot_be_read_still_gives_the_missing_glyph() {
    let dir = scratch("unread");
    std::fs::write(dir.join("fonts/sample.ttf"), b"not a font").unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(
        said.iter().any(|line| line.contains("rejected")),
        "{said:?}"
    );
    assert_eq!(codes(&mut session), vec![-1]);
    assert_eq!(codes(&mut session), vec![-1]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_check_counts_a_font_nodes_codes_as_read() {
    let dir = scratch("check");
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    assert!(
        checked
            .lines
            .iter()
            .all(|(_, line)| !line.contains("nothing reads")),
        "{:?}",
        checked.lines
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_replay_reads_the_codes_its_frames_do_not_ask_for() {
    let dir = scratch("replay");
    let text = "scene scene.esn\nframe 0.0\nframe 0.05\nexpect copy.result[0].code = 111\nexpect copy.result[1].code = -1\nexpect copy.result[2].code = 73\n";
    let replay = entelechy_session::replay::parse(text, &dir).unwrap();
    let runtime = Session::runtime_for("cpu", entelechy_session::Limits::default()).unwrap();
    let outcome = entelechy_session::replay::run(&replay, runtime).unwrap();
    assert_eq!(
        (outcome.checks, outcome.failures.len()),
        (3, 0),
        "{outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_players_frames_give_the_glyphs_a_frame_later() {
    use entelechy_session::play::{Frame, Play, View};
    let dir = scratch("play");
    let mut session = open(&dir);
    let mut play = Play::new(|| 0.0);
    let view = View {
        window: [8.0, 4.0],
        scale: 1.0,
        rate: 0.0,
        queued: 0,
    };
    for _ in 0..2 {
        assert!(
            matches!(play.frame(&mut session, view), Frame::Shown(_)),
            "{}",
            play.status()
        );
    }
    let handle = session.output(Some("result")).unwrap();
    let words = session.field(handle).unwrap().values().to_vec();
    let codes: Vec<i32> = words.chunks(8).map(|row| row[0] as i32).collect();
    assert_eq!(codes, vec![111, -1, 73]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_session_started_over_gives_the_glyphs_of_no_codes_until_it_is_given_codes_again() {
    let dir = scratch("over");
    let mut session = open(&dir);
    codes(&mut session);
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    session.start_over();
    assert_eq!(codes(&mut session), vec![-1]);
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_font_through_a_package_name_is_read_from_the_package() {
    let dir = scratch("package");
    // The font only in a package kept elsewhere, named in the scene by the
    // package's name, as an input's file is.
    let packages = dir.join("packages");
    std::fs::create_dir_all(packages.join("glyphs.entpkg/fonts")).unwrap();
    std::fs::write(
        packages.join("glyphs.entpkg/package.esn"),
        "package glyphs@1 { language = 1; ir = 1; }\n",
    )
    .unwrap();
    std::fs::rename(
        dir.join("fonts/sample.ttf"),
        packages.join("glyphs.entpkg/fonts/sample.ttf"),
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("fonts/sample.ttf", "glyphs.entpkg/fonts/sample.ttf"),
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
    let said: Vec<String> = events.iter().map(ToString::to_string).collect();
    assert!(
        !said.iter().any(|line| line.contains("rejected")),
        "{said:?}"
    );
    codes(&mut session);
    assert_eq!(codes(&mut session), vec![111, -1, 73]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A store that counts the times `sample.ttf` is read.
#[derive(Default)]
struct Counted {
    files: entelechy_documents::store::Memory,
    fonts_read: std::sync::atomic::AtomicUsize,
}

impl entelechy_documents::store::Store for Counted {
    fn read(&self, path: &std::path::Path) -> std::io::Result<Vec<u8>> {
        if path.ends_with("sample.ttf") {
            self.fonts_read
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
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
    fn kind(&self, path: &std::path::Path) -> Option<entelechy_documents::store::Kind> {
        self.files.kind(path)
    }
    fn list(&self, dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
        self.files.list(dir)
    }
    fn stamp(&self, path: &std::path::Path) -> Option<entelechy_documents::store::Stamp> {
        self.files.stamp(path)
    }
    fn canonical(&self, path: &std::path::Path) -> std::io::Result<PathBuf> {
        self.files.canonical(path)
    }
    fn whole(&self, path: &std::path::Path) -> PathBuf {
        self.files.whole(path)
    }
}

/// Two font nodes of one file hold it once, each giving the glyphs of its
/// own codes; when the file changes, both read the new one, once.
#[test]
fn font_nodes_of_one_file_read_it_once() {
    let scene = "
struct Letterform { code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2; }
kernel t.codes@1 { requires { field.write; } param code: Int = 0; output result: Rows<Int>;
    compute { return code; } }
kernel t.copy@1 { requires { field.read; field.write; } input glyphs: Rows<Letterform>; output result: Rows<Letterform>;
    compute { return sample(glyphs); } }
node i: t.codes@1 { code = 73; domain = 1; }
node o: t.codes@1 { code = 111; domain = 1; }
node a: font \"/s/sample.ttf\" { codes <- i.result; }
node b: font \"/s/sample.ttf\" { codes <- o.result; }
node first: t.copy@1 { glyphs <- a.glyphs; }
node second: t.copy@1 { glyphs <- b.glyphs; }
output result <- first.result;
output other <- second.result;
";
    let store = std::sync::Arc::new(Counted::default());
    store
        .files
        .put(std::path::Path::new("/s/scene.esn"), scene.into());
    let font = std::path::Path::new("/s/sample.ttf");
    store
        .files
        .put(font, entelechy_import::font::sample::latin());
    let mut session = Session::open_in(
        store.clone(),
        std::path::Path::new("/s/scene.esn"),
        None,
        entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(entelechy_cpu::CpuBackend)),
    );
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    let reads = || store.fonts_read.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(reads(), 1);
    let code = |session: &mut Session, output: &str| {
        let handle = session.output(Some(output)).unwrap();
        session.field(handle).unwrap().values()[0] as i32
    };
    session.evaluate(&Context::default()).unwrap();
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(code(&mut session, "result"), 73);
    assert_eq!(code(&mut session, "other"), 111);
    assert_eq!(reads(), 1, "glyphs of other codes read no file");

    store
        .files
        .put(font, entelechy_import::font::sample::latin());
    session.sync();
    assert_eq!(reads(), 2);
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(code(&mut session, "result"), 73);
    assert_eq!(code(&mut session, "other"), 111);
}
