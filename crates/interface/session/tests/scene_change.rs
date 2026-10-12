//! A scene change is one change: the Kernels of its documents and the
//! graph they make publish together or not at all. The running scene
//! keeps its Kernels until every Kernel of its replacement stands.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_kernel::{Context, KernelId};
use entelechy_session::{Event, Memory, Session};

/// The root: `k.src` makes a picture under the output `port`, read by a
/// placed graph.
fn root(port: &str) -> String {
    format!(
        "node src: kernel k.src@1 {{ requires {{ field.write; }} output {port}: Image; domain = vec2(4, 4); compute {{ return vec4(0.5, 0.25, 0.75, 1.0); }} }}\n\
         node shown: graph \"sub.esn\" {{ source <- src.{port}; }}\n\
         output result <- shown.result;\n"
    )
}

/// The placed graph, its one Kernel's body `body`.
fn sub(body: &str) -> String {
    format!(
        "input source: Image;\n\
         node m: kernel k.m@1 {{ requires {{ field.read; field.write; }} input source: Image <- source; output result: Image; compute {{ {body} }} }}\n\
         output result <- m.result;\n"
    )
}

fn open(files: &Arc<Memory>, runtime: entelechy_kernel::Runtime) -> Session {
    let store: Arc<dyn entelechy_session::Store> = Arc::clone(files) as _;
    Session::open_in(store, Path::new("/s/scene.esn"), None, runtime)
}

fn put(files: &Memory, name: &str, text: &str) {
    files.put(&PathBuf::from("/s").join(name), text.as_bytes().to_vec());
}

fn generation(session: &Session, id: &str) -> Option<u64> {
    let id: KernelId = id.parse().unwrap();
    session.runtime().kernel(&id).map(|v| v.generation().0)
}

/// One refused document refuses the change whole: the root's Kernel is
/// not published, and the running scene runs on.
#[test]
fn a_refused_document_publishes_none_of_the_scenes_change() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    let before = generation(&session, "k.src@1");

    // The root renames its output and rewires; the placed graph's Kernel
    // fails to compile.
    put(&files, "scene.esn", &root("out"));
    put(&files, "sub.esn", &sub("return nothing;"));
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneRejected(_))),
        "{events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, Event::Published { .. })),
        "{events:?}"
    );
    assert_eq!(generation(&session, "k.src@1"), before);
    session
        .evaluate(&Context::default())
        .expect("the scene running runs on");

    // Mended, the change is taken whole.
    put(&files, "sub.esn", &sub("return sample(source) * 0.5;"));
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneLoaded)),
        "{events:?}"
    );
    assert!(generation(&session, "k.src@1") > before);
    session.evaluate(&Context::default()).unwrap();
}

/// With off-frame preparation, the new graph waits for its Kernels: every
/// frame runs one whole scene, old or new.
#[test]
fn a_scene_prepared_off_the_frame_goes_live_with_its_kernels() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let runtime = entelechy_kernel::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();

    put(&files, "scene.esn", &root("out"));
    let mut loaded = false;
    for _ in 0..5_000 {
        let events = session.sync();
        session
            .evaluate(&Context::default())
            .expect("every frame runs a whole scene");
        if events.iter().any(|e| matches!(e, Event::SceneLoaded)) {
            loaded = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(loaded, "the change lands");
    session.evaluate(&Context::default()).unwrap();
}

/// The interpreter with a slow prepare, so an off-frame replacement is
/// still preparing when its sync returns; it refuses to prepare any
/// Kernel with an output named `refused`.
struct Deliberate;

impl entelechy_kernel::Backend for Deliberate {
    fn name(&self) -> &'static str {
        "deliberate"
    }

    fn prepare(
        &self,
        module: &entelechy_kernel::Validated,
    ) -> Result<Box<dyn entelechy_kernel::Prepared>, String> {
        std::thread::sleep(std::time::Duration::from_millis(20));
        if module
            .manifest
            .outputs
            .iter()
            .any(|port| port.name == "refused")
        {
            return Err("this device refuses it".into());
        }
        entelechy_kernel::Backend::prepare(&entelechy_cpu::CpuBackend, module)
    }

    fn execute(
        &self,
        prepared: &dyn entelechy_kernel::Prepared,
        module: &entelechy_kernel::Validated,
        invocation: &entelechy_kernel::Invocation,
        scope: &mut entelechy_kernel::Scope<'_>,
    ) -> Result<Vec<entelechy_kernel::Handle>, entelechy_kernel::ExecutionError> {
        entelechy_kernel::Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
}

/// A replacement refused off the frame refuses the scene needing it: the
/// running scene stays, with its Kernels.
#[test]
fn a_scene_whose_kernel_fails_off_the_frame_is_refused_and_the_old_one_runs() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let runtime = entelechy_kernel::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();
    let before = generation(&session, "k.src@1");

    put(&files, "scene.esn", &root("refused"));
    let mut refused = false;
    for _ in 0..5_000 {
        let events = session.sync();
        assert!(
            !events.iter().any(|e| matches!(e, Event::SceneLoaded)),
            "{events:?}"
        );
        session
            .evaluate(&Context::default())
            .expect("the scene running runs on");
        if events.iter().any(|e| matches!(e, Event::SceneRejected(_))) {
            refused = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(refused, "the change is refused");
    assert_eq!(generation(&session, "k.src@1"), before);
    session.evaluate(&Context::default()).unwrap();
}

/// A scene a request claimed is the request's in the history, though its
/// Kernels were prepared off the frame and it landed on a later sync.
#[test]
fn a_claimed_scene_landing_later_is_the_requests() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let runtime = entelechy_kernel::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    let (since, _) = session.position();

    let asked = entelechy_session::Origin::Protocol("socket 1".into());
    let before = session.act_for(asked.clone());
    put(&files, "scene.esn", &root("out"));
    session.claim_scene();
    let mut landed = session
        .sync()
        .iter()
        .any(|e| matches!(e, Event::SceneLoaded));
    session.act_for(before);
    for _ in 0..5_000 {
        if landed {
            break;
        }
        landed = session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded));
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(landed);
    let scene = session
        .changes_since(since)
        .into_iter()
        .rfind(|record| matches!(record.change, entelechy_session::Change::Scene { .. }))
        .expect("the scene is recorded");
    assert_eq!(scene.origin, asked);
}

/// A graph param set while a scene waits for its Kernels is judged by
/// that newest-read scene, whatever the worker's progress.
#[test]
fn a_param_is_judged_by_the_newest_scene_read() {
    let scene = |param: &str, red: &str| {
        format!(
            "param {param}: Float = 0.5;\n\
             kernel k.src@1 {{ requires {{ field.write; }} param level: Float = 0.5; output result: Image; compute {{ return vec4({red}, level, 0.0, 1.0); }} }}\n\
             node src: k.src@1 {{ level = {param}; domain = vec2(4, 4); }}\n\
             output result <- src.result;\n"
        )
    };
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &scene("speed", "0.25"));
    let runtime = entelechy_kernel::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();

    // The new document declares `zoom`, not `speed`, and its Kernel
    // changes, so it waits for the worker.
    put(&files, "scene.esn", &scene("zoom", "0.75"));
    session.sync();
    assert!(
        session
            .set_param("speed", entelechy_kernel::Value::Float(2.0))
            .is_err()
    );
    session
        .set_param("zoom", entelechy_kernel::Value::Float(5.0))
        .expect("the newest scene declares it");
    for _ in 0..5_000 {
        if session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded))
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        session.params().get("zoom"),
        Some(&entelechy_kernel::Value::Float(5.0)),
        "the value set is the scene's once it runs"
    );
    assert!(session.params().get("speed").is_none());
}

/// A session opened on `best` runs its scene on the first backend that
/// opens and takes it; if none does, it is refused naming every backend
/// passed over.
#[test]
fn best_opens_where_the_scene_loads_and_says_why_when_it_loads_nowhere() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let store: Arc<dyn entelechy_session::Store> = Arc::clone(&files) as _;
    let open = |backend: &str| {
        Session::open_loaded(
            Arc::clone(&store),
            Path::new("/s/scene.esn"),
            None,
            backend,
            Default::default(),
            entelechy_gpu::Wait::blocking(),
            true,
        )
    };
    let mut loaded = open("best").map_err(|e| e.to_string()).unwrap();
    assert!(
        loaded
            .events
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded))
    );
    loaded.session.evaluate(&Context::default()).unwrap();

    put(&files, "sub.esn", &sub("return nothing;"));
    for backend in ["best", "cpu"] {
        let Err(why) = open(backend) else {
            panic!("{backend}: a scene that loads nowhere opens");
        };
        assert!(why.contains("nothing"), "{backend}: {why}");
    }
}

/// A claimed scene superseded by a document edit on disk before it lands
/// is the document's in the history, not the request's.
#[test]
fn a_claimed_scene_superseded_on_disk_is_the_documents() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    let runtime = entelechy_kernel::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    let (since, _) = session.position();

    let asked = entelechy_session::Origin::Protocol("socket 1".into());
    let before = session.act_for(asked.clone());
    put(&files, "scene.esn", &root("out"));
    session.claim_scene();
    session.sync();
    session.act_for(before);
    assert!(
        session.scene_waiting(),
        "the claimed scene waits for its Kernel"
    );
    // A person's edit on disk replaces it before it lands.
    put(&files, "scene.esn", &root("again"));
    let mut landed = false;
    for _ in 0..5_000 {
        if session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded))
        {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(landed);
    let scene = session
        .changes_since(since)
        .into_iter()
        .rfind(|record| matches!(record.change, entelechy_session::Change::Scene { .. }))
        .expect("the scene is recorded");
    assert_eq!(
        scene.origin,
        entelechy_session::Origin::Document(PathBuf::from("/s/scene.esn"))
    );
}

/// A library's Kernels recompile when a function it declares changes, and
/// when a function of a library it uses changes; the Kernels that call
/// the changed function run its new body.
#[test]
fn a_changed_function_republishes_the_kernels_that_call_it() {
    let files = Arc::new(Memory::default());
    let base = |gain: &str| format!("fn gain(x: Float) -> Float {{ return x * {gain}; }}\n");
    let mid = "use \"base.ent\";\n\
               fn twice(x: Float) -> Float { return x * 2.0; }\n\
               kernel l.own@1 { requires { field.write; } output result: Rows<Float>; compute { return twice(1.0); } }\n\
               kernel l.used@1 { requires { field.write; } output result: Rows<Float>; compute { return gain(1.0); } }\n";
    put(&files, "base.ent", &base("3.0"));
    put(&files, "mid.ent", mid);
    put(
        &files,
        "scene.esn",
        "use \"mid.ent\";\nnode a: l.own@1 { domain = 1; }\nnode b: l.used@1 { domain = 1; }\noutput a <- a.result;\noutput b <- b.result;\n",
    );
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    let value = |session: &mut Session, node: &str| {
        let report = session.evaluate(&Context::default()).unwrap();
        let handle = report
            .outputs
            .iter()
            .find(|(name, _)| name == node)
            .map(|(_, h)| *h)
            .unwrap();
        f32::from_bits(session.field(handle).unwrap().values()[0])
    };
    assert_eq!(
        (value(&mut session, "a"), value(&mut session, "b")),
        (2.0, 3.0)
    );

    // Its own function changes.
    let before = generation(&session, "l.own@1");
    put(&files, "mid.ent", &mid.replace("x * 2.0", "x * 7.0"));
    session.sync();
    assert!(generation(&session, "l.own@1") > before);
    assert_eq!(value(&mut session, "a"), 7.0);

    // A function of the library it uses changes.
    put(&files, "base.ent", &base("5.0"));
    session.sync();
    assert_eq!(value(&mut session, "b"), 5.0);
}

/// A changed function republishes only the Kernels whose IR it changes: a
/// Kernel that does not reach it keeps its version, and a change to a
/// Kernel's own text, its doc included, republishes it.
#[test]
fn a_changed_function_republishes_only_the_kernels_it_reaches() {
    let files = Arc::new(Memory::default());
    let lib = |gain: &str, doc: &str| {
        format!(
            "fn gain(x: Float) -> Float {{ return x * {gain}; }}\n\
             /// {doc}\n\
             kernel l.calls@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return gain(1.0); }} }}\n\
             kernel l.apart@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 4.0; }} }}\n"
        )
    };
    put(&files, "lib.ent", &lib("2.0", "Calls."));
    put(
        &files,
        "scene.esn",
        "use \"lib.ent\";\nnode a: l.calls@1 { domain = 1; }\nnode b: l.apart@1 { domain = 1; }\noutput a <- a.result;\noutput b <- b.result;\n",
    );
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    let (calls, apart) = (
        generation(&session, "l.calls@1"),
        generation(&session, "l.apart@1"),
    );

    put(&files, "lib.ent", &lib("3.0", "Calls."));
    session.sync();
    assert!(generation(&session, "l.calls@1") > calls);
    assert_eq!(generation(&session, "l.apart@1"), apart);

    // Its doc is its text.
    let calls = generation(&session, "l.calls@1");
    put(&files, "lib.ent", &lib("3.0", "Calls gain."));
    session.sync();
    assert!(generation(&session, "l.calls@1") > calls);
    assert_eq!(generation(&session, "l.apart@1"), apart);
}

fn refusals(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Event::SceneRejected(_)))
        .count()
}

/// With no scene running, an unchanged root refused for the same reasons
/// is said once, however many syncs read it again; a change says again.
#[test]
fn an_unchanged_refusal_is_said_once() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", "");
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    assert_eq!(refusals(&session.sync()), 1);
    assert_eq!(refusals(&session.sync()), 0);
    assert_eq!(refusals(&session.sync()), 0);

    // Written again, still empty: a change, said again.
    put(&files, "scene.esn", "// nothing yet\n");
    assert_eq!(refusals(&session.sync()), 1);
    assert_eq!(refusals(&session.sync()), 0);

    put(&files, "scene.esn", &root("result"));
    put(&files, "sub.esn", &sub("return sample(source);"));
    session.sync();
    assert!(session.graph().is_some());
}

/// A graph that appears under an unchanged root is found: the root is
/// read again every sync while it is refused, only not said again.
#[test]
fn a_graph_appearing_under_a_refused_root_loads_it() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.esn", &root("result"));
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    let first = session.sync();
    assert_eq!(refusals(&first), 1, "{first:?}");
    let again = session.sync();
    assert_eq!(refusals(&again), 0, "{again:?}");
    assert!(session.graph().is_none());

    put(&files, "sub.esn", &sub("return sample(source);"));
    let found = session.sync();
    assert!(session.graph().is_some(), "{found:?}");
}

/// A library refused while the scene runs is said when it is read, not
/// every sync after.
#[test]
fn a_refused_library_is_said_once() {
    let files = Arc::new(Memory::default());
    put(
        &files,
        "scene.esn",
        "use \"lib.ent\";\nnode n: kernel { requires { field.write; } output result: Image; domain = vec2(2, 2); compute { return vec4(1.0, 1.0, 1.0, 1.0); } }\noutput result <- n.result;\n",
    );
    put(&files, "lib.ent", "fn tint() -> Vec4 { return nothing; }\n");
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    let again = session.sync();
    assert!(again.is_empty(), "{again:?}");
}

/// Each document a scene loads is remembered at the version it loaded,
/// whatever changed it: a placed graph rewired on disk and a library of
/// functions alone are read back by path and version.
#[test]
fn every_document_a_scene_loads_is_remembered_by_its_version() {
    let files = Arc::new(Memory::default());
    let base = |gain: &str| format!("fn gain(x: Float) -> Float {{ return x * {gain}; }}\n");
    let wired = |by: &str| {
        format!(
            "use \"base.ent\";\ninput source: Image;\n\
             node m: kernel k.m@1 {{ requires {{ field.read; field.write; }} input source: Image <- source; output result: Image; compute {{ return sample(source) * gain(1.0); }} }}\n\
             output result <- {by}.result;\n"
        )
    };
    put(&files, "base.ent", &base("1.0"));
    put(&files, "sub.esn", &wired("m"));
    put(&files, "scene.esn", &root("result"));
    let mut session = open(
        &files,
        entelechy_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    put(&files, "base.ent", &base("2.0"));
    session.sync();
    // The graph's output is rewired through a node of its own; no Kernel
    // of it changes.
    let rewired = wired("n").replace(
        "output result <-",
        "node n: kernel k.n@1 { requires { field.read; field.write; } input source: Image <- m.result; output result: Image; compute { return sample(source); } }\noutput result <-",
    );
    put(&files, "sub.esn", &rewired);
    session.sync();
    let version = |text: &str| entelechy_session::history::digest(text);
    for (path, text) in [
        ("base.ent", base("1.0")),
        ("base.ent", base("2.0")),
        ("sub.esn", wired("m")),
        ("sub.esn", rewired.clone()),
    ] {
        assert_eq!(
            session.document_at(Path::new(path), version(&text)),
            Ok(text),
            "{path}"
        );
    }
    // A version is its own document's, and is recorded once.
    assert!(
        session
            .document_at(Path::new("sub.esn"), version(&base("1.0")))
            .is_err()
    );
    let named = session
        .changes_since(0)
        .into_iter()
        .filter_map(|record| match record.change {
            entelechy_session::Change::Scene { documents, .. } => Some(documents),
            _ => None,
        })
        .flatten()
        .filter(|(path, _)| path.ends_with("base.ent"))
        .count();
    assert_eq!(named, 2);
}
