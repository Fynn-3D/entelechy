//! A scene's change is one change: the Kernels of its own documents, the
//! root and the graphs it places, and the graph they make publish
//! together or not at all. The scene running keeps running, on the
//! Kernels it ran on, until every Kernel of the change it is replaced by
//! stands.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_runtime::{Context, KernelId};
use orla_session::{Event, Memory, Session};

/// The root: `k.src` makes a picture under the output `port`, and a
/// placed graph reads it.
fn root(port: &str) -> String {
    format!(
        "node src: kernel k.src@1 {{ requires {{ field.write; }} output {port}: Image; domain = vec2(4, 4); compute {{ return vec4(0.5, 0.25, 0.75, 1.0); }} }}\n\
         node shown: graph \"sub.osn\" {{ source <- src.{port}; }}\n\
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

fn open(files: &Arc<Memory>, runtime: orla_runtime::Runtime) -> Session {
    let store: Arc<dyn orla_session::Store> = Arc::clone(files) as _;
    Session::open_in(store, Path::new("/s/scene.osn"), None, runtime)
}

fn put(files: &Memory, name: &str, text: &str) {
    files.put(&PathBuf::from("/s").join(name), text.as_bytes().to_vec());
}

fn generation(session: &Session, id: &str) -> Option<u64> {
    let id: KernelId = id.parse().unwrap();
    session.runtime().kernel(&id).map(|v| v.generation().0)
}

/// One of the scene's documents refused refuses the change whole: the
/// root's Kernel it would have published is not, and the scene running
/// runs on as it did. (The root's Kernel used to publish anyway, and the
/// graph kept running asked it for an output it no longer had.)
#[test]
fn a_refused_document_publishes_none_of_the_scenes_change() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.osn", &root("result"));
    put(&files, "sub.osn", &sub("return sample(source);"));
    let mut session = open(
        &files,
        orla_session::Session::runtime_for("cpu", Default::default()).unwrap(),
    );
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    let before = generation(&session, "k.src@1");

    // The root renames its output and rewires; the placed graph's Kernel
    // no longer compiles.
    put(&files, "scene.osn", &root("out"));
    put(&files, "sub.osn", &sub("return nothing;"));
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
    put(&files, "sub.osn", &sub("return sample(source) * 0.5;"));
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneLoaded)),
        "{events:?}"
    );
    assert!(generation(&session, "k.src@1") > before);
    session.evaluate(&Context::default()).unwrap();
}

/// Where replacements are prepared off the frame, the new graph waits for
/// its Kernels: every frame runs one whole scene, the old on its Kernels
/// or the new on its own, never the new graph on the old Kernels.
#[test]
fn a_scene_prepared_off_the_frame_goes_live_with_its_kernels() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.osn", &root("result"));
    put(&files, "sub.osn", &sub("return sample(source);"));
    let runtime = orla_runtime::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();

    put(&files, "scene.osn", &root("out"));
    let mut loaded = false;
    for _ in 0..10_000 {
        let events = session.sync();
        session
            .evaluate(&Context::default())
            .expect("every frame runs a whole scene");
        if events.iter().any(|e| matches!(e, Event::SceneLoaded)) {
            loaded = true;
            break;
        }
        std::thread::yield_now();
    }
    assert!(loaded, "the change lands");
    session.evaluate(&Context::default()).unwrap();
}

/// The interpreter, slow to prepare, so that a replacement prepared off
/// the frame is still being prepared when the sync that asked for it
/// returns; and refusing any Kernel with an output named `refused`, a
/// replacement that validates and fails to prepare.
struct Deliberate;

impl orla_runtime::Backend for Deliberate {
    fn name(&self) -> &'static str {
        "deliberate"
    }

    fn prepare(
        &self,
        module: &orla_runtime::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        std::thread::sleep(std::time::Duration::from_millis(20));
        if module
            .manifest
            .outputs
            .iter()
            .any(|port| port.name == "refused")
        {
            return Err("this device refuses it".into());
        }
        orla_runtime::Backend::prepare(&orla_cpu::CpuBackend, module)
    }

    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_runtime::Validated,
        invocation: &orla_runtime::Invocation,
        scope: &mut orla_runtime::Scope<'_>,
    ) -> Result<Vec<orla_runtime::Handle>, orla_runtime::ExecutionError> {
        orla_runtime::Backend::execute(&orla_cpu::CpuBackend, prepared, module, invocation, scope)
    }
}

/// A replacement refused off the frame refuses the scene that needs it:
/// the scene running stays, with the Kernels it runs on. (The new graph
/// used to be installed before its Kernels, and stayed when one failed.)
#[test]
fn a_scene_whose_kernel_fails_off_the_frame_is_refused_and_the_old_one_runs() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.osn", &root("result"));
    put(&files, "sub.osn", &sub("return sample(source);"));
    let runtime = orla_runtime::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();
    let before = generation(&session, "k.src@1");

    put(&files, "scene.osn", &root("refused"));
    let mut refused = false;
    for _ in 0..10_000 {
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
        std::thread::yield_now();
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
    put(&files, "scene.osn", &root("result"));
    put(&files, "sub.osn", &sub("return sample(source);"));
    let runtime = orla_runtime::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    let (since, _) = session.position();

    let asked = orla_session::Origin::Protocol("socket 1".into());
    let before = session.act_for(asked.clone());
    put(&files, "scene.osn", &root("out"));
    session.claim_scene();
    let mut landed = session
        .sync()
        .iter()
        .any(|e| matches!(e, Event::SceneLoaded));
    session.act_for(before);
    for _ in 0..10_000 {
        if landed {
            break;
        }
        landed = session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded));
        std::thread::yield_now();
    }
    assert!(landed);
    let scene = session
        .changes_since(since)
        .into_iter()
        .rfind(|record| matches!(record.change, orla_session::Change::Scene { .. }))
        .expect("the scene is recorded");
    assert_eq!(scene.origin, asked);
}

/// A graph parameter set while a scene waits for its Kernels is judged
/// by that scene, the newest the session has read, not by whichever
/// runs at that moment: the same request has one answer however far the
/// worker has got. (A param the new document dropped was taken and then
/// silently lost, and one it declared refused.)
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
    put(&files, "scene.osn", &scene("speed", "0.25"));
    let runtime = orla_runtime::Runtime::with_backend(1 << 20, Box::new(Deliberate));
    let mut session = open(&files, runtime);
    session.sync();
    session.set_background(true);
    session.evaluate(&Context::default()).unwrap();

    // The new document declares `zoom`, not `speed`, and its Kernel
    // changes, so it waits for the worker.
    put(&files, "scene.osn", &scene("zoom", "0.75"));
    session.sync();
    assert!(
        session
            .set_param("speed", orla_runtime::Value::Float(2.0))
            .is_err()
    );
    session
        .set_param("zoom", orla_runtime::Value::Float(5.0))
        .expect("the newest scene declares it");
    for _ in 0..10_000 {
        if session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded))
        {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(
        session.params().get("zoom"),
        Some(&orla_runtime::Value::Float(5.0)),
        "the value set is the scene's once it runs"
    );
    assert!(session.params().get("speed").is_none());
}

/// A session opened on `best` reads its scene on the first backend that
/// opens and takes it, and one it takes on none is refused with why;
/// every backend passed over is said.
#[test]
fn best_opens_where_the_scene_loads_and_says_why_when_it_loads_nowhere() {
    let files = Arc::new(Memory::default());
    put(&files, "scene.osn", &root("result"));
    put(&files, "sub.osn", &sub("return sample(source);"));
    let store: Arc<dyn orla_session::Store> = Arc::clone(&files) as _;
    let open = |backend: &str| {
        Session::open_loaded(
            Arc::clone(&store),
            Path::new("/s/scene.osn"),
            None,
            backend,
            Default::default(),
            orla_gpu::Wait::blocking(),
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

    put(&files, "sub.osn", &sub("return nothing;"));
    for backend in ["best", "cpu"] {
        let Err(why) = open(backend) else {
            panic!("{backend}: a scene that loads nowhere opens");
        };
        assert!(why.contains("nothing"), "{backend}: {why}");
    }
}
