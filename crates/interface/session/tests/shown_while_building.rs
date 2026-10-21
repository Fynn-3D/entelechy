//! What only shows is shown from the ordered backend while the device's
//! compiler builds it: a node reached by no state and no output but the
//! picture and the sound. A session that shows while building waits for
//! the device only on what state or a kept output reads; the rest runs
//! on the ordered backend until the device has built it, and on the
//! device from then on. Off, nothing changes where it runs.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_graph::NodeId;
use entelechy_ir::Validated;
use entelechy_kernel::{Backend, Context, ExecutionError, Prepared, Runtime};
use entelechy_session::Session;

const SCENE: &str = "kernel t.zero@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }
kernel t.pace@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }
kernel t.step@1 { requires { field.read; field.write; } input previous: Rows<Float>; input pace: Rows<Float>; output result: Rows<Float>; compute { return at(previous, 0) + at(pace, 0); } }
kernel t.paint@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Image; compute { return vec4(at(x, 0) / 10.0, 0.0, 0.0, 1.0); } }
kernel t.tally@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Int>; compute { return int(at(x, 0)); } }
kernel t.tone@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.25; } }
node zero: t.zero@1 { domain = 1; }
node pace: t.pace@1 { domain = 1; }
node count: t.step@1 { previous <~ count.result from zero.result; pace <- pace.result; domain = 1; }
node paint: t.paint@1 { x <- count.result; domain = vec2(2.0, 2.0); }
node tally: t.tally@1 { x <- count.result; domain = 1; }
node tone: t.tone@1 { domain = 64; }
output result <- paint.result;
output sound <- tone.result;
output kept <- tally.result;
";

fn scene(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-shown-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

/// The interpreter as a device that builds on first need, whose builds
/// of Kernels named `paint` or `tone` wait until it is let go.
#[derive(Default)]
struct Device {
    held: Mutex<bool>,
    let_go: Condvar,
}

struct Lazy {
    inner: Box<dyn Prepared>,
    id: String,
    built: OnceLock<()>,
}

impl Prepared for Lazy {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Device {
    fn holding() -> &'static Device {
        let device: &'static Device = Box::leak(Box::default());
        *device.held.lock().unwrap() = true;
        device
    }

    fn release(&self) {
        *self.held.lock().unwrap() = false;
        self.let_go.notify_all();
    }

    fn build(&self, prepared: &dyn Prepared) {
        let lazy = prepared.as_any().downcast_ref::<Lazy>().unwrap();
        lazy.built.get_or_init(|| {
            if lazy.id.contains("paint") || lazy.id.contains("tone") {
                let mut held = self.held.lock().unwrap();
                while *held {
                    held = self.let_go.wait(held).unwrap();
                }
            }
        });
    }
}

impl Backend for &'static Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn has_device(&self) -> bool {
        true
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        Ok(Box::new(Lazy {
            inner: entelechy_cpu::CpuBackend.prepare(module)?,
            id: module.manifest.id.to_string(),
            built: OnceLock::new(),
        }))
    }
    fn finish(&self, prepared: &dyn Prepared) {
        self.build(prepared);
    }
    fn finished(&self, prepared: &dyn Prepared) -> bool {
        prepared
            .as_any()
            .downcast_ref::<Lazy>()
            .is_none_or(|lazy| lazy.built.get().is_some())
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.build(prepared);
        let inner = &prepared.as_any().downcast_ref::<Lazy>().unwrap().inner;
        Backend::execute(
            &entelechy_cpu::CpuBackend,
            inner.as_ref(),
            module,
            invocation,
            scope,
        )
    }
}

fn session_over(device: &'static Device, dir: &std::path::Path) -> Session {
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    // Every run on the device, however few its elements.
    runtime.set_small(0);
    Session::open_with(&dir.join("scene.esn"), None, runtime)
}

/// Where each node ran in the last evaluation's placements, by name.
fn placed(session: &Session, node: &str) -> (&'static str, Option<String>) {
    let (_, placement) = session
        .placements()
        .into_iter()
        .find(|(name, _)| name == node)
        .unwrap();
    (
        placement.backend,
        placement.routed.map(|why| why.to_string()),
    )
}

#[test]
fn only_what_reaches_the_picture_and_the_sound_alone_only_shows() {
    let dir = scene("graph");
    let device = Device::holding();
    device.release();
    let mut session = session_over(device, &dir);
    session.sync();
    let graph = session.graph().unwrap();
    let presented: BTreeSet<String> = ["result".to_owned(), "sound".to_owned()].into();
    let shown = graph.shown_only(&presented);
    let names: BTreeSet<&str> = shown.iter().map(|id| id.0.as_str()).collect();
    // `count` is state, `zero` its initial value and `pace` read by it;
    // `tally` is kept as an output nothing presents.
    assert_eq!(names, BTreeSet::from(["paint", "tone"]));
    // With nothing presented, everything an output reads is kept.
    let none = graph.shown_only(&BTreeSet::new());
    assert!(none.is_empty(), "{none:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_only_shows_runs_on_the_ordered_backend_until_the_device_built_it() {
    let dir = scene("building");
    let device = Device::holding();
    let mut session = session_over(device, &dir);
    session.show_while_building(true);
    // The sync waits only for what state and the kept output read: the
    // held builds go on beside it.
    session.sync();
    let paint = NodeId("paint".into());
    for frame in 1..=2 {
        session.evaluate(&Context::default()).unwrap();
        let handle = session.cached(&paint, "result").unwrap();
        let red = f32::from_bits(session.field(handle).unwrap().values()[0]);
        assert_eq!(red, frame as f32 / 10.0, "frame {frame}");
        let (backend, why) = placed(&session, "paint");
        assert_eq!(backend, "cpu");
        assert!(why.as_deref().unwrap().contains("builds it"), "{why:?}");
        assert_eq!(placed(&session, "tone").0, "cpu", "the sound only plays");
        assert_eq!(placed(&session, "count"), ("device", None), "state waits");
        assert_eq!(placed(&session, "tally"), ("device", None), "kept waits");
    }
    device.release();
    let deadline = Instant::now() + Duration::from_secs(10);
    while placed(&session, "paint").0 != "device" {
        assert!(Instant::now() < deadline, "the held build never finished");
        std::thread::sleep(Duration::from_millis(5));
    }
    session.evaluate(&Context::default()).unwrap();
    let handle = session.cached(&paint, "result").unwrap();
    let red = f32::from_bits(session.field(handle).unwrap().values()[0]);
    assert_eq!(red, 0.3);
    assert_eq!(placed(&session, "paint"), ("device", None));
    assert_eq!(placed(&session, "tone"), ("device", None));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn off_every_node_waits_for_the_device() {
    let dir = scene("off");
    let device = Device::holding();
    device.release();
    let mut session = session_over(device, &dir);
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    for node in ["zero", "pace", "count", "paint", "tally", "tone"] {
        assert_eq!(placed(&session, node), ("device", None), "{node}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Kernel that only shows but whose runs are so few that nothing builds
/// it on the device is not shown "while building": nothing builds it, and
/// its runs go where their size sends them.
#[test]
fn what_only_shows_but_nothing_builds_is_not_held_off_the_device() {
    let dir = scene("few");
    let device = Device::holding();
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    // Runs as small as these go to the ordered backend, so the device
    // never builds what only they run.
    runtime.set_small(entelechy_kernel::DEFAULT_SMALL);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.show_while_building(true);
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    for node in ["paint", "tone"] {
        assert!(
            !placed(&session, node)
                .1
                .unwrap_or_default()
                .contains("builds it"),
            "{node}"
        );
        let ran = report.placements.iter().find(|p| p.node.0 == node).unwrap();
        assert_eq!(ran.backend, "cpu", "{node}");
        let why = ran
            .routed
            .as_ref()
            .map(|why| why.to_string())
            .unwrap_or_default();
        assert!(!why.contains("builds it"), "{node}: {why}");
    }
    device.release();
    let _ = std::fs::remove_dir_all(&dir);
}
