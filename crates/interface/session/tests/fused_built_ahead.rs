//! A fused group is built on the device as it is prepared, where the
//! Kernel of the node it stands in for is built there: its first run
//! waits for no compiler, as the group's members' first runs did not.

use std::sync::{Mutex, OnceLock};

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_ir::Validated;
use entelechy_kernel::{Backend, Context, ExecutionError, Prepared, Runtime};
use entelechy_session::Session;

/// The interpreter as a device that builds on first need, saying whether
/// each build came from finishing or from a run.
#[derive(Default)]
struct Device {
    builds: Mutex<Vec<(String, &'static str)>>,
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
    fn build(&self, prepared: &dyn Prepared, by: &'static str) {
        let lazy = prepared.as_any().downcast_ref::<Lazy>().unwrap();
        lazy.built.get_or_init(|| {
            self.builds.lock().unwrap().push((lazy.id.clone(), by));
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
        self.build(prepared, "finish");
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
        self.build(prepared, "run");
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

#[test]
fn a_fused_group_is_built_as_it_is_prepared_not_by_its_first_run() {
    let dir = std::env::temp_dir().join(format!("entelechy-fused-ahead-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.ramp@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()); } }
kernel t.twice@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return sample(x) * 2.0; } }
node ramp: t.ramp@1 { domain = 1000; }
node twice: t.twice@1 { x <- ramp.result; }
output result <- twice.result;
",
    )
    .unwrap();
    let device: &'static Device = Box::leak(Box::default());
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    let mut fused = false;
    for _ in 0..4 {
        let report = session.evaluate(&Context::default()).unwrap();
        fused |= !report.fused.is_empty();
        session.sync();
    }
    assert!(fused, "the two nodes fuse");
    // The node's version and the group, each built once by finishing,
    // none by a run.
    let builds = device.builds.lock().unwrap().clone();
    let count = |name: &str| builds.iter().filter(|(id, _)| id == name).count();
    assert_eq!(count("t.twice@1"), 1, "{builds:?}");
    assert_eq!(count("fused.twice@1"), 1, "{builds:?}");
    assert!(builds.iter().all(|(_, by)| *by == "finish"), "{builds:?}");
    // What preparing shows names the group, not the node it stands in for.
    let shown: Vec<String> = session
        .progress()
        .now()
        .seeds
        .iter()
        .map(|seed| seed.id.to_string())
        .collect();
    assert!(shown.iter().any(|id| id == "fused.twice@1"), "{shown:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
