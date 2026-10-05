//! A Kernel the main backend found in preparing it that it cannot run
//! (a device's own compiler failing on a valid shader) runs on the ordered
//! backend, said once as any Kernel off the main one is; a backend that
//! fails without declining, its own fault, still refuses the document.

use std::sync::Mutex;

use orla_abi::{Handle, Invocation, Scope};
use orla_ir::Validated;
use orla_kernel::{Backend, ExecutionError, Prepared, Runtime};
use orla_session::{Event, Session};

/// The interpreter as a device whose compiler fails on Kernels named
/// `big`; with `remember`, it declines them from then on, as the GPU
/// backend does.
struct Device {
    remember: bool,
    unbuilt: Mutex<Vec<Validated>>,
}

impl Backend for Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        let unbuilt = self.unbuilt.lock().unwrap();
        unbuilt
            .iter()
            .any(|seen| seen == module)
            .then(|| "the device's own compiler could not build it (it crashed)".to_owned())
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        if module.manifest.id.to_string().contains("big") {
            if self.remember {
                self.unbuilt.lock().unwrap().push(module.clone());
            }
            return Err("the compiler crashed".to_owned());
        }
        orla_cpu::CpuBackend.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        Backend::execute(&orla_cpu::CpuBackend, prepared, module, invocation, scope)
    }
}

fn scene(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-unbuilt-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "kernel t.small@1 { requires { field.write; } output result: Rows<Float>; compute { return 2.0; } }
kernel t.big@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return at(x, 0) * 3.0; } }
node small: t.small@1 { domain = 1; }
node big: t.big@1 { x <- small.result; domain = 1; }
output result <- big.result;
",
    )
    .unwrap();
    dir
}

fn session(dir: &std::path::Path, remember: bool) -> Session {
    let device = Device {
        remember,
        unbuilt: Mutex::new(Vec::new()),
    };
    let runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    Session::open_with(&dir.join("scene.osn"), None, runtime)
}

#[test]
fn a_kernel_the_device_could_not_build_runs_on_the_ordered_backend_said_once() {
    let dir = scene("routed");
    let mut session = session(&dir, true);
    let mut events = session.sync();
    assert!(
        !events.iter().any(|e| matches!(e, Event::Rejected { .. })),
        "{events:?}"
    );
    session.evaluate(&orla_kernel::Context::default()).unwrap();
    events.extend(session.sync());
    let routed: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Routed { node, why, .. } => Some((node.clone(), why.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(routed.len(), 1, "{events:?}");
    assert_eq!(routed[0].0, "big");
    assert!(routed[0].1.contains("could not build"), "{routed:?}");
    let handle = session
        .cached(&orla_graph::NodeId("big".into()), "result")
        .unwrap();
    assert_eq!(
        f32::from_bits(session.field(handle).unwrap().values()[0]),
        6.0
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_backend_failing_without_declining_still_refuses_the_document() {
    let dir = scene("refused");
    let mut session = session(&dir, false);
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::Rejected { error, .. } if error.message.contains("the compiler crashed"))),
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
