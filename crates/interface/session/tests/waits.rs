//! A frame names the nodes that kept its thread waiting for the device,
//! and how long each did: what the backend counts as waited while a node
//! is bound and run is that node's.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_kernel::{Backend, ExecutionError, Runtime};
use entelechy_session::Session;

/// A device that runs on the CPU and counts three milliseconds waited
/// for each run of a Kernel named `count`.
struct Device {
    waited: Arc<AtomicU64>,
    cpu: entelechy_cpu::CpuBackend,
}

impl Backend for Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn has_device(&self) -> bool {
        true
    }
    fn waited(&self) -> Duration {
        Duration::from_nanos(self.waited.load(Ordering::SeqCst))
    }
    fn prepare(
        &self,
        module: &entelechy_ir::Validated,
    ) -> Result<Box<dyn entelechy_kernel::Prepared>, String> {
        self.cpu.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn entelechy_kernel::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        if module.manifest.id.to_string().contains("count") {
            self.waited.fetch_add(3_000_000, Ordering::SeqCst);
        }
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
}

#[test]
fn a_frame_names_the_nodes_that_waited_for_the_device() {
    let waited = Arc::new(AtomicU64::new(0));
    let device = Device {
        waited: Arc::clone(&waited),
        cpu: entelechy_cpu::CpuBackend,
    };
    let mut runtime = Runtime::with_backends(1 << 24, Box::new(device), None);
    runtime.set_small(0);
    let dir = std::env::temp_dir().join(format!("entelechy-waits-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.esn");
    std::fs::write(
        &scene,
        "node seed: kernel {
    requires { field.write; time.read; }
    output result: Rows<Float>;
    compute { return float(element()) + time(); }
    domain = 8;
}
node count: kernel {
    requires { field.read; field.write; }
    input x: Rows<Float> <- seed.result;
    output result: Rows<Float>;
    compute { return sample(x) * 2.0; }
}
output result <- count.result;
",
    )
    .unwrap();
    let mut session = Session::open_in(
        Arc::new(entelechy_session::Disk::default()),
        &scene,
        None,
        runtime,
    );
    let _ = session.sync();
    let report = session
        .evaluate(&entelechy_graph::Context {
            time: 1.0,
            ..entelechy_graph::Context::default()
        })
        .unwrap();
    assert_eq!(
        report.waits,
        vec![(
            entelechy_graph::NodeId("count".into()),
            Duration::from_millis(3)
        )],
        "the node that waited, and only it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
