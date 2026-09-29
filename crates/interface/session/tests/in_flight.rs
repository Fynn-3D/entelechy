//! A session keeps one frame in flight on a device: each evaluation waits
//! for the last one's device work before it submits its own, so a caller
//! evaluating faster than the device holds one frame's device memory.

use std::path::Path;
use std::sync::{Arc, Mutex};

use orla_abi::{Handle, Invocation, Scope};
use orla_kernel::{Backend, Runtime};
use orla_session::Session;

/// A device that runs on the CPU and records each run and each wait.
struct Device {
    log: Arc<Mutex<Vec<&'static str>>>,
    cpu: orla_cpu::CpuBackend,
}

impl Backend for Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn has_device(&self) -> bool {
        true
    }
    fn settle_device(&self) {
        self.log.lock().unwrap().push("settle");
    }
    fn prepare(
        &self,
        module: &orla_ir::Validated,
    ) -> Result<Box<dyn orla_kernel::Prepared>, String> {
        self.cpu.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn orla_kernel::Prepared,
        module: &orla_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_kernel::ExecutionError> {
        self.log.lock().unwrap().push("run");
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
}

#[test]
fn each_evaluation_waits_for_the_last_ones_device_work_first() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = Runtime::with_backends(
        1 << 24,
        Box::new(Device {
            log: Arc::clone(&log),
            cpu: orla_cpu::CpuBackend,
        }),
        None,
    );
    // Every run on the device, however small.
    runtime.set_small(0);
    let scene =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/life/tests/blinker.osn");
    let mut session = Session::open_in(
        Arc::new(orla_session::Disk::default()),
        &scene,
        None,
        runtime,
    );
    let _ = session.sync();
    for frame in 0..3 {
        let context = orla_graph::Context {
            time: frame as f32 * 0.1,
            ..orla_graph::Context::default()
        };
        log.lock().unwrap().clear();
        let report = session.evaluate(&context).unwrap();
        assert!(!report.executed.is_empty());
        let log = log.lock().unwrap();
        assert_eq!(log.first(), Some(&"settle"), "frame {frame}: {log:?}");
        assert!(log.contains(&"run"), "frame {frame}: {log:?}");
    }
}
