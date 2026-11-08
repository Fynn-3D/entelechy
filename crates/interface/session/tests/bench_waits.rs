//! What `bench` says a frame waited for, node by node and at the frame's
//! end, is the best run's, as the frame's own wait is: a slower run
//! before it (a device still building) does not show in its breakdown.

use std::sync::Mutex;
use std::time::Duration;

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_ir::Validated;
use entelechy_kernel::{Backend, Context, ExecutionError, Prepared, Runtime};
use entelechy_session::Session;

/// The interpreter as a device whose runs `slow` to `slow + 3` (the first
/// run's frames) wait for it 5 ms each.
#[derive(Default)]
struct Device {
    runs: Mutex<u32>,
    waited: Mutex<Duration>,
}

impl Backend for &'static Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        entelechy_cpu::CpuBackend.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let run = {
            let mut runs = self.runs.lock().unwrap();
            *runs += 1;
            *runs
        };
        if (2..=4).contains(&run) {
            std::thread::sleep(Duration::from_millis(5));
            *self.waited.lock().unwrap() += Duration::from_millis(5);
        }
        Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
    fn waited(&self) -> Duration {
        *self.waited.lock().unwrap()
    }
}

#[test]
fn the_wait_breakdown_is_the_best_runs() {
    let dir = std::env::temp_dir().join(format!("entelechy-bench-waits-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.clock@1 { requires { field.write; time.read; } output result: Rows<Float>; compute { return time(); } }
node clock: t.clock@1 { domain = 4; }
output result <- clock.result;
",
    )
    .unwrap();
    let device: &'static Device = Box::leak(Box::default());
    let mut runtime = Runtime::with_backend(1 << 20, Box::new(device));
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    // The warm-up frame is run 1; the first run's three frames wait.
    let measured = entelechy_session::bench::measure(
        &mut session,
        "device",
        3,
        2,
        60.0,
        Context::default(),
        false,
    )
    .unwrap();
    assert!(
        measured.wait < 1.0,
        "the best run waited {} ms a frame",
        measured.wait
    );
    assert!(
        measured.waits.is_empty(),
        "the best run's nodes waited for nothing: {:?}",
        measured.waits
    );
    assert!(measured.wait_after < 1.0, "{}", measured.wait_after);
    let _ = std::fs::remove_dir_all(&dir);
}
