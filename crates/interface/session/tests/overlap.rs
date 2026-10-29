//! Overlapping frames: a session that overlaps commits each frame without
//! waiting for its device, and waits for it once the next frame's work has
//! gone to the device. A frame found failed then takes back its commit and
//! the next one's: the state is what the frame before them left. What a
//! frame writes to a file is written once it is confirmed, never waiting
//! for it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_kernel::{Backend, Check, ExecutionError, Runtime};
use entelechy_rules::backend::Checked;
use entelechy_session::Session;

type Log = Arc<Mutex<Vec<&'static str>>>;

/// A device that runs on the CPU, answers each run with a check that fails
/// while `failing` is set, and records each run and each wait.
struct Device {
    log: Log,
    failing: Arc<AtomicBool>,
    cpu: entelechy_cpu::CpuBackend,
}

/// A run's verdict, recorded when waited on.
struct Verdict {
    log: Log,
    failed: bool,
}

impl Check for Verdict {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        self.log.lock().unwrap().push("check");
        if self.failed {
            return Err(ExecutionError::Device("the run failed".into()));
        }
        Ok(())
    }
}

/// Every run submitted so far, recorded when waited on.
struct Mark(Log);

impl Check for Mark {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        self.0.lock().unwrap().push("mark");
        Ok(())
    }
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
    fn submitted(&self) -> Option<Box<dyn Check>> {
        Some(Box::new(Mark(Arc::clone(&self.log))))
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
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
    fn execute_checked(
        &self,
        prepared: &dyn entelechy_kernel::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        self.log.lock().unwrap().push("run");
        let outputs = Backend::execute(&self.cpu, prepared, module, invocation, scope)?;
        let verdict = Verdict {
            log: Arc::clone(&self.log),
            failed: self.failing.load(Ordering::SeqCst),
        };
        Ok((outputs, Some(Box::new(verdict))))
    }
}

/// A count that adds the time to itself each frame, from the elements'
/// numbers, on `device`, with `more` of the scene after it.
fn opened(name: &str, device: Device, more: &str) -> (Session, std::path::PathBuf) {
    let mut runtime = Runtime::with_backends(1 << 24, Box::new(device), None);
    // Every run on the device, however small.
    runtime.set_small(0);
    let dir = std::env::temp_dir().join(format!("entelechy-overlap-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.esn");
    std::fs::write(
        &scene,
        format!(
            "node seed: kernel {{
    requires {{ field.write; }}
    output result: Rows<Float>;
    compute {{ return float(element()); }}
    domain = 64;
}}
node count: kernel {{
    requires {{ field.read; field.write; time.read; }}
    input previous: Rows<Float> <~ count.result from seed.result;
    output result: Rows<Float>;
    compute {{ return sample(previous) + time(); }}
}}
output result <- count.result;
{more}"
        ),
    )
    .unwrap();
    let mut session = Session::open_in(
        Arc::new(entelechy_session::Disk::default()),
        &scene,
        None,
        runtime,
    );
    let _ = session.sync();
    session.set_overlap(true).unwrap();
    (session, dir)
}

fn at(time: f64) -> entelechy_graph::Context {
    entelechy_graph::Context {
        time,
        ..entelechy_graph::Context::default()
    }
}

/// The count's first element, the frame in flight confirmed.
fn first(session: &mut Session) -> f32 {
    session.confirm().unwrap();
    let handle = session
        .cached(&entelechy_graph::NodeId("count".into()), "result")
        .unwrap();
    f32::from_bits(session.field(handle).unwrap().values()[0])
}

#[test]
fn an_overlapping_frame_is_waited_for_once_the_next_has_gone_to_the_device() {
    let log = Log::default();
    let (mut session, dir) = opened(
        "waits",
        Device {
            log: Arc::clone(&log),
            failing: Arc::default(),
            cpu: entelechy_cpu::CpuBackend,
        },
        "",
    );
    session.evaluate(&at(1.0)).unwrap();
    {
        let log = log.lock().unwrap();
        assert!(log.contains(&"run"), "{log:?}");
        assert!(
            !log.contains(&"check") && !log.contains(&"settle"),
            "{log:?}"
        );
    }
    log.lock().unwrap().clear();
    session.evaluate(&at(2.0)).unwrap();
    {
        let log = log.lock().unwrap();
        let last_run = log.iter().rposition(|entry| *entry == "run").unwrap();
        let first_wait = log.iter().position(|entry| *entry == "check").unwrap();
        assert!(last_run < first_wait, "{log:?}");
        assert_eq!(log.last(), Some(&"mark"), "{log:?}");
        assert!(!log.contains(&"settle"), "{log:?}");
    }
    // Each frame's state is the one before's, whether confirmed or not.
    assert_eq!(first(&mut session), 3.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_frame_found_failed_takes_back_its_commit_and_the_next_ones() {
    let failing = Arc::new(AtomicBool::new(false));
    let (mut session, dir) = opened(
        "failed",
        Device {
            log: Log::default(),
            failing: Arc::clone(&failing),
            cpu: entelechy_cpu::CpuBackend,
        },
        "",
    );
    session.evaluate(&at(1.0)).unwrap();
    failing.store(true, Ordering::SeqCst);
    // Committed unconfirmed: its failure is not known yet.
    session.evaluate(&at(2.0)).unwrap();
    failing.store(false, Ordering::SeqCst);
    // Confirming the frame before finds it failed: both are taken back.
    let (_, before) = session.position();
    let error = session.evaluate(&at(3.0)).unwrap_err();
    assert!(error.to_string().contains("the run failed"), "{error}");
    // The frame answered as committed, taken back, is not counted.
    assert_eq!(session.position().1, before - 1);
    // What the first frame left, and this frame's time: 0 + 1 + 4.
    session.evaluate(&at(4.0)).unwrap();
    assert_eq!(first(&mut session), 5.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The count as written to the file.
fn written(file: &std::path::Path) -> f32 {
    f32::from_bits(entelechy_formats::read(file, None).unwrap().0.values()[0])
}

const WRITTEN: &str = "output kept <- count.result to \"count.field\";\n";

#[test]
fn a_frame_is_written_once_confirmed_and_never_waited_for_to_be() {
    let log = Log::default();
    let (mut session, dir) = opened(
        "written",
        Device {
            log: Arc::clone(&log),
            failing: Arc::default(),
            cpu: entelechy_cpu::CpuBackend,
        },
        WRITTEN,
    );
    session.allow_writes(&dir).unwrap();
    let file = dir.join("count.field");
    session.evaluate(&at(1.0)).unwrap();
    // The frame changed what is written and is not waited for.
    {
        let log = log.lock().unwrap();
        assert!(
            !log.contains(&"check") && !log.contains(&"settle"),
            "{log:?}"
        );
    }
    assert!(!file.exists());
    // Confirmed once the next has gone to the device, it is written as it
    // left the count: 0 + 1.
    session.evaluate(&at(2.0)).unwrap();
    assert_eq!(written(&file), 1.0);
    session.confirm().unwrap();
    assert_eq!(written(&file), 3.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_frame_taken_back_is_never_written() {
    let failing = Arc::new(AtomicBool::new(false));
    let (mut session, dir) = opened(
        "unwritten",
        Device {
            log: Log::default(),
            failing: Arc::clone(&failing),
            cpu: entelechy_cpu::CpuBackend,
        },
        WRITTEN,
    );
    session.allow_writes(&dir).unwrap();
    let file = dir.join("count.field");
    session.evaluate(&at(1.0)).unwrap();
    failing.store(true, Ordering::SeqCst);
    session.evaluate(&at(2.0)).unwrap();
    assert_eq!(written(&file), 1.0);
    failing.store(false, Ordering::SeqCst);
    // The second frame is found failed: what it made, 1 + 2, is not
    // written, and the third's commit is taken back with it.
    session.evaluate(&at(3.0)).unwrap_err();
    assert_eq!(written(&file), 1.0);
    session.evaluate(&at(4.0)).unwrap();
    assert_eq!(written(&file), 1.0);
    session.confirm().unwrap();
    assert_eq!(written(&file), 5.0);
    let _ = std::fs::remove_dir_all(&dir);
}
