//! Where a Kernel runs changes nothing it computes: every example's
//! replay, played on the JIT and on a main backend that leaves half the
//! Kernels to the interpreter, gives every output of the scene the same
//! words at every frame, each half left over in one of two runs. Each
//! backend refuses to run what the other prepared, as a device does, so a
//! Kernel prepared in one place and run in another, on the frame, off it
//! or repeated, fails its example.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_abi::{Handle, Invocation, Scope};
use orla_ir::Validated;
use orla_rules::backend::Checked;
use orla_rules::{Backend, ExecutionError, NativeOp, Prepared};
use orla_session::Session;
use orla_session::replay::{self, Replayer};

/// A backend that knows its own prepared state, as a device backend does:
/// it marks what it prepares with its name and refuses to run what
/// another backend prepared. It may decline half the Kernels, those whose
/// id's bytes sum to `half` modulo 2, and may say it is a device, so a
/// run small enough leaves it.
struct Owned {
    name: &'static str,
    inner: Box<dyn Backend>,
    half: Option<u32>,
    device: bool,
}

/// Prepared state marked with the backend that prepared it.
struct Marked {
    by: &'static str,
    inner: Box<dyn Prepared>,
}

impl Prepared for Marked {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Owned {
    /// The inner backend's state, when this backend prepared it.
    fn own<'p>(&self, prepared: &'p dyn Prepared) -> Result<&'p dyn Prepared, ExecutionError> {
        match prepared.as_any().downcast_ref::<Marked>() {
            Some(marked) if marked.by == self.name => Ok(marked.inner.as_ref()),
            Some(marked) => Err(ExecutionError::Device(format!(
                "prepared by `{}`, run on `{}`",
                marked.by, self.name
            ))),
            None => Err(ExecutionError::Device(format!(
                "prepared by no marking backend, run on `{}`",
                self.name
            ))),
        }
    }
}

impl Backend for Owned {
    fn name(&self) -> &'static str {
        self.name
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        let id = module.manifest.id.to_string();
        let sum = id.bytes().map(u32::from).sum::<u32>();
        self.inner.declines(module).or_else(|| {
            self.half
                .is_some_and(|half| sum % 2 == half)
                .then(|| "left to the other backend".to_owned())
        })
    }
    fn uploaded(&self) -> u64 {
        self.inner.uploaded()
    }
    fn has_device(&self) -> bool {
        self.device
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let inner = self.inner.prepare(module)?;
        Ok(Box::new(Marked {
            by: self.name,
            inner,
        }))
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner
            .execute(self.own(prepared)?, module, invocation, scope)
    }
    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        self.inner
            .execute_checked(self.own(prepared)?, module, invocation, scope)
    }
    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner.execute_native(op, invocation, scope)
    }
    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        match self.own(prepared) {
            Ok(own) => self
                .inner
                .execute_repeated(own, module, invocation, scope, repeat, chain),
            Err(error) => Some(Err(error)),
        }
    }
    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        self.inner.download(device, byte_len)
    }
}

fn replays() -> Vec<PathBuf> {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let mut out = Vec::new();
    for example in std::fs::read_dir(&examples).unwrap().flatten() {
        let tests = example.path().join("tests");
        for entry in std::fs::read_dir(tests).into_iter().flatten().flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "replay") {
                out.push(entry.path());
            }
        }
    }
    out.sort();
    out
}

/// How a session leaves Kernels to the interpreter: the half a sum of
/// their ids names, or every run small enough, the JIT saying it is a
/// device.
#[derive(Clone, Copy)]
enum Split {
    Half(u32),
    Sized,
}

/// A session over the replay's scene: on the JIT, or on the JIT with what
/// `split` names left to the interpreter.
fn opened(replay: &replay::Replay, split: Option<Split>) -> Session {
    let limits = orla_session::Limits::default();
    let runtime = if let Some(split) = split {
        let (half, device) = match split {
            Split::Half(half) => (Some(half), false),
            Split::Sized => (None, true),
        };
        let mut runtime = orla_kernel::Runtime::with_backends(
            limits.words,
            Box::new(Owned {
                name: "main",
                inner: Box::new(orla_jit::JitBackend::new().unwrap()),
                half,
                device,
            }),
            Some(Box::new(Owned {
                name: "ordered",
                inner: Box::new(orla_cpu::CpuBackend),
                half: None,
                device: false,
            })),
        );
        runtime.host_mut().set_work(limits.work);
        runtime.set_small(limits.small);
        runtime
    } else {
        Session::runtime_for("jit", limits).unwrap()
    };
    let mut session = Session::open_in(
        Arc::new(orla_session::Disk::default()),
        &replay.scene,
        replay.kernels.as_deref(),
        runtime,
    );
    let _ = session.sync();
    session
}

/// Every output of the scene by name, as words.
fn outputs(session: &mut Session, report: &orla_graph::Report) -> Vec<(String, Vec<u32>)> {
    report
        .outputs
        .iter()
        .map(|(name, handle)| {
            let words = session.field(*handle).unwrap().values().into_owned();
            (name.clone(), words)
        })
        .collect()
}

/// What `path`'s replay found different with what `split` names left to
/// the interpreter, how many frames it compared, and how many Kernels ran
/// routed, or, split by size, how many runs left the main backend.
fn compared(path: &Path, split: Split) -> (Vec<String>, usize, usize) {
    let mut wrong = Vec::new();
    let mut frames = 0;
    let text = std::fs::read_to_string(path).unwrap();
    let replay = replay::parse(&text, path.parent().unwrap()).unwrap();
    let mut whole = opened(&replay, None);
    let mut halved = opened(&replay, Some(split));
    let mut routed = halved
        .runtime()
        .kernels()
        .filter(|version| halved.runtime().placement(version).routed.is_some())
        .count();
    let mut one = Replayer::start(&replay, &mut whole);
    let mut other = Replayer::start(&replay, &mut halved);
    loop {
        let (Some(a), Some(b)) = (
            one.next_frame(&mut whole).unwrap(),
            other.next_frame(&mut halved).unwrap(),
        ) else {
            break;
        };
        let (x, y) = match (whole.evaluate(&a.context), halved.evaluate(&b.context)) {
            (Ok(x), Ok(y)) => (x, y),
            (x, y) => {
                wrong.push(format!(
                    "{}: frame {frames}: {:?} against {:?}",
                    path.display(),
                    x.err().map(|e| e.to_string()),
                    y.err().map(|e| e.to_string())
                ));
                break;
            }
        };
        frames += 1;
        if let Split::Sized = split {
            routed += y
                .placements
                .iter()
                .filter(|placed| placed.routed.is_some())
                .count();
        }
        let (p, q) = (outputs(&mut whole, &x), outputs(&mut halved, &y));
        for ((name, a), (_, b)) in p.iter().zip(&q) {
            if a != b {
                let at = a.iter().zip(b).position(|(p, q)| p != q);
                wrong.push(format!(
                    "{}: frame {frames}: `{name}` differs halved, first at word {at:?}",
                    path.display()
                ));
            }
        }
        one.evaluated(x);
        other.evaluated(y);
    }
    (wrong, frames, routed)
}

#[test]
fn where_a_kernel_runs_changes_no_output_of_any_example() {
    let paths = replays();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(path) = paths.get(at) else { break };
                    // Each half of the Kernels routed in one of two runs.
                    for half in [0, 1] {
                        let found = compared(path, Split::Half(half));
                        results.lock().unwrap().push(found);
                    }
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let frames: usize = results.iter().map(|(_, frames, _)| frames).sum();
    let routed: usize = results.iter().map(|(_, _, routed)| routed).sum();
    let wrong: Vec<String> = results.into_iter().flat_map(|(wrong, ..)| wrong).collect();
    assert!(frames > 0, "no frame was compared");
    assert!(routed > 0, "no Kernel ran routed");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A run small enough goes to the ordered backend, which prepares the
/// Kernel for itself the first time; every output of every example is the
/// same words as on the JIT alone.
#[test]
fn a_run_placed_by_its_size_changes_no_output_of_any_example() {
    let paths = replays();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(path) = paths.get(at) else { break };
                    let found = compared(path, Split::Sized);
                    results.lock().unwrap().push(found);
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let frames: usize = results.iter().map(|(_, frames, _)| frames).sum();
    let placed: usize = results.iter().map(|(_, _, placed)| placed).sum();
    let wrong: Vec<String> = results.into_iter().flat_map(|(wrong, ..)| wrong).collect();
    assert!(frames > 0, "no frame was compared");
    assert!(placed > 0, "no run left the main backend by its size");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
