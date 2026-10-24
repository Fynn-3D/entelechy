//! Where a Kernel runs changes nothing it computes: a scene of every kind
//! of work, played on the JIT and on a main backend that leaves half the
//! Kernels to the interpreter, gives every output the same words at
//! every frame, each half left over in one of two runs. Each
//! backend refuses to run what the other prepared, as a device does, so a
//! Kernel prepared in one place and run in another, on the frame, off it
//! or repeated, fails its example.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_ir::Validated;
use entelechy_rules::backend::Checked;
use entelechy_rules::{Backend, ExecutionError, NativeOp, Prepared};
use entelechy_session::Session;
use entelechy_session::replay::{self, Replayer};

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
    ) -> Result<Vec<u32>, String> {
        self.inner.download(device, byte_len)
    }
}

/// The Kernels of [`SCENE`].
const KERNELS: &str = r#"kernel k.seed@1 { requires { field.write; time.read; } output result: Rows<Float>; compute { return float(element()) * 0.25 + time(); } }
kernel k.step@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return sample(x) * 0.9 + 0.1; } }
kernel k.total@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute over x { return sample(x); } merge(a, b) { return a + b; } }
kernel k.counts@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Int>; compute { return int(sample(x)) % 4; } }
kernel k.ends@1 { requires { field.read; field.write; } input counts: Rows<Int>; output result: Rows<Int>; prefix over counts with sum { return sample(counts); } }
kernel k.zeros@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }
kernel k.spread@1 {
    requires { field.read; field.write; }
    input counts: Rows<Int>;
    input slots: Rows<Int>;
    output result: Rows<Int>;
    struct Hit { at: Int; value: Int; }
    scatter over counts into slots with sum {
        let n = sample(counts);
        loop n with k = 0 { emit(Hit { at: (element() + k) % 16, value: 1 }); k + 1 };
        return Hit { at: element() % 16, value: n };
    }
}
kernel k.walk@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; scan over x with acc = 0.0 { return acc * 0.5 + sample(x); } }
kernel k.rounds@1 {
    requires { field.read; field.write; }
    input ends: Rows<Int>;
    output result: Rows<Int>;
    compute {
        let n = at(ends, element());
        return loop (n + 65535) / 65536 with t = 0 { t + loop clamp(n - index() * 65536, 0, 65536) with u = 0 { u + 1 } };
    }
}
kernel k.shift@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return at(x, (element() + 1) % count(x)) * 0.5 + sample(x) * 0.5; } }
"#;

/// Every kind of work: maps fused into one, a map repeated with a carry,
/// a reduction, a prefix, a scatter that emits, a scan, a walk in rounds
/// of the loop limit, and a run off the frame.
const SCENE: &str = r#"// A scene of every kind of work: maps fused into one, a map repeated
// with a carry, a reduction, a prefix, a scatter that emits, a scan, a
// walk in rounds of the loop limit, and a run off the frame.
use "kinds.ent";
node seed: k.seed@1 { domain = 64; }
node steps: k.step@1 { times = 6; x <- seed.result; carry x <- result; }
node a: k.step@1 { x <- steps.result; }
node b: k.step@1 { x <- a.result; }
node total: k.total@1 { x <- b.result; }
node counts: k.counts@1 { x <- b.result; }
node ends: k.ends@1 { counts <- counts.result; }
node zeros: k.zeros@1 { domain = 16; }
node spread: k.spread@1 { counts <- counts.result; slots <- zeros.result; }
node walk: k.walk@1 { x <- b.result; }
node rounds: k.rounds@1 { ends <- ends.result; }
node turned: k.shift@1 { times = 5; x <- b.result; carry x <- result; }
node later: k.step@1 { x <- walk.result; async = true; }
output total <- total.result;
output ends <- ends.result;
output spread <- spread.result;
output walk <- walk.result;
output rounds <- rounds.result;
output later <- later.result;
output turned <- turned.result;
"#;

/// Its frames, each settling the run off the frame: when an unsettled
/// run lands is the machine's timing, which the two sessions compared
/// need not share.
const REPLAY: &str = r#"scene kinds.esn
frame 0.0
settle
frame 0.5
settle
frame 1.0
settle
frame 1.5
settle
frame 2.0
settle
expect count spread.result = 16
"#;

/// [`REPLAY`] written beside its scene, in a directory of its own.
fn replays() -> Vec<PathBuf> {
    // Each call its own directory: tests running at once write here.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("entelechy-routing-{}-{call}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("kinds.ent"), KERNELS).unwrap();
    std::fs::write(dir.join("kinds.esn"), SCENE).unwrap();
    std::fs::write(dir.join("kinds.replay"), REPLAY).unwrap();
    vec![dir.join("kinds.replay")]
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
    let limits = entelechy_session::Limits::default();
    let runtime = if let Some(split) = split {
        let (half, device) = match split {
            Split::Half(half) => (Some(half), false),
            Split::Sized => (None, true),
        };
        let mut runtime = entelechy_kernel::Runtime::with_backends(
            limits.words,
            Box::new(Owned {
                name: "main",
                inner: Box::new(entelechy_jit::JitBackend::new().unwrap()),
                half,
                device,
            }),
            Some(Box::new(Owned {
                name: "ordered",
                inner: Box::new(entelechy_cpu::CpuBackend),
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
        Arc::new(entelechy_session::Disk::default()),
        &replay.scene,
        replay.kernels.as_deref(),
        runtime,
    );
    let _ = session.sync();
    session
}

/// Every output of the scene by name, as words.
fn outputs(session: &mut Session, report: &entelechy_graph::Report) -> Vec<(String, Vec<u32>)> {
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
fn where_a_kernel_runs_changes_no_output() {
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
/// Kernel for itself the first time; every output is the same words as
/// on the JIT alone.
#[test]
fn a_run_placed_by_its_size_changes_no_output() {
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
