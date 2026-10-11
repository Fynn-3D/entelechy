//! A run remembered frame by frame: played again, a frame takes from the
//! same frame of the last run what nothing it reads has changed; a changed
//! Kernel whose outputs are what they were leaves what reads it as it
//! was; outputs that part from the remembered ones run what reads them;
//! and what it answers is what a fresh session answers.

use std::collections::BTreeSet;
use std::path::Path;

use entelechy_kernel::{Backend, Context, Handle, Invocation, Prepared, Runtime, Scope, Validated};
use entelechy_session::Session;

/// A count advanced a frame at a time, a value made of it (`made`), what
/// reads that (`shown`), and a large Field made once (`big`, over a
/// mebibyte, remembered by key and epoch alone) summed by `total`.
fn kernels(made: &str) -> String {
    format!(
        "kernel t.zero@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 0.0; }} }}
kernel t.count@1 {{ requires {{ field.read; field.write; }} input previous: Rows<Float>; output result: Rows<Float>; compute {{ return at(previous, 0) + 1.0; }} }}
kernel t.made@1 {{ requires {{ field.read; field.write; }} input count: Rows<Float>; output result: Rows<Float>; compute {{ let x = at(count, 0); return {made}; }} }}
kernel t.shown@1 {{ requires {{ field.read; field.write; }} input made: Rows<Float>; output result: Rows<Float>; compute {{ return at(made, 0) + 0.5; }} }}
kernel t.big@1 {{ requires {{ field.write; }} param seed: Int = 7; output result: Rows<Float>; compute {{ return float(element() % seed); }} }}
kernel t.total@1 {{ requires {{ field.read; field.write; }} input big: Rows<Float>; input count: Rows<Float>; output result: Rows<Float>; compute {{ return at(big, 300000) + at(count, 0); }} }}
"
    )
}

const SCENE: &str = "use \"k.ent\";
param seed: Int = 7;
node zero: t.zero@1 { domain = 1; }
node count: t.count@1 { previous <~ count.result from zero.result; domain = 1; }
node made: t.made@1 { count <- count.result; domain = 1; }
node shown: t.shown@1 { made <- made.result; domain = 1; }
node big: t.big@1 { seed = seed; domain = 400000; }
node total: t.total@1 { big <- big.result; count <- count.result; domain = 1; }
output result <- shown.result;
";

fn scratch(name: &str, made: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-recall-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.ent"), kernels(made)).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

/// Plays four frames asking for `shown` and `total`: what each frame
/// executed, and what they answered on the last.
fn play(session: &mut Session) -> (Vec<BTreeSet<String>>, (f32, f32)) {
    let demand: BTreeSet<entelechy_graph::NodeId> = ["shown", "total"]
        .into_iter()
        .map(|n| entelechy_graph::NodeId(n.into()))
        .collect();
    let mut executed = Vec::new();
    for frame in 0..4 {
        let context = Context {
            time: frame as f32 * 0.1,
            ..Context::default()
        };
        let report = session.evaluate_for(&context, &demand).unwrap();
        executed.push(report.executed.iter().map(|n| n.0.clone()).collect());
    }
    let read = |session: &mut Session, node: &str| {
        let handle = session
            .cached(&entelechy_graph::NodeId(node.into()), "result")
            .unwrap();
        f32::from_bits(session.field(handle).unwrap().values()[0])
    };
    let answered = (read(session, "shown"), read(session, "total"));
    (executed, answered)
}

fn fresh(dir: &Path) -> (f32, f32) {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
    session.sync();
    play(&mut session).1
}

#[test]
fn a_run_played_again_computes_only_what_changed_and_answers_alike() {
    let dir = scratch("again", "x * 2.0");
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 24);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.recall_as(Some("run".into()));
    let (first, answered) = play(&mut session);
    assert!(first[1].contains("count") && first[1].contains("shown"));
    // The same run again: nothing runs.
    session.start_over();
    session.recall_as(Some("run".into()));
    let (again, answered_again) = play(&mut session);
    assert_eq!(answered_again, answered);
    assert!(again.iter().all(BTreeSet::is_empty), "{again:?}");
    // Another run between, the large Field made of another seed; then the
    // first again: the large Field, remembered by key alone, is made once
    // more, and what reads it is taken as it was.
    session.start_over();
    session
        .set_param("seed", entelechy_kernel::Value::Int(5))
        .unwrap();
    session.sync();
    session.recall_as(Some("other".into()));
    play(&mut session);
    session.start_over();
    session.recall_as(Some("run".into()));
    let (after, answered_after) = play(&mut session);
    assert_eq!(answered_after, answered);
    assert_eq!(after[0], BTreeSet::from(["big".to_owned()]), "{after:?}");
    assert!(after[1..].iter().all(BTreeSet::is_empty), "{after:?}");
    // A changed Kernel computing the same: it runs; what reads it does not.
    std::fs::write(dir.join("k.ent"), kernels("x + x")).unwrap();
    session.start_over();
    session.recall_as(Some("run".into()));
    let (alike, answered_alike) = play(&mut session);
    assert_eq!(answered_alike, answered);
    assert!(
        alike[1..]
            .iter()
            .all(|frame| frame.contains("made") && !frame.contains("shown")),
        "{alike:?}"
    );
    // Computing otherwise: what reads it runs, and answers as a fresh
    // session does.
    std::fs::write(dir.join("k.ent"), kernels("x * 3.0")).unwrap();
    session.start_over();
    session.recall_as(Some("run".into()));
    let (parted, answered_parted) = play(&mut session);
    assert!(
        parted[1..].iter().all(|frame| frame.contains("shown")),
        "{parted:?}"
    );
    assert_eq!(answered_parted, fresh(&dir));
    assert_ne!(answered_parted, answered);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The interpreter, not saying it repeats exactly.
struct Unpromised(entelechy_cpu::CpuBackend);

impl Backend for Unpromised {
    fn name(&self) -> &'static str {
        "unpromised"
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        self.0.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, entelechy_kernel::ExecutionError> {
        Backend::execute(&self.0, prepared, module, invocation, scope)
    }
}

#[test]
fn a_run_made_again_on_a_backend_that_does_not_repeat_exactly_runs_what_reads_it() {
    let dir = scratch("unpromised", "x * 2.0");
    let runtime = Runtime::with_backends(
        1 << 24,
        Box::new(Unpromised(entelechy_cpu::CpuBackend)),
        None,
    );
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    session.recall_as(Some("run".into()));
    let (_, answered) = play(&mut session);
    session.start_over();
    session
        .set_param("seed", entelechy_kernel::Value::Int(5))
        .unwrap();
    session.sync();
    session.recall_as(Some("other".into()));
    play(&mut session);
    session.start_over();
    session.recall_as(Some("run".into()));
    let (after, answered_after) = play(&mut session);
    assert_eq!(answered_after, answered);
    // The large Field, made again, is not trusted to be the one
    // remembered: what reads it runs.
    assert_eq!(
        after[0],
        BTreeSet::from(["big".to_owned(), "total".to_owned()]),
        "{after:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
