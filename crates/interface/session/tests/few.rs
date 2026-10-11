//! A run of so few elements that one of a device's groups holds them all
//! goes to the processor however much each takes, as the device runs them
//! a thread each; many elements of the same work stay on the device. One
//! that reads what the device made in the same evaluation goes only when
//! it is long, the device's work kept whole otherwise.

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_ir::Validated;
use entelechy_kernel::{Backend, ExecutionError, Prepared, Runtime};
use entelechy_session::Session;

/// The interpreter, as a device.
struct Device;

impl Backend for Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn has_device(&self) -> bool {
        true
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
        Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
}

#[test]
fn few_elements_of_much_work_go_to_the_processor_and_many_stay() {
    let dir = std::env::temp_dir().join(format!("entelechy-few-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Work enough that the size alone keeps either on the device.
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.heavy@1 { requires { field.write; } output result: Rows<Float>; compute {
    return loop 60000 with a = float(element()) { a * 0.999 + sin(a) * cos(a) + fract(a) * 0.5 };
} }
kernel t.made@1 { requires { field.write; } output result: Rows<Float>; compute { return float(element()) * 0.5; } }
kernel t.light@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return at(x, 3) + 1.0; } }
kernel t.long@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute {
    return loop 60000 with a = at(x, 3) { a * 0.999 + sin(a) * cos(a) + fract(a) * 0.5 };
} }
kernel t.mark@1 { requires { field.read; field.write; } input at: Rows<Int>; input canvas: Rows<Int>; output result: Rows<Int>;
    struct Hit { place: Int; value: Int; }
    scatter into canvas with min { return Hit { place: sample(at), value: element() }; } }
kernel t.ats@1 { requires { field.write; time.read; } output result: Rows<Int>; compute { return element() * 7 + int(time()); } }
kernel t.blank@1 { requires { field.write; } output result: Rows<Int>; compute { return 1000; } }
node one: t.heavy@1 { domain = 1; }
node ats: t.ats@1 { domain = 2; }
node canvas: t.blank@1 { domain = 1000000; }
node small_canvas: t.blank@1 { domain = 16; }
node mark: t.mark@1 { at <- ats.result; canvas <- canvas.result; }
node mark_small: t.mark@1 { at <- ats.result; canvas <- small_canvas.result; }
node many: t.heavy@1 { domain = 65; }
node made: t.made@1 { domain = 200000; }
node light: t.light@1 { x <- made.result; domain = 1; }
node long: t.long@1 { x <- made.result; domain = 1; }
output result <- one.result;
output all <- many.result;
output lit <- light.result;
output longer <- long.result;
output marked <- mark.result;
output marked_small <- mark_small.result;
",
    )
    .unwrap();
    let runtime = Runtime::with_backends(
        1 << 24,
        Box::new(Device),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    let report = session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    let placed = |node: &str| {
        report
            .placements
            .iter()
            .find(|placed| placed.node.0 == node)
            .map(|placed| placed.routed.clone())
            .unwrap_or_else(|| panic!("{node} ran: {:?}", report.placements))
    };
    let one = placed("one");
    assert!(
        one.as_deref()
            .is_some_and(|why| why.contains("few: 1 elements")),
        "{one:?}"
    );
    assert_eq!(placed("many"), None);
    // What the device made this evaluation, read: the short run stays,
    // the long one goes.
    assert_eq!(placed("made"), None);
    assert_eq!(placed("light"), None);
    assert!(placed("long").is_some_and(|why| why.contains("few")));
    // A scatter of two elements run again into a target of a million made
    // before carries the target whole, so it stays where the target is;
    // into a target of a few, it goes.
    let later = entelechy_kernel::Context {
        time: 1.0,
        ..Default::default()
    };
    let report = session.evaluate(&later).unwrap();
    let placed = |node: &str| {
        report
            .placements
            .iter()
            .find(|placed| placed.node.0 == node)
            .map(|placed| placed.routed.clone())
            .unwrap_or_else(|| panic!("{node} ran: {:?}", report.placements))
    };
    assert_eq!(placed("mark"), None);
    assert!(placed("mark_small").is_some_and(|why| why.contains("few")));
    let _ = std::fs::remove_dir_all(&dir);
}
