//! Ordered work on a second backend: a runtime over one backend with
//! another for scans prepares and executes each mode where it belongs,
//! and the scan's inputs reach it through Host memory.

use std::any::Any;
use std::sync::{Arc, Mutex};

use orla_abi::{Capabilities, Capability, Domain, Handle, Invocation, Scope};
use orla_runtime::{Backend, Expect, Runtime};

/// A backend that records what it is asked to do, and does it on the CPU.
struct Recorder {
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
    cpu: orla_cpu::CpuBackend,
}

impl Backend for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn prepare(
        &self,
        module: &orla_ir::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{} prepares {}", self.name, module.manifest.id));
        self.cpu.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_runtime::ExecutionError> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{} executes {}", self.name, module.manifest.id));
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
}

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

#[test]
fn scans_go_to_the_ordered_backend_and_the_rest_stays() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let main = Recorder {
        name: "main",
        log: log.clone(),
        cpu: orla_cpu::CpuBackend,
    };
    let ordered = Recorder {
        name: "ordered",
        log: log.clone(),
        cpu: orla_cpu::CpuBackend,
    };
    let mut runtime = Runtime::with_backends(1 << 20, Box::new(main), Some(Box::new(ordered)));
    let map = prepare(&runtime, "kernel t.map@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 2; } }")
        .unwrap();
    let scan = prepare(&runtime, "kernel t.scan@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { return acc + sample(rows); } }")
        .unwrap();
    let map = runtime.publish(map, Expect::Absent).unwrap();
    let scan = runtime.publish(scan, Expect::Absent).unwrap();
    assert_eq!(runtime.executes_on(&map), "main");
    assert_eq!(runtime.executes_on(&scan), "ordered");
    assert_eq!(runtime.ordered_backend_name(), Some("ordered"));
    let rows = runtime
        .execute(
            &map,
            &Invocation {
                inputs: vec![],
                params: vec![],
                domain: Domain::Rows(5),
                context: Default::default(),
            },
        )
        .unwrap()[0];
    let sums = runtime
        .execute(
            &scan,
            &Invocation {
                inputs: vec![rows],
                params: vec![],
                domain: Domain::Rows(5),
                context: Default::default(),
            },
        )
        .unwrap()[0];
    let words: Vec<u32> = runtime.host().data(sums).unwrap().words().to_vec();
    assert_eq!(words, [0, 2, 6, 12, 20]);
    let log = log.lock().unwrap().clone();
    assert_eq!(
        log,
        [
            "main prepares t.map@1",
            "ordered prepares t.scan@1",
            "main executes t.map@1",
            "ordered executes t.scan@1",
        ]
    );
    // Without a second backend, everything is the main one's.
    let plain = cpu(1 << 16);
    assert_eq!(plain.ordered_backend_name(), None);
    let _ = all();
}

/// A backend that claims a device time for every Kernel it runs and
/// none for a native operation, which is what a device whose clock
/// answers looks like beside a rasterizer that is not asked.
struct Clocked {
    cpu: orla_cpu::CpuBackend,
    /// Whether the clock has something to say, taken when it is read:
    /// what a device's timestamp is, so a call nobody timed answers
    /// nothing rather than the call before it.
    timed: std::sync::atomic::AtomicBool,
}

impl Backend for Clocked {
    fn name(&self) -> &'static str {
        "clocked"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn prepare(
        &self,
        module: &orla_ir::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        self.cpu.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_runtime::ExecutionError> {
        self.timed.store(true, std::sync::atomic::Ordering::Relaxed);
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
    fn execute_native(
        &self,
        op: orla_runtime::NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_runtime::ExecutionError> {
        Backend::execute_native(&self.cpu, op, invocation, scope)
    }
    fn last_device_time(&self) -> Option<std::time::Duration> {
        // A Kernel is timed; a native operation is not, and the runtime
        // must not hand the Kernel's number to it.
        self.timed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
            .then(|| std::time::Duration::from_millis(7))
    }
}

#[test]
fn a_node_the_backend_did_not_time_is_not_given_its_neighbours_number() {
    let mut runtime = Runtime::with_backend(
        1 << 22,
        Box::new(Clocked {
            cpu: orla_cpu::CpuBackend,
            timed: std::sync::atomic::AtomicBool::new(false),
        }),
    );
    let points = prepare(&runtime, "kernel t.points@1 { requires { field.write; } output result: Rows<Vertex>; struct Vertex { clip: Vec4; normal: Vec3; uv: Vec2; } compute { let i = element(); return Vertex { clip: vec4(float(i) - 0.5, -0.5, 0.2, 1.0), normal: vec3(0.0, 0.0, 1.0), uv: vec2(0.0, 0.0) }; } }")
        .unwrap();
    let faces = prepare(&runtime, "kernel t.faces@1 { requires { field.write; } output result: Rows<Tri>; struct Tri { a: Int; b: Int; c: Int; } compute { return Tri { a: 0, b: 1, c: 2 }; } }")
        .unwrap();
    let points = runtime.publish(points, Expect::Absent).unwrap();
    let faces = runtime.publish(faces, Expect::Absent).unwrap();

    let run = |runtime: &mut Runtime, version: &Arc<orla_runtime::KernelVersion>, count: u32| {
        runtime
            .execute(
                version,
                &Invocation {
                    inputs: vec![],
                    params: vec![],
                    domain: Domain::Rows(count),
                    context: Default::default(),
                },
            )
            .unwrap()
    };
    let vertices = run(&mut runtime, &points, 3)[0];
    // A Kernel: the backend's clock answered.
    assert_eq!(
        runtime.last_measured(),
        Some(std::time::Duration::from_millis(7))
    );
    let triangles = run(&mut runtime, &faces, 1)[0];
    assert_eq!(
        runtime.last_measured(),
        Some(std::time::Duration::from_millis(7))
    );

    let raster = runtime
        .kernel(&orla_runtime::NativeOp::Rasterize.id())
        .unwrap();
    runtime
        .execute(
            &raster,
            &Invocation {
                inputs: vec![vertices, triangles],
                params: vec![],
                domain: Domain::Grid(8, 8),
                context: Default::default(),
            },
        )
        .unwrap();
    // The reference rasterizer has no device and reports no time; the
    // runtime must answer `None` rather than the Kernel's seven
    // milliseconds, which is what a profile printed for a rasterize
    // until it was asked to.
    assert_eq!(
        runtime.last_measured(),
        None,
        "a native operation must not be given the previous node's time"
    );
}

/// A runtime on the reference backend, as a session opens one.
fn cpu(budget_words: usize) -> orla_runtime::Runtime {
    orla_runtime::Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend))
}

/// Compiles and prepares a Kernel without publishing it.
fn prepare(
    runtime: &orla_runtime::Runtime,
    source: &str,
) -> Result<orla_runtime::Candidate, orla_runtime::PrepareError> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.prepare_module(module, source)
}
