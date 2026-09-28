//! Ordered work on a second backend: a runtime with another backend for
//! scans prepares and executes each mode where it belongs, and the scan's
//! inputs reach it through Host memory.

use std::sync::{Arc, Mutex};

use orla_abi::{Capabilities, Capability, Domain, Handle, Invocation, Scope};
use orla_kernel::{Backend, Expect, Runtime};

/// A backend that records what it is asked, and does it on the CPU.
struct Recorder {
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
    cpu: orla_cpu::CpuBackend,
    /// Declines a module holding a `Double`, like a device without one.
    no_doubles: bool,
    /// Leaves scans to another backend, like a device that would walk one
    /// on a single thread.
    no_scans: bool,
}

impl Backend for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }
    fn declines(&self, module: &orla_ir::Validated) -> Option<String> {
        if self.no_scans && module.manifest.mode == orla_ir::Mode::Scan {
            return Some("a scan runs its steps in order, on one thread".to_owned());
        }
        (self.no_doubles && module.holds(orla_abi::Type::Double))
            .then(|| "it holds a Double".to_owned())
    }
    fn prepare(
        &self,
        module: &orla_ir::Validated,
    ) -> Result<Box<dyn orla_kernel::Prepared>, String> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{} prepares {}", self.name, module.manifest.id));
        self.cpu.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn orla_kernel::Prepared,
        module: &orla_ir::Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_kernel::ExecutionError> {
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
        no_doubles: false,
        no_scans: true,
    };
    let ordered = Recorder {
        name: "ordered",
        log: log.clone(),
        cpu: orla_cpu::CpuBackend,
        no_doubles: false,
        no_scans: false,
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
    let words: Vec<u32> = runtime.host().data(sums).unwrap().storage().to_vec();
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
    // Where each runs, and why the scan is routed.
    assert_eq!(runtime.placement(&map).routed, None);
    assert_eq!(
        runtime.placement(&scan).routed.as_deref(),
        Some("not on main: a scan runs its steps in order, on one thread")
    );
    // Backends computing in Host memory move nothing.
    assert_eq!(runtime.take_moved(), orla_kernel::Moved::default());
    // Without a second backend, everything is the main one's.
    let plain = cpu(1 << 16);
    assert_eq!(plain.ordered_backend_name(), None);
    let _ = all();
}

#[test]
fn what_the_main_backend_declines_goes_to_the_ordered_one_and_says_why() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let backend = |name, no_doubles| Recorder {
        name,
        log: log.clone(),
        cpu: orla_cpu::CpuBackend,
        no_doubles,
        no_scans: false,
    };
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(backend("main", true)),
        Some(Box::new(backend("ordered", false))),
    );
    let wide = prepare(&runtime, "kernel t.wide@1 { requires { field.write; } output result: Rows<Double>; compute { return double(element()) * 0.5; } }")
        .unwrap();
    let wide = runtime.publish(wide, Expect::Absent).unwrap();
    let placement = runtime.placement(&wide);
    assert_eq!(placement.backend, "ordered");
    assert_eq!(
        placement.routed.as_deref(),
        Some("not on main: it holds a Double")
    );
    let invocation = Invocation {
        inputs: vec![],
        params: vec![],
        domain: Domain::Rows(3),
        context: Default::default(),
    };
    runtime.execute(&wide, &invocation).unwrap();
    // A run off the frame goes where the version was prepared too.
    let run = runtime.execute_later(&wide, &invocation).unwrap();
    runtime.wait_run(run).unwrap();
    assert_eq!(
        log.lock().unwrap().clone(),
        [
            "ordered prepares t.wide@1",
            "ordered executes t.wide@1",
            "ordered executes t.wide@1"
        ]
    );
}

/// A backend that reports a device time for every Kernel and none for a
/// native operation.
struct Clocked {
    cpu: orla_cpu::CpuBackend,
    /// Whether the clock has a reading, taken when read, like a device
    /// timestamp: an untimed call answers nothing.
    timed: std::sync::atomic::AtomicBool,
}

impl Backend for Clocked {
    fn name(&self) -> &'static str {
        "clocked"
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
        self.timed.store(true, std::sync::atomic::Ordering::Relaxed);
        Backend::execute(&self.cpu, prepared, module, invocation, scope)
    }
    fn execute_native(
        &self,
        op: orla_kernel::NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, orla_kernel::ExecutionError> {
        Backend::execute_native(&self.cpu, op, invocation, scope)
    }
    fn last_device_time(&self) -> Option<std::time::Duration> {
        // A Kernel is timed; a native operation is not.
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

    let run = |runtime: &mut Runtime, version: &Arc<orla_kernel::KernelVersion>, count: u32| {
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
    // A Kernel: the backend's clock answers.
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
        .kernel(&orla_kernel::NativeOp::Rasterize.id())
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
    // The reference rasterizer has no device and reports no time: the
    // runtime answers `None`, not the Kernel's seven milliseconds.
    assert_eq!(
        runtime.last_measured(),
        None,
        "a native operation must not be given the previous node's time"
    );
}

/// A runtime on the reference backend, as a session opens one.
fn cpu(budget_words: usize) -> orla_kernel::Runtime {
    orla_kernel::Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend))
}

/// Compiles and prepares a Kernel without publishing it.
fn prepare(
    runtime: &orla_kernel::Runtime,
    source: &str,
) -> Result<orla_kernel::Candidate, orla_kernel::PrepareError> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.prepare_module(module, source)
}
