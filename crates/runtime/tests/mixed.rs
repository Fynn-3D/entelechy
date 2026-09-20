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
    let map = runtime
        .prepare("kernel t.map@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 2; } }")
        .unwrap();
    let scan = runtime
        .prepare("kernel t.scan@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; scan with acc = 0 { return acc + sample(rows); } }")
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
    let plain = Runtime::new(1 << 16);
    assert_eq!(plain.ordered_backend_name(), None);
    let _ = all();
}
