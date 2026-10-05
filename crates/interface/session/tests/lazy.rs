//! A device that prepares a Kernel and builds its pipelines later: a
//! session builds what its graph's nodes use when it publishes it, and a
//! run builds what nothing built before. A Kernel the device's compiler
//! then fails on goes to the ordered backend, said once, a run that found
//! it made again there. A replacement for a version that ran on the device
//! is built before it is ready; one for a version that never did is not.

use std::sync::{Mutex, OnceLock};

use orla_abi::{Handle, Invocation, Scope};
use orla_ir::Validated;
use orla_kernel::{Backend, Context, Domain, ExecutionError, Expect, Prepared, Runtime, Value};
use orla_session::{Event, Session};

/// The interpreter as a device that builds on first need, whose compiler
/// fails on Kernels named `big`, declining them from then on.
#[derive(Default)]
struct Device {
    /// Every build, by Kernel id, in order.
    builds: Mutex<Vec<String>>,
    unbuilt: Mutex<Vec<Validated>>,
}

/// What the device prepared: the interpreter's state, built once.
struct Lazy {
    inner: Box<dyn Prepared>,
    module: Validated,
    built: OnceLock<bool>,
}

impl Prepared for Lazy {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Device {
    fn build(&self, prepared: &dyn Prepared) -> Result<&'static (), ExecutionError> {
        let lazy = prepared.as_any().downcast_ref::<Lazy>().unwrap();
        let built = *lazy.built.get_or_init(|| {
            let id = lazy.module.manifest.id.to_string();
            self.builds.lock().unwrap().push(id.clone());
            let fails = id.contains("big");
            if fails {
                self.unbuilt.lock().unwrap().push(lazy.module.clone());
            }
            !fails
        });
        built
            .then_some(&())
            .ok_or_else(|| ExecutionError::Device("the compiler crashed".into()))
    }

    fn builds_of(&self, id: &str) -> usize {
        self.builds
            .lock()
            .unwrap()
            .iter()
            .filter(|b| *b == id)
            .count()
    }
}

impl Backend for &'static Device {
    fn name(&self) -> &'static str {
        "device"
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        self.unbuilt
            .lock()
            .unwrap()
            .iter()
            .any(|seen| seen == module)
            .then(|| "the device's own compiler could not build it (it crashed)".to_owned())
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        Ok(Box::new(Lazy {
            inner: orla_cpu::CpuBackend.prepare(module)?,
            module: module.clone(),
            built: OnceLock::new(),
        }))
    }
    fn finish(&self, prepared: &dyn Prepared) {
        let _ = self.build(prepared);
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.build(prepared)?;
        let inner = &prepared.as_any().downcast_ref::<Lazy>().unwrap().inner;
        Backend::execute(
            &orla_cpu::CpuBackend,
            inner.as_ref(),
            module,
            invocation,
            scope,
        )
    }
}

fn device() -> &'static Device {
    Box::leak(Box::default())
}

#[test]
fn a_kernel_a_node_uses_is_built_as_it_publishes_and_routed_if_it_cannot_be() {
    let dir = std::env::temp_dir().join(format!("orla-lazy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "kernel t.small@1 { requires { field.write; } output result: Rows<Float>; compute { return 2.0; } }
kernel t.big@1 { requires { field.read; field.write; } input x: Rows<Float>; output result: Rows<Float>; compute { return at(x, 0) * 3.0; } }
kernel t.unused@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }
node small: t.small@1 { domain = 1; }
node big: t.big@1 { x <- small.result; domain = 1; }
output result <- big.result;
",
    )
    .unwrap();
    let device = device();
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    // Every run on the device, however few its elements.
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    let mut events = session.sync();
    assert_eq!(device.builds_of("t.small@1"), 1);
    assert_eq!(device.builds_of("t.big@1"), 1);
    assert_eq!(device.builds_of("t.unused@1"), 0, "no node uses it");
    for _ in 0..2 {
        session.evaluate(&Context::default()).unwrap();
        let handle = session
            .cached(&orla_graph::NodeId("big".into()), "result")
            .unwrap();
        assert_eq!(
            f32::from_bits(session.field(handle).unwrap().values()[0]),
            6.0
        );
        events.extend(session.sync());
    }
    let routed: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Routed { node, why, .. } => Some((node.clone(), why.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(routed.len(), 1, "{events:?}");
    assert_eq!(routed[0].0, "big");
    assert!(routed[0].1.contains("could not build"), "{routed:?}");
    assert_eq!(device.builds_of("t.big@1"), 1, "built once");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_that_finds_its_kernel_unbuildable_is_made_again_on_the_ordered_backend() {
    let device = device();
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    runtime.set_small(0);
    let big = source("big", "3.0");
    let candidate = runtime
        .prepare_module(orla_compiler::compile(&big).unwrap(), &big)
        .unwrap();
    let version = runtime.publish(candidate, Expect::Any).unwrap();
    assert_eq!(runtime.placement(&version).routed, None);
    let invocation = Invocation {
        inputs: vec![],
        params: Vec::<Value>::new(),
        domain: Domain::Rows(100),
        context: Context::default(),
    };
    for _ in 0..2 {
        let outputs = runtime.execute(&version, &invocation).unwrap();
        let words = runtime.host().data(outputs[0]).unwrap().values().to_vec();
        assert!(words.iter().all(|w| f32::from_bits(*w) == 3.0));
        runtime.host_mut().release(outputs[0]).unwrap();
    }
    let placement = runtime.placement(&version);
    assert_eq!(placement.backend, "cpu");
    assert!(
        placement
            .routed
            .is_some_and(|why| why.contains("could not build")),
    );
    assert_eq!(device.builds_of("t.big@1"), 1, "built once");
}

fn source(k: &str, value: &str) -> String {
    format!(
        "kernel t.{k}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value}; }} }}"
    )
}

#[test]
fn a_replacement_is_built_before_it_is_ready_only_where_its_version_ran() {
    let device = device();
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    runtime.set_small(0);
    runtime.set_background(true);
    let publish = |runtime: &mut Runtime, k: &str, value: &str| {
        let candidate = runtime
            .prepare_module(
                orla_compiler::compile(&source(k, value)).unwrap(),
                &source(k, value),
            )
            .unwrap();
        runtime.publish(candidate, Expect::Any).unwrap()
    };
    let ran = publish(&mut runtime, "ran", "1.0");
    publish(&mut runtime, "idle", "1.0");
    let invocation = Invocation {
        inputs: vec![],
        params: Vec::<Value>::new(),
        domain: Domain::Rows(100),
        context: Context::default(),
    };
    let outputs = runtime.execute(&ran, &invocation).unwrap();
    runtime.host_mut().release(outputs[0]).unwrap();
    assert_eq!(device.builds_of("t.ran@1"), 1);
    assert_eq!(device.builds_of("t.idle@1"), 0);

    for k in ["ran", "idle"] {
        let ticket = runtime.prepare_later(
            orla_compiler::compile(&source(k, "2.0")).unwrap(),
            source(k, "2.0"),
        );
        let candidate = loop {
            if let Some(result) = runtime.poll(ticket) {
                break result.unwrap();
            }
            std::thread::yield_now();
        };
        runtime.publish(candidate, Expect::Any).unwrap();
    }
    // Built before it was ready, so its first run waits for nothing.
    assert_eq!(device.builds_of("t.ran@1"), 2);
    // Never run on the device: left for a run to build.
    assert_eq!(device.builds_of("t.idle@1"), 0);
}

#[test]
fn a_session_without_a_worker_makes_its_replacements_ahead_built_where_they_ran() {
    let dir = std::env::temp_dir().join(format!("orla-lazy-ahead-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scene = |value: &str| {
        format!(
            "kernel t.a@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value}; }} }}
kernel t.b@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value} + 1.0; }} }}
kernel t.unused@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value}; }} }}
node a: t.a@1 {{ domain = 100; }}
node b: t.b@1 {{ domain = 100; }}
output result <- a.result;
output other <- b.result;
"
        )
    };
    std::fs::write(dir.join("scene.osn"), scene("1.0")).unwrap();
    let device = device();
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    std::fs::write(dir.join("scene.osn"), scene("2.0")).unwrap();
    let events = session.sync();
    let published = events
        .iter()
        .filter(|e| matches!(e, Event::Published { .. }))
        .count();
    assert_eq!(published, 3, "{events:?}");
    // Built before they were ready: two builds each.
    assert_eq!(device.builds_of("t.a@1"), 2);
    assert_eq!(device.builds_of("t.b@1"), 2);
    assert_eq!(device.builds_of("t.unused@1"), 0, "no node uses it");
    session.evaluate(&Context::default()).unwrap();
    let handle = session
        .cached(&orla_graph::NodeId("a".into()), "result")
        .unwrap();
    assert_eq!(
        f32::from_bits(session.field(handle).unwrap().values()[0]),
        2.0
    );
    let _ = std::fs::remove_dir_all(&dir);
}
