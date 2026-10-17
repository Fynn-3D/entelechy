//! The GPU held to the interpreter on graphs keeping state on the
//! device. These need the scene language and the runtime together, held
//! by the session layer.

use entelechy_abi::{Context, Value};
use entelechy_gpu::GpuBackend;
use entelechy_rules::Backend;

fn gpu() -> Option<GpuBackend> {
    match GpuBackend::new() {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

/// Compiles and publishes a Kernel as a session does: compiling is the
/// caller's step, the runtime takes validated IR.
fn load(
    runtime: &mut entelechy_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<entelechy_kernel::KernelVersion> {
    let module = entelechy_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

#[test]
fn chains_and_state_stay_resident_until_observed() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.zero@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    );
    load(
        &mut runtime,
        "kernel test.increment@1 { requires { field.read; field.write; } input previous: Image; output result: Image; compute { let c = sample(previous); return vec4(c.r + 1.0, c.g, c.b, 1.0); } }",
    );
    load(
        &mut runtime,
        "kernel test.blur@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = coord(); return (fetch(source, c) + fetch(source, c + vec2(1.0, 0.0))) * 0.5; } }",
    );
    let graph = entelechy_scene::parse(
        "node seed: test.zero@1 { domain = vec2(8, 8); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node soft: test.blur@1 { source <- count.result; }
         output result <- soft.result;",
    )
    .unwrap();
    let mut evaluator = entelechy_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
        std::collections::BTreeMap::new();
    let no_params: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for frame in 1..=3 {
        let report = evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &empty,
                &Context::default(),
                &no_params,
            )
            .unwrap();
        assert_eq!(
            runtime.host().resident_count(),
            runtime.host().live(),
            "nothing came back to the Host during evaluation"
        );
        if frame == 3 {
            let image = runtime.image(report.output).unwrap();
            assert_eq!(image.pixels()[0][0], 3.0, "state advanced on the device");
            // Observed, it is in Host memory and still on the device.
            assert_eq!(runtime.host().resident_count(), runtime.host().live());
        }
    }
    // The observed image is an input again, read from the device as is.
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &empty,
            &Context::default(),
            &no_params,
        )
        .unwrap();
    assert_eq!(runtime.image(report.output).unwrap().pixels()[0][0], 4.0);
}

#[test]
fn state_at_a_fixed_point_on_the_device_stops_advancing() {
    // State that keeps itself: the run its committed state calls for gives
    // the same words, compared on the device, so the state keeps its
    // Field, the node and what reads it are not run again, and nothing
    // comes back to the Host.
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.ramp@1 { requires { field.write; } output result: Image; compute { return vec4(coord().x, coord().y, 0.0, 1.0); } }",
    );
    load(
        &mut runtime,
        "kernel test.keep@1 { requires { field.read; field.write; } input previous: Image; output result: Image; compute { return sample(previous); } }",
    );
    load(
        &mut runtime,
        "kernel test.double@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { return sample(source) * 2.0; } }",
    );
    let graph = entelechy_scene::parse(
        "node seed: test.ramp@1 { domain = vec2(64, 64); }
         node kept: test.keep@1 { previous <~ kept.result from seed.result; }
         node twice: test.double@1 { source <- kept.result; }
         output result <- twice.result;",
    )
    .unwrap();
    let mut evaluator = entelechy_graph::Evaluator::new();
    evaluator.set_fusion(false);
    let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
        std::collections::BTreeMap::new();
    let no_params: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    let mut frame = || {
        evaluator
            .evaluate(
                &graph,
                &mut runtime,
                &empty,
                &Context::default(),
                &no_params,
            )
            .unwrap()
    };
    let names =
        |nodes: &[entelechy_graph::NodeId]| nodes.iter().map(|n| n.0.clone()).collect::<Vec<_>>();
    let first = frame();
    assert_eq!(names(&first.executed), ["seed", "kept", "twice"]);
    // Its state changed from the seed to its own first Field: it runs for
    // that alone, gives the same words, and keeps its Field.
    let second = frame();
    assert_eq!(names(&second.executed), ["kept"]);
    assert_eq!(names(&second.reused.to_vec()), ["seed", "twice"]);
    assert_eq!(second.output, first.output);
    let third = frame();
    assert!(third.executed.is_empty(), "the state is at rest");
    assert_eq!(
        runtime.host().resident_count(),
        runtime.host().live(),
        "nothing came back to the Host"
    );
    let image = runtime.image(third.output).unwrap();
    assert_eq!(image.pixels()[3 * 64 + 5][0], 10.0);
}

#[test]
fn fields_flow_through_the_gpu_exactly() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.seed@1 { requires { field.write; } output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let i = element(); return Particle { position: vec2(float(i), float(i) * 0.5), life: i * 3, alive: i / 2 * 2 == i }; } }",
    );
    load(
        &mut runtime,
        "kernel test.step@1 { requires { field.read; field.write; } input previous: Rows<Particle>; output result: Rows<Particle>; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = sample(previous); let n = at(previous, element() + 1); return Particle { position: p.position + n.position, life: p.life - 1 + count(previous), alive: !p.alive }; } }",
    );
    load(
        &mut runtime,
        "kernel test.draw@1 { requires { field.read; field.write; } input particles: Rows<Particle>; output result: Image; struct Particle { position: Vec2; life: Int; alive: Bool; } compute { let p = at(particles, int(coord().x)); return vec4(p.position.x, p.position.y, float(p.life), float(p.alive)); } }",
    );
    let graph = entelechy_scene::parse(
        "node seed: test.seed@1 { domain = 8; }
         node step: test.step@1 { previous <~ step.result from seed.result; }
         node draw: test.draw@1 { source <- step.result; domain = vec2(8, 1); }
         output result <- draw.result;"
            .replace("source <-", "particles <-")
            .as_str(),
    )
    .unwrap();
    let mut evaluator = entelechy_graph::Evaluator::new();
    let none: std::collections::BTreeMap<String, entelechy_abi::Handle> = Default::default();
    let params: std::collections::BTreeMap<String, Value> = Default::default();
    let report = evaluator
        .evaluate(&graph, &mut runtime, &none, &Context::default(), &params)
        .unwrap();
    assert_eq!(runtime.host().resident_count(), runtime.host().live());
    let image = runtime.image(report.output).unwrap();
    // Element 2 after one step: position (2,1)+(3,1.5), life 6-1+8,
    // alive flipped.
    assert_eq!(image.pixels()[2], [5.0, 2.5, 13.0, 0.0]);
    // Element 7 reads its clamped neighbour (itself).
    assert_eq!(image.pixels()[7], [14.0, 7.0, 28.0, 1.0]);
}

#[test]
fn a_small_output_found_unchanged_on_the_device_spares_what_reads_it() {
    // A choice made each frame from state that moves (`pick`, one word,
    // run on the device for fresh inputs) is compared there: unchanged,
    // the costly Field made from it is not made again; changed, it is.
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.zero@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }",
    );
    load(
        &mut runtime,
        "kernel test.clock@1 { requires { field.read; field.write; time.read; } input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) * 0.0 + time(); } }",
    );
    load(
        &mut runtime,
        "kernel test.pick@1 { requires { field.read; field.write; } input clock: Rows<Float>; param every: Float = 0.0; output result: Rows<Int>; compute { return 3 + int(at(clock, 0) * every); } }",
    );
    load(
        &mut runtime,
        "kernel test.costly@1 { requires { field.read; field.write; } input pick: Rows<Int>; output result: Image; compute { return vec4(float(at(pick, 0)) + coord().x, 0.0, 0.0, 1.0); } }",
    );
    let names =
        |nodes: &[entelechy_graph::NodeId]| nodes.iter().map(|n| n.0.clone()).collect::<Vec<_>>();
    let mut run = |every: f32| {
        let graph = entelechy_scene::parse(&format!(
            "node zero: test.zero@1 {{ domain = 1; }}
             node clock: test.clock@1 {{ previous <~ clock.result from zero.result; domain = 1; }}
             node pick: test.pick@1 {{ clock <- clock.result; every = {every}; domain = 1; }}
             node costly: test.costly@1 {{ pick <- pick.result; domain = vec2(256, 256); }}
             output result <- costly.result;"
        ))
        .unwrap();
        let mut evaluator = entelechy_graph::Evaluator::new();
        evaluator.set_fusion(false);
        let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
            std::collections::BTreeMap::new();
        let no_params: std::collections::BTreeMap<String, Value> =
            std::collections::BTreeMap::new();
        (1..=4)
            .map(|k| {
                let context = Context {
                    time: k as f64,
                    ..Context::default()
                };
                let report = evaluator
                    .evaluate(&graph, &mut runtime, &empty, &context, &no_params)
                    .unwrap();
                names(&report.executed).contains(&"costly".to_owned())
            })
            .collect::<Vec<bool>>()
    };
    // The pick stays 3: made once (the first frame, and the one before the
    // first comparison finds it at rest), then spared.
    let steady = run(0.0);
    assert!(steady[0] && !steady[2] && !steady[3], "{steady:?}");
    // The pick moves with the clock: made every frame.
    assert_eq!(run(1.0), [true, true, true, true]);
}

/// A device that counts its comparisons, otherwise the GPU.
struct Counted {
    gpu: GpuBackend,
    compared: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Backend for Counted {
    fn name(&self) -> &'static str {
        self.gpu.name()
    }
    fn declines(&self, module: &entelechy_ir::Validated) -> Option<String> {
        self.gpu.declines(module)
    }
    fn repeats_exactly(&self) -> bool {
        self.gpu.repeats_exactly()
    }
    fn uploaded(&self) -> u64 {
        self.gpu.uploaded()
    }
    fn prepare(
        &self,
        module: &entelechy_ir::Validated,
    ) -> Result<Box<dyn entelechy_rules::Prepared>, String> {
        self.gpu.prepare(module)
    }
    fn finish(&self, prepared: &dyn entelechy_rules::Prepared) {
        self.gpu.finish(prepared);
    }
    fn execute(
        &self,
        prepared: &dyn entelechy_rules::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_abi::Invocation,
        scope: &mut entelechy_abi::Scope<'_>,
    ) -> Result<Vec<entelechy_abi::Handle>, entelechy_rules::ExecutionError> {
        self.gpu.execute(prepared, module, invocation, scope)
    }
    fn execute_checked(
        &self,
        prepared: &dyn entelechy_rules::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_abi::Invocation,
        scope: &mut entelechy_abi::Scope<'_>,
    ) -> Result<entelechy_rules::backend::Checked, entelechy_rules::ExecutionError> {
        self.gpu
            .execute_checked(prepared, module, invocation, scope)
    }
    fn has_device(&self) -> bool {
        true
    }
    fn settle_device(&self) {
        self.gpu.settle_device();
    }
    fn submitted(&self) -> Option<Box<dyn entelechy_rules::Check>> {
        self.gpu.submitted()
    }
    fn download(
        &self,
        device: &(dyn std::any::Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        self.gpu.download(device, byte_len)
    }
    fn same_resident(
        &self,
        a: &(dyn std::any::Any + Send + Sync),
        b: &(dyn std::any::Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        self.compared
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.gpu.same_resident(a, b, byte_len)
    }
}

#[test]
fn a_small_output_found_unchanged_while_sparing_nothing_rests() {
    // The pick stays 3 while the clock moves. Where the costly Field reads
    // the clock as well, finding the pick unchanged spares nothing, and
    // the pick is compared seldom; where it reads the pick alone, it is
    // compared each frame.
    let compares = |reads_clock: bool| {
        let gpu = gpu()?;
        let compared = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend: Box<dyn Backend> = Box::new(Counted {
            gpu,
            compared: std::sync::Arc::clone(&compared),
        });
        let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
        runtime.set_small(0);
        load(
            &mut runtime,
            "kernel test.zero@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }",
        );
        load(
            &mut runtime,
            "kernel test.clock@1 { requires { field.read; field.write; time.read; } input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) * 0.0 + time(); } }",
        );
        load(
            &mut runtime,
            "kernel test.pick@1 { requires { field.read; field.write; } input clock: Rows<Float>; output result: Rows<Int>; compute { return 3 + int(at(clock, 0) * 0.0); } }",
        );
        load(
            &mut runtime,
            "kernel test.costly@1 { requires { field.read; field.write; } input pick: Rows<Int>; input clock: Rows<Float>; param every: Float = 0.0; output result: Image; compute { return vec4(float(at(pick, 0)) + at(clock, 0) * every + coord().x, 0.0, 0.0, 1.0); } }",
        );
        let costly_clock = if reads_clock {
            "clock.result"
        } else {
            "zero.result"
        };
        let graph = entelechy_scene::parse(&format!(
            "node zero: test.zero@1 {{ domain = 1; }}
             node clock: test.clock@1 {{ previous <~ clock.result from zero.result; domain = 1; }}
             node pick: test.pick@1 {{ clock <- clock.result; domain = 1; }}
             node costly: test.costly@1 {{ pick <- pick.result; clock <- {costly_clock}; every = 1.0; domain = vec2(64, 64); }}
             output result <- costly.result;"
        ))
        .unwrap();
        let mut evaluator = entelechy_graph::Evaluator::new();
        evaluator.set_fusion(false);
        let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
            std::collections::BTreeMap::new();
        let no_params: std::collections::BTreeMap<String, Value> =
            std::collections::BTreeMap::new();
        for k in 1..=40 {
            let context = Context {
                time: k as f64,
                ..Context::default()
            };
            evaluator
                .evaluate(&graph, &mut runtime, &empty, &context, &no_params)
                .unwrap();
        }
        Some(compared.load(std::sync::atomic::Ordering::SeqCst))
    };
    let (Some(sparing), Some(sparing_nothing)) = (compares(false), compares(true)) else {
        return;
    };
    assert!(sparing >= 35, "{sparing}");
    assert!(sparing_nothing <= 16, "{sparing_nothing}");
}

#[test]
fn every_wait_for_the_device_is_counted() {
    // A download waits for the run before it; the time it was blocked is
    // the backend's to tell, summed.
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    load(
        &mut runtime,
        "kernel test.heavy@1 { requires { field.write; } output result: Rows<Float>; compute { return loop 4000 with a = float(element()) { a * 0.999 + sin(a) }; } }",
    );
    let graph = entelechy_scene::parse(
        "node heavy: test.heavy@1 { domain = 65536; }
         output result <- heavy.result;",
    )
    .unwrap();
    let mut evaluator = entelechy_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
        std::collections::BTreeMap::new();
    let no_params: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    let before = runtime.device_waited();
    let report = evaluator
        .evaluate(
            &graph,
            &mut runtime,
            &empty,
            &Context::default(),
            &no_params,
        )
        .unwrap();
    runtime.materialize(report.output).unwrap();
    assert!(runtime.device_waited() > before);
}

#[test]
fn a_clocked_frame_gives_the_devices_busy_time_without_waiting_for_it() {
    // Clocked, each frame's runs are stamped by the device's clock and read
    // once it has done them: a frame later, its busy time is there.
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 24, backend);
    runtime.set_small(0);
    load(
        &mut runtime,
        "kernel test.wave@1 { requires { field.write; time.read; } output result: Image; compute { let p = coord(); return vec4(sin(p.x * 0.01 + time()), cos(p.y * 0.01), 0.0, 1.0); } }",
    );
    let graph = entelechy_scene::parse(
        "node wave: test.wave@1 { domain = vec2(512, 512); } output result <- wave.result;",
    )
    .unwrap();
    let mut evaluator = entelechy_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, entelechy_abi::Handle> =
        std::collections::BTreeMap::new();
    let no_params: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    assert_eq!(runtime.frame_device_time(), None, "nothing clocked yet");
    runtime.set_clocked(true);
    let mut busy = None;
    for k in 0..200 {
        let context = Context {
            time: k as f64 * 0.1,
            ..Context::default()
        };
        evaluator
            .evaluate(&graph, &mut runtime, &empty, &context, &no_params)
            .unwrap();
        runtime.end_frame();
        busy = runtime.frame_device_time();
        if busy.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    // An adapter without timestamps keeps no clock: nothing to read.
    let Some(busy) = busy else { return };
    assert!(
        busy > std::time::Duration::ZERO && busy < std::time::Duration::from_secs(1),
        "{busy:?}"
    );
    runtime.set_clocked(false);
    assert_eq!(
        runtime.frame_device_time(),
        None,
        "unclocked, nothing is kept"
    );
}

#[test]
fn a_clocked_frames_device_time_is_shared_out_to_the_nodes_that_ran() {
    // Two nodes on the device, clocked: once the device has done a frame,
    // each has its runs' busy time by name, and with what ran between
    // nodes they make up the frame's.
    let Some(gpu) = gpu() else { return };
    let dir = std::env::temp_dir().join(format!("entelechy-node-clock-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("lib.ent"),
        "kernel t.wave@1 { requires { field.write; time.read; } output result: Image; compute { let p = coord(); return vec4(sin(p.x * 0.01 + time()), cos(p.y * 0.01), 0.0, 1.0); } }
kernel t.blur@1 { requires { field.read; field.write; } input picture: Image; output result: Image; compute { let p = coord(); let n = size(picture); return (at(picture, p) + at(picture, min(p + vec2(1.0, 0.0), n - vec2(1.0, 1.0)))) * 0.5; } }",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"lib.ent\";\nnode wave: t.wave@1 { domain = vec2(512, 512); }\nnode blur: t.blur@1 { picture <- wave.result; }\noutput result <- blur.result;\n",
    )
    .unwrap();
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 26, backend);
    runtime.set_small(0);
    let mut session = entelechy_session::Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    assert_eq!(session.device_times(), None, "nothing clocked yet");
    session.set_clocked(true);
    let mut told = None;
    for k in 0..300 {
        let context = Context {
            time: k as f64 * 0.1,
            ..Context::default()
        };
        session.evaluate(&context).unwrap();
        told = session.device_times();
        if told.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let _ = std::fs::remove_dir_all(&dir);
    // An adapter that tells nothing done keeps no clock.
    let Some(times) = told else { return };
    let names: Vec<String> = times.nodes.iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(names, ["wave", "blur"], "in evaluation order, by name");
    assert!(
        times
            .nodes
            .iter()
            .all(|(_, busy)| *busy > std::time::Duration::ZERO),
        "{times:?}"
    );
    let shared: std::time::Duration = times
        .nodes
        .iter()
        .map(|(_, busy)| *busy)
        .sum::<std::time::Duration>()
        + times.between;
    assert_eq!(
        shared, times.busy,
        "the nodes and what ran between them make up the frame"
    );
    session.set_clocked(false);
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(session.device_times(), None, "unclocked, nothing is told");
}
