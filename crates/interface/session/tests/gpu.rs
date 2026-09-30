//! The GPU held to the interpreter on graphs keeping state on the
//! device. These need the scene language and the runtime together, held
//! by the session layer.

use orla_abi::{Context, Value};
use orla_gpu::GpuBackend;
use orla_rules::Backend;

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
    runtime: &mut orla_kernel::Runtime,
    source: &str,
) -> std::sync::Arc<orla_kernel::KernelVersion> {
    let module = orla_compiler::compile(source).unwrap_or_else(|error| panic!("{error}"));
    runtime.load(module, source).unwrap()
}

#[test]
fn chains_and_state_stay_resident_until_observed() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, backend);
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
    let graph = orla_scene::parse(
        "node seed: test.zero@1 { domain = vec2(8, 8); }
         node count: test.increment@1 { previous <~ count.result from seed.result; }
         node soft: test.blur@1 { source <- count.result; }
         output result <- soft.result;",
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let empty: std::collections::BTreeMap<String, orla_abi::Handle> =
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
            assert_eq!(runtime.host().resident_count(), runtime.host().live() - 1);
        }
    }
    // The observed image is an input again, re-uploaded once.
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
fn fields_flow_through_the_gpu_exactly() {
    let Some(gpu) = gpu() else { return };
    let backend: Box<dyn Backend> = Box::new(gpu);
    let mut runtime = orla_kernel::Runtime::with_backend(1 << 24, backend);
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
    let graph = orla_scene::parse(
        "node seed: test.seed@1 { domain = 8; }
         node step: test.step@1 { previous <~ step.result from seed.result; }
         node draw: test.draw@1 { source <- step.result; domain = vec2(8, 1); }
         output result <- draw.result;"
            .replace("source <-", "particles <-")
            .as_str(),
    )
    .unwrap();
    let mut evaluator = orla_graph::Evaluator::new();
    let none: std::collections::BTreeMap<String, orla_abi::Handle> = Default::default();
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
