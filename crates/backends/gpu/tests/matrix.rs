//! The feature matrix on the GPU: every feature under every mode, ints
//! and booleans exact and floats to rounding against the interpreter.

use orla_abi::{Capabilities, Capability, Host, Invocation, Scope};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_gpu::GpuBackend;
use orla_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

#[test]
fn every_feature_under_every_mode_matches_the_interpreter() {
    let Some(gpu) = GpuBackend::new().ok() else {
        return;
    };
    let cases = orla_conformance::cases();
    let failures = differences(&gpu, &cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn every_long_feature_matches_the_interpreter_where_the_device_has_longs() {
    let Some(gpu) = GpuBackend::new().ok() else {
        return;
    };
    let cases = orla_conformance::long_cases();
    let module = compile(&cases[0].source).unwrap();
    if gpu.declines(&module).is_some() {
        return;
    }
    let failures = differences(&gpu, &cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// What each case's GPU run says against the interpreter's: integers
/// and booleans exact, floats to rounding.
fn differences(gpu: &GpuBackend, cases: &[orla_conformance::Case]) -> Vec<String> {
    let mut failures = Vec::new();
    for case in cases {
        let module = match compile(&case.source) {
            Ok(module) => module,
            Err(error) => {
                failures.push(format!("{}: does not compile: {error}", case.name));
                continue;
            }
        };
        let mut host = Host::new(1 << 22);
        let handles: Vec<_> = case
            .inputs
            .iter()
            .map(|(schema, domain, words)| {
                host.adopt(
                    orla_abi::FieldData::from_values(schema.clone(), *domain, words).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let invocation = Invocation {
            inputs: handles,
            params: case.params.clone(),
            domain: case.domain,
            context: case.context,
        };
        let mut scope = Scope::new(&mut host, all());
        let cpu = match CpuBackend.execute(&module, &invocation, &mut scope) {
            Ok(outputs) => outputs,
            Err(error) => {
                failures.push(format!("{}: the interpreter fails: {error}", case.name));
                continue;
            }
        };
        let prepared = match Backend::prepare(gpu, &module) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!(
                    "{}: the GPU refuses: {}",
                    case.name,
                    error.lines().take(6).collect::<Vec<_>>().join(" | ")
                ));
                continue;
            }
        };
        let mut scope = Scope::new(&mut host, all());
        let out = match Backend::execute(gpu, prepared.as_ref(), &module, &invocation, &mut scope) {
            Ok(outputs) => outputs,
            Err(error) => {
                failures.push(format!("{}: the GPU fails: {error}", case.name));
                continue;
            }
        };
        if cpu.len() != out.len() {
            failures.push(format!(
                "{}: {} output(s) against the interpreter's {}",
                case.name,
                out.len(),
                cpu.len()
            ));
            continue;
        }
        for (which, (a, b)) in cpu.iter().zip(&out).enumerate() {
            // Compared as the Kernel's words: what a device kept narrow is
            // downloaded as kept and widened by the Host's rule.
            let schema = host.schema(*a).unwrap().clone();
            let count = host.domain(*a).unwrap().count();
            let reference = host.data(*a).unwrap().values().into_owned();
            let scalars = schema.scalars().to_vec();
            let candidate: Vec<u32> = match host.resident(*b).unwrap() {
                Some(device) => {
                    let storage: Vec<u32> =
                        Backend::download(gpu, device, schema.storage_words(count) * 4)
                            .unwrap()
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                    // What a device kept narrow is the Host's rule applied
                    // to the same words, so it is the same bits.
                    if !schema.is_wide() {
                        let kept = host.data(*a).unwrap().storage().to_vec();
                        if storage != kept {
                            failures.push(format!(
                                "{}: output {which}: kept as {:08x?} against {:08x?}",
                                case.name, storage, kept
                            ));
                        }
                    }
                    schema.unpack(&storage, count)
                }
                None => host.data(*b).unwrap().values().into_owned(),
            };
            if let Some(mismatch) =
                orla_conformance::compare(&scalars, &reference, &candidate, 2e-4, 1e-3)
            {
                failures.push(format!("{}: output {which}: {mismatch}", case.name));
            }
        }
    }
    failures
}

/// Every element shape a scatter's target may hold, on the device where
/// it takes it. It declines only a target of Longs, and of those takes
/// every one under `min` or `max` whose scalars are all Longs, where it
/// has 64-bit atomics.
#[test]
fn every_element_shape_a_scatter_holds_matches_the_interpreter() {
    let Some(gpu) = GpuBackend::new().ok() else {
        return;
    };
    let cases = orla_conformance::shape_cases();
    let mut taken = Vec::new();
    for case in &cases {
        let module = compile(&case.source).unwrap();
        match gpu.declines(&module) {
            None => taken.push(case.clone()),
            Some(why) => assert!(
                case.name.contains("Long")
                    && !case.name.contains("with min")
                    && !case.name.contains("with max")
                    || !gpu_takes_long_extremes(&gpu),
                "{}: declined: {why}",
                case.name
            ),
        }
    }
    let failures = differences(&gpu, &taken);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        taken.len(),
        failures.join("\n")
    );
}

/// Whether the device lands Longs under `min` and `max`: whether it takes
/// the smallest such scatter.
fn gpu_takes_long_extremes(gpu: &GpuBackend) -> bool {
    let module = compile(
        "kernel probe.low@1 { requires { field.read; field.write; }
         input keys: Rows<Int>; input target: Rows<Long>; output result: Rows<Long>;
         struct Pair { at: Int; value: Long; }
         scatter into target with min { return Pair { at: sample(keys), value: 0l }; } }",
    )
    .unwrap();
    gpu.declines(&module).is_none()
}

#[test]
fn a_64_bit_feature_is_not_taken_and_says_why() {
    let Some(gpu) = GpuBackend::new().ok() else {
        return;
    };
    for case in orla_conformance::wide_cases() {
        let module = compile(&case.source).unwrap();
        assert!(gpu.declines(&module).is_some(), "{}: taken", case.name);
        let Err(error) = Backend::prepare(&gpu, &module) else {
            panic!("{}: prepared", case.name);
        };
        assert!(error.contains("holds a Double"), "{}: {error}", case.name);
    }
    // Everything else it takes, but a scan, which it leaves to an
    // ordered backend where there is one and runs where there is none.
    for case in orla_conformance::cases() {
        let module = compile(&case.source).unwrap();
        match module.manifest.mode {
            orla_ir::Mode::Scan => {
                let why = gpu.declines(&module).unwrap_or_default();
                assert!(
                    why.contains("a scan runs its steps in order"),
                    "{}: {why}",
                    case.name
                );
                assert!(Backend::prepare(&gpu, &module).is_ok(), "{}", case.name);
            }
            _ => assert!(gpu.declines(&module).is_none(), "{}: not taken", case.name),
        }
    }
}
