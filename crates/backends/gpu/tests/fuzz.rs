//! Random programs on the device against the interpreter: integer and
//! boolean programs word for word, and failures alike. Floats match only
//! to rounding, which random programs amplify without bound; the feature
//! matrix holds them. `ENTELECHY_FUZZ=<n>` runs more.

use entelechy_abi::{Capabilities, Capability, Host, Invocation, Scalar, Scope};
use entelechy_compiler::compile;
use entelechy_cpu::CpuBackend;
use entelechy_gpu::GpuBackend;
use entelechy_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

fn count() -> usize {
    std::env::var("ENTELECHY_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(150)
}

#[test]
fn random_programs_give_the_interpreters_integers_on_the_device() {
    let Ok(gpu) = GpuBackend::new() else {
        return;
    };
    let cases = entelechy_conformance::random::exact_cases(0x6f1a, count());
    let mut failures = Vec::new();
    for case in &cases {
        let module = match compile(&case.source) {
            Ok(module) => module,
            Err(error) => {
                failures.push(format!("{}: does not compile: {error}", case.name));
                continue;
            }
        };
        // A module the device declines runs elsewhere and is skipped.
        if gpu.declines(&module).is_some() {
            continue;
        }
        let mut host = Host::new(1 << 22);
        let handles: Vec<_> = case
            .inputs
            .iter()
            .map(|(schema, domain, words)| {
                host.adopt(
                    entelechy_abi::FieldData::from_values(schema.clone(), *domain, words).unwrap(),
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
        let cpu = CpuBackend.execute(&module, &invocation, &mut scope);
        let prepared = match Backend::prepare(&gpu, &module) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!("{}: the device refuses: {error}", case.name));
                continue;
            }
        };
        let mut scope = Scope::new(&mut host, all());
        let out = Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope);
        let (cpu, out) = match (cpu, out) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(_), Err(_)) => continue,
            (a, b) => {
                failures.push(format!(
                    "{}: the interpreter {} but the device {}\n{}",
                    case.name,
                    a.map_or_else(|e| format!("fails: {e}"), |_| "runs".into()),
                    b.map_or_else(|e| format!("fails: {e}"), |_| "runs".into()),
                    case.source
                ));
                continue;
            }
        };
        for (which, (a, b)) in cpu.iter().zip(&out).enumerate() {
            let schema = host.schema(*a).unwrap().clone();
            let count = host.domain(*a).unwrap().count();
            let reference = host.data(*a).unwrap().values().into_owned();
            let candidate: Vec<u32> = match host.resident(*b).unwrap() {
                Some(device) => {
                    let storage: Vec<u32> =
                        Backend::download(&gpu, device, schema.storage_words(count) * 4)
                            .unwrap()
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                    schema.unpack(&storage, count)
                }
                None => host.data(*b).unwrap().values().into_owned(),
            };
            let scalars = schema.scalars();
            let stride = scalars.len().max(1);
            let differs = reference
                .iter()
                .zip(&candidate)
                .enumerate()
                .find(|(k, (r, c))| {
                    matches!(scalars[k % stride], Scalar::Int | Scalar::Bool) && r != c
                });
            if let Some((k, (r, c))) = differs {
                failures.push(format!(
                    "{}: output {which} element {} word {}: {} against {}\n{}",
                    case.name,
                    k / stride,
                    k % stride,
                    *r as i32,
                    *c as i32,
                    case.source
                ));
            }
        }
    }
    let said: Vec<String> = failures
        .iter()
        .enumerate()
        .map(|(n, failure)| match n {
            0..3 => failure.clone(),
            _ => failure.lines().next().unwrap_or_default().to_owned(),
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} programs differ:\n{}",
        failures.len(),
        cases.len(),
        said.join("\n")
    );
}
