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
    let mut failures = Vec::new();
    for case in orla_matrix::cases() {
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
                    orla_abi::FieldData::from_words(schema.clone(), *domain, words.clone())
                        .unwrap(),
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
        let prepared = match Backend::prepare(&gpu, &module) {
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
        let out = match Backend::execute(&gpu, prepared.as_ref(), &module, &invocation, &mut scope)
        {
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
            let reference = host.data(*a).unwrap().words().to_vec();
            let scalars = host.schema(*a).unwrap().scalars().to_vec();
            let candidate: Vec<u32> = match host.resident(*b).unwrap() {
                Some(device) => Backend::download(&gpu, device, reference.len() * 4)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                None => host.data(*b).unwrap().words().to_vec(),
            };
            if let Some(mismatch) =
                orla_matrix::compare(&scalars, &reference, &candidate, 2e-4, 1e-3)
            {
                failures.push(format!("{}: output {which}: {mismatch}", case.name));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        orla_matrix::cases().len(),
        failures.join("\n")
    );
}
