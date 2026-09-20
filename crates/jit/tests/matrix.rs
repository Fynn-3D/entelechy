//! The feature matrix on the JIT: every feature under every mode, bit
//! for bit against the interpreter.

use orla_abi::{Capabilities, Capability, Host, Invocation, Scope};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_jit::JitBackend;
use orla_runtime::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

#[test]
fn every_feature_under_every_mode_matches_the_interpreter() {
    let jit = JitBackend::new().unwrap();
    let mut failures = Vec::new();
    for case in orla_matrix::cases() {
        let module = match compile(&case.source) {
            Ok(module) => module,
            Err(error) => {
                failures.push(format!(
                    "{}: does not compile: {error}\n{}",
                    case.name, case.source
                ));
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
        let prepared = match jit.prepare(&module) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!("{}: the JIT refuses: {error}", case.name));
                continue;
            }
        };
        let mut scope = Scope::new(&mut host, all());
        let out = match jit.execute(prepared.as_ref(), &module, &invocation, &mut scope) {
            Ok(outputs) => outputs,
            Err(error) => {
                failures.push(format!("{}: the JIT fails: {error}", case.name));
                continue;
            }
        };
        for (which, (a, b)) in cpu.iter().zip(&out).enumerate() {
            let reference = host.data(*a).unwrap().words().to_vec();
            let candidate = host.data(*b).unwrap().words().to_vec();
            let scalars = host.schema(*a).unwrap().scalars().to_vec();
            if let Some(mismatch) = orla_matrix::compare(&scalars, &reference, &candidate, 0.0, 0.0)
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
