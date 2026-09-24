//! The feature matrix on the JIT: every feature under every mode, bit
//! for bit against the interpreter.

use orla_abi::{Capabilities, Capability, Host, Invocation, Scope};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_jit::JitBackend;
use orla_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

#[test]
fn every_feature_under_every_mode_matches_the_interpreter() {
    let cases = orla_conformance::cases();
    let failures = differences(&cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn every_64_bit_feature_under_every_mode_matches_the_interpreter() {
    let cases = orla_conformance::wide_cases();
    let failures = differences(&cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn every_long_feature_matches_the_interpreter() {
    let cases = orla_conformance::long_cases();
    let failures = differences(&cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn every_element_shape_a_scatter_holds_matches_the_interpreter() {
    let cases = orla_conformance::shape_cases();
    let failures = differences(&cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// What each case's JIT run says against the interpreter's, bit for bit.
fn differences(cases: &[orla_conformance::Case]) -> Vec<String> {
    let jit = JitBackend::new().unwrap();
    let mut failures = Vec::new();
    for case in cases {
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
            let reference = host.data(*a).unwrap().values().into_owned();
            let candidate = host.data(*b).unwrap().values().into_owned();
            let scalars = host.schema(*a).unwrap().scalars().to_vec();
            if let Some(mismatch) =
                orla_conformance::compare(&scalars, &reference, &candidate, 0.0, 0.0)
            {
                failures.push(format!("{}: output {which}: {mismatch}", case.name));
            }
        }
    }
    failures
}
