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

/// Each case's JIT run against the interpreter's, bit for bit.
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
        let mut host = Host::new(case.room());
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

/// An integer sum is judged by each element's whole total on every
/// backend: a total that fits passes though a partial sum overflows; one
/// that does not fails at the first such element, in any addition order.
#[test]
fn integer_sums_are_whole_totals_on_jit_as_everywhere() {
    let Some(backend) = orla_jit::JitBackend::new().ok() else {
        return;
    };
    let failures: Vec<String> = orla_conformance::sum_cases()
        .iter()
        .filter_map(|(case, want)| {
            want.judge(&summed(&backend, case))
                .map(|why| format!("{}: {why}", case.name))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs one integer sum case on `backend`: the output's words, or why not.
fn summed(
    backend: &dyn orla_rules::Backend,
    case: &orla_conformance::Case,
) -> Result<Vec<u32>, orla_rules::ExecutionError> {
    let module =
        orla_compiler::compile(&case.source).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let mut host = orla_abi::Host::new(1 << 22);
    let inputs = case
        .inputs
        .iter()
        .map(|(schema, domain, words)| {
            host.adopt(orla_abi::FieldData::from_values(schema.clone(), *domain, words).unwrap())
                .unwrap()
        })
        .collect();
    let invocation = orla_abi::Invocation {
        inputs,
        params: case.params.clone(),
        domain: case.domain,
        context: case.context,
    };
    let all: orla_abi::Capabilities = orla_abi::Capability::ALL.into_iter().collect();
    let mut scope = orla_abi::Scope::new(&mut host, all);
    let prepared = backend
        .prepare(&module)
        .map_err(orla_rules::ExecutionError::Device)?;
    let outputs = backend.execute(prepared.as_ref(), &module, &invocation, &mut scope)?;
    Ok(scope.values(outputs[0])?.into_owned())
}

/// Extents past 2^24, which a float cannot hold whole, under every mode:
/// each element, count, size and domain as the interpreter gives them.
#[test]
fn extents_past_a_floats_integers_match_the_interpreter() {
    let cases = orla_conformance::large_cases();
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
fn every_case_run_twice_writes_the_same_words() {
    let backend = JitBackend::new().unwrap();
    assert!(backend.repeats_exactly());
    let cases: Vec<_> = orla_conformance::repeat_cases()
        .into_iter()
        .filter_map(|case| compile(&case.source).ok().map(|module| (case, module)))
        .collect();
    let failures = orla_conformance::unrepeated(&backend, &cases);
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
