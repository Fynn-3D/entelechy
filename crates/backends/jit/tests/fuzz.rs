//! Random programs on the JIT against the interpreter: the same words
//! out, bit for bit, or the same failure at the same element. The
//! programs come from `orla_conformance::random` by seed, so a difference
//! found here is reproduced by its name. `ORLA_FUZZ=<n>` runs more.

use orla_abi::{Capabilities, Capability, Host, Invocation, Scope};
use orla_compiler::compile;
use orla_cpu::CpuBackend;
use orla_jit::JitBackend;
use orla_rules::Backend;

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

/// How many programs a run tries.
fn count() -> usize {
    std::env::var("ORLA_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(400)
}

#[test]
fn random_programs_run_on_the_jit_as_on_the_interpreter() {
    let jit = JitBackend::new().unwrap();
    let cases = orla_conformance::random::cases(0x0f1a, count());
    let mut failures = Vec::new();
    let mut failed_alike = 0;
    for case in &cases {
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
        let cpu = CpuBackend.execute(&module, &invocation, &mut scope);
        let prepared = match jit.prepare(&module) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!("{}: the JIT refuses: {error}", case.name));
                continue;
            }
        };
        let mut scope = Scope::new(&mut host, all());
        let out = jit.execute(prepared.as_ref(), &module, &invocation, &mut scope);
        match (cpu, out) {
            (Ok(a), Ok(b)) => {
                for (which, (a, b)) in a.iter().zip(&b).enumerate() {
                    let reference = host.data(*a).unwrap().values().into_owned();
                    let candidate = host.data(*b).unwrap().values().into_owned();
                    let scalars = host.schema(*a).unwrap().scalars().to_vec();
                    if let Some(mismatch) =
                        orla_conformance::compare(&scalars, &reference, &candidate, 0.0, 0.0)
                    {
                        failures.push(format!(
                            "{}: output {which}: {mismatch}\n{}",
                            case.name, case.source
                        ));
                    }
                }
            }
            (Err(a), Err(b)) if a == b => failed_alike += 1,
            (a, b) => failures.push(format!(
                "{}: the interpreter {} but the JIT {}\n{}",
                case.name,
                a.map_or_else(|e| format!("fails: {e}"), |_| "runs".into()),
                b.map_or_else(|e| format!("fails: {e}"), |_| "runs".into()),
                case.source
            )),
        }
    }
    // Each failure's first line, and the whole program for the first few.
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
    // The generator must mostly write programs that run, or it tests the
    // failure path alone.
    assert!(
        failed_alike * 2 < cases.len(),
        "{failed_alike} of {} failed",
        cases.len()
    );
}
