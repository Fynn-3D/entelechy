//! The optimizer is held to the interpreter: every feature under every
//! mode computes the same words from the optimized IR as from the IR
//! the lowering made, and the optimized IR is smaller.

use orla_abi::{Capabilities, Capability, Host, Invocation, Scope};
use orla_compiler::{compile, compile_unoptimized};
use orla_cpu::CpuBackend;
use orla_ir::{Instruction, Op};

fn all() -> Capabilities {
    Capability::ALL.into_iter().collect()
}

fn count(instructions: &[Instruction]) -> usize {
    instructions
        .iter()
        .map(|i| {
            1 + match &i.op {
                Op::If {
                    then, otherwise, ..
                } => count(&then.body) + count(&otherwise.body),
                Op::Match {
                    arms, otherwise, ..
                } => {
                    arms.iter().map(|(_, b)| count(&b.body)).sum::<usize>() + count(&otherwise.body)
                }
                Op::Loop { until, body, .. } => {
                    until.as_ref().map_or(0, |b| count(&b.body)) + count(&body.body)
                }
                _ => 0,
            }
        })
        .sum()
}

#[test]
fn every_feature_under_every_mode_computes_the_same_words_optimized() {
    let mut failures = Vec::new();
    let (mut before, mut after) = (0, 0);
    for case in orla_matrix::cases() {
        let plain = compile_unoptimized(&case.source).unwrap();
        let optimized = compile(&case.source).unwrap();
        let size = |module: &orla_ir::Validated| {
            module
                .functions
                .iter()
                .chain([&module.entry])
                .chain(module.merge.iter())
                .chain(module.init.iter())
                .map(|f| count(&f.body))
                .sum::<usize>()
        };
        before += size(&plain);
        after += size(&optimized);
        let run = |module: &orla_ir::Validated| -> Result<Vec<Vec<u32>>, String> {
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
            let outputs = CpuBackend
                .execute(module, &invocation, &mut scope)
                .map_err(|e| e.to_string())?;
            Ok(outputs
                .iter()
                .map(|handle| host.data(*handle).unwrap().words().to_vec())
                .collect())
        };
        match (run(&plain), run(&optimized)) {
            (Ok(a), Ok(b)) if a == b => {}
            (Ok(_), Ok(_)) => failures.push(format!("{}: the words differ", case.name)),
            (a, b) => failures.push(format!("{}: {a:?} against {b:?}", case.name)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        after < before,
        "{after} instructions optimized against {before}"
    );
    eprintln!("feature matrix: {before} instructions lowered, {after} optimized");
}
