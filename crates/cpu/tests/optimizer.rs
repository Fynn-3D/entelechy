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

/// A loop whose count and initial value are known and whose body only
/// computes is run once, when the Kernel is compiled, rather than for
/// every element. A table built in a loop is what this is for.
#[test]
fn a_loop_over_literals_is_run_when_the_kernel_is_compiled() {
    let loops = |body: &[Instruction]| {
        body.iter()
            .filter(|instruction| matches!(instruction.op, Op::Loop { .. }))
            .count()
    };
    let table = "kernel test.table@1 { requires { field.write; } output result: Rows<Int>; fn side(bit: Int) -> Int { return loop 16 with m = 0 { if ((index() >>> bit) & 1) == 1 { m | (1 << index()) } else { m } }; } compute { return side(0) + side(1) * 2 + element(); } }";
    let plain = compile_unoptimized(table).unwrap();
    assert_eq!(
        plain
            .functions
            .iter()
            .map(|f| loops(&f.body))
            .sum::<usize>(),
        1,
        "the lowering writes the loop"
    );
    let module = compile(table).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(loops(&module.entry.body), 0, "{module}");
    // 0xAAAA + 0xCCCC * 2, the masks of two sides, added up here.
    let text = module.entry.to_string();
    assert!(text.contains("148546"), "{text}");

    // A loop that reads a Field is not run here, and neither is one
    // that would fail: the failure belongs to the element that has it.
    let reads = "kernel test.reads@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return loop 8 with a = 0 { a + at(rows, index()) }; } }";
    assert_eq!(loops(&compile(reads).unwrap().entry.body), 1);
    let fails = "kernel test.fails@1 { requires { field.write; } output result: Rows<Int>; compute { return loop 4 with a = 1 { a + 100 / (index() - 2) } + element(); } }";
    assert_eq!(loops(&compile(fails).unwrap().entry.body), 1);
    // A run-time count is not known here either.
    let counted = "kernel test.counted@1 { requires { field.write; } param n: Int = 4; output result: Rows<Int>; compute { return loop n with a = 1 { a * 2 } + element(); } }";
    assert_eq!(loops(&compile(counted).unwrap().entry.body), 1);
}

/// Every operation of a function, nested blocks included.
fn ops(instructions: &[Instruction]) -> Vec<&Op> {
    let mut out = Vec::new();
    for instruction in instructions {
        out.push(&instruction.op);
        match &instruction.op {
            Op::If {
                then, otherwise, ..
            } => {
                out.extend(ops(&then.body));
                out.extend(ops(&otherwise.body));
            }
            Op::Match {
                arms, otherwise, ..
            } => {
                for (_, block) in arms {
                    out.extend(ops(&block.body));
                }
                out.extend(ops(&otherwise.body));
            }
            Op::Loop { until, body, .. } => {
                if let Some(until) = until {
                    out.extend(ops(&until.body));
                }
                out.extend(ops(&body.body));
            }
            _ => {}
        }
    }
    out
}

/// A field or an element taken out of a whole element of an input is a
/// read of those words alone, and the whole read goes when nothing else
/// wants it: a Kernel holding a record of arrays would otherwise copy
/// every word of it to reach three.
#[test]
fn a_part_of_an_element_is_read_without_the_rest_of_it() {
    let head = "kernel test.part@1 { requires { field.read; field.write; }
        struct Patch { points: [Vec3; 8]; label: Int; }
        input patches: Rows<Patch>;
        output result: Rows<Float>;
        compute { ";
    let source = format!(
        "{head} return at(patches, element()).points[element() % 8].y
            + float(at(patches, element()).label); }} }}"
    );
    let module = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    let entry = ops(&module.entry.body);
    let paths: Vec<usize> = entry
        .iter()
        .filter_map(|op| match op {
            Op::Part { path, .. } => Some(path.len()),
            _ => None,
        })
        .collect();
    // `.points[k].y` is three steps in, `.label` one.
    assert_eq!(paths, vec![3, 1], "{module}");
    assert!(
        !entry.iter().any(|op| matches!(op, Op::At(..))),
        "nothing wants the whole element: {module}"
    );
    // A name is not a want: the `let` the read was given goes with
    // the read, since showing it would mean doing work the Kernel no
    // longer does. A trace of a Kernel a document owns walks the
    // source, where every `let` still is.
    let named = format!("{head} let p = at(patches, element()); return p.points[0].y; }} }}");
    let module = compile(&named).unwrap_or_else(|error| panic!("{error}"));
    assert!(
        !ops(&module.entry.body)
            .iter()
            .any(|op| matches!(op, Op::At(..))),
        "the name does not keep it: {module}"
    );
    assert!(
        !module.entry.bindings.iter().any(|(_, name)| name == "p"),
        "and the name goes with it: {:?}",
        module.entry.bindings
    );
    // What wants the whole element keeps it: here one of two elements
    // is chosen before anything is taken out of it.
    let whole = format!(
        "{head} let p = at(patches, element());
            let q = if element() > 2 {{ p }} else {{ at(patches, 0) }};
            return q.points[0].y; }} }}"
    );
    let module = compile(&whole).unwrap_or_else(|error| panic!("{error}"));
    assert!(
        ops(&module.entry.body)
            .iter()
            .any(|op| matches!(op, Op::At(..))),
        "{module}"
    );
}

/// A memory the steps write keeps its whole read: a `put` between the
/// read and the part would be seen by the part and not by the copy.
#[test]
fn a_memory_the_steps_write_is_read_whole() {
    let source = "kernel test.memory@1 { requires { field.read; field.write; }
        struct Patch { tally: Int; label: Int; }
        input rows: Rows<Int>;
        memory mem: Rows<Patch>;
        output result: Rows<Int>;
        scan with acc = 0 {
            let seen = at(mem, index() % 4).label;
            put(mem, index() % 4, Patch { tally: acc, label: index() });
            return acc + seen + at(rows, index());
        } }";
    let module = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let entry = ops(&module.entry.body);
    assert!(
        !entry.iter().any(|op| matches!(op, Op::Part { .. })),
        "a memory is read whole: {module}"
    );
    assert!(entry.iter().any(|op| matches!(op, Op::At(..))), "{module}");
}
