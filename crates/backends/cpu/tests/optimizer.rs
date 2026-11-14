//! The optimizer is held to the interpreter: every feature under every
//! mode computes the same words from optimized and lowered IR, and the
//! optimized IR is smaller.

use entelechy_abi::{Capabilities, Capability, Host, Invocation, Scope};
use entelechy_compiler::{compile, compile_unoptimized};
use entelechy_cpu::CpuBackend;
use entelechy_ir::{Instruction, Op};

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
    for case in entelechy_conformance::cases() {
        let plain = compile_unoptimized(&case.source).unwrap();
        let optimized = compile(&case.source).unwrap();
        let size = |module: &entelechy_ir::Validated| {
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
        let run = |module: &entelechy_ir::Validated| -> Result<Vec<Vec<u32>>, String> {
            let mut host = Host::new(1 << 22);
            let handles: Vec<_> = case
                .inputs
                .iter()
                .map(|(schema, domain, words)| {
                    host.adopt(
                        entelechy_abi::FieldData::from_values(schema.clone(), *domain, words)
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
                .map(|handle| host.data(*handle).unwrap().storage().to_vec())
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

/// A loop with a known count and initial value whose body only computes
/// runs once at compile time, as for a table built in a loop.
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
    // 0xAAAA + 0xCCCC * 2, the masks of two sides, folded.
    let text = module.entry.to_string();
    assert!(text.contains("148546"), "{text}");

    // A loop reading a Field is not folded, nor one that would fail: the
    // failure belongs to its element.
    let reads = "kernel test.reads@1 { requires { field.read; field.write; } input rows: Rows<Int>; output result: Rows<Int>; compute { return loop 8 with a = 0 { a + at(rows, index()) }; } }";
    assert_eq!(loops(&compile(reads).unwrap().entry.body), 1);
    let fails = "kernel test.fails@1 { requires { field.write; } output result: Rows<Int>; compute { return loop 4 with a = 1 { a + 100 / (index() - 2) } + element(); } }";
    assert_eq!(loops(&compile(fails).unwrap().entry.body), 1);
    // Nor a loop with a run-time count.
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

/// A field or element taken out of a whole input element reads only those
/// words, and the whole read goes when nothing else wants it.
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
    // The read's `let` name goes with the read; a trace of a document's
    // Kernel walks the source, where every `let` stays.
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
    // A whole element still wanted is kept: here one of two is chosen
    // before anything is taken out.
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

/// A function too long to write out, handed a whole wide input element
/// it only takes apart, is given the element's address instead and reads
/// the parts there: the caller holds no copy of the element, and the
/// words are the same.
#[test]
fn a_wide_element_handed_to_a_long_function_is_read_where_it_lies() {
    // A body past the longest a function with a wide argument is written
    // out at.
    let steps: String = (0..1100)
        .map(|i| format!("let a{} = a{} * 1.0001 + 0.5;\n", i + 1, i))
        .collect();
    let source = format!(
        "kernel test.through@1 {{ requires {{ field.read; field.write; }}
            struct Patch {{ points: [Vec3; 16]; label: Int; }}
            input patches: Rows<Patch>;
            output result: Rows<Float>;
            fn heavy(p: Patch, k: Int) -> Float {{
                let a0 = p.points[k].y + float(p.label);
                {steps}
                return a1100 + p.points[0].x;
            }}
            fn handed(p: Patch, k: Int) -> Float {{ return heavy(p, k) + float(p.label); }}
            compute {{ return handed(at(patches, element()), element() % 16) + heavy(at(patches, 0), 3); }}
        }}"
    );
    let optimized = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    assert!(
        !ops(&optimized.entry.body)
            .iter()
            .any(|op| matches!(op, Op::At(..))),
        "no whole element is read: {optimized}"
    );
    let made: Vec<&entelechy_ir::Function> = optimized
        .functions
        .iter()
        .filter(|f| f.name.contains('~'))
        .collect();
    assert!(!made.is_empty(), "{optimized}");
    assert!(
        made.iter()
            .all(|f| ops(&f.body).iter().any(|op| matches!(op, Op::Part { .. }))),
        "{optimized}"
    );
    // The same words as the plain lowering.
    let plain = compile_unoptimized(&source).unwrap();
    let patch_words = 16 * 3 + 1;
    let schema = entelechy_abi::Schema::new(
        std::iter::repeat_n(entelechy_abi::Scalar::Float, 48)
            .chain([entelechy_abi::Scalar::Int])
            .collect(),
    );
    let words: Vec<u32> = (0..4 * patch_words)
        .map(|w| {
            if w % patch_words == patch_words - 1 {
                (w / patch_words) as u32
            } else {
                (w as f32 * 0.25).to_bits()
            }
        })
        .collect();
    let run = |module: &entelechy_ir::Validated| -> Vec<u32> {
        let mut host = Host::new(1 << 22);
        let patches = host
            .adopt(
                entelechy_abi::FieldData::from_values(
                    schema.clone(),
                    entelechy_abi::Domain::Rows(4),
                    &words,
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = Invocation {
            inputs: vec![patches],
            params: Vec::new(),
            domain: entelechy_abi::Domain::Rows(4),
            context: Default::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let outputs = CpuBackend.execute(module, &invocation, &mut scope).unwrap();
        host.data(outputs[0]).unwrap().storage().to_vec()
    };
    assert_eq!(run(&plain), run(&optimized));
}

/// A wide part of an input element, reached by fields (`v.b`), handed to
/// a function that only takes it apart, is read where it lies as a whole
/// element is: the function is given the element's address and reads its
/// parts past the fields that lead to it; the words are the same.
#[test]
fn a_wide_part_of_an_element_handed_on_is_read_where_it_lies() {
    let steps: String = (0..1100)
        .map(|i| format!("let a{} = a{} * 1.0001 + 0.5;\n", i + 1, i))
        .collect();
    let source = format!(
        "kernel test.part@1 {{ requires {{ field.read; field.write; }}
            struct Patch {{ points: [Vec3; 16]; label: Int; }}
            struct View {{ a: Patch; b: Patch; t: Float; }}
            input views: Rows<View>;
            output result: Rows<Float>;
            fn heavy(p: Patch, k: Int) -> Float {{
                let a0 = p.points[k].y + float(p.label);
                {steps}
                return a1100 + p.points[0].x;
            }}
            fn handed(p: Patch, k: Int) -> Float {{ return heavy(p, k) + float(p.label); }}
            compute {{ let v = at(views, element()); return handed(v.b, element() % 16) + v.t; }}
        }}"
    );
    let optimized = compile(&source).unwrap_or_else(|error| panic!("{error}"));
    let entry = ops(&optimized.entry.body);
    assert!(
        !entry.iter().any(|op| matches!(op, Op::At(..))
            || matches!(op, Op::Part { path, .. } if path.len() == 1 && path[0] == entelechy_ir::Step::Field(1))),
        "neither the element nor its part `b` is read whole: {optimized}"
    );
    let made: Vec<&entelechy_ir::Function> = optimized
        .functions
        .iter()
        .filter(|f| f.name.contains('~'))
        .collect();
    assert!(!made.is_empty(), "{optimized}");
    assert!(
        made.iter().all(|f| ops(&f.body).iter().any(|op| matches!(op,
            Op::Part { path, .. } if path.first() == Some(&entelechy_ir::Step::Field(1)) && path.len() > 1))),
        "each made function reads past `b`: {optimized}"
    );
    // The same words as the plain lowering.
    let plain = compile_unoptimized(&source).unwrap();
    let patch = 16 * 3 + 1;
    let view = 2 * patch + 1;
    let schema = entelechy_abi::Schema::new(
        (0..2)
            .flat_map(|_| {
                std::iter::repeat_n(entelechy_abi::Scalar::Float, 48)
                    .chain([entelechy_abi::Scalar::Int])
            })
            .chain([entelechy_abi::Scalar::Float])
            .collect(),
    );
    let words: Vec<u32> = (0..4 * view)
        .map(|w| {
            let at = w % view;
            if at == patch - 1 || at == 2 * patch - 1 {
                (w / view + at) as u32
            } else {
                (w as f32 * 0.25).to_bits()
            }
        })
        .collect();
    let run = |module: &entelechy_ir::Validated| -> Vec<u32> {
        let mut host = Host::new(1 << 22);
        let views = host
            .adopt(
                entelechy_abi::FieldData::from_values(
                    schema.clone(),
                    entelechy_abi::Domain::Rows(4),
                    &words,
                )
                .unwrap(),
            )
            .unwrap();
        let invocation = Invocation {
            inputs: vec![views],
            params: Vec::new(),
            domain: entelechy_abi::Domain::Rows(4),
            context: Default::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let outputs = CpuBackend.execute(module, &invocation, &mut scope).unwrap();
        host.data(outputs[0]).unwrap().storage().to_vec()
    };
    assert_eq!(run(&plain), run(&optimized));
}

/// A memory the steps write keeps its whole read: a part would see a
/// `put` between read and part that the copy would not.
#[test]
fn a_memory_the_steps_write_is_read_whole() {
    let source = "kernel test.memory@1 { requires { field.read; field.write; }
        struct Patch { tally: Int; label: Int; }
        input rows: Rows<Int>;
        memory mem: Rows<Patch>;
        output result: Rows<Int>;
        scan over rows with acc = 0 {
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

/// A `match` or an `if` on a literal is the block it takes, as for a
/// table written `each` with a `match` on its index: the other blocks are
/// gone, a failure only they would have raised is gone with them, and one
/// the taken block raises is still its element's.
#[test]
fn a_branch_on_a_literal_is_the_block_it_takes() {
    let branches = |body: &[Instruction]| {
        ops(body)
            .into_iter()
            .filter(|op| matches!(op, Op::Match { .. } | Op::If { .. }))
            .count()
    };
    let run = |module: &entelechy_ir::Validated| -> Result<Vec<u32>, String> {
        let mut host = Host::new(1 << 20);
        let invocation = Invocation {
            inputs: vec![],
            params: vec![],
            domain: entelechy_abi::Domain::Rows(8),
            context: Default::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out = CpuBackend
            .execute(module, &invocation, &mut scope)
            .map_err(|e| e.to_string())?[0];
        Ok(host.data(out).unwrap().storage().to_vec())
    };
    let table = "kernel test.table@1 { requires { field.write; } output result: Rows<Int>;
        compute {
            let t = [each 6 { match index() { 1 => 10, 2 | 3 => 20 + index(), 5 => 7 / (index() - 5), _ => index() * 3 } }];
            let pick = if 2 > 1 { t[1] } else { 1 / 0 };
            return t[element() % 5] + pick + element();
        } }";
    let plain = compile_unoptimized(table).unwrap();
    let module = compile(table).unwrap_or_else(|error| panic!("{error}"));
    assert!(
        branches(&plain.entry.body) > 0,
        "the lowering writes the branches"
    );
    assert_eq!(branches(&module.entry.body), 0, "{module}");
    // Arm 5 divides by zero: it is taken, so both fail, at the same element.
    let (a, b) = (run(&plain), run(&module));
    assert!(a.is_err(), "{a:?}");
    assert_eq!(a, b);
    // Without the failing arm taken, the same words.
    let kept = table.replace("each 6", "each 5");
    let (a, b) = (
        run(&compile_unoptimized(&kept).unwrap()),
        run(&compile(&kept).unwrap()),
    );
    assert!(a.is_ok(), "{a:?}");
    assert_eq!(a, b);
}

/// A chain of `if`s testing one `Int` against literals, disjunctions and
/// short closed ranges collapses into one `match`, carrying a wide result
/// through one join and keeping the first matching arm on overlap.
#[test]
fn an_integer_if_chain_collapses_into_one_match() {
    let source = "kernel test.chain@1 {
        requires { field.read; field.write; }
        struct Wide { v: [Vec4; 8]; tag: Int; }
        fn pack(base: Float, tag: Int) -> Wide {
            return Wide {
                v: [each 8 { vec4(base + float(index()), base * 2.0, -base, 1.0) }],
                tag: tag,
            };
        }
        input keys: Rows<Int>;
        output result: Rows<Wide>;
        compute {
            let k = sample(keys);
            if k == 1 { return pack(10.0, 1); }
            if k == 2 || k == 7 {
                let two = k == 2;
                return pack(if two { 20.0 } else { 70.0 }, 27);
            }
            if k >= 10 && k <= 12 { return pack(float(k) * 3.0, 1012); }
            if 14 < k && k < 18 { return pack(float(k) * 5.0, 1517); }
            if k == 7 || k == 11 || k == 20 { return pack(200.0, 20); }
            if k == -2147483647 - 1 { return pack(-1.0, -99); }
            return pack(0.0, 0);
        }
    }";
    let plain = compile_unoptimized(source).unwrap();
    let optimized = compile(source).unwrap_or_else(|error| panic!("{error}"));
    let matches = optimized
        .entry
        .body
        .iter()
        .filter(|inst| matches!(inst.op, Op::Match { .. }))
        .count();
    assert_eq!(
        matches, 1,
        "expected one top-level match: {}",
        optimized.entry
    );
    assert!(
        entelechy_ir::carried(&optimized).total < entelechy_ir::carried(&plain).total / 2,
        "carried words did not drop: {} vs {}",
        entelechy_ir::carried(&optimized).total,
        entelechy_ir::carried(&plain).total,
    );

    let probes: Vec<i32> = (-2..=25)
        .chain([i32::MIN, i32::MIN + 1, i32::MAX - 1, i32::MAX])
        .collect();
    let words: Vec<u32> = probes.iter().map(|&v| v as u32).collect();
    let schema = entelechy_ir::Schema::new(vec![entelechy_ir::Scalar::Int]);
    let domain = entelechy_abi::Domain::Rows(probes.len() as u32);
    let run = |module: &entelechy_ir::Validated| -> Vec<u32> {
        let mut host = Host::new(1 << 22);
        let input = host
            .adopt(entelechy_abi::FieldData::from_values(schema.clone(), domain, &words).unwrap())
            .unwrap();
        let invocation = Invocation {
            inputs: vec![input],
            params: vec![],
            domain,
            context: Default::default(),
        };
        let mut scope = Scope::new(&mut host, all());
        let out = CpuBackend.execute(module, &invocation, &mut scope).unwrap()[0];
        host.data(out).unwrap().storage().to_vec()
    };
    assert_eq!(run(&plain), run(&optimized));
}
