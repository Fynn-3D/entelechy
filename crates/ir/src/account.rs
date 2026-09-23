//! A module's work, accounted in its own IR. Every loop is followed by
//! an [`Op::Spend`] of the iterations whose body ran; every element
//! looks ([`Op::Look`]) when it finishes, and so does every iteration of
//! a loop that can run more than [`crate::LOOK_WORK`] by itself, which leaves
//! the loop once the element must stop. The rule of where work is
//! counted and where it is looked at is written here once, and a backend
//! runs it as it runs any other operation: it implements what a spend
//! and a look are, never where they go.

use crate::optimize::renumbered;
use crate::{
    BinaryOp, Block, Function, Instruction, Module, Op, Type, Validated, ValueId, loop_looks,
    validate,
};

/// The module with its work accounted, when it counts its work
/// ([`Module::counts_work`]); the module as it is otherwise. A module
/// already accounted is given back as it is.
pub fn account(module: &Validated) -> Validated {
    if !module.counts_work() || module.accounted() {
        return module.clone();
    }
    let functions = module.functions_work();
    let rewrite = |function: &Function, element: bool| {
        let mut next = function.slot_count() as u32;
        let mut body = block(&function.body, &functions, &mut next);
        if element {
            body.push(Instruction {
                id: ValueId(next),
                ty: Type::Bool,
                op: Op::Look,
            });
        }
        renumbered(Function {
            body,
            ..function.clone()
        })
    };
    let accounted = Module {
        manifest: module.manifest.clone(),
        structs: module.structs.clone(),
        functions: module
            .functions
            .iter()
            .map(|function| rewrite(function, false))
            .collect(),
        entry: rewrite(&module.entry, true),
        merge: module.merge.as_ref().map(|merge| rewrite(merge, true)),
        init: module.init.as_ref().map(|init| rewrite(init, true)),
    };
    validate(accounted).expect("accounting keeps a valid module valid")
}

/// The instructions with every loop followed by its spend, and the
/// loops that look given a look at the start of their `until`. New
/// values take ids from `next` up; the function is numbered again after.
fn block(instructions: &[Instruction], functions: &[u64], next: &mut u32) -> Vec<Instruction> {
    let fresh = |ty: Type, op: Op, next: &mut u32| {
        let id = ValueId(*next);
        *next += 1;
        Instruction { id, ty, op }
    };
    let mut out = Vec::with_capacity(instructions.len());
    for instruction in instructions {
        let nested = |b: &Block, next: &mut u32| Block {
            body: block(&b.body, functions, next),
            result: b.result,
        };
        let op = match &instruction.op {
            Op::If {
                condition,
                then,
                otherwise,
            } => Op::If {
                condition: *condition,
                then: nested(then, next),
                otherwise: nested(otherwise, next),
            },
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => Op::Match {
                scrutinee: *scrutinee,
                arms: arms
                    .iter()
                    .map(|(values, arm)| (values.clone(), nested(arm, next)))
                    .collect(),
                otherwise: nested(otherwise, next),
            },
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                let looks = loop_looks(*count, until.as_ref(), body, functions);
                let mut until = until.as_ref().map(|until| nested(until, next));
                if looks {
                    // The look comes first: an element that must stop
                    // leaves before its iteration's `until` runs.
                    let look = fresh(Type::Bool, Op::Look, next);
                    until = Some(match until {
                        None => Block {
                            result: look.id,
                            body: vec![look],
                        },
                        Some(mut until) => {
                            let either = fresh(
                                Type::Bool,
                                Op::Binary(BinaryOp::Or, look.id, until.result),
                                next,
                            );
                            until.body.insert(0, look);
                            until.result = either.id;
                            until.body.push(either);
                            until
                        }
                    });
                }
                Op::Loop {
                    count: *count,
                    bound: *bound,
                    init: *init,
                    until,
                    body: nested(body, next),
                }
            }
            other => other.clone(),
        };
        let is_loop = matches!(op, Op::Loop { .. });
        out.push(Instruction {
            id: instruction.id,
            ty: instruction.ty,
            op,
        });
        if is_loop {
            out.push(fresh(Type::Int, Op::Spend(instruction.id), next));
        }
    }
    out
}
