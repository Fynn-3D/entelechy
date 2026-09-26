//! A module's work, accounted in its own IR. Every loop is followed by an
//! [`Op::Spend`] of its body's iterations; every element checks
//! ([`Op::Look`]) when it finishes, as does every iteration of a loop that
//! can run more than [`crate::LOOK_WORK`], leaving once the element must
//! stop. Placement is decided here; a backend implements only what a spend
//! and a look do.

use crate::optimize::renumbered;
use crate::{
    BinaryOp, Block, Function, Instruction, Module, Op, Type, Validated, ValueId, loop_looks,
    validate,
};

/// The module with its work accounted when [`Module::counts_work`] and not
/// yet accounted; otherwise unchanged. The result is validated again, and a
/// broken rewrite is an error for the caller to report.
pub fn account(module: &Validated) -> Result<Validated, crate::ValidationError> {
    if !module.counts_work() || module.accounted() {
        return Ok(module.clone());
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
    validate(accounted)
}

/// The instructions with every loop followed by its spend and a look at
/// the start of each checking loop's `until`. New ids start at `next`;
/// the function is renumbered after.
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
                    // The look precedes `until`, so a stopped element
                    // leaves first.
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
