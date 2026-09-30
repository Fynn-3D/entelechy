//! A module's work, accounted in its own IR. Every loop is followed by an
//! [`Op::Spend`] of its body's iterations; every element checks
//! ([`Op::Look`]) when it finishes, as does every iteration of a loop that
//! can run more than [`crate::LOOK_WORK`], leaving once the element must
//! stop. Placement is decided here; a backend implements only what a spend
//! and a look do.

use std::collections::BTreeMap;

use crate::optimize::renumbered_from;
use crate::{
    BinaryOp, Block, Function, Instruction, Module, Op, Type, Validated, ValueId, loop_looks,
    validate,
};

/// The module with its work accounted when [`Module::counts_work`] and not
/// yet accounted; otherwise unchanged. The result is validated again, and a
/// broken rewrite is an error for the caller to report.
pub fn account(module: &Validated) -> Result<Validated, crate::ValidationError> {
    account_renumbered(module).map(|(module, _)| module)
}

/// Which value of the module [`account`] was given each value of the
/// accounted one was: the accounting renumbers every function.
#[derive(Clone, Debug, Default)]
pub struct Renumbering {
    /// The entry's values, accounted id to given id; empty when the
    /// module was not rewritten.
    pub entry: BTreeMap<ValueId, ValueId>,
    /// Each function's, by its index.
    pub functions: Vec<BTreeMap<ValueId, ValueId>>,
    /// Whether the module was rewritten.
    pub rewritten: bool,
}

impl Renumbering {
    /// The given id of the entry's (`None`) or a function's value `id`;
    /// `None` for a value the accounting added.
    pub fn given(&self, function: Option<usize>, id: ValueId) -> Option<ValueId> {
        if !self.rewritten {
            return Some(id);
        }
        let ids = match function {
            None => &self.entry,
            Some(index) => self.functions.get(index)?,
        };
        ids.get(&id).copied()
    }
}

/// [`account`], and which given value each accounted one was.
pub fn account_renumbered(
    module: &Validated,
) -> Result<(Validated, Renumbering), crate::ValidationError> {
    if !module.counts_work() || module.accounted() {
        return Ok((module.clone(), Renumbering::default()));
    }
    let functions = module.functions_work();
    let rewrite = |function: &Function, element: bool| {
        let given = function.slot_count() as u32;
        let mut next = given;
        let mut body = block(&function.body, &functions, &mut next);
        if element {
            body.push(Instruction {
                id: ValueId(next),
                ty: Type::Bool,
                op: Op::Look,
            });
        }
        let (function, ids) = renumbered_from(Function {
            body,
            ..function.clone()
        });
        // Accounted id to given id, for the values that were given.
        let back = ids
            .into_iter()
            .filter(|(old, _)| *old < given)
            .map(|(old, new)| (ValueId(new), ValueId(old)))
            .collect::<BTreeMap<_, _>>();
        (function, back)
    };
    let functions_renumbered: Vec<(Function, BTreeMap<ValueId, ValueId>)> = module
        .functions
        .iter()
        .map(|function| rewrite(function, false))
        .collect();
    let (entry, entry_ids) = rewrite(&module.entry, true);
    let renumbering = Renumbering {
        entry: entry_ids,
        functions: functions_renumbered
            .iter()
            .map(|(_, ids)| ids.clone())
            .collect(),
        rewritten: true,
    };
    let accounted = Module {
        manifest: module.manifest.clone(),
        structs: module.structs.clone(),
        functions: functions_renumbered
            .into_iter()
            .map(|(function, _)| function)
            .collect(),
        entry,
        merge: module.merge.as_ref().map(|merge| rewrite(merge, true).0),
        init: module.init.as_ref().map(|init| rewrite(init, true).0),
    };
    Ok((validate(accounted)?, renumbering))
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
