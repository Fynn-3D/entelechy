//! A group's lanes meet at each exchange: every lane gives its value and
//! takes another's. A device runs a group's lanes together and an
//! exchange is a barrier they all pass, so every lane must reach each
//! exchange alike: the branches and loops round one may turn only on
//! what every lane of the group holds the same. A value a lane may hold
//! alone (its lane, its element, what it reads, what it was handed) is
//! varying; the rest (literals, params, the domain, the context, the
//! counts of inputs) is the group's own. An exchange under a branch or a
//! loop that turns on a varying value is refused, as one in a function.

use crate::optimize::operands;
use crate::{Block, Instruction, Mode, Module, Op, ValueId};

/// Where a group's exchanges fail to be met by every lane alike, and
/// why: the first exchange in a function, or under a branch or a loop
/// that turns on what a lane holds alone. `None` for a module of another
/// mode, or for a group whose every exchange every lane reaches.
pub fn group_fault(module: &Module) -> Option<(ValueId, String)> {
    let Mode::Group { .. } = module.manifest.mode else {
        return None;
    };
    for function in &module.functions {
        if let Some(at) = first_exchange(&function.body) {
            return Some((
                at,
                "`exchange` meets every lane of a group where the entry reaches it, so it is written in `group`'s body, never in a function it calls".into(),
            ));
        }
    }
    let reads = reading_functions(module);
    let mut walk = Walk {
        varying: Vec::new(),
        carried: Vec::new(),
        reads: &reads,
        fault: None,
    };
    for instruction in &module.entry.body {
        walk.instruction(instruction, true);
    }
    walk.fault
}

/// The first exchange in instructions, nested blocks included.
fn first_exchange(instructions: &[Instruction]) -> Option<ValueId> {
    instructions
        .iter()
        .find_map(|instruction| match instruction.op {
            Op::Exchange(..) => Some(instruction.id),
            ref other => other
                .blocks()
                .into_iter()
                .find_map(|block| first_exchange(&block.body)),
        })
}

/// Whether each function reads an input, itself or through a call: what
/// it gives may differ lane to lane whatever it is given.
fn reading_functions(module: &Module) -> Vec<bool> {
    let mut reads = vec![false; module.functions.len()];
    loop {
        let mut changed = false;
        for (index, function) in module.functions.iter().enumerate() {
            if !reads[index] && block_reads(&function.body, &reads) {
                reads[index] = true;
                changed = true;
            }
        }
        if !changed {
            return reads;
        }
    }
}

fn block_reads(instructions: &[Instruction], reads: &[bool]) -> bool {
    instructions
        .iter()
        .any(|instruction| match &instruction.op {
            Op::Sample(_)
            | Op::At(..)
            | Op::AtLinear(..)
            | Op::AtFiltered { .. }
            | Op::Part { .. } => true,
            Op::Call(callee, _) => reads.get(*callee as usize).copied().unwrap_or(true),
            other => other
                .blocks()
                .into_iter()
                .any(|block| block_reads(&block.body, reads)),
        })
}

/// The walk of a group's entry: which values may differ lane to lane, the
/// carried value of each loop open round the walk, and the first fault.
struct Walk<'a> {
    varying: Vec<bool>,
    carried: Vec<bool>,
    reads: &'a [bool],
    fault: Option<(ValueId, String)>,
}

impl Walk<'_> {
    fn is_varying(&self, id: ValueId) -> bool {
        self.varying.get(id.0 as usize).copied().unwrap_or(true)
    }

    fn set(&mut self, id: ValueId, varying: bool) {
        let at = id.0 as usize;
        if self.varying.len() <= at {
            self.varying.resize(at + 1, false);
        }
        self.varying[at] = varying;
    }

    fn fail(&mut self, at: ValueId, why: &str) {
        if self.fault.is_none() {
            self.fault = Some((at, why.to_owned()));
        }
    }

    /// Walks a block whose every lane runs alike when `alike`, and answers
    /// whether its result may differ lane to lane.
    fn block(&mut self, block: &Block, alike: bool) -> bool {
        for instruction in &block.body {
            self.instruction(instruction, alike);
        }
        self.is_varying(block.result)
    }

    fn instruction(&mut self, instruction: &Instruction, alike: bool) {
        let id = instruction.id;
        let varying = match &instruction.op {
            Op::Exchange(..) => {
                if !alike {
                    self.fail(
                        id,
                        "`exchange` is reached by every lane of a group alike, and this one is under a branch or a loop that turns on what a lane holds alone (its lane, its element, what it reads or was handed); turn it on what the whole group holds (params, literals, the domain)",
                    );
                }
                true
            }
            Op::Lane
            | Op::Element
            | Op::Coord
            | Op::Sample(_)
            | Op::At(..)
            | Op::AtLinear(..)
            | Op::AtFiltered { .. }
            | Op::Part { .. } => true,
            Op::Carried => self.carried.last().copied().unwrap_or(true),
            Op::Index => false,
            Op::Call(callee, args) => {
                self.reads.get(*callee as usize).copied().unwrap_or(true)
                    || args.iter().any(|arg| self.is_varying(*arg))
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let turns = self.is_varying(*condition);
                let inner = alike && !turns;
                let a = self.block(then, inner);
                let b = self.block(otherwise, inner);
                turns || a || b
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let turns = self.is_varying(*scrutinee);
                let inner = alike && !turns;
                let mut any = turns;
                for (_, block) in arms {
                    any |= self.block(block, inner);
                }
                any | self.block(otherwise, inner)
            }
            Op::Loop {
                bound,
                init,
                until,
                body,
                ..
            } => {
                // The carried value differs lane to lane when it starts so
                // or any iteration makes it so: walked until that settles,
                // faults found only on the last walk.
                let mut carried = self.is_varying(*init);
                let bound_varies = bound.is_some_and(|b| self.is_varying(b));
                let fault = self.fault.take();
                loop {
                    self.carried.push(carried);
                    let stops = until.as_ref().is_some_and(|u| self.block(u, false));
                    let next = self.block(body, false);
                    self.carried.pop();
                    let settled = carried || next || stops;
                    if settled == carried {
                        break;
                    }
                    carried = settled;
                }
                self.fault = fault;
                self.carried.push(carried);
                let stops = until.as_ref().is_some_and(|u| self.block(u, false));
                let inner = alike && !bound_varies && !stops;
                if let Some(until) = until {
                    self.block(until, inner);
                }
                self.block(body, inner);
                self.carried.pop();
                carried || bound_varies || stops
            }
            Op::Each { body, .. } => self.block(body, alike),
            other => operands(other)
                .iter()
                .any(|operand| self.is_varying(*operand)),
        };
        self.set(id, varying);
    }
}
