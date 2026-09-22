//! The optimizer: what a Kernel computes, in fewer instructions, with
//! every word the same. Four rewrites, each exact by construction:
//!
//! - a component, field, column or element taken straight out of what
//!   built it is that part; a vector rebuilt from all of its own
//!   components in order is the vector;
//! - an operation on literals alone is its literal, where the backends'
//!   own arithmetic gives one value and no failure: integers that
//!   overflow or divide by zero stay for the run to report;
//! - a pure operation computed twice on the same operands, in one block
//!   or an enclosing one, is computed once (a Field read is pure unless
//!   the Field is a memory the steps write; a call is pure unless the
//!   function writes);
//! - a value nothing reads is not computed, unless it is a `put`, a
//!   call that writes, a block holding one, or a name a trace shows;
//! - a call to a short function is what the function does, written out
//!   where it was called, which costs the interpreter no frame and the
//!   JIT no call;
//! - work inside a loop that does not change while the loop runs is
//!   done once, before it, when it cannot fail (so a run that fails
//!   still reports the failure it did, of the element it did).
//!
//! Values are then numbered again in pre-order, as the IR requires.
//! Nothing here reorders floats or changes which failure a run reports.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Module, Op, Step, StructDef, Type, UnaryOp,
    Value, ValueId,
};

/// A function of at most this many instructions, nested blocks counted,
/// is written out where it is called instead: a call costs a frame on
/// the interpreter and a real call on the JIT, and a short function is
/// cheaper spelled out. Beyond it the code would grow faster than the
/// calls it saves.
const INLINE_LIMIT: usize = 96;

/// A function taking a record or an array of at least this many words
/// is written out where it is called even when it is longer: the
/// argument is copied at every element, which costs more than the
/// instructions do, and written out its reads narrow too. Beyond
/// [`INLINE_BIG_LIMIT`] instructions the code would grow too far for
/// what one copy is worth. Giving one back was tried and measured as
/// nothing: a function that builds a wide value pays for building it,
/// not for handing it over.
const BIG_ARG_WORDS: usize = 32;
/// How long a function taking a wide argument may be and still be
/// written out where it is called.
const INLINE_BIG_LIMIT: usize = 1024;

/// Optimizes every function of a module. The result computes what the
/// input did, word for word, and validates whenever the input does.
pub fn optimize(module: Module) -> Module {
    let writers = crate::validate::writers(&module.functions);
    let memories: Vec<bool> = module
        .manifest
        .inputs
        .iter()
        .map(|port| module.manifest.memories.contains(&port.name))
        .collect();
    let ctx = Ctx {
        structs: &module.structs,
        writers: &writers,
        memories: &memories,
    };
    // Functions are defined before they are called, so taking them in
    // order means a call is always to one already optimized: what is
    // written out is its short form, and depth costs nothing.
    let mut functions: Vec<Function> = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        let flattened = inline_calls(function, &functions, &module.structs);
        functions.push(ctx.function(&flattened));
    }
    let entry = ctx.function(&inline_calls(&module.entry, &functions, &module.structs));
    let merge = module
        .merge
        .as_ref()
        .map(|f| ctx.function(&inline_calls(f, &functions, &module.structs)));
    let init = module
        .init
        .as_ref()
        .map(|f| ctx.function(&inline_calls(f, &functions, &module.structs)));
    Module {
        manifest: module.manifest,
        structs: module.structs,
        functions,
        entry,
        merge,
        init,
    }
}

/// How many instructions a body holds, nested blocks counted.
fn size(body: &[Instruction]) -> usize {
    body.iter()
        .map(|instruction| {
            1 + match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => size(&then.body) + size(&otherwise.body),
                Op::Match {
                    arms, otherwise, ..
                } => {
                    arms.iter()
                        .map(|(_, block)| size(&block.body))
                        .sum::<usize>()
                        + size(&otherwise.body)
                }
                Op::Loop { until, body, .. } => {
                    until.as_ref().map_or(0, |block| size(&block.body)) + size(&body.body)
                }
                _ => 0,
            }
        })
        .sum()
}

/// The greatest value id a body defines, nested blocks counted.
fn highest(body: &[Instruction]) -> u32 {
    body.iter()
        .map(|instruction| {
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => highest(&then.body).max(highest(&otherwise.body)),
                Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .map(|(_, block)| highest(&block.body))
                    .max()
                    .unwrap_or(0)
                    .max(highest(&otherwise.body)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .map_or(0, |block| highest(&block.body))
                    .max(highest(&body.body)),
                _ => 0,
            };
            instruction.id.0.max(nested)
        })
        .max()
        .unwrap_or(0)
}

/// `function` with its calls to short functions written out. The callee's
/// values take fresh ids above the caller's, its `Arg(i)` reads become
/// the call's own operands, and what it returns takes the call's place.
fn inline_calls(function: &Function, functions: &[Function], structs: &[StructDef]) -> Function {
    let inlinable: Vec<bool> = functions
        .iter()
        .map(|callee| {
            if callee.results.len() != 1 {
                return false;
            }
            let size = size(&callee.body);
            let widest = callee
                .args
                .iter()
                .map(|ty| crate::words(structs, *ty))
                .max()
                .unwrap_or(0);
            size <= INLINE_LIMIT || (widest >= BIG_ARG_WORDS && size <= INLINE_BIG_LIMIT)
        })
        .collect();
    if !inlinable.iter().any(|yes| *yes) {
        return function.clone();
    }
    let mut pass = Inline {
        functions,
        inlinable: &inlinable,
        alias: HashMap::new(),
        next: highest(&function.body) + 1,
    };
    let body = pass.block(&function.body);
    Function {
        name: function.name.clone(),
        args: function.args.clone(),
        body,
        results: function.results.iter().map(|id| pass.map(*id)).collect(),
        bindings: function
            .bindings
            .iter()
            .map(|(id, name)| (pass.map(*id), name.clone()))
            .collect(),
    }
}

/// Writing short functions out where they are called.
struct Inline<'a> {
    functions: &'a [Function],
    inlinable: &'a [bool],
    /// A value that stands for another: a call's value is what the
    /// written-out body returned, and an argument read is the operand.
    alias: HashMap<u32, ValueId>,
    /// The next free value id; the numbering is made dense again later.
    next: u32,
}

impl Inline<'_> {
    fn map(&self, id: ValueId) -> ValueId {
        match self.alias.get(&id.0) {
            Some(other) if *other != id => self.map(*other),
            _ => id,
        }
    }

    fn fresh(&mut self) -> ValueId {
        let id = ValueId(self.next);
        self.next += 1;
        id
    }

    fn nested(&mut self, block: &Block) -> Block {
        let body = self.block(&block.body);
        Block {
            body,
            result: self.map(block.result),
        }
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            if let Op::Call(callee, args) = &instruction.op
                && self
                    .inlinable
                    .get(*callee as usize)
                    .copied()
                    .unwrap_or(false)
            {
                let operands: Vec<ValueId> = args.iter().map(|id| self.map(*id)).collect();
                let body = self.functions[*callee as usize].clone();
                let mut ids: HashMap<u32, ValueId> = HashMap::new();
                self.written(&body.body, &operands, &mut ids, &mut out);
                let result = Self::renamed(body.results[0], &ids, &operands, &body);
                self.alias.insert(instruction.id.0, result);
                continue;
            }
            let op = match &instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => Op::If {
                    condition: self.map(*condition),
                    then: self.nested(then),
                    otherwise: self.nested(otherwise),
                },
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: self.map(*scrutinee),
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.nested(block)))
                        .collect(),
                    otherwise: self.nested(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
                    bound: bound.map(|id| self.map(id)),
                    init: self.map(*init),
                    until: until.as_ref().map(|block| self.nested(block)),
                    body: self.nested(body),
                },
                other => {
                    let mut op = other.clone();
                    rename(&mut op, &|id| self.map(id));
                    op
                }
            };
            out.push(Instruction {
                id: instruction.id,
                ty: instruction.ty,
                op,
            });
        }
        out
    }

    /// A value of the callee as the caller names it: an argument read is
    /// the operand it was called with, anything else its fresh id.
    fn renamed(
        id: ValueId,
        ids: &HashMap<u32, ValueId>,
        operands: &[ValueId],
        callee: &Function,
    ) -> ValueId {
        if let Some(fresh) = ids.get(&id.0) {
            return *fresh;
        }
        if let Some(Op::Arg(index)) = find_op(&callee.body, id) {
            return operands[*index as usize];
        }
        id
    }

    /// The callee's body written into `out` with fresh ids.
    fn written(
        &mut self,
        body: &[Instruction],
        operands: &[ValueId],
        ids: &mut HashMap<u32, ValueId>,
        out: &mut Vec<Instruction>,
    ) {
        for instruction in body {
            if let Op::Arg(index) = &instruction.op {
                ids.insert(instruction.id.0, operands[*index as usize]);
                continue;
            }
            let id = self.fresh();
            let mut op = instruction.op.clone();
            let take = |value: ValueId, ids: &HashMap<u32, ValueId>| -> ValueId {
                ids.get(&value.0).copied().unwrap_or(value)
            };
            match &mut op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    *condition = take(*condition, ids);
                    *then = self.written_block(then, operands, ids);
                    *otherwise = self.written_block(otherwise, operands, ids);
                }
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    *scrutinee = take(*scrutinee, ids);
                    for (_, block) in arms.iter_mut() {
                        *block = self.written_block(block, operands, ids);
                    }
                    *otherwise = self.written_block(otherwise, operands, ids);
                }
                Op::Loop {
                    bound,
                    init,
                    until,
                    body,
                    ..
                } => {
                    *bound = bound.map(|value| take(value, ids));
                    *init = take(*init, ids);
                    *until = until
                        .as_ref()
                        .map(|block| self.written_block(block, operands, ids));
                    *body = self.written_block(body, operands, ids);
                }
                other => rename(other, &|value| take(value, ids)),
            }
            ids.insert(instruction.id.0, id);
            out.push(Instruction {
                id,
                ty: instruction.ty,
                op,
            });
        }
    }

    fn written_block(
        &mut self,
        block: &Block,
        operands: &[ValueId],
        ids: &mut HashMap<u32, ValueId>,
    ) -> Block {
        let mut body = Vec::with_capacity(block.body.len());
        self.written(&block.body, operands, ids, &mut body);
        Block {
            body,
            result: ids.get(&block.result.0).copied().unwrap_or(block.result),
        }
    }
}

/// The operation a body gives an id, nested blocks included.
fn find_op(body: &[Instruction], id: ValueId) -> Option<&Op> {
    for instruction in body {
        if instruction.id == id {
            return Some(&instruction.op);
        }
        let nested = match &instruction.op {
            Op::If {
                then, otherwise, ..
            } => find_op(&then.body, id).or_else(|| find_op(&otherwise.body, id)),
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter()
                .find_map(|(_, block)| find_op(&block.body, id))
                .or_else(|| find_op(&otherwise.body, id)),
            Op::Loop { until, body, .. } => until
                .as_ref()
                .and_then(|block| find_op(&block.body, id))
                .or_else(|| find_op(&body.body, id)),
            _ => None,
        };
        if nested.is_some() {
            return nested;
        }
    }
    None
}

struct Ctx<'a> {
    structs: &'a [StructDef],
    writers: &'a [bool],
    memories: &'a [bool],
}

/// Where a value came from when it is part of one element of an input:
/// which input, the address, the element's own type and the way from
/// the whole to this part.
#[derive(Clone)]
struct Read {
    input: u32,
    at: Option<ValueId>,
    element: Type,
    /// The read of the whole element this part came out of.
    whole: ValueId,
    path: Vec<Step>,
}

impl Ctx<'_> {
    fn function(&self, function: &Function) -> Function {
        // The short-circuits whose other side is free come first, so
        // what they compute is one instruction again and a loop may
        // lift it out.
        let body = self.folded(&function.body);
        let body = self.hoisted(&body);
        let (body, narrowed) = self.narrowed(&body);
        // A read every part of which is now read on its own is not a
        // read any more, so the name it was given goes with it: showing
        // it would mean doing work the Kernel no longer does. A trace
        // of a Kernel a document owns walks the source instead, where
        // every `let` still is.
        let bindings: Vec<(ValueId, String)> = function
            .bindings
            .iter()
            .filter(|(id, _)| !narrowed.contains(&id.0))
            .cloned()
            .collect();
        let function = &Function {
            name: function.name.clone(),
            args: function.args.clone(),
            body,
            results: function.results.clone(),
            bindings,
        };
        let mut pass = Simplify {
            ctx: self,
            alias: HashMap::new(),
            defs: HashMap::new(),
            scopes: Vec::new(),
        };
        let body = pass.block(&function.body);
        let results: Vec<ValueId> = function.results.iter().map(|id| pass.map(*id)).collect();
        let bindings: Vec<(ValueId, String)> = function
            .bindings
            .iter()
            .map(|(id, name)| (pass.map(*id), name.clone()))
            .collect();
        // Dead values go; what a result, a binding or an effect needs stays.
        let mut used: HashMap<u32, ()> = HashMap::new();
        for id in results.iter().chain(bindings.iter().map(|(id, _)| id)) {
            used.insert(id.0, ());
        }
        let body = self.sweep(body, &mut used);
        // Pre-order numbering again.
        let mut renumber = Renumber {
            ids: BTreeMap::new(),
            next: 0,
        };
        let body = renumber.block(&body);
        Function {
            name: function.name.clone(),
            args: function.args.clone(),
            body,
            results: results.iter().map(|id| renumber.map(*id)).collect(),
            bindings: bindings
                .into_iter()
                .map(|(id, name)| (renumber.map(id), name))
                .collect(),
        }
    }

    /// A body that reads only the words it uses: a field, a column or an
    /// element taken out of a whole element of an input becomes a read
    /// of that part alone. A Kernel whose input holds a `struct` of
    /// arrays would otherwise copy every word of an element into every
    /// element of its own domain to reach three of them, which is the
    /// whole of such a frame.
    ///
    /// The whole read stays where anything still wants the whole; the
    /// sweep takes it when nothing does. Memories are left alone: a
    /// `put` between the read and the part would be seen by the part
    /// and not by the copy that was there before.
    /// The body, and the whole reads the parts taken out of them left
    /// with nothing to do.
    fn narrowed(&self, body: &[Instruction]) -> (Vec<Instruction>, HashSet<u32>) {
        let mut reads: HashMap<u32, Read> = HashMap::new();
        let mut narrowed: HashSet<u32> = HashSet::new();
        let body = self.narrow_body(body, &mut reads, &mut narrowed);
        (body, narrowed)
    }

    fn narrow_body(
        &self,
        body: &[Instruction],
        reads: &mut HashMap<u32, Read>,
        narrowed: &mut HashSet<u32>,
    ) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(body.len());
        for instruction in body {
            // A part of something already known to be part of an element.
            let step = match &instruction.op {
                Op::Extract(source, k) => reads
                    .get(&source.0)
                    .map(|read| (read.clone(), Step::Field(*k))),
                Op::Subscript(array, index) => reads
                    .get(&array.0)
                    .map(|read| (read.clone(), Step::Index(*index))),
                _ => None,
            };
            if let Some((read, step)) = step {
                let mut path = read.path;
                path.push(step);
                // The path names this very value, or the rewrite is
                // not the one the parts made and is left alone.
                let leads = crate::part_layout(self.structs, read.element, &path)
                    .is_some_and(|layout| layout.ty == instruction.ty);
                if leads {
                    narrowed.insert(read.whole.0);
                    reads.insert(
                        instruction.id.0,
                        Read {
                            input: read.input,
                            at: read.at,
                            element: read.element,
                            whole: read.whole,
                            path: path.clone(),
                        },
                    );
                    out.push(Instruction {
                        id: instruction.id,
                        ty: instruction.ty,
                        op: Op::Part {
                            input: read.input,
                            at: read.at,
                            path,
                        },
                    });
                    continue;
                }
            }
            let op = match &instruction.op {
                Op::At(input, at) if self.narrowable(*input, instruction.ty) => {
                    reads.insert(
                        instruction.id.0,
                        Read {
                            input: *input,
                            at: Some(*at),
                            element: instruction.ty,
                            whole: instruction.id,
                            path: Vec::new(),
                        },
                    );
                    instruction.op.clone()
                }
                Op::Sample(input) if self.narrowable(*input, instruction.ty) => {
                    reads.insert(
                        instruction.id.0,
                        Read {
                            input: *input,
                            at: None,
                            element: instruction.ty,
                            whole: instruction.id,
                            path: Vec::new(),
                        },
                    );
                    instruction.op.clone()
                }
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => Op::If {
                    condition: *condition,
                    then: self.narrow_block(then, reads, narrowed),
                    otherwise: self.narrow_block(otherwise, reads, narrowed),
                },
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: *scrutinee,
                    arms: arms
                        .iter()
                        .map(|(values, block)| {
                            (values.clone(), self.narrow_block(block, reads, narrowed))
                        })
                        .collect(),
                    otherwise: self.narrow_block(otherwise, reads, narrowed),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
                    bound: *bound,
                    init: *init,
                    until: until
                        .as_ref()
                        .map(|block| self.narrow_block(block, reads, narrowed)),
                    body: self.narrow_block(body, reads, narrowed),
                },
                other => other.clone(),
            };
            out.push(Instruction {
                id: instruction.id,
                ty: instruction.ty,
                op,
            });
        }
        out
    }

    fn narrow_block(
        &self,
        block: &Block,
        reads: &mut HashMap<u32, Read>,
        narrowed: &mut HashSet<u32>,
    ) -> Block {
        Block {
            body: self.narrow_body(&block.body, reads, narrowed),
            result: block.result,
        }
    }

    /// Whether a read of this input is worth narrowing: an element that
    /// is a record or an array, which is where the words add up, and
    /// not a memory the steps write.
    fn narrowable(&self, input: u32, element: Type) -> bool {
        matches!(element, Type::Struct(_))
            && !self.memories.get(input as usize).copied().unwrap_or(true)
    }

    /// A body with every short-circuit whose other side is free written
    /// as one operation: `a && b` asks `b` either way when asking costs
    /// nothing and cannot fail, which leaves a value a loop can lift
    /// out and a backend need not branch for.
    fn folded(&self, body: &[Instruction]) -> Vec<Instruction> {
        let settled = |block: &Block| match (block.body.len(), block.body.first()) {
            (1, Some(instruction)) if instruction.id == block.result => match instruction.op {
                Op::Const(Value::Bool(value)) => Some(value),
                _ => None,
            },
            _ => None,
        };
        let cheap = |block: &Block| {
            block.body.len() <= 4
                && block.body.iter().all(|instruction| {
                    !matches!(
                        instruction.op,
                        Op::If { .. } | Op::Match { .. } | Op::Loop { .. } | Op::Call(..)
                    ) && !self.has_effects(&instruction.op)
                        && self.cse_safe(&instruction.op)
                        && !can_fail(instruction.ty, &instruction.op)
                })
        };
        let mut out: Vec<Instruction> = Vec::with_capacity(body.len());
        for instruction in body {
            let op = match &instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    let then = self.folded_block(then);
                    let otherwise = self.folded_block(otherwise);
                    match (settled(&then), settled(&otherwise)) {
                        (Some(true), None) if cheap(&otherwise) => {
                            let result = otherwise.result;
                            out.extend(otherwise.body);
                            Op::Binary(BinaryOp::Or, *condition, result)
                        }
                        (None, Some(false)) if cheap(&then) => {
                            let result = then.result;
                            out.extend(then.body);
                            Op::Binary(BinaryOp::And, *condition, result)
                        }
                        _ => Op::If {
                            condition: *condition,
                            then,
                            otherwise,
                        },
                    }
                }
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: *scrutinee,
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.folded_block(block)))
                        .collect(),
                    otherwise: self.folded_block(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
                    bound: *bound,
                    init: *init,
                    until: until.as_ref().map(|block| self.folded_block(block)),
                    body: self.folded_block(body),
                },
                other => other.clone(),
            };
            out.push(Instruction {
                id: instruction.id,
                ty: instruction.ty,
                op,
            });
        }
        out
    }

    fn folded_block(&self, block: &Block) -> Block {
        Block {
            body: self.folded(&block.body),
            result: block.result,
        }
    }

    /// A body with every loop's unchanging work lifted out of it: an
    /// instruction whose operands all come from outside the loop gives
    /// the same value every iteration, so the loop computes it once.
    /// Only operations that cannot fail move, so a run that fails still
    /// reports the failure it did, of the element it did.
    fn hoisted(&self, body: &[Instruction]) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(body.len());
        for instruction in body {
            let op = match &instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => Op::If {
                    condition: *condition,
                    then: self.hoisted_block(then),
                    otherwise: self.hoisted_block(otherwise),
                },
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: *scrutinee,
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.hoisted_block(block)))
                        .collect(),
                    otherwise: self.hoisted_block(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => {
                    // Innermost first, so what an inner loop lifts can
                    // leave the outer one too.
                    let inner = self.hoisted_block(body);
                    let mut kept = Vec::with_capacity(inner.body.len());
                    // Everything the loop defines is inside it, until an
                    // instruction leaves: then its value is outside.
                    let mut within: HashSet<u32> = defined(&inner.body);
                    if let Some(until) = until {
                        within.extend(defined(&until.body));
                    }
                    for instruction in inner.body {
                        let liftable = !self.has_effects(&instruction.op)
                            && self.cse_safe(&instruction.op)
                            && !can_fail(instruction.ty, &instruction.op)
                            && !matches!(
                                instruction.op,
                                Op::If { .. } | Op::Match { .. } | Op::Loop { .. } | Op::Call(..)
                            )
                            && operands(&instruction.op)
                                .iter()
                                .all(|operand| !within.contains(&operand.0));
                        if liftable {
                            within.remove(&instruction.id.0);
                            out.push(instruction);
                        } else {
                            kept.push(instruction);
                        }
                    }
                    Op::Loop {
                        count: *count,
                        bound: *bound,
                        init: *init,
                        until: until.as_ref().map(|block| self.hoisted_block(block)),
                        body: Block {
                            body: kept,
                            result: inner.result,
                        },
                    }
                }
                other => other.clone(),
            };
            out.push(Instruction {
                id: instruction.id,
                ty: instruction.ty,
                op,
            });
        }
        out
    }

    fn hoisted_block(&self, block: &Block) -> Block {
        Block {
            body: self.hoisted(&block.body),
            result: block.result,
        }
    }

    /// Whether an instruction must run for what it does, not for its value.
    fn has_effects(&self, op: &Op) -> bool {
        match op {
            Op::Put(..) => true,
            Op::Call(callee, _) => self.writers[*callee as usize],
            Op::If {
                then, otherwise, ..
            } => self.block_has_effects(then) || self.block_has_effects(otherwise),
            Op::Match {
                arms, otherwise, ..
            } => {
                arms.iter().any(|(_, block)| self.block_has_effects(block))
                    || self.block_has_effects(otherwise)
            }
            Op::Loop { until, body, .. } => {
                until.as_ref().is_some_and(|b| self.block_has_effects(b))
                    || self.block_has_effects(body)
            }
            _ => false,
        }
    }

    fn block_has_effects(&self, block: &Block) -> bool {
        block.body.iter().any(|i| self.has_effects(&i.op))
    }

    /// Whether two instances of an operation on the same operands give
    /// the same value: not a write, not a loop's own carried value or
    /// index, not a read of a memory the steps change, not a block.
    fn cse_safe(&self, op: &Op) -> bool {
        match op {
            Op::Put(..) | Op::Carried | Op::Index => false,
            Op::If { .. } | Op::Match { .. } | Op::Loop { .. } => false,
            Op::Call(callee, _) => !self.writers[*callee as usize],
            Op::Sample(input)
            | Op::At(input, _)
            | Op::AtLinear(input, _)
            | Op::Part { input, .. } => {
                !self.memories.get(*input as usize).copied().unwrap_or(true)
            }
            _ => true,
        }
    }

    /// Drops the instructions of a block nothing needs, marking what the
    /// kept ones need; instructions are visited last to first, so a use
    /// is seen before its definition.
    fn sweep(&self, body: Vec<Instruction>, used: &mut HashMap<u32, ()>) -> Vec<Instruction> {
        let mut kept: Vec<Instruction> = Vec::with_capacity(body.len());
        for instruction in body.into_iter().rev() {
            if !used.contains_key(&instruction.id.0) && !self.has_effects(&instruction.op) {
                continue;
            }
            let op = match instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    used.insert(condition.0, ());
                    Op::If {
                        condition,
                        then: self.sweep_block(then, used),
                        otherwise: self.sweep_block(otherwise, used),
                    }
                }
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    used.insert(scrutinee.0, ());
                    Op::Match {
                        scrutinee,
                        arms: arms
                            .into_iter()
                            .map(|(values, block)| (values, self.sweep_block(block, used)))
                            .collect(),
                        otherwise: self.sweep_block(otherwise, used),
                    }
                }
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => {
                    used.insert(init.0, ());
                    if let Some(bound) = bound {
                        used.insert(bound.0, ());
                    }
                    Op::Loop {
                        count,
                        bound,
                        init,
                        until: until.map(|block| self.sweep_block(block, used)),
                        body: self.sweep_block(body, used),
                    }
                }
                other => {
                    for operand in operands(&other) {
                        used.insert(operand.0, ());
                    }
                    other
                }
            };
            kept.push(Instruction {
                id: instruction.id,
                ty: instruction.ty,
                op,
            });
        }
        kept.reverse();
        kept
    }

    fn sweep_block(&self, block: Block, used: &mut HashMap<u32, ()>) -> Block {
        used.insert(block.result.0, ());
        Block {
            body: self.sweep(block.body, used),
            result: block.result,
        }
    }
}

/// The values an operation reads, blocks aside.
/// Every value id a body defines, nested blocks counted.
fn defined(body: &[Instruction]) -> HashSet<u32> {
    let mut ids = HashSet::new();
    fn walk(body: &[Instruction], ids: &mut HashSet<u32>) {
        for instruction in body {
            ids.insert(instruction.id.0);
            match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => {
                    walk(&then.body, ids);
                    walk(&otherwise.body, ids);
                }
                Op::Match {
                    arms, otherwise, ..
                } => {
                    for (_, block) in arms {
                        walk(&block.body, ids);
                    }
                    walk(&otherwise.body, ids);
                }
                Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, ids);
                    }
                    walk(&body.body, ids);
                }
                _ => {}
            }
        }
    }
    walk(body, &mut ids);
    ids
}

/// Whether an operation can fail where it stands: integer arithmetic
/// overflows and divides by zero, and a run says so of the element it
/// happened in. Floats have no failures; a Field read is clamped.
/// How many operations one compile may run while folding a function's
/// loops. A table built in a loop is a hundred or so; this stops a
/// Kernel whose loops are large from making a compile long.
const LOOP_BUDGET: u32 = 20_000;

fn can_fail(ty: Type, op: &Op) -> bool {
    let integer = matches!(ty, Type::Int | Type::IVec2 | Type::IVec3);
    integer && matches!(op, Op::Binary(..) | Op::Intrinsic(..))
}

fn operands(op: &Op) -> Vec<ValueId> {
    match op {
        Op::Unary(_, a) => vec![*a],
        Op::Binary(_, a, b) => vec![*a, *b],
        Op::Construct(parts) => parts.clone(),
        Op::Extract(source, _) => vec![*source],
        Op::Subscript(array, index) => vec![*array, *index],
        Op::Call(_, args) | Op::Intrinsic(_, args) => args.clone(),
        Op::At(_, at) | Op::AtLinear(_, at) | Op::Key(at) | Op::Button(at) => vec![*at],
        Op::Part { at, path, .. } => at
            .iter()
            .copied()
            .chain(path.iter().filter_map(|step| match step {
                Step::Index(index) => Some(*index),
                Step::Field(_) => None,
            }))
            .collect(),
        Op::Put(_, address, value) => vec![*address, *value],
        _ => Vec::new(),
    }
}

/// The first pass: operands renamed through the aliases so far, parts
/// taken straight from constructions, literals folded, repeats shared.
struct Simplify<'a> {
    ctx: &'a Ctx<'a>,
    /// A value that is another value, by id.
    alias: HashMap<u32, u32>,
    /// What each kept value is, for the rewrites that look at it.
    defs: HashMap<u32, (Type, Op)>,
    /// The values available for sharing, one table per open block.
    scopes: Vec<HashMap<String, u32>>,
}

impl Simplify<'_> {
    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.alias.get(&id.0).unwrap_or(&id.0))
    }

    /// An `if`, simplified, and as one `and` or `or` where it is a
    /// short-circuit whose other side is cheap: that side's
    /// instructions are emitted before it, so they must be pure and
    /// unable to fail. `a && b` asks `b` either way when asking is free.
    fn folded_if(
        &mut self,
        condition: ValueId,
        then: &Block,
        otherwise: &Block,
        out: &mut Vec<Instruction>,
    ) -> Op {
        let condition = self.map(condition);
        let then = self.nested(then);
        let otherwise = self.nested(otherwise);
        let settled = |block: &Block| match (block.body.len(), block.body.first()) {
            (1, Some(instruction)) if instruction.id == block.result => match instruction.op {
                Op::Const(Value::Bool(value)) => Some(value),
                _ => None,
            },
            _ => None,
        };
        let ctx = self.ctx;
        let cheap = |block: &Block| {
            block.body.len() <= 4
                && block.body.iter().all(|instruction| {
                    !matches!(
                        instruction.op,
                        Op::If { .. } | Op::Match { .. } | Op::Loop { .. } | Op::Call(..)
                    ) && !ctx.has_effects(&instruction.op)
                        && ctx.cse_safe(&instruction.op)
                        && !can_fail(instruction.ty, &instruction.op)
                })
        };
        match (settled(&then), settled(&otherwise)) {
            // `a || b`: true where `a` holds, else what `b` says.
            (Some(true), None) if cheap(&otherwise) => {
                let result = otherwise.result;
                out.extend(otherwise.body);
                Op::Binary(BinaryOp::Or, condition, result)
            }
            // `a && b`: what `b` says where `a` holds, else false.
            (None, Some(false)) if cheap(&then) => {
                let result = then.result;
                out.extend(then.body);
                Op::Binary(BinaryOp::And, condition, result)
            }
            _ => Op::If {
                condition,
                then,
                otherwise,
            },
        }
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        self.scopes.push(HashMap::new());
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            let ty = instruction.ty;
            if let Op::If {
                condition,
                then,
                otherwise,
            } = &instruction.op
            {
                let op = self.folded_if(*condition, then, otherwise, &mut out);
                self.defs.insert(instruction.id.0, (ty, op.clone()));
                out.push(Instruction {
                    id: instruction.id,
                    ty,
                    op,
                });
                continue;
            }
            let op = match &instruction.op {
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: self.map(*scrutinee),
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.nested(block)))
                        .collect(),
                    otherwise: self.nested(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => {
                    let loop_op = Op::Loop {
                        count: *count,
                        bound: bound.map(|bound| self.map(bound)),
                        init: self.map(*init),
                        until: until.as_ref().map(|block| self.nested(block)),
                        body: self.nested(body),
                    };
                    // A loop that only computes, over a count and an
                    // initial value known here, is run here: a Kernel
                    // would run it again for every element.
                    match self.run_loop(&loop_op) {
                        Some(value) => Op::Const(value),
                        None => loop_op,
                    }
                }
                other => {
                    let mut op = other.clone();
                    rename(&mut op, &|id| self.map(id));
                    op
                }
            };
            // A part straight out of its construction is that part.
            if let Op::Extract(source, k) = &op
                && let Some((_, Op::Construct(parts))) = self.defs.get(&source.0)
                && let Some(part) = parts.get(*k as usize)
                && self
                    .defs
                    .get(&part.0)
                    .is_some_and(|(part_ty, _)| *part_ty == ty)
            {
                self.alias.insert(instruction.id.0, part.0);
                continue;
            }
            // A vector rebuilt from all of its own components is the vector.
            if let Op::Construct(parts) = &op
                && let Some(whole) = self.rebuilt(parts, ty)
            {
                self.alias.insert(instruction.id.0, whole);
                continue;
            }
            // Literals fold when the backends' arithmetic gives one value.
            let op = match self.fold(&op, ty) {
                Some(value) => Op::Const(value),
                None => op,
            };
            // A repeat of a pure operation is the first one.
            let key = self.ctx.cse_safe(&op).then(|| format!("{ty}|{op:?}"));
            if let Some(key) = &key
                && let Some(first) = self.scopes.iter().rev().find_map(|scope| scope.get(key))
            {
                self.alias.insert(instruction.id.0, *first);
                continue;
            }
            if let Some(key) = key
                && let Some(scope) = self.scopes.last_mut()
            {
                scope.insert(key, instruction.id.0);
            }
            self.defs.insert(instruction.id.0, (ty, op.clone()));
            out.push(Instruction {
                id: instruction.id,
                ty,
                op,
            });
        }
        self.scopes.pop();
        out
    }

    fn nested(&mut self, block: &Block) -> Block {
        let body = self.block(&block.body);
        Block {
            body,
            result: self.map(block.result),
        }
    }

    /// The vector, matrix, array or struct these parts are all of, in
    /// order, when they are.
    fn rebuilt(&self, parts: &[ValueId], ty: Type) -> Option<u32> {
        let (Op::Extract(first, 0), _) = (&self.defs.get(&parts[0].0)?.1, ()) else {
            return None;
        };
        let whole = first.0;
        let (whole_ty, _) = self.defs.get(&whole)?;
        if *whole_ty != ty {
            return None;
        }
        let count = match ty {
            Type::Struct(index) => self.ctx.structs[index as usize].fields.len(),
            other => other
                .order()
                .or(other.lanes().map(|(n, _)| n))
                .map(usize::from)?,
        };
        if parts.len() != count {
            return None;
        }
        let all = parts.iter().enumerate().all(|(k, part)| {
            matches!(self.defs.get(&part.0), Some((_, Op::Extract(source, at))) if source.0 == whole && *at as usize == k)
        });
        all.then_some(whole)
    }

    /// A loop with a known count and a literal initial value whose body
    /// only computes: run here, so what it gives is a literal. Nothing
    /// that reads a Field, writes a memory, calls a function or can
    /// fail is run, and the budget bounds what a compile spends on it.
    fn run_loop(&self, op: &Op) -> Option<Value> {
        let Op::Loop {
            count,
            bound: None,
            init,
            until,
            body,
        } = op
        else {
            return None;
        };
        let mut env: HashMap<u32, Value> = HashMap::new();
        let mut carried = self.known(*init, &env)?;
        let mut budget = LOOP_BUDGET;
        for index in 0..*count {
            if let Some(until) = until
                && self.run_block(until, carried, index, &mut env, &mut budget)?
                    == Value::Bool(true)
            {
                return Some(carried);
            }
            carried = self.run_block(body, carried, index, &mut env, &mut budget)?;
        }
        Some(carried)
    }

    /// A block of the loop being run: every instruction in order, then
    /// what it yields.
    fn run_block(
        &self,
        block: &Block,
        carried: Value,
        index: u32,
        env: &mut HashMap<u32, Value>,
        budget: &mut u32,
    ) -> Option<Value> {
        for instruction in &block.body {
            *budget = budget.checked_sub(1)?;
            let value = self.run_op(&instruction.op, carried, index, env, budget)?;
            env.insert(instruction.id.0, value);
        }
        self.known(block.result, env)
    }

    /// One operation of the loop being run.
    fn run_op(
        &self,
        op: &Op,
        carried: Value,
        index: u32,
        env: &mut HashMap<u32, Value>,
        budget: &mut u32,
    ) -> Option<Value> {
        match op {
            Op::Const(value) => Some(*value),
            Op::Carried => Some(carried),
            Op::Index => Some(Value::Int(index as i32)),
            Op::Unary(unary, a) => fold_unary(*unary, self.known(*a, env)?),
            Op::Binary(binary, a, b) => {
                fold_binary(*binary, self.known(*a, env)?, self.known(*b, env)?)
            }
            Op::Intrinsic(intrinsic, args) => {
                let values: Option<Vec<Value>> =
                    args.iter().map(|arg| self.known(*arg, env)).collect();
                fold_intrinsic(*intrinsic, &values?)
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let taken = match self.known(*condition, env)? {
                    Value::Bool(true) => then,
                    Value::Bool(false) => otherwise,
                    _ => return None,
                };
                self.run_block(taken, carried, index, env, budget)
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let Value::Int(value) = self.known(*scrutinee, env)? else {
                    return None;
                };
                let taken = arms
                    .iter()
                    .find(|(values, _)| values.contains(&value))
                    .map(|(_, block)| block)
                    .unwrap_or(otherwise);
                self.run_block(taken, carried, index, env, budget)
            }
            // A loop inside carries its own value and counts its own
            // iterations.
            Op::Loop {
                count,
                bound: None,
                init,
                until,
                body,
            } => {
                let mut inner = self.known(*init, env)?;
                for step in 0..*count {
                    *budget = budget.checked_sub(1)?;
                    if let Some(until) = until
                        && self.run_block(until, inner, step, env, budget)? == Value::Bool(true)
                    {
                        return Some(inner);
                    }
                    inner = self.run_block(body, inner, step, env, budget)?;
                }
                Some(inner)
            }
            _ => None,
        }
    }

    /// What a value is where the loop is being run: one the run has
    /// computed, or a literal from outside it.
    fn known(&self, id: ValueId, env: &HashMap<u32, Value>) -> Option<Value> {
        if let Some(value) = env.get(&id.0) {
            return Some(*value);
        }
        match self.defs.get(&id.0) {
            Some((_, Op::Const(value))) => Some(*value),
            _ => None,
        }
    }

    /// The literal an operation on literals is, when one and never a
    /// failure.
    fn fold(&self, op: &Op, ty: Type) -> Option<Value> {
        if !matches!(
            ty,
            Type::Float | Type::Int | Type::Bool | Type::Vec2 | Type::Vec3 | Type::Vec4
        ) {
            return None;
        }
        let literal = |id: &ValueId| match self.defs.get(&id.0) {
            Some((_, Op::Const(value))) => Some(*value),
            _ => None,
        };
        match op {
            Op::Unary(unary, a) => fold_unary(*unary, literal(a)?),
            Op::Binary(binary, a, b) => fold_binary(*binary, literal(a)?, literal(b)?),
            Op::Intrinsic(intrinsic, args) => {
                let values: Option<Vec<Value>> = args.iter().map(literal).collect();
                fold_intrinsic(*intrinsic, &values?)
            }
            _ => None,
        }
    }
}

/// Renames an operation's operands, blocks aside.
fn rename(op: &mut Op, map: &dyn Fn(ValueId) -> ValueId) {
    match op {
        Op::Unary(_, a) => *a = map(*a),
        Op::Binary(_, a, b) => {
            *a = map(*a);
            *b = map(*b);
        }
        Op::Construct(parts) => {
            for part in parts {
                *part = map(*part);
            }
        }
        Op::Extract(source, _) => *source = map(*source),
        Op::Subscript(array, index) => {
            *array = map(*array);
            *index = map(*index);
        }
        Op::Call(_, args) | Op::Intrinsic(_, args) => {
            for arg in args {
                *arg = map(*arg);
            }
        }
        Op::At(_, at) | Op::AtLinear(_, at) | Op::Key(at) | Op::Button(at) => *at = map(*at),
        Op::Part { at, path, .. } => {
            if let Some(at) = at {
                *at = map(*at);
            }
            for step in path {
                if let Step::Index(index) = step {
                    *index = map(*index);
                }
            }
        }
        Op::Put(_, address, value) => {
            *address = map(*address);
            *value = map(*value);
        }
        _ => {}
    }
}

/// The components of a literal as floats, and back.
fn floats(value: Value) -> Option<Vec<f32>> {
    match value {
        Value::Float(v) => Some(vec![v]),
        Value::Vec2(v) => Some(v.to_vec()),
        Value::Vec3(v) => Some(v.to_vec()),
        Value::Vec4(v) => Some(v.to_vec()),
        _ => None,
    }
}

fn from_floats(parts: &[f32]) -> Option<Value> {
    match parts {
        [x] => Some(Value::Float(*x)),
        [x, y] => Some(Value::Vec2([*x, *y])),
        [x, y, z] => Some(Value::Vec3([*x, *y, *z])),
        [x, y, z, w] => Some(Value::Vec4([*x, *y, *z, *w])),
        _ => None,
    }
}

/// Two float literals componentwise, a `Float` beside a vector broadcast.
fn zip(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Option<Value> {
    let (a, b) = (floats(a)?, floats(b)?);
    let n = a.len().max(b.len());
    if a.len() != 1 && b.len() != 1 && a.len() != b.len() {
        return None;
    }
    let parts: Vec<f32> = (0..n)
        .map(|k| {
            let x = if a.len() == 1 { a[0] } else { a[k] };
            let y = if b.len() == 1 { b[0] } else { b[k] };
            f(x, y)
        })
        .collect();
    from_floats(&parts)
}

fn map(a: Value, f: impl Fn(f32) -> f32) -> Option<Value> {
    let parts: Vec<f32> = floats(a)?.into_iter().map(f).collect();
    from_floats(&parts)
}

fn fold_unary(op: UnaryOp, a: Value) -> Option<Value> {
    match (op, a) {
        (UnaryOp::Neg, Value::Int(v)) => v.checked_neg().map(Value::Int),
        (UnaryOp::Neg, other) => map(other, |v| -v),
        (UnaryOp::Not, Value::Bool(v)) => Some(Value::Bool(!v)),
        _ => None,
    }
}

fn fold_binary(op: BinaryOp, a: Value, b: Value) -> Option<Value> {
    use BinaryOp::*;
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => match op {
            Add => x.checked_add(y).map(Value::Int),
            Sub => x.checked_sub(y).map(Value::Int),
            Mul => x.checked_mul(y).map(Value::Int),
            Div if y != 0 => x.checked_div(y).map(Value::Int),
            Rem if y != 0 => x.checked_rem(y).map(Value::Int),
            Less => Some(Value::Bool(x < y)),
            LessEqual => Some(Value::Bool(x <= y)),
            Greater => Some(Value::Bool(x > y)),
            GreaterEqual => Some(Value::Bool(x >= y)),
            Equal => Some(Value::Bool(x == y)),
            NotEqual => Some(Value::Bool(x != y)),
            BitAnd => Some(Value::Int(x & y)),
            BitOr => Some(Value::Int(x | y)),
            BitXor => Some(Value::Int(x ^ y)),
            // The count modulo 32, as every backend takes it.
            Shl => Some(Value::Int(x.wrapping_shl(y as u32))),
            Shr => Some(Value::Int(x.wrapping_shr(y as u32))),
            Ushr => Some(Value::Int((x as u32).wrapping_shr(y as u32) as i32)),
            _ => None,
        },
        (Value::Bool(x), Value::Bool(y)) => match op {
            And => Some(Value::Bool(x && y)),
            Or => Some(Value::Bool(x || y)),
            Equal => Some(Value::Bool(x == y)),
            NotEqual => Some(Value::Bool(x != y)),
            _ => None,
        },
        (a, b) => match op {
            Add => zip(a, b, |x, y| x + y),
            Sub => zip(a, b, |x, y| x - y),
            Mul => zip(a, b, |x, y| x * y),
            Div => zip(a, b, |x, y| x / y),
            Rem => zip(a, b, |x, y| x % y),
            Less | LessEqual | Greater | GreaterEqual => {
                let (Value::Float(x), Value::Float(y)) = (a, b) else {
                    return None;
                };
                Some(Value::Bool(match op {
                    Less => x < y,
                    LessEqual => x <= y,
                    Greater => x > y,
                    _ => x >= y,
                }))
            }
            Equal | NotEqual => {
                let (x, y) = (floats(a)?, floats(b)?);
                if x.len() != y.len() {
                    return None;
                }
                let same = x.iter().zip(&y).all(|(p, q)| p == q);
                Some(Value::Bool(if op == Equal { same } else { !same }))
            }
            _ => None,
        },
    }
}

fn fold_intrinsic(intrinsic: Intrinsic, args: &[Value]) -> Option<Value> {
    use Intrinsic::*;
    let sign = |v: f32| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    match (intrinsic, args) {
        (Floor, [a]) => map(*a, f32::floor),
        (Ceil, [a]) => map(*a, f32::ceil),
        (Round, [a]) => map(*a, f32::round_ties_even),
        (Trunc, [a]) => map(*a, f32::trunc),
        (Fract, [a]) => map(*a, |v| v - v.floor()),
        (Abs, [Value::Int(v)]) => Some(Value::Int(v.wrapping_abs())),
        (Abs, [a]) => map(*a, f32::abs),
        (Sign, [Value::Int(v)]) => Some(Value::Int(v.signum())),
        (Sign, [a]) => map(*a, sign),
        (Min, [Value::Int(x), Value::Int(y)]) => Some(Value::Int((*x).min(*y))),
        (Max, [Value::Int(x), Value::Int(y)]) => Some(Value::Int((*x).max(*y))),
        (Min, [a, b]) => zip(*a, *b, f32::min),
        (Max, [a, b]) => zip(*a, *b, f32::max),
        (Clamp, [Value::Int(v), Value::Int(low), Value::Int(high)]) => {
            Some(Value::Int((*v).max(*low).min(*high)))
        }
        (Clamp, [v, low, high]) => {
            let floored = zip(*v, *low, f32::max)?;
            zip(floored, *high, f32::min)
        }
        (ToFloat, [Value::Int(v)]) => Some(Value::Float(*v as f32)),
        (ToFloat, [Value::Bool(v)]) => Some(Value::Float(if *v { 1.0 } else { 0.0 })),
        (ToInt, [Value::Float(v)]) => Some(Value::Int(*v as i32)),
        (ToInt, [Value::Bool(v)]) => Some(Value::Int(i32::from(*v))),
        _ => None,
    }
}

/// Pre-order numbering of the kept instructions.
struct Renumber {
    ids: BTreeMap<u32, u32>,
    next: u32,
}

impl Renumber {
    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.ids.get(&id.0).unwrap_or(&id.0))
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            let id = ValueId(self.next);
            self.next += 1;
            self.ids.insert(instruction.id.0, id.0);
            let op = match &instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => Op::If {
                    condition: self.map(*condition),
                    then: self.nested(then),
                    otherwise: self.nested(otherwise),
                },
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: self.map(*scrutinee),
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.nested(block)))
                        .collect(),
                    otherwise: self.nested(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
                    bound: bound.map(|bound| self.map(bound)),
                    init: self.map(*init),
                    until: until.as_ref().map(|block| self.nested(block)),
                    body: self.nested(body),
                },
                other => {
                    let mut op = other.clone();
                    rename(&mut op, &|id| self.map(id));
                    op
                }
            };
            out.push(Instruction {
                id,
                ty: instruction.ty,
                op,
            });
        }
        out
    }

    fn nested(&mut self, block: &Block) -> Block {
        let body = self.block(&block.body);
        Block {
            body,
            result: self.map(block.result),
        }
    }
}
