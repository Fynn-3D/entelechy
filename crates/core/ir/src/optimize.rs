//! The optimizer: a Kernel's computation in fewer instructions, every
//! word the same. Each rewrite is exact by construction:
//!
//! - a part taken straight out of its construction is that part; a
//!   vector rebuilt from its own components in order is the vector;
//! - an operation on literals folds where the backends' arithmetic gives
//!   one value and no failure;
//! - an `if` or a `match` on a literal is the block it takes;
//! - a pure operation repeated on the same operands in one block or an
//!   enclosing one is computed once (a memory read and a writing call are
//!   impure);
//! - an unread value is dropped, except a `put`, a writing call, a block
//!   holding one, or a name a trace shows;
//! - a call to a short function is written out in place;
//! - loop-invariant work that cannot fail moves before the loop;
//! - an `each` builder short enough is written out, its `Index` the
//!   element's position; a longer one holding a loop counted at run time
//!   runs as a loop carrying its array.
//!
//! Values are then renumbered in pre-order. Floats keep their order and
//! a run reports the same failure.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Module, Op, Step, StructDef, Type, UnaryOp,
    Value, ValueId,
};

/// A function of at most this many instructions, nested blocks counted,
/// is written out where it is called.
const INLINE_LIMIT: usize = 96;

/// A function taking a record or array of at least this many words is
/// written out where it is called up to [`INLINE_BIG_LIMIT`] instructions,
/// so its argument is not copied per element and its reads narrow.
const BIG_ARG_WORDS: usize = 32;
/// The most instructions of a function taking a wide argument that is
/// still written out where it is called.
const INLINE_BIG_LIMIT: usize = 1024;
/// An `each` builder whose elements together hold at most this many
/// instructions, nested blocks counted, is written out element by
/// element; a longer one runs as a builder's loop, or, holding a loop
/// counted at run time, as a loop carrying its array
/// ([`counts_at_run_time`]).
const UNROLL_LIMIT: usize = 512;

/// Optimizes every function of a module. The result computes the same
/// words and validates whenever the input does.
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
    // Functions precede their callers, so a written-out call is already
    // optimized.
    // Whether each is short enough to write out, judged once, as it is
    // optimized.
    let mut functions: Vec<Function> = Vec::with_capacity(module.functions.len());
    let mut inlinable: Vec<bool> = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        let flattened = inline_calls(function, &functions, &inlinable);
        let optimized = ctx.function(&flattened);
        inlinable.push(writes_out(&optimized, &module.structs));
        functions.push(optimized);
    }
    let entry = ctx.function(&inline_calls(&module.entry, &functions, &inlinable));
    let merge = module
        .merge
        .as_ref()
        .map(|f| ctx.function(&inline_calls(f, &functions, &inlinable)));
    let init = module
        .init
        .as_ref()
        .map(|f| ctx.function(&inline_calls(f, &functions, &inlinable)));
    let (functions, entry, merge, init) = ctx.read_through(functions, entry, merge, init);
    prune(Module {
        manifest: module.manifest,
        structs: module.structs,
        functions,
        entry,
        merge,
        init,
    })
}

// ---------------------------------------------------------------- arguments read where they are

/// A value a function's argument leads to: the argument's read (`root`)
/// and the way from it.
#[derive(Clone)]
struct Lead {
    root: u32,
    path: Vec<Step>,
}

/// Every value a function's argument `k` leads to by fields and
/// elements, if the argument itself is used only so or handed whole to
/// calls (returned as those calls, callee and place); `None` when it is
/// wanted whole otherwise.
/// What an argument leads to, and the calls it is handed to whole.
type Leads = (HashMap<u32, Lead>, Vec<(u32, usize)>);

fn leads_of(function: &Function, k: u32) -> Option<Leads> {
    fn walk(body: &[Instruction], k: u32, leads: &mut HashMap<u32, Lead>) {
        for instruction in body {
            let lead = match &instruction.op {
                Op::Arg(i) if *i == k => Some(Lead {
                    root: instruction.id.0,
                    path: Vec::new(),
                }),
                Op::Extract(source, field) => leads.get(&source.0).map(|l| {
                    let mut path = l.path.clone();
                    path.push(Step::Field(*field));
                    Lead { root: l.root, path }
                }),
                Op::Subscript(source, index) => leads.get(&source.0).map(|l| {
                    let mut path = l.path.clone();
                    path.push(Step::Index(*index));
                    Lead { root: l.root, path }
                }),
                _ => None,
            };
            if let Some(lead) = lead {
                leads.insert(instruction.id.0, lead);
            }
            for block in instruction.op.blocks() {
                walk(&block.body, k, leads);
            }
        }
    }
    let mut leads = HashMap::new();
    walk(&function.body, k, &mut leads);
    let roots: HashSet<u32> = leads
        .iter()
        .filter(|(_, l)| l.path.is_empty())
        .map(|(id, _)| *id)
        .collect();
    if roots.is_empty() {
        return None;
    }
    // A root is used only to take parts out of, or handed whole to a call.
    fn check(body: &[Instruction], roots: &HashSet<u32>, calls: &mut Vec<(u32, usize)>) -> bool {
        for instruction in body {
            let ok = match &instruction.op {
                Op::Extract(_, _) => true,
                Op::Subscript(_, index) => !roots.contains(&index.0),
                Op::Call(callee, args) => {
                    for (place, arg) in args.iter().enumerate() {
                        if roots.contains(&arg.0) {
                            calls.push((*callee, place));
                        }
                    }
                    true
                }
                Op::If { condition, .. } => !roots.contains(&condition.0),
                Op::Match { scrutinee, .. } => !roots.contains(&scrutinee.0),
                Op::Loop { init, bound, .. } => {
                    !roots.contains(&init.0) && !bound.is_some_and(|b| roots.contains(&b.0))
                }
                other => !operands(other).iter().any(|o| roots.contains(&o.0)),
            };
            if !ok {
                return false;
            }
            for block in instruction.op.blocks() {
                if roots.contains(&block.result.0) || !check(&block.body, roots, calls) {
                    return false;
                }
            }
        }
        true
    }
    let mut calls = Vec::new();
    let fine = check(&function.body, &roots, &mut calls)
        && !function.results.iter().any(|id| roots.contains(&id.0));
    fine.then_some((leads, calls))
}

/// An input read at an address, whole or a part of it by fields: the
/// input, the address and the fields.
type ElementRead = (u32, ValueId, Vec<u32>);

/// The functions made reading an argument where it lies, by what they
/// were made from: the function, the argument, the input, the address's
/// type and the fields from the input's element to the argument.
type Made = HashMap<(u32, u32, u32, Type, Vec<u32>), u32>;

impl Ctx<'_> {
    /// Whether a function's argument `k`, a record or array of at least
    /// [`BIG_ARG_WORDS`] words, can be read where it lies: it is only taken
    /// apart, or handed whole to functions that can read it so too.
    fn reads_through(
        &self,
        functions: &[Function],
        g: u32,
        k: u32,
        seen: &mut HashMap<(u32, u32), bool>,
    ) -> bool {
        if let Some(known) = seen.get(&(g, k)) {
            return *known;
        }
        let function = &functions[g as usize];
        let wide = function
            .args
            .get(k as usize)
            .is_some_and(|ty| crate::words(self.structs, *ty) >= BIG_ARG_WORDS);
        let answer = wide
            && !self.writers.get(g as usize).copied().unwrap_or(false)
            && leads_of(function, k).is_some_and(|(_, calls)| {
                calls.iter().all(|(callee, place)| {
                    self.reads_through(functions, *callee, *place as u32, seen)
                })
            });
        seen.insert((g, k), answer);
        answer
    }

    /// `g` with its argument `k` an address of `input`, of type `address`,
    /// the argument the part of its element `base` names (its fields,
    /// outermost first; none for the whole element), every part of it read
    /// there; made once, appended to `functions`.
    #[allow(clippy::too_many_arguments)]
    fn made_reading(
        &self,
        functions: &mut Vec<Function>,
        made: &mut Made,
        g: u32,
        k: u32,
        input: u32,
        address: Type,
        base: &[u32],
    ) -> u32 {
        if let Some(index) = made.get(&(g, k, input, address, base.to_vec())) {
            return *index;
        }
        let original = functions[g as usize].clone();
        let (leads, _) = leads_of(&original, k).expect("judged readable through");
        let mut body = original.body.clone();
        self.rewrite(&mut body, &leads, functions, made, input, address, base);
        let mut args = original.args.clone();
        args[k as usize] = address;
        let index = functions.len() as u32;
        functions.push(Function {
            name: format!("{}~{k}", original.name),
            args,
            body,
            results: original.results.clone(),
            bindings: original.bindings.clone(),
        });
        made.insert((g, k, input, address, base.to_vec()), index);
        index
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite(
        &self,
        body: &mut [Instruction],
        leads: &HashMap<u32, Lead>,
        functions: &mut Vec<Function>,
        made: &mut Made,
        input: u32,
        address: Type,
        base: &[u32],
    ) {
        for instruction in body.iter_mut() {
            if let Some(lead) = leads.get(&instruction.id.0) {
                if lead.path.is_empty() {
                    instruction.ty = address;
                } else {
                    let mut path: Vec<Step> = base.iter().map(|f| Step::Field(*f)).collect();
                    path.extend(lead.path.iter().copied());
                    instruction.op = Op::Part {
                        input,
                        at: Some(ValueId(lead.root)),
                        path,
                    };
                }
            }
            if let Op::Call(callee, args) = &instruction.op {
                let mut callee = *callee;
                for (place, arg) in args.iter().enumerate() {
                    if leads.get(&arg.0).is_some_and(|l| l.path.is_empty()) {
                        callee = self.made_reading(
                            functions,
                            made,
                            callee,
                            place as u32,
                            input,
                            address,
                            base,
                        );
                    }
                }
                instruction.op = Op::Call(callee, args.clone());
            }
            for block in instruction.op.blocks_mut() {
                self.rewrite(
                    &mut block.body,
                    leads,
                    functions,
                    made,
                    input,
                    address,
                    base,
                );
            }
        }
    }

    /// Calls handing a function a whole input element it only takes apart
    /// hand it the element's address instead, to a function made to read
    /// the parts there (`made_reading`): no Kernel holds a wide element
    /// as a value only to pass it on. Returns the functions, made ones
    /// among them, callees first, and the entry points.
    fn read_through(
        &self,
        functions: Vec<Function>,
        entry: Function,
        merge: Option<Function>,
        init: Option<Function>,
    ) -> (Vec<Function>, Function, Option<Function>, Option<Function>) {
        let count = functions.len();
        let mut functions = functions;
        let mut made: Made = HashMap::new();
        let mut touched: HashSet<usize> = HashSet::new();
        for index in 0..count {
            let mut body = std::mem::take(&mut functions[index].body);
            let args = functions[index].args.clone();
            if self.redirect(&mut body, &args, &mut functions, &mut made) {
                touched.insert(index);
            }
            functions[index].body = body;
        }
        let mut points: Vec<(Function, bool)> = Vec::new();
        for mut point in std::iter::once(entry)
            .chain(merge.clone())
            .chain(init.clone())
        {
            let mut body = std::mem::take(&mut point.body);
            let args = point.args.clone();
            let changed = self.redirect(&mut body, &args, &mut functions, &mut made);
            point.body = body;
            points.push((point, changed));
        }
        let points: Vec<Function> = points
            .into_iter()
            .map(|(point, changed)| {
                if changed {
                    self.function(&point)
                } else {
                    point
                }
            })
            .collect();
        if made.is_empty() {
            let mut points = points.into_iter();
            let entry = points.next().expect("an entry");
            let merge = merge.map(|_| points.next().expect("the merge"));
            let init = init.map(|_| points.next().expect("the init"));
            return (functions, entry, merge, init);
        }
        for (index, function) in functions.iter_mut().enumerate() {
            if index >= count || touched.contains(&index) {
                *function = self.function(function);
            }
        }
        let (functions, mut points) = callees_first(functions, points);
        let entry = points.remove(0);
        let merge = merge.map(|_| points.remove(0));
        let init = init.map(|_| points.remove(0));
        (functions, entry, merge, init)
    }

    /// A body's calls handed a whole element of an input read at an
    /// address, to a function that only takes it apart, redirected to one
    /// made to read the parts there; whether any was.
    fn redirect(
        &self,
        body: &mut [Instruction],
        args: &[Type],
        functions: &mut Vec<Function>,
        made: &mut Made,
    ) -> bool {
        let mut types: HashMap<u32, Type> = HashMap::new();
        let mut reads: HashMap<u32, ElementRead> = HashMap::new();
        fn scan(
            body: &[Instruction],
            memories: &[bool],
            types: &mut HashMap<u32, Type>,
            reads: &mut HashMap<u32, ElementRead>,
        ) {
            for instruction in body {
                types.insert(instruction.id.0, instruction.ty);
                let outside = |input: u32| !memories.get(input as usize).copied().unwrap_or(true);
                match &instruction.op {
                    Op::At(input, at) if outside(*input) => {
                        reads.insert(instruction.id.0, (*input, *at, Vec::new()));
                    }
                    // A part of an element at an address, by fields alone:
                    // read where it lies as the whole element is.
                    Op::Part {
                        input,
                        at: Some(at),
                        path,
                    } if outside(*input) => {
                        let fields: Option<Vec<u32>> = path
                            .iter()
                            .map(|step| match step {
                                Step::Field(f) => Some(*f),
                                Step::Index(_) => None,
                            })
                            .collect();
                        if let Some(fields) = fields {
                            reads.insert(instruction.id.0, (*input, *at, fields));
                        }
                    }
                    _ => {}
                }
                for block in instruction.op.blocks() {
                    scan(&block.body, memories, types, reads);
                }
            }
        }
        scan(body, self.memories, &mut types, &mut reads);
        let _ = args;
        if reads.is_empty() {
            return false;
        }
        let mut changed = false;
        self.redirect_body(body, &types, &reads, functions, made, &mut changed);
        changed
    }

    fn redirect_body(
        &self,
        body: &mut [Instruction],
        types: &HashMap<u32, Type>,
        reads: &HashMap<u32, ElementRead>,
        functions: &mut Vec<Function>,
        made: &mut Made,
        changed: &mut bool,
    ) {
        for instruction in body.iter_mut() {
            if let Op::Call(callee, args) = &instruction.op {
                let mut callee = *callee;
                let mut args = args.clone();
                for (place, arg) in args.iter_mut().enumerate() {
                    let Some((input, at, base)) = reads.get(&arg.0).cloned() else {
                        continue;
                    };
                    let mut seen = HashMap::new();
                    if !self.reads_through(functions, callee, place as u32, &mut seen) {
                        continue;
                    }
                    let address = types.get(&at.0).copied().unwrap_or(Type::Int);
                    callee = self.made_reading(
                        functions,
                        made,
                        callee,
                        place as u32,
                        input,
                        address,
                        &base,
                    );
                    *arg = at;
                    *changed = true;
                }
                instruction.op = Op::Call(callee, args);
            }
            for block in instruction.op.blocks_mut() {
                self.redirect_body(&mut block.body, types, reads, functions, made, changed);
            }
        }
    }
}

/// The functions ordered so each comes after those it calls, the first
/// written first where they are free to be, and every call, the entry
/// points' too, renumbered to match.
fn callees_first(
    functions: Vec<Function>,
    points: Vec<Function>,
) -> (Vec<Function>, Vec<Function>) {
    fn callees(body: &[Instruction], out: &mut Vec<u32>) {
        for instruction in body {
            if let Op::Call(callee, _) = &instruction.op {
                out.push(*callee);
            }
            for block in instruction.op.blocks() {
                callees(&block.body, out);
            }
        }
    }
    fn visit(index: usize, functions: &[Function], placed: &mut Vec<bool>, order: &mut Vec<usize>) {
        if placed[index] {
            return;
        }
        placed[index] = true;
        let mut called = Vec::new();
        callees(&functions[index].body, &mut called);
        for callee in called {
            visit(callee as usize, functions, placed, order);
        }
        order.push(index);
    }
    fn renumber(body: &mut [Instruction], to: &[u32]) {
        for instruction in body {
            if let Op::Call(callee, _) = &mut instruction.op {
                *callee = to[*callee as usize];
            }
            for block in instruction.op.blocks_mut() {
                renumber(&mut block.body, to);
            }
        }
    }
    let mut placed = vec![false; functions.len()];
    let mut order = Vec::with_capacity(functions.len());
    for index in 0..functions.len() {
        visit(index, &functions, &mut placed, &mut order);
    }
    let mut to = vec![0u32; functions.len()];
    for (new, old) in order.iter().enumerate() {
        to[*old] = new as u32;
    }
    let mut slots: Vec<Option<Function>> = functions.into_iter().map(Some).collect();
    let mut ordered: Vec<Function> = order
        .iter()
        .map(|old| slots[*old].take().expect("each once"))
        .collect();
    for function in &mut ordered {
        renumber(&mut function.body, &to);
    }
    let points = points
        .into_iter()
        .map(|mut point| {
            renumber(&mut point.body, &to);
            point
        })
        .collect();
    (ordered, points)
}

/// The module without the functions its entry points never reach, so an
/// unused document helper neither decides what devices take it nor is
/// emitted. The rest keep their order.
fn prune(mut module: Module) -> Module {
    fn calls(body: &[Instruction], found: &mut Vec<bool>) {
        for instruction in body {
            match &instruction.op {
                Op::Call(callee, _) => found[*callee as usize] = true,
                op => {
                    for block in op.blocks() {
                        calls(&block.body, found);
                    }
                }
            }
        }
    }
    fn renumber(body: &mut [Instruction], to: &[u32]) {
        for instruction in body {
            match &mut instruction.op {
                Op::Call(callee, _) => *callee = to[*callee as usize],
                op => {
                    for block in op.blocks_mut() {
                        renumber(&mut block.body, to);
                    }
                }
            }
        }
    }
    let count = module.functions.len();
    let mut used = vec![false; count];
    calls(&module.entry.body, &mut used);
    for function in module.merge.iter().chain(module.init.iter()) {
        calls(&function.body, &mut used);
    }
    // Callees precede callers, so one backward pass finds every use.
    for index in (0..count).rev() {
        if used[index] {
            let body = module.functions[index].body.clone();
            calls(&body, &mut used);
        }
    }
    if used.iter().all(|&u| u) {
        return module;
    }
    let mut to = vec![u32::MAX; count];
    let mut kept = 0u32;
    for (index, &u) in used.iter().enumerate() {
        if u {
            to[index] = kept;
            kept += 1;
        }
    }
    let functions = std::mem::take(&mut module.functions);
    module.functions = functions
        .into_iter()
        .zip(&used)
        .filter(|(_, u)| **u)
        .map(|(mut function, _)| {
            renumber(&mut function.body, &to);
            function
        })
        .collect();
    renumber(&mut module.entry.body, &to);
    for function in module.merge.iter_mut().chain(module.init.iter_mut()) {
        renumber(&mut function.body, &to);
    }
    module
}

/// How many instructions a body holds, nested blocks counted.
fn size(body: &[Instruction]) -> usize {
    body.iter()
        .map(|instruction| {
            1 + instruction
                .op
                .blocks()
                .iter()
                .map(|block| size(&block.body))
                .sum::<usize>()
        })
        .sum()
}

/// The greatest value id a body defines, nested blocks counted.
fn highest(body: &[Instruction]) -> u32 {
    body.iter()
        .map(|instruction| {
            let nested = instruction
                .op
                .blocks()
                .iter()
                .map(|block| highest(&block.body))
                .max()
                .unwrap_or(0);
            instruction.id.0.max(nested)
        })
        .max()
        .unwrap_or(0)
}

/// `body` with each short `each` builder written out, innermost first:
/// its body once per element under fresh ids, the body's own `Index`
/// that element's position, and the array built from the elements.
fn unrolled(body: &[Instruction], structs: &[StructDef], next: &mut u32) -> Vec<Instruction> {
    let mut out = Vec::with_capacity(body.len());
    for instruction in body {
        let mut instruction = instruction.clone();
        for block in instruction.op.blocks_mut() {
            block.body = unrolled(&block.body, structs, next);
        }
        match &instruction.op {
            Op::Each { count, body }
                if *count as usize * size(&body.body) > UNROLL_LIMIT
                    && counts_at_run_time(&body.body)
                    && carried_loop(&instruction, *count, body, structs, next, &mut out) => {}
            Op::Each { count, body }
                if *count as usize * size(&body.body) <= UNROLL_LIMIT
                    || counts_at_run_time(&body.body) =>
            {
                let mut elements = Vec::with_capacity(*count as usize);
                for position in 0..*count {
                    let mut ids = HashMap::new();
                    out.extend(copied(&body.body, position as i32, true, &mut ids, next));
                    elements.push(*ids.get(&body.result.0).unwrap_or(&body.result));
                }
                out.push(Instruction {
                    id: instruction.id,
                    ty: instruction.ty,
                    op: Op::Construct(elements),
                });
            }
            _ => out.push(instruction),
        }
    }
    out
}

/// Whether a block holds a loop whose count is read at run time. A
/// builder holding one runs as a loop carrying its array
/// ([`carried_loop`]): Chrome's WebGPU computes such a builder wrong as
/// a builder's loop.
fn counts_at_run_time(body: &[Instruction]) -> bool {
    body.iter().any(|instruction| {
        matches!(instruction.op, Op::Loop { bound: Some(_), .. })
            || instruction
                .op
                .blocks()
                .iter()
                .any(|block| counts_at_run_time(&block.body))
    })
}

/// `builder`, an `each` of `count` elements, as a loop carrying its
/// array: the body once a turn, its element taking the turn's place, the
/// rest kept. The enclosing loop's value, which the body read as
/// `Carried`, is read before it. False, writing nothing, for an element
/// with no zero to start the array from.
fn carried_loop(
    builder: &Instruction,
    count: u32,
    body: &Block,
    structs: &[StructDef],
    next: &mut u32,
    out: &mut Vec<Instruction>,
) -> bool {
    let fresh = |next: &mut u32| {
        let id = ValueId(*next);
        *next += 1;
        id
    };
    let Type::Struct(index) = builder.ty else {
        return false;
    };
    let Some(element) = structs.get(index as usize).and_then(|def| def.element) else {
        return false;
    };
    let mut before = Vec::new();
    let mark = *next;
    let Some(zero) = zero_of(element, structs, next, &mut before) else {
        *next = mark;
        return false;
    };
    let init = fresh(next);
    before.push(Instruction {
        id: init,
        ty: builder.ty,
        op: Op::Construct(vec![zero; count as usize]),
    });
    let outer = fresh(next);
    let mut outer_ty = None;
    let mut map = HashMap::new();
    let mut turn = rehoisted(&body.body, true, outer, &mut map, &mut outer_ty);
    if let Some(ty) = outer_ty {
        before.push(Instruction {
            id: outer,
            ty,
            op: Op::Carried,
        });
    }
    let made = *map.get(&body.result.0).unwrap_or(&body.result);
    let index = fresh(next);
    turn.push(Instruction {
        id: index,
        ty: Type::Int,
        op: Op::Index,
    });
    let held = fresh(next);
    turn.push(Instruction {
        id: held,
        ty: builder.ty,
        op: Op::Carried,
    });
    let mut parts = Vec::with_capacity(count as usize);
    for place in 0..count {
        let at = fresh(next);
        turn.push(Instruction {
            id: at,
            ty: Type::Int,
            op: Op::Const(Value::Int(place as i32)),
        });
        let here = fresh(next);
        turn.push(Instruction {
            id: here,
            ty: Type::Bool,
            op: Op::Binary(BinaryOp::Equal, index, at),
        });
        let kept = fresh(next);
        let part = fresh(next);
        turn.push(Instruction {
            id: part,
            ty: element,
            op: Op::If {
                condition: here,
                then: Block {
                    body: Vec::new(),
                    result: made,
                },
                otherwise: Block {
                    body: vec![Instruction {
                        id: kept,
                        ty: element,
                        op: Op::Extract(held, place),
                    }],
                    result: kept,
                },
            },
        });
        parts.push(part);
    }
    let array = fresh(next);
    turn.push(Instruction {
        id: array,
        ty: builder.ty,
        op: Op::Construct(parts),
    });
    out.extend(before);
    out.push(Instruction {
        id: builder.id,
        ty: builder.ty,
        op: Op::Loop {
            count,
            bound: None,
            init,
            until: None,
            body: Block {
                body: turn,
                result: array,
            },
        },
    });
    true
}

/// `body` with its own `Carried` (outside any loop in it) read as
/// `outer`, whose type is recorded; uses of them renamed.
fn rehoisted(
    body: &[Instruction],
    own: bool,
    outer: ValueId,
    map: &mut HashMap<u32, ValueId>,
    outer_ty: &mut Option<Type>,
) -> Vec<Instruction> {
    let mut out = Vec::with_capacity(body.len());
    for instruction in body {
        if own && matches!(instruction.op, Op::Carried) {
            *outer_ty = Some(instruction.ty);
            map.insert(instruction.id.0, outer);
            continue;
        }
        let named = map.clone();
        let renamed = |id: ValueId| *named.get(&id.0).unwrap_or(&id);
        let mut op = instruction.op.clone();
        match &mut op {
            Op::If { condition, .. } => *condition = renamed(*condition),
            Op::Match { scrutinee, .. } => *scrutinee = renamed(*scrutinee),
            Op::Loop { init, bound, .. } => {
                *init = renamed(*init);
                if let Some(bound) = bound {
                    *bound = renamed(*bound);
                }
            }
            other => rename(other, &renamed),
        }
        let inner = own && !matches!(op, Op::Loop { .. });
        for block in op.blocks_mut() {
            block.body = rehoisted(&block.body, inner, outer, map, outer_ty);
            block.result = *map.get(&block.result.0).unwrap_or(&block.result);
        }
        out.push(Instruction {
            id: instruction.id,
            ty: instruction.ty,
            op,
        });
    }
    out
}

/// A value of `ty` with every word zero, made in `out`; `None` for a type
/// with no such literal.
fn zero_of(
    ty: Type,
    structs: &[StructDef],
    next: &mut u32,
    out: &mut Vec<Instruction>,
) -> Option<ValueId> {
    use crate::Scalar::{Bool, Double, Float, Int, Long};
    let value = match ty {
        Type::Scalar(Float) => Value::Float(0.0),
        Type::Scalar(Int) => Value::Int(0),
        Type::Scalar(Bool) => Value::Bool(false),
        Type::Scalar(Double) => Value::Double(0.0),
        Type::Scalar(Long) => Value::Long(0),
        Type::Vector(Float, 2) => Value::Vec2([0.0; 2]),
        Type::Vector(Float, 3) => Value::Vec3([0.0; 3]),
        Type::Vector(Float, 4) => Value::Vec4([0.0; 4]),
        Type::Vector(Int, 2) => Value::IVec2([0; 2]),
        Type::Vector(Int, 3) => Value::IVec3([0; 3]),
        Type::Vector(Int, 4) => Value::IVec4([0; 4]),
        Type::Vector(Double, 2) => Value::DVec2([0.0; 2]),
        Type::Vector(Double, 3) => Value::DVec3([0.0; 3]),
        Type::Vector(Double, 4) => Value::DVec4([0.0; 4]),
        Type::Vector(Long, 2) => Value::LVec2([0; 2]),
        Type::Vector(Long, 3) => Value::LVec3([0; 3]),
        Type::Matrix(Float, 2) => Value::Mat2([0.0; 4]),
        Type::Matrix(Float, 3) => Value::Mat3([0.0; 9]),
        Type::Matrix(Float, 4) => Value::Mat4([0.0; 16]),
        Type::Matrix(Double, 2) => Value::DMat2([0.0; 4]),
        Type::Matrix(Double, 3) => Value::DMat3([0.0; 9]),
        Type::Matrix(Double, 4) => Value::DMat4([0.0; 16]),
        Type::Struct(index) => {
            let def = structs.get(index as usize)?;
            let parts = def
                .fields
                .iter()
                .map(|field| zero_of(field.ty, structs, next, out))
                .collect::<Option<Vec<_>>>()?;
            let id = ValueId(*next);
            *next += 1;
            out.push(Instruction {
                id,
                ty,
                op: Op::Construct(parts),
            });
            return Some(id);
        }
        _ => return None,
    };
    let id = ValueId(*next);
    *next += 1;
    out.push(Instruction {
        id,
        ty,
        op: Op::Const(value),
    });
    Some(id)
}

/// A copy of `body` under fresh ids, recorded in `ids`; where `own`, an
/// `Index` is `position`, and inside a nested loop or builder it is that
/// one's.
fn copied(
    body: &[Instruction],
    position: i32,
    own: bool,
    ids: &mut HashMap<u32, ValueId>,
    next: &mut u32,
) -> Vec<Instruction> {
    let mut copy = Vec::with_capacity(body.len());
    for instruction in body {
        let fresh = ValueId(*next);
        *next += 1;
        let map = |ids: &HashMap<u32, ValueId>, id: ValueId| *ids.get(&id.0).unwrap_or(&id);
        let mut op = match &instruction.op {
            Op::Index if own => Op::Const(Value::Int(position)),
            other => other.clone(),
        };
        match &mut op {
            Op::If { condition, .. } => *condition = map(ids, *condition),
            Op::Match { scrutinee, .. } => *scrutinee = map(ids, *scrutinee),
            Op::Loop { init, bound, .. } => {
                *init = map(ids, *init);
                if let Some(bound) = bound {
                    *bound = map(ids, *bound);
                }
            }
            other => rename(other, &|id| map(ids, id)),
        }
        let inner = own && !matches!(op, Op::Loop { .. } | Op::Each { .. });
        for block in op.blocks_mut() {
            block.body = copied(&block.body, position, inner, ids, next);
            block.result = map(ids, block.result);
        }
        ids.insert(instruction.id.0, fresh);
        copy.push(Instruction {
            id: fresh,
            ty: instruction.ty,
            op,
        });
    }
    copy
}

/// `function` with its calls to short functions written out: callee
/// values take fresh ids, `Arg(i)` reads become the call's operands, and
/// the return value takes the call's place.
/// Whether a call to `callee` is written out where it is made: it gives
/// one value and is short, or takes a wide argument and is not long.
fn writes_out(callee: &Function, structs: &[StructDef]) -> bool {
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
}

fn inline_calls(function: &Function, functions: &[Function], inlinable: &[bool]) -> Function {
    if !inlinable.iter().any(|yes| *yes) {
        return function.clone();
    }
    let mut pass = Inline {
        functions,
        inlinable,
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
    /// Values standing for others: a call's value for what the body
    /// returned, an argument read for its operand.
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
                Op::Each { count, body } => Op::Each {
                    count: *count,
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

    /// A callee value as the caller names it: an argument read is its
    /// operand, anything else its fresh id.
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
                Op::Each { body, .. } => *body = self.written_block(body, operands, ids),
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
        let nested = instruction
            .op
            .blocks()
            .into_iter()
            .find_map(|block| find_op(&block.body, id));
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

/// Where a part of one input element comes from: the input, the address,
/// the element's type and the path from the whole.
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
        // Short builders are written out first, so their positions fold.
        let mut next = highest(&function.body) + 1;
        let body = unrolled(&function.body, self.structs, &mut next);
        // Short-circuits fold first, so a loop may lift what they compute.
        let body = self.folded(&body);
        let body = self.hoisted(&body);
        let (body, narrowed) = self.narrowed(&body);
        // A narrowed-away read loses its binding name, so a trace shows
        // only work the Kernel does.
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

    /// The body reading only the words it uses: a part taken out of a
    /// whole input element becomes a read of that part alone. The whole
    /// read stays while anything wants it; the sweep drops it otherwise.
    /// Memories are left alone, since a `put` would split part from copy.
    /// Returns the body and the whole reads left unused.
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
                // Only a path naming this very value is rewritten.
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
                Op::Each { count, body } => Op::Each {
                    count: *count,
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

    /// Whether a read of this input is worth narrowing: a record or array
    /// element outside the memories.
    fn narrowable(&self, input: u32, element: Type) -> bool {
        matches!(element, Type::Struct(_))
            && !self.memories.get(input as usize).copied().unwrap_or(true)
    }

    /// The body with every short-circuit whose other side is free and
    /// cannot fail written as one operation (`a && b` evaluates `b` always).
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
                        Op::If { .. }
                            | Op::Match { .. }
                            | Op::Loop { .. }
                            | Op::Each { .. }
                            | Op::Call(..)
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
                Op::Each { count, body } => Op::Each {
                    count: *count,
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

    /// The body with each loop's invariant instructions (all operands from
    /// outside the loop) lifted before it. Only operations that cannot fail
    /// move, so a failing run reports the same failure.
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
                    // Innermost first, so lifted work can leave outer loops.
                    let inner = self.hoisted_block(body);
                    let mut kept = Vec::with_capacity(inner.body.len());
                    // The loop's values are inside it until lifted out.
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
                                Op::If { .. }
                                    | Op::Match { .. }
                                    | Op::Loop { .. }
                                    | Op::Each { .. }
                                    | Op::Call(..)
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
                Op::Each { count, body } => {
                    // As a loop's: what is the same for every element is
                    // made once, before it.
                    let inner = self.hoisted_block(body);
                    let mut kept = Vec::with_capacity(inner.body.len());
                    let mut within: HashSet<u32> = defined(&inner.body);
                    for instruction in inner.body {
                        let liftable = !self.has_effects(&instruction.op)
                            && self.cse_safe(&instruction.op)
                            && !can_fail(instruction.ty, &instruction.op)
                            && !matches!(
                                instruction.op,
                                Op::If { .. }
                                    | Op::Match { .. }
                                    | Op::Loop { .. }
                                    | Op::Each { .. }
                                    | Op::Call(..)
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
                    Op::Each {
                        count: *count,
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
            Op::Put(..) | Op::Emit(_) | Op::Spend(_) | Op::Look => true,
            Op::Call(callee, _) => self.writers.get(*callee as usize).copied().unwrap_or(false),
            other => other
                .blocks()
                .into_iter()
                .any(|block| self.block_has_effects(block)),
        }
    }

    fn block_has_effects(&self, block: &Block) -> bool {
        block.body.iter().any(|i| self.has_effects(&i.op))
    }

    /// Whether two instances of an operation on the same operands give the
    /// same value: pure operations outside blocks, loop state and memories.
    fn cse_safe(&self, op: &Op) -> bool {
        match op {
            Op::Put(..) | Op::Emit(_) | Op::Spend(_) | Op::Look | Op::Carried | Op::Index => false,
            Op::If { .. } | Op::Match { .. } | Op::Loop { .. } | Op::Each { .. } => false,
            Op::Call(callee, _) => !self.writers.get(*callee as usize).copied().unwrap_or(false),
            Op::Sample(input)
            | Op::At(input, _)
            | Op::AtLinear(input, _)
            | Op::Part { input, .. } => {
                !self.memories.get(*input as usize).copied().unwrap_or(true)
            }
            _ => true,
        }
    }

    /// Drops unneeded instructions of a block, marking what the kept ones
    /// need; visited last to first, so a use precedes its definition.
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
                Op::Each { count, body } => Op::Each {
                    count,
                    body: self.sweep_block(body, used),
                },
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

/// Every value id a body defines, nested blocks counted.
fn defined(body: &[Instruction]) -> HashSet<u32> {
    let mut ids = HashSet::new();
    fn walk(body: &[Instruction], ids: &mut HashSet<u32>) {
        for instruction in body {
            ids.insert(instruction.id.0);
            for block in instruction.op.blocks() {
                walk(&block.body, ids);
            }
        }
    }
    walk(body, &mut ids);
    ids
}

/// How many operations one compile may run while folding a function's
/// loops.
const LOOP_BUDGET: u32 = 20_000;

/// Whether an operation can fail: integer arithmetic overflows and
/// divides by zero. Floats never fail; a Field read is clamped.
fn can_fail(ty: Type, op: &Op) -> bool {
    ty.is_int_kind() && matches!(op, Op::Binary(..) | Op::Intrinsic(..))
}

/// The values an operation reads, blocks aside.
fn operands(op: &Op) -> Vec<ValueId> {
    match op {
        Op::Unary(_, a) => vec![*a],
        Op::Binary(_, a, b) => vec![*a, *b],
        Op::Construct(parts) => parts.clone(),
        Op::Extract(source, _) => vec![*source],
        Op::Subscript(array, index) => vec![*array, *index],
        Op::Call(_, args) | Op::Intrinsic(_, args) => args.clone(),
        Op::At(_, at)
        | Op::AtLinear(_, at)
        | Op::Key(at)
        | Op::Typed(at)
        | Op::Button(at)
        | Op::Stick(at) => {
            vec![*at]
        }
        Op::Part { at, path, .. } => at
            .iter()
            .copied()
            .chain(path.iter().filter_map(|step| match step {
                Step::Index(index) => Some(*index),
                Step::Field(_) => None,
            }))
            .collect(),
        Op::Put(_, address, value) => vec![*address, *value],
        Op::Spend(counted) | Op::Emit(counted) => vec![*counted],
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
    /// Each value's type, and its operation when a rewrite reads it (a
    /// literal, a construction, an extraction); a nested block's
    /// operation is not copied.
    defs: HashMap<u32, (Type, Option<Op>)>,
    /// The values available for sharing, one table per open block.
    scopes: Vec<HashMap<String, u32>>,
}

impl Simplify<'_> {
    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.alias.get(&id.0).unwrap_or(&id.0))
    }

    /// An `if`, simplified, and written as one `and` or `or` when it is a
    /// short-circuit whose other side is pure and cannot fail: that side
    /// is emitted before it.
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
                        Op::If { .. }
                            | Op::Match { .. }
                            | Op::Loop { .. }
                            | Op::Each { .. }
                            | Op::Call(..)
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
            // A branch on a literal is the block it takes, which joins
            // this one: only that block would have run.
            if let Some(taken) = self.taken(&instruction.op) {
                let taken = self.nested(taken);
                out.extend(taken.body);
                self.alias.insert(instruction.id.0, taken.result.0);
                continue;
            }
            if let Op::If {
                condition,
                then,
                otherwise,
            } = &instruction.op
            {
                let op = self.folded_if(*condition, then, otherwise, &mut out);
                self.defs.insert(instruction.id.0, (ty, read_later(&op)));
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
                    // A pure loop over a known count and initial value
                    // runs here, once.
                    match self.run_loop(&loop_op) {
                        Some(value) => Op::Const(value),
                        None => loop_op,
                    }
                }
                Op::Each { count, body } => Op::Each {
                    count: *count,
                    body: self.nested(body),
                },
                other => {
                    let mut op = other.clone();
                    rename(&mut op, &|id| self.map(id));
                    op
                }
            };
            // A part straight out of its construction is that part.
            if let Op::Extract(source, k) = &op
                && let Some((_, Some(Op::Construct(parts)))) = self.defs.get(&source.0)
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
            self.defs.insert(instruction.id.0, (ty, read_later(&op)));
            out.push(Instruction {
                id: instruction.id,
                ty,
                op,
            });
        }
        self.scopes.pop();
        out
    }

    /// The block an `if` or a `match` takes when what it tests is a
    /// literal.
    fn taken<'op>(&self, op: &'op Op) -> Option<&'op Block> {
        let literal = |id: &ValueId| match self.defs.get(&self.map(*id).0) {
            Some((_, Some(Op::Const(value)))) => Some(value.clone()),
            _ => None,
        };
        match op {
            Op::If {
                condition,
                then,
                otherwise,
            } => match literal(condition)? {
                Value::Bool(true) => Some(then),
                Value::Bool(false) => Some(otherwise),
                _ => None,
            },
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => match literal(scrutinee)? {
                Value::Int(value) => Some(
                    arms.iter()
                        .find(|(values, _)| values.contains(&value))
                        .map_or(otherwise, |(_, block)| block),
                ),
                _ => None,
            },
            _ => None,
        }
    }

    fn nested(&mut self, block: &Block) -> Block {
        let body = self.block(&block.body);
        Block {
            body,
            result: self.map(block.result),
        }
    }

    /// The vector, matrix, array or struct these parts are all of, in
    /// order, if any.
    fn rebuilt(&self, parts: &[ValueId], ty: Type) -> Option<u32> {
        let (Some(Op::Extract(first, 0)), _) = (&self.defs.get(&parts[0].0)?.1, ()) else {
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
            matches!(self.defs.get(&part.0), Some((_, Some(Op::Extract(source, at)))) if source.0 == whole && *at as usize == k)
        });
        all.then_some(whole)
    }

    /// A loop with a known count, a literal initial value and a pure,
    /// infallible body without Field reads or calls, run to a literal
    /// within [`LOOP_BUDGET`].
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
                && self.run_block(until, &carried, index, &mut env, &mut budget)?
                    == Value::Bool(true)
            {
                return Some(carried);
            }
            carried = self.run_block(body, &carried, index, &mut env, &mut budget)?;
        }
        Some(carried)
    }

    /// Runs a block of the loop: every instruction in order, then its yield.
    fn run_block(
        &self,
        block: &Block,
        carried: &Value,
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
        carried: &Value,
        index: u32,
        env: &mut HashMap<u32, Value>,
        budget: &mut u32,
    ) -> Option<Value> {
        match op {
            Op::Const(value) => Some(value.clone()),
            Op::Carried => Some(carried.clone()),
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
            // An inner loop has its own carried value and index.
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
                        && self.run_block(until, &inner, step, env, budget)? == Value::Bool(true)
                    {
                        return Some(inner);
                    }
                    inner = self.run_block(body, &inner, step, env, budget)?;
                }
                Some(inner)
            }
            _ => None,
        }
    }

    /// A value during the run: computed by it, or a literal from outside.
    fn known(&self, id: ValueId, env: &HashMap<u32, Value>) -> Option<Value> {
        if let Some(value) = env.get(&id.0) {
            return Some(value.clone());
        }
        match self.defs.get(&id.0) {
            Some((_, Some(Op::Const(value)))) => Some(value.clone()),
            _ => None,
        }
    }

    /// The literal an operation on literals gives, when it cannot fail.
    fn fold(&self, op: &Op, ty: Type) -> Option<Value> {
        if !matches!(
            ty,
            Type::Float | Type::Int | Type::Bool | Type::Vec2 | Type::Vec3 | Type::Vec4
        ) {
            return None;
        }
        let literal = |id: &ValueId| match self.defs.get(&id.0) {
            Some((_, Some(Op::Const(value)))) => Some(value.clone()),
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
        Op::At(_, at)
        | Op::AtLinear(_, at)
        | Op::Key(at)
        | Op::Typed(at)
        | Op::Button(at)
        | Op::Stick(at) => *at = map(*at),
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
        Op::Spend(counted) | Op::Emit(counted) => *counted = map(*counted),
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
        (Floor, [a]) => map(a.clone(), f32::floor),
        (Ceil, [a]) => map(a.clone(), f32::ceil),
        (Round, [a]) => map(a.clone(), f32::round_ties_even),
        (Trunc, [a]) => map(a.clone(), f32::trunc),
        (Fract, [a]) => map(a.clone(), |v| v - v.floor()),
        (Abs, [Value::Int(v)]) => Some(Value::Int(v.wrapping_abs())),
        (Abs, [a]) => map(a.clone(), f32::abs),
        (Sign, [Value::Int(v)]) => Some(Value::Int(v.signum())),
        (Sign, [a]) => map(a.clone(), sign),
        (Min, [Value::Int(x), Value::Int(y)]) => Some(Value::Int((*x).min(*y))),
        (Max, [Value::Int(x), Value::Int(y)]) => Some(Value::Int((*x).max(*y))),
        (Min, [a, b]) => zip(a.clone(), b.clone(), f32::min),
        (Max, [a, b]) => zip(a.clone(), b.clone(), f32::max),
        (Clamp, [Value::Int(v), Value::Int(low), Value::Int(high)]) => {
            Some(Value::Int((*v).max(*low).min(*high)))
        }
        (Clamp, [v, low, high]) => {
            let floored = zip(v.clone(), low.clone(), f32::max)?;
            zip(floored, high.clone(), f32::min)
        }
        (ToFloat, [Value::Int(v)]) => Some(Value::Float(*v as f32)),
        (ToFloat, [Value::Bool(v)]) => Some(Value::Float(if *v { 1.0 } else { 0.0 })),
        (ToInt, [Value::Float(v)]) => Some(Value::Int(*v as i32)),
        (ToInt, [Value::Bool(v)]) => Some(Value::Int(i32::from(*v))),
        _ => None,
    }
}

/// A function numbered again in pre-order, its operands following, as
/// [`validate`](crate::validate) asks: for a pass that adds instructions or
/// takes them away.
pub fn renumbered(function: Function) -> Function {
    renumbered_from(function).0
}

/// [`renumbered`], and each kept instruction's old id with its new one.
pub(crate) fn renumbered_from(function: Function) -> (Function, BTreeMap<u32, u32>) {
    let mut renumber = Renumber {
        ids: BTreeMap::new(),
        next: 0,
    };
    let body = renumber.block(&function.body);
    let function = Function {
        name: function.name,
        args: function.args,
        body,
        results: function
            .results
            .iter()
            .map(|id| renumber.map(*id))
            .collect(),
        bindings: function
            .bindings
            .into_iter()
            .map(|(id, name)| (renumber.map(id), name))
            .collect(),
    };
    (function, renumber.ids)
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
                Op::Each { count, body } => Op::Each {
                    count: *count,
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

/// The operation a later rewrite may read: a literal, a construction or an
/// extraction, which hold no block.
fn read_later(op: &Op) -> Option<Op> {
    matches!(op, Op::Const(_) | Op::Construct(_) | Op::Extract(..)).then(|| op.clone())
}
