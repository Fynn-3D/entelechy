//! The most stack a module's compiled code can take. The code runs on the
//! caller's thread and on threads of the default size, where a frame
//! larger than the guard page steps past it into other memory, so a
//! module whose frames pass [`MAX_FRAME_BYTES`] is left to the
//! interpreter, which keeps its values on the heap.
//!
//! The stack a module takes is the deepest chain of calls from its entry,
//! its merge or its init, each frame as compiled ([`Measured`]). Before
//! compiling there is only an estimate from the IR ([`bound`]), an upper
//! bound: it counts each value's words, though a struct built from
//! another shares the other's words in the compiled code.

use orla_ir::{Function, Instruction, Module, Op, Type, type_name, words};

/// Stack the compiled functions of one module may take together: a
/// small part of the smallest thread stack the code runs on.
pub const MAX_FRAME_BYTES: usize = 256 * 1024;

/// The bytes of stack each function of a module takes as compiled: its
/// functions by index, its entry, its init and its merge (0 when it has
/// none).
#[derive(Clone, Debug)]
pub struct Measured {
    pub functions: Vec<usize>,
    pub entry: usize,
    pub init: usize,
    pub merge: usize,
}

/// An upper bound on the bytes the frames of `module`'s functions take
/// together, as deep as calls nest: the deepest chain of calls from the
/// entry, the merge or the init, each frame counting each value's words
/// twice (held and spilled) and the words of each array subscripted from
/// a stack slot. A function calls only those before it, so no chain
/// returns to a function already on it.
#[cfg(not(target_arch = "wasm32"))]
pub fn bound(module: &Module) -> usize {
    deepest(module, None).iter().map(|frame| frame.bytes).sum()
}

/// Why `module` is left to the interpreter, when its frames pass what
/// compiled code may take: the bytes, and the deepest chain of calls,
/// each frame with the type whose values are the most of it. With the
/// frames `measured` as compiled, they decide; without, the estimate.
pub fn refusal(module: &Module, measured: Option<&Measured>) -> Option<String> {
    let chain = deepest(module, measured);
    let bytes: usize = chain.iter().map(|frame| frame.bytes).sum();
    (bytes > MAX_FRAME_BYTES).then(|| {
        let calls = chain
            .iter()
            .map(|frame| {
                let (ty, count, _) = frame.largest;
                format!(
                    "`{}` {} bytes, the most of its values {count} of `{}`",
                    frame.name,
                    frame.bytes,
                    type_name(ty, &module.structs)
                )
            })
            .collect::<Vec<_>>()
            .join(", calling ");
        let taken = if measured.is_some() {
            "its frames take"
        } else {
            "its values may take"
        };
        format!(
            "{taken} {bytes} bytes of stack; compiled code takes at most {MAX_FRAME_BYTES}; the deepest calls: {calls}"
        )
    })
}

/// One function's frame: its bytes, the functions it calls, and the
/// type whose values take the most of its words (the type, how many
/// values, their words).
struct Frame<'a> {
    name: &'a str,
    bytes: usize,
    calls: Vec<usize>,
    largest: (Type, usize, usize),
}

/// The frames on the deepest chain of calls from the entry, the merge or
/// the init, outermost first: each as `measured`, or as estimated.
fn deepest<'a>(module: &'a Module, measured: Option<&Measured>) -> Vec<Frame<'a>> {
    let sized = |mut frame: Frame<'a>, bytes: Option<usize>| {
        if let Some(bytes) = bytes {
            frame.bytes = bytes;
        }
        frame
    };
    let functions: Vec<Frame<'_>> = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| {
            sized(
                frame(module, function),
                measured.and_then(|m| m.functions.get(index).copied()),
            )
        })
        .collect();
    // The deepest chain from each function, in bytes, and the callee it
    // goes on through; a function calls only those before it.
    let mut depth: Vec<(usize, Option<usize>)> = Vec::with_capacity(functions.len());
    let below = |frame: &Frame<'_>, depth: &[(usize, Option<usize>)]| {
        frame
            .calls
            .iter()
            .filter_map(|&callee| depth.get(callee).map(|(bytes, _)| (*bytes, callee)))
            .max()
    };
    for function in &functions {
        let next = below(function, &depth);
        depth.push((
            function.bytes + next.map_or(0, |(bytes, _)| bytes),
            next.map(|(_, callee)| callee),
        ));
    }
    let roots = [
        Some((&module.entry, measured.map(|m| m.entry))),
        module
            .merge
            .as_ref()
            .map(|merge| (merge, measured.map(|m| m.merge))),
        module
            .init
            .as_ref()
            .map(|init| (init, measured.map(|m| m.init))),
    ]
    .into_iter()
    .flatten()
    .map(|(root, bytes)| sized(frame(module, root), bytes));
    let Some(root) =
        roots.max_by_key(|root| root.bytes + below(root, &depth).map_or(0, |(bytes, _)| bytes))
    else {
        return Vec::new();
    };
    let mut next = below(&root, &depth).map(|(_, callee)| callee);
    let mut chain = vec![root];
    let mut functions: Vec<Option<Frame<'_>>> = functions.into_iter().map(Some).collect();
    while let Some(callee) = next {
        next = depth[callee].1;
        if let Some(frame) = functions[callee].take() {
            chain.push(frame);
        }
    }
    chain
}

fn frame<'a>(module: &Module, function: &'a Function) -> Frame<'a> {
    // Each value's words twice (held and spilled), and the words of each
    // array subscripted from a stack slot; by type, how many values and
    // their words; and the functions called.
    fn walk(
        module: &Module,
        body: &[Instruction],
        types: &mut Vec<Type>,
        tally: &mut Vec<(Type, usize, usize)>,
        calls: &mut Vec<usize>,
    ) -> usize {
        let mut total = 0;
        for instruction in body {
            types.push(instruction.ty);
            let held = 2 * words(&module.structs, instruction.ty);
            total += held;
            match tally.iter_mut().find(|(ty, ..)| *ty == instruction.ty) {
                Some(entry) => {
                    entry.1 += 1;
                    entry.2 += held;
                }
                None => tally.push((instruction.ty, 1, held)),
            }
            match &instruction.op {
                Op::Subscript(array, _) => {
                    if let Some(ty) = types.get(array.0 as usize) {
                        total += words(&module.structs, *ty);
                    }
                }
                Op::Call(callee, _) => {
                    if !calls.contains(&(*callee as usize)) {
                        calls.push(*callee as usize);
                    }
                }
                _ => {}
            }
            for block in instruction.op.blocks() {
                total += walk(module, &block.body, types, tally, calls);
            }
        }
        total
    }
    let mut tally = Vec::new();
    let mut calls = Vec::new();
    let words = walk(
        module,
        &function.body,
        &mut Vec::new(),
        &mut tally,
        &mut calls,
    );
    let largest = tally
        .into_iter()
        .max_by_key(|(_, _, of_type)| *of_type)
        .unwrap_or((Type::Int, 0, 0));
    Frame {
        name: &function.name,
        bytes: words * 4,
        calls,
        largest,
    }
}
