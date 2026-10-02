//! The most stack a module's compiled code can take. The code runs on the
//! caller's thread and on threads of the default size, where a frame
//! larger than the guard page steps past it into other memory, so a
//! module whose bound passes [`MAX_FRAME_BYTES`] is left to the
//! interpreter, which keeps its values on the heap.

use orla_ir::{Function, Instruction, Module, Op, Type, type_name, words};

/// Stack the compiled functions of one module may take together: a
/// small part of the smallest thread stack the code runs on.
pub const MAX_FRAME_BYTES: usize = 256 * 1024;

/// An upper bound on the bytes the frames of `module`'s functions take
/// together, as deep as calls nest: the deepest chain of calls from the
/// entry, the merge or the init, each frame counting each value's words
/// twice (held and spilled) and the words of each array subscripted from
/// a stack slot. A function calls only those before it, so no chain
/// returns to a function already on it.
pub fn bound(module: &Module) -> usize {
    deepest(module)
        .iter()
        .map(|frame| frame.words)
        .sum::<usize>()
        * 4
}

/// Why `module` is left to the interpreter, when its frames may pass the
/// bound: the bytes, and the deepest chain of calls, each frame with the
/// type whose values take the most of it.
pub fn refusal(module: &Module) -> Option<String> {
    let bytes = bound(module);
    (bytes > MAX_FRAME_BYTES).then(|| {
        let chain = deepest(module)
            .iter()
            .map(|frame| {
                let (ty, count, of_type) = frame.largest;
                format!(
                    "`{}` {} bytes, {} of them {count} values of `{}`",
                    frame.name,
                    frame.words * 4,
                    of_type * 4,
                    type_name(ty, &module.structs)
                )
            })
            .collect::<Vec<_>>()
            .join(", calling ");
        format!(
            "its values may take {bytes} bytes of stack; compiled code takes at most {MAX_FRAME_BYTES}; the deepest calls: {chain}"
        )
    })
}

/// One function's frame: its words, the functions it calls, and the
/// type whose values take the most of its words (the type, how many
/// values, their words).
struct Frame<'a> {
    name: &'a str,
    words: usize,
    calls: Vec<usize>,
    largest: (Type, usize, usize),
}

/// The frames on the deepest chain of calls from the entry, the merge or
/// the init, outermost first.
fn deepest(module: &Module) -> Vec<Frame<'_>> {
    let functions: Vec<Frame<'_>> = module
        .functions
        .iter()
        .map(|function| frame(module, function))
        .collect();
    // The deepest chain from each function, in words, and the callee it
    // goes on through; a function calls only those before it.
    let mut depth: Vec<(usize, Option<usize>)> = Vec::with_capacity(functions.len());
    let below = |frame: &Frame<'_>, depth: &[(usize, Option<usize>)]| {
        frame
            .calls
            .iter()
            .filter_map(|&callee| depth.get(callee).map(|(words, _)| (*words, callee)))
            .max()
    };
    for function in &functions {
        let next = below(function, &depth);
        depth.push((
            function.words + next.map_or(0, |(words, _)| words),
            next.map(|(_, callee)| callee),
        ));
    }
    let roots = Some(&module.entry)
        .into_iter()
        .chain(&module.merge)
        .chain(&module.init)
        .map(|root| frame(module, root));
    let Some(root) =
        roots.max_by_key(|root| root.words + below(root, &depth).map_or(0, |(words, _)| words))
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
        words,
        calls,
        largest,
    }
}
