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
/// together, as deep as calls nest: each value's words twice (held and
/// spilled) and the words of each array subscripted from a stack slot.
pub fn bound(module: &Module) -> usize {
    frames(module)
        .iter()
        .map(|frame| frame.words)
        .sum::<usize>()
        * 4
}

/// Why `module` is left to the interpreter, when its frames may pass the
/// bound: the bytes, and the functions taking the most, each with the
/// type whose values take the most of it.
pub fn refusal(module: &Module) -> Option<String> {
    let bytes = bound(module);
    (bytes > MAX_FRAME_BYTES).then(|| {
        let mut frames = frames(module);
        frames.sort_by(|a, b| b.words.cmp(&a.words));
        let most = frames
            .iter()
            .take(3)
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
            .join("; ");
        format!(
            "its values may take {bytes} bytes of stack; compiled code takes at most {MAX_FRAME_BYTES}; the most: {most}"
        )
    })
}

/// One function's frame: its words, and the type whose values take the
/// most of them (the type, how many values, their words).
struct Frame<'a> {
    name: &'a str,
    words: usize,
    largest: (Type, usize, usize),
}

/// The frame of each of `module`'s functions, the entry, merge and init
/// included.
fn frames(module: &Module) -> Vec<Frame<'_>> {
    module
        .functions
        .iter()
        .chain(Some(&module.entry))
        .chain(&module.merge)
        .chain(&module.init)
        .map(|function| frame(module, function))
        .collect()
}

fn frame<'a>(module: &Module, function: &'a Function) -> Frame<'a> {
    // Each value's words twice (held and spilled), and the words of each
    // array subscripted from a stack slot; by type, how many values and
    // their words.
    fn walk(
        module: &Module,
        body: &[Instruction],
        types: &mut Vec<Type>,
        tally: &mut Vec<(Type, usize, usize)>,
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
            if let Op::Subscript(array, _) = instruction.op
                && let Some(ty) = types.get(array.0 as usize)
            {
                total += words(&module.structs, *ty);
            }
            for block in instruction.op.blocks() {
                total += walk(module, &block.body, types, tally);
            }
        }
        total
    }
    let mut tally = Vec::new();
    let words = walk(module, &function.body, &mut Vec::new(), &mut tally);
    let largest = tally
        .into_iter()
        .max_by_key(|(_, _, of_type)| *of_type)
        .unwrap_or((Type::Int, 0, 0));
    Frame {
        name: &function.name,
        words,
        largest,
    }
}
