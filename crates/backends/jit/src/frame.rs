//! The most stack a module's compiled code can take. The code runs on the
//! caller's thread and on threads of the default size, where a frame
//! larger than the guard page steps past it into other memory, so a
//! module whose bound passes [`MAX_FRAME_BYTES`] is left to the
//! interpreter, which keeps its values on the heap.

use orla_ir::{Function, Instruction, Module, Op, Type, words};

/// Stack the compiled functions of one module may take together: a
/// small part of the smallest thread stack the code runs on.
pub const MAX_FRAME_BYTES: usize = 256 * 1024;

/// An upper bound on the bytes the frames of `module`'s functions take
/// together, as deep as calls nest: each value's words twice (held and
/// spilled) and the words of each array subscripted from a stack slot.
pub fn bound(module: &Module) -> usize {
    let functions = module
        .functions
        .iter()
        .chain(Some(&module.entry))
        .chain(&module.merge)
        .chain(&module.init);
    functions
        .map(|function| frame_words(module, function))
        .sum::<usize>()
        * 4
}

/// Why `module` is left to the interpreter, when its frames may pass the
/// bound.
pub fn refusal(module: &Module) -> Option<String> {
    let bytes = bound(module);
    (bytes > MAX_FRAME_BYTES).then(|| {
        format!(
            "its values may take {bytes} bytes of stack; compiled code takes at most {MAX_FRAME_BYTES}"
        )
    })
}

fn frame_words(module: &Module, function: &Function) -> usize {
    fn walk(module: &Module, body: &[Instruction], types: &mut Vec<Type>) -> usize {
        let mut total = 0;
        for instruction in body {
            types.push(instruction.ty);
            total += 2 * words(&module.structs, instruction.ty);
            if let Op::Subscript(array, _) = instruction.op
                && let Some(ty) = types.get(array.0 as usize)
            {
                total += words(&module.structs, *ty);
            }
            for block in instruction.op.blocks() {
                total += walk(module, &block.body, types);
            }
        }
        total
    }
    walk(module, &function.body, &mut Vec::new())
}
