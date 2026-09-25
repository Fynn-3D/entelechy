//! How a value of an IR type lies in memory for the JIT's code, native
//! or WebAssembly alike: its scalars in layout order, each a lane of one
//! width. A 32-bit scalar is one word; a `Double` or a `Long` is one
//! 64-bit value that memory keeps as two words, the low one first
//! (little-endian, as every target the JIT builds for is). Each code
//! generator names a lane in its own machine's types.

use orla_abi::Type;
use orla_ir::Module;

/// One scalar of a value, by what holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lane {
    F32,
    F64,
    I32,
    I64,
}

impl Lane {
    /// Bytes it takes in memory.
    pub(crate) fn bytes(self) -> u32 {
        match self {
            Lane::F32 | Lane::I32 => 4,
            Lane::F64 | Lane::I64 => 8,
        }
    }
}

/// The lanes of a value of `ty`, in layout order.
pub(crate) fn lanes(module: &Module, ty: Type) -> Vec<Lane> {
    match ty {
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .flat_map(|field| lanes(module, field.ty))
            .collect(),
        other => {
            let (scalar, n) = other.shape().unwrap_or((Type::Float, 1));
            let lane = match scalar {
                Type::Float => Lane::F32,
                Type::Double => Lane::F64,
                Type::Long => Lane::I64,
                _ => Lane::I32,
            };
            vec![lane; usize::from(n)]
        }
    }
}

/// A scalar's width in memory, however a code generator names it.
pub(crate) trait Width: Copy {
    /// Bytes it takes.
    fn width(self) -> u32;
}

impl Width for Lane {
    fn width(self) -> u32 {
        self.bytes()
    }
}

/// Where each of `words` begins in memory, in bytes.
pub(crate) fn byte_offsets<W: Width>(words: &[W]) -> Vec<u32> {
    let mut at = 0;
    words
        .iter()
        .map(|word| {
            let here = at;
            at += word.width();
            here
        })
        .collect()
}

/// Bytes `words` take in memory.
pub(crate) fn byte_size<W: Width>(words: &[W]) -> u32 {
    words.iter().map(|word| word.width()).sum()
}

/// Whether a value of `ty` holds a 64-bit scalar anywhere.
pub(crate) fn holds_wide(module: &Module, ty: Type) -> bool {
    lanes(module, ty).iter().any(|lane| lane.bytes() == 8)
}

/// The lane at which field `field` of struct `index` begins.
pub(crate) fn field_offset(module: &Module, index: u32, field: usize) -> usize {
    module.structs[index as usize].fields[..field]
        .iter()
        .map(|f| lanes(module, f.ty).len())
        .sum()
}
