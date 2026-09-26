//! The JIT on WebAssembly: a Kernel's IR written as a module
//! ([`translate`]) that the host loads into Orla's function table
//! (`orla.load`, imported under `wasm-host`). The module imports Orla's
//! memory and table, so it reads Fields in place and calls helpers by
//! their table index. The returned index is a function pointer, called
//! like native code.

mod encode;
mod translate;

use std::any::Any;
use std::collections::HashMap;

use orla_ir::{Mode, Validated};
use orla_rules::Prepared;

use encode::{Module, Val};
use translate::{Fx, Kind, Shared, signature};

#[cfg(feature = "wasm-host")]
#[link(wasm_import_module = "orla")]
unsafe extern "C" {
    /// Compiles `len` bytes of a module, instantiates it over Orla's
    /// memory and table, puts the functions it exports as "0", "1", ...
    /// into the table, writes their indices to `slots` (at most `most`),
    /// and answers how many; below zero when the host refuses the module.
    fn load(bytes: *const u8, len: usize, slots: *mut u32, most: u32) -> i32;
    /// Gives the table's slots back.
    fn unload(slots: *const u32, count: u32);
}

#[cfg(not(feature = "wasm-host"))]
unsafe fn load(_: *const u8, _: usize, _: *mut u32, _: u32) -> i32 {
    -1
}

#[cfg(not(feature = "wasm-host"))]
unsafe fn unload(_: *const u32, _: u32) {}

/// A loaded module: the entry, a scan's `init` and a reduction's
/// `merge` as table indices, and the slot and constant-table memory,
/// which lives as long as the code.
pub struct Compiled {
    pub entry: usize,
    pub init: Option<usize>,
    pub merge: Option<usize>,
    slots: Vec<u32>,
    _frame: Vec<u64>,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // The runtime drops prepared code after its last execution.
        unsafe { unload(self.slots.as_ptr(), self.slots.len() as u32) };
    }
}

impl Prepared for Compiled {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Compiles the module and has the host load it.
pub fn compile(module: &Validated) -> Result<Compiled, String> {
    for port in module
        .manifest
        .inputs
        .iter()
        .chain(&module.manifest.outputs)
    {
        if port.ty.schema.uniform_encoding().is_none() && !port.ty.schema.is_wide() {
            return Err(format!(
                "`{}` keeps its scalars in more than one encoding ({}); a port keeps them one way",
                port.name, port.ty.schema
            ));
        }
    }
    // A pointer to a helper is its index in Orla's table.
    let helpers: HashMap<&'static str, u32> = crate::symbols()
        .into_iter()
        .map(|(name, address)| (name, address as usize as u32))
        .collect();
    let pair = match module.manifest.mode {
        Mode::Scatter { base, .. } => {
            module.manifest.inputs[base as usize]
                .ty
                .schema
                .value_words()
                + 3
        }
        _ => 0,
    };
    let max_stride = module
        .manifest
        .inputs
        .iter()
        .map(|port| port.ty.schema.value_words())
        .max()
        .unwrap_or(1)
        .max(pair)
        .max(1) as u32;
    let mut shared = Shared::default();
    let mut functions = Vec::new();
    // The module's functions first, so a call's index is the IR's.
    for function in &module.functions {
        let sig = signature(module, function, Kind::Function);
        let params = sig.params.clone();
        let ty = shared.type_index(sig);
        let body = Fx::new(module, &mut shared, &helpers, params, max_stride)
            .function(function, Kind::Function);
        functions.push((ty, body));
    }
    let scan = module.manifest.mode == Mode::Scan;
    let mut exports = Vec::new();
    let entry_kind = if scan { Kind::Scan } else { Kind::Map };
    let sig = signature(module, &module.entry, entry_kind);
    let params = sig.params.clone();
    let ty = shared.type_index(sig);
    let body = Fx::new(module, &mut shared, &helpers, params, max_stride)
        .function(&module.entry, entry_kind);
    exports.push(("0".to_owned(), functions.len() as u32));
    functions.push((ty, body));
    let mut has_init = false;
    if let (true, Some(init)) = (scan, &module.init) {
        let sig = encode::Signature {
            params: vec![Val::I32, Val::I32],
            results: Vec::new(),
        };
        let params = sig.params.clone();
        let ty = shared.type_index(sig);
        let body = Fx::new(module, &mut shared, &helpers, params, max_stride)
            .function(init, Kind::Function);
        exports.push((exports.len().to_string(), functions.len() as u32));
        functions.push((ty, body));
        has_init = true;
    }
    let mut has_merge = false;
    if let (Mode::Reduce, Some(merge)) = (module.manifest.mode, &module.merge) {
        let sig = signature(module, merge, Kind::Merge);
        let params = sig.params.clone();
        let ty = shared.type_index(sig);
        let body =
            Fx::new(module, &mut shared, &helpers, params, max_stride).function(merge, Kind::Merge);
        exports.push((exports.len().to_string(), functions.len() as u32));
        functions.push((ty, body));
        has_merge = true;
    }
    // The slots and the constant tables, in the Kernel's own memory.
    let mut frame = vec![0u64; (shared.frame as usize).div_ceil(8).max(1)];
    for (offset, words) in &shared.tables {
        for (k, word) in words.iter().enumerate() {
            let at = *offset as usize / 4 + k;
            let slot = &mut frame[at / 2];
            let shift = (at % 2) * 32;
            *slot = (*slot & !(0xffff_ffffu64 << shift)) | (u64::from(*word) << shift);
        }
    }
    let bytes = Module {
        types: shared.types,
        functions,
        exports,
        global: frame.as_mut_ptr() as usize as i32,
    }
    .encode();
    let mut slots = [0u32; 3];
    let loaded = unsafe { load(bytes.as_ptr(), bytes.len(), slots.as_mut_ptr(), 3) };
    let wanted = 1 + usize::from(has_init) + usize::from(has_merge);
    if loaded < wanted as i32 {
        return Err("the host refused the compiled Kernel".into());
    }
    let slots = slots[..wanted].to_vec();
    let mut rest = slots.iter().skip(1).map(|slot| *slot as usize);
    let init = has_init.then(|| rest.next()).flatten();
    let merge = has_merge.then(|| rest.next()).flatten();
    Ok(Compiled {
        entry: slots[0] as usize,
        init,
        merge,
        slots,
        _frame: frame,
    })
}
