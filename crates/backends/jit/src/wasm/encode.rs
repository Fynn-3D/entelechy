//! The WebAssembly binary subset the JIT writes: types, imported memory
//! and table, functions, one global, exports and code. The host
//! validates the module when it compiles it.

/// A value type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Val {
    I32,
    I64,
    F32,
    F64,
}

impl Val {
    fn byte(self) -> u8 {
        match self {
            Val::I32 => 0x7f,
            Val::I64 => 0x7e,
            Val::F32 => 0x7d,
            Val::F64 => 0x7c,
        }
    }

    /// Bytes the value takes in memory.
    pub fn bytes(self) -> u32 {
        match self {
            Val::I32 | Val::F32 => 4,
            Val::I64 | Val::F64 => 8,
        }
    }
}

/// A function type: its parameters and at most one result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    pub params: Vec<Val>,
    pub results: Vec<Val>,
}

pub fn unsigned(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

pub fn signed(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn name(out: &mut Vec<u8>, text: &str) {
    unsigned(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

fn section(module: &mut Vec<u8>, id: u8, body: &[u8]) {
    module.push(id);
    unsigned(module, body.len() as u64);
    module.extend_from_slice(body);
}

/// One function's body: its locals beyond the parameters, by type in
/// the order declared, and its code, ending in `end`.
pub struct Body {
    pub locals: Vec<Val>,
    pub code: Vec<u8>,
}

impl Body {
    fn encode(&self, out: &mut Vec<u8>) {
        let mut entry = Vec::new();
        // Runs of one type are one declaration.
        let mut runs: Vec<(u32, Val)> = Vec::new();
        for local in &self.locals {
            match runs.last_mut() {
                Some((count, ty)) if ty == local => *count += 1,
                _ => runs.push((1, *local)),
            }
        }
        unsigned(&mut entry, runs.len() as u64);
        for (count, ty) in runs {
            unsigned(&mut entry, u64::from(count));
            entry.push(ty.byte());
        }
        entry.extend_from_slice(&self.code);
        unsigned(out, entry.len() as u64);
        out.extend_from_slice(&entry);
    }
}

/// A module: every function's type index and body, the exports by
/// function index, and the one immutable `i32` global's value.
pub struct Module {
    pub types: Vec<Signature>,
    pub functions: Vec<(u32, Body)>,
    pub exports: Vec<(String, u32)>,
    pub global: i32,
}

impl Module {
    pub fn encode(&self) -> Vec<u8> {
        let mut module = vec![0x00, b'a', b's', b'm', 0x01, 0x00, 0x00, 0x00];
        let mut types = Vec::new();
        unsigned(&mut types, self.types.len() as u64);
        for signature in &self.types {
            types.push(0x60);
            unsigned(&mut types, signature.params.len() as u64);
            types.extend(signature.params.iter().map(|v| v.byte()));
            unsigned(&mut types, signature.results.len() as u64);
            types.extend(signature.results.iter().map(|v| v.byte()));
        }
        section(&mut module, 1, &types);
        // The host's memory and function table.
        let mut imports = Vec::new();
        unsigned(&mut imports, 2);
        name(&mut imports, "env");
        name(&mut imports, "memory");
        imports.extend([0x02, 0x00, 0x00]);
        name(&mut imports, "env");
        name(&mut imports, "table");
        imports.extend([0x01, 0x70, 0x00, 0x00]);
        section(&mut module, 2, &imports);
        let mut functions = Vec::new();
        unsigned(&mut functions, self.functions.len() as u64);
        for (ty, _) in &self.functions {
            unsigned(&mut functions, u64::from(*ty));
        }
        section(&mut module, 3, &functions);
        let mut globals = Vec::new();
        unsigned(&mut globals, 1);
        globals.extend([Val::I32.byte(), 0x00, 0x41]);
        signed(&mut globals, i64::from(self.global));
        globals.push(0x0b);
        section(&mut module, 6, &globals);
        let mut exports = Vec::new();
        unsigned(&mut exports, self.exports.len() as u64);
        for (export, function) in &self.exports {
            name(&mut exports, export);
            exports.push(0x00);
            unsigned(&mut exports, u64::from(*function));
        }
        section(&mut module, 7, &exports);
        let mut code = Vec::new();
        unsigned(&mut code, self.functions.len() as u64);
        for (_, body) in &self.functions {
            body.encode(&mut code);
        }
        section(&mut module, 10, &code);
        module
    }
}
