//! Entelechy IR to WebAssembly, instruction for instruction as `codegen`
//! does, so the words are the interpreter's: every IR value is its
//! words, one local each; `if`, `match` and `loop` are WebAssembly
//! blocks; integer arithmetic is checked, the first failure recorded in
//! the context. Host reads and Rust-rounded math call the crate's
//! helpers through Entelechy's function table.
//!
//! Stack data in `codegen` (the scratch, a callee's result, an array
//! indexed at run time) is here a slot in memory the prepared Kernel
//! owns, based at the module's one global. WebAssembly runs on one
//! thread and Kernel functions never recurse, so each slot has one user
//! at a time.

use std::collections::HashMap;

use entelechy_abi::{Encoding, Scalar, Type, encoding};
use entelechy_ir::{BinaryOp, Function, Instruction, Intrinsic, Module, Op, UnaryOp};

use super::encode::{Body, Signature, Val, signed, unsigned};
use crate::failure;
use crate::layout::{self, Lane, byte_offsets, byte_size, field_offset, holds_wide};
use crate::offsets;

/// The failure codes ([`crate::failure`]) as this code stores them.
const DIVISION_BY_ZERO: i32 = failure::DIVISION_BY_ZERO as i32;
const OVERFLOW: i32 = failure::OVERFLOW as i32;
const EXHAUSTED: i32 = failure::EXHAUSTED as i32;
const OVERRUN: i32 = failure::OVERRUN as i32;

/// A word: the local that holds it, and its type.
pub type W = (u32, Val);

/// What a module's functions share: function types, slot memory, and
/// constant-table contents.
#[derive(Default)]
pub struct Shared {
    pub types: Vec<Signature>,
    /// Bytes of slots so far.
    pub frame: u32,
    /// The constant tables' words, written into their slots before the
    /// code runs.
    pub tables: Vec<(u32, Vec<u32>)>,
}

impl Shared {
    pub fn type_index(&mut self, signature: Signature) -> u32 {
        if let Some(at) = self.types.iter().position(|s| *s == signature) {
            return at as u32;
        }
        self.types.push(signature);
        (self.types.len() - 1) as u32
    }

    /// A slot of `bytes`, eight-aligned.
    pub fn reserve(&mut self, bytes: u32) -> u32 {
        let at = self.frame;
        self.frame += bytes.max(8).div_ceil(8) * 8;
        at
    }
}

/// What the function being translated is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The context, its result's slot, its arguments' words.
    Function,
    /// A map's entry: the context, the run's first `x`, its `y` and
    /// `z`, its length, and where its words go.
    Map,
    /// A scan's entry: the context, the index, the carried words, the
    /// output slot.
    Scan,
    /// A reduction's `merge`: the context, the two values' words, the
    /// output slot.
    Merge,
}
/// The WebAssembly type of each scalar of a value of `ty`, in layout
/// order ([`layout::lanes`]).
pub fn word_types(module: &Module, ty: Type) -> Vec<Val> {
    layout::lanes(module, ty)
        .into_iter()
        .map(|lane| match lane {
            Lane::F32 => Val::F32,
            Lane::F64 => Val::F64,
            Lane::I32 => Val::I32,
            Lane::I64 => Val::I64,
        })
        .collect()
}

impl layout::Width for Val {
    fn width(self) -> u32 {
        self.bytes()
    }
}

/// The signature every function of `kind` has.
pub fn signature(module: &Module, function: &Function, kind: Kind) -> Signature {
    let params = match kind {
        Kind::Function => {
            let mut params = vec![Val::I32, Val::I32];
            for arg in &function.args {
                params.extend(word_types(module, *arg));
            }
            params
        }
        Kind::Map => vec![Val::I32; 6],
        Kind::Scan | Kind::Merge => vec![Val::I32; 4],
    };
    Signature {
        params,
        results: Vec::new(),
    }
}

/// An input's view as read before the run.
#[derive(Clone, Copy)]
struct Settled {
    words: W,
    element: W,
    last_x: W,
    row: W,
    last: W,
}

// Opcodes.
const BLOCK: u8 = 0x02;
const LOOP: u8 = 0x03;
const IF: u8 = 0x04;
const ELSE: u8 = 0x05;
const END: u8 = 0x0b;
const RETURN: u8 = 0x0f;
const BR: u8 = 0x0c;
const BR_IF: u8 = 0x0d;
const BR_TABLE: u8 = 0x0e;
const CALL: u8 = 0x10;
const CALL_INDIRECT: u8 = 0x11;
const SELECT: u8 = 0x1b;
const LOCAL_GET: u8 = 0x20;
const LOCAL_SET: u8 = 0x21;
const GLOBAL_GET: u8 = 0x23;
const I32_CONST: u8 = 0x41;
const I64_CONST: u8 = 0x42;
const F32_CONST: u8 = 0x43;
const F64_CONST: u8 = 0x44;
const I32_EQZ: u8 = 0x45;
const I32_EQ: u8 = 0x46;
const I32_NE: u8 = 0x47;
const I32_LT_S: u8 = 0x48;
const I32_LT_U: u8 = 0x49;
const I32_GT_S: u8 = 0x4a;
const I32_GE_U: u8 = 0x4f;
const I32_ADD: u8 = 0x6a;
const I32_SUB: u8 = 0x6b;
const I32_MUL: u8 = 0x6c;
const I32_DIV_U: u8 = 0x6e;
const I32_REM_U: u8 = 0x70;
const I32_AND: u8 = 0x71;
const I32_OR: u8 = 0x72;
const I32_XOR: u8 = 0x73;
const I32_SHR_U: u8 = 0x76;
const I64_ADD: u8 = 0x7c;
const I64_MUL: u8 = 0x7e;
const I64_SHR_S: u8 = 0x87;
const I32_WRAP_I64: u8 = 0xa7;
const I64_EXTEND_I32_S: u8 = 0xac;
const I64_EXTEND_I32_U: u8 = 0xad;
const F32_CONVERT_I32_S: u8 = 0xb2;
const F32_CONVERT_I32_U: u8 = 0xb3;
const F32_FLOOR: u8 = 0x8e;
const F32_REINTERPRET_I32: u8 = 0xbe;
const I32_REINTERPRET_F32: u8 = 0xbc;

/// Per type: (i32, i64, f32, f64) opcodes of one operation.
fn by_type(v: Val, ops: [u8; 4]) -> u8 {
    match v {
        Val::I32 => ops[0],
        Val::I64 => ops[1],
        Val::F32 => ops[2],
        Val::F64 => ops[3],
    }
}

/// One function being translated.
pub struct Fx<'a> {
    module: &'a Module,
    shared: &'a mut Shared,
    /// The IR functions' indices in the module, by IR function index.
    code: Vec<u8>,
    params: Vec<Val>,
    locals: Vec<Val>,
    constants: HashMap<u32, Vec<u32>>,
    pointers: HashMap<u32, W>,
    tables: HashMap<u32, u32>,
    values: Vec<Vec<W>>,
    types: Vec<Type>,
    args: Vec<Vec<W>>,
    loops: Vec<(Vec<W>, W)>,
    scan: Option<(Vec<W>, W)>,
    ctx: W,
    coords: Option<(W, W, W, W)>,
    settled: Vec<Option<Settled>>,
    scratch: u32,
    budget: Option<u32>,
    work: Option<(W, W, W)>,
    ran: HashMap<u32, W>,
    helpers: &'a HashMap<&'static str, u32>,
    /// A group's entry running its lanes: where each lane's state is.
    group: Option<GroupSlots>,
    /// In a group's lanes, the lane running.
    running: Option<W>,
}

/// A group's lanes in slots of the module's memory, lane after lane: each
/// lane's work and work at its last look, whether it failed (a word
/// each), and the lowest lane failed in the stretch with its failure's
/// code and element (three words); and lane 0's element.
#[derive(Clone, Copy)]
struct GroupSlots {
    size: u32,
    work: u32,
    seen: u32,
    failed: u32,
    held: u32,
    first: W,
}

impl<'a> Fx<'a> {
    pub fn new(
        module: &'a Module,
        shared: &'a mut Shared,
        helpers: &'a HashMap<&'static str, u32>,
        params: Vec<Val>,
        max_stride: u32,
    ) -> Self {
        let scratch = shared.reserve(max_stride * 4);
        Self {
            module,
            shared,
            code: Vec::new(),
            params,
            locals: Vec::new(),
            constants: HashMap::new(),
            pointers: HashMap::new(),
            tables: HashMap::new(),
            values: Vec::new(),
            types: Vec::new(),
            args: Vec::new(),
            loops: Vec::new(),
            scan: None,
            ctx: (0, Val::I32),
            coords: None,
            settled: Vec::new(),
            scratch,
            budget: module.accounted().then_some(module.manifest.budget),
            work: None,
            ran: HashMap::new(),
            helpers,
            group: None,
            running: None,
        }
    }

    // --- Emission ---

    fn byte(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn uleb(&mut self, value: u32) {
        unsigned(&mut self.code, u64::from(value));
    }

    fn local(&mut self, ty: Val) -> W {
        self.locals.push(ty);
        ((self.params.len() + self.locals.len() - 1) as u32, ty)
    }

    fn get(&mut self, w: W) {
        self.byte(LOCAL_GET);
        self.uleb(w.0);
    }

    fn set(&mut self, w: W) {
        self.byte(LOCAL_SET);
        self.uleb(w.0);
    }

    /// Pops the stack into a new local of `ty`.
    fn take(&mut self, ty: Val) -> W {
        let w = self.local(ty);
        self.set(w);
        w
    }

    fn i32c(&mut self, v: i32) -> W {
        self.push_i32(v);
        self.take(Val::I32)
    }

    fn push_i32(&mut self, v: i32) {
        self.byte(I32_CONST);
        signed(&mut self.code, i64::from(v));
    }

    fn push_i64(&mut self, v: i64) {
        self.byte(I64_CONST);
        signed(&mut self.code, v);
    }

    fn f32c(&mut self, v: f32) -> W {
        self.byte(F32_CONST);
        self.code.extend_from_slice(&v.to_bits().to_le_bytes());
        self.take(Val::F32)
    }

    fn f64c(&mut self, v: f64) -> W {
        self.byte(F64_CONST);
        self.code.extend_from_slice(&v.to_bits().to_le_bytes());
        self.take(Val::F64)
    }

    fn i64c(&mut self, v: i64) -> W {
        self.push_i64(v);
        self.take(Val::I64)
    }

    /// A constant of `like`'s type.
    fn fconst(&mut self, like: W, v: f64) -> W {
        match like.1 {
            Val::F64 => self.f64c(v),
            _ => self.f32c(v as f32),
        }
    }

    fn iconst(&mut self, like: W, v: i64) -> W {
        match like.1 {
            Val::I64 => self.i64c(v),
            _ => self.i32c(v as i32),
        }
    }

    /// `op` over one word, to a word of `out`.
    fn un(&mut self, a: W, op: u8, out: Val) -> W {
        self.get(a);
        self.byte(op);
        self.take(out)
    }

    /// `op` over two words, to a word of `out`.
    fn bin(&mut self, a: W, b: W, op: u8, out: Val) -> W {
        self.get(a);
        self.get(b);
        self.byte(op);
        self.take(out)
    }

    /// `op` over two words of `a`'s type, one opcode per type.
    fn arith(&mut self, a: W, b: W, ops: [u8; 4]) -> W {
        let op = by_type(a.1, ops);
        self.bin(a, b, op, a.1)
    }

    /// A comparison of two words of `a`'s type, to a `Bool` word.
    fn compare(&mut self, a: W, b: W, ops: [u8; 4]) -> W {
        let op = by_type(a.1, ops);
        self.bin(a, b, op, Val::I32)
    }

    /// A comparison with a constant, to a `Bool` word.
    fn compare_imm(&mut self, a: W, v: i64, ops: [u8; 4]) -> W {
        self.get(a);
        match a.1 {
            Val::I64 => self.push_i64(v),
            _ => self.push_i32(v as i32),
        }
        self.byte(by_type(a.1, ops));
        self.take(Val::I32)
    }

    /// `a` when `cond`, else `b`.
    fn select(&mut self, cond: W, a: W, b: W) -> W {
        self.get(a);
        self.get(b);
        self.get(cond);
        self.byte(SELECT);
        self.take(a.1)
    }

    fn load(&mut self, ty: Val, base: W, offset: u32) -> W {
        self.get(base);
        self.load_op(ty, offset);
        self.take(ty)
    }

    fn load_op(&mut self, ty: Val, offset: u32) {
        self.byte(by_type(ty, [0x28, 0x29, 0x2a, 0x2b]));
        self.uleb(0);
        self.uleb(offset);
    }

    fn store(&mut self, base: W, offset: u32, value: W) {
        self.get(base);
        self.get(value);
        self.byte(by_type(value.1, [0x36, 0x37, 0x38, 0x39]));
        self.uleb(0);
        self.uleb(offset);
    }

    /// The address of a slot: the module's global plus its offset.
    fn slot(&mut self, offset: u32) -> W {
        self.byte(GLOBAL_GET);
        self.uleb(0);
        self.push_i32(offset as i32);
        self.byte(I32_ADD);
        self.take(Val::I32)
    }

    fn scratch_addr(&mut self) -> W {
        let at = self.scratch;
        self.slot(at)
    }

    /// Calls a helper through the table at its pointer's index.
    fn helper(&mut self, name: &'static str, args: &[W], result: Option<Val>) -> Option<W> {
        let index = *self.helpers.get(name).expect("a helper the crate defines");
        let ty = self.shared.type_index(Signature {
            params: args.iter().map(|a| a.1).collect(),
            results: result.into_iter().collect(),
        });
        for arg in args {
            self.get(*arg);
        }
        self.push_i32(index as i32);
        self.byte(CALL_INDIRECT);
        self.uleb(ty);
        self.uleb(0);
        result.map(|ty| self.take(ty))
    }

    fn ctx_load(&mut self, ty: Val, offset: i32) -> W {
        let ctx = self.ctx;
        self.load(ty, ctx, offset as u32)
    }

    fn ctx_store(&mut self, offset: i32, value: W) {
        let ctx = self.ctx;
        self.store(ctx, offset as u32, value);
    }

    /// Records the first failure when `cond` holds.
    fn fail_if(&mut self, cond: W, code: i32) {
        let current = self.ctx_load(Val::I32, offsets::ERROR);
        let none = self.un(current, I32_EQZ, Val::I32);
        let set = self.bin(none, cond, I32_AND, Val::I32);
        let code = self.i32c(code);
        let next = self.select(set, code, current);
        self.ctx_store(offsets::ERROR, next);
    }

    fn umin(&mut self, a: W, b: W) -> W {
        let below = self.compare(a, b, [I32_LT_U, 0x54, 0, 0]);
        self.select(below, a, b)
    }

    fn smin(&mut self, a: W, b: W) -> W {
        let below = self.compare(a, b, [I32_LT_S, 0x53, 0, 0]);
        self.select(below, a, b)
    }

    fn smax(&mut self, a: W, b: W) -> W {
        let above = self.compare(a, b, [I32_GT_S, 0x55, 0, 0]);
        self.select(above, a, b)
    }

    fn add_imm(&mut self, a: W, v: i32) -> W {
        let c = self.i32c(v);
        self.bin(a, c, I32_ADD, Val::I32)
    }

    fn mul_imm(&mut self, a: W, v: i32) -> W {
        let c = self.i32c(v);
        self.bin(a, c, I32_MUL, Val::I32)
    }

    fn u32_to_f32(&mut self, v: W) -> W {
        self.un(v, F32_CONVERT_I32_U, Val::F32)
    }

    // --- A group's lanes ---

    /// The address of word `index` of the slot at `offset`.
    fn lane_word(&mut self, offset: u32, index: W) -> W {
        let base = self.slot(offset);
        let scaled = self.mul_imm(index, 4);
        self.bin(base, scaled, I32_ADD, Val::I32)
    }

    /// Runs `each` once for every lane of `size`, its index a local.
    fn over_lanes(&mut self, size: u32, each: impl FnOnce(&mut Self, W)) {
        let lane = self.local(Val::I32);
        self.push_i32(0);
        self.set(lane);
        self.byte(LOOP);
        self.byte(0x40);
        each(self, lane);
        self.get(lane);
        self.push_i32(1);
        self.byte(I32_ADD);
        self.set(lane);
        self.get(lane);
        self.push_i32(size as i32);
        self.byte(I32_LT_U);
        self.byte(BR_IF);
        self.uleb(0);
        self.byte(END);
    }

    /// A group's lanes laid out for its entry: none has worked or failed,
    /// none is held; lane 0 is the element the entry was called at.
    fn open_lanes(&mut self, size: u32) {
        let work = self.shared.reserve(size * 4);
        let seen = self.shared.reserve(size * 4);
        let failed = self.shared.reserve(size * 4);
        let held = self.shared.reserve(12);
        let first = self.element_index();
        self.over_lanes(size, |fx, lane| {
            let zero = fx.i32c(0);
            for offset in [work, seen, failed] {
                let at = fx.lane_word(offset, lane);
                fx.store(at, 0, zero);
            }
        });
        let none = self.i32c(size as i32);
        let at = self.slot(held);
        self.store(at, 0, none);
        self.group = Some(GroupSlots {
            size,
            work,
            seen,
            failed,
            held,
            first,
        });
    }

    /// Each lane's work short of a block, handed to the run as an
    /// element's is when it ends.
    fn settle_lanes(&mut self) {
        let Some(group) = self.group else {
            return;
        };
        if self.work.is_none() {
            return;
        }
        self.over_lanes(group.size, |fx, lane| {
            let at = fx.lane_word(group.work, lane);
            let work = fx.load(Val::I32, at, 0);
            let mask = fx.i32c((entelechy_ir::WORK_BLOCK - 1) as i32);
            let short = fx.bin(work, mask, I32_AND, Val::I32);
            let short = fx.un(short, I64_EXTEND_I32_U, Val::I64);
            let left = fx.ctx_load(Val::I64, offsets::LEFT);
            let left = fx.bin(left, short, I64_ADD, Val::I64);
            fx.ctx_store(offsets::LEFT, left);
        });
    }

    /// [`Op::Lanes`]: the body once for each lane not failed, as that lane
    /// (its element's place, its work), its result the lane's element of
    /// the array; a lane that fails is marked, the lowest held with its
    /// failure, and the context's failure cleared for the next.
    fn lanes(&mut self, ty: Type, body: &entelechy_ir::Block) -> Vec<W> {
        let group = self.group.expect("lanes run in a group's entry");
        let size = group.size;
        let array_types = word_types(self.module, ty);
        let element_types = array_types[..array_types.len() / size as usize].to_vec();
        let stride = byte_size(&element_types) as i32;
        let array = self.shared.reserve(byte_size(&array_types));
        let around = self.coords;
        let counted = self.work.is_some();
        self.over_lanes(size, |fx, lane| {
            let flag_at = fx.lane_word(group.failed, lane);
            let flag = fx.load(Val::I32, flag_at, 0);
            fx.get(flag);
            fx.byte(I32_EQZ);
            fx.byte(IF);
            fx.byte(0x40);
            // The lane's place: its element in the domain, `x` fastest.
            let element = fx.bin(group.first, lane, I32_ADD, Val::I32);
            let width = fx.ctx_load(Val::I32, offsets::WIDTH);
            let height = fx.ctx_load(Val::I32, offsets::HEIGHT);
            let x = fx.bin(element, width, I32_REM_U, Val::I32);
            let row = fx.bin(element, width, I32_DIV_U, Val::I32);
            let y = fx.bin(row, height, I32_REM_U, Val::I32);
            let z = fx.bin(row, height, I32_DIV_U, Val::I32);
            let out = around.expect("the entry's place").3;
            fx.coords = Some((x, y, z, out));
            fx.running = Some(lane);
            let kept = counted.then(|| {
                let work_at = fx.lane_word(group.work, lane);
                let seen_at = fx.lane_word(group.seen, lane);
                let work = fx.load(Val::I32, work_at, 0);
                let seen = fx.load(Val::I32, seen_at, 0);
                fx.count_from(work, seen);
                (work_at, seen_at)
            });
            fx.block(&body.body);
            let words = fx.words(&body.result);
            let base = fx.slot(array);
            let scaled = fx.mul_imm(lane, stride);
            let at = fx.bin(base, scaled, I32_ADD, Val::I32);
            for (word, offset) in words.iter().zip(byte_offsets(&element_types)) {
                fx.store(at, offset, *word);
            }
            if let (Some((work_at, seen_at)), Some((count, _, seen))) = (kept, fx.work) {
                fx.store(work_at, 0, count);
                fx.store(seen_at, 0, seen);
            }
            let error = fx.ctx_load(Val::I32, offsets::ERROR);
            fx.get(error);
            fx.byte(IF);
            fx.byte(0x40);
            {
                let one = fx.i32c(1);
                fx.store(flag_at, 0, one);
                let held = fx.slot(group.held);
                let held_lane = fx.load(Val::I32, held, 0);
                let held_code = fx.load(Val::I32, held, 4);
                let held_element = fx.load(Val::I32, held, 8);
                let lower = fx.compare(lane, held_lane, [I32_LT_U, 0, 0, 0]);
                let lane_kept = fx.select(lower, lane, held_lane);
                let code_kept = fx.select(lower, error, held_code);
                let element_kept = fx.select(lower, element, held_element);
                fx.store(held, 0, lane_kept);
                fx.store(held, 4, code_kept);
                fx.store(held, 8, element_kept);
                let clear = fx.i32c(0);
                fx.ctx_store(offsets::ERROR, clear);
            }
            fx.byte(END);
            fx.byte(END);
        });
        self.coords = around;
        self.running = None;
        let base = self.slot(array);
        self.loads(&array_types, base, 0)
    }

    /// [`Op::Meet`]: once a lane is held, the run fails with it, its
    /// element named, and the entry returns.
    fn meet(&mut self) -> Vec<W> {
        let group = self.group.expect("a group's entry meets");
        let held = self.slot(group.held);
        let held_lane = self.load(Val::I32, held, 0);
        let any = self.compare_imm(held_lane, i64::from(group.size), [I32_LT_U, 0, 0, 0]);
        self.get(any);
        self.byte(IF);
        self.byte(0x40);
        let code = self.load(Val::I32, held, 4);
        let element = self.load(Val::I32, held, 8);
        self.ctx_store(offsets::ERROR, code);
        self.ctx_store(offsets::ERROR_AT, element);
        self.byte(RETURN);
        self.byte(END);
        vec![self.i32c(1)]
    }

    // --- The work count ---

    fn count_from(&mut self, work: W, looked: W) {
        let (Some(budget), Some((count, next, seen))) = (self.budget, self.work) else {
            return;
        };
        self.get(work);
        self.set(count);
        self.get(looked);
        self.set(seen);
        let mask = self.i32c((entelechy_ir::WORK_BLOCK - 1) as i32);
        let block = self.bin(looked, mask, I32_OR, Val::I32);
        let block_end = self.add_imm(block, 1);
        let past = self.i32c(budget.wrapping_add(1) as i32);
        let at = self.umin(block_end, past);
        self.get(at);
        self.set(next);
    }

    fn load_count(&mut self) {
        if self.work.is_none() {
            return;
        }
        let work = self.ctx_load(Val::I32, offsets::WORK);
        let looked = self.ctx_load(Val::I32, offsets::LOOKED);
        self.count_from(work, looked);
    }

    fn begin_count(&mut self) {
        if self.budget.is_none() {
            return;
        }
        let count = self.local(Val::I32);
        let next = self.local(Val::I32);
        let looked = self.local(Val::I32);
        self.work = Some((count, next, looked));
        self.load_count();
    }

    fn spend(&mut self, ran: W) -> W {
        if let Some((count, _, _)) = self.work {
            self.get(count);
            self.get(ran);
            self.byte(I32_ADD);
            self.set(count);
        }
        ran
    }

    /// [`Op::Look`], as the Cranelift translation does it.
    fn look(&mut self) -> W {
        let (Some(budget), Some((count, next, seen))) = (self.budget, self.work) else {
            return self.i32c(0);
        };
        let stop = self.local(Val::I32);
        self.push_i32(0);
        self.set(stop);
        // Due: the count has reached the next look.
        self.get(count);
        self.get(next);
        self.byte(I32_GE_U);
        self.byte(IF);
        self.byte(0x40);
        {
            let work = self.local(Val::I32);
            self.get(count);
            self.set(work);
            let limit = self.i32c(budget as i32);
            let over = self.compare(work, limit, [0x4b, 0, 0, 0]);
            self.fail_if(over, EXHAUSTED);
            self.get(over);
            self.byte(IF);
            self.byte(0x40);
            self.push_i32(1);
            self.set(stop);
            self.byte(ELSE);
            {
                let shift = self.i32c(entelechy_ir::WORK_BLOCK.trailing_zeros() as i32);
                let now = self.bin(work, shift, I32_SHR_U, Val::I32);
                let before = self.bin(seen, shift, I32_SHR_U, Val::I32);
                let filled = self.bin(now, before, I32_SUB, Val::I32);
                let blocks = self.ctx_load(Val::I32, offsets::RUN_BLOCKS);
                let limit = self.ctx_load(Val::I32, offsets::RUN_LIMIT);
                // One thread: the run's total is updated in place.
                let handed = self.load(Val::I32, blocks, 0);
                let total = self.bin(handed, filled, I32_ADD, Val::I32);
                self.store(blocks, 0, total);
                let passed = self.compare(total, limit, [0x4b, 0, 0, 0]);
                self.fail_if(passed, OVERRUN);
                self.get(passed);
                self.byte(IF);
                self.byte(0x40);
                {
                    self.push_i32(budget.wrapping_add(1) as i32);
                    self.set(count);
                    self.push_i32(1);
                    self.set(stop);
                }
                self.byte(ELSE);
                {
                    self.count_from(work, work);
                }
                self.byte(END);
            }
            self.byte(END);
        }
        self.byte(END);
        stop
    }

    fn store_count(&mut self) {
        if let Some((count, _, seen)) = self.work {
            self.ctx_store(offsets::WORK, count);
            self.ctx_store(offsets::LOOKED, seen);
        }
    }

    // --- Functions ---

    /// Translates a whole function into its body.
    pub fn function(mut self, function: &Function, kind: Kind) -> Body {
        let slots = function.slot_count();
        self.values = vec![Vec::new(); slots];
        self.types = vec![Type::Float; slots];
        let params: Vec<W> = self
            .params
            .iter()
            .enumerate()
            .map(|(k, ty)| (k as u32, *ty))
            .collect();
        self.begin_count();
        if kind == Kind::Map {
            return self.map_run(function, &params);
        }
        let out = match kind {
            Kind::Map => unreachable!("a map's entry walks its run"),
            Kind::Scan => {
                let zero = self.i32c(0);
                self.coords = Some((params[1], zero, zero, params[3]));
                let carried_types = word_types(self.module, function.result_type());
                let carried = self.loads(&carried_types, params[2], 0);
                self.scan = Some((carried, params[1]));
                params[3]
            }
            Kind::Function => {
                let mut cursor = 2;
                for arg in &function.args {
                    let n = word_types(self.module, *arg).len();
                    self.args.push(params[cursor..cursor + n].to_vec());
                    cursor += n;
                }
                params[1]
            }
            Kind::Merge => {
                for (arg, pointer) in function.args.iter().zip([params[1], params[2]]) {
                    let tys = word_types(self.module, *arg);
                    let words = self.loads(&tys, pointer, 0);
                    self.args.push(words);
                }
                params[3]
            }
        };
        self.block(&function.body);
        let mut offset = 0u32;
        for result in &function.results {
            for word in self.values[result.0 as usize].clone() {
                self.store(out, offset, word);
                offset += word.1.bytes();
            }
        }
        self.store_count();
        self.byte(END);
        Body {
            locals: self.locals,
            code: self.code,
        }
    }

    fn map_run(mut self, function: &Function, params: &[W]) -> Body {
        let (x0, y, z, count, out0) = (params[1], params[2], params[3], params[4], params[5]);
        // A group's lanes are elements in rows of their own: each reads
        // at its own place.
        let group = match self.module.manifest.mode {
            entelechy_ir::Mode::Group { size } => Some(size),
            _ => None,
        };
        if group.is_none() {
            self.settle_views(y, z);
        }
        let taken = self.i32c(0);
        let out = self.local(Val::I32);
        self.get(out0);
        self.set(out);
        let x = self.local(Val::I32);
        self.byte(BLOCK);
        self.byte(0x40);
        self.byte(LOOP);
        self.byte(0x40);
        {
            // Past the run: out.
            self.get(taken);
            self.get(count);
            self.byte(I32_GE_U);
            self.byte(BR_IF);
            self.uleb(1);
            self.get(x0);
            self.get(taken);
            self.byte(I32_ADD);
            self.set(x);
            self.coords = Some((x, y, z, out));
            if self.budget.is_some() {
                let zero = self.i32c(0);
                self.count_from(zero, zero);
            }
            if let Some(size) = group {
                self.open_lanes(size);
            }
            self.block(&function.body);
            let mut offset = 0u32;
            for result in &function.results {
                for word in self.values[result.0 as usize].clone() {
                    self.store(out, offset, word);
                    offset += word.1.bytes();
                }
            }
            if group.is_some() {
                self.settle_lanes();
            } else if let Some((count, _, _)) = self.work {
                let mask = self.i32c((entelechy_ir::WORK_BLOCK - 1) as i32);
                let short = self.bin(count, mask, I32_AND, Val::I32);
                let short = self.un(short, I64_EXTEND_I32_U, Val::I64);
                let left = self.ctx_load(Val::I64, offsets::LEFT);
                let left = self.bin(left, short, I64_ADD, Val::I64);
                self.ctx_store(offsets::LEFT, left);
            }
            // A failure stops the run where it happened; a group's names
            // its element where its lanes met.
            let error = self.ctx_load(Val::I32, offsets::ERROR);
            self.get(error);
            self.byte(IF);
            self.byte(0x40);
            if group.is_none() {
                self.ctx_store(offsets::ERROR_AT, x);
            }
            self.byte(BR);
            self.uleb(2);
            self.byte(END);
            self.get(taken);
            self.push_i32(1);
            self.byte(I32_ADD);
            self.set(taken);
            self.get(out);
            self.push_i32(offset as i32);
            self.byte(I32_ADD);
            self.set(out);
            self.byte(BR);
            self.uleb(0);
        }
        self.byte(END);
        self.byte(END);
        self.byte(END);
        Body {
            locals: self.locals,
            code: self.code,
        }
    }

    fn block(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.types[instruction.id.0 as usize] = instruction.ty;
            let words = self.instruction(instruction);
            self.values[instruction.id.0 as usize] = words;
            self.note_constant(instruction);
        }
    }

    fn note_constant(&mut self, instruction: &Instruction) {
        let id = instruction.id.0;
        if holds_wide(self.module, instruction.ty) || self.reads_wide(&instruction.op) {
            return;
        }
        let words = match &instruction.op {
            Op::Const(value) => Some(value.words()),
            Op::Construct(parts)
                if instruction.ty.components().is_none()
                    || parts.iter().all(|part| self.ty(part) == Type::Float) =>
            {
                parts
                    .iter()
                    .map(|part| self.constants.get(&part.0).cloned())
                    .collect::<Option<Vec<_>>>()
                    .map(|parts| parts.concat())
            }
            Op::Extract(source, component) => self.constants.get(&source.0).and_then(|words| {
                let (start, n) = match self.ty(source) {
                    Type::Struct(index) => (
                        field_offset(self.module, index, *component as usize),
                        word_types(self.module, instruction.ty).len(),
                    ),
                    other if other.order().is_some() => {
                        let n = usize::from(other.order().unwrap_or(1));
                        (*component as usize * n, n)
                    }
                    _ => (*component as usize, 1),
                };
                words.get(start..start + n).map(<[u32]>::to_vec)
            }),
            _ => None,
        };
        if let Some(words) = words {
            self.constants.insert(id, words);
        }
    }

    fn reads_wide(&self, op: &Op) -> bool {
        match op {
            Op::Extract(source, _) => holds_wide(self.module, self.ty(source)),
            _ => false,
        }
    }

    /// The address of a constant array's words, in its own slot written
    /// once before the code runs.
    fn table(&mut self, id: u32) -> W {
        let offset = match self.tables.get(&id) {
            Some(offset) => *offset,
            None => {
                let words = self.constants.get(&id).cloned().unwrap_or_default();
                let offset = self.shared.reserve(words.len() as u32 * 4);
                self.shared.tables.push((offset, words));
                self.tables.insert(id, offset);
                offset
            }
        };
        self.slot(offset)
    }

    fn words(&self, id: &entelechy_ir::ValueId) -> Vec<W> {
        self.values[id.0 as usize].clone()
    }

    fn ty(&self, id: &entelechy_ir::ValueId) -> Type {
        self.types[id.0 as usize]
    }

    fn loads(&mut self, tys: &[Val], base: W, offset: u32) -> Vec<W> {
        tys.iter()
            .zip(byte_offsets(tys))
            .map(|(t, at)| self.load(*t, base, offset + at))
            .collect()
    }

    fn stores(&mut self, words: &[W], base: W) {
        let mut at = 0;
        for word in words {
            self.store(base, at, *word);
            at += word.1.bytes();
        }
    }

    fn load_scratch(&mut self, tys: &[Val]) -> Vec<W> {
        let base = self.scratch_addr();
        self.loads(tys, base, 0)
    }

    fn element_index(&mut self) -> W {
        let (x, y, z, _) = self.coords.expect("entry point");
        let w = self.ctx_load(Val::I32, offsets::WIDTH);
        let h = self.ctx_load(Val::I32, offsets::HEIGHT);
        let zh = self.bin(z, h, I32_MUL, Val::I32);
        let row = self.bin(zh, y, I32_ADD, Val::I32);
        let base = self.bin(row, w, I32_MUL, Val::I32);
        self.bin(base, x, I32_ADD, Val::I32)
    }

    fn view(&mut self, input: u32) -> W {
        let views = self.ctx_load(Val::I32, offsets::VIEWS);
        self.add_imm(views, (offsets::VIEW_SIZE * i64::from(input)) as i32)
    }

    fn settle_views(&mut self, y: W, z: W) {
        let inputs = self.module.manifest.inputs.len();
        let mut settled = Vec::with_capacity(inputs);
        for input in 0..inputs as u32 {
            let view = self.view(input);
            let words = self.load(Val::I32, view, offsets::VIEW_WORDS as u32);
            let width = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
            let height = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
            let depth = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
            let element = self.load(Val::I32, view, offsets::VIEW_BYTES as u32);
            let last_x = self.add_imm(width, -1);
            let last_y = self.add_imm(height, -1);
            let last_z = self.add_imm(depth, -1);
            let cy = self.umin(y, last_y);
            let cz = self.umin(z, last_z);
            let layer = self.bin(cz, height, I32_MUL, Val::I32);
            let row = self.bin(layer, cy, I32_ADD, Val::I32);
            let row = self.bin(row, width, I32_MUL, Val::I32);
            let rows = self.bin(width, height, I32_MUL, Val::I32);
            let count = self.bin(rows, depth, I32_MUL, Val::I32);
            let last = self.add_imm(count, -1);
            settled.push(Some(Settled {
                words,
                element,
                last_x,
                row,
                last,
            }));
        }
        self.settled = settled;
    }

    fn settled_view(&self, input: u32) -> Option<Settled> {
        self.settled.get(input as usize).copied().flatten()
    }

    /// The view's words and the bytes an element takes.
    fn words_and_element(&mut self, input: u32) -> (W, W) {
        match self.settled_view(input) {
            Some(view) => (view.words, view.element),
            None => {
                let view = self.view(input);
                let words = self.load(Val::I32, view, offsets::VIEW_WORDS as u32);
                let bytes = self.load(Val::I32, view, offsets::VIEW_BYTES as u32);
                (words, bytes)
            }
        }
    }

    fn element_words(&mut self, input: u32, index: W, ty: Type, tys: &[Val]) -> Vec<W> {
        let (words, element) = self.words_and_element(input);
        let bytes = self.bin(index, element, I32_MUL, Val::I32);
        let at = self.bin(words, bytes, I32_ADD, Val::I32);
        self.field_loads(input, at, ty, tys)
    }

    fn field_loads(&mut self, input: u32, base: W, ty: Type, tys: &[Val]) -> Vec<W> {
        let schema = &self.module.manifest.inputs[input as usize].ty.schema;
        let encoding = schema.uniform_encoding().unwrap_or(Encoding::Word);
        if encoding == Encoding::Word {
            return self.loads(tys, base, 0);
        }
        let kinds: Vec<Scalar> = entelechy_ir::schema_of(ty, &self.module.structs)
            .map(|schema| schema.scalars().to_vec())
            .unwrap_or_default();
        let size = encoding.bytes() as u32;
        (0..tys.len())
            .map(|k| {
                let word = self.narrow_load(encoding, base, k as u32 * size);
                if kinds.get(k) == Some(&Scalar::Bool) {
                    self.compare_imm(word, 0, [I32_NE, 0, 0, 0])
                } else {
                    word
                }
            })
            .collect()
    }

    fn narrow_load(&mut self, encoding: Encoding, base: W, offset: u32) -> W {
        let raw = |fx: &mut Self, op: u8| {
            fx.get(base);
            fx.byte(op);
            fx.uleb(0);
            fx.uleb(offset);
            fx.take(Val::I32)
        };
        let unit = |fx: &mut Self, raw: W, signed: bool, scale: f32| {
            let op = if signed {
                F32_CONVERT_I32_S
            } else {
                F32_CONVERT_I32_U
            };
            let x = fx.un(raw, op, Val::F32);
            let scale = fx.f32c(scale);
            fx.bin(x, scale, 0x94, Val::F32)
        };
        match encoding {
            Encoding::U8 => raw(self, 0x2d),
            Encoding::I8 => raw(self, 0x2c),
            Encoding::U16 => raw(self, 0x2f),
            Encoding::I16 => raw(self, 0x2e),
            Encoding::Unorm8 => {
                let r = raw(self, 0x2d);
                unit(self, r, false, encoding::UNORM8)
            }
            Encoding::Unorm16 => {
                let r = raw(self, 0x2f);
                unit(self, r, false, encoding::UNORM16)
            }
            Encoding::Snorm8 => {
                let r = raw(self, 0x2c);
                let x = unit(self, r, true, encoding::SNORM8);
                let floor = self.f32c(-1.0);
                self.bin(x, floor, 0x97, Val::F32)
            }
            Encoding::F16 => {
                let r = raw(self, 0x2f);
                let bits = self
                    .helper("entelechy_jit_half", &[r], Some(Val::I32))
                    .expect("a result");
                self.un(bits, F32_REINTERPRET_I32, Val::F32)
            }
            Encoding::Word => self.load(Val::I32, base, offset),
        }
    }

    fn part_offset(
        &mut self,
        input: u32,
        path: &[entelechy_ir::Step],
        words: usize,
        unit: i32,
    ) -> W {
        let element = self.module.manifest.inputs[input as usize].element();
        let layout = entelechy_ir::part_layout(&self.module.structs, element, path)
            .expect("a path validation walked");
        let mut offset = self.i32c(layout.base as i32);
        for (index, count, stride) in layout.steps {
            let at = self.words(&index)[0];
            let zero = self.i32c(0);
            let last = self.i32c(count as i32 - 1);
            let floored = self.smax(at, zero);
            let clamped = self.smin(floored, last);
            let scaled = self.mul_imm(clamped, stride as i32);
            offset = self.bin(offset, scaled, I32_ADD, Val::I32);
        }
        let stride = self.module.manifest.inputs[input as usize]
            .ty
            .schema
            .value_words();
        let last = self.i32c(stride.saturating_sub(words) as i32);
        let held = self.smin(offset, last);
        let zero = self.i32c(0);
        let inside = self.smax(held, zero);
        self.mul_imm(inside, unit)
    }

    fn sample_index(&mut self, input: u32, x: W, y: W, z: W) -> W {
        if let Some(view) = self.settled_view(input) {
            let cx = self.umin(x, view.last_x);
            return self.bin(view.row, cx, I32_ADD, Val::I32);
        }
        let view = self.view(input);
        let width = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
        let height = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
        let depth = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
        let last = |fx: &mut Self, size: W, at: W| {
            let end = fx.add_imm(size, -1);
            fx.umin(at, end)
        };
        let cx = last(self, width, x);
        let cy = last(self, height, y);
        let cz = last(self, depth, z);
        let layer = self.bin(cz, height, I32_MUL, Val::I32);
        let row = self.bin(layer, cy, I32_ADD, Val::I32);
        let scaled = self.bin(row, width, I32_MUL, Val::I32);
        self.bin(scaled, cx, I32_ADD, Val::I32)
    }

    fn rows_index(&mut self, input: u32, address: W) -> W {
        let last = match self.settled_view(input) {
            Some(view) => view.last,
            None => {
                let view = self.view(input);
                let width = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
                let height = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
                let depth = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
                let rows = self.bin(width, height, I32_MUL, Val::I32);
                let count = self.bin(rows, depth, I32_MUL, Val::I32);
                self.add_imm(count, -1)
            }
        };
        let zero = self.i32c(0);
        let floored = self.smax(address, zero);
        self.smin(floored, last)
    }

    /// A float address into a grid or volume, each axis floored and
    /// clamped into the Field, NaN as zero: the native code's rule.
    fn float_index(&mut self, input: u32, x: W, y: W, z: W) -> W {
        let view = self.view(input);
        let width = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
        let height = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
        let depth = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
        let axis = |fx: &mut Self, v: W, limit: W| {
            let zero = fx.f32c(0.0);
            let nan = fx.is_nan(v);
            let v = fx.select(nan, zero, v);
            let floored = fx.un(v, F32_FLOOR, Val::F32);
            // i32.trunc_sat_f32_u: below zero is zero, past the largest the
            // largest; then clamped as an integer.
            fx.get(floored);
            fx.byte(0xfc);
            fx.uleb(1);
            let whole = fx.take(Val::I32);
            let last = fx.add_imm(limit, -1);
            fx.umin(whole, last)
        };
        let cx = axis(self, x, width);
        let cy = axis(self, y, height);
        let cz = axis(self, z, depth);
        let layer = self.bin(cz, height, I32_MUL, Val::I32);
        let row = self.bin(layer, cy, I32_ADD, Val::I32);
        let scaled = self.bin(row, width, I32_MUL, Val::I32);
        self.bin(scaled, cx, I32_ADD, Val::I32)
    }

    /// The element `at` reads at `address`: an `Int` over rows, a `Vec2`
    /// or `Vec3` over a grid or volume.
    fn address_index(&mut self, input: u32, address: &entelechy_ir::ValueId) -> W {
        let words = self.words(address);
        match self.ty(address) {
            Type::Int => self.rows_index(input, words[0]),
            Type::Vec2 => {
                let zero = self.f32c(0.0);
                self.float_index(input, words[0], words[1], zero)
            }
            _ => self.float_index(input, words[0], words[1], words[2]),
        }
    }

    /// Reads an element through the Host's helper into the scratch.
    fn read_into_scratch(&mut self, helper: &'static str, input: u32, x: W, y: W, z: W) -> W {
        let out = self.scratch_addr();
        let index = self.i32c(input as i32);
        let ctx = self.ctx;
        self.helper(helper, &[ctx, index, x, y, z, out], None);
        out
    }

    // --- Instructions ---

    fn instruction(&mut self, instruction: &Instruction) -> Vec<W> {
        let ty = instruction.ty;
        let id = instruction.id;
        match &instruction.op {
            Op::Const(value) => value
                .numbers()
                .into_iter()
                .map(|number| match number {
                    entelechy_ir::Number::Float(v) => self.f32c(v),
                    entelechy_ir::Number::Int(v) => self.i32c(v),
                    entelechy_ir::Number::Bool(v) => self.i32c(i32::from(v)),
                    entelechy_ir::Number::Double(v) => self.f64c(v),
                    entelechy_ir::Number::Long(v) => self.i64c(v),
                })
                .collect(),
            Op::Param(index) => {
                let offset: u32 = self.module.manifest.params[..*index as usize]
                    .iter()
                    .map(|p| byte_size(&word_types(self.module, p.ty)))
                    .sum();
                let base = self.ctx_load(Val::I32, offsets::PARAMS);
                let tys = word_types(self.module, ty);
                self.loads(&tys, base, offset)
            }
            Op::Arg(index) => self.args[*index as usize].clone(),
            Op::Unary(UnaryOp::Neg, a) => {
                let words = self.words(a);
                if self.ty(a).is_int_kind() {
                    words
                        .iter()
                        .map(|w| {
                            let zero = self.iconst(*w, 0);
                            self.checked(zero, *w, BinaryOp::Sub)
                        })
                        .collect()
                } else {
                    words
                        .iter()
                        .map(|w| self.un(*w, by_type(w.1, [0, 0, 0x8c, 0x9a]), w.1))
                        .collect()
                }
            }
            Op::Unary(UnaryOp::Not, a) => {
                let word = self.words(a)[0];
                vec![self.un(word, I32_EQZ, Val::I32)]
            }
            Op::Binary(op, a, b) => self.binary(*op, a, b),
            Op::Construct(parts) => {
                if ty.components().is_none() {
                    parts.iter().flat_map(|part| self.words(part)).collect()
                } else {
                    parts
                        .iter()
                        .map(|part| {
                            let word = self.words(part)[0];
                            match self.ty(part) {
                                Type::Int | Type::Bool => {
                                    self.un(word, F32_CONVERT_I32_S, Val::F32)
                                }
                                _ => word,
                            }
                        })
                        .collect()
                }
            }
            Op::Extract(source, component) => {
                let words = self.words(source);
                match self.ty(source) {
                    Type::Struct(index) => {
                        let start = field_offset(self.module, index, *component as usize);
                        let n = word_types(self.module, ty).len();
                        words[start..start + n].to_vec()
                    }
                    other if other.order().is_some() => {
                        let n = usize::from(other.order().unwrap_or(1));
                        let start = *component as usize * n;
                        words[start..start + n].to_vec()
                    }
                    _ => vec![words[*component as usize]],
                }
            }
            Op::Subscript(array, index) => {
                let count = match self.ty(array) {
                    Type::Struct(struct_index) => {
                        self.module.structs[struct_index as usize].fields.len()
                    }
                    other => other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from),
                };
                let tys = word_types(self.module, ty);
                let (base, readonly) = if let Some(pointer) = self.pointers.get(&array.0) {
                    (*pointer, true)
                } else if self.constants.contains_key(&array.0)
                    && matches!(self.ty(array), Type::Struct(_))
                {
                    (self.table(array.0), true)
                } else {
                    // Stored in its own slot and indexed there.
                    let words = self.words(array);
                    let array_types = word_types(self.module, self.ty(array));
                    let offset = self.shared.reserve(byte_size(&array_types));
                    let address = self.slot(offset);
                    self.stores(&words, address);
                    (address, false)
                };
                let raw = self.words(index)[0];
                let zero = self.i32c(0);
                let last = self.i32c(count as i32 - 1);
                let low = self.smax(raw, zero);
                let clamped = self.smin(low, last);
                let scaled = self.mul_imm(clamped, byte_size(&tys) as i32);
                let address = self.bin(base, scaled, I32_ADD, Val::I32);
                if readonly
                    && matches!(ty, Type::Struct(index) if self.module.structs[index as usize].is_array())
                {
                    self.pointers.insert(id.0, address);
                }
                self.loads(&tys, address, 0)
            }
            Op::Call(callee, args) => {
                let tys = word_types(self.module, ty);
                let offset = self.shared.reserve(byte_size(&tys));
                let out = self.slot(offset);
                self.store_count();
                let ctx = self.ctx;
                self.get(ctx);
                self.get(out);
                for arg in args {
                    for word in self.words(arg) {
                        self.get(word);
                    }
                }
                self.byte(CALL);
                self.uleb(*callee);
                self.load_count();
                self.loads(&tys, out, 0)
            }
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args),
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let cond = self.words(condition)[0];
                let results: Vec<W> = word_types(self.module, ty)
                    .into_iter()
                    .map(|t| self.local(t))
                    .collect();
                self.get(cond);
                self.byte(IF);
                self.byte(0x40);
                self.block(&then.body);
                let words = self.words(&then.result);
                self.assign(&results, &words);
                self.byte(ELSE);
                self.block(&otherwise.body);
                let words = self.words(&otherwise.result);
                self.assign(&results, &words);
                self.byte(END);
                results
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => self.match_(ty, *scrutinee, arms, otherwise),
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => self.loop_(ty, id, *count, *bound, *init, until.as_ref(), body),
            Op::Each { count, body } => self.each(ty, *count, body),
            Op::Carried => self
                .loops
                .last()
                .or(self.scan.as_ref())
                .expect("a loop or a scan")
                .0
                .clone(),
            Op::Index => vec![
                self.loops
                    .last()
                    .or(self.scan.as_ref())
                    .expect("a loop or a scan")
                    .1,
            ],
            Op::Coord => {
                let (x, y, z, _) = self.coords.expect("entry point");
                let mut words = vec![self.u32_to_f32(x), self.u32_to_f32(y)];
                if ty == Type::Vec3 {
                    words.push(self.u32_to_f32(z));
                }
                words
            }
            Op::Domain => {
                let w = self.ctx_load(Val::I32, offsets::WIDTH);
                let h = self.ctx_load(Val::I32, offsets::HEIGHT);
                let d = self.ctx_load(Val::I32, offsets::DEPTH);
                match ty {
                    Type::Int => vec![w],
                    Type::Vec2 => vec![self.u32_to_f32(w), self.u32_to_f32(h)],
                    _ => vec![self.u32_to_f32(w), self.u32_to_f32(h), self.u32_to_f32(d)],
                }
            }
            Op::Element => vec![self.element_index()],
            Op::Time => vec![self.ctx_load(Val::F32, offsets::TIME)],
            Op::Clock => vec![self.ctx_load(Val::F64, offsets::CLOCK)],
            Op::Rate => vec![self.ctx_load(Val::F32, offsets::RATE)],
            Op::Queued => vec![self.ctx_load(Val::I32, offsets::QUEUED)],
            Op::Pointer => vec![
                self.ctx_load(Val::F32, offsets::POINTER),
                self.ctx_load(Val::F32, offsets::POINTER + 4),
            ],
            Op::Stick(which) => {
                let which = self.words(which)[0];
                let zero = self.i32c(0);
                let last = self.i32c(2);
                let floored = self.smax(which, zero);
                let held = self.smin(floored, last);
                let off = self.mul_imm(held, 8);
                let ctx = self.ctx;
                let base = self.add_imm(ctx, offsets::STICKS);
                let at = self.bin(base, off, I32_ADD, Val::I32);
                let inside = self.compare(which, held, [I32_EQ, 0, 0, 0]);
                let x = self.load(Val::F32, at, 0);
                let y = self.load(Val::F32, at, 4);
                let none = self.f32c(0.0);
                vec![self.select(inside, x, none), self.select(inside, y, none)]
            }
            Op::Wheel => vec![
                self.ctx_load(Val::F32, offsets::WHEEL),
                self.ctx_load(Val::F32, offsets::WHEEL + 4),
            ],
            Op::Pads => vec![self.ctx_load(Val::I32, offsets::PADS)],
            Op::Window => vec![
                self.ctx_load(Val::F32, offsets::WINDOW),
                self.ctx_load(Val::F32, offsets::WINDOW + 4),
            ],
            Op::Scale => vec![self.ctx_load(Val::F32, offsets::SCALE)],
            Op::PointerDown => {
                let zero = self.i32c(0);
                let ctx = self.ctx;
                vec![
                    self.helper("entelechy_jit_button", &[ctx, zero], Some(Val::I32))
                        .expect("a result"),
                ]
            }
            Op::Button(code) => {
                let code = self.words(code)[0];
                let ctx = self.ctx;
                vec![
                    self.helper("entelechy_jit_button", &[ctx, code], Some(Val::I32))
                        .expect("a result"),
                ]
            }
            Op::Key(code) => {
                let code = self.words(code)[0];
                let ctx = self.ctx;
                vec![
                    self.helper("entelechy_jit_key", &[ctx, code], Some(Val::I32))
                        .expect("a result"),
                ]
            }
            Op::Typed(k) => {
                let k = self.words(k)[0];
                let ctx = self.ctx;
                vec![
                    self.helper("entelechy_jit_typed", &[ctx, k], Some(Val::I32))
                        .expect("a result"),
                ]
            }
            Op::Count(input) => {
                let view = self.view(*input);
                let w = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
                let h = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
                let d = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
                let wh = self.bin(w, h, I32_MUL, Val::I32);
                vec![self.bin(wh, d, I32_MUL, Val::I32)]
            }
            Op::Size(input) => {
                let view = self.view(*input);
                let w = self.load(Val::I32, view, offsets::VIEW_WIDTH as u32);
                let h = self.load(Val::I32, view, offsets::VIEW_HEIGHT as u32);
                let d = self.load(Val::I32, view, offsets::VIEW_DEPTH as u32);
                match ty {
                    Type::Int => vec![w],
                    Type::Vec2 => vec![self.u32_to_f32(w), self.u32_to_f32(h)],
                    _ => vec![self.u32_to_f32(w), self.u32_to_f32(h), self.u32_to_f32(d)],
                }
            }
            Op::Sample(input) => {
                let tys = word_types(self.module, ty);
                let (x, y, z, _) = self.coords.expect("entry point");
                let index = self.sample_index(*input, x, y, z);
                self.element_words(*input, index, ty, &tys)
            }
            Op::At(input, address) => {
                let at = self.address_index(*input, address);
                let tys = word_types(self.module, ty);
                self.element_words(*input, at, ty, &tys)
            }
            Op::Part { input, at, path } => {
                let tys = word_types(self.module, ty);
                let unit = self.module.manifest.inputs[*input as usize]
                    .ty
                    .schema
                    .uniform_encoding()
                    .map_or(4, |encoding| encoding.bytes() as i32);
                let Some(at) = at else {
                    let within = self.part_offset(*input, path, byte_size(&tys) as usize / 4, unit);
                    let (x, y, z, _) = self.coords.expect("entry point");
                    let index = self.sample_index(*input, x, y, z);
                    return self.part_words(*input, index, within, ty, &tys);
                };
                let within = self.part_offset(*input, path, byte_size(&tys) as usize / 4, unit);
                let index = self.address_index(*input, at);
                self.part_words(*input, index, within, ty, &tys)
            }
            Op::AtLinear(input, position) => {
                let words = self.words(position);
                let half = self.f32c(0.5);
                let (x, y, z) = match self.ty(position) {
                    Type::Float => (words[0], half, half),
                    Type::Vec2 => (words[0], words[1], half),
                    _ => (words[0], words[1], words[2]),
                };
                self.read_into_scratch("entelechy_jit_linear", *input, x, y, z);
                let tys = word_types(self.module, ty);
                self.load_scratch(&tys)
            }
            Op::Spend(counted) => {
                let ran = match self.ran.get(&counted.0) {
                    Some(ran) => *ran,
                    None => self.i32c(0),
                };
                vec![self.spend(ran)]
            }
            Op::Look => vec![self.look()],
            Op::Emit(pair) => {
                let words = self.words(pair);
                let slot = self.scratch_addr();
                self.stores(&words, slot);
                let element = self.element_index();
                let bytes: u32 = words.iter().map(|word| word.1.bytes()).sum();
                let n = self.i32c((bytes / 4) as i32);
                let ctx = self.ctx;
                self.helper("entelechy_jit_emit", &[ctx, element, slot, n], None);
                words
            }
            Op::Lane => vec![self.running.expect("a lane is read in a group's lanes")],
            Op::Exchange(..) => unreachable!("a group's entry runs its lanes, its exchanges taken"),
            Op::Lanes { body } => self.lanes(ty, body),
            Op::Meet(_) => self.meet(),
            Op::Put(input, address, value) => {
                let words = self.words(value);
                let slot = self.scratch_addr();
                self.stores(&words, slot);
                let index = self.i32c(*input as i32);
                let address_words = self.words(address);
                let ctx = self.ctx;
                match self.ty(address) {
                    Type::Int => {
                        self.helper(
                            "entelechy_jit_put_rows",
                            &[ctx, index, address_words[0], slot],
                            None,
                        );
                    }
                    ty => {
                        let zero = self.f32c(0.0);
                        let (x, y, z) = match ty {
                            Type::Vec2 => (address_words[0], address_words[1], zero),
                            _ => (address_words[0], address_words[1], address_words[2]),
                        };
                        self.helper("entelechy_jit_put", &[ctx, index, x, y, z, slot], None);
                    }
                }
                words
            }
        }
    }

    fn part_words(&mut self, input: u32, index: W, within: W, ty: Type, tys: &[Val]) -> Vec<W> {
        let (words, element) = self.words_and_element(input);
        let bytes = self.bin(index, element, I32_MUL, Val::I32);
        let total = self.bin(bytes, within, I32_ADD, Val::I32);
        let at = self.bin(words, total, I32_ADD, Val::I32);
        self.field_loads(input, at, ty, tys)
    }

    /// `targets` take `sources`' words at once, since a loop's next words
    /// may permute its carried ones.
    fn assign(&mut self, targets: &[W], sources: &[W]) {
        for source in sources {
            self.get(*source);
        }
        for target in targets.iter().rev() {
            self.set(*target);
        }
    }

    fn match_(
        &mut self,
        ty: Type,
        scrutinee: entelechy_ir::ValueId,
        arms: &[(Vec<i32>, entelechy_ir::Block)],
        otherwise: &entelechy_ir::Block,
    ) -> Vec<W> {
        let value = self.words(&scrutinee)[0];
        let results: Vec<W> = word_types(self.module, ty)
            .into_iter()
            .map(|t| self.local(t))
            .collect();
        let n = arms.len() as u32;
        // block $join { block $default { block $arm(n-1) { ... block $arm0 {
        //   dispatch } arm0 } arm1 ... } default }
        self.byte(BLOCK);
        self.byte(0x40);
        self.byte(BLOCK);
        self.byte(0x40);
        for _ in 0..n {
            self.byte(BLOCK);
            self.byte(0x40);
        }
        let all: Vec<(i32, u32)> = arms
            .iter()
            .enumerate()
            .flat_map(|(index, (values, _))| values.iter().map(move |v| (*v, index as u32)))
            .collect();
        let bounds = all.iter().fold(None, |bounds: Option<(i32, i32)>, (v, _)| {
            Some(bounds.map_or((*v, *v), |(lo, hi)| (lo.min(*v), hi.max(*v))))
        });
        let dense = bounds
            .is_some_and(|(lo, hi)| (hi as i64 - lo as i64 + 1) as usize <= all.len() * 4 + 16);
        // From inside the innermost block, arm k is `k` out and the
        // default `n`.
        match (dense, bounds) {
            (true, Some((lo, hi))) => {
                let span = (hi as i64 - lo as i64 + 1) as usize;
                let mut targets = vec![n; span];
                for (v, index) in &all {
                    targets[(*v as i64 - lo as i64) as usize] = *index;
                }
                self.get(value);
                if lo != 0 {
                    self.push_i32(lo);
                    self.byte(I32_SUB);
                }
                self.byte(BR_TABLE);
                self.uleb(span as u32);
                for target in targets {
                    self.uleb(target);
                }
                self.uleb(n);
            }
            _ => {
                for (v, index) in &all {
                    self.get(value);
                    self.push_i32(*v);
                    self.byte(I32_EQ);
                    self.byte(BR_IF);
                    self.uleb(*index);
                }
                self.byte(BR);
                self.uleb(n);
            }
        }
        for (k, (_, arm)) in arms.iter().enumerate() {
            self.byte(END);
            self.block(&arm.body);
            let words = self.words(&arm.result);
            self.assign(&results, &words);
            // Out past the remaining arms, the default, to the join.
            self.byte(BR);
            self.uleb(n - k as u32);
        }
        self.byte(END);
        self.block(&otherwise.body);
        let words = self.words(&otherwise.result);
        self.assign(&results, &words);
        self.byte(END);
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn loop_(
        &mut self,
        ty: Type,
        id: entelechy_ir::ValueId,
        count: u32,
        bound: Option<entelechy_ir::ValueId>,
        init: entelechy_ir::ValueId,
        until: Option<&entelechy_ir::Block>,
        body: &entelechy_ir::Block,
    ) -> Vec<W> {
        let limit = match bound {
            Some(bound) => {
                let given = self.words(&bound)[0];
                let zero = self.i32c(0);
                let most = self.i32c(count as i32);
                let floored = self.smax(given, zero);
                self.smin(floored, most)
            }
            None => self.i32c(count as i32),
        };
        let carried: Vec<W> = word_types(self.module, ty)
            .into_iter()
            .map(|t| self.local(t))
            .collect();
        let index = self.local(Val::I32);
        let ran = self.local(Val::I32);
        let init_words = self.words(&init);
        self.assign(&carried, &init_words);
        self.push_i32(0);
        self.set(index);
        self.push_i32(0);
        self.set(ran);
        // block $exit { [if limit > 0] loop $header { ... } }
        self.byte(BLOCK);
        self.byte(0x40);
        if bound.is_some() {
            // A run-time count of zero yields the carried values.
            self.get(limit);
            self.push_i32(0);
            self.byte(0x4c); // i32.le_s
            self.byte(BR_IF);
            self.uleb(0);
        }
        self.byte(LOOP);
        self.byte(0x40);
        self.loops.push((carried.clone(), index));
        if let Some(until) = until {
            self.block(&until.body);
            let done = self.words(&until.result)[0];
            self.get(index);
            self.set(ran);
            self.get(done);
            self.byte(BR_IF);
            self.uleb(1);
        }
        self.block(&body.body);
        let next = self.words(&body.result);
        self.assign(&carried, &next);
        self.get(index);
        self.push_i32(1);
        self.byte(I32_ADD);
        self.set(index);
        self.get(index);
        self.set(ran);
        self.get(index);
        self.get(limit);
        self.byte(I32_LT_U);
        self.byte(BR_IF);
        self.uleb(0);
        self.loops.pop();
        self.byte(END);
        self.byte(END);
        if self.work.is_some() {
            self.ran.insert(id.0, ran);
        }
        carried
    }

    /// An array built in its own slot: each run of the body stores its
    /// element at its position, and the words are read back once all are
    /// made. Its position is the innermost index; what it reads as
    /// carried is the enclosing loop's (or the scan's).
    fn each(&mut self, ty: Type, count: u32, body: &entelechy_ir::Block) -> Vec<W> {
        let array_types = word_types(self.module, ty);
        let element_types = array_types[..array_types.len() / count as usize].to_vec();
        let stride = byte_size(&element_types) as i32;
        let offset = self.shared.reserve(byte_size(&array_types));
        let base = self.slot(offset);
        let index = self.local(Val::I32);
        self.push_i32(0);
        self.set(index);
        let carried = self
            .loops
            .last()
            .or(self.scan.as_ref())
            .map(|(carried, _)| carried.clone())
            .unwrap_or_default();
        // loop $each { ... }
        self.byte(LOOP);
        self.byte(0x40);
        self.loops.push((carried, index));
        self.block(&body.body);
        let element = self.words(&body.result);
        let scaled = self.mul_imm(index, stride);
        let at = self.bin(base, scaled, I32_ADD, Val::I32);
        self.stores(&element, at);
        self.get(index);
        self.push_i32(1);
        self.byte(I32_ADD);
        self.set(index);
        self.get(index);
        self.push_i32(count as i32);
        self.byte(I32_LT_U);
        self.byte(BR_IF);
        self.uleb(0);
        self.loops.pop();
        self.byte(END);
        self.loads(&array_types, base, 0)
    }

    /// Checked `x op y` on integers: the result wraps; overflow is
    /// recorded.
    fn checked(&mut self, x: W, y: W, op: BinaryOp) -> W {
        match x.1 {
            Val::I32 => {
                // In 64 bits: overflow when the 32-bit result, widened,
                // differs from the 64-bit one.
                let a = self.un(x, I64_EXTEND_I32_S, Val::I64);
                let b = self.un(y, I64_EXTEND_I32_S, Val::I64);
                let wide = match op {
                    BinaryOp::Add => self.bin(a, b, I64_ADD, Val::I64),
                    BinaryOp::Sub => self.bin(a, b, 0x7d, Val::I64),
                    _ => self.bin(a, b, I64_MUL, Val::I64),
                };
                let narrow = self.un(wide, I32_WRAP_I64, Val::I32);
                let back = self.un(narrow, I64_EXTEND_I32_S, Val::I64);
                let overflow = self.bin(back, wide, 0x52, Val::I32);
                self.fail_if(overflow, OVERFLOW);
                narrow
            }
            _ => match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let r = if op == BinaryOp::Add {
                        self.bin(x, y, I64_ADD, Val::I64)
                    } else {
                        self.bin(x, y, 0x7d, Val::I64)
                    };
                    // Add: both operands differ in sign from the result.
                    // Sub: the operands differ in sign, and the result
                    // differs from the first.
                    let (p, q) = if op == BinaryOp::Add {
                        let p = self.bin(x, r, 0x85, Val::I64);
                        let q = self.bin(y, r, 0x85, Val::I64);
                        (p, q)
                    } else {
                        let p = self.bin(x, y, 0x85, Val::I64);
                        let q = self.bin(x, r, 0x85, Val::I64);
                        (p, q)
                    };
                    let both = self.bin(p, q, 0x83, Val::I64);
                    let overflow = self.compare_imm(both, 0, [0, 0x53, 0, 0]);
                    self.fail_if(overflow, OVERFLOW);
                    r
                }
                _ => {
                    // No wider type: the product divided by `x` is `y`
                    // unless it wrapped; MIN * -1 is checked apart.
                    let r = self.bin(x, y, I64_MUL, Val::I64);
                    let zero = self.compare_imm(x, 0, [0, 0x51, 0, 0]);
                    let minus_one = self.compare_imm(x, -1, [0, 0x51, 0, 0]);
                    let unsafe_ = self.bin(zero, minus_one, I32_OR, Val::I32);
                    let one = self.i64c(1);
                    let divisor = self.select(unsafe_, one, x);
                    let back = self.bin(r, divisor, 0x7f, Val::I64);
                    let differs = self.bin(back, y, 0x52, Val::I32);
                    let y_min = self.compare_imm(y, i64::MIN, [0, 0x51, 0, 0]);
                    let no = self.i32c(0);
                    let by_division = self.select(zero, no, differs);
                    let overflow = self.select(minus_one, y_min, by_division);
                    self.fail_if(overflow, OVERFLOW);
                    r
                }
            },
        }
    }

    fn zip2(&mut self, a: &[W], b: &[W], mut f: impl FnMut(&mut Self, W, W) -> W) -> Vec<W> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|k| {
                let x = if a.len() == 1 { a[0] } else { a[k] };
                let y = if b.len() == 1 { b[0] } else { b[k] };
                f(self, x, y)
            })
            .collect()
    }

    fn zip3(
        &mut self,
        a: &[W],
        b: &[W],
        c: &[W],
        mut f: impl FnMut(&mut Self, W, W, W) -> W,
    ) -> Vec<W> {
        let n = a.len().max(b.len()).max(c.len());
        (0..n)
            .map(|k| {
                let x = if a.len() == 1 { a[0] } else { a[k] };
                let y = if b.len() == 1 { b[0] } else { b[k] };
                let z = if c.len() == 1 { c[0] } else { c[k] };
                f(self, x, y, z)
            })
            .collect()
    }

    fn is_nan(&mut self, v: W) -> W {
        self.compare(v, v, [0, 0, 0x5c, 0x62])
    }

    /// `m`, unless `a` or `b` is NaN: then the other.
    fn nan_gives_way(&mut self, a: W, b: W, m: W) -> W {
        let b_nan = self.is_nan(b);
        let m = self.select(b_nan, a, m);
        let a_nan = self.is_nan(a);
        self.select(a_nan, b, m)
    }

    fn fmin(&mut self, a: W, b: W) -> W {
        let m = self.arith(a, b, [0, 0, 0x96, 0xa4]);
        self.nan_gives_way(a, b, m)
    }

    fn fmax(&mut self, a: W, b: W) -> W {
        let m = self.arith(a, b, [0, 0, 0x97, 0xa5]);
        self.nan_gives_way(a, b, m)
    }

    /// A scalar of kind `from` as a word of `to`: a float truncates
    /// toward zero and saturates (a NaN is zero), a long wraps to an int.
    fn convert(&mut self, v: W, from: Type, to: Val) -> W {
        let float = matches!(from, Type::Float | Type::Double);
        let to_float = matches!(to, Val::F32 | Val::F64);
        if v.1 == to {
            return v;
        }
        match (float, to_float) {
            (true, true) if to == Val::F64 => self.un(v, 0xbb, to),
            (true, true) => self.un(v, 0xb6, to),
            (true, false) => {
                let sub = match (v.1, to) {
                    (Val::F32, Val::I32) => 0,
                    (Val::F64, Val::I32) => 2,
                    (Val::F32, _) => 4,
                    _ => 6,
                };
                self.get(v);
                self.byte(0xfc);
                self.uleb(sub);
                self.take(to)
            }
            (false, true) => {
                let op = match (v.1, to) {
                    (Val::I32, Val::F32) => F32_CONVERT_I32_S,
                    (Val::I64, Val::F32) => 0xb4,
                    (Val::I32, _) => 0xb7,
                    _ => 0xb9,
                };
                self.un(v, op, to)
            }
            (false, false) if to == Val::I64 => self.un(v, I64_EXTEND_I32_S, to),
            (false, false) => self.un(v, I32_WRAP_I64, to),
        }
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        a: &entelechy_ir::ValueId,
        b: &entelechy_ir::ValueId,
    ) -> Vec<W> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        let (wa, wb) = (self.words(a), self.words(b));
        let int_like = ta.is_int_kind() || tb.is_int_kind();
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul if int_like => {
                self.zip2(&wa, &wb, |fx, x, y| fx.checked(x, y, op))
            }
            BinaryOp::Div | BinaryOp::Rem if int_like => self.zip2(&wa, &wb, |fx, x, y| {
                let zero = fx.compare_imm(y, 0, [I32_EQ, 0x51, 0, 0]);
                fx.fail_if(zero, DIVISION_BY_ZERO);
                let min = if y.1 == Val::I64 {
                    i64::MIN
                } else {
                    i64::from(i32::MIN)
                };
                let is_min = fx.compare_imm(x, min, [I32_EQ, 0x51, 0, 0]);
                let minus_one = fx.compare_imm(y, -1, [I32_EQ, 0x51, 0, 0]);
                let overflow = fx.bin(is_min, minus_one, I32_AND, Val::I32);
                fx.fail_if(overflow, OVERFLOW);
                let unsafe_ = fx.bin(zero, overflow, I32_OR, Val::I32);
                let one = fx.iconst(y, 1);
                let divisor = fx.select(unsafe_, one, y);
                if op == BinaryOp::Div {
                    fx.arith(x, divisor, [0x6d, 0x7f, 0, 0])
                } else {
                    fx.arith(x, divisor, [0x6f, 0x81, 0, 0])
                }
            }),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                let ops = match op {
                    BinaryOp::Add => [0, 0, 0x92, 0xa0],
                    BinaryOp::Sub => [0, 0, 0x93, 0xa1],
                    BinaryOp::Mul => [0, 0, 0x94, 0xa2],
                    _ => [0, 0, 0x95, 0xa3],
                };
                self.zip2(&wa, &wb, |fx, x, y| fx.arith(x, y, ops))
            }
            BinaryOp::Rem => self.zip2(&wa, &wb, |fx, x, y| {
                let (name, ty) = if x.1 == Val::F64 {
                    ("entelechy_jit_fmod", Val::F64)
                } else {
                    ("entelechy_jit_fmodf", Val::F32)
                };
                fx.helper(name, &[x, y], Some(ty)).expect("a result")
            }),
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                let ops = match op {
                    BinaryOp::Less => [I32_LT_S, 0x53, 0x5d, 0x63],
                    BinaryOp::LessEqual => [0x4c, 0x57, 0x5f, 0x65],
                    BinaryOp::Greater => [I32_GT_S, 0x55, 0x5e, 0x64],
                    _ => [0x4e, 0x59, 0x60, 0x66],
                };
                vec![self.compare(wa[0], wb[0], ops)]
            }
            BinaryOp::Equal | BinaryOp::NotEqual => {
                let mut all: Option<W> = None;
                for (x, y) in wa.iter().zip(&wb) {
                    let same = self.compare(*x, *y, [I32_EQ, 0x51, 0x5b, 0x61]);
                    all = Some(match all {
                        Some(prior) => self.bin(prior, same, I32_AND, Val::I32),
                        None => same,
                    });
                }
                let equal = all.expect("an operand has words");
                if op == BinaryOp::NotEqual {
                    let one = self.i32c(1);
                    vec![self.bin(equal, one, I32_XOR, Val::I32)]
                } else {
                    vec![equal]
                }
            }
            BinaryOp::And => vec![self.bin(wa[0], wb[0], I32_AND, Val::I32)],
            BinaryOp::Or => vec![self.bin(wa[0], wb[0], I32_OR, Val::I32)],
            BinaryOp::BitAnd => {
                self.zip2(&wa, &wb, |fx, x, y| fx.arith(x, y, [I32_AND, 0x83, 0, 0]))
            }
            BinaryOp::BitOr => self.zip2(&wa, &wb, |fx, x, y| fx.arith(x, y, [I32_OR, 0x84, 0, 0])),
            BinaryOp::BitXor => {
                self.zip2(&wa, &wb, |fx, x, y| fx.arith(x, y, [I32_XOR, 0x85, 0, 0]))
            }
            BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Ushr => self.zip2(&wa, &wb, |fx, x, y| {
                // WebAssembly shifts take the count modulo the width; the
                // count is converted to the value's type.
                let y = if x.1 == Val::I64 && y.1 == Val::I32 {
                    fx.un(y, I64_EXTEND_I32_U, Val::I64)
                } else if x.1 == Val::I32 && y.1 == Val::I64 {
                    fx.un(y, I32_WRAP_I64, Val::I32)
                } else {
                    y
                };
                let ops = match op {
                    BinaryOp::Shl => [0x74, 0x86, 0, 0],
                    BinaryOp::Shr => [0x75, I64_SHR_S, 0, 0],
                    _ => [I32_SHR_U, 0x88, 0, 0],
                };
                fx.arith(x, y, ops)
            }),
        }
    }

    fn intrinsic(&mut self, intrinsic: Intrinsic, args: &[entelechy_ir::ValueId]) -> Vec<W> {
        let w: Vec<Vec<W>> = args.iter().map(|a| self.words(a)).collect();
        let is_int = |fx: &Self, k: usize| fx.ty(&args[k]).is_int_kind();
        let map1 = |fx: &mut Self, f: &mut dyn FnMut(&mut Self, W) -> W| -> Vec<W> {
            w[0].iter().map(|v| f(fx, *v)).collect()
        };
        let float_op = |fx: &mut Self, v: W, ops: [u8; 2]| {
            let op = if v.1 == Val::F64 { ops[1] } else { ops[0] };
            fx.un(v, op, v.1)
        };
        match intrinsic {
            Intrinsic::Floor => map1(self, &mut |fx, v| float_op(fx, v, [0x8e, 0x9c])),
            Intrinsic::Ceil => map1(self, &mut |fx, v| float_op(fx, v, [0x8d, 0x9b])),
            Intrinsic::Round => map1(self, &mut |fx, v| float_op(fx, v, [0x90, 0x9e])),
            Intrinsic::Sqrt => map1(self, &mut |fx, v| float_op(fx, v, [0x91, 0x9f])),
            Intrinsic::Trunc => map1(self, &mut |fx, v| float_op(fx, v, [0x8f, 0x9d])),
            Intrinsic::Fract => map1(self, &mut |fx, v| {
                let f = float_op(fx, v, [0x8e, 0x9c]);
                fx.arith(v, f, [0, 0, 0x93, 0xa1])
            }),
            Intrinsic::Abs if is_int(self, 0) => map1(self, &mut |fx, v| {
                // Wrapping, as Cranelift's `iabs`: MIN stays MIN.
                let (shift, sra, xor, sub) = match v.1 {
                    Val::I64 => (63, I64_SHR_S, 0x85, 0x7d),
                    _ => (31, 0x75, I32_XOR, I32_SUB),
                };
                let s = fx.iconst(v, shift);
                let sign = fx.bin(v, s, sra, v.1);
                let flipped = fx.bin(v, sign, xor, v.1);
                fx.bin(flipped, sign, sub, v.1)
            }),
            Intrinsic::Abs => map1(self, &mut |fx, v| float_op(fx, v, [0x8b, 0x99])),
            Intrinsic::Sign if is_int(self, 0) => map1(self, &mut |fx, v| {
                let pos = fx.compare_imm(v, 0, [I32_GT_S, 0x55, 0, 0]);
                let neg = fx.compare_imm(v, 0, [I32_LT_S, 0x53, 0, 0]);
                let one = fx.iconst(v, 1);
                let minus = fx.iconst(v, -1);
                let zero = fx.iconst(v, 0);
                let low = fx.select(neg, minus, zero);
                fx.select(pos, one, low)
            }),
            Intrinsic::Sign => map1(self, &mut |fx, v| {
                let zero = fx.fconst(v, 0.0);
                let pos = fx.compare(v, zero, [0, 0, 0x5e, 0x64]);
                let neg = fx.compare(v, zero, [0, 0, 0x5d, 0x63]);
                let one = fx.fconst(v, 1.0);
                let minus = fx.fconst(v, -1.0);
                let low = fx.select(neg, minus, zero);
                fx.select(pos, one, low)
            }),
            Intrinsic::Fma => self.zip3(&w[0], &w[1], &w[2], |fx, a, b, c| {
                let name = if a.1 == Val::F64 {
                    "entelechy_jit_fma"
                } else {
                    "entelechy_jit_fmaf"
                };
                fx.helper(name, &[a, b, c], Some(a.1)).expect("a result")
            }),
            Intrinsic::Popcount => {
                let v = w[0][0];
                vec![self.un(v, by_type(v.1, [0x69, 0x7b, 0, 0]), v.1)]
            }
            Intrinsic::Clz => {
                let v = w[0][0];
                vec![self.un(v, by_type(v.1, [0x67, 0x79, 0, 0]), v.1)]
            }
            Intrinsic::Ctz => {
                let v = w[0][0];
                vec![self.un(v, by_type(v.1, [0x68, 0x7a, 0, 0]), v.1)]
            }
            Intrinsic::Rotl | Intrinsic::Rotr => {
                // The count masked to five bits, as in `codegen`.
                let (v, count) = (w[0][0], w[1][0]);
                let count = if count.1 == Val::I64 {
                    self.un(count, I32_WRAP_I64, Val::I32)
                } else {
                    count
                };
                let mask = self.i32c(31);
                let count = self.bin(count, mask, I32_AND, Val::I32);
                let count = if v.1 == Val::I64 {
                    self.un(count, I64_EXTEND_I32_U, Val::I64)
                } else {
                    count
                };
                let ops = if intrinsic == Intrinsic::Rotl {
                    [0x77, 0x89, 0, 0]
                } else {
                    [0x78, 0x8a, 0, 0]
                };
                vec![self.arith(v, count, ops)]
            }
            Intrinsic::Ult => vec![self.compare(w[0][0], w[1][0], [I32_LT_U, 0x54, 0, 0])],
            Intrinsic::Udiv | Intrinsic::Urem => {
                let zero = self.compare_imm(w[1][0], 0, [I32_EQ, 0x51, 0, 0]);
                self.fail_if(zero, DIVISION_BY_ZERO);
                let one = self.iconst(w[1][0], 1);
                let divisor = self.select(zero, one, w[1][0]);
                let ops = if intrinsic == Intrinsic::Udiv {
                    [0x6e, 0x80, 0, 0]
                } else {
                    [0x70, 0x82, 0, 0]
                };
                vec![self.arith(w[0][0], divisor, ops)]
            }
            Intrinsic::Mulhi | Intrinsic::Umulhi => {
                let extend = if intrinsic == Intrinsic::Mulhi {
                    I64_EXTEND_I32_S
                } else {
                    I64_EXTEND_I32_U
                };
                let a = self.un(w[0][0], extend, Val::I64);
                let b = self.un(w[1][0], extend, Val::I64);
                let product = self.bin(a, b, I64_MUL, Val::I64);
                let shift = self.i64c(32);
                let high = self.bin(product, shift, I64_SHR_S, Val::I64);
                vec![self.un(high, I32_WRAP_I64, Val::I32)]
            }
            Intrinsic::Bits => vec![self.un(w[0][0], I32_REINTERPRET_F32, Val::I32)],
            Intrinsic::FromBits => vec![self.un(w[0][0], F32_REINTERPRET_I32, Val::F32)],
            Intrinsic::Sin
            | Intrinsic::Cos
            | Intrinsic::Tan
            | Intrinsic::Asin
            | Intrinsic::Acos
            | Intrinsic::Atan
            | Intrinsic::Exp
            | Intrinsic::Log
            | Intrinsic::Exp2
            | Intrinsic::Log2
            | Intrinsic::Sinh
            | Intrinsic::Cosh
            | Intrinsic::Tanh => {
                let wide = self.ty(&args[0]).is_64bit();
                let name = unary_helper(intrinsic, wide);
                map1(self, &mut |fx, v| {
                    fx.helper(name, &[v], Some(v.1)).expect("a result")
                })
            }
            Intrinsic::Min if is_int(self, 0) => self.zip2(&w[0], &w[1], |fx, x, y| fx.smin(x, y)),
            Intrinsic::Max if is_int(self, 0) => self.zip2(&w[0], &w[1], |fx, x, y| fx.smax(x, y)),
            Intrinsic::AddWrap => vec![self.arith(w[0][0], w[1][0], [I32_ADD, I64_ADD, 0, 0])],
            Intrinsic::SubWrap => vec![self.arith(w[0][0], w[1][0], [I32_SUB, 0x7d, 0, 0])],
            Intrinsic::MulWrap => vec![self.arith(w[0][0], w[1][0], [I32_MUL, I64_MUL, 0, 0])],
            Intrinsic::ToFloat | Intrinsic::ToInt if self.ty(&args[0]).is_64bit() => {
                self.conversion(intrinsic, args, &w[0])
            }
            Intrinsic::ToDouble | Intrinsic::ToLong => self.conversion(intrinsic, args, &w[0]),
            Intrinsic::Min => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmin(x, y)),
            Intrinsic::Max => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmax(x, y)),
            Intrinsic::Clamp if is_int(self, 0) => {
                self.zip3(&w[0], &w[1], &w[2], |fx, v, low, high| {
                    let floored = fx.smax(v, low);
                    fx.smin(floored, high)
                })
            }
            Intrinsic::Clamp => self.zip3(&w[0], &w[1], &w[2], |fx, v, low, high| {
                let floored = fx.fmax(v, low);
                fx.fmin(floored, high)
            }),
            Intrinsic::Mix => self.zip3(&w[0], &w[1], &w[2], |fx, a, b, t| {
                let span = fx.arith(b, a, [0, 0, 0x93, 0xa1]);
                let scaled = fx.arith(span, t, [0, 0, 0x94, 0xa2]);
                fx.arith(a, scaled, [0, 0, 0x92, 0xa0])
            }),
            Intrinsic::ToFloat => map1(self, &mut |fx, v| fx.un(v, F32_CONVERT_I32_S, Val::F32)),
            Intrinsic::ToInt => {
                if matches!(self.ty(&args[0]), Type::Int | Type::Bool) {
                    vec![w[0][0]]
                } else {
                    map1(self, &mut |fx, v| fx.convert(v, Type::Float, Val::I32))
                }
            }
            Intrinsic::Pow => self.zip2(&w[0], &w[1], |fx, x, y| {
                let name = if x.1 == Val::F64 {
                    "entelechy_jit_pow"
                } else {
                    "entelechy_jit_powf"
                };
                fx.helper(name, &[x, y], Some(x.1)).expect("a result")
            }),
            Intrinsic::Step => self.zip2(&w[0], &w[1], |fx, edge, x| {
                let ge = fx.compare(x, edge, [0, 0, 0x60, 0x66]);
                let one = fx.fconst(x, 1.0);
                let zero = fx.fconst(x, 0.0);
                fx.select(ge, one, zero)
            }),
            Intrinsic::Smoothstep => self.zip3(&w[0], &w[1], &w[2], |fx, low, high, x| {
                let num = fx.arith(x, low, [0, 0, 0x93, 0xa1]);
                let den = fx.arith(high, low, [0, 0, 0x93, 0xa1]);
                let t = fx.arith(num, den, [0, 0, 0x95, 0xa3]);
                let zero = fx.fconst(t, 0.0);
                let one = fx.fconst(t, 1.0);
                let t = fx.fmax(t, zero);
                let t = fx.fmin(t, one);
                let two = fx.fconst(t, 2.0);
                let three = fx.fconst(t, 3.0);
                let tt = fx.arith(t, t, [0, 0, 0x94, 0xa2]);
                let twice = fx.arith(two, t, [0, 0, 0x94, 0xa2]);
                let rest = fx.arith(three, twice, [0, 0, 0x93, 0xa1]);
                fx.arith(tt, rest, [0, 0, 0x94, 0xa2])
            }),
            Intrinsic::Atan2 => {
                let (name, ty) = if self.ty(&args[0]) == Type::Double {
                    ("entelechy_jit_atan2", Val::F64)
                } else {
                    ("entelechy_jit_atan2f", Val::F32)
                };
                vec![
                    self.helper(name, &[w[0][0], w[1][0]], Some(ty))
                        .expect("a result"),
                ]
            }
        }
    }

    fn conversion(
        &mut self,
        intrinsic: Intrinsic,
        args: &[entelechy_ir::ValueId],
        words: &[W],
    ) -> Vec<W> {
        let from = self.ty(&args[0]).scalar().unwrap_or(Type::Float);
        let to = match intrinsic {
            Intrinsic::ToFloat => Val::F32,
            Intrinsic::ToInt => Val::I32,
            Intrinsic::ToDouble => Val::F64,
            _ => Val::I64,
        };
        words.iter().map(|v| self.convert(*v, from, to)).collect()
    }
}

fn unary_helper(intrinsic: Intrinsic, wide: bool) -> &'static str {
    let (narrow, wide_name) = match intrinsic {
        Intrinsic::Sin => ("entelechy_jit_sinf", "entelechy_jit_sin"),
        Intrinsic::Cos => ("entelechy_jit_cosf", "entelechy_jit_cos"),
        Intrinsic::Tan => ("entelechy_jit_tanf", "entelechy_jit_tan"),
        Intrinsic::Asin => ("entelechy_jit_asinf", "entelechy_jit_asin"),
        Intrinsic::Acos => ("entelechy_jit_acosf", "entelechy_jit_acos"),
        Intrinsic::Atan => ("entelechy_jit_atanf", "entelechy_jit_atan"),
        Intrinsic::Exp => ("entelechy_jit_expf", "entelechy_jit_exp"),
        Intrinsic::Log => ("entelechy_jit_logf", "entelechy_jit_log"),
        Intrinsic::Exp2 => ("entelechy_jit_exp2f", "entelechy_jit_exp2"),
        Intrinsic::Log2 => ("entelechy_jit_log2f", "entelechy_jit_log2"),
        Intrinsic::Sinh => ("entelechy_jit_sinhf", "entelechy_jit_sinh"),
        Intrinsic::Cosh => ("entelechy_jit_coshf", "entelechy_jit_cosh"),
        _ => ("entelechy_jit_tanhf", "entelechy_jit_tanh"),
    };
    if wide { wide_name } else { narrow }
}
