//! The word code: what a function compiles to. Offsets are relative
//! to the running frame; a value is a run of words there.

use orla_ir::BinaryOp;

use crate::ArithmeticError;

/// Where a value lives: a frame-relative word offset and a width in words.
#[derive(Clone, Copy, Debug)]
pub(super) struct Loc {
    pub(super) at: usize,
    pub(super) n: usize,
}

/// Word layout of one struct type.
pub(super) struct StructLayout {
    pub(super) words: usize,
    pub(super) field_offsets: Vec<usize>,
}

/// How a vector component is made from a part.
#[derive(Clone, Copy)]
pub(super) enum Widen {
    /// Already a float word.
    Float,
    /// An `Int` or `Bool` word, converted.
    Int,
}

/// How an address is read for `at` and `at_linear`.
#[derive(Clone, Copy)]
pub(super) enum Address {
    Int,
    Float,
    Vec2,
    Vec3,
}

pub(super) type F1 = fn(f32) -> f32;
pub(super) type F2 = fn(f32, f32) -> f32;
pub(super) type F3 = fn(f32, f32, f32) -> f32;

pub(super) type D1 = fn(f64) -> f64;
pub(super) type D2 = fn(f64, f64) -> f64;
pub(super) type D3 = fn(f64, f64, f64) -> f64;

/// What a scalar is, for a conversion between kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lane {
    Float,
    Int,
    Bool,
    Double,
    Long,
}

impl Lane {
    /// Words a scalar of this kind takes.
    pub(super) fn words(self) -> usize {
        match self {
            Self::Double | Self::Long => 2,
            _ => 1,
        }
    }
}

/// One compiled instruction. Offsets are relative to the running frame;
/// `n` is a width in words; `sa`, `sb`, `sc` are operand strides, zero
/// for a scalar broadcast beside a vector.
pub(super) enum Instr {
    Const {
        dst: usize,
        words: [u32; 4],
        n: usize,
    },
    Param {
        dst: usize,
        at: usize,
        n: usize,
    },
    Move {
        dst: usize,
        src: usize,
        n: usize,
    },
    Gather {
        dst: usize,
        parts: Vec<Loc>,
    },
    /// A value known when the function compiled: its words.
    Words {
        dst: usize,
        words: Vec<u32>,
    },
    /// `dst` = the `n` words of element `index` (clamped to `count`) of
    /// the array at `base`.
    Subscript {
        dst: usize,
        base: usize,
        index: usize,
        count: usize,
        n: usize,
    },
    /// The 64-bit operations: `n` lanes of two words each, the low word
    /// first; strides count lanes, zero for a scalar broadcast.
    DMap1 {
        f: D1,
        dst: usize,
        src: usize,
        n: usize,
    },
    DMap2 {
        f: D2,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    DMap3 {
        f: D3,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        c: usize,
        sc: usize,
        n: usize,
    },
    DCmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    DEq {
        dst: usize,
        a: usize,
        b: usize,
        n: usize,
        negate: bool,
    },
    /// Arithmetic, bit operations and shifts on longs; the arithmetic
    /// fails on overflow as an `Int`'s does, and a shift takes its count
    /// modulo 64.
    LBin {
        op: BinaryOp,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    LNeg {
        dst: usize,
        src: usize,
        n: usize,
    },
    LMap1 {
        f: fn(i64) -> i64,
        dst: usize,
        src: usize,
        n: usize,
    },
    LMap2 {
        f: fn(i64, i64) -> i64,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    LMap3 {
        f: fn(i64, i64, i64) -> i64,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        c: usize,
        sc: usize,
        n: usize,
    },
    LCmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    /// `n` scalars of one kind as another: what `float`, `int`,
    /// `double` and `long` do where a 64-bit kind is on either side.
    Convert {
        from: Lane,
        to: Lane,
        dst: usize,
        src: usize,
        n: usize,
    },
    Vector {
        dst: usize,
        parts: Vec<(usize, Widen)>,
    },
    FNeg {
        dst: usize,
        src: usize,
        n: usize,
    },
    INeg {
        dst: usize,
        src: usize,
        n: usize,
    },
    Not {
        dst: usize,
        src: usize,
    },
    FBin {
        f: F2,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    /// Two float operations as one: `g(f(a, b), c)`, or `g(c, f(a, b))`
    /// with `flip`, in one dispatch; the intermediate stays off the frame.
    FBin2 {
        f: F2,
        g: F2,
        flip: bool,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        c: usize,
        sc: usize,
        n: usize,
    },
    IBin {
        op: BinaryOp,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    FCmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    ICmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    FEq {
        dst: usize,
        a: usize,
        b: usize,
        n: usize,
        negate: bool,
    },
    IEq {
        dst: usize,
        a: usize,
        b: usize,
        n: usize,
        negate: bool,
    },
    And {
        dst: usize,
        a: usize,
        b: usize,
    },
    Or {
        dst: usize,
        a: usize,
        b: usize,
    },
    Map1 {
        f: F1,
        dst: usize,
        src: usize,
        n: usize,
    },
    Map2 {
        f: F2,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    Map3 {
        f: F3,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        c: usize,
        sc: usize,
        n: usize,
    },
    IMap1 {
        f: fn(i32) -> i32,
        dst: usize,
        src: usize,
        n: usize,
    },
    /// An integer operation that may fail, a division by zero say.
    IMap2Checked {
        f: fn(i32, i32) -> Result<i32, ArithmeticError>,
        dst: usize,
        a: usize,
        b: usize,
    },
    IMap2 {
        f: fn(i32, i32) -> i32,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    IClamp {
        dst: usize,
        v: usize,
        low: usize,
        sl: usize,
        high: usize,
        sh: usize,
        n: usize,
    },
    ToFloat {
        dst: usize,
        src: usize,
        n: usize,
    },
    ToInt {
        dst: usize,
        src: usize,
        n: usize,
    },
    Atan2 {
        dst: usize,
        y: usize,
        x: usize,
    },
    If {
        condition: usize,
        then: Vec<Instr>,
        then_result: usize,
        otherwise: Vec<Instr>,
        otherwise_result: usize,
        dst: usize,
        n: usize,
    },
    Match {
        scrutinee: usize,
        /// Each arm's values, code and result; the first holding the
        /// value runs.
        arms: Vec<(Vec<i32>, Vec<Instr>, usize)>,
        otherwise: Vec<Instr>,
        otherwise_result: usize,
        dst: usize,
        n: usize,
    },
    Loop {
        count: u32,
        /// The slot of a run-time count, clamped to `0..=count`.
        bound: Option<usize>,
        init: usize,
        carried: usize,
        n: usize,
        until: Option<(Vec<Instr>, usize)>,
        body: Vec<Instr>,
        body_result: usize,
    },
    Index {
        dst: usize,
    },
    Call {
        function: usize,
        args: Vec<Loc>,
        dst: usize,
        n: usize,
    },
    Coord {
        dst: usize,
        n: usize,
    },
    Extent {
        dst: usize,
    },
    Element {
        dst: usize,
    },
    Time {
        dst: usize,
    },
    Rate {
        dst: usize,
    },
    Pointer {
        dst: usize,
    },
    PointerDown {
        dst: usize,
    },
    Key {
        dst: usize,
        code: usize,
    },
    Button {
        dst: usize,
        code: usize,
    },
    /// `dst` = the pad axis pair numbered by the slot at `which`.
    Stick {
        dst: usize,
        which: usize,
    },
    Wheel {
        dst: usize,
    },
    /// `dst` = how many pads are connected.
    Pads {
        dst: usize,
    },
    Window {
        dst: usize,
    },
    Scale {
        dst: usize,
    },
    Count {
        dst: usize,
        input: usize,
    },
    Size {
        dst: usize,
        input: usize,
    },
    Sample {
        dst: usize,
        input: usize,
        n: usize,
    },
    At {
        dst: usize,
        input: usize,
        address: usize,
        kind: Address,
        n: usize,
    },
    /// `dst` = the `n` words a path names inside one element of `input`:
    /// `base` words in, then per index step the clamped index times its
    /// stride. The element is at `address`, or this element's own (as
    /// `Sample` reads it) without one.
    Part {
        dst: usize,
        input: usize,
        address: Option<(usize, Address)>,
        base: usize,
        steps: Vec<(usize, usize, usize)>,
        n: usize,
    },
    AtLinear {
        dst: usize,
        input: usize,
        position: usize,
        kind: Address,
        n: usize,
    },
    Put {
        dst: usize,
        input: usize,
        address: usize,
        kind: Address,
        value: usize,
        n: usize,
    },
    /// Appends the pair's words to the machine's emitted pairs, in order.
    Emit {
        dst: usize,
        pair: usize,
        n: usize,
    },
    /// Adds the finished loop's run iterations to the element's work.
    Spend {
        dst: usize,
    },
    /// Checks the element's work: fails past a bound, and hands filled
    /// blocks to the run.
    Look {
        dst: usize,
    },
}

/// A compiled function: its frame size in words, where its arguments
/// and results live in the frame, and its code.
pub(super) struct Code {
    pub(super) frame: usize,
    pub(super) args: Vec<Loc>,
    pub(super) results: Vec<Loc>,
    pub(super) code: Vec<Instr>,
    /// Where every value of the function lives in its frame, by id.
    pub(super) locs: Vec<Loc>,
}
