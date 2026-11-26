//! Kernel fusion: inlining a producer's per-element computation into a
//! consumer that samples it at the current coordinate, or reads it at
//! rows its own work decides.
//!
//! For `Y = C(P(X))`, the fused module computes `P`'s result first and uses
//! it where `C` sampled, or computes it as a function of the row where `C`
//! read, so the intermediate Field is never stored. A Kernel that only asks
//! a Field its count or size reads the extent from params instead. Each
//! result is an ordinary module that validates and runs on any backend.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{
    BinaryOp, Block, Capabilities, Encoding, Function, Instruction, Intrinsic, Manifest, Mode,
    Module, Op, Param, Port, PortType, Schema, Step, StructDef, Type, Validated, Value, ValueId,
    validate,
};

/// Internal helper name emitted by [`fuse_outputs`] / [`fuse_indexed`] /
/// [`fuse_stencil`] when a fused link crosses an [`Encoding::F16`] port,
/// rounding IEEE-754 `f32 -> f16 -> f32` bit-for-bit identically to Host
/// storage across every backend.
pub const QUANTIZE_F16_FN: &str = "#quantize_f16";

/// Prefix for internal helper functions emitted by [`fuse_stencil`]
/// (`#tile_r{radius}_{slot}`), recognized by the GPU backend to cooperatively
/// cache the producer's 2D tile + halo in workgroup shared memory.
pub const TILE_FN_PREFIX: &str = "#tile_r";

/// Maximum 2D stencil halo radius supported by [`stencil_radius`] and
/// [`fuse_stencil`] (`1..=2`, covering 3x3 cross/box and 5x5 À-Trous stencils).
pub const MAX_STENCIL_RADIUS: u32 = 2;

/// Parses the halo radius `R` (`1..=MAX_STENCIL_RADIUS`) from a [`TILE_FN_PREFIX`]
/// helper function name (`#tile_r{R}_{slot}`), returning `None` for any other name.
pub fn tile_fn_radius(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(TILE_FN_PREFIX)?;
    let (digits, suffix) = rest.split_once('_')?;
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let radius: u32 = digits.parse().ok()?;
    (1..=MAX_STENCIL_RADIUS).contains(&radius).then_some(radius)
}

/// Why two modules cannot fuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuseError {
    /// What went wrong.
    pub message: String,
    /// Whether the planner runs the two separately without reporting: the
    /// consumer's port cannot take the producer's Field (the binding check
    /// reports it), or a Kernel's work is counted. Any other refusal is a
    /// fault of this crate and is reported.
    pub declines: bool,
}

impl FuseError {
    fn new(message: String) -> Self {
        Self {
            message,
            declines: false,
        }
    }

    fn declined(message: String) -> Self {
        Self {
            message,
            declines: true,
        }
    }
}

impl fmt::Display for FuseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FuseError {}

/// How a module reads one of its inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputUse {
    /// Read at the current coordinate.
    pub sampled: bool,
    /// Read at other coordinates.
    pub addressed: bool,
    /// Asked for its count or size.
    pub counted: bool,
}

fn use_in(
    instructions: &[Instruction],
    input: u32,
    self_coord_ok: bool,
    coords: &mut BTreeSet<u32>,
    use_: &mut InputUse,
) {
    for instruction in instructions {
        if matches!(instruction.op, Op::Coord) {
            coords.insert(instruction.id.0);
        }
        match &instruction.op {
            Op::Sample(i) if *i == input => use_.sampled = true,
            Op::Part {
                input: i, at: None, ..
            } if *i == input => use_.sampled = true,
            Op::At(i, addr) if *i == input && self_coord_ok && coords.contains(&addr.0) => {
                use_.sampled = true;
            }
            Op::Part {
                input: i,
                at: Some(addr),
                ..
            } if *i == input && self_coord_ok && coords.contains(&addr.0) => {
                use_.sampled = true;
            }
            Op::Count(i) | Op::Size(i) if *i == input => use_.counted = true,
            Op::At(i, _)
            | Op::AtLinear(i, _)
            | Op::AtFiltered { input: i, .. }
            | Op::Part { input: i, .. }
                if *i == input =>
            {
                use_.addressed = true
            }
            op => {
                for block in op.blocks() {
                    use_in(&block.body, input, self_coord_ok, coords, use_);
                }
            }
        }
    }
}

/// Reports how `module` reads `input`.
pub fn input_use(module: &Module, input: u32) -> InputUse {
    let mut use_ = InputUse::default();
    let grid_or_volume = module
        .manifest
        .inputs
        .get(input as usize)
        .is_some_and(|port| matches!(port.shape(), crate::Shape::Grid | crate::Shape::Volume));
    for function in &module.functions {
        let mut coords = BTreeSet::new();
        use_in(&function.body, input, false, &mut coords, &mut use_);
    }
    let mut coords = BTreeSet::new();
    use_in(
        &module.entry.body,
        input,
        grid_or_volume,
        &mut coords,
        &mut use_,
    );
    use_
}

/// Whether `module` reads `input` only by `sample`, so its producer can
/// fuse into it.
pub fn samples_only(module: &Module, input: u32) -> bool {
    let use_ = input_use(module, input);
    use_.sampled && !use_.addressed && !use_.counted
}

/// Whether `module` asks `input` only its count or size, so the input can
/// leave it and the extent come in by params ([`counted`]).
pub fn counts_only(module: &Module, input: u32) -> bool {
    let use_ = input_use(module, input);
    use_.counted && !use_.sampled && !use_.addressed
}

/// Whether `module` reads `input`, a row, only at an offset of its own
/// element: `element()` plus or less a value every element shares, read
/// in the entry outside any loop and at that one address, so an element
/// reads at most one of the input's and two elements two. Besides, it may
/// ask the input's count or size. The input's producer can then fuse into
/// it as a function of the row ([`fuse_indexed`]).
pub fn offset_read(module: &Module, input: u32) -> bool {
    let rows = module
        .manifest
        .inputs
        .get(input as usize)
        .is_some_and(|port| port.shape() == crate::Shape::Rows);
    if !rows {
        return false;
    }
    // A function may ask the count or size, nothing more.
    let mut in_functions = InputUse::default();
    for function in &module.functions {
        let mut coords = BTreeSet::new();
        use_in(&function.body, input, false, &mut coords, &mut in_functions);
    }
    if in_functions.sampled || in_functions.addressed {
        return false;
    }
    let mut reads = Reads {
        input,
        defs: BTreeMap::new(),
        addresses: Vec::new(),
        refused: false,
    };
    reads.walk(&module.entry.body, false);
    let Some(&address) = reads.addresses.first() else {
        return false;
    };
    if reads.refused || reads.addresses.iter().any(|other| *other != address) {
        return false;
    }
    reads.offset(address, &mut BTreeMap::new())
}

/// Whether `module` reads `input`, a row, only at addresses, each read
/// written in the entry outside any loop, and if so at how many places:
/// an element then asks the input for at most that many of its elements,
/// wherever its own work leads it. Besides, it may ask the input's count
/// or size. The input's producer can then fuse into it as a function of
/// the row ([`fuse_indexed`]), called at each place; whether calling it
/// costs less than storing it is the planner's to say, from that number.
pub fn row_reads(module: &Module, input: u32) -> Option<usize> {
    let rows = module
        .manifest
        .inputs
        .get(input as usize)
        .is_some_and(|port| port.shape() == crate::Shape::Rows);
    if !rows {
        return None;
    }
    // A function may ask the count or size, nothing more.
    let mut in_functions = InputUse::default();
    for function in &module.functions {
        let mut coords = BTreeSet::new();
        use_in(&function.body, input, false, &mut coords, &mut in_functions);
    }
    if in_functions.sampled || in_functions.addressed {
        return None;
    }
    let mut reads = Reads {
        input,
        defs: BTreeMap::new(),
        addresses: Vec::new(),
        refused: false,
    };
    reads.walk(&module.entry.body, false);
    // Two parts of one element are read at one address: one place.
    let places: std::collections::BTreeSet<u32> =
        reads.addresses.iter().map(|address| address.0).collect();
    (!reads.refused && !places.is_empty()).then_some(places.len())
}

/// Sound interval analysis over a 2D `Grid` map's entry point: returns
/// `Some(radius)` (`0..=MAX_STENCIL_RADIUS`) when every read of `input` is
/// provably within integer offset `[-radius, +radius]` of `coord()` on both
/// axes (optionally clamped to `[0, size(input) - 1]` / `[0, domain() - 1]`),
/// with no helper-function reads and no filtered (`at_linear` / `at_repeat`)
/// reads. `const_params` supplies known literal/default parameter values
/// (excluding runtime-overridable graph parameters).
pub fn stencil_radius(
    module: &Module,
    input: u32,
    const_params: &BTreeMap<u32, Value>,
) -> Option<u32> {
    stencil_radius_with_peers(module, input, &[input], const_params)
}

/// Like [`stencil_radius`], also recognizing `size(peer)` for any input in
/// `same_extent_inputs` (e.g. multiple outputs of the same producer) as the
/// exact extent bound of `input`.
pub fn stencil_radius_with_peers(
    module: &Module,
    input: u32,
    same_extent_inputs: &[u32],
    const_params: &BTreeMap<u32, Value>,
) -> Option<u32> {
    if module.manifest.mode != Mode::Map {
        return None;
    }
    let is_grid = module
        .manifest
        .inputs
        .get(input as usize)
        .is_some_and(|port| port.shape() == crate::Shape::Grid);
    let out_grid = module
        .manifest
        .outputs
        .first()
        .is_some_and(|port| port.shape() == crate::Shape::Grid);
    if !is_grid || !out_grid {
        return None;
    }
    let mut in_functions = InputUse::default();
    for function in &module.functions {
        let mut coords = BTreeSet::new();
        use_in(&function.body, input, false, &mut coords, &mut in_functions);
    }
    if in_functions.sampled || in_functions.addressed {
        return None;
    }
    let mut scan = StencilScan {
        input,
        same_extent_inputs,
        const_params,
        defs: BTreeMap::new(),
        loop_indices: Vec::new(),
        saw_read: false,
        max_radius: 0,
        refused: false,
    };
    scan.walk(&module.entry.body);
    (!scan.refused && scan.saw_read && scan.max_radius <= MAX_STENCIL_RADIUS)
        .then_some(scan.max_radius)
}

/// Whether stencil-fusing `input` of `module` requires extent params (`#count`
/// / `#size`) because `module` calls `count(input)` anywhere or `size(input)`
/// inside a helper function (where `Op::Domain` is disallowed).
pub fn stencil_needs_extent(module: &Module, input: u32) -> bool {
    fn walk(body: &[Instruction], input: u32, in_fn: bool) -> bool {
        for instr in body {
            match &instr.op {
                Op::Count(i) if *i == input => return true,
                Op::Size(i) if *i == input && in_fn => return true,
                op => {
                    if op.blocks().iter().any(|b| walk(&b.body, input, in_fn)) {
                        return true;
                    }
                }
            }
        }
        false
    }
    module.functions.iter().any(|f| walk(&f.body, input, true))
        || walk(&module.entry.body, input, false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    lo: i64,
    hi: i64,
}

impl Interval {
    const fn point(v: i64) -> Self {
        Self { lo: v, hi: v }
    }

    fn join(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            lo: self.lo.checked_add(other.lo)?,
            hi: self.hi.checked_add(other.hi)?,
        })
    }

    fn sub(self, other: Self) -> Option<Self> {
        Some(Self {
            lo: self.lo.checked_sub(other.hi)?,
            hi: self.hi.checked_sub(other.lo)?,
        })
    }

    fn mul(self, other: Self) -> Option<Self> {
        let p0 = self.lo.checked_mul(other.lo)?;
        let p1 = self.lo.checked_mul(other.hi)?;
        let p2 = self.hi.checked_mul(other.lo)?;
        let p3 = self.hi.checked_mul(other.hi)?;
        Some(Self {
            lo: p0.min(p1).min(p2).min(p3),
            hi: p0.max(p1).max(p2).max(p3),
        })
    }

    fn neg(self) -> Option<Self> {
        Some(Self {
            lo: self.hi.checked_neg()?,
            hi: self.lo.checked_neg()?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Const(Interval),
    Coord(Interval),
    MaxCoord,
    Extent,
    Unknown,
}

impl Axis {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => Self::Const(a.join(b)),
            (Self::Coord(a), Self::Coord(b)) => Self::Coord(a.join(b)),
            (Self::MaxCoord, Self::MaxCoord) => Self::MaxCoord,
            (Self::Extent, Self::Extent) => Self::Extent,
            _ => Self::Unknown,
        }
    }

    fn is_zero(self) -> bool {
        self == Self::Const(Interval::point(0))
    }

    fn is_one(self) -> bool {
        self == Self::Const(Interval::point(1))
    }

    fn is_neg_one(self) -> bool {
        self == Self::Const(Interval::point(-1))
    }

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a.add(b).map_or(Self::Unknown, Self::Const),
            (Self::Coord(a), Self::Const(b)) | (Self::Const(b), Self::Coord(a)) => {
                a.add(b).map_or(Self::Unknown, Self::Coord)
            }
            (Self::Extent, b) if b.is_neg_one() => Self::MaxCoord,
            (a, Self::Extent) if a.is_neg_one() => Self::MaxCoord,
            _ => Self::Unknown,
        }
    }

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a.sub(b).map_or(Self::Unknown, Self::Const),
            (Self::Coord(a), Self::Const(b)) => a.sub(b).map_or(Self::Unknown, Self::Coord),
            (Self::Extent, b) if b.is_one() => Self::MaxCoord,
            _ => Self::Unknown,
        }
    }

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a.mul(b).map_or(Self::Unknown, Self::Const),
            _ => Self::Unknown,
        }
    }

    fn div_int(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) if a.lo >= 0 && b.lo == b.hi && b.lo > 0 => {
                Self::Const(Interval {
                    lo: a.lo / b.lo,
                    hi: a.hi / b.lo,
                })
            }
            _ => Self::Unknown,
        }
    }

    fn rem_int(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) if a.lo >= 0 && b.lo == b.hi && b.lo > 0 => {
                let d = b.lo;
                if a.hi - a.lo < d && a.lo / d == a.hi / d {
                    Self::Const(Interval {
                        lo: a.lo % d,
                        hi: a.hi % d,
                    })
                } else {
                    Self::Const(Interval { lo: 0, hi: d - 1 })
                }
            }
            _ => Self::Unknown,
        }
    }

    fn clamp(self, lo: Self, hi: Self) -> Self {
        match (self, lo, hi) {
            (Self::Coord(iv), lo, Self::MaxCoord) if lo.is_zero() => Self::Coord(iv),
            (Self::Const(v), Self::Const(l), Self::Const(h)) if l.lo == l.hi && h.lo == h.hi => {
                Self::Const(Interval {
                    lo: v.lo.clamp(l.lo, h.lo),
                    hi: v.hi.clamp(l.lo, h.lo),
                })
            }
            _ => Self::Unknown,
        }
    }

    fn min(self, other: Self) -> Self {
        match (self, other) {
            (Self::Coord(iv), Self::MaxCoord) | (Self::MaxCoord, Self::Coord(iv)) => {
                Self::Coord(iv)
            }
            (Self::Const(a), Self::Const(b)) => Self::Const(Interval {
                lo: a.lo.min(b.lo),
                hi: a.hi.min(b.hi),
            }),
            _ => Self::Unknown,
        }
    }

    fn max(self, other: Self) -> Self {
        match (self, other) {
            (Self::Coord(iv), b) if b.is_zero() => Self::Coord(iv),
            (a, Self::Coord(iv)) if a.is_zero() => Self::Coord(iv),
            (Self::Const(a), Self::Const(b)) => Self::Const(Interval {
                lo: a.lo.max(b.lo),
                hi: a.hi.max(b.hi),
            }),
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AbsVal {
    Scalar(Axis),
    Vec2(Axis, Axis),
    Unknown,
}

impl AbsVal {
    fn from_float(f: f32) -> Axis {
        if f.is_finite() && f.fract() == 0.0 && f.abs() <= 1_000_000.0 {
            Axis::Const(Interval::point(f as i64))
        } else {
            Axis::Unknown
        }
    }

    fn from_value(val: &Value) -> Self {
        match val {
            Value::Int(n) => Self::Scalar(Axis::Const(Interval::point(i64::from(*n)))),
            Value::Float(f) => Self::Scalar(Self::from_float(*f)),
            Value::Vec2([x, y]) => Self::Vec2(Self::from_float(*x), Self::from_float(*y)),
            Value::IVec2([x, y]) => Self::Vec2(
                Axis::Const(Interval::point(i64::from(*x))),
                Axis::Const(Interval::point(i64::from(*y))),
            ),
            _ => Self::Unknown,
        }
    }

    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => Self::Scalar(a.join(b)),
            (Self::Vec2(ax, ay), Self::Vec2(bx, by)) => Self::Vec2(ax.join(bx), ay.join(by)),
            _ => Self::Unknown,
        }
    }

    fn scalar(self) -> Axis {
        match self {
            Self::Scalar(a) => a,
            _ => Axis::Unknown,
        }
    }

    fn map_axes(self, f: impl Fn(Axis) -> Axis) -> Self {
        match self {
            Self::Scalar(a) => Self::Scalar(f(a)),
            Self::Vec2(x, y) => Self::Vec2(f(x), f(y)),
            Self::Unknown => Self::Unknown,
        }
    }

    fn zip2(self, other: Self, f: impl Fn(Axis, Axis) -> Axis) -> Self {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => Self::Scalar(f(a, b)),
            (Self::Vec2(ax, ay), Self::Vec2(bx, by)) => Self::Vec2(f(ax, bx), f(ay, by)),
            (Self::Vec2(ax, ay), Self::Scalar(b)) => Self::Vec2(f(ax, b), f(ay, b)),
            (Self::Scalar(a), Self::Vec2(bx, by)) => Self::Vec2(f(a, bx), f(a, by)),
            _ => Self::Unknown,
        }
    }

    fn zip3(self, b: Self, c: Self, f: impl Fn(Axis, Axis, Axis) -> Axis) -> Self {
        match (self, b, c) {
            (Self::Scalar(va), Self::Scalar(vb), Self::Scalar(vc)) => Self::Scalar(f(va, vb, vc)),
            (Self::Vec2(ax, ay), b, c) => {
                let (bx, by) = match b {
                    Self::Vec2(x, y) => (x, y),
                    Self::Scalar(s) => (s, s),
                    Self::Unknown => return Self::Unknown,
                };
                let (cx, cy) = match c {
                    Self::Vec2(x, y) => (x, y),
                    Self::Scalar(s) => (s, s),
                    Self::Unknown => return Self::Unknown,
                };
                Self::Vec2(f(ax, bx, cx), f(ay, by, cy))
            }
            _ => Self::Unknown,
        }
    }
}

struct StencilScan<'a> {
    input: u32,
    same_extent_inputs: &'a [u32],
    const_params: &'a BTreeMap<u32, Value>,
    defs: BTreeMap<u32, AbsVal>,
    loop_indices: Vec<Interval>,
    saw_read: bool,
    max_radius: u32,
    refused: bool,
}

impl<'a> StencilScan<'a> {
    fn get(&self, id: ValueId) -> AbsVal {
        self.defs.get(&id.0).copied().unwrap_or(AbsVal::Unknown)
    }

    fn check_addr(&mut self, addr: ValueId) {
        match self.get(addr) {
            AbsVal::Vec2(Axis::Coord(ix), Axis::Coord(iy)) => {
                let r = ix
                    .lo
                    .unsigned_abs()
                    .max(ix.hi.unsigned_abs())
                    .max(iy.lo.unsigned_abs())
                    .max(iy.hi.unsigned_abs());
                if r <= u64::from(MAX_STENCIL_RADIUS) {
                    self.saw_read = true;
                    self.max_radius = self.max_radius.max(r as u32);
                } else {
                    self.refused = true;
                }
            }
            _ => self.refused = true,
        }
    }

    fn walk(&mut self, instructions: &[Instruction]) {
        for instr in instructions {
            let val = match &instr.op {
                Op::Coord if instr.ty == Type::Vec2 => AbsVal::Vec2(
                    Axis::Coord(Interval::point(0)),
                    Axis::Coord(Interval::point(0)),
                ),
                Op::Domain if instr.ty == Type::Vec2 => AbsVal::Vec2(Axis::Extent, Axis::Extent),
                Op::Size(i) if *i == self.input || self.same_extent_inputs.contains(i) => {
                    AbsVal::Vec2(Axis::Extent, Axis::Extent)
                }
                Op::Const(v) => AbsVal::from_value(v),
                Op::Param(k) => self
                    .const_params
                    .get(k)
                    .map_or(AbsVal::Unknown, AbsVal::from_value),
                Op::Index => self
                    .loop_indices
                    .last()
                    .copied()
                    .map_or(AbsVal::Unknown, |iv| AbsVal::Scalar(Axis::Const(iv))),
                Op::Extract(src, k) => match (self.get(*src), *k) {
                    (AbsVal::Vec2(x, _), 0) => AbsVal::Scalar(x),
                    (AbsVal::Vec2(_, y), 1) => AbsVal::Scalar(y),
                    _ => AbsVal::Unknown,
                },
                Op::Construct(args)
                    if args.len() == 2 && matches!(instr.ty, Type::Vec2 | Type::IVec2) =>
                {
                    AbsVal::Vec2(self.get(args[0]).scalar(), self.get(args[1]).scalar())
                }
                Op::Unary(crate::UnaryOp::Neg, a) => self.get(*a).map_axes(|ax| match ax {
                    Axis::Const(iv) => iv.neg().map_or(Axis::Unknown, Axis::Const),
                    _ => Axis::Unknown,
                }),
                Op::Binary(op, a, b) => {
                    let (va, vb) = (self.get(*a), self.get(*b));
                    match op {
                        BinaryOp::Add => va.zip2(vb, Axis::add),
                        BinaryOp::Sub => va.zip2(vb, Axis::sub),
                        BinaryOp::Mul => va.zip2(vb, Axis::mul),
                        BinaryOp::Div if instr.ty.is_int_kind() => va.zip2(vb, Axis::div_int),
                        BinaryOp::Rem if instr.ty.is_int_kind() => va.zip2(vb, Axis::rem_int),
                        _ => AbsVal::Unknown,
                    }
                }
                Op::Intrinsic(intr, args) => match (intr, args.as_slice()) {
                    (
                        Intrinsic::ToFloat
                        | Intrinsic::ToInt
                        | Intrinsic::Floor
                        | Intrinsic::Ceil
                        | Intrinsic::Round
                        | Intrinsic::Trunc,
                        [a],
                    ) => self.get(*a),
                    (Intrinsic::AddWrap, [a, b]) => self.get(*a).zip2(self.get(*b), Axis::add),
                    (Intrinsic::SubWrap, [a, b]) => self.get(*a).zip2(self.get(*b), Axis::sub),
                    (Intrinsic::MulWrap, [a, b]) => self.get(*a).zip2(self.get(*b), Axis::mul),
                    (Intrinsic::Clamp, [v, lo, hi]) => {
                        self.get(*v).zip3(self.get(*lo), self.get(*hi), Axis::clamp)
                    }
                    (Intrinsic::Min, [a, b]) => self.get(*a).zip2(self.get(*b), Axis::min),
                    (Intrinsic::Max, [a, b]) => self.get(*a).zip2(self.get(*b), Axis::max),
                    _ => AbsVal::Unknown,
                },
                Op::Sample(i)
                | Op::Part {
                    input: i, at: None, ..
                } if *i == self.input => {
                    self.saw_read = true;
                    AbsVal::Unknown
                }
                Op::At(i, addr)
                | Op::Part {
                    input: i,
                    at: Some(addr),
                    ..
                } if *i == self.input => {
                    self.check_addr(*addr);
                    AbsVal::Unknown
                }
                Op::AtLinear(i, _) | Op::AtFiltered { input: i, .. } if *i == self.input => {
                    self.refused = true;
                    AbsVal::Unknown
                }
                Op::If {
                    then, otherwise, ..
                } => {
                    self.walk(&then.body);
                    self.walk(&otherwise.body);
                    self.get(then.result).join(self.get(otherwise.result))
                }
                Op::Match {
                    arms, otherwise, ..
                } => {
                    for (_, block) in arms {
                        self.walk(&block.body);
                    }
                    self.walk(&otherwise.body);
                    let mut acc = self.get(otherwise.result);
                    for (_, block) in arms {
                        acc = acc.join(self.get(block.result));
                    }
                    acc
                }
                Op::Loop {
                    count, until, body, ..
                } => {
                    let hi = i64::from(*count).saturating_sub(1).max(0);
                    self.loop_indices.push(Interval { lo: 0, hi });
                    if let Some(until) = until {
                        self.walk(&until.body);
                    }
                    self.walk(&body.body);
                    self.loop_indices.pop();
                    AbsVal::Unknown
                }
                Op::Each { count, body } => {
                    let hi = i64::from(*count).saturating_sub(1).max(0);
                    self.loop_indices.push(Interval { lo: 0, hi });
                    self.walk(&body.body);
                    self.loop_indices.pop();
                    AbsVal::Unknown
                }
                op => {
                    for block in op.blocks() {
                        self.walk(&block.body);
                    }
                    AbsVal::Unknown
                }
            };
            self.defs.insert(instr.id.0, val);
        }
    }
}

/// The entry's reads of one input, and every value's definition.
struct Reads<'a> {
    input: u32,
    defs: BTreeMap<u32, &'a Op>,
    addresses: Vec<ValueId>,
    /// Read otherwise than at an address, or at one inside a loop.
    refused: bool,
}

impl<'a> Reads<'a> {
    fn walk(&mut self, instructions: &'a [Instruction], looped: bool) {
        for instruction in instructions {
            let op = &instruction.op;
            self.defs.insert(instruction.id.0, op);
            match op {
                Op::Sample(i)
                | Op::AtLinear(i, _)
                | Op::AtFiltered { input: i, .. }
                | Op::Part {
                    input: i, at: None, ..
                } if *i == self.input => self.refused = true,
                Op::At(i, address)
                | Op::Part {
                    input: i,
                    at: Some(address),
                    ..
                } if *i == self.input => {
                    self.refused |= looped;
                    self.addresses.push(*address);
                }
                Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        self.walk(&until.body, true);
                    }
                    self.walk(&body.body, true);
                }
                // An each's body runs once for each position, as a loop's.
                Op::Each { body, .. } => self.walk(&body.body, true),
                op => {
                    for block in op.blocks() {
                        self.walk(&block.body, looped);
                    }
                }
            }
        }
    }

    /// Whether `id` is `element()` plus or less a shared value.
    fn offset(&self, id: ValueId, shared: &mut BTreeMap<u32, bool>) -> bool {
        use crate::{BinaryOp, Intrinsic};
        match self.defs.get(&id.0) {
            Some(Op::Element) => true,
            Some(Op::Binary(BinaryOp::Add, a, b)) => self.sum(*a, *b, shared),
            Some(Op::Binary(BinaryOp::Sub, a, b)) => {
                self.offset(*a, shared) && self.shared(*b, shared)
            }
            Some(Op::Intrinsic(Intrinsic::AddWrap, args)) if args.len() == 2 => {
                self.sum(args[0], args[1], shared)
            }
            Some(Op::Intrinsic(Intrinsic::SubWrap, args)) if args.len() == 2 => {
                self.offset(args[0], shared) && self.shared(args[1], shared)
            }
            _ => false,
        }
    }

    /// Whether `a + b` is an offset: one of them is, the other shared.
    fn sum(&self, a: ValueId, b: ValueId, shared: &mut BTreeMap<u32, bool>) -> bool {
        (self.offset(a, shared) && self.shared(b, shared))
            || (self.shared(a, shared) && self.offset(b, shared))
    }

    /// Whether `id` is the same for every element, remembered by value.
    fn shared(&self, id: ValueId, known: &mut BTreeMap<u32, bool>) -> bool {
        if let Some(&shared) = known.get(&id.0) {
            return shared;
        }
        let shared = match self.defs.get(&id.0) {
            Some(
                Op::Const(_)
                | Op::Param(_)
                | Op::Count(_)
                | Op::Size(_)
                | Op::Domain
                | Op::Time
                | Op::Clock
                | Op::Rate
                | Op::Queued
                | Op::Window
                | Op::Scale
                | Op::Pads,
            ) => true,
            Some(Op::Unary(_, a) | Op::Extract(a, _)) => self.shared(*a, known),
            Some(Op::Binary(_, a, b)) => self.shared(*a, known) && self.shared(*b, known),
            Some(Op::Intrinsic(_, args) | Op::Construct(args)) => {
                args.iter().all(|arg| self.shared(*arg, known))
            }
            _ => false,
        };
        known.insert(id.0, shared);
        shared
    }
}

/// The params a Kernel reads a producer's extent from once that producer
/// is fused into it or is not stored for it: the element count, as
/// `count` gives it, and the extent, as `size` gives it. They are named
/// under the producer's node, as no written name can be; the planner sets
/// them.
pub fn extent_params(prefix: &str) -> [String; 2] {
    [format!("{prefix}/#count"), format!("{prefix}/#size")]
}

/// Whether `name` is one of [`extent_params`]'s.
fn is_extent_param(name: &str) -> bool {
    name.rsplit_once('/')
        .is_some_and(|(_, last)| last == "#count" || last == "#size")
}

/// The index of the param `name` of type `ty`, appended unless there.
fn extent_param(params: &mut Vec<Param>, name: &str, ty: Type) -> u32 {
    if let Some(index) = params.iter().position(|param| param.name == name) {
        return index as u32;
    }
    let default = match ty {
        Type::Vec2 => Value::Vec2([0.0; 2]),
        Type::Vec3 => Value::Vec3([0.0; 3]),
        _ => Value::Int(0),
    };
    params.push(Param {
        name: name.to_owned(),
        ty,
        default,
    });
    (params.len() - 1) as u32
}

/// The count and size params of a producer of `shape` under `prefix`.
fn extent_pair(params: &mut Vec<Param>, prefix: &str, shape: crate::Shape) -> (u32, u32) {
    use crate::ShapeIndex;
    let [count, size] = extent_params(prefix);
    (
        extent_param(params, &count, Type::Int),
        extent_param(params, &size, shape.index_type()),
    )
}

/// Drops the inputs `module` asks only the count or size of
/// ([`counts_only`]), each to read its producer's extent from the params
/// [`extent_params`] names under the prefix given with it, so nothing has
/// to be stored for it.
pub fn counted(module: &Module, inputs: &[(u32, String)]) -> Result<Validated, FuseError> {
    let manifest = &module.manifest;
    if module.merge.is_some() || module.init.is_some() || !manifest.memories.is_empty() {
        return Err(FuseError::new(
            "only a map without memories reads an extent from params".into(),
        ));
    }
    for (index, (input, _)) in inputs.iter().enumerate() {
        if *input as usize >= manifest.inputs.len() {
            return Err(FuseError::new(format!("the Kernel has no input {input}")));
        }
        if inputs[..index].iter().any(|(earlier, _)| earlier == input) {
            return Err(FuseError::new(format!("input {input} is counted twice")));
        }
        if !counts_only(module, *input) {
            return Err(FuseError::new(format!(
                "input `{}` is read, not only counted",
                manifest.inputs[*input as usize].name
            )));
        }
    }
    let mut params = manifest.params.clone();
    let extents: BTreeMap<u32, (u32, u32)> = inputs
        .iter()
        .map(|(input, prefix)| {
            let shape = manifest.inputs[*input as usize].shape();
            (*input, extent_pair(&mut params, prefix, shape))
        })
        .collect();
    let removed: Vec<u32> = inputs.iter().map(|(input, _)| *input).collect();
    let ops = |op: &Op| -> Op {
        match op {
            Op::Count(i) if extents.contains_key(i) => Op::Param(extents[i].0),
            Op::Size(i) if extents.contains_key(i) => Op::Param(extents[i].1),
            other => shift_inputs(other, |i| {
                i - removed.iter().filter(|gone| **gone < i).count() as u32
            }),
        }
    };
    let identity = |ty: Type| ty;
    let renumbered =
        |function: &Function| Renumber::new(&ops, &identity, BTreeMap::new()).function(function);
    let manifest = Manifest {
        inputs: manifest
            .inputs
            .iter()
            .enumerate()
            .filter(|(index, _)| !removed.contains(&(*index as u32)))
            .map(|(_, port)| port.clone())
            .collect(),
        params,
        ..manifest.clone()
    };
    validate(Module {
        manifest,
        structs: module.structs.clone(),
        functions: module.functions.iter().map(renumbered).collect(),
        entry: renumbered(&module.entry),
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError::new(format!("counted module is invalid: {error}")))
}

/// `op` with each input it names renumbered by `index`.
fn shift_inputs(op: &Op, index: impl Fn(u32) -> u32) -> Op {
    match op {
        Op::Sample(i) => Op::Sample(index(*i)),
        Op::At(i, c) => Op::At(index(*i), *c),
        Op::AtLinear(i, c) => Op::AtLinear(index(*i), *c),
        Op::AtFiltered {
            input: i,
            at,
            repeat,
            lod,
            grad,
        } => Op::AtFiltered {
            input: index(*i),
            at: *at,
            repeat: *repeat,
            lod: *lod,
            grad: *grad,
        },
        Op::Count(i) => Op::Count(index(*i)),
        Op::Size(i) => Op::Size(index(*i)),
        Op::Part { input: i, at, path } => Op::Part {
            input: index(*i),
            at: *at,
            path: path.clone(),
        },
        Op::Put(i, a, v) => Op::Put(index(*i), *a, *v),
        other => other.clone(),
    }
}

/// A name the producer brings into a fused Kernel: `prefix/name`, or the
/// name as it is when an earlier fusion already qualified it with its own
/// node's (no written name holds a `/`), so names stay short however long
/// a chain fuses and stay unique, each qualified once by its node.
pub fn fused_name(prefix: &str, name: &str) -> String {
    if name.contains('/') {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

/// Fuses `producer` into `consumer` at the consumer's `input`, which reads
/// the producer's first output: [`fuse_outputs`] with one link.
pub fn fuse(
    consumer: &Module,
    input: u32,
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_outputs(consumer, &[(input, 0)], producer, prefix)
}

/// Fuses `producer` into `consumer`. Each link `(input, output)` binds a
/// consumer input to a producer output; every sample of that input becomes
/// the producer's value for it, and its count and size come in by the
/// params [`extent_params`] names. Unread producer outputs are computed and
/// dropped. The fused manifest keeps the consumer's other inputs and params,
/// then the producer's, renamed by [`fused_name`], then the extent's.
pub fn fuse_outputs(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_links(consumer, links, producer, prefix, Reading::Pointwise)
}

/// Fuses `producer` into `consumer` as a function of the row: every read
/// of a linked input, at an address written in the consumer's entry
/// outside any loop ([`row_reads`]), becomes the producer's computation at
/// that address,
/// clamped into the producer's extent as `at` clamps it; the input's count
/// and size come in by the params [`extent_params`] names, and so does the
/// producer's own extent. The manifest is [`fuse_outputs`]'s.
pub fn fuse_indexed(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_links(consumer, links, producer, prefix, Reading::Indexed)
}

/// Fuses `producer` into `consumer` as a 2D stencil helper `#tile_r{radius}_{k}`:
/// every read of a linked 2D `Grid` input (proven within halo `radius` by
/// [`stencil_radius`]) becomes a clamped call to `#tile_r{radius}_{k}(pos, dom)`.
/// On CPU/JIT this runs as ordinary validated IR; on GPU the WGSL emitter
/// cooperatively fills a `var<workgroup>` tile + halo before `workgroupBarrier()`
/// so every neighbor tap hits fast on-chip SRAM with zero intermediate DRAM I/O.
pub fn fuse_stencil(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
    radius: u32,
) -> Result<Validated, FuseError> {
    fuse_links(
        consumer,
        links,
        producer,
        prefix,
        Reading::Stencil { radius },
    )
}

/// How a fused consumer reads its producer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// At its own element: the producer's body runs first.
    Pointwise,
    /// At addresses: the producer's body is a function of the row, called
    /// where the consumer reads.
    Indexed,
    /// At 2D `Grid` neighborhood offsets within `[-radius, +radius]` of
    /// `coord()`: the producer's body becomes a `#tile_r{radius}_{k}(pos, dom)`
    /// function that the GPU caches in workgroup shared memory.
    Stencil { radius: u32 },
}

fn body_uses_element(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instr| {
        matches!(instr.op, Op::Element)
            || instr
                .op
                .blocks()
                .iter()
                .any(|block| body_uses_element(&block.body))
    })
}

fn next_tile_slot(functions: &[Function]) -> u32 {
    functions
        .iter()
        .filter_map(|f| {
            let rest = f.name.strip_prefix(TILE_FN_PREFIX)?;
            let (_, suffix) = rest.split_once('_')?;
            suffix.parse::<u32>().ok()
        })
        .max()
        .map_or(0, |m| m + 1)
}

fn fuse_links(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
    reading: Reading,
) -> Result<Validated, FuseError> {
    let consumer_manifest = &consumer.manifest;
    let producer_manifest = &producer.manifest;
    if links.is_empty() {
        return Err(FuseError::new("fusion needs a link".into()));
    }
    for (index, (input, output)) in links.iter().enumerate() {
        if *input as usize >= consumer_manifest.inputs.len() {
            return Err(FuseError::new(format!("consumer has no input {input}")));
        }
        if *output as usize >= producer_manifest.outputs.len() {
            return Err(FuseError::new(format!("producer has no output {output}")));
        }
        if links[..index].iter().any(|(earlier, _)| earlier == input) {
            return Err(FuseError::new(format!("input {input} is linked twice")));
        }
        let name = &consumer_manifest.inputs[*input as usize].name;
        match reading {
            Reading::Pointwise if input_use(consumer, *input).addressed => {
                return Err(FuseError::new(format!(
                    "consumer reads input `{name}` by address, not only by sample"
                )));
            }
            Reading::Indexed
                if row_reads(consumer, *input).is_none() && !counts_only(consumer, *input) =>
            {
                return Err(FuseError::new(format!(
                    "consumer reads input `{name}` otherwise than at addresses written in its entry outside any loop"
                )));
            }
            Reading::Stencil { radius } => {
                if !(1..=MAX_STENCIL_RADIUS).contains(&radius) {
                    return Err(FuseError::new(format!(
                        "stencil radius {radius} is outside 1..={MAX_STENCIL_RADIUS}"
                    )));
                }
                if consumer_manifest.inputs[*input as usize].shape() != crate::Shape::Grid {
                    return Err(FuseError::new(format!(
                        "stencil fusion requires a Grid input, but `{name}` is {}",
                        consumer_manifest.inputs[*input as usize].shape()
                    )));
                }
                let mut in_functions = InputUse::default();
                for function in &consumer.functions {
                    let mut coords = BTreeSet::new();
                    use_in(
                        &function.body,
                        *input,
                        false,
                        &mut coords,
                        &mut in_functions,
                    );
                }
                if in_functions.sampled || in_functions.addressed {
                    return Err(FuseError::new(format!(
                        "consumer reads stencil input `{name}` inside a helper function"
                    )));
                }
            }
            _ => {}
        }
    }
    // A Kernel whose work is counted keeps its own budget and run, so it
    // runs alone.
    if producer.counts_work() || consumer.counts_work() {
        return Err(FuseError::declined(
            "a Kernel whose work is counted runs on its own, held to its own budget".into(),
        ));
    }
    if consumer.manifest.mode != Mode::Map || producer.manifest.mode != Mode::Map {
        return Err(FuseError::new(
            "only map Kernels fuse; reductions and scans run on their own".into(),
        ));
    }
    let producer_has_tile = producer
        .functions
        .iter()
        .any(|f| tile_fn_radius(&f.name).is_some());
    if producer_has_tile
        && (reading != Reading::Pointwise
            || consumer_manifest
                .outputs
                .first()
                .is_none_or(|p| p.shape() != crate::Shape::Grid))
    {
        return Err(FuseError::declined(
            "a tiled stencil Kernel fuses downstream only into a pointwise Grid consumer".into(),
        ));
    }
    if matches!(reading, Reading::Stencil { .. })
        && consumer_manifest
            .outputs
            .first()
            .is_none_or(|p| p.shape() != crate::Shape::Grid)
    {
        return Err(FuseError::new(
            "stencil fusion requires a 2D Grid consumer output".into(),
        ));
    }
    // Ports match by shape and layout; struct indices are module-local.
    for (input, output) in links {
        let (written, read) = (
            &producer_manifest.outputs[*output as usize].ty,
            &consumer_manifest.inputs[*input as usize].ty,
        );
        if written.shape != read.shape || written.schema != read.schema {
            return Err(FuseError {
                message: format!("producer writes {written} but the consumer reads {read}"),
                declines: true,
            });
        }
    }
    // Linked inputs leave the consumer; later inputs move down to close up.
    let removed: Vec<u32> = links.iter().map(|(input, _)| *input).collect();
    let kept_inputs = (consumer_manifest.inputs.len() - removed.len()) as u32;
    let consumer_functions = consumer.functions.len() as u32;
    let rename = |name: &str| fused_name(prefix, name);

    // The producer's structs join the consumer's table once each, so a
    // spliced element carries the consumer's own struct and types match.
    // A linked output's struct is the struct its consumer reads (and so
    // are the structs within them, field by field), whatever each names
    // it; any other is the consumer's struct of its name and fields, as a
    // struct declared alike in two documents is one; a struct of another
    // name stays apart, though its fields are the same. Structs refer
    // only to earlier ones, so one ordered pass maps them.
    let mut linked: BTreeMap<u32, u32> = BTreeMap::new();
    for (input, output) in links {
        link_structs(
            &producer.structs,
            &consumer.structs,
            producer_manifest.outputs[*output as usize].element(),
            consumer_manifest.inputs[*input as usize].ty.element,
            &mut linked,
        );
    }
    let mut structs = consumer.structs.clone();
    let mut moved: Vec<u32> = Vec::with_capacity(producer.structs.len());
    for (at, def) in producer.structs.iter().enumerate() {
        if let Some(&index) = linked.get(&(at as u32)) {
            moved.push(index);
            continue;
        }
        let mapped = |ty: Type| match ty {
            Type::Struct(index) => Type::Struct(moved[index as usize]),
            other => other,
        };
        let fields: Vec<crate::Field> = def
            .fields
            .iter()
            .map(|field| crate::Field {
                name: field.name.clone(),
                ty: mapped(field.ty),
            })
            .collect();
        let element = def.element.map(mapped);
        match structs.iter().position(|other| {
            other.name == def.name && other.fields == fields && other.element == element
        }) {
            Some(index) => moved.push(index as u32),
            None => {
                moved.push(structs.len() as u32);
                structs.push(StructDef {
                    name: rename(&def.name),
                    fields,
                    element,
                });
            }
        }
    }
    let shift_struct = |ty: Type| match ty {
        Type::Struct(index) => Type::Struct(moved[index as usize]),
        other => other,
    };
    let mut inputs: Vec<Port> = consumer_manifest
        .inputs
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, port)| port.clone())
        .collect();
    inputs.extend(producer_manifest.inputs.iter().map(|port| Port {
        name: rename(&port.name),
        ty: PortType {
            element: shift_struct(port.ty.element),
            shape: port.ty.shape,
            schema: port.ty.schema.clone(),
        },
    }));
    // A producer's param keeps its place after the consumer's, but for an
    // extent both read already, which is one value.
    let mut params: Vec<Param> = consumer_manifest.params.clone();
    let mut producer_params = Vec::with_capacity(producer_manifest.params.len());
    for param in &producer_manifest.params {
        let name = rename(&param.name);
        let ty = shift_struct(param.ty);
        let index = if is_extent_param(&name) {
            extent_param(&mut params, &name, ty)
        } else {
            params.push(Param {
                name,
                ty,
                default: param.default.clone(),
            });
            (params.len() - 1) as u32
        };
        producer_params.push(index);
    }
    // The producer's extent, for a count or size of a linked input and for
    // the producer's own element in an indexed fusion. Stencil fusion passes
    // `Op::Domain` directly into `#tile_r*` and only allocates extent params
    // when the consumer uses `count(input)` or `size(input)` in a helper fn.
    let counted_links = links
        .iter()
        .any(|(input, _)| input_use(consumer, *input).counted);
    let stencil_extent = matches!(reading, Reading::Stencil { .. })
        && links
            .iter()
            .any(|(input, _)| stencil_needs_extent(consumer, *input));
    let needs_extent = match reading {
        Reading::Pointwise => counted_links,
        Reading::Indexed => true,
        Reading::Stencil { .. } => stencil_extent,
    };
    let extent = needs_extent.then(|| {
        let shape = producer_manifest.outputs[links[0].1 as usize].shape();
        extent_pair(&mut params, prefix, shape)
    });
    let mut requires: Capabilities = consumer_manifest.requires;
    for capability in producer_manifest.requires.iter() {
        requires.insert(capability);
    }
    let manifest = Manifest {
        id: consumer_manifest.id.clone(),
        mode: consumer_manifest.mode,
        requires,
        inputs,
        params,
        memories: Vec::new(),
        outputs: consumer_manifest.outputs.clone(),
        // One element runs both, so it may run what both may.
        budget: consumer_manifest
            .budget
            .saturating_add(producer_manifest.budget)
            .min(crate::MAX_BUDGET),
    };

    // Consumer ops: inputs after a fused one move down, and a fused one's
    // count and size are the extent's params (or `Op::Domain` for Grid `size`
    // in a stencil consumer's entry).
    let is_stencil = matches!(reading, Reading::Stencil { .. });
    let consumer_ops = |op: &Op| -> Op {
        match (op, extent) {
            (Op::Count(i), Some((count, _))) if removed.contains(i) => Op::Param(count),
            (Op::Size(i), Some((_, size))) if removed.contains(i) => Op::Param(size),
            (Op::Size(i), None) if is_stencil && removed.contains(i) => Op::Domain,
            _ => shift_inputs(op, |i| {
                i - removed.iter().filter(|gone| **gone < i).count() as u32
            }),
        }
    };
    // Producer ops: inputs, params, functions and structs follow the
    // consumer's.
    let producer_ops = |op: &Op| -> Op {
        match op {
            Op::Param(k) => Op::Param(producer_params[*k as usize]),
            Op::Call(f, args) => Op::Call(f + consumer_functions, args.clone()),
            other => shift_inputs(other, |i| i + kept_inputs),
        }
    };
    let identity = |ty: Type| ty;

    let mut functions: Vec<Function> = consumer
        .functions
        .iter()
        .map(|function| {
            let mut renumber = Renumber::new(&consumer_ops, &identity, BTreeMap::new());
            renumber.function(function)
        })
        .collect();
    let mut tile_slot = next_tile_slot(&functions);
    functions.extend(producer.functions.iter().map(|function| {
        let mut renumber = Renumber::new(&producer_ops, &shift_struct, BTreeMap::new());
        // `function` has already shifted the argument types.
        let mut fused = renumber.function(function);
        if let Some(radius) = tile_fn_radius(&function.name) {
            fused.name = format!("{TILE_FN_PREFIX}{radius}_{tile_slot}");
            tile_slot += 1;
        } else if function.name != QUANTIZE_F16_FN {
            fused.name = rename(&function.name);
        }
        fused
    }));
    // Ensure `#quantize_f16` exists once if any narrow fused link uses `Encoding::F16`.
    let needs_f16_helper = links.iter().any(|(_, output)| {
        producer_manifest.outputs[*output as usize]
            .ty
            .schema
            .encodings()
            .contains(&Encoding::F16)
    });
    let quantize_f16_fn = if needs_f16_helper {
        if let Some(existing) = functions.iter().position(|f| f.name == QUANTIZE_F16_FN) {
            Some(existing as u32)
        } else {
            let idx = functions.len() as u32;
            functions.push(build_quantize_f16_function());
            Some(idx)
        }
    } else {
        None
    };
    let producer_bindings = |renumber: &Renumber| -> Vec<(ValueId, String)> {
        producer
            .entry
            .bindings
            .iter()
            .map(|(id, name)| {
                (
                    renumber.map(*id),
                    format!("{}/{name}", producer.manifest.id),
                )
            })
            .collect()
    };

    let (body, results, bindings) = match reading {
        Reading::Pointwise => {
            // The producer's body first, then explicit bit-exact narrowing for
            // any narrow linked output, then the consumer's body with its
            // sample of the fused input replaced by the (rounded) value.
            let mut entry_renumber = Renumber::new(&producer_ops, &shift_struct, BTreeMap::new());
            let mut body = entry_renumber.block(&producer.entry.body);
            // Cache rounded outputs so two consumer inputs reading the same
            // narrow producer output only round once.
            let mut rounded_outputs: BTreeMap<u32, ValueId> = BTreeMap::new();
            let mut replace = BTreeMap::new();
            for &(input, output) in links {
                let output_port = &producer_manifest.outputs[output as usize];
                let element = shift_struct(output_port.element());
                let produced = *rounded_outputs.entry(output).or_insert_with(|| {
                    let raw = entry_renumber.map(producer.entry.results[output as usize]);
                    if output_port.ty.schema.is_wide() {
                        raw
                    } else {
                        emit_quantize_value(
                            &mut body,
                            &mut entry_renumber.next,
                            raw,
                            element,
                            &output_port.ty.schema,
                            &structs,
                            quantize_f16_fn,
                        )
                    }
                });
                replace.insert(input, (produced, element));
            }
            let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, replace);
            consumer_renumber.next = entry_renumber.next;
            consumer_renumber.structs = Some(&structs);
            body.extend(consumer_renumber.block(&consumer.entry.body));
            let results = consumer_renumber.results(&consumer.entry.results);
            let mut bindings = producer_bindings(&entry_renumber);
            bindings.extend(consumer_renumber.bindings(&consumer.entry.bindings));
            (body, results, bindings)
        }
        Reading::Indexed => {
            let (count, size) = extent.expect("an indexed fusion reads the extent");
            // The producer's own extent is the param's, its element the row
            // it is called at.
            let row_ops = |op: &Op| -> Op {
                match op {
                    Op::Domain => Op::Param(size),
                    other => producer_ops(other),
                }
            };
            // One function per output read by address: the producer's
            // entry at a row, yielding that output (narrowed if non-wide).
            let mut rows: BTreeMap<u32, u32> = BTreeMap::new();
            let mut called = BTreeMap::new();
            for (input, output) in links {
                if !input_use(consumer, *input).addressed {
                    continue;
                }
                let function = match rows.get(output) {
                    Some(function) => *function,
                    None => {
                        let mut renumber = Renumber::new(&row_ops, &shift_struct, BTreeMap::new());
                        let row = ValueId(0);
                        renumber.next = 1;
                        renumber.own = Some(row);
                        let mut body = vec![Instruction {
                            id: row,
                            ty: Type::Int,
                            op: Op::Arg(0),
                        }];
                        body.extend(renumber.block(&producer.entry.body));
                        let output_port = &producer_manifest.outputs[*output as usize];
                        let element = shift_struct(output_port.element());
                        let raw = renumber.map(producer.entry.results[*output as usize]);
                        let result = if output_port.ty.schema.is_wide() {
                            raw
                        } else {
                            emit_quantize_value(
                                &mut body,
                                &mut renumber.next,
                                raw,
                                element,
                                &output_port.ty.schema,
                                &structs,
                                quantize_f16_fn,
                            )
                        };
                        functions.push(Function {
                            name: format!("{prefix}/{} at a row", output_port.name),
                            args: vec![Type::Int],
                            body,
                            results: vec![result],
                            bindings: producer_bindings(&renumber),
                        });
                        let function = (functions.len() - 1) as u32;
                        rows.insert(*output, function);
                        function
                    }
                };
                let element = shift_struct(producer_manifest.outputs[*output as usize].element());
                called.insert(*input, (function, element));
            }
            let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, BTreeMap::new());
            consumer_renumber.called = called;
            consumer_renumber.count = count;
            consumer_renumber.structs = Some(&structs);
            let body = consumer_renumber.block(&consumer.entry.body);
            let results = consumer_renumber.results(&consumer.entry.results);
            let bindings = consumer_renumber.bindings(&consumer.entry.bindings);
            (body, results, bindings)
        }
        Reading::Stencil { radius } => {
            // Collect unique producer outputs actually read by the consumer.
            let mut unique_outputs: Vec<u32> = Vec::new();
            for &(input, output) in links {
                let u = input_use(consumer, input);
                if (u.sampled || u.addressed) && !unique_outputs.contains(&output) {
                    unique_outputs.push(output);
                }
            }
            if unique_outputs.is_empty() {
                return Err(FuseError::declined(
                    "stencil fusion requires at least one read linked input".into(),
                ));
            }
            let slot = tile_slot;
            let mut renumber = Renumber::new(&producer_ops, &shift_struct, BTreeMap::new());
            let pos = ValueId(0);
            let dom = ValueId(1);
            let mut tile_body = vec![
                Instruction {
                    id: pos,
                    ty: Type::Vec2,
                    op: Op::Arg(0),
                },
                Instruction {
                    id: dom,
                    ty: Type::Vec2,
                    op: Op::Arg(1),
                },
            ];
            renumber.next = 2;
            renumber.tile_pos = Some(pos);
            renumber.tile_dom = Some(dom);
            if body_uses_element(&producer.entry.body) {
                let mut b = IrBuilder {
                    out: &mut tile_body,
                    next: &mut renumber.next,
                };
                let w_f = b.push(Type::Float, Op::Extract(dom, 0));
                let w_i = b.intr(Type::Int, Intrinsic::ToInt, vec![w_f]);
                let px_f = b.push(Type::Float, Op::Extract(pos, 0));
                let px_i = b.intr(Type::Int, Intrinsic::ToInt, vec![px_f]);
                let py_f = b.push(Type::Float, Op::Extract(pos, 1));
                let py_i = b.intr(Type::Int, Intrinsic::ToInt, vec![py_f]);
                let row = b.intr(Type::Int, Intrinsic::MulWrap, vec![py_i, w_i]);
                let elem = b.intr(Type::Int, Intrinsic::AddWrap, vec![row, px_i]);
                renumber.tile_elem = Some(elem);
            }
            tile_body.extend(renumber.block(&producer.entry.body));
            let mut quantized = Vec::with_capacity(unique_outputs.len());
            for &out_idx in &unique_outputs {
                let output_port = &producer_manifest.outputs[out_idx as usize];
                let element = shift_struct(output_port.element());
                let raw = renumber.map(producer.entry.results[out_idx as usize]);
                let value = if output_port.ty.schema.is_wide() {
                    raw
                } else {
                    emit_quantize_value(
                        &mut tile_body,
                        &mut renumber.next,
                        raw,
                        element,
                        &output_port.ty.schema,
                        &structs,
                        quantize_f16_fn,
                    )
                };
                quantized.push((value, element));
            }
            let (tile_ret_ty, tile_result) = if quantized.len() == 1 {
                (quantized[0].1, quantized[0].0)
            } else {
                let struct_idx = structs.len() as u32;
                structs.push(StructDef {
                    name: format!("#tile_out_{slot}"),
                    fields: quantized
                        .iter()
                        .enumerate()
                        .map(|(k, &(_, ty))| crate::Field {
                            name: format!("o{k}"),
                            ty,
                        })
                        .collect(),
                    element: None,
                });
                let ty = Type::Struct(struct_idx);
                let id = ValueId(renumber.next);
                renumber.next += 1;
                tile_body.push(Instruction {
                    id,
                    ty,
                    op: Op::Construct(quantized.iter().map(|&(v, _)| v).collect()),
                });
                (ty, id)
            };
            let tile_fn = functions.len() as u32;
            functions.push(Function {
                name: format!("{TILE_FN_PREFIX}{radius}_{slot}"),
                args: vec![Type::Vec2, Type::Vec2],
                body: tile_body,
                results: vec![tile_result],
                bindings: producer_bindings(&renumber),
            });
            let mut stencil = BTreeMap::new();
            for &(input, output) in links {
                if let Some(field_pos) = unique_outputs.iter().position(|&o| o == output) {
                    let element =
                        shift_struct(producer_manifest.outputs[output as usize].element());
                    let field = (unique_outputs.len() > 1).then_some(field_pos as u32);
                    stencil.insert(input, (tile_fn, tile_ret_ty, field, element));
                }
            }
            let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, BTreeMap::new());
            consumer_renumber.stencil = stencil;
            consumer_renumber.structs = Some(&structs);
            let body = consumer_renumber.block(&consumer.entry.body);
            let results = consumer_renumber.results(&consumer.entry.results);
            let bindings = consumer_renumber.bindings(&consumer.entry.bindings);
            (body, results, bindings)
        }
    };
    let entry = Function {
        name: consumer.entry.name.clone(),
        args: Vec::new(),
        body,
        results,
        bindings,
    };
    let fused = validate(Module {
        manifest,
        structs,
        functions,
        entry,
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError::new(format!("fused module is invalid: {error}")))?;
    // A fusion that would be counted where neither part is stays unfused.
    if fused.counts_work() {
        return Err(FuseError::declined(
            "together the two would run long enough to be counted, which neither is on its own"
                .into(),
        ));
    }
    Ok(fused)
}

/// Tiny helper for building pre-order SSA instructions cleanly.
struct IrBuilder<'a> {
    out: &'a mut Vec<Instruction>,
    next: &'a mut u32,
}

impl<'a> IrBuilder<'a> {
    fn push(&mut self, ty: Type, op: Op) -> ValueId {
        let id = ValueId(*self.next);
        *self.next += 1;
        self.out.push(Instruction { id, ty, op });
        id
    }

    fn int(&mut self, v: i32) -> ValueId {
        self.push(Type::Int, Op::Const(Value::Int(v)))
    }

    fn float(&mut self, v: f32) -> ValueId {
        self.push(Type::Float, Op::Const(Value::Float(v)))
    }

    fn bin(&mut self, ty: Type, op: BinaryOp, a: ValueId, b: ValueId) -> ValueId {
        self.push(ty, Op::Binary(op, a, b))
    }

    fn intr(&mut self, ty: Type, i: Intrinsic, args: Vec<ValueId>) -> ValueId {
        self.push(ty, Op::Intrinsic(i, args))
    }

    fn push_if(
        &mut self,
        ty: Type,
        condition: ValueId,
        then_fn: impl FnOnce(&mut IrBuilder<'_>) -> ValueId,
        else_fn: impl FnOnce(&mut IrBuilder<'_>) -> ValueId,
    ) -> ValueId {
        let id = ValueId(*self.next);
        *self.next += 1;
        let mut then_body = Vec::new();
        let then_res = {
            let mut b = IrBuilder {
                out: &mut then_body,
                next: self.next,
            };
            then_fn(&mut b)
        };
        let mut else_body = Vec::new();
        let else_res = {
            let mut b = IrBuilder {
                out: &mut else_body,
                next: self.next,
            };
            else_fn(&mut b)
        };
        self.out.push(Instruction {
            id,
            ty,
            op: Op::If {
                condition,
                then: Block {
                    body: then_body,
                    result: then_res,
                },
                otherwise: Block {
                    body: else_body,
                    result: else_res,
                },
            },
        });
        id
    }
}

/// Builds the pure, non-failing IR function [`QUANTIZE_F16_FN`] rounding an
/// `f32` through IEEE-754 binary16 and back (`f16_to_f32(f32_to_f16(bits(x)))`),
/// matching `entelechy_abi::Encoding::F16` bit-for-bit across all backends.
fn build_quantize_f16_function() -> Function {
    let mut body = Vec::new();
    let mut next = 0u32;
    let result = {
        let mut b = IrBuilder {
            out: &mut body,
            next: &mut next,
        };
        let arg = b.push(Type::Float, Op::Arg(0));
        let x = b.intr(Type::Int, Intrinsic::Bits, vec![arg]);
        let c16 = b.int(16);
        let c0x8000 = b.int(0x8000);
        let x_shr16 = b.bin(Type::Int, BinaryOp::Ushr, x, c16);
        let sign16 = b.bin(Type::Int, BinaryOp::BitAnd, x_shr16, c0x8000);
        let c23 = b.int(23);
        let c0xff = b.int(0xFF);
        let x_shr23 = b.bin(Type::Int, BinaryOp::Ushr, x, c23);
        let exponent = b.bin(Type::Int, BinaryOp::BitAnd, x_shr23, c0xff);
        let c_mant_mask = b.int(0x7F_FFFF);
        let mantissa = b.bin(Type::Int, BinaryOp::BitAnd, x, c_mant_mask);
        let is_inf_nan = b.bin(Type::Bool, BinaryOp::Equal, exponent, c0xff);

        let h = b.push_if(
            Type::Int,
            is_inf_nan,
            |b| {
                let c0 = b.int(0);
                let is_nan = b.bin(Type::Bool, BinaryOp::NotEqual, mantissa, c0);
                let c0x7c00 = b.int(0x7C00);
                let inf = b.bin(Type::Int, BinaryOp::BitOr, sign16, c0x7c00);
                b.push_if(
                    Type::Int,
                    is_nan,
                    |b| {
                        let c13 = b.int(13);
                        let m_hi = b.bin(Type::Int, BinaryOp::Ushr, mantissa, c13);
                        let c0x200 = b.int(0x200);
                        let qnan = b.bin(Type::Int, BinaryOp::BitOr, c0x200, m_hi);
                        b.bin(Type::Int, BinaryOp::BitOr, inf, qnan)
                    },
                    |_| inf,
                )
            },
            |b| {
                let c112 = b.int(112);
                let e = b.intr(Type::Int, Intrinsic::SubWrap, vec![exponent, c112]);
                let c31 = b.int(31);
                let overflow = b.bin(Type::Bool, BinaryOp::GreaterEqual, e, c31);
                b.push_if(
                    Type::Int,
                    overflow,
                    |b| {
                        let c0x7c00 = b.int(0x7C00);
                        b.bin(Type::Int, BinaryOp::BitOr, sign16, c0x7c00)
                    },
                    |b| {
                        let c0 = b.int(0);
                        let subnormal = b.bin(Type::Bool, BinaryOp::LessEqual, e, c0);
                        b.push_if(
                            Type::Int,
                            subnormal,
                            |b| {
                                let cm10 = b.int(-10);
                                let underflow = b.bin(Type::Bool, BinaryOp::Less, e, cm10);
                                b.push_if(
                                    Type::Int,
                                    underflow,
                                    |_| sign16,
                                    |b| {
                                        let c_imp = b.int(0x80_0000);
                                        let full =
                                            b.bin(Type::Int, BinaryOp::BitOr, mantissa, c_imp);
                                        let c14 = b.int(14);
                                        let shift =
                                            b.intr(Type::Int, Intrinsic::SubWrap, vec![c14, e]);
                                        let val = b.bin(Type::Int, BinaryOp::Ushr, full, shift);
                                        let c1 = b.int(1);
                                        let one_sh = b.bin(Type::Int, BinaryOp::Shl, c1, shift);
                                        let mask =
                                            b.intr(Type::Int, Intrinsic::SubWrap, vec![one_sh, c1]);
                                        let dropped =
                                            b.bin(Type::Int, BinaryOp::BitAnd, full, mask);
                                        let sh_m1 =
                                            b.intr(Type::Int, Intrinsic::SubWrap, vec![shift, c1]);
                                        let half = b.bin(Type::Int, BinaryOp::Shl, c1, sh_m1);
                                        let gt =
                                            b.bin(Type::Bool, BinaryOp::Greater, dropped, half);
                                        let eq = b.bin(Type::Bool, BinaryOp::Equal, dropped, half);
                                        let lsb = b.bin(Type::Int, BinaryOp::BitAnd, val, c1);
                                        let odd = b.bin(Type::Bool, BinaryOp::Equal, lsb, c1);
                                        let tie = b.bin(Type::Bool, BinaryOp::And, eq, odd);
                                        let up = b.bin(Type::Bool, BinaryOp::Or, gt, tie);
                                        let inc =
                                            b.push_if(Type::Int, up, |b| b.int(1), |b| b.int(0));
                                        let rounded =
                                            b.intr(Type::Int, Intrinsic::AddWrap, vec![val, inc]);
                                        b.bin(Type::Int, BinaryOp::BitOr, sign16, rounded)
                                    },
                                )
                            },
                            |b| {
                                let c10 = b.int(10);
                                let c13 = b.int(13);
                                let e_sh = b.bin(Type::Int, BinaryOp::Shl, e, c10);
                                let m_hi = b.bin(Type::Int, BinaryOp::Ushr, mantissa, c13);
                                let val = b.bin(Type::Int, BinaryOp::BitOr, e_sh, m_hi);
                                let c0x1fff = b.int(0x1FFF);
                                let dropped = b.bin(Type::Int, BinaryOp::BitAnd, mantissa, c0x1fff);
                                let c0x1000 = b.int(0x1000);
                                let gt = b.bin(Type::Bool, BinaryOp::Greater, dropped, c0x1000);
                                let eq = b.bin(Type::Bool, BinaryOp::Equal, dropped, c0x1000);
                                let c1 = b.int(1);
                                let lsb = b.bin(Type::Int, BinaryOp::BitAnd, val, c1);
                                let odd = b.bin(Type::Bool, BinaryOp::Equal, lsb, c1);
                                let tie = b.bin(Type::Bool, BinaryOp::And, eq, odd);
                                let up = b.bin(Type::Bool, BinaryOp::Or, gt, tie);
                                let inc = b.push_if(Type::Int, up, |b| b.int(1), |b| b.int(0));
                                let rounded = b.intr(Type::Int, Intrinsic::AddWrap, vec![val, inc]);
                                b.bin(Type::Int, BinaryOp::BitOr, sign16, rounded)
                            },
                        )
                    },
                )
            },
        );

        // Decode f16 bits `h` back into f32 bits.
        let h_sign = b.bin(Type::Int, BinaryOp::BitAnd, h, c0x8000);
        let sign32 = b.bin(Type::Int, BinaryOp::Shl, h_sign, c16);
        let c10 = b.int(10);
        let c0x1f = b.int(0x1F);
        let h_shr10 = b.bin(Type::Int, BinaryOp::Ushr, h, c10);
        let he = b.bin(Type::Int, BinaryOp::BitAnd, h_shr10, c0x1f);
        let c0x3ff = b.int(0x3FF);
        let hm = b.bin(Type::Int, BinaryOp::BitAnd, h, c0x3ff);
        let c0 = b.int(0);
        let he_zero = b.bin(Type::Bool, BinaryOp::Equal, he, c0);
        let out_bits = b.push_if(
            Type::Int,
            he_zero,
            |b| {
                let hm_zero = b.bin(Type::Bool, BinaryOp::Equal, hm, c0);
                b.push_if(
                    Type::Int,
                    hm_zero,
                    |_| sign32,
                    |b| {
                        let lz = b.intr(Type::Int, Intrinsic::Clz, vec![hm]);
                        let c21 = b.int(21);
                        let sh = b.intr(Type::Int, Intrinsic::SubWrap, vec![lz, c21]);
                        let hm_sh = b.bin(Type::Int, BinaryOp::Shl, hm, sh);
                        let nm = b.bin(Type::Int, BinaryOp::BitAnd, hm_sh, c0x3ff);
                        let c113 = b.int(113);
                        let exp = b.intr(Type::Int, Intrinsic::SubWrap, vec![c113, sh]);
                        let exp_sh = b.bin(Type::Int, BinaryOp::Shl, exp, c23);
                        let c13 = b.int(13);
                        let nm_sh = b.bin(Type::Int, BinaryOp::Shl, nm, c13);
                        let s_e = b.bin(Type::Int, BinaryOp::BitOr, sign32, exp_sh);
                        b.bin(Type::Int, BinaryOp::BitOr, s_e, nm_sh)
                    },
                )
            },
            |b| {
                let he_inf = b.bin(Type::Bool, BinaryOp::Equal, he, c0x1f);
                b.push_if(
                    Type::Int,
                    he_inf,
                    |b| {
                        let c_inf = b.int(0x7F80_0000);
                        let c13 = b.int(13);
                        let hm_sh = b.bin(Type::Int, BinaryOp::Shl, hm, c13);
                        let s_i = b.bin(Type::Int, BinaryOp::BitOr, sign32, c_inf);
                        b.bin(Type::Int, BinaryOp::BitOr, s_i, hm_sh)
                    },
                    |b| {
                        let c112 = b.int(112);
                        let exp = b.intr(Type::Int, Intrinsic::AddWrap, vec![he, c112]);
                        let exp_sh = b.bin(Type::Int, BinaryOp::Shl, exp, c23);
                        let c13 = b.int(13);
                        let hm_sh = b.bin(Type::Int, BinaryOp::Shl, hm, c13);
                        let s_e = b.bin(Type::Int, BinaryOp::BitOr, sign32, exp_sh);
                        b.bin(Type::Int, BinaryOp::BitOr, s_e, hm_sh)
                    },
                )
            },
        );
        b.intr(Type::Float, Intrinsic::FromBits, vec![out_bits])
    };
    Function {
        name: QUANTIZE_F16_FN.to_owned(),
        args: vec![Type::Float],
        body,
        results: vec![result],
        bindings: Vec::new(),
    }
}

/// Emits bit-exact narrow-encoding round-trip operations for `value` of type
/// `ty` matching `schema`'s per-scalar [`Encoding`]s.
fn emit_quantize_value(
    out: &mut Vec<Instruction>,
    next: &mut u32,
    value: ValueId,
    ty: Type,
    schema: &Schema,
    structs: &[StructDef],
    quantize_f16_fn: Option<u32>,
) -> ValueId {
    let encodings = schema.encodings();
    let mut cursor = 0usize;
    let mut b = IrBuilder { out, next };
    quantize_rec(
        &mut b,
        value,
        ty,
        encodings,
        &mut cursor,
        structs,
        quantize_f16_fn,
    )
}

fn scalar_count(ty: Type, structs: &[StructDef]) -> usize {
    match ty {
        Type::Struct(index) => structs[index as usize]
            .fields
            .iter()
            .map(|f| scalar_count(f.ty, structs))
            .sum(),
        Type::Vector(_, lanes) => usize::from(lanes),
        Type::Matrix(_, order) => usize::from(order) * usize::from(order),
        Type::Scalar(_) => 1,
    }
}

fn quantize_scalar(
    b: &mut IrBuilder<'_>,
    value: ValueId,
    ty: Type,
    encoding: Encoding,
    quantize_f16_fn: Option<u32>,
) -> ValueId {
    if encoding == Encoding::Word || ty == Type::Bool {
        return value;
    }
    match (ty, encoding) {
        (Type::Float, Encoding::F16) => {
            let f = quantize_f16_fn.expect("F16 quantize helper registered");
            b.push(Type::Float, Op::Call(f, vec![value]))
        }
        (Type::Float, Encoding::Unorm8 | Encoding::Unorm16 | Encoding::Snorm8) => {
            let (low, scale, recip) = match encoding {
                Encoding::Unorm8 => (0.0f32, 255.0f32, 1.0f32 / 255.0f32),
                Encoding::Unorm16 => (0.0f32, 65535.0f32, 1.0f32 / 65535.0f32),
                Encoding::Snorm8 => (-1.0f32, 127.0f32, 1.0f32 / 127.0f32),
                _ => unreachable!(),
            };
            let is_not_nan = b.bin(Type::Bool, BinaryOp::Equal, value, value);
            let clamped = b.push_if(
                Type::Float,
                is_not_nan,
                |b| {
                    let lo = b.float(low);
                    let hi = b.float(1.0);
                    b.intr(Type::Float, Intrinsic::Clamp, vec![value, lo, hi])
                },
                |b| b.float(0.0),
            );
            let s = b.float(scale);
            let scaled = b.bin(Type::Float, BinaryOp::Mul, clamped, s);
            let rounded = b.intr(Type::Float, Intrinsic::Round, vec![scaled]);
            // Round-trip through Int so `-0.0` canonicalizes to `+0.0`,
            // matching `Encoding::encode` (`as i32`) + `Encoding::decode` (`as f32`).
            let as_int = b.intr(Type::Int, Intrinsic::ToInt, vec![rounded]);
            let as_float = b.intr(Type::Float, Intrinsic::ToFloat, vec![as_int]);
            let r = b.float(recip);
            b.bin(Type::Float, BinaryOp::Mul, as_float, r)
        }
        (Type::Int, Encoding::U8) => {
            let mask = b.int(0xFF);
            b.bin(Type::Int, BinaryOp::BitAnd, value, mask)
        }
        (Type::Int, Encoding::U16) => {
            let mask = b.int(0xFFFF);
            b.bin(Type::Int, BinaryOp::BitAnd, value, mask)
        }
        (Type::Int, Encoding::I8) => {
            let c24 = b.int(24);
            let hi = b.bin(Type::Int, BinaryOp::Shl, value, c24);
            b.bin(Type::Int, BinaryOp::Shr, hi, c24)
        }
        (Type::Int, Encoding::I16) => {
            let c16 = b.int(16);
            let hi = b.bin(Type::Int, BinaryOp::Shl, value, c16);
            b.bin(Type::Int, BinaryOp::Shr, hi, c16)
        }
        _ => value,
    }
}

fn quantize_rec(
    b: &mut IrBuilder<'_>,
    value: ValueId,
    ty: Type,
    encodings: &[Encoding],
    cursor: &mut usize,
    structs: &[StructDef],
    quantize_f16_fn: Option<u32>,
) -> ValueId {
    let n = scalar_count(ty, structs);
    if encodings[*cursor..*cursor + n]
        .iter()
        .all(|&e| e == Encoding::Word)
    {
        *cursor += n;
        return value;
    }
    match ty {
        Type::Scalar(_) => {
            let enc = encodings[*cursor];
            *cursor += 1;
            quantize_scalar(b, value, ty, enc, quantize_f16_fn)
        }
        Type::Vector(scalar, lanes) => {
            let lane_ty = Type::Scalar(scalar);
            let mut parts = Vec::with_capacity(usize::from(lanes));
            for k in 0..u32::from(lanes) {
                let extracted = b.push(lane_ty, Op::Extract(value, k));
                let enc = encodings[*cursor];
                *cursor += 1;
                parts.push(quantize_scalar(b, extracted, lane_ty, enc, quantize_f16_fn));
            }
            b.push(ty, Op::Construct(parts))
        }
        Type::Matrix(scalar, order) => {
            let col_ty = Type::Vector(scalar, order);
            let mut cols = Vec::with_capacity(usize::from(order));
            for k in 0..u32::from(order) {
                let extracted = b.push(col_ty, Op::Extract(value, k));
                cols.push(quantize_rec(
                    b,
                    extracted,
                    col_ty,
                    encodings,
                    cursor,
                    structs,
                    quantize_f16_fn,
                ));
            }
            b.push(ty, Op::Construct(cols))
        }
        Type::Struct(index) => {
            let fields: Vec<Type> = structs[index as usize]
                .fields
                .iter()
                .map(|f| f.ty)
                .collect();
            let mut parts = Vec::with_capacity(fields.len());
            for (k, field_ty) in fields.into_iter().enumerate() {
                let extracted = b.push(field_ty, Op::Extract(value, k as u32));
                parts.push(quantize_rec(
                    b,
                    extracted,
                    field_ty,
                    encodings,
                    cursor,
                    structs,
                    quantize_f16_fn,
                ));
            }
            b.push(ty, Op::Construct(parts))
        }
    }
}

/// Rewrites a function body with fresh pre-order ids, applying `ops` to
/// every operation and `types` to every type, and replacing samples of
/// linked inputs by values in hand and their reads by calls.
/// Pairs the producer's struct `written` with the consumer's `read` (a
/// linked port's two ends, laid out alike), and so the structs within
/// them, field by field.
fn link_structs(
    producer: &[StructDef],
    consumer: &[StructDef],
    written: Type,
    read: Type,
    linked: &mut BTreeMap<u32, u32>,
) {
    let (Type::Struct(p), Type::Struct(c)) = (written, read) else {
        return;
    };
    if linked.insert(p, c).is_some() {
        return;
    }
    let (Some(pd), Some(cd)) = (producer.get(p as usize), consumer.get(c as usize)) else {
        return;
    };
    for (pf, cf) in pd.fields.iter().zip(&cd.fields) {
        link_structs(producer, consumer, pf.ty, cf.ty, linked);
    }
    if let (Some(pe), Some(ce)) = (pd.element, cd.element) {
        link_structs(producer, consumer, pe, ce, linked);
    }
}

struct Renumber<'a> {
    next: u32,
    ids: BTreeMap<u32, u32>,
    ops: &'a dyn Fn(&Op) -> Op,
    types: &'a dyn Fn(Type) -> Type,
    /// Inputs whose samples become values in hand, with each value's
    /// element type for walking paths.
    replace: BTreeMap<u32, (ValueId, Type)>,
    /// Inputs whose reads become calls: the function computing the
    /// element at a row, and the element's type.
    called: BTreeMap<u32, (u32, Type)>,
    /// Inputs whose reads become 2D stencil tile calls:
    /// `(tile_fn, tile_ret_ty, optional_struct_field, element_ty)`.
    stencil: BTreeMap<u32, (u32, Type, Option<u32>, Type)>,
    /// Cached tile calls per enclosing block, keyed by `(tile_fn, optional_mapped_addr)`
    /// (`None` for self-coord `Op::Sample` reads), so multi-output producers
    /// evaluate once per unique address.
    tile_calls: Vec<BTreeMap<(u32, Option<u32>), ValueId>>,
    /// The param holding those inputs' count.
    count: u32,
    /// The calls made, by input and address, in each enclosing block.
    calls: Vec<BTreeMap<(u32, u32), ValueId>>,
    /// In a function of the row, the row: the element, and where its own
    /// samples read.
    own: Option<ValueId>,
    /// In a stencil `#tile_r*` function: the evaluated 2D coordinate, domain,
    /// and computed flat element index.
    tile_pos: Option<ValueId>,
    tile_dom: Option<ValueId>,
    tile_elem: Option<ValueId>,
    /// The structs such a path is read against.
    structs: Option<&'a [StructDef]>,
    /// Original value ids in the consumer's entry that evaluate `Op::Coord`,
    /// so `at(grid_or_volume, coord())` folds identically to `sample(...)`.
    coords: BTreeSet<u32>,
}

impl<'a> Renumber<'a> {
    fn new(
        ops: &'a dyn Fn(&Op) -> Op,
        types: &'a dyn Fn(Type) -> Type,
        replace: BTreeMap<u32, (ValueId, Type)>,
    ) -> Self {
        Self {
            next: 0,
            ids: BTreeMap::new(),
            ops,
            types,
            replace,
            called: BTreeMap::new(),
            stencil: BTreeMap::new(),
            tile_calls: vec![BTreeMap::new()],
            count: 0,
            calls: vec![BTreeMap::new()],
            own: None,
            tile_pos: None,
            tile_dom: None,
            tile_elem: None,
            structs: None,
            coords: BTreeSet::new(),
        }
    }

    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.ids.get(&id.0).unwrap_or(&id.0))
    }

    fn results(&self, results: &[ValueId]) -> Vec<ValueId> {
        results.iter().map(|id| self.map(*id)).collect()
    }

    fn bindings(&self, bindings: &[(ValueId, String)]) -> Vec<(ValueId, String)> {
        bindings
            .iter()
            .map(|(id, name)| (self.map(*id), name.clone()))
            .collect()
    }

    fn function(&mut self, function: &Function) -> Function {
        let body = self.block(&function.body);
        Function {
            name: function.name.clone(),
            args: function.args.iter().map(|ty| (self.types)(*ty)).collect(),
            body,
            results: self.results(&function.results),
            bindings: self.bindings(&function.bindings),
        }
    }

    fn nested(&mut self, block: &Block) -> Block {
        self.calls.push(BTreeMap::new());
        self.tile_calls.push(BTreeMap::new());
        let body = self.block(&block.body);
        self.calls.pop();
        self.tile_calls.pop();
        Block {
            body,
            result: self.map(block.result),
        }
    }

    /// Calls `tile_fn(clamped_pos, domain())` once per unique `(tile_fn, address)`
    /// within the enclosing block scopes.
    fn tile_call(
        &mut self,
        out: &mut Vec<Instruction>,
        tile_fn: u32,
        tile_ret_ty: Type,
        address: Option<ValueId>,
        is_exact_coord: bool,
    ) -> ValueId {
        let key = (tile_fn, address.map(|a| a.0));
        if let Some(value) = self
            .tile_calls
            .iter()
            .rev()
            .find_map(|scope| scope.get(&key))
        {
            return *value;
        }
        let dom = self.push(out, Type::Vec2, Op::Domain);
        let pos = match address {
            None => self.push(out, Type::Vec2, Op::Coord),
            Some(addr) if is_exact_coord => addr,
            Some(addr) => {
                let floored =
                    self.push(out, Type::Vec2, Op::Intrinsic(Intrinsic::Floor, vec![addr]));
                let zero = self.push(out, Type::Vec2, Op::Const(Value::Vec2([0.0, 0.0])));
                let one = self.push(out, Type::Vec2, Op::Const(Value::Vec2([1.0, 1.0])));
                let hi = self.push(out, Type::Vec2, Op::Binary(BinaryOp::Sub, dom, one));
                self.push(
                    out,
                    Type::Vec2,
                    Op::Intrinsic(Intrinsic::Clamp, vec![floored, zero, hi]),
                )
            }
        };
        let value = self.push(out, tile_ret_ty, Op::Call(tile_fn, vec![pos, dom]));
        self.tile_calls
            .last_mut()
            .expect("a block's tile calls")
            .insert(key, value);
        value
    }

    /// Appends an instruction with the next id.
    fn push(&mut self, out: &mut Vec<Instruction>, ty: Type, op: Op) -> ValueId {
        let id = ValueId(self.next);
        self.next += 1;
        out.push(Instruction { id, ty, op });
        id
    }

    /// The element of `input` at `address`, computed once per block it is
    /// read in: the address clamped into the extent, as `at` clamps it,
    /// and the producer called there.
    fn call(
        &mut self,
        out: &mut Vec<Instruction>,
        input: u32,
        address: ValueId,
        (function, element): (u32, Type),
    ) -> ValueId {
        let key = (input, address.0);
        if let Some(value) = self.calls.iter().rev().find_map(|scope| scope.get(&key)) {
            return *value;
        }
        use crate::{BinaryOp, Intrinsic};
        let count = self.push(out, Type::Int, Op::Param(self.count));
        let zero = self.push(out, Type::Int, Op::Const(Value::Int(0)));
        let one = self.push(out, Type::Int, Op::Const(Value::Int(1)));
        let last = self.push(out, Type::Int, Op::Binary(BinaryOp::Sub, count, one));
        let row = self.push(
            out,
            Type::Int,
            Op::Intrinsic(Intrinsic::Clamp, vec![address, zero, last]),
        );
        let value = self.push(out, element, Op::Call(function, vec![row]));
        self.calls
            .last_mut()
            .expect("a block's calls")
            .insert(key, value);
        value
    }

    /// A part of an element in hand: extracts and subscripts along `path`.
    fn part(
        &mut self,
        out: &mut Vec<Instruction>,
        mut source: ValueId,
        mut ty: Type,
        path: &[Step],
    ) -> ValueId {
        let structs = self
            .structs
            .expect("a path is walked against the fused structs");
        for step in path {
            let (part, op) = match step {
                Step::Field(k) => (
                    crate::part_at(structs, ty, *k as usize).map(|(_, ty)| ty),
                    Op::Extract(source, *k),
                ),
                Step::Index(index) => (
                    crate::elements_of(structs, ty).map(|(ty, _, _)| ty),
                    Op::Subscript(source, self.map(*index)),
                ),
            };
            let Some(part) = part else { break };
            // The types walked are already the fused table's.
            source = self.push(out, part, op);
            ty = part;
        }
        source
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            if matches!(instruction.op, Op::Coord) {
                self.coords.insert(instruction.id.0);
            }
            if let Op::Sample(i) = &instruction.op
                && let Some((replacement, _)) = self.replace.get(i)
            {
                self.ids.insert(instruction.id.0, replacement.0);
                continue;
            }
            if let Op::At(i, addr) = &instruction.op
                && self.coords.contains(&addr.0)
                && let Some((replacement, _)) = self.replace.get(i)
            {
                self.ids.insert(instruction.id.0, replacement.0);
                continue;
            }
            // A part of that sample (or self-coord `at`) becomes extracts and
            // subscripts of the producer's value.
            if let Op::Part { input: i, at, path } = &instruction.op
                && at.is_none_or(|addr| self.coords.contains(&addr.0))
                && let Some(&(replacement, element)) = self.replace.get(i)
                && self.structs.is_some()
            {
                let source = self.part(&mut out, replacement, element, path);
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            // A read of a called input is the producer's element there.
            let read = match &instruction.op {
                Op::At(i, address) => Some((*i, *address, None)),
                Op::Part {
                    input: i,
                    at: Some(address),
                    path,
                } => Some((*i, *address, Some(path))),
                _ => None,
            };
            if let Some((i, address, path)) = read
                && let Some(&called) = self.called.get(&i)
            {
                let address = self.map(address);
                let value = self.call(&mut out, i, address, called);
                let source = match path {
                    Some(path) => self.part(&mut out, value, called.1, path),
                    None => value,
                };
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            // A read of a stenciled Grid input becomes a call to `#tile_r*`.
            let stencil_read = match &instruction.op {
                Op::Sample(i) => Some((*i, None, None)),
                Op::At(i, address) => Some((*i, Some(*address), None)),
                Op::Part { input: i, at, path } => Some((*i, *at, Some(path))),
                _ => None,
            };
            if let Some((i, orig_addr, path)) = stencil_read
                && let Some(&(tile_fn, tile_ret_ty, field, elem_ty)) = self.stencil.get(&i)
            {
                let exact = orig_addr.is_none_or(|a| self.coords.contains(&a.0));
                let mapped_addr = orig_addr.map(|a| self.map(a));
                let tile_val = self.tile_call(&mut out, tile_fn, tile_ret_ty, mapped_addr, exact);
                let value = match field {
                    Some(k) => self.push(&mut out, elem_ty, Op::Extract(tile_val, k)),
                    None => tile_val,
                };
                let source = match path {
                    Some(path) => self.part(&mut out, value, elem_ty, path),
                    None => value,
                };
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            // In a stencil `#tile_r*` helper function, `coord()`, `domain()`,
            // `element()`, and `sample()` evaluate at the passed `(pos, dom)`.
            if let Some(tile_pos) = self.tile_pos {
                match &instruction.op {
                    Op::Coord => {
                        self.ids.insert(instruction.id.0, tile_pos.0);
                        continue;
                    }
                    Op::Domain => {
                        let dom = self.tile_dom.expect("tile domain arg");
                        self.ids.insert(instruction.id.0, dom.0);
                        continue;
                    }
                    Op::Element => {
                        let elem = self.tile_elem.expect("tile element computed");
                        self.ids.insert(instruction.id.0, elem.0);
                        continue;
                    }
                    Op::Sample(i) => {
                        let op = (self.ops)(&Op::At(*i, tile_pos));
                        let id = self.push(&mut out, (self.types)(instruction.ty), op);
                        self.ids.insert(instruction.id.0, id.0);
                        continue;
                    }
                    Op::Part {
                        input: i,
                        at: None,
                        path,
                    } => {
                        let mut path = path.clone();
                        for step in &mut path {
                            if let Step::Index(index) = step {
                                *index = self.map(*index);
                            }
                        }
                        let op = (self.ops)(&Op::Part {
                            input: *i,
                            at: Some(tile_pos),
                            path,
                        });
                        let id = self.push(&mut out, (self.types)(instruction.ty), op);
                        self.ids.insert(instruction.id.0, id.0);
                        continue;
                    }
                    _ => {}
                }
            }
            // In a function of the row, the element is the row, and a
            // sample of the producer's own input reads there.
            if let Some(own) = self.own {
                let op = match &instruction.op {
                    Op::Element => {
                        self.ids.insert(instruction.id.0, own.0);
                        continue;
                    }
                    Op::Sample(i) => Some((self.ops)(&Op::At(*i, own))),
                    Op::Part {
                        input: i,
                        at: None,
                        path,
                    } => {
                        let mut path = path.clone();
                        for step in &mut path {
                            if let Step::Index(index) = step {
                                *index = self.map(*index);
                            }
                        }
                        Some((self.ops)(&Op::Part {
                            input: *i,
                            at: Some(own),
                            path,
                        }))
                    }
                    _ => None,
                };
                if let Some(op) = op {
                    let id = self.push(&mut out, (self.types)(instruction.ty), op);
                    self.ids.insert(instruction.id.0, id.0);
                    continue;
                }
            }
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
                    let mut op = (self.ops)(other);
                    self.map_operands(&mut op);
                    op
                }
            };
            out.push(Instruction {
                id,
                ty: (self.types)(instruction.ty),
                op,
            });
        }
        out
    }

    fn map_operands(&self, op: &mut Op) {
        match op {
            Op::Unary(_, a) => *a = self.map(*a),
            Op::Binary(_, a, b) => {
                *a = self.map(*a);
                *b = self.map(*b);
            }
            Op::Construct(parts) => {
                for part in parts {
                    *part = self.map(*part);
                }
            }
            Op::Extract(source, _) => *source = self.map(*source),
            Op::Subscript(array, index) => {
                *array = self.map(*array);
                *index = self.map(*index);
            }
            Op::Call(_, args) | Op::Intrinsic(_, args) => {
                for arg in args {
                    *arg = self.map(*arg);
                }
            }
            Op::At(_, coord) | Op::AtLinear(_, coord) => *coord = self.map(*coord),
            Op::AtFiltered { at, lod, grad, .. } => {
                *at = self.map(*at);
                if let Some(lod) = lod {
                    *lod = self.map(*lod);
                }
                if let Some((dx, dy)) = grad {
                    *dx = self.map(*dx);
                    *dy = self.map(*dy);
                }
            }
            Op::Part { at, path, .. } => {
                if let Some(at) = at {
                    *at = self.map(*at);
                }
                for step in path {
                    if let Step::Index(index) = step {
                        *index = self.map(*index);
                    }
                }
            }
            Op::Put(_, address, value) | Op::Exchange(address, value) => {
                *address = self.map(*address);
                *value = self.map(*value);
            }
            Op::Key(code) | Op::Typed(code) | Op::Button(code) | Op::Stick(code) => {
                *code = self.map(*code)
            }
            Op::Spend(counted) | Op::Emit(counted) => *counted = self.map(*counted),
            Op::Look => {}
            Op::Meet(given) => {
                if let Some(given) = given {
                    *given = self.map(*given);
                }
            }
            Op::Const(_)
            | Op::Param(_)
            | Op::Arg(_)
            | Op::Carried
            | Op::Index
            | Op::Coord
            | Op::Domain
            | Op::Time
            | Op::Clock
            | Op::Rate
            | Op::Queued
            | Op::Pointer
            | Op::PointerDown
            | Op::Wheel
            | Op::Pads
            | Op::Window
            | Op::Scale
            | Op::Element
            | Op::Lane
            | Op::Sample(_)
            | Op::Count(_)
            | Op::Size(_) => {}
            Op::If { .. }
            | Op::Match { .. }
            | Op::Loop { .. }
            | Op::Each { .. }
            | Op::Lanes { .. } => {
                unreachable!("blocks are rewritten separately")
            }
        }
    }
}
