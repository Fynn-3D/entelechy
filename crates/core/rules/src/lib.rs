#![warn(missing_docs)]
//! The rules of Kernel execution every backend keeps: invocation
//! checks, scan and scatter sizing, scatter placement and combining,
//! the reduction tree, float ordering, the rasterizer's contract, and
//! the [`Backend`] trait. The backends execute; this crate rules.

pub mod backend;
pub mod placement;
pub mod raster;
pub mod reduce;

pub use backend::{Backend, Check, Downloading, NativeOp, Prepared};

use std::fmt;

use orla_abi::{AbiError, Domain, Invocation, Scalar, Scope, Shape};
use orla_ir::{Combine, Mode, Validated};

/// Integer arithmetic without a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Integer division or remainder by zero.
    DivisionByZero,
    /// Integer result does not fit in 32 bits.
    Overflow,
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => f.write_str("integer division by zero"),
            Self::Overflow => f.write_str("integer overflow"),
        }
    }
}

/// Why an execution failed. The Kernel version stays valid; the run
/// produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    /// The invocation supplied the wrong number of inputs.
    InputCount {
        /// Inputs the manifest declares.
        expected: usize,
        /// Inputs supplied.
        actual: usize,
    },
    /// The invocation supplied the wrong number of parameters.
    ParamCount {
        /// Parameters the manifest declares.
        expected: usize,
        /// Parameters supplied.
        actual: usize,
    },
    /// A parameter's value is not of its type's layout.
    ParamType {
        /// Parameter name.
        name: String,
        /// Declared type, as the Kernel names it.
        expected: String,
        /// The value supplied.
        actual: String,
    },
    /// An input's Field differs from its port in shape or element layout.
    InputType {
        /// Input name.
        name: String,
        /// What differs.
        message: String,
    },
    /// The output domain has no elements.
    EmptyDomain,
    /// The output domain's shape is not the output port's.
    DomainShape {
        /// The port's shape.
        expected: Shape,
        /// The invocation's shape.
        actual: Shape,
    },
    /// A scatter's domain is not its first input's shape, which
    /// `element()`, `coord()` and `domain()` read.
    ScatterDomain {
        /// The first input's name.
        input: String,
        /// Its shape.
        expected: Shape,
        /// The invocation's shape.
        actual: Shape,
    },
    /// A Host call was denied or rejected.
    Abi(AbiError),
    /// The backend device failed; the message is backend specific.
    Device(String),
    /// Integer arithmetic failed at the given element.
    Arithmetic {
        /// Element column.
        x: u32,
        /// Element row, rows of every layer counted through.
        y: u32,
        /// What failed.
        error: ArithmeticError,
    },
    /// The element at the given place passed the Kernel's loop budget and
    /// stopped there.
    Exhausted {
        /// Element column.
        x: u32,
        /// Element row, rows of every layer counted through.
        y: u32,
        /// The budget it had.
        budget: u32,
    },
    /// The run's elements together passed the Host's loop bound for a run.
    Overrun {
        /// The iterations the run could take.
        work: u64,
    },
    /// What a run held in Host memory beyond its Fields (a scatter's
    /// pairs, a reduction's partial values) passed the room the Host had
    /// when the run began.
    OverRoom {
        /// The words the Host had room for.
        room: usize,
    },
}

/// A run's total of loop iterations, shared by every thread that runs
/// its elements. Elements hand in full blocks of [`orla_ir::WORK_BLOCK`]
/// iterations as they go, so a run past the Host's bound stops early;
/// remainders are summed per thread and handed in at the end. The run
/// fails when the exact total passes the bound, whatever the schedule.
pub struct Run {
    blocks: std::sync::atomic::AtomicU32,
    left: std::sync::atomic::AtomicU64,
    limit: u32,
    work: u64,
}

impl Run {
    /// A run that may take `work` iterations ([`orla_abi::Scope::work`]).
    pub fn new(work: u64) -> Self {
        let blocks = (work / u64::from(orla_ir::WORK_BLOCK)).min(u64::from(u32::MAX / 2));
        Self {
            blocks: std::sync::atomic::AtomicU32::new(0),
            left: std::sync::atomic::AtomicU64::new(0),
            limit: blocks as u32,
            work,
        }
    }

    /// Hands the run `blocks` full blocks; `true` once the run has passed
    /// its bound and the element should stop.
    pub fn spend(&self, blocks: u32) -> bool {
        self.blocks
            .fetch_add(blocks, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(blocks)
            > self.limit
    }

    /// Hands the run a thread's summed remainders. `true` once the run
    /// has passed its bound.
    pub fn settle(&self, left: u64) -> bool {
        let block = u64::from(orla_ir::WORK_BLOCK);
        self.left
            .fetch_add(left % block, std::sync::atomic::Ordering::Relaxed);
        self.spend((left / block).min(u64::from(u32::MAX / 2)) as u32)
    }

    /// The run's verdict once every element has handed in: the failure
    /// when the total passes the bound.
    pub fn outcome(&self) -> Result<(), ExecutionError> {
        let blocks = u64::from(self.blocks.load(std::sync::atomic::Ordering::Relaxed));
        let left = self.left.load(std::sync::atomic::Ordering::Relaxed);
        Self::judge(self.work, blocks, left)
    }

    /// The verdict for blocks and remainder counted elsewhere (a
    /// device's record read back).
    pub fn judge(work: u64, blocks: u64, left: u64) -> Result<(), ExecutionError> {
        match blocks * u64::from(orla_ir::WORK_BLOCK) + left > work {
            true => Err(ExecutionError::Overrun { work }),
            false => Ok(()),
        }
    }

    /// The iterations the run may take.
    pub fn work(&self) -> u64 {
        self.work
    }

    /// The blocks a run may take, as a device is told.
    pub fn limit(&self) -> u32 {
        self.limit
    }

    /// The count of blocks handed, for generated code that adds to it:
    /// the run has passed once the count exceeds [`Run::limit`].
    pub fn blocks(&self) -> &std::sync::atomic::AtomicU32 {
        &self.blocks
    }

    /// The failure of a run that passed its bound.
    pub fn overrun(&self) -> ExecutionError {
        ExecutionError::Overrun { work: self.work }
    }
}

/// The words a reduction or a prefix over `count` elements of `stride`
/// words holds at once, every backend alike: every element's value, and
/// for a reduction the level above it, half as many, for a prefix the
/// output it then fills, as many.
pub fn held_values(reduction: bool, count: usize, stride: usize) -> usize {
    let also = if reduction { count.div_ceil(2) } else { count };
    count.saturating_add(also).saturating_mul(stride)
}

/// The Host words a run holds beyond its Fields (a scatter's pairs while
/// applied, a reduction's or a prefix's values), against the room the
/// Host had when the run began
/// ([`orla_abi::Scope::room`]). Shared by every thread of the run.
pub struct Held {
    words: std::sync::atomic::AtomicUsize,
    room: usize,
}

impl Held {
    /// Nothing held yet, against `room` words.
    pub fn new(room: usize) -> Self {
        Self {
            words: std::sync::atomic::AtomicUsize::new(0),
            room,
        }
    }

    /// Holds `words` more; `false`, holding nothing, when that would pass
    /// the room.
    pub fn take(&self, words: usize) -> bool {
        let before = self
            .words
            .fetch_add(words, std::sync::atomic::Ordering::Relaxed);
        if before.saturating_add(words) > self.room {
            self.words
                .fetch_sub(words, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        true
    }

    /// The failure of a run whose held words passed the room.
    pub fn over(&self) -> ExecutionError {
        ExecutionError::OverRoom { room: self.room }
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCount { expected, actual } => {
                write!(f, "kernel expects {expected} input(s), received {actual}")
            }
            Self::ParamCount { expected, actual } => {
                write!(
                    f,
                    "kernel expects {expected} parameter(s), received {actual}"
                )
            }
            Self::ParamType {
                name,
                expected,
                actual,
            } => write!(
                f,
                "parameter `{name}` expects {expected}, received {actual}"
            ),
            Self::EmptyDomain => write!(f, "output domain has no elements"),
            Self::InputType { name, message } => write!(f, "input `{name}` {message}"),
            Self::DomainShape { expected, actual } => {
                write!(f, "output is a {expected} but the domain is a {actual}")
            }
            Self::ScatterDomain {
                input,
                expected,
                actual,
            } => write!(
                f,
                "a scatter runs over its first input's shape: `{input}` is a {expected} but the domain is a {actual}; give it a {expected} domain, or declare an input of {actual} first"
            ),
            Self::Abi(error) => write!(f, "{error}"),
            Self::Device(message) => write!(f, "device failed: {message}"),
            Self::Arithmetic { x, y, error } => write!(f, "at element ({x}, {y}): {error}"),
            Self::Exhausted { x, y, budget } => write!(
                f,
                "at element ({x}, {y}): more than {budget} loop iterations, the Kernel's budget (`budget N;` sets another)"
            ),
            Self::Overrun { work } => write!(
                f,
                "the run's elements together took more than {work} loop iterations, what the Host lets one run take (`--work` sets another)"
            ),
            Self::OverRoom { room } => write!(
                f,
                "the run held more than the {room} words the Host had room for (a scatter's pairs, a reduction's values); a larger Host budget holds them"
            ),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<AbiError> for ExecutionError {
    fn from(error: AbiError) -> Self {
        Self::Abi(error)
    }
}

/// Checks a scan's invocation: its domain is its first input's rows.
pub fn check_scan(invocation: &Invocation, scope: &Scope<'_>) -> Result<Domain, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    if domain != invocation.domain {
        return Err(ExecutionError::Device(format!(
            "a scan writes one element per element of its first input ({domain}); its domain cannot be {}",
            invocation.domain
        )));
    }
    Ok(domain)
}

/// Checks a scatter's invocation: its body runs once per domain element
/// (the first input's extent by default). Returns the extent of the
/// input it scatters into, which it writes.
pub fn check_scatter(
    invocation: &Invocation,
    scope: &Scope<'_>,
    base: usize,
) -> Result<Domain, ExecutionError> {
    if invocation.domain.is_empty() {
        return Err(ExecutionError::EmptyDomain);
    }
    Ok(scope.size(invocation.inputs[base])?)
}

/// Where a module's outputs take their domain from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    /// The invocation's domain: the elements the entry runs over.
    Invocation,
    /// One row.
    OneRow,
    /// The extent of the input at this index.
    Input(usize),
}

/// Where a module's outputs take their domain from, by mode: a map's and
/// a scan's the invocation's, a reduction's one row, a prefix's its first
/// input's extent, a scatter's its target's. The invocation's domain is
/// the elements the entry runs over; an output's is what its readers see.
pub fn extent(mode: Mode) -> Extent {
    match mode {
        Mode::Map | Mode::Scan => Extent::Invocation,
        Mode::Reduce => Extent::OneRow,
        Mode::Prefix { .. } => Extent::Input(0),
        Mode::Scatter { base, .. } => Extent::Input(base as usize),
    }
}

/// The domain a module's outputs have ([`extent`]). Known before the run,
/// so its storage can be reserved.
pub fn output_domain(mode: Mode, domain: Domain, inputs: &[Domain]) -> Domain {
    match extent(mode) {
        Extent::Invocation => domain,
        Extent::OneRow => Domain::Rows(1),
        Extent::Input(at) => inputs.get(at).copied().unwrap_or(domain),
    }
}

/// The threads this machine runs at once, asked of the system once.
pub fn cores() -> usize {
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Words an address into a domain takes in a scatter's pair.
pub fn address_words(domain: Domain) -> usize {
    domain.shape().dims() as usize
}

/// The element one axis of a float address picks: floored, NaN as zero,
/// and clamped into `0..limit` as an integer, so no rounding of `limit`
/// to a float moves it past the last element.
pub fn clamp_axis(value: f32, limit: u32) -> u32 {
    let floored = if value.is_nan() { 0.0 } else { value.floor() };
    // `as` saturates: below zero is zero, past the largest is the largest.
    (floored as u32).min(limit.saturating_sub(1))
}

/// One axis of a linear read at `value`: element centres sit at
/// half-integers, so the position less a half gives the lower neighbour,
/// the upper one, each clamped as [`clamp_axis`] clamps, and the upper
/// one's weight, none before the first element.
pub fn linear_axis(value: f32, limit: u32) -> (u32, u32, f32) {
    let t = if value.is_nan() { 0.0 } else { value - 0.5 };
    let last = limit.saturating_sub(1);
    let lower = (t.floor() as u32).min(last);
    let upper = lower.saturating_add(1).min(last);
    let weight = (t - t.floor()).clamp(0.0, 1.0);
    let weight = if t < 0.0 { 0.0 } else { weight };
    (lower, upper, weight)
}

/// The linear index a scattered address names, or `None` when it falls
/// outside the target (NaN included): the write is dropped, never clamped.
pub fn scatter_index(domain: Domain, address: &[u32]) -> Option<usize> {
    if domain.shape() == Shape::Rows {
        let i = address[0] as i32;
        return (i >= 0 && (i as usize) < domain.count()).then_some(i as usize);
    }
    let axis = |word: u32, limit: u32| -> Option<u32> {
        let v = f32::from_bits(word).floor();
        (v >= 0.0 && v < limit as f32).then_some(v as u32)
    };
    let x = axis(address[0], domain.width())?;
    let y = axis(address[1], domain.height())?;
    let z = if domain.shape() == Shape::Volume {
        axis(address[2], domain.depth())?
    } else {
        0
    };
    Some(((z * domain.height() + y) * domain.width() + x) as usize)
}

/// `max` of two floats as every backend computes it (IEEE 754-2019
/// maximumNumber): a NaN gives way to the other operand, and
/// `max(-0.0, 0.0)` is `0.0` in either order.
pub fn max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a > b {
        a
    } else if b > a {
        b
    } else {
        f32::from_bits(a.to_bits() & b.to_bits())
    }
}

/// `min` of two floats: [`max_f32`]'s rule the other way, so of two
/// equal values the one with a sign, `min(-0.0, 0.0)` is `-0.0`.
pub fn min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a < b {
        a
    } else if b < a {
        b
    } else {
        f32::from_bits(a.to_bits() | b.to_bits())
    }
}

/// [`max_f32`] for doubles.
pub fn max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a > b {
        a
    } else if b > a {
        b
    } else {
        f64::from_bits(a.to_bits() & b.to_bits())
    }
}

/// [`min_f32`] for doubles.
pub fn min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a < b {
        a
    } else if b < a {
        b
    } else {
        f64::from_bits(a.to_bits() | b.to_bits())
    }
}

/// The total order `min` and `max` use over floats, as a key: negatives
/// below positives, negative zero below positive zero, NaN beyond the
/// infinities by its bits. Every backend compares by it.
pub fn float_order_key(word: u32) -> i32 {
    let bits = word as i32;
    if bits >= 0 { bits } else { bits ^ 0x7fff_ffff }
}

/// Combines a scattered value into the element already there, scalar by
/// scalar.
pub fn combine_words(
    combine: Combine,
    scalars: &[Scalar],
    element: &mut [u32],
    value: &[u32],
) -> Result<(), ArithmeticError> {
    let mut k = 0;
    while k < element.len().min(value.len()).min(scalars.len()) {
        let scalar = scalars[k];
        if scalar.words() == 2 {
            let pair = |words: &[u32]| u64::from(words[k]) | u64::from(words[k + 1]) << 32;
            let (a, b) = (pair(element), pair(value));
            let combined = match (combine, scalar) {
                (Combine::Last, _) => b,
                (Combine::Sum, Scalar::Double) => (f64::from_bits(a) + f64::from_bits(b)).to_bits(),
                (Combine::Sum, _) => (a as i64)
                    .checked_add(b as i64)
                    .ok_or(ArithmeticError::Overflow)? as u64,
                (Combine::Min, Scalar::Double) => {
                    if double_order_key(b) < double_order_key(a) {
                        b
                    } else {
                        a
                    }
                }
                (Combine::Min, _) => (a as i64).min(b as i64) as u64,
                (Combine::Max, Scalar::Double) => {
                    if double_order_key(b) > double_order_key(a) {
                        b
                    } else {
                        a
                    }
                }
                (Combine::Max, _) => (a as i64).max(b as i64) as u64,
            };
            element[k] = combined as u32;
            element[k + 1] = (combined >> 32) as u32;
            k += 2;
            continue;
        }
        let (a, b) = (element[k], value[k]);
        element[k] = match (combine, scalar) {
            (Combine::Last, _) => b,
            (Combine::Sum, Scalar::Float) => (f32::from_bits(a) + f32::from_bits(b)).to_bits(),
            (Combine::Sum, _) => (a as i32)
                .checked_add(b as i32)
                .ok_or(ArithmeticError::Overflow)? as u32,
            (Combine::Min, Scalar::Float) => {
                if float_order_key(b) < float_order_key(a) {
                    b
                } else {
                    a
                }
            }
            (Combine::Min, _) => (a as i32).min(b as i32) as u32,
            (Combine::Max, Scalar::Float) => {
                if float_order_key(b) > float_order_key(a) {
                    b
                } else {
                    a
                }
            }
            (Combine::Max, _) => (a as i32).max(b as i32) as u32,
        };
        k += 1;
    }
    Ok(())
}

/// A scatter's `sum`, the rule every backend's scatter keeps. An integer
/// element becomes the exact total of what it held and every value
/// scattered to it, failing only when that total overflows its scalar;
/// the first element that overflows is the one reported, in any arrival
/// order. A float is added in element order.
pub struct Sums {
    scalars: Vec<Scalar>,
    stride: usize,
    /// The word each integer scalar starts at within an element, and
    /// whether it is a `Long`.
    integers: Vec<(usize, bool)>,
    /// Each element's integer totals, whole.
    totals: Vec<i128>,
}

impl Sums {
    /// The Host words the sums of `elements` elements of `scalars` hold,
    /// four per total, to be held ([`Held`]) before they are made.
    pub fn held_words(scalars: &[Scalar], elements: usize) -> usize {
        let integers = starts(scalars)
            .filter(|(_, scalar)| matches!(scalar, Scalar::Int | Scalar::Long))
            .count();
        elements.saturating_mul(integers).saturating_mul(4)
    }

    /// The sums of a scatter into `words`, the target's elements of
    /// `scalars` in order: its totals start at what each element holds.
    pub fn new(scalars: &[Scalar], words: &[u32]) -> Self {
        let integers: Vec<(usize, bool)> = starts(scalars)
            .filter_map(|(at, scalar)| match scalar {
                Scalar::Int => Some((at, false)),
                Scalar::Long => Some((at, true)),
                _ => None,
            })
            .collect();
        let stride = scalars.len().max(1);
        let count = words.len() / stride * integers.len();
        let mut totals = Vec::with_capacity(count);
        for element in words.chunks_exact(stride) {
            for &(at, long) in &integers {
                totals.push(integer(element, at, long));
            }
        }
        Self {
            scalars: scalars.to_vec(),
            stride,
            integers,
            totals,
        }
    }

    /// Adds `value` into `element`: its integers to their totals, its
    /// floats to the words in `words`, in the order they are added.
    pub fn add(&mut self, words: &mut [u32], element: usize, value: &[u32]) {
        let slot = &mut words[element * self.stride..(element + 1) * self.stride];
        for (at, scalar) in starts(&self.scalars) {
            match scalar {
                Scalar::Float => {
                    slot[at] = (f32::from_bits(slot[at]) + f32::from_bits(value[at])).to_bits();
                }
                Scalar::Double => {
                    let pair = |w: &[u32]| u64::from(w[at]) | u64::from(w[at + 1]) << 32;
                    let sum = (f64::from_bits(pair(slot)) + f64::from_bits(pair(value))).to_bits();
                    slot[at] = sum as u32;
                    slot[at + 1] = (sum >> 32) as u32;
                }
                _ => {}
            }
        }
        let n = self.integers.len();
        for (k, &(at, long)) in self.integers.iter().enumerate() {
            let total = &mut self.totals[element * n + k];
            *total = total.saturating_add(integer(value, at, long));
        }
    }

    /// Writes every total into `words`, or fails at the first element
    /// whose total overflows its scalar.
    pub fn finish(self, words: &mut [u32]) -> Result<(), (usize, ArithmeticError)> {
        let n = self.integers.len();
        if n == 0 {
            return Ok(());
        }
        for (element, totals) in self.totals.chunks_exact(n).enumerate() {
            for (&(at, long), &total) in self.integers.iter().zip(totals) {
                let at = element * self.stride + at;
                if long {
                    let total =
                        i64::try_from(total).map_err(|_| (element, ArithmeticError::Overflow))?;
                    words[at] = total as u32;
                    words[at + 1] = (total >> 32) as u32;
                } else {
                    let total =
                        i32::try_from(total).map_err(|_| (element, ArithmeticError::Overflow))?;
                    words[at] = total as u32;
                }
            }
        }
        Ok(())
    }
}

/// Each scalar of a layout at its first word; a layout names a `Double`
/// or a `Long` once per word.
fn starts(scalars: &[Scalar]) -> impl Iterator<Item = (usize, Scalar)> + '_ {
    let mut at = 0;
    std::iter::from_fn(move || {
        let scalar = *scalars.get(at)?;
        let start = at;
        at += scalar.words();
        Some((start, scalar))
    })
}

/// The integer at word `at` of `words`, a `Long` over two.
fn integer(words: &[u32], at: usize, long: bool) -> i128 {
    if long {
        i128::from((u64::from(words[at]) | u64::from(words[at + 1]) << 32) as i64)
    } else {
        i128::from(words[at] as i32)
    }
}

/// The order [`float_order_key`] is, for doubles.
pub fn double_order_key(bits: u64) -> i64 {
    let bits = bits as i64;
    if bits >= 0 {
        bits
    } else {
        bits ^ 0x7fff_ffff_ffff_ffff
    }
}

/// Asks the scope for room for what a run on the Host holds while it
/// computes: each output's elements in whole words, before a narrow
/// output is packed. Asked before any words are computed.
pub fn outputs_fit_words(
    module: &Validated,
    domain: Domain,
    scope: &Scope<'_>,
) -> Result<(), ExecutionError> {
    let wide: Vec<orla_abi::Schema> = module
        .manifest
        .outputs
        .iter()
        .map(|port| orla_abi::Schema::new(port.ty.schema.scalars().to_vec()))
        .collect();
    let fields: Vec<_> = wide.iter().map(|schema| (schema, domain)).collect();
    Ok(scope.can_hold(&fields)?)
}

/// Checks that an invocation matches the manifest: input and parameter
/// counts, each input's shape and element layout, parameter types, and a
/// non-empty domain. Every backend applies it, so reading an input by its
/// port's layout is safe for any caller.
pub fn check_invocation(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(), ExecutionError> {
    let manifest = &module.manifest;
    if invocation.inputs.len() != manifest.inputs.len() {
        return Err(ExecutionError::InputCount {
            expected: manifest.inputs.len(),
            actual: invocation.inputs.len(),
        });
    }
    for (port, handle) in manifest.inputs.iter().zip(&invocation.inputs) {
        let (shape, schema) = scope.bound(*handle)?;
        let message = if shape != port.ty.shape {
            format!("expects {} but is bound to a {shape}", port.ty.shape)
        } else if schema != port.ty.schema {
            format!(
                "expects an element of {} but is bound to a Field of {}",
                port.ty.schema.describe(),
                schema.describe()
            )
        } else {
            continue;
        };
        return Err(ExecutionError::InputType {
            name: port.name.clone(),
            message,
        });
    }
    if invocation.params.len() != manifest.params.len() {
        return Err(ExecutionError::ParamCount {
            expected: manifest.params.len(),
            actual: invocation.params.len(),
        });
    }
    // A value is its type's words, scalar for scalar.
    for (param, value) in manifest.params.iter().zip(&invocation.params) {
        let wanted = orla_ir::schema_of(param.ty, &module.structs).map(|s| s.scalars().to_vec());
        if wanted.as_deref() != Some(value.scalars().as_slice()) {
            return Err(ExecutionError::ParamType {
                name: param.name.clone(),
                expected: orla_ir::type_name(param.ty, &module.structs),
                actual: value.to_string(),
            });
        }
    }
    if invocation.domain.is_empty() {
        return Err(ExecutionError::EmptyDomain);
    }
    if manifest.mode == Mode::Reduce {
        if invocation.domain != Domain::Rows(1) {
            return Err(ExecutionError::Device(
                "a reduction produces one element; its domain is 1".into(),
            ));
        }
        return Ok(());
    }
    // A scatter's output has the target's shape; its domain has its first
    // input's shape, the one `element()`, `coord()` and `domain()` read.
    if let (Mode::Scatter { .. }, Some(first)) = (manifest.mode, manifest.inputs.first())
        && first.ty.shape != invocation.domain.shape()
    {
        return Err(ExecutionError::ScatterDomain {
            input: first.name.clone(),
            expected: first.ty.shape,
            actual: invocation.domain.shape(),
        });
    }
    if let Some(output) = manifest.outputs.first()
        && !matches!(manifest.mode, Mode::Scatter { .. })
        && output.ty.shape != invocation.domain.shape()
    {
        return Err(ExecutionError::DomainShape {
            expected: output.ty.shape,
            actual: invocation.domain.shape(),
        });
    }
    Ok(())
}

/// The grouping a prefix fold takes on a backend that blocks it. A
/// prefix's rule is order-free, so grouping changes no word; a block of
/// about the square root of the count balances the two walks.
pub mod prefix {
    /// The fewest elements a block holds.
    pub const LEAST: usize = 64;

    /// How many elements a block holds, for `count` of them.
    pub fn block(count: usize) -> usize {
        let root = (count as f64).sqrt().ceil() as usize;
        root.max(LEAST).min(count.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Double` and a `Long` are each summed once, from their first
    /// word, beside an `Int` and a `Float`; an overflow names the first
    /// element whose total overflows.
    #[test]
    fn sums_take_each_scalar_once_where_it_starts() {
        use Scalar::{Double, Float, Int, Long};
        let scalars = [Double, Double, Long, Long, Int, Float];
        let element = |d: f64, l: i64, i: i32, f: f32| {
            let (d, l) = (d.to_bits(), l as u64);
            vec![
                d as u32,
                (d >> 32) as u32,
                l as u32,
                (l >> 32) as u32,
                i as u32,
                f.to_bits(),
            ]
        };
        let mut words = [element(1.5, 5_000_000_000, 7, 0.5), element(0.0, 0, 0, 0.0)].concat();
        assert_eq!(Sums::held_words(&scalars, 2), 2 * 2 * 4);
        let mut sums = Sums::new(&scalars, &words);
        sums.add(
            &mut words,
            0,
            &element(0.25, i64::MAX - 5_000_000_000, i32::MAX, 1.0),
        );
        sums.add(&mut words, 0, &element(0.25, -1, -8, 1.0));
        sums.finish(&mut words).unwrap();
        assert_eq!(
            &words[..6],
            &element(2.0, i64::MAX - 1, i32::MAX - 1, 2.5)[..]
        );
        // Past the end of a `Long`, at the second element.
        let mut words = [element(0.0, 0, 0, 0.0), element(0.0, i64::MAX, 0, 0.0)].concat();
        let mut sums = Sums::new(&scalars, &words);
        sums.add(&mut words, 1, &element(0.0, 1, 0, 0.0));
        assert_eq!(sums.finish(&mut words), Err((1, ArithmeticError::Overflow)));
    }
}
