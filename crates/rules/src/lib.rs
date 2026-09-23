#![warn(missing_docs)]
//! The rules of Kernel execution that every backend keeps, in one place
//! so the interpreter, the JIT and the GPU are one law: what an
//! invocation must look like, how a scan and a scatter are sized, where
//! a scattered value lands and how it meets what is there, the tree a
//! reduction folds in, the order floats take as keys, the matrix
//! formulas, the rasterizer's contract, and the [`Backend`] trait each
//! backend implements. Nothing here executes a Kernel; the backends do,
//! by these.

pub mod backend;
pub mod raster;
pub mod reduce;

pub use backend::{Backend, Check, NativeOp, Prepared};

use std::fmt;

use orla_abi::{AbiError, Domain, Invocation, Scalar, Scope, Shape};
use orla_ir::{Combine, Mode, Validated};

/// Integer arithmetic that cannot produce a value.
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

/// Why an execution failed. The Kernel version stays valid; only this run
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
    /// The output domain has no elements.
    EmptyDomain,
    /// The output domain's shape is not the output port's.
    DomainShape {
        /// The port's shape.
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
    /// The element at the given place ran more loop iterations than the
    /// Kernel's budget, and stopped there.
    Exhausted {
        /// Element column.
        x: u32,
        /// Element row, rows of every layer counted through.
        y: u32,
        /// The budget it had.
        budget: u32,
    },
    /// The run's elements together ran more loop iterations than the Host
    /// lets one run take.
    Overrun {
        /// The iterations the run could take.
        work: u64,
    },
}

/// A run's total of loop iterations, shared by every thread or
/// invocation that runs its elements. An element hands it the blocks of
/// [`orla_ir::WORK_BLOCK`] iterations it fills as its loops finish, so a
/// run that passes the Host's bound is stopped while it runs; what is
/// left of each element, short of a block, is summed where the element
/// ran and handed on in blocks, and at the end as it is. The run fails
/// when the whole, blocks and what is left, is more than the Host lets
/// it take: an exact sum of every element's iterations, so it is decided
/// alike however the elements are scheduled.
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

    /// Hands the run `blocks` full blocks; `true` once the run has
    /// certainly passed its bound, when the element should stop.
    pub fn spend(&self, blocks: u32) -> bool {
        self.blocks
            .fetch_add(blocks, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(blocks)
            > self.limit
    }

    /// Hands the run what a thread's elements left short of their
    /// blocks, summed: its full blocks as blocks, the rest as it is.
    /// `true` once the run has certainly passed its bound.
    pub fn settle(&self, left: u64) -> bool {
        let block = u64::from(orla_ir::WORK_BLOCK);
        self.left
            .fetch_add(left % block, std::sync::atomic::Ordering::Relaxed);
        self.spend((left / block).min(u64::from(u32::MAX / 2)) as u32)
    }

    /// The run's answer once every element has handed in what it ran:
    /// the failure when the whole is more than its bound.
    pub fn outcome(&self) -> Result<(), ExecutionError> {
        let blocks = u64::from(self.blocks.load(std::sync::atomic::Ordering::Relaxed));
        let left = self.left.load(std::sync::atomic::Ordering::Relaxed);
        Self::judge(self.work, blocks, left)
    }

    /// The answer for a run whose blocks and remainder were counted
    /// elsewhere, a device's record read back.
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

    /// The count of blocks handed, for a backend that adds to it in the
    /// code it generates: the run has passed once the count after an
    /// addition is more than [`Run::limit`].
    pub fn blocks(&self) -> &std::sync::atomic::AtomicU32 {
        &self.blocks
    }

    /// The failure of a run that passed its bound.
    pub fn overrun(&self) -> ExecutionError {
        ExecutionError::Overrun { work: self.work }
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
            Self::DomainShape { expected, actual } => {
                write!(f, "output is a {expected} but the domain is a {actual}")
            }
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
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<AbiError> for ExecutionError {
    fn from(error: AbiError) -> Self {
        Self::Abi(error)
    }
}

/// Checks a scan's invocation: it runs over its first input's rows and
/// writes as many; the domain says so. Every backend applies this.
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

/// Checks a scatter's invocation: its body runs once per element of the
/// domain (the first input's extent unless the node says otherwise), and
/// it writes the extent of the input it scatters into, which is what this
/// returns. Every backend applies this.
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

/// Words an address into a domain takes in a scatter's pair.
pub fn address_words(domain: Domain) -> usize {
    domain.shape().dims() as usize
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

/// The total order `min` and `max` use over floats, as a key that orders
/// like the float: every negative below every positive, negative zero
/// below positive zero, NaN beyond the infinities by its bits. Both
/// backends compare by it, so a parallel minimum on the GPU and a
/// sequential one here pick the same word.
pub fn float_order_key(word: u32) -> i32 {
    let bits = word as i32;
    if bits >= 0 { bits } else { bits ^ 0x7fff_ffff }
}

/// Combines a scattered value into the element already there, scalar by
/// scalar; the one rule every backend follows.
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

/// The order [`float_order_key`] is, for doubles.
pub fn double_order_key(bits: u64) -> i64 {
    let bits = bits as i64;
    if bits >= 0 {
        bits
    } else {
        bits ^ 0x7fff_ffff_ffff_ffff
    }
}

/// Checks that an invocation matches the manifest: input and parameter
/// counts, parameter types, and a non-empty domain. Every backend applies
/// the same rule.
pub fn check_invocation(module: &Validated, invocation: &Invocation) -> Result<(), ExecutionError> {
    let manifest = &module.manifest;
    if invocation.inputs.len() != manifest.inputs.len() {
        return Err(ExecutionError::InputCount {
            expected: manifest.inputs.len(),
            actual: invocation.inputs.len(),
        });
    }
    if invocation.params.len() != manifest.params.len() {
        return Err(ExecutionError::ParamCount {
            expected: manifest.params.len(),
            actual: invocation.params.len(),
        });
    }
    // A value is its type's words, scalar for scalar: what a caller
    // widened or reshaped (`orla_ir::value_as`) arrives exact.
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
    // A scatter's output has the target's shape, whatever its domain.
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

/// The grouping a prefix fold takes on a backend that blocks it.
///
/// The rule a prefix declares is order-free, so where the fold is
/// grouped changes no word; what it changes is how long the two walks
/// are. A block of about the square root of the count makes them the
/// same length: as many blocks as there are elements in one, so the
/// walk over the blocks' totals is no longer than a block's own.
pub mod prefix {
    /// The fewest elements a block holds, below which the blocks
    /// outnumber what a walk over them should cost.
    pub const LEAST: usize = 64;

    /// How many elements a block holds, for `count` of them.
    pub fn block(count: usize) -> usize {
        let root = (count as f64).sqrt().ceil() as usize;
        root.max(LEAST).min(count.max(1))
    }
}
