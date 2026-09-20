#![warn(missing_docs)]
//! The rules of Kernel execution that every backend keeps, in one place
//! so the interpreter, the JIT and the GPU are one law: what an
//! invocation must look like, how a scan and a scatter are sized, where
//! a scattered value lands and how it meets what is there, the order
//! floats take as keys, the matrix formulas, and the rasterizer's
//! contract. Nothing here executes a Kernel; the backends do, by these.

pub mod matrix;
pub mod raster;

use std::fmt;

use orla_abi::{AbiError, Domain, Invocation, Scalar, Scope, Shape, Type};
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
    /// A parameter has the wrong type.
    ParamType {
        /// Parameter name.
        name: String,
        /// Declared type.
        expected: Type,
        /// Supplied type.
        actual: Type,
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

/// Checks a scatter's invocation: it writes the extent of the input it
/// scatters into; the domain says so. Every backend applies this.
pub fn check_scatter(
    invocation: &Invocation,
    scope: &Scope<'_>,
    base: usize,
) -> Result<Domain, ExecutionError> {
    let domain = scope.size(invocation.inputs[base])?;
    if domain != invocation.domain {
        return Err(ExecutionError::Device(format!(
            "a scatter writes the extent of the input it scatters into ({domain}); its domain cannot be {}",
            invocation.domain
        )));
    }
    Ok(domain)
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
    for ((slot, incoming), scalar) in element.iter_mut().zip(value).zip(scalars) {
        let (a, b) = (*slot, *incoming);
        *slot = match (combine, scalar) {
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
    }
    Ok(())
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
    for (param, value) in manifest.params.iter().zip(&invocation.params) {
        if value.ty() != param.default.ty() {
            return Err(ExecutionError::ParamType {
                name: param.name.clone(),
                expected: param.default.ty(),
                actual: value.ty(),
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
    if let Some(output) = manifest.outputs.first()
        && output.ty.shape != invocation.domain.shape()
    {
        return Err(ExecutionError::DomainShape {
            expected: output.ty.shape,
            actual: invocation.domain.shape(),
        });
    }
    Ok(())
}
