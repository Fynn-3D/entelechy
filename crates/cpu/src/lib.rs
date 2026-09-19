#![warn(missing_docs)]
//! CPU backend.
//!
//! Interprets a [`Validated`] module once per output element. The backend is
//! trusted Rust, but the Kernel it runs is not: every Host access goes through
//! the capability-gated [`Scope`], and the backend creates and releases the
//! outputs it allocates so a failed execution leaks nothing.
//!
//! Execution has two phases. The compute phase only reads through the scope
//! and runs rows in parallel; the write phase stores results sequentially.
//! Results are identical to a single-threaded run, and the first failing
//! pixel in row-major order is the one reported.

mod eval;
pub mod raster;

use std::fmt;

use orla_abi::{AbiError, Domain, ImageHandle, Invocation, Scope, Shape, Type};
use orla_ir::{Mode, Validated};

pub use eval::ArithmeticError;

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

/// Elements per thread below which the backend stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// The CPU backend. Stateless; one instance can execute any module.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Executes `module` over `invocation.domain`, returning one handle per
    /// manifest output. The handles are owned by the caller's Host.
    pub fn execute(
        &self,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
        check_invocation(module, invocation)?;
        let strides = output_strides(module);
        if module.manifest.mode == Mode::Reduce {
            return reduce(module, invocation, scope, strides[0]);
        }
        let results = compute(module, invocation, scope, &strides, invocation.domain)?;
        let domain = invocation.domain;
        let stride: usize = strides.iter().sum();
        let count = domain.count();
        let mut outputs = Vec::with_capacity(module.manifest.outputs.len());
        let mut offset = 0;
        for (port, words) in module.manifest.outputs.iter().zip(&strides) {
            // Gather this output's words, element after element.
            let mut own: Vec<u32> = Vec::with_capacity(count * words);
            for element in 0..count {
                let base = element * stride + offset;
                own.extend_from_slice(&results[base..base + words]);
            }
            offset += words;
            let created = scope
                .create(port.ty.schema.clone(), domain)
                .and_then(|handle| scope.write_all(handle, &own).map(|()| handle));
            match created {
                Ok(handle) => outputs.push(handle),
                Err(error) => {
                    release_all(scope, &outputs);
                    return Err(error.into());
                }
            }
        }
        Ok(outputs)
    }
}

/// Words per element of each output: its schema's count.
fn output_strides(module: &Validated) -> Vec<usize> {
    module
        .manifest
        .outputs
        .iter()
        .map(|port| port.ty.schema.words())
        .collect()
}

/// A reduction: the entry over the first input's domain, then a pairwise
/// fold with `merge` in a fixed tree, so the result does not depend on
/// how many threads ran the map. One `Rows` element comes out.
fn reduce(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<ImageHandle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    let mut level = compute(module, invocation, scope, &[stride], domain)?;
    let mut machine = eval::Machine::new(module, invocation, domain);
    let mut count = domain.count();
    while count > 1 {
        let pairs = count / 2;
        let mut next = vec![0u32; count.div_ceil(2) * stride];
        for pair in 0..pairs {
            let (a, b) = (
                &level[2 * pair * stride..(2 * pair + 1) * stride],
                &level[(2 * pair + 1) * stride..(2 * pair + 2) * stride],
            );
            machine.run_merge_words(scope, a, b, &mut next[pair * stride..(pair + 1) * stride])?;
        }
        if count % 2 == 1 {
            next[pairs * stride..].copy_from_slice(&level[(count - 1) * stride..count * stride]);
        }
        level = next;
        count = count.div_ceil(2);
    }
    let port = &module.manifest.outputs[0];
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_all(output, &level) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// Runs the entry point over every element of `domain`, reading through
/// `scope` only. Returns every output's words per element, outputs
/// concatenated.
fn compute(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
    strides: &[usize],
    domain: Domain,
) -> Result<Vec<u32>, ExecutionError> {
    let width = domain.width() as usize;
    let height = domain.height();
    // Rows of every layer, one after another: the third axis is outermost.
    let rows = height as usize * domain.depth() as usize;
    let stride: usize = strides.iter().sum();
    let row_words = width * stride;
    let mut results = vec![0u32; rows * row_words];
    let element_count = width * rows;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows)
        .min(element_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
    let rows_per_thread = rows.div_ceil(threads);
    let run_rows = |first_row: usize, chunk: &mut [u32]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(module, invocation, domain);
        for (offset, row) in chunk.chunks_exact_mut(row_words.max(1)).enumerate() {
            let flat = (first_row + offset) as u32;
            let (y, z) = (flat % height, flat / height);
            for x in 0..domain.width() {
                let slot = x as usize * stride;
                machine.run_entry_words(scope, [x, y, z], &mut row[slot..slot + stride])?;
            }
        }
        Ok(())
    };
    if threads == 1 {
        run_rows(0, &mut results)?;
        return Ok(results);
    }
    let chunk_len = rows_per_thread * row_words.max(1);
    let outcomes: Vec<Result<(), ExecutionError>> = std::thread::scope(|s| {
        let handles: Vec<_> = results
            .chunks_mut(chunk_len)
            .enumerate()
            .map(|(index, chunk)| {
                let run_rows = &run_rows;
                s.spawn(move || run_rows(index * rows_per_thread, chunk))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("row worker panicked"))
            .collect()
    });
    for outcome in outcomes {
        outcome?;
    }
    Ok(results)
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

fn release_all(scope: &mut Scope<'_>, outputs: &[ImageHandle]) {
    for output in outputs {
        // The output was created through this scope, so release cannot be
        // denied; a Host failure here would mean the handle is already gone.
        let _ = scope.release(*output);
    }
}
