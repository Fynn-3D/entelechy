#![warn(missing_docs)]
//! CPU backend.
//!
//! Interprets a [`Validated`] module once per output pixel. The backend is
//! trusted Rust, but the Kernel it runs is not: every Host access goes through
//! the capability-gated [`Scope`], and the backend creates and releases the
//! output images it allocates so a failed execution leaks nothing.
//!
//! Execution has two phases. The compute phase only reads through the scope
//! and runs rows in parallel; the write phase stores results sequentially.
//! Results are identical to a single-threaded run, and the first failing
//! pixel in row-major order is the one reported.

mod eval;

use std::fmt;

use orla_abi::{AbiError, ImageHandle, Invocation, Pixel, Scope, Type, Value};
use orla_ir::Validated;

pub use eval::ArithmeticError;

/// Why an execution failed. The Kernel version stays valid; only this run
/// produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    /// The invocation supplied the wrong number of image inputs.
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
    /// The output domain has no pixels.
    EmptyDomain,
    /// A Host call was denied or rejected.
    Abi(AbiError),
    /// Integer arithmetic failed at the given pixel.
    Arithmetic {
        /// Pixel column.
        x: u32,
        /// Pixel row.
        y: u32,
        /// What failed.
        error: ArithmeticError,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCount { expected, actual } => {
                write!(
                    f,
                    "kernel expects {expected} image input(s), received {actual}"
                )
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
            Self::EmptyDomain => write!(f, "output domain has no pixels"),
            Self::Abi(error) => write!(f, "{error}"),
            Self::Arithmetic { x, y, error } => write!(f, "at pixel ({x}, {y}): {error}"),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<AbiError> for ExecutionError {
    fn from(error: AbiError) -> Self {
        Self::Abi(error)
    }
}

/// Pixels per thread below which the backend stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// The CPU backend. Stateless; one instance can execute any module.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Executes `module` over `invocation.domain`, returning one image handle
    /// per manifest output. The handles are owned by the caller's Host.
    pub fn execute(
        &self,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
        check_invocation(module, invocation)?;
        let results = compute(module, invocation, scope)?;
        let domain = invocation.domain;
        let output_count = module.manifest.outputs.len();
        let mut outputs = Vec::with_capacity(output_count);
        for _ in 0..output_count {
            match scope.image_create(domain.width, domain.height) {
                Ok(handle) => outputs.push(handle),
                Err(error) => {
                    release_all(scope, &outputs);
                    return Err(error.into());
                }
            }
        }
        let mut pixels = results.chunks_exact(output_count.max(1));
        for y in 0..domain.height {
            for x in 0..domain.width {
                let pixel = pixels.next().unwrap_or(&[]);
                for (output, value) in outputs.iter().zip(pixel) {
                    if let Err(error) = scope.image_write(*output, x, y, *value) {
                        release_all(scope, &outputs);
                        return Err(error.into());
                    }
                }
            }
        }
        Ok(outputs)
    }
}

/// Runs the entry point over every pixel, reading through `scope` only.
/// Returns `outputs` pixels per position, row-major.
fn compute(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<Vec<Pixel>, ExecutionError> {
    let domain = invocation.domain;
    let width = domain.width as usize;
    let output_count = module.manifest.outputs.len();
    let stride = width * output_count;
    let mut results = vec![[0.0; 4]; domain.height as usize * stride];
    let pixel_count = width * domain.height as usize;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(domain.height as usize)
        .min(pixel_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
    let rows_per_thread = (domain.height as usize).div_ceil(threads);
    let run_rows = |first_row: usize, rows: &mut [Pixel]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(module, invocation);
        for (offset, row) in rows.chunks_exact_mut(stride.max(1)).enumerate() {
            let y = (first_row + offset) as u32;
            for x in 0..domain.width {
                let values = machine.run_entry(scope, x, y)?;
                let slot = x as usize * output_count;
                for (target, value) in row[slot..slot + output_count].iter_mut().zip(values) {
                    let Value::Vec4(pixel) = value else {
                        unreachable!("validation guarantees Vec4 results for image outputs")
                    };
                    *target = *pixel;
                }
            }
        }
        Ok(())
    };
    if threads == 1 {
        run_rows(0, &mut results)?;
        return Ok(results);
    }
    let chunk_len = rows_per_thread * stride.max(1);
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

fn check_invocation(module: &Validated, invocation: &Invocation) -> Result<(), ExecutionError> {
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
    if invocation.domain.width == 0 || invocation.domain.height == 0 {
        return Err(ExecutionError::EmptyDomain);
    }
    Ok(())
}

fn release_all(scope: &mut Scope<'_>, outputs: &[ImageHandle]) {
    for output in outputs {
        // The image was created through this scope, so release cannot be
        // denied; a Host failure here would mean the handle is already gone.
        let _ = scope.image_release(*output);
    }
}
