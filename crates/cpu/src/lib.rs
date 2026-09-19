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

use orla_abi::{AbiError, ImageHandle, Invocation, Pixel, PortType, Scope, Type};
use orla_ir::{Validated, schema_of};

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
    /// The backend device failed; the message is backend specific.
    Device(String),
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
            Self::Device(message) => write!(f, "device failed: {message}"),
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
        let strides = output_strides(module);
        let results = compute(module, invocation, scope, &strides)?;
        let domain = invocation.domain;
        let stride: usize = strides.iter().sum();
        let count = domain.width as usize * domain.height as usize;
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
            let created = match &port.ty {
                PortType::Image => {
                    scope
                        .image_create(domain.width, domain.height)
                        .and_then(|handle| {
                            let pixels: Vec<Pixel> = own
                                .chunks_exact(4)
                                .map(|c| {
                                    [
                                        f32::from_bits(c[0]),
                                        f32::from_bits(c[1]),
                                        f32::from_bits(c[2]),
                                        f32::from_bits(c[3]),
                                    ]
                                })
                                .collect();
                            scope.image_write_all(handle, &pixels).map(|()| handle)
                        })
                }
                PortType::Field { schema, .. } => scope
                    .field_create(schema.clone(), domain.width)
                    .and_then(|handle| scope.field_write_all(handle, &own).map(|()| handle)),
            };
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

/// Words per element of each output: four for an image, the schema's count
/// for a Field.
fn output_strides(module: &Validated) -> Vec<usize> {
    module
        .manifest
        .outputs
        .iter()
        .map(|port| match &port.ty {
            PortType::Image => 4,
            PortType::Field { element, .. } => schema_of(*element, &module.structs)
                .map(|schema| schema.words())
                .unwrap_or(0),
        })
        .collect()
}

/// Runs the entry point over every element, reading through `scope` only.
/// Returns every output's words per element, outputs concatenated.
fn compute(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
    strides: &[usize],
) -> Result<Vec<u32>, ExecutionError> {
    let domain = invocation.domain;
    let width = domain.width as usize;
    let stride: usize = strides.iter().sum();
    let row_words = width * stride;
    let mut results = vec![0u32; domain.height as usize * row_words];
    let pixel_count = width * domain.height as usize;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(domain.height as usize)
        .min(pixel_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
    let rows_per_thread = (domain.height as usize).div_ceil(threads);
    let run_rows = |first_row: usize, rows: &mut [u32]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(module, invocation);
        for (offset, row) in rows.chunks_exact_mut(row_words.max(1)).enumerate() {
            let y = (first_row + offset) as u32;
            for x in 0..domain.width {
                let slot = x as usize * stride;
                machine.run_entry_words(scope, x, y, &mut row[slot..slot + stride])?;
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
