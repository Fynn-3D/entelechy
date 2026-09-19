#![warn(missing_docs)]
//! CPU backend.
//!
//! Interprets a [`Validated`] module once per output pixel. The backend is
//! trusted Rust, but the Kernel it runs is not: every Host access goes through
//! the capability-gated [`Scope`], and the backend creates and releases the
//! output images it allocates so a failed execution leaks nothing.

mod eval;

use std::fmt;

use orla_abi::{AbiError, ImageHandle, Invocation, Scope, Type, Value};
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
        let domain = invocation.domain;
        let mut outputs = Vec::with_capacity(module.manifest.outputs.len());
        for _ in &module.manifest.outputs {
            match scope.image_create(domain.width, domain.height) {
                Ok(handle) => outputs.push(handle),
                Err(error) => {
                    release_all(scope, &outputs);
                    return Err(error.into());
                }
            }
        }
        let mut machine = eval::Machine::new(module, invocation);
        for y in 0..domain.height {
            for x in 0..domain.width {
                let result = machine
                    .run_entry(scope, x, y)
                    .and_then(|results| write_results(scope, &outputs, x, y, &results));
                if let Err(error) = result {
                    release_all(scope, &outputs);
                    return Err(error);
                }
            }
        }
        Ok(outputs)
    }
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

fn write_results(
    scope: &mut Scope<'_>,
    outputs: &[ImageHandle],
    x: u32,
    y: u32,
    results: &[Value],
) -> Result<(), ExecutionError> {
    for (output, value) in outputs.iter().zip(results) {
        let Value::Vec4(pixel) = value else {
            unreachable!("validation guarantees Vec4 results for image outputs")
        };
        scope.image_write(*output, x, y, *pixel)?;
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
