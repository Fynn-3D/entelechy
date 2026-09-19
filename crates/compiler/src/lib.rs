#![warn(missing_docs)]
//! Compiler: `.orla` source → AST → semantic analysis → validated Orla IR.
//!
//! Every diagnostic carries a source position so an agent can repair the
//! Kernel without guessing. The compiler never executes anything.

mod lower;

use std::fmt;

pub use orla_ir::Validated;
use orla_language::{ParseError, Span};

/// Why compilation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// The source does not parse.
    Parse(ParseError),
    /// The source parses but is not meaningful.
    Semantic {
        /// Where the problem is.
        span: Span,
        /// What is wrong.
        message: String,
    },
    /// The produced IR failed validation. This indicates a compiler bug and
    /// is reported rather than hidden.
    Validation(orla_ir::ValidationError),
}

impl CompileError {
    /// Source position, when the error has one.
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::Parse(error) => Some(error.span),
            Self::Semantic { span, .. } => Some(*span),
            Self::Validation(_) => None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "{error}"),
            Self::Semantic { span, message } => write!(f, "{span}: {message}"),
            Self::Validation(error) => write!(f, "invalid IR: {error}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<ParseError> for CompileError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<orla_ir::ValidationError> for CompileError {
    fn from(error: orla_ir::ValidationError) -> Self {
        Self::Validation(error)
    }
}

/// Compiles one Kernel source into validated IR.
pub fn compile(source: &str) -> Result<Validated, CompileError> {
    let kernel = orla_language::parse(source)?;
    let module = lower::lower(&kernel)?;
    Ok(orla_ir::validate(module)?)
}
