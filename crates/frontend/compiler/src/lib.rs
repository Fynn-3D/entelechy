#![warn(missing_docs)]
//! Compiler: `.orla` source → AST → semantic analysis → validated Orla IR.
//!
//! Every diagnostic carries a source position. The compiler executes
//! nothing.

pub mod help;
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
    /// The produced IR failed validation: a compiler bug.
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
    orla_language::on_a_deep_stack(|| {
        let kernel = orla_language::parse(source)?;
        compile_kernel(&kernel, None)
    })
}

/// Compiles a parsed Kernel. `id` names an anonymous inline Kernel; a
/// Kernel that declares its own id keeps it.
pub fn compile_kernel(
    kernel: &orla_language::Kernel,
    id: Option<&orla_ir::KernelId>,
) -> Result<Validated, CompileError> {
    // Every pass below walks the tree or the IR by recursion.
    orla_language::on_a_deep_stack(|| {
        let module = lower::lower(kernel, id)?;
        Ok(orla_ir::validate(orla_ir::optimize(module))?)
    })
}

/// [`compile_kernel`] without optimization: the IR a trace walks, with
/// its calls and the source's own `let`s.
pub fn compile_kernel_unoptimized(
    kernel: &orla_language::Kernel,
    id: Option<&orla_ir::KernelId>,
) -> Result<Validated, CompileError> {
    orla_language::on_a_deep_stack(|| {
        let module = lower::lower(kernel, id)?;
        Ok(orla_ir::validate(module)?)
    })
}

/// [`compile`] without optimization: the reference the optimizer is held
/// to.
pub fn compile_unoptimized(source: &str) -> Result<Validated, CompileError> {
    orla_language::on_a_deep_stack(|| {
        let kernel = orla_language::parse(source)?;
        let module = lower::lower(&kernel, None)?;
        Ok(orla_ir::validate(module)?)
    })
}

/// Evaluates a literal as a plain value: numbers, booleans, negation, and
/// `vec2`/`vec3`/`vec4` of floats. Parameter defaults and graph documents
/// share this grammar.
pub fn literal(expr: &orla_language::Expr) -> Result<orla_ir::Value, CompileError> {
    lower::literal(expr)
}

/// A literal read exact and shaped into the type `ty` names (`Double`,
/// `[DVec2; 4]`, a struct), over a document's struct declarations.
pub fn literal_of_type(
    expr: &orla_language::Expr,
    ty: &orla_language::Spanned<String>,
    structs: &[orla_language::StructDecl],
) -> Result<orla_ir::Value, CompileError> {
    lower::literal_of_type(expr, ty, structs)
}

/// [`literal_in`] for a place that knows its type: written floats stay
/// exact until the type sets their width.
pub fn literal_exact(
    expr: &orla_language::Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<orla_ir::Value, CompileError> {
    lower::literal_exact(expr, fields_of)
}

/// Evaluates an untyped literal of any Kernel value: a struct's fields
/// in the order `fields_of` gives, an array or struct as a record.
/// [`orla_ir::value_as`] shapes it into a type.
pub fn literal_in(
    expr: &orla_language::Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<orla_ir::Value, CompileError> {
    lower::literal_in(expr, fields_of)
}
