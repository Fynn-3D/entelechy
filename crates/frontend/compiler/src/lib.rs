#![warn(missing_docs)]
//! Compiler: `.ent` source → AST → semantic analysis → validated Entelechy IR.
//!
//! Every diagnostic carries a source position. The compiler executes
//! nothing.

pub mod help;
mod lower;

use std::fmt;

pub use entelechy_ir::Validated;
use entelechy_language::{ParseError, Span};

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
        /// The declaration it is in, when one holds it: a function (its
        /// body or its signature), a struct or a type; a declaration may
        /// be written in another document than the Kernel's.
        declaration: Option<String>,
    },
    /// The produced IR failed validation: a compiler bug.
    Validation(entelechy_ir::ValidationError),
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
            Self::Semantic { span, message, .. } if span.is_placed() => {
                write!(f, "{span}: {message}")
            }
            Self::Semantic { message, .. } => f.write_str(message),
            Self::Validation(error) => write!(f, "invalid IR: {error}"),
        }
    }
}

impl CompileError {
    /// The declaration the error is in, when one holds it: a function, a
    /// struct or a type, by the name the Kernel knows it by.
    pub fn declaration(&self) -> Option<&str> {
        match self {
            Self::Semantic { declaration, .. } => declaration.as_deref(),
            _ => None,
        }
    }

    /// The error, placed in the declaration `name` unless an inner one
    /// holds it.
    pub(crate) fn in_declaration(self, name: &str) -> Self {
        match self {
            Self::Semantic {
                span,
                message,
                declaration: None,
            } => Self::Semantic {
                span,
                message,
                declaration: Some(name.to_owned()),
            },
            other => other,
        }
    }
}

impl std::error::Error for CompileError {}

impl From<ParseError> for CompileError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<entelechy_ir::ValidationError> for CompileError {
    fn from(error: entelechy_ir::ValidationError) -> Self {
        Self::Validation(error)
    }
}

/// Compiles one Kernel source into validated IR.
pub fn compile(source: &str) -> Result<Validated, CompileError> {
    entelechy_language::on_a_deep_stack(|| {
        let kernel = entelechy_language::parse(source)?;
        compile_kernel(&kernel, None)
    })
}

/// Compiles a parsed Kernel. `id` names an anonymous inline Kernel; a
/// Kernel that declares its own id keeps it.
pub fn compile_kernel(
    kernel: &entelechy_language::Kernel,
    id: Option<&entelechy_ir::KernelId>,
) -> Result<Validated, CompileError> {
    // Every pass below walks the tree or the IR by recursion.
    entelechy_language::on_a_deep_stack(|| {
        let module = lower::lower(kernel, id, &[])?;
        let module = entelechy_ir::validate(entelechy_ir::optimize(module))?;
        // A backend that runs a group's lanes one at a time holds what
        // crosses an exchange as an array of its lanes: one past what a
        // value may hold is refused here, where the group is written.
        if let (Some(group), Err(error)) = (&kernel.group, entelechy_ir::lanes(&module)) {
            return Err(CompileError::Semantic {
                span: group.span,
                message: format!(
                    "a group of {} lanes holds each value that crosses an exchange as the array of its lanes, and here one does not fit ({error}); exchange less, or make the group smaller",
                    group.size.value
                ),
                declaration: None,
            });
        }
        Ok(module)
    })
}

/// Checks the functions a document declares, each lowered once whether a
/// Kernel calls it or not, against what the document declares and
/// `inherited`, the structs, types and functions of the documents it
/// uses. A Kernel lowers only the functions it calls; this is where the
/// others are held to the language, a template with its Field
/// parameters bound to ports of their declared types. A function's
/// capabilities are its callers' to hold, so none is refused here.
pub fn check_functions(
    document: &entelechy_language::Document,
    inherited: (
        &[entelechy_language::StructDecl],
        &[entelechy_language::TypeAlias],
        &[entelechy_language::FnDecl],
    ),
) -> Result<(), CompileError> {
    if document.functions.is_empty() {
        return Ok(());
    }
    entelechy_language::on_a_deep_stack(|| {
        let (kernel, own) = functions_kernel(document, inherited)?;
        lower::lower(&kernel, None, &own)?;
        match lower::check_templates(&kernel, &own).into_iter().next() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

/// [`check_functions`], saying every fault its functions hold, each once:
/// what a check lists, so one run shows all a document must mend.
pub fn check_functions_each(
    document: &entelechy_language::Document,
    inherited: (
        &[entelechy_language::StructDecl],
        &[entelechy_language::TypeAlias],
        &[entelechy_language::FnDecl],
    ),
) -> Vec<CompileError> {
    if document.functions.is_empty() {
        return Vec::new();
    }
    entelechy_language::on_a_deep_stack(|| match functions_kernel(document, inherited) {
        Ok((kernel, own)) => {
            let mut faults = lower::check_each(&kernel, &own);
            for error in lower::check_templates(&kernel, &own) {
                if !faults.iter().any(|seen| {
                    seen.to_string() == error.to_string()
                        && seen.declaration() == error.declaration()
                }) {
                    faults.push(error);
                }
            }
            faults
        }
        Err(error) => vec![error],
    })
}

/// A Kernel holding `document`'s functions and what it inherits, and the
/// names of the document's own functions.
fn functions_kernel(
    document: &entelechy_language::Document,
    inherited: (
        &[entelechy_language::StructDecl],
        &[entelechy_language::TypeAlias],
        &[entelechy_language::FnDecl],
    ),
) -> Result<(entelechy_language::Kernel, Vec<String>), CompileError> {
    let mut kernel = entelechy_language::parse(
        "kernel entelechy.functions@1 { requires { field.read; field.write; time.read; input.read; } output result: Rows<Float>; compute { return 0.0; } }",
    )?;
    let (structs, types, functions) = inherited;
    kernel.structs = structs.iter().chain(&document.structs).cloned().collect();
    kernel.types = types.iter().chain(&document.types).cloned().collect();
    kernel.functions = functions
        .iter()
        .chain(&document.functions)
        .cloned()
        .collect();
    let own: Vec<String> = document
        .functions
        .iter()
        .map(|f| f.name.value.clone())
        .collect();
    Ok((kernel, own))
}

/// Where each value of a Kernel's function is written in its document.
pub type Places = std::collections::HashMap<entelechy_ir::ValueId, entelechy_language::Span>;

/// Where the values of an unoptimized module are written: its entry, its
/// `merge` or `init`, and each function by its index. Kept beside the
/// IR, never in it, so where a Kernel is written changes nothing it
/// computes.
#[derive(Clone, Debug, Default)]
pub struct SourcePlaces {
    /// The entry's values.
    pub entry: Places,
    /// A reduction's `merge` or a scan's `init`.
    pub others: Vec<Places>,
    /// Each function's values, by its index in the module.
    pub functions: Vec<Places>,
}

/// [`compile_kernel`] without optimization: the IR a trace walks, with
/// its calls and the source's own `let`s, and where each value is
/// written.
pub fn compile_kernel_unoptimized(
    kernel: &entelechy_language::Kernel,
    id: Option<&entelechy_ir::KernelId>,
) -> Result<(Validated, SourcePlaces), CompileError> {
    entelechy_language::on_a_deep_stack(|| {
        let (module, places) = lower::lower_placed(kernel, id, &[])?;
        Ok((entelechy_ir::validate(module)?, places))
    })
}

/// [`compile`] without optimization: the reference the optimizer is held
/// to.
pub fn compile_unoptimized(source: &str) -> Result<Validated, CompileError> {
    entelechy_language::on_a_deep_stack(|| {
        let kernel = entelechy_language::parse(source)?;
        let module = lower::lower(&kernel, None, &[])?;
        Ok(entelechy_ir::validate(module)?)
    })
}

/// Evaluates a literal as a plain value: numbers, booleans, negation, and
/// `vec2`/`vec3`/`vec4` of floats. Parameter defaults and graph documents
/// share this grammar.
pub fn literal(expr: &entelechy_language::Expr) -> Result<entelechy_ir::Value, CompileError> {
    lower::literal(expr)
}

/// A literal read exact and shaped into the type `ty` names (`Double`,
/// `[DVec2; 4]`, a struct), over a document's struct declarations.
pub fn literal_of_type(
    expr: &entelechy_language::Expr,
    ty: &entelechy_language::Spanned<String>,
    structs: &[entelechy_language::StructDecl],
) -> Result<entelechy_ir::Value, CompileError> {
    lower::literal_of_type(expr, ty, structs)
}

/// [`literal_in`] for a place that knows its type: written floats stay
/// exact until the type sets their width.
pub fn literal_exact(
    expr: &entelechy_language::Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<entelechy_ir::Value, CompileError> {
    lower::literal_exact(expr, fields_of)
}

/// Evaluates an untyped literal of any Kernel value: a struct's fields
/// in the order `fields_of` gives, an array or struct as a record.
/// [`entelechy_ir::value_as`] shapes it into a type.
pub fn literal_in(
    expr: &entelechy_language::Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<entelechy_ir::Value, CompileError> {
    lower::literal_in(expr, fields_of)
}

/// An enum's variants in order, each with the `Int` it is.
pub type Variants = Vec<(String, i32)>;

/// `expr` with each enum variant it names (`Mode.Walk`) written as its
/// `Int`, for a literal outside a Kernel (a scene's); `variants_of` gives
/// an enum's variants, each with its number, by the enum's name.
pub fn with_variants(
    expr: &entelechy_language::Expr,
    variants_of: &dyn Fn(&str) -> Option<Variants>,
) -> Result<entelechy_language::Expr, CompileError> {
    lower::with_variants(expr, variants_of)
}
