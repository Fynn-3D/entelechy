#![warn(missing_docs)]
//! Orla Kernel Language (L5) frontend.
//!
//! The language is deliberately small: a Kernel declares its identity,
//! required capabilities, inputs, parameters, outputs, optional helper
//! functions, and one `compute` block that runs per output pixel. The
//! frontend produces an AST only; it knows nothing about IR or backends.
//!
//! ```text
//! kernel image.grayscale@1 {
//!     requires { image.read; image.write; }
//!     input source: Image;
//!     param amount: Float = 1.0;
//!     output result: Image;
//!     fn luma(c: Vec4) -> Float {
//!         return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
//!     }
//!     compute {
//!         let c = sample(source);
//!         let g = luma(c);
//!         return mix(c, vec4(g, g, g, c.a), amount);
//!     }
//! }
//! ```

mod lexer;
mod parser;

use std::fmt;

pub use parser::parse;

/// A position in the source, 1-based.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    /// Line number.
    pub line: u32,
    /// Column number in characters.
    pub column: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A value with the position it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spanned<T> {
    /// The value.
    pub value: T,
    /// Where it was written.
    pub span: Span,
}

/// A parsed Kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// The raw id text, e.g. `image.grayscale@1`.
    pub id: Spanned<String>,
    /// Declared capability names.
    pub requires: Vec<Spanned<String>>,
    /// Inputs in declaration order.
    pub inputs: Vec<PortDecl>,
    /// Parameters in declaration order.
    pub params: Vec<ParamDecl>,
    /// Outputs in declaration order.
    pub outputs: Vec<PortDecl>,
    /// Helper functions in declaration order.
    pub functions: Vec<FnDecl>,
    /// The per-pixel body.
    pub compute: Block,
}

/// `input name: Type;` or `output name: Type;`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortDecl {
    /// Port name.
    pub name: Spanned<String>,
    /// Type name as written.
    pub ty: Spanned<String>,
}

/// `param name: Type = default;`
#[derive(Clone, Debug, PartialEq)]
pub struct ParamDecl {
    /// Parameter name.
    pub name: Spanned<String>,
    /// Type name as written.
    pub ty: Spanned<String>,
    /// Default value expression.
    pub default: Expr,
}

/// `fn name(arg: Type, ...) -> Type { ... }`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    /// Function name.
    pub name: Spanned<String>,
    /// Arguments as `(name, type)`.
    pub args: Vec<(Spanned<String>, Spanned<String>)>,
    /// Return type name as written.
    pub ret: Spanned<String>,
    /// Body.
    pub body: Block,
}

/// A sequence of `let` bindings ending in `return`.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Bindings in order.
    pub lets: Vec<(Spanned<String>, Expr)>,
    /// The returned expression.
    pub result: Expr,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    /// The expression.
    pub kind: ExprKind,
    /// Where it starts.
    pub span: Span,
}

/// Expression forms.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// A float literal such as `1.0`.
    Float(f32),
    /// An integer literal such as `8`.
    Int(i32),
    /// A reference to an input, parameter, argument, or binding.
    Name(String),
    /// A call to a builtin, intrinsic, or helper function.
    Call(String, Vec<Expr>),
    /// Component access such as `c.r`.
    Member(Box<Expr>, String),
    /// Negation.
    Neg(Box<Expr>),
    /// Binary arithmetic.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
}

/// A parse failure with a position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Where the problem was detected.
    pub span: Span,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}
