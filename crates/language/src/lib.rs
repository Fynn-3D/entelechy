#![warn(missing_docs)]
//! Orla Kernel Language (L5) frontend.
//!
//! The language is deliberately small: a Kernel declares its identity,
//! required capabilities, inputs, parameters, outputs, optional helper
//! functions, and one `compute` block that runs per output pixel. Control
//! flow is expression-shaped: `if c { a } else { b }` and bounded
//! `loop N with x = init until done { next }`. Structs bundle values:
//! `struct Hit { t: Float; id: Int; }` is built with `Hit { t: 1.0, id: 2 }`
//! or updated with `Hit { ..hit, t: 2.0 }`. The frontend produces an AST
//! only; it knows nothing about IR or backends.
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
    /// Struct types in declaration order.
    pub structs: Vec<StructDecl>,
    /// Helper functions in declaration order.
    pub functions: Vec<FnDecl>,
    /// The per-pixel body.
    pub compute: Block,
}

/// `struct Name { field: Type; ... }`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    /// Struct name; must start with an uppercase letter.
    pub name: Spanned<String>,
    /// Fields as `(name, type)`.
    pub fields: Vec<(Spanned<String>, Spanned<String>)>,
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

/// A sequence of `let` bindings ending in a result: `return expr;` in
/// function and `compute` bodies, or a tail expression in `if` and `loop`
/// blocks (both forms are accepted everywhere).
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Bindings in order.
    pub lets: Vec<(Spanned<String>, Expr)>,
    /// The resulting expression.
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
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
    /// `==`
    Equal,
    /// `!=`
    NotEqual,
    /// `&&`
    And,
    /// `||`
    Or,
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
    /// `true` or `false`.
    Bool(bool),
    /// A reference to an input, parameter, argument, or binding.
    Name(String),
    /// A call to a builtin, intrinsic, or helper function.
    Call(String, Vec<Expr>),
    /// Component access such as `c.r`, or a struct field such as `hit.distance`.
    Member(Box<Expr>, String),
    /// `Name { field: expr, ... }`, optionally starting from `..base` whose
    /// unlisted fields are kept.
    Struct {
        /// The struct name.
        name: Spanned<String>,
        /// The value whose fields fill in whatever is not listed.
        base: Option<Box<Expr>>,
        /// Listed fields.
        fields: Vec<(Spanned<String>, Expr)>,
    },
    /// Negation.
    Neg(Box<Expr>),
    /// Logical not.
    Not(Box<Expr>),
    /// Binary arithmetic, comparison, or logic.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// `if condition { then } else { otherwise }`.
    If {
        /// The condition.
        condition: Box<Expr>,
        /// Block taken when the condition holds.
        then: Box<Block>,
        /// Block taken otherwise.
        otherwise: Box<Block>,
    },
    /// `loop count with name = init until done { body }`.
    Loop {
        /// Compile-time iteration bound.
        count: Spanned<u32>,
        /// Name of the carried value inside `until` and `body`.
        name: Spanned<String>,
        /// Initial carried value.
        init: Box<Expr>,
        /// Optional exit test evaluated before each iteration.
        until: Option<Box<Expr>>,
        /// The body, yielding the next carried value.
        body: Box<Block>,
    },
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
