#![warn(missing_docs)]
//! Orla language frontend (L5).
//!
//! One grammar serves two file roles. `.orla` files hold Kernels: pure
//! capabilities, one or more per file. `.osn` files are node graphs: the
//! file is the graph. A graph uses libraries, declares its inputs and
//! outputs, and places nodes; a node either references a Kernel by id or
//! carries its own Kernel inline, wiring its inputs where they are declared.
//!
//! ```text
//! use "kernels/";
//!
//! input photo: Image;
//!
//! node gray: image.grayscale@1 {
//!     source <- photo;
//!     amount = 1.0;
//! }
//! node trail: kernel {
//!     requires { image.read; image.write; }
//!     input current: Image <- gray.result;
//!     input previous: Image <~ trail.result from gray.result;
//!     output result: Image;
//!     compute { ... }
//! }
//!
//! output result <- trail.result;
//! ```
//!
//! Wiring is written only at the consumer: `port <- source` reads the source
//! in this evaluation, `port <~ source from initial` reads it as committed by
//! the previous evaluation. A source is a graph input by name or `node.port`.
//!
//! Kernel syntax:
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
use std::ops::Range;

pub use parser::{parse, parse_document, parse_literal};

/// A whole file. For a `.osn` the file is the graph; for a `.orla` only
/// `kernels` is populated.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    /// Files or directories whose Kernels this document uses.
    pub uses: Vec<Use>,
    /// Struct types declared at the top level; every Kernel of the
    /// document sees them before its own.
    pub structs: Vec<StructDecl>,
    /// Kernels defined at the top level, document structs already prepended.
    pub kernels: Vec<Kernel>,
    /// External images the caller supplies: `input name: Image;`.
    pub inputs: Vec<PortDecl>,
    /// Graph parameters: `param name: Type = default;`. Node values may
    /// name them, and a parent graph or the caller may override them.
    pub params: Vec<ParamDecl>,
    /// Named results: `output name <- source;`.
    pub outputs: Vec<(Spanned<String>, SourceRef)>,
    /// Nodes in document order.
    pub nodes: Vec<NodeDecl>,
}

/// `use "path";`: a file of Kernels, or a directory of such files when the
/// path ends with `/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Use {
    /// The path as written, relative to the document.
    pub path: Spanned<String>,
}

/// `node name: kernel@1 { ... }` or `node name: kernel [id@1] { ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDecl {
    /// Node name.
    pub name: Spanned<String>,
    /// What the node executes.
    pub kind: NodeKind,
    /// Members of a reference node's block; empty for inline Kernels, whose
    /// wiring sits on their `input` lines and whose values are their
    /// parameter defaults.
    pub members: Vec<NodeMember>,
}

/// How a node gets its Kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// A published Kernel, by id as written.
    Reference(Spanned<String>),
    /// A Kernel written in place. With an id it is also published under
    /// that id; without one it is anonymous and the loader names it after
    /// the document and the node.
    Inline(Box<Kernel>),
    /// Another `.osn` document placed as a node: `node fx: graph "path";`.
    /// Its inputs are wired and its params set like a Kernel's.
    SubGraph(Spanned<String>),
}

/// One line inside a reference node's block.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeMember {
    /// `name = expr;`: a parameter value, or the reserved `domain`.
    Set(Spanned<String>, Expr),
    /// `port <- source;`
    Bind(Spanned<String>, SourceRef),
    /// `port <~ source from initial;`
    Previous {
        /// The port.
        port: Spanned<String>,
        /// The node output read from the last commit.
        source: SourceRef,
        /// What feeds the port before the first commit.
        initial: SourceRef,
    },
}

/// Wiring attached to an inline Kernel's `input` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingDecl {
    /// `<- source`
    Now(SourceRef),
    /// `<~ source from initial`
    Previous {
        /// The node output read from the last commit.
        source: SourceRef,
        /// What feeds the port before the first commit.
        initial: SourceRef,
    },
}

/// `name` (a graph input) or `node.port`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    /// The node, when the reference is `node.port`.
    pub node: Option<String>,
    /// The graph input name or the port.
    pub name: String,
    /// Where it was written.
    pub span: Span,
}

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
    /// Byte range of the whole `kernel ... }` item in its document, so a
    /// Kernel can be quoted on its own.
    pub range: Range<usize>,
    /// The raw id text, e.g. `image.grayscale@1`; `None` for an anonymous
    /// inline Kernel.
    pub id: Option<Spanned<String>>,
    /// `domain = expr;`, allowed only in an inline node Kernel.
    pub domain: Option<Expr>,
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

/// `input name: Type;` or `output name: Type;`, with wiring on inline
/// Kernel inputs: `input name: Type <- source;`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortDecl {
    /// Port name.
    pub name: Spanned<String>,
    /// Type name as written.
    pub ty: Spanned<String>,
    /// Wiring, only on inline Kernel inputs.
    pub binding: Option<BindingDecl>,
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
