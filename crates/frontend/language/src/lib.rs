#![warn(missing_docs)]
//! Orla language frontend (L5).
//!
//! One grammar, two file roles: a `.orla` file holds Kernels; a `.osn`
//! file is a graph that uses libraries, declares inputs and outputs, and
//! places nodes, each referencing a Kernel by id or carrying one inline.
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
//!     requires { field.read; field.write; }
//!     input current: Image <- gray.result;
//!     input previous: Image <~ trail.result from gray.result;
//!     output result: Image;
//!     compute { ... }
//! }
//!
//! output result <- trail.result;
//! ```
//!
//! Wiring sits at the consumer: `port <- source` reads this evaluation's
//! value, `port <~ source from initial` the previous commit's. A source is
//! a graph input or `node.port`.
//!
//! Kernel syntax:
//!
//! ```text
//! kernel image.grayscale@1 {
//!     requires { field.read; field.write; }
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

pub mod doc;
mod lexer;
mod parser;

use std::fmt;
use std::ops::Range;

pub use orla_abi::names::{did_you_mean, edit_distance, nearest, suggestion};
pub use parser::{outline, parse, parse_document, parse_literal};

/// The version of the Kernel language and the document format.
pub const LANGUAGE_VERSION: u32 = 1;

/// `package name@version { language = N; ir = N; }`: a package manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageDecl {
    /// The package's id, `name@version`.
    pub id: Spanned<String>,
    /// The least language version it needs.
    pub language: u32,
    /// The least IR version it needs.
    pub ir: u32,
}

/// A whole file. For a `.osn` the file is the graph; for a `.orla` only
/// `kernels` is populated; a package's `package.osn` has `package`.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    /// The package this document is the manifest of, if it is one.
    pub package: Option<PackageDecl>,
    /// Files or directories whose Kernels this document uses.
    pub uses: Vec<Use>,
    /// Top-level struct types, seen by every Kernel before its own.
    pub structs: Vec<StructDecl>,
    /// Top-level port type aliases, shared the same way.
    pub types: Vec<TypeAlias>,
    /// Top-level functions, shared the same way; one with a Field
    /// parameter is instantiated per port it is called with.
    pub functions: Vec<FnDecl>,
    /// Kernels defined at the top level, document structs already prepended.
    pub kernels: Vec<Kernel>,
    /// External images the caller supplies: `input name: Image;`.
    pub inputs: Vec<PortDecl>,
    /// Graph parameters: `param name: Type = default;`, overridable by a
    /// parent graph or the caller.
    pub params: Vec<ParamDecl>,
    /// Named results: `output name <- source;`.
    pub outputs: Vec<(Spanned<String>, SourceRef)>,
    /// Outputs also written to a file: `output name <- source to "file";`.
    pub writes: Vec<WriteDecl>,
    /// Nodes in document order.
    pub nodes: Vec<NodeDecl>,
}

/// Where each statement of a document sits in its text: what an edit
/// names and the bytes it replaces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outline {
    /// Top-level statements in document order.
    pub statements: Vec<Statement>,
}

/// One top-level statement: its kind, the name it declares, and its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    /// What it declares.
    pub kind: StatementKind,
    /// The name it declares as written (a `use`'s path, a Kernel's id);
    /// empty for `package`.
    pub name: String,
    /// From its first token to its last, the `;` or `}` included.
    pub range: Range<usize>,
    /// A reference node's block of members.
    pub block: Option<NodeBlock>,
}

/// The kinds of top-level statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementKind {
    /// `use "path";`
    Use,
    /// `package name@version { ... }`
    Package,
    /// `struct Name { ... }`
    Struct,
    /// `type Name = ...;`
    Type,
    /// `fn name(...) { ... }`
    Fn,
    /// `kernel id@version { ... }`
    Kernel,
    /// `input name: Type ...;`
    Input,
    /// `param name: Type = value;`
    Param,
    /// `output name <- source ...;`
    Output,
    /// `node name: ...`
    Node,
}

impl StatementKind {
    /// The keyword that begins it.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Use => "use",
            Self::Package => "package",
            Self::Struct => "struct",
            Self::Type => "type",
            Self::Fn => "fn",
            Self::Kernel => "kernel",
            Self::Input => "input",
            Self::Param => "param",
            Self::Output => "output",
            Self::Node => "node",
        }
    }
}

/// Where a node's `{ ... }` and the members in it were written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeBlock {
    /// The byte of its `{`.
    pub open: usize,
    /// The byte of its `}`.
    pub close: usize,
    /// Its members in order.
    pub members: Vec<NodeMemberAt>,
}

/// Where one member of a node's block was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeMemberAt {
    /// The parameter or port it sets or wires.
    pub name: String,
    /// From its first token to its `;`.
    pub range: Range<usize>,
}

/// `to "file"` or `to bytes "file"` after an output: the file, relative
/// to the document, the output's Field is written to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteDecl {
    /// The output written.
    pub output: String,
    /// The file.
    pub path: Spanned<String>,
    /// Written as bytes, one per `Rows<Int>` element; otherwise in the
    /// format the file's extension names.
    pub bytes: bool,
}

impl Document {
    /// Puts declarations from used documents in front of every Kernel's
    /// own, so a shared struct or type has one layout everywhere.
    pub fn inherit(&mut self, structs: &[StructDecl], types: &[TypeAlias], functions: &[FnDecl]) {
        if structs.is_empty() && types.is_empty() && functions.is_empty() {
            return;
        }
        let prepend = |kernel: &mut Kernel| {
            let mut all = structs.to_vec();
            all.append(&mut kernel.structs);
            kernel.structs = all;
            let mut aliases = types.to_vec();
            aliases.append(&mut kernel.types);
            kernel.types = aliases;
            let mut shared = functions.to_vec();
            shared.append(&mut kernel.functions);
            kernel.functions = shared;
        };
        for kernel in &mut self.kernels {
            prepend(kernel);
        }
        for node in &mut self.nodes {
            if let NodeKind::Inline(kernel) = &mut node.kind {
                prepend(kernel);
            }
        }
    }
}

/// `use "path";`: a file of Kernels, or a directory when it ends with `/`.
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
    /// Members of a reference node's block; empty for an inline Kernel,
    /// which wires on its `input` lines.
    pub members: Vec<NodeMember>,
}

/// How a node gets its Kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// A published Kernel, by id as written.
    Reference(Spanned<String>),
    /// A Kernel written in place, published under its id if it has one;
    /// the loader names an anonymous one after the document and node.
    Inline(Box<Kernel>),
    /// Another `.osn` document as a node: `node fx: graph "path";`, wired
    /// and set like a Kernel.
    SubGraph(Spanned<String>),
    /// A filled Field of a type: zeros, or `value = 0.5;`. An alias or
    /// struct element type needs the `value`.
    Fill(Spanned<String>),
}

/// One line inside a reference node's block.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeMember {
    /// `name = expr;`: a parameter value, or the reserved `domain`.
    Set(Spanned<String>, Expr),
    /// `port <- source;`
    Bind(Spanned<String>, SourceRef),
    /// `carry port <- output;`: with `times = N;`, each time after the
    /// first reads the previous time's output.
    Carry(Spanned<String>, Spanned<String>),
    /// `port <~ source from initial [migrate kernel.id@v];`
    Previous {
        /// The port.
        port: Spanned<String>,
        /// The node output read from the last commit.
        source: SourceRef,
        /// What feeds the port before the first commit.
        initial: SourceRef,
        /// The Kernel (`id@version`) that migrates committed state into
        /// this port's layout when the layouts differ.
        migration: Option<Spanned<String>>,
    },
}

/// Wiring attached to an inline Kernel's `input` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingDecl {
    /// `<- source`
    Now(SourceRef),
    /// `<~ source from initial [migrate kernel.id@v]`
    Previous {
        /// The node output read from the last commit.
        source: SourceRef,
        /// What feeds the port before the first commit.
        initial: SourceRef,
        /// The Kernel (`id@version`) that migrates committed state into
        /// this port's layout when the layouts differ.
        migration: Option<Spanned<String>>,
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
    /// Byte range of the whole `kernel ... }` item in its document.
    pub range: Range<usize>,
    /// The raw id text, e.g. `image.grayscale@1`; `None` for an anonymous
    /// inline Kernel.
    pub id: Option<Spanned<String>>,
    /// `domain = expr;`, allowed only in an inline node Kernel.
    pub domain: Option<Expr>,
    /// Byte range of the `domain = expr;` statement in its document.
    pub domain_range: Option<Range<usize>>,
    /// Declared capability names.
    pub requires: Vec<Spanned<String>>,
    /// `budget N;`: the loop iterations an element may run.
    pub budget: Option<Spanned<u32>>,
    /// Inputs in declaration order.
    pub inputs: Vec<PortDecl>,
    /// Parameters in declaration order.
    pub params: Vec<ParamDecl>,
    /// Outputs in declaration order.
    pub outputs: Vec<PortDecl>,
    /// Memories in declaration order: `memory ram: Rows<Int>;`, a port an
    /// ordered Kernel reads and writes step by step.
    pub memories: Vec<PortDecl>,
    /// Struct types in declaration order.
    pub structs: Vec<StructDecl>,
    /// Port type aliases in declaration order.
    pub types: Vec<TypeAlias>,
    /// Helper functions in declaration order.
    pub functions: Vec<FnDecl>,
    /// The per-element body.
    pub compute: Block,
    /// `merge(a, b) { ... }`: a reduction's fold of two results.
    pub merge: Option<MergeDecl>,
    /// `scan with name = init { ... }`: a scan's value carried through
    /// the elements in order.
    pub carry: Option<CarryDecl>,
    /// `scatter into base with combine { ... }`: a scatter, whose
    /// `compute` yields an address and a value per first-input element.
    pub scatter: Option<ScatterDecl>,
    /// `prefix with combine { ... }`: a prefix fold.
    pub prefix: Option<PrefixDecl>,
}

/// `prefix with combine { ... }`: the rule an element's value meets
/// everything before it with.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixDecl {
    /// `sum`, `min`, or `max`.
    pub combine: Spanned<String>,
    /// Where `prefix` was written.
    pub span: Span,
}

/// `scatter into base with combine { ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterDecl {
    /// The input port the output starts from.
    pub into: Spanned<String>,
    /// `sum`, `min`, `max`, or `last`.
    pub combine: Spanned<String>,
    /// Where `scatter` was written.
    pub span: Span,
}

/// The carried value of a scan.
#[derive(Clone, Debug, PartialEq)]
pub struct CarryDecl {
    /// Name of the carried value inside the body.
    pub name: Spanned<String>,
    /// Its value before the first element; evaluated once.
    pub init: Expr,
    /// Where `scan` was written.
    pub span: Span,
}

/// `merge(a, b) { ... }`: the fold of a reduce Kernel. Both arguments and
/// the result have the output's element type.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeDecl {
    /// Name of the first value.
    pub a: Spanned<String>,
    /// Name of the second value.
    pub b: Spanned<String>,
    /// Body yielding the merged value.
    pub body: Block,
    /// Where `merge` was written.
    pub span: Span,
}

/// `type Name = Shape<Element>;`: a transparent alias of a port type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeAlias {
    /// Alias name; must start with an uppercase letter.
    pub name: Spanned<String>,
    /// The port type it stands for, as written, e.g. `Grid<Tile>`.
    pub target: Spanned<String>,
}

/// `struct Name { field: Type; ... }`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    /// Struct name; must start with an uppercase letter.
    pub name: Spanned<String>,
    /// Fields as `(name, type)`; an array type is written `[Type; N]`.
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
    /// Document inputs only: a file read, `= "atlas.ppm"`.
    pub file: Option<Spanned<String>>,
    /// Document inputs only: text held as character codes, `= text "ORLA"`.
    pub text: Option<Spanned<String>>,
    /// Document inputs only: a directory read frame by frame,
    /// `= frames "clip/" at 30`.
    pub frames: Option<FramesDecl>,
    /// Document inputs only: a file read as bytes, one per element,
    /// `= bytes "level.bin"`.
    pub bytes: Option<Spanned<String>>,
    /// What a file input reads while its file is unreadable:
    /// `... else fresh.result`.
    pub otherwise: Option<SourceRef>,
    /// Byte range of the wiring (`<- source`, `<~ source from initial`)
    /// in the document.
    pub binding_range: Option<Range<usize>>,
}

/// `frames "dir/" at N`: a directory's files in name order, one per frame
/// at `rate` frames per second of session time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramesDecl {
    /// The directory, as written.
    pub dir: Spanned<String>,
    /// Frames a second.
    pub rate: u32,
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
    /// Byte range of the default expression in the document.
    pub default_range: Range<usize>,
}

/// `fn name(arg: Type, ...) -> Type { ... }`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    /// Function name.
    pub name: Spanned<String>,
    /// Type parameters as written, `fn twice<T>(v: T) -> T`.
    pub generics: Vec<Spanned<String>>,
    /// Arguments as `(name, type)`.
    pub args: Vec<(Spanned<String>, Spanned<String>)>,
    /// Return type name as written.
    pub ret: Spanned<String>,
    /// Body.
    pub body: Block,
}

/// How deep an expression or block may nest, counting enclosing nodes and
/// chained operators (`a + b + c` is three deep at `c`). A deeper document
/// is refused with an error, so no Kernel can exhaust the compiling
/// thread's stack.
pub const MAX_NESTING: usize = 256;

/// The stack of a thread that reads or compiles a document: room for a
/// tree [`MAX_NESTING`] deep through every recursive pass, in a debug
/// build too. Reserved, committed only as touched.
pub const DEEP_STACK: usize = 256 << 20;

thread_local! {
    /// Whether this thread is one [`on_a_deep_stack`] made.
    static DEEP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `work` on a thread with [`DEEP_STACK`] of stack and waits for it;
/// runs it here when this thread is one already or none can be spawned.
/// Reading and compiling go through here, so the nesting limit is the
/// same whichever thread asks.
pub fn on_a_deep_stack<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    if DEEP.with(std::cell::Cell::get) {
        return work();
    }
    let mut work = Some(work);
    let spawned = std::thread::scope(|scope| {
        let work = &mut work;
        std::thread::Builder::new()
            .name("orla-compile".into())
            .stack_size(DEEP_STACK)
            .spawn_scoped(scope, move || {
                DEEP.with(|deep| deep.set(true));
                (work.take().expect("run once"))()
            })
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .ok()
    });
    match spawned {
        Some(done) => done,
        None => (work.take().expect("not run"))(),
    }
}

/// `let` bindings ending in a result, written `return expr;` or as a tail
/// expression, in any block.
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
    /// `%`: the remainder, truncated like `/`
    Rem,
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
    /// `&` on integers
    BitAnd,
    /// `|` on integers
    BitOr,
    /// `^` on integers
    BitXor,
    /// `<<` on integers
    Shl,
    /// `>>` on integers, arithmetic
    Shr,
    /// `>>>` on integers, logical
    Ushr,
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
    /// A float literal such as `1.0`, exact as written: a `Float`, or a
    /// `Double` where one is wanted.
    Float(f64),
    /// An integer literal such as `8`: an `Int`, or a `Long` where one
    /// is wanted or the value needs 64 bits.
    Int(i64),
    /// A float literal written with `d`, such as `0.1d`: a `Double`.
    Double(f64),
    /// An integer literal written with `l`, such as `8l`: a `Long`.
    Long(i64),
    /// `true` or `false`.
    Bool(bool),
    /// A reference to an input, parameter, argument, or binding.
    Name(String),
    /// A call to a builtin, intrinsic, or helper function.
    Call(String, Vec<Expr>),
    /// Component access such as `c.r`, or a struct field such as `hit.distance`.
    Member(Box<Expr>, String),
    /// `[a, b, c]`: an array of the listed elements, all of one type.
    Array(Vec<Expr>),
    /// `[value; n]`: an array of `n` copies of a value.
    Repeat(Box<Expr>, Spanned<u32>),
    /// `[each n { body }]`: an array built element by element, the body
    /// run once per element with `index()` its position.
    Each(Spanned<u32>, Box<Block>),
    /// `array[index]`: the element at an `Int` index, clamped into the
    /// array; a literal index is checked when compiled.
    Subscript(Box<Expr>, Box<Expr>),
    /// `Name { field: expr, ..base }`.
    Struct {
        /// The struct name.
        name: Spanned<String>,
        /// The value supplying unlisted fields.
        base: Option<Box<Expr>>,
        /// Listed fields.
        fields: Vec<(Spanned<String>, Expr)>,
    },
    /// `{ name: expr, ... }`: each output once by name, the result of a
    /// multi-output Kernel's `compute`.
    Record(Vec<(Spanned<String>, Expr)>),
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
    /// `match value { 1 => a, 2 | 3 => { ... }, _ => b }` on an `Int`.
    Match {
        /// The `Int` matched.
        scrutinee: Box<Expr>,
        /// The arms in order, each with the values it takes.
        arms: Vec<MatchArm>,
        /// The `_` arm.
        otherwise: Box<Block>,
    },
    /// `break expr;` in a loop body: the loop ends after this iteration
    /// with that value.
    Break(Box<Expr>),
    /// `loop count with name = init until done { body }`.
    Loop {
        /// The iteration count, an `Int` read once before the loop.
        count: Box<Expr>,
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

/// One arm of a `match`: the values it takes and what it yields.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchArm {
    /// The `Int` values, as written, `1 | 2 | 3`.
    pub values: Vec<Spanned<i32>>,
    /// The arm's block: `{ lets; value }`, or a bare expression.
    pub body: Block,
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
