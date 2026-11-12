#![warn(missing_docs)]
//! Scene documents: a parsed `.esn` lowered into the Graph model, with
//! sub-graphs flattened in place.
//!
//! The frontend half of composition: this crate reads and checks what
//! was written (nodes, wiring, domains, `times`, placed sub-graphs,
//! inline Kernels) and hands the loader the inline Kernels to publish.
//! The caller supplies files through a [`Loader`].

pub mod domain;
pub mod edit;

pub use domain::domain_expr;

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use entelechy_abi::Value;
use entelechy_ir::KernelId;
use entelechy_language::NodeDecl;
use entelechy_language::{
    BindingDecl, Document, Expr, ExprKind, Kernel, NodeKind, NodeMember, ParseError, SourceRef,
    Span, Spanned,
};

use entelechy_graph::{
    Awake, Binding, DomainExpr, DomainSpec, Graph, GraphParam, Node, NodeId, ParamValue,
    Requirement, Source,
};

/// The port type an alias stands for, following aliases to the end.
fn resolved(ty: &str, aliases: &[entelechy_language::TypeAlias]) -> String {
    let mut ty = ty.to_owned();
    for _ in 0..8 {
        match aliases.iter().find(|alias| alias.name.value == ty) {
            // An enum is an `Int`.
            Some(alias) if alias.variants().is_some() => return "Int".to_owned(),
            Some(alias) => ty = alias.target.value.clone(),
            None => break,
        }
    }
    ty
}

/// The default fill of a Field of this port type: the zero of its
/// element, for element types a parameter can hold.
fn zero_of(ty: &str) -> Option<Value> {
    if ty == "Image" {
        return Some(Value::Vec4([0.0; 4]));
    }
    let element = ty
        .split_once('<')
        .and_then(|(_, rest)| rest.strip_suffix('>'))?;
    let element = element.split(" as ").next().unwrap_or(element);
    match element {
        "Float" => Some(Value::Float(0.0)),
        "Int" => Some(Value::Int(0)),
        "Bool" => Some(Value::Bool(false)),
        "Vec2" => Some(Value::Vec2([0.0; 2])),
        "Vec3" => Some(Value::Vec3([0.0; 3])),
        "Vec4" => Some(Value::Vec4([0.0; 4])),
        "Double" => Some(Value::Double(0.0)),
        "Long" => Some(Value::Long(0)),
        "DVec2" => Some(Value::DVec2([0.0; 2])),
        "DVec3" => Some(Value::DVec3(Box::new([0.0; 3]))),
        "DVec4" => Some(Value::DVec4(Box::new([0.0; 4]))),
        _ => None,
    }
}

/// A document does not parse or does not describe a valid graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentError {
    /// Where the problem is.
    pub span: Span,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.span.is_placed() {
            write!(f, "{}: {}", self.span, self.message)
        } else {
            f.write_str(&self.message)
        }
    }
}

impl std::error::Error for DocumentError {}

impl From<ParseError> for DocumentError {
    fn from(error: ParseError) -> Self {
        Self {
            span: error.span,
            message: error.message,
        }
    }
}

/// Reads a document at a path a placement writes: where the file is
/// (one path for one file, however it is written or reached) and its
/// text.
pub type Loader<'a> = dyn FnMut(&Path) -> Result<(PathBuf, String), String> + 'a;

/// Resolves a document's `use` items, given its directory and what it
/// declares itself, to the structs, type aliases and functions every
/// Kernel of the document sees, and the names it writes qualified by a
/// package. The caller reads the files.
pub type Declare<'a> = dyn FnMut(
        &Path,
        &[entelechy_language::Use],
        entelechy_language::Own<'_>,
    ) -> Result<entelechy_language::Declarations, String>
    + 'a;

/// A Kernel written inside a node, for the loader to publish.
#[derive(Clone, Debug, PartialEq)]
pub struct InlineKernel {
    /// The flattened node that carries it.
    pub node: NodeId,
    /// The id it publishes as: its own, or an anonymous one.
    pub id: KernelId,
    /// The parsed Kernel, with graph-param defaults resolved to literals.
    pub kernel: Kernel,
    /// Its source text.
    pub text: String,
    /// The document it was written in.
    pub document: PathBuf,
}

/// An external a document reads from a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asset {
    /// The input's name.
    pub name: String,
    /// The file, resolved against the document's directory.
    pub path: PathBuf,
    /// The type, aliases resolved: it decides how a picture file comes in
    /// and how the Field keeps it (`Grid<Vec4 as unorm8>`).
    pub ty: String,
    /// Whether the file comes in as its bytes, one per element of a
    /// `Rows<Int>`.
    pub bytes: bool,
}

/// A file output: `output name <- source to "file";`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Write {
    /// The output's name.
    pub output: String,
    /// The file, resolved against the document's directory.
    pub path: PathBuf,
    /// Whether the Field goes out as its bytes, one per element of a
    /// `Rows<Int>`, instead of in the extension's format.
    pub bytes: bool,
    /// Whether the Field is a list of operations on the files under the
    /// directory `path` (`to files "dir/"`).
    pub files: bool,
}

/// An input a Host device feeds: `input name: Type = device;`. The device
/// is a name the Host's table knows (`entelechy_documents::devices`); the
/// scene knows none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Feed {
    /// The input's name.
    pub input: String,
    /// The device, as written.
    pub device: String,
    /// The input's type, aliases resolved: the device gives one structure.
    pub ty: String,
    /// Where the device's name is written: a fault in it is said there.
    pub at: Span,
}

/// An output a Host device takes: `output name <- source to device;`. The
/// device is a name the Host's table knows (`entelechy_documents::devices`);
/// the scene knows none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Present {
    /// The output's name.
    pub output: String,
    /// The device, as written.
    pub device: String,
    /// Where the device's name is written: a fault in it is said there.
    pub at: Span,
}

/// A scene's file contract: inputs read from files, directories read as
/// frames, and outputs written. The session carries them out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Files {
    /// Externals read from files, loaded and watched by the session.
    pub assets: Vec<Asset>,
    /// Externals read frame by frame from a directory: name, resolved
    /// directory, type as written, and frames per second of session time.
    pub sequences: Vec<(String, PathBuf, String, u32)>,
    /// Externals holding what the language finds wrong in the documents
    /// of a file or a directory, as text: name, resolved path. The session
    /// checks them again when one changes.
    pub checks: Vec<(String, PathBuf)>,
    /// Externals holding every file under a directory, its path and its
    /// bytes: name, resolved directory. The session reads them again when
    /// one changes, or which there are.
    pub files: Vec<(String, PathBuf)>,
    /// Externals holding what the documents of a file or a directory
    /// define, as text: name, resolved path. The session reads them again
    /// when one changes.
    pub outlines: Vec<(String, PathBuf)>,
    /// Externals a Host device feeds, the root's only: `= clipboard`,
    /// `= progress`. The session reads each again when its device changes.
    pub feeds: Vec<Feed>,
    /// The root document's file outputs, written after a committed
    /// evaluation that changed them.
    pub writes: Vec<Write>,
    /// The root document's outputs a Host device takes, one an output.
    pub presented: Vec<Present>,
    /// Font nodes, at every level: the session reads their fonts and
    /// gives each the glyphs of the characters its `codes` held when last
    /// evaluated.
    pub fonts: Vec<FontNode>,
    /// Files nodes, at every level: the session lists their directories
    /// and gives each the files its `paths` named when last evaluated.
    pub file_nodes: Vec<FilesNode>,
    /// Fields nodes, at every level: the session reads, off the frames,
    /// the Field files each node's `paths` named when last evaluated.
    pub field_nodes: Vec<FieldsNode>,
}

impl Files {
    /// The inputs device `device` feeds, by name.
    pub fn fed_by(&self, device: &str) -> Vec<String> {
        self.feeds
            .iter()
            .filter(|feed| feed.device == device)
            .map(|feed| feed.input.clone())
            .collect()
    }
}

/// A `font` node: `node g: font "x.ttf" { codes <- c.result; }`. It is
/// three externals the session fills, `glyphs`, `outlines` and `metrics`
/// in the layouts `entelechy import` writes a font's Fields in, read as
/// `g.glyphs` and the others.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FontNode {
    /// The node's id, placed.
    pub node: String,
    /// The font file, resolved against the document's directory.
    pub path: PathBuf,
    /// Where the characters wanted come from, read as last evaluated.
    pub codes: Source,
    /// Each output's port and the external that holds it.
    pub outputs: Vec<(String, String)>,
}

/// A font node's outputs, in order.
pub const FONT_OUTPUTS: [&str; 3] = ["glyphs", "outlines", "metrics"];

/// A `files` node: `node d: files "dir/" { paths <- p.result; }`. It is two
/// externals the session fills: `listing`, every file under the directory
/// that is not ignored, its path and size, and `opened`, the files `paths`
/// named when last evaluated, their bytes, in the layouts of
/// `entelechy_formats::directory`; read as `d.listing` and `d.opened`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilesNode {
    /// The node's id, placed.
    pub node: String,
    /// The directory, resolved against the document's directory.
    pub dir: PathBuf,
    /// Where the paths wanted come from, read as last evaluated.
    pub paths: Source,
    /// Each output's port and the external that holds it.
    pub outputs: Vec<(String, String)>,
}

/// A files node's outputs, in order.
pub const FILES_OUTPUTS: [&str; 2] = ["listing", "opened"];

/// A `fields` node: `node near: Rows<Vertex> fields "chunks/" { paths <-
/// p.result; }`. It is two externals the session fills: `result`, the
/// elements of the Field files `paths` named when last evaluated that are
/// in, in the order named, and `ranges`, a row a path, where its elements
/// start in `result` and how many there are (-1 and 0 for one not in);
/// read as `near.result` and `near.ranges`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldsNode {
    /// The node's id, placed.
    pub node: String,
    /// The directory, resolved against the document's directory.
    pub dir: PathBuf,
    /// The type of `result`, aliases resolved: `Rows` of the element.
    pub ty: String,
    /// Where the paths wanted come from, read as last evaluated.
    pub paths: Source,
    /// Each output's port and the external that holds it.
    pub outputs: Vec<(String, String)>,
}

/// A fields node's outputs, in order.
pub const FIELDS_OUTPUTS: [&str; 2] = ["result", "ranges"];

/// A lowered document: the graph plus everything the loader must publish
/// and watch.
#[derive(Clone, Debug, PartialEq)]
pub struct Lowered {
    /// The graph.
    pub graph: Graph,
    /// The document's file contract, carried out by the session.
    pub files: Files,
    /// Inline Kernels from every level, in document order.
    pub inline: Vec<InlineKernel>,
    /// `use` targets from every level, resolved, with whether the path
    /// named a directory.
    pub uses: Vec<(PathBuf, bool)>,
    /// Sub-graph documents that were loaded, resolved.
    pub documents: Vec<PathBuf>,
}

/// An inline Kernel's text as `publish` takes it: the node's Kernel with
/// its `domain` statement and input wiring removed, graph-param defaults
/// replaced by `edits` (document byte ranges and their text), and the
/// node's id written in when it has none.
pub fn publishable_text(
    document: &str,
    kernel: &Kernel,
    id: &KernelId,
    mut edits: Vec<(Range<usize>, String)>,
) -> String {
    let Some(mut text) = document.get(kernel.range.clone()).map(str::to_owned) else {
        return String::new();
    };
    let base = kernel.range.start;
    if let Some(range) = &kernel.domain_range {
        let (a, b) = (range.start - base, range.end - base);
        // A statement alone on its line takes the line.
        let line_start = text[..a].rfind('\n').map_or(0, |i| i + 1);
        let line_end = text[b..].find('\n').map_or(text.len(), |i| b + i + 1);
        let alone = text[line_start..a].trim().is_empty() && text[b..line_end].trim().is_empty();
        let cut = if alone { line_start..line_end } else { a..b };
        edits.push((cut.start + base..cut.end + base, String::new()));
    }
    for port in &kernel.inputs {
        if let Some(range) = &port.binding_range {
            // The space before the arrow goes with it: `input x: Image;`.
            let mut start = range.start;
            while start > base && document.as_bytes()[start - 1] == b' ' {
                start -= 1;
            }
            edits.push((start..range.end, String::new()));
        }
    }
    // Later ranges first, so earlier offsets stay valid.
    edits.sort_by(|a, b| b.0.start.cmp(&a.0.start));
    for (range, replacement) in edits {
        text.replace_range(range.start - base..range.end - base, &replacement);
    }
    if kernel.id.is_none() {
        text = format!("kernel {id}{}", &text["kernel".len()..]);
    }
    text
}

/// An anonymous inline Kernel's id: document and node names, lowercased
/// with unsupported characters replaced, at version 1.
pub fn anonymous_id(document: &str, node: &str) -> KernelId {
    let segment = |text: &str| {
        let mut out: String = text
            .chars()
            .map(|c| match c.to_ascii_lowercase() {
                c @ ('a'..='z' | '0'..='9' | '_') => c,
                _ => '_',
            })
            .collect();
        if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
            out.insert(0, 'n');
        }
        out
    };
    KernelId::new(&format!("{}.{}", segment(document), segment(node)), 1)
        .expect("sanitized segments form a valid id")
}

fn error(span: Span, message: impl Into<String>) -> DocumentError {
    DocumentError {
        span,
        message: message.into(),
    }
}

/// A compiler's error as the document's: where the compiler placed it, else
/// at `span`, said once.
fn compiled(span: Span, e: entelechy_compiler::CompileError) -> DocumentError {
    match e {
        entelechy_compiler::CompileError::Semantic {
            span: placed,
            message,
            ..
        } if placed.is_placed() => error(placed, message),
        other => error(span, other.to_string()),
    }
}

/// A compiler's error in a struct another document declares, said with
/// that document and the place in it; the caller places it in the scene.
fn declared_elsewhere(
    structs: &[entelechy_language::StructDecl],
    e: entelechy_compiler::CompileError,
) -> Result<DocumentError, entelechy_compiler::CompileError> {
    let Some(origin) = e
        .declaration()
        .and_then(|name| structs.iter().find(|d| d.name.value == name))
        .and_then(|d| d.origin.as_ref())
    else {
        return Err(e);
    };
    let (place, message) = match &e {
        entelechy_compiler::CompileError::Semantic { span, message, .. } => {
            (*span, message.clone())
        }
        other => (Span::default(), other.to_string()),
    };
    let at = if place.is_placed() {
        format!("{}:{place}", origin.document)
    } else {
        origin.document.to_string()
    };
    Ok(error(
        Span::default(),
        format!("struct `{}` ({at}): {message}", origin.name),
    ))
}

/// A node's `domain`: a literal resolves now, an expression when the
/// node binds.
fn domain_spec(expr: &entelechy_language::Expr) -> Result<DomainSpec, DocumentError> {
    spec_of(domain_expr(expr)?, expr.span)
}

/// A parameter's value as a domain term, when it is a number.
fn number(value: &Value) -> Option<DomainExpr> {
    match value {
        Value::Int(v) => Some(DomainExpr::Int(i64::from(*v))),
        Value::Long(v) => Some(DomainExpr::Int(*v)),
        Value::Float(v) => Some(DomainExpr::Float(f64::from(*v))),
        Value::Double(v) => Some(DomainExpr::Float(*v)),
        _ => None,
    }
}

/// A domain expression as a node keeps it: an extent when constant, the
/// expression otherwise.
fn spec_of(domain: DomainExpr, span: Span) -> Result<DomainSpec, DocumentError> {
    if domain.is_constant() {
        let literal = domain
            .resolve(&|_| None, &|_| None, &|_| None, &|_, _| None)
            .map_err(|e| error(span, e.message))?;
        Ok(DomainSpec::Literal(literal))
    } else {
        Ok(DomainSpec::Expr(domain))
    }
}

/// A document's nodes and outputs after `times` expansion.
type Expanded = (
    Vec<NodeDecl>,
    Vec<(entelechy_language::Spanned<String>, SourceRef)>,
    BTreeMap<String, String>,
);

/// What a placed sub-graph receives from its parent.
struct Placement {
    /// Node-name prefix, e.g. `fx.`.
    prefix: String,
    /// Parent bindings for the sub-graph's inputs, in the parent's
    /// flattened names.
    inputs: BTreeMap<String, Binding>,
    /// Parent values for the sub-graph's params, already resolved.
    params: BTreeMap<String, ParamValue>,
    /// What every node of the sub-graph is awake by (its placement's and
    /// theirs), in the parent's flattened names.
    awake: Vec<Awake>,
}

/// State shared across one lowering.
struct Lowering<'a, 'b> {
    loader: &'a mut Loader<'b>,
    declare: &'a mut Declare<'b>,
    /// The caller's parameter values, which a root node's `times` reads.
    overrides: &'a BTreeMap<String, Value>,
    out: Lowered,
    /// Documents on the current placement path, to refuse recursion.
    stack: Vec<PathBuf>,
    /// The declarations of the document being lowered, own and
    /// inherited, which fills and synthesized Kernels resolve through.
    aliases: Vec<entelechy_language::TypeAlias>,
    structs: Vec<entelechy_language::StructDecl>,
    functions: Vec<entelechy_language::FnDecl>,
    /// Nodes placed so far, every level together.
    placed: usize,
    /// Each sub-graph document placed so far, read, parsed and declared
    /// once however often it is placed, by where its file is.
    read: BTreeMap<PathBuf, std::rc::Rc<Read>>,
    /// Where the file a written path names is, as the loader said.
    located: BTreeMap<PathBuf, PathBuf>,
}

/// A sub-graph document as a placement uses it: its text, its parse with
/// what it inherits, and the declarations its level sees.
struct Read {
    text: String,
    document: Document,
    aliases: Vec<entelechy_language::TypeAlias>,
    structs: Vec<entelechy_language::StructDecl>,
    functions: Vec<entelechy_language::FnDecl>,
}

/// The most nodes one scene places, every level of `times` and sub-graph
/// placement together, bounding the loader's memory and work.
pub const MAX_PLACED: usize = 1 << 16;

/// A place in a document, as the `file:line:column` of Kernel
/// diagnostics.
fn written_at(path: &Path, span: entelechy_language::Span) -> String {
    format!("{}:{}:{}", path.display(), span.line, span.column)
}

fn lower(
    document: &Document,
    text: &str,
    name: &str,
    base: &Path,
    loader: &mut Loader<'_>,
    declare: &mut Declare<'_>,
    overrides: &BTreeMap<String, Value>,
) -> Result<Lowered, DocumentError> {
    let mut document = document.clone();
    let entelechy_language::Declarations {
        structs,
        types,
        functions,
        qualified,
    } = declare(base, &document.uses, entelechy_language::Own::of(&document))
        .map_err(|message| error(Span::default(), message))?;
    document.unqualify(&qualified);
    document.resolve((&structs, &types, &functions));
    let document = &document;
    let mut lowering = Lowering {
        loader,
        declare,
        overrides,
        out: Lowered {
            graph: Graph {
                name: name.to_owned(),
                externals: Vec::new(),
                texts: Vec::new(),
                params: Vec::new(),
                requirements: Vec::new(),
                times_params: Vec::new(),
                times_rows: Vec::new(),
                nodes: Vec::new(),
                outputs: Vec::new(),
                fallbacks: Vec::new(),
                placed: Vec::new(),
                repeated: Vec::new(),
            },
            files: Files::default(),
            inline: Vec::new(),
            uses: Vec::new(),
            documents: Vec::new(),
        },
        aliases: types.iter().chain(&document.types).cloned().collect(),
        structs: structs.iter().chain(&document.structs).cloned().collect(),
        functions: functions
            .iter()
            .chain(&document.functions)
            .cloned()
            .collect(),
        stack: Vec::new(),
        placed: 0,
        read: BTreeMap::new(),
        located: BTreeMap::new(),
    };
    let level = lowering.level(document, text, name, base, &PathBuf::from(name), None)?;
    // The root's inputs, then the font and files nodes' Fields, every
    // level's.
    let fonts: Vec<String> = std::mem::take(&mut lowering.out.graph.externals);
    lowering.out.graph.externals = level.externals;
    lowering.out.graph.externals.extend(fonts);
    lowering.out.files.assets = level.assets;
    lowering.out.graph.texts = level.texts;
    lowering.out.files.sequences = level.sequences;
    lowering.out.files.checks = level.checks;
    lowering.out.files.files = level.files;
    lowering.out.files.outlines = level.outlines;
    lowering.out.files.feeds = level.feeds;
    lowering.out.graph.params = level.params;
    lowering.out.graph.outputs = level.outputs;
    lowering.out.graph.fallbacks = level.fallbacks;
    lowering.out.files.writes = level.writes;
    lowering.out.files.presented = level.presented;
    // A requirement the parameters as given break is refused as the
    // document loads; one that reads the window waits for an evaluation,
    // which checks every one again.
    let graph = &lowering.out.graph;
    let param = |name: &str| {
        overrides.get(name).cloned().or_else(|| {
            graph
                .params
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.default.clone())
        })
    };
    for requirement in graph.requirements.iter().filter(|r| !r.reads_context()) {
        requirement
            .check(&param, &|_| None)
            .map_err(|e| error(Span::default(), format!("requirement {}", e.message)))?;
    }
    Ok(lowering.out)
}

/// What one document level exposes to its parent.
struct Level {
    externals: Vec<String>,
    assets: Vec<Asset>,
    texts: Vec<(String, String)>,
    sequences: Vec<(String, PathBuf, String, u32)>,
    checks: Vec<(String, PathBuf)>,
    files: Vec<(String, PathBuf)>,
    outlines: Vec<(String, PathBuf)>,
    feeds: Vec<Feed>,
    params: Vec<GraphParam>,
    /// Outputs in declaration order; the first is the primary.
    outputs: Vec<(String, Source)>,
    /// What each input read from a file falls back on.
    fallbacks: Vec<(String, Source)>,
    /// The outputs written to files (the root's only).
    writes: Vec<Write>,
    /// The outputs Host devices take (the root's only).
    presented: Vec<Present>,
}

/// Whether a placed graph and its placer agree: every input wired, and
/// every input or param given is declared. A refusal names the node that
/// places the graph, the name nearest one it lacks, and those it has.
fn check_placement(
    placement: &Placement,
    externals: &[String],
    params: &[GraphParam],
) -> Result<(), DocumentError> {
    let node = placement.prefix.trim_end_matches('.');
    let lacks = |what: &str, name: &str, names: Vec<&str>| {
        let listed = if names.is_empty() {
            "none".to_owned()
        } else {
            names
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        error(
            Span::default(),
            format!(
                "sub-graph `{node}` has no {what} `{name}`{}; its {what}s are {listed}",
                entelechy_language::suggestion(entelechy_language::nearest(
                    name,
                    names.iter().copied()
                ))
            ),
        )
    };
    for input in placement.inputs.keys() {
        if !externals.contains(input) {
            return Err(lacks(
                "input",
                input,
                externals.iter().map(String::as_str).collect(),
            ));
        }
    }
    for external in externals {
        if !placement.inputs.contains_key(external) {
            return Err(error(
                Span::default(),
                format!("sub-graph `{node}` input `{external}` must be wired where it is placed"),
            ));
        }
    }
    for param in placement.params.keys() {
        if !params.iter().any(|p| &p.name == param) {
            return Err(lacks(
                "param",
                param,
                params.iter().map(|p| p.name.as_str()).collect(),
            ));
        }
    }
    Ok(())
}

/// What a document's inputs come to: every one's name, and those the
/// caller supplies from a file, as text, or as frames.
struct Inputs {
    externals: Vec<String>,
    assets: Vec<Asset>,
    texts: Vec<(String, String)>,
    sequences: Vec<(String, PathBuf, String, u32)>,
    checks: Vec<(String, PathBuf)>,
    files: Vec<(String, PathBuf)>,
    outlines: Vec<(String, PathBuf)>,
    feeds: Vec<Feed>,
}

/// Where a document is lowered: its node prefix, what its placing graph
/// gives it, and its params. The one place a written name becomes a
/// flattened one, for values, domains and sources.
struct Place<'a> {
    /// The prefix of every node the document names.
    prefix: &'a str,
    /// What the placing graph gives; `None` at the root.
    placement: Option<&'a Placement>,
    /// The params the document declares.
    params: &'a [GraphParam],
    /// The structs a literal may build and the enums it may name.
    known: Known,
    /// The nodes the document declares, by name.
    nodes: Vec<String>,
}

impl Place<'_> {
    /// A value written in the document: a literal, or a graph param,
    /// which in a placed sub-graph is what the parent passed or the
    /// default.
    fn value(&self, expr: &Expr) -> Result<ParamValue, DocumentError> {
        if let ExprKind::Name(param) = &expr.kind {
            let declared = self.params.iter().find(|p| &p.name == param).ok_or_else(|| {
                error(
                    expr.span,
                    format!("`{param}` is not a param of this graph; values are literals or graph params"),
                )
            })?;
            return Ok(match self.placement.and_then(|p| p.params.get(param)) {
                Some(from_parent) => from_parent.clone(),
                None if self.placement.is_some() => ParamValue::Literal(declared.default.clone()),
                None => ParamValue::Graph(param.clone()),
            });
        }
        let value = literal(expr, &self.known)?;
        Ok(ParamValue::Literal(value))
    }

    /// A value before any override.
    /// The default the document declares for its param `name`.
    fn declared(&self, name: &str) -> Option<Value> {
        self.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default.clone())
    }

    fn default_of(&self, value: &ParamValue) -> Value {
        match value {
            ParamValue::Literal(v) => v.clone(),
            ParamValue::Graph(name) => self
                .params
                .iter()
                .find(|p| &p.name == name)
                .map(|p| p.default.clone())
                .unwrap_or(Value::Float(0.0)),
        }
    }

    /// A domain in the flattened graph: each name of the placed graph's
    /// params becomes what the parent passed, as a value does; a name
    /// the node sets itself stays the node's.
    fn domain(
        &self,
        domain: Option<DomainSpec>,
        node_params: &BTreeMap<String, ParamValue>,
    ) -> Result<Option<DomainSpec>, DocumentError> {
        let (Some(placement), Some(DomainSpec::Expr(expr))) = (self.placement, &domain) else {
            return Ok(domain);
        };
        let given = |name: &str| -> Option<DomainExpr> {
            if node_params.contains_key(name) {
                return None;
            }
            self.passed(placement, name)
        };
        spec_of(expr.with_names(&given), Span::default()).map(Some)
    }

    /// What the placing graph passes for the placed graph's param `name`
    /// in a count: its own param, the value it gave, or the default.
    fn passed(&self, placement: &Placement, name: &str) -> Option<DomainExpr> {
        let declared = self.params.iter().find(|p| p.name == name)?;
        match placement.params.get(name) {
            Some(ParamValue::Graph(root)) => Some(DomainExpr::Name(root.clone())),
            Some(ParamValue::Literal(value)) => number(value),
            None => number(&declared.default),
        }
    }

    /// A requirement in the flattened graph: named with the prefix, each
    /// param what the parent passed. Every name it reads is a param of
    /// this graph.
    fn requirement(
        &self,
        decl: &entelechy_language::RequireDecl,
    ) -> Result<Requirement, DocumentError> {
        let requirement = domain::requirement(decl)?;
        let unknown = std::cell::RefCell::new(None);
        let _ = requirement.with_names(&|name| {
            if !self.params.iter().any(|p| p.name == name) {
                unknown.borrow_mut().get_or_insert_with(|| name.to_owned());
            }
            None
        });
        if let Some(name) = unknown.into_inner() {
            return Err(error(
                decl.condition.span,
                format!(
                    "`require {}` names `{name}`, which is not a param of this graph",
                    decl.name.value
                ),
            ));
        }
        let requirement = match self.placement {
            Some(placement) => requirement.with_names(&|name| self.passed(placement, name)),
            None => requirement,
        };
        Ok(Requirement {
            name: format!("{}{}", self.prefix, requirement.name),
            ..requirement
        })
    }

    /// A source in the flattened graph: a node's name prefixed, a
    /// document input replaced by the parent's binding.
    /// `awake = value(source)` or `awake = value(source, row)`: what the
    /// node is awake by, a source as a `<-` names one.
    fn awake(&self, expr: &Expr) -> Result<Awake, DocumentError> {
        let usage = "`awake` is `value(source)` or `value(source, row)`: the node sleeps while that row is 0";
        let ExprKind::Call(name, args) = &expr.kind else {
            return Err(error(expr.span, usage));
        };
        if name != "value" || args.is_empty() || args.len() > 2 {
            return Err(error(expr.span, usage));
        }
        let reference = match &args[0].kind {
            ExprKind::Name(name) => SourceRef {
                node: None,
                name: name.clone(),
                span: args[0].span,
            },
            ExprKind::Member(node, port) => match &node.kind {
                ExprKind::Name(node) => SourceRef {
                    node: Some(node.clone()),
                    name: port.clone(),
                    span: args[0].span,
                },
                _ => return Err(error(args[0].span, usage)),
            },
            _ => return Err(error(args[0].span, usage)),
        };
        let row = match args.get(1).map(|arg| &arg.kind) {
            None => 0,
            Some(ExprKind::Int(row)) if (0..=i64::from(u32::MAX)).contains(row) => *row as u32,
            Some(_) => {
                return Err(error(
                    args[1].span,
                    "`awake`'s row is a whole number, 0 or more",
                ));
            }
        };
        let Binding::Now(source) = self.now(&reference)? else {
            return Err(error(args[0].span, usage));
        };
        Ok(Awake { source, row })
    }

    fn now(&self, reference: &SourceRef) -> Result<Binding, DocumentError> {
        let prefix = self.prefix;
        if reference.node.is_none() && self.nodes.contains(&reference.name) {
            let node = &reference.name;
            return Err(error(
                reference.span,
                format!(
                    "`{node}` is a node; a source names one of its outputs, `{node}.<output>` (`{node}.result` for a Kernel's `result`)"
                ),
            ));
        }
        match &reference.node {
            Some(node) => Ok(Binding::Now(Source::Node {
                node: NodeId(format!("{prefix}{node}").into()),
                port: reference.name.clone(),
            })),
            None => match self.placement {
                Some(placement) => {
                    placement
                        .inputs
                        .get(&reference.name)
                        .cloned()
                        .ok_or_else(|| {
                            error(
                                reference.span,
                                format!("`{}` is not an input of this graph", reference.name),
                            )
                        })
                }
                None => Ok(Binding::Now(Source::External(reference.name.clone()))),
            },
        }
    }

    /// A `<~` binding written in the document, in the flattened graph.
    fn previous(
        &self,
        source: &SourceRef,
        initial: &SourceRef,
        migration: Option<&Spanned<String>>,
    ) -> Result<Binding, DocumentError> {
        let prefix = self.prefix;
        let migration = match migration {
            Some(named) => Some(
                named
                    .value
                    .parse::<KernelId>()
                    .map_err(|e| error(named.span, format!("`migrate` names a kernel id: {e}")))?,
            ),
            None => None,
        };
        let Some(node) = &source.node else {
            return Err(error(
                source.span,
                "`<~` reads a node output; graph inputs carry no state",
            ));
        };
        let initial = match self.now(initial)? {
            Binding::Now(source) => source,
            Binding::Previous { .. } => {
                return Err(error(
                    initial.span,
                    "the `from` source of a `<~` binding must be a current value",
                ));
            }
        };
        Ok(Binding::Previous {
            source: Source::Node {
                node: NodeId(format!("{prefix}{node}").into()),
                port: source.name.clone(),
            },
            initial,
            migration,
        })
    }
}

impl Lowering<'_, '_> {
    /// What a literal of this document may name: the structs, own and
    /// used, and the enums.
    fn known(&self) -> Known {
        Known {
            structs: self.known_structs(),
            enums: self
                .aliases
                .iter()
                .filter_map(|alias| {
                    let variants = alias.variants()?;
                    Some((
                        alias.name.value.clone(),
                        variants
                            .into_iter()
                            .map(|(variant, number)| (variant.to_owned(), number))
                            .collect(),
                    ))
                })
                .collect(),
        }
    }

    /// The structs, own and used, a literal of this document may name,
    /// with their fields in order.
    fn known_structs(&self) -> Vec<(String, Vec<String>)> {
        self.structs
            .iter()
            .map(|decl| {
                let fields = decl
                    .fields
                    .iter()
                    .map(|(field, _)| field.value.clone())
                    .collect();
                (decl.name.value.clone(), fields)
            })
            .collect()
    }

    /// A node with `times = N;` stands for nodes `name/1` to `name/N`,
    /// each wired as written, except that `carry port <- output;` feeds
    /// the port from the previous time's output from the second time on;
    /// `name.port` means the last time's. At `times = 0` there is no node
    /// and a carried output is what its port is first bound to; an output
    /// no carry names has no value. `N` is a literal or a param, resolved
    /// from the placing graph, the caller's overrides, or the default; a
    /// root param that sizes the graph is recorded so changing it lowers
    /// the document again.
    fn expand_times(
        &mut self,
        document: &Document,
        placement: Option<&Placement>,
        params: &[GraphParam],
        fallbacks: &mut [SourceRef],
    ) -> Result<Expanded, DocumentError> {
        let mut nodes: Vec<NodeDecl> = Vec::new();
        let mut renames: BTreeMap<String, String> = BTreeMap::new();
        // Nodes placed no times: each carried output and what it passes
        // through.
        let mut passed: BTreeMap<String, BTreeMap<String, SourceRef>> = BTreeMap::new();
        for decl in &document.nodes {
            let times = decl.members.iter().find_map(|member| match member {
                NodeMember::Set(name, expr) if name.value == "times" => Some((name, expr)),
                _ => None,
            });
            let Some((times_name, expr)) = times else {
                nodes.push(decl.clone());
                continue;
            };
            let value = match &expr.kind {
                ExprKind::Name(param) => {
                    if !params.iter().any(|p| &p.name == param) {
                        return Err(error(
                            expr.span,
                            format!(
                                "`{param}` is not a param of this graph; `times` is a literal or a graph param"
                            ),
                        ));
                    }
                    match placement.and_then(|p| p.params.get(param)) {
                        Some(ParamValue::Literal(v)) => v.clone(),
                        Some(ParamValue::Graph(root)) => {
                            // The root param it is passed from sizes the
                            // graph too, however deep it is placed.
                            if !self.out.graph.times_params.contains(root) {
                                self.out.graph.times_params.push(root.clone());
                            }
                            self.overrides
                                .get(root)
                                .cloned()
                                .or_else(|| {
                                    self.out
                                        .graph
                                        .params
                                        .iter()
                                        .find(|p| &p.name == root)
                                        .map(|p| p.default.clone())
                                })
                                .unwrap_or_else(|| {
                                    params
                                        .iter()
                                        .find(|p| &p.name == param)
                                        .map(|p| p.default.clone())
                                        .unwrap()
                                })
                        }
                        None => {
                            if placement.is_none() && !self.out.graph.times_params.contains(param) {
                                self.out.graph.times_params.push(param.clone());
                            }
                            self.overrides
                                .get(param)
                                .cloned()
                                .filter(|_| placement.is_none())
                                .unwrap_or_else(|| {
                                    params
                                        .iter()
                                        .find(|p| &p.name == param)
                                        .map(|p| p.default.clone())
                                        .unwrap()
                                })
                        }
                    }
                }
                // A row says how many: the count the caller holds for
                // this node, none before a frame has made the row.
                ExprKind::Call(name, args) if name == "value" => {
                    let usage = "`times` is a whole number, a graph param, or `value(node.port)` / `value(node.port, row)`: as many times as that row held when the last frame ended";
                    if placement.is_some() {
                        return Err(error(
                            expr.span,
                            "`times = value(...)` is written in the scene itself; a placed graph takes its count by a param",
                        ));
                    }
                    let source = match args.first().map(|arg| &arg.kind) {
                        Some(ExprKind::Member(node, port)) if args.len() <= 2 => match &node.kind {
                            ExprKind::Name(node) => (node.clone(), port.clone()),
                            _ => return Err(error(args[0].span, usage)),
                        },
                        _ => return Err(error(expr.span, usage)),
                    };
                    if source.0 == decl.name.value
                        || !document
                            .nodes
                            .iter()
                            .any(|other| other.name.value == source.0)
                    {
                        return Err(error(
                            args[0].span,
                            format!(
                                "`{}` is not another node of this scene; `times = value(node.port)` reads a node's output",
                                source.0
                            ),
                        ));
                    }
                    let row = match args.get(1).map(|arg| &arg.kind) {
                        None => 0,
                        Some(ExprKind::Int(row)) if (0..=i64::from(u32::MAX)).contains(row) => {
                            *row as u32
                        }
                        Some(_) => {
                            return Err(error(
                                args[1].span,
                                "`times`'s row is a whole number, 0 or more",
                            ));
                        }
                    };
                    let held = self
                        .overrides
                        .get(&entelechy_graph::TimesRow::param(&decl.name.value))
                        .cloned()
                        .unwrap_or(Value::Int(0));
                    self.out.graph.times_rows.push(entelechy_graph::TimesRow {
                        node: decl.name.value.clone(),
                        source,
                        row,
                    });
                    held
                }
                _ => literal(expr, &self.known())?,
            };
            let count = match value {
                Value::Int(n) if n >= 0 => n as usize,
                other => {
                    return Err(error(
                        times_name.span,
                        format!("`times` must be a whole number of at least 0, found {other:?}"),
                    ));
                }
            };
            // Refused before its copies are made, with the nodes placed
            // and those expanded so far; each is counted as it is placed.
            if self
                .placed
                .saturating_add(nodes.len())
                .saturating_add(count)
                > MAX_PLACED
            {
                return Err(error(
                    times_name.span,
                    format!(
                        "`times = {count}` places more than {MAX_PLACED} nodes in the scene, every level together"
                    ),
                ));
            }
            let carries: Vec<(String, String)> = decl
                .members
                .iter()
                .filter_map(|member| match member {
                    NodeMember::Carry(port, output) => {
                        Some((port.value.clone(), output.value.clone()))
                    }
                    _ => None,
                })
                .collect();
            for (port, _) in &carries {
                let bound = decl.members.iter().any(|member| match member {
                    NodeMember::Bind(p, _) | NodeMember::Previous { port: p, .. } => {
                        &p.value == port
                    }
                    _ => false,
                });
                if !bound {
                    return Err(error(
                        times_name.span,
                        format!("carried port `{port}` needs a binding for the first time"),
                    ));
                }
            }
            let base_name = decl.name.value.clone();
            if count == 0 {
                let mut through = BTreeMap::new();
                for (port, output) in &carries {
                    let first = decl.members.iter().find_map(|member| match member {
                        NodeMember::Bind(p, source) if &p.value == port => Some(Ok(source)),
                        NodeMember::Previous { port: p, .. } if &p.value == port => Some(Err(p)),
                        _ => None,
                    });
                    match first {
                        Some(Ok(source)) => {
                            through.insert(output.clone(), source.clone());
                        }
                        Some(Err(p)) => {
                            return Err(error(
                                p.span,
                                format!(
                                    "carried port `{port}` is bound with `<~`, which has no value to pass through at `times = 0`"
                                ),
                            ));
                        }
                        None => {}
                    }
                }
                passed.insert(base_name, through);
                continue;
            }
            if count == 1 {
                // One time is the node itself, under its own name.
                let mut copy = decl.clone();
                copy.members.retain(|member| {
                    !matches!(member, NodeMember::Carry(..))
                        && !matches!(member, NodeMember::Set(name, _) if name.value == "times")
                });
                nodes.push(copy);
                continue;
            }
            renames.insert(base_name.clone(), format!("{base_name}/{count}"));
            for k in 1..=count {
                let mut copy = decl.clone();
                copy.name.value = format!("{base_name}/{k}");
                copy.members = decl
                    .members
                    .iter()
                    .filter_map(|member| match member {
                        NodeMember::Set(name, _) if name.value == "times" => None,
                        NodeMember::Carry(..) => None,
                        NodeMember::Bind(port, _) | NodeMember::Previous { port, .. }
                            if k > 1 && carries.iter().any(|(p, _)| p == &port.value) =>
                        {
                            let output = &carries.iter().find(|(p, _)| p == &port.value).unwrap().1;
                            Some(NodeMember::Bind(
                                port.clone(),
                                SourceRef {
                                    node: Some(format!("{base_name}/{}", k - 1)),
                                    name: output.clone(),
                                    span: port.span,
                                },
                            ))
                        }
                        other => Some(other.clone()),
                    })
                    .collect();
                nodes.push(copy);
            }
        }
        // `name.port` means the last time's; a carried output of a node
        // placed no times resolves, through chains, to what it passes.
        let rename = |reference: &mut SourceRef| -> Result<(), DocumentError> {
            for _ in 0..=passed.len() {
                let Some(through) = reference.node.as_ref().and_then(|node| passed.get(node))
                else {
                    break;
                };
                let Some(source) = through.get(&reference.name) else {
                    return Err(error(
                        reference.span,
                        format!(
                            "`{}.{}` has no value at `times = 0`: only a carried output passes through",
                            reference.node.as_deref().unwrap_or(""),
                            reference.name
                        ),
                    ));
                };
                *reference = SourceRef {
                    span: reference.span,
                    ..source.clone()
                };
            }
            if let Some(node) = &reference.node
                && let Some(last) = renames.get(node)
            {
                reference.node = Some(last.clone());
            }
            Ok(())
        };
        if !renames.is_empty() || !passed.is_empty() {
            for node in &mut nodes {
                for member in &mut node.members {
                    match member {
                        NodeMember::Bind(_, reference) => rename(reference)?,
                        NodeMember::Previous {
                            source, initial, ..
                        } => {
                            rename(source)?;
                            rename(initial)?;
                        }
                        NodeMember::Set(..) | NodeMember::Carry(..) => {}
                    }
                }
                if let NodeKind::Inline(kernel) = &mut node.kind {
                    for input in &mut kernel.inputs {
                        match &mut input.binding {
                            Some(BindingDecl::Now(reference)) => rename(reference)?,
                            Some(BindingDecl::Previous {
                                source, initial, ..
                            }) => {
                                rename(source)?;
                                rename(initial)?;
                            }
                            None => {}
                        }
                    }
                }
            }
        }
        let mut outputs = document.outputs.clone();
        for (_, reference) in &mut outputs {
            rename(reference)?;
        }
        for reference in fallbacks {
            rename(reference)?;
        }
        Ok((nodes, outputs, renames))
    }

    /// The inputs the document declares: every name, and those the
    /// caller supplies from a file, as text, or as frames. A placed
    /// graph's inputs are wired by its placer only.
    fn inputs(
        &self,
        document: &Document,
        base: &Path,
        placement: Option<&Placement>,
    ) -> Result<Inputs, DocumentError> {
        let mut externals = Vec::new();
        let mut assets = Vec::new();
        let mut texts = Vec::new();
        let mut sequences = Vec::new();
        let mut checks = Vec::new();
        let mut files = Vec::new();
        let mut outlines = Vec::new();
        let mut feeds = Vec::new();
        for input in &document.inputs {
            // What arrives is checked by shape and layout where it binds.
            if input.binding.is_some() {
                return Err(error(
                    input.name.span,
                    "a graph input is supplied by the caller and cannot be wired",
                ));
            }
            if let Some(file) = &input.file {
                if placement.is_some() {
                    return Err(error(
                        file.span,
                        "a placed graph's input is wired by the placing graph; it cannot name a file",
                    ));
                }
                assets.push(Asset {
                    name: input.name.value.clone(),
                    path: base.join(&file.value),
                    ty: resolved(&input.ty.value, &self.aliases),
                    bytes: false,
                });
            }
            if let Some(bytes) = &input.bytes {
                if placement.is_some() {
                    return Err(error(
                        bytes.span,
                        "a placed graph's input is wired by the placing graph; it cannot name a file",
                    ));
                }
                assets.push(Asset {
                    name: input.name.value.clone(),
                    path: base.join(&bytes.value),
                    ty: resolved(&input.ty.value, &self.aliases),
                    bytes: true,
                });
            }
            if let Some(text) = &input.text {
                if placement.is_some() {
                    return Err(error(
                        text.span,
                        "a placed graph's input is wired by the placing graph; it cannot hold text",
                    ));
                }
                if text.value.is_empty() {
                    return Err(error(
                        text.span,
                        "an input's text needs at least one character; a Field has no empty extent",
                    ));
                }
                texts.push((input.name.value.clone(), text.value.clone()));
            }
            if let Some(frames) = &input.frames {
                if placement.is_some() {
                    return Err(error(
                        frames.dir.span,
                        "a placed graph's input is wired by the placing graph; it cannot read frames",
                    ));
                }
                sequences.push((
                    input.name.value.clone(),
                    base.join(&frames.dir.value),
                    resolved(&input.ty.value, &self.aliases),
                    frames.rate,
                ));
            }
            if let Some(checked) = &input.checks {
                if placement.is_some() {
                    return Err(error(
                        checked.span,
                        "a placed graph's input is wired by the placing graph; it cannot check documents",
                    ));
                }
                let ty = resolved(&input.ty.value, &self.aliases);
                if ty != "Rows<Int>" {
                    return Err(error(
                        input.ty.span,
                        format!(
                            "what `checks` finds is text, a character a row: `Rows<Int>`, not `{ty}`"
                        ),
                    ));
                }
                checks.push((input.name.value.clone(), base.join(&checked.value)));
            }
            if let Some(dir) = &input.files {
                if placement.is_some() {
                    return Err(error(
                        dir.span,
                        "a placed graph's input is wired by the placing graph; it cannot read a directory's files",
                    ));
                }
                let ty = resolved(&input.ty.value, &self.aliases);
                if ty != "Rows<Int>" {
                    return Err(error(
                        input.ty.span,
                        format!(
                            "a directory's files are read as words, their paths and bytes: `Rows<Int>`, not `{ty}`"
                        ),
                    ));
                }
                files.push((input.name.value.clone(), base.join(&dir.value)));
            }
            if let Some(outlined) = &input.outline {
                if placement.is_some() {
                    return Err(error(
                        outlined.span,
                        "a placed graph's input is wired by the placing graph; it cannot read what documents define",
                    ));
                }
                let ty = resolved(&input.ty.value, &self.aliases);
                if ty != "Rows<Int>" {
                    return Err(error(
                        input.ty.span,
                        format!(
                            "what `outline` finds is text, a character a row: `Rows<Int>`, not `{ty}`"
                        ),
                    ));
                }
                outlines.push((input.name.value.clone(), base.join(&outlined.value)));
            }
            // A device fills the scene's own inputs; placed, an input is
            // wired by the placing graph, as a placed graph's outputs are.
            if let Some(device) = input.device.as_ref().filter(|_| placement.is_none()) {
                feeds.push(Feed {
                    input: input.name.value.clone(),
                    device: device.value.clone(),
                    ty: resolved(&input.ty.value, &self.aliases),
                    at: device.span,
                });
            }
            externals.push(input.name.value.clone());
        }
        Ok(Inputs {
            externals,
            assets,
            texts,
            sequences,
            checks,
            files,
            outlines,
            feeds,
        })
    }

    /// The structs in reach, their fields' types read through the
    /// aliases in reach, as the compiler reads a literal of them: an enum a
    /// used document declares is an `Int` there too.
    fn resolved_structs(&self) -> Vec<entelechy_language::StructDecl> {
        let field_type = |ty: &str| match ty
            .strip_prefix('[')
            .and_then(|inner| inner.strip_suffix(']'))
        {
            Some(array) => match array.split_once(';') {
                Some((element, count)) => {
                    format!("[{};{count}]", resolved(element.trim(), &self.aliases))
                }
                None => ty.to_owned(),
            },
            None => resolved(ty, &self.aliases),
        };
        self.structs
            .iter()
            .map(|decl| entelechy_language::StructDecl {
                fields: decl
                    .fields
                    .iter()
                    .map(|(name, ty)| {
                        (
                            name.clone(),
                            entelechy_language::Spanned {
                                value: field_type(&ty.value),
                                span: ty.span,
                            },
                        )
                    })
                    .collect(),
                ..decl.clone()
            })
            .collect()
    }

    /// The params the document declares, each read in its own type.
    fn params(&self, document: &Document) -> Result<Vec<GraphParam>, DocumentError> {
        let mut params = Vec::new();
        for decl in &document.params {
            // Read in the param's declared type, with the document's
            // structs.
            let ty = entelechy_language::Spanned {
                value: resolved(&decl.ty.value, &self.aliases),
                span: decl.ty.span,
            };
            let default = self.known().with_variants(&decl.default)?;
            let structs = self.resolved_structs();
            let default =
                entelechy_compiler::literal_of_type(&default, &ty, &structs).map_err(|e| {
                    match declared_elsewhere(&structs, e) {
                        Ok(said) => DocumentError {
                            span: decl.default.span,
                            ..said
                        },
                        Err(e) => compiled(decl.default.span, e),
                    }
                })?;
            if params
                .iter()
                .any(|p: &GraphParam| p.name == decl.name.value)
            {
                return Err(error(
                    decl.name.span,
                    format!("param `{}` is declared twice", decl.name.value),
                ));
            }
            params.push(GraphParam {
                name: decl.name.value.clone(),
                ty: resolved(&decl.ty.value, &self.aliases),
                default,
            });
        }
        Ok(params)
    }

    /// Lowers the document a `graph` node `id` places, with the inputs
    /// and params the node gives, and returns its declared outputs. The
    /// placed document's declarations are in scope while it is lowered.
    fn place(
        &mut self,
        sub_path: &Spanned<String>,
        base: &Path,
        id: &NodeId,
        inputs: BTreeMap<String, Binding>,
        node_params: BTreeMap<String, ParamValue>,
        awake: Vec<Awake>,
    ) -> Result<Vec<(String, Source)>, DocumentError> {
        // A document is the file it is in: placed from two levels, or
        // written two ways, it is one document, its Kernels one set.
        let written = base.join(&sub_path.value);
        let (resolved, text) = match self.located.get(&written) {
            Some(at) => (at.clone(), None),
            None => {
                let (at, text) = (self.loader)(&written).map_err(|message| {
                    error(
                        sub_path.span,
                        format!("cannot load `{}`: {message}", sub_path.value),
                    )
                })?;
                self.located.insert(written, at.clone());
                (at, Some(text))
            }
        };
        if self.stack.contains(&resolved) {
            return Err(error(
                sub_path.span,
                format!("sub-graph `{}` places itself", sub_path.value),
            ));
        }
        let read = match (self.read.get(&resolved), text) {
            (Some(read), _) => std::rc::Rc::clone(read),
            (None, None) => unreachable!("a located document is read where it is located"),
            (None, Some(text)) => {
                let mut document = entelechy_language::parse_document(&text)
                    .map_err(|e| error(sub_path.span, format!("{}:{e}", sub_path.value)))?;
                let sub_base = resolved.parent().unwrap_or_else(|| Path::new("."));
                let entelechy_language::Declarations {
                    structs,
                    types,
                    functions,
                    qualified,
                } = (self.declare)(
                    sub_base,
                    &document.uses,
                    entelechy_language::Own::of(&document),
                )
                .map_err(|m| error(sub_path.span, format!("{}: {m}", sub_path.value)))?;
                document.unqualify(&qualified);
                document.resolve((&structs, &types, &functions));
                let read = std::rc::Rc::new(Read {
                    aliases: types.iter().chain(&document.types).cloned().collect(),
                    structs: structs.iter().chain(&document.structs).cloned().collect(),
                    functions: functions
                        .iter()
                        .chain(&document.functions)
                        .cloned()
                        .collect(),
                    text,
                    document,
                });
                self.read
                    .insert(resolved.clone(), std::rc::Rc::clone(&read));
                read
            }
        };
        let outer = (
            std::mem::replace(&mut self.aliases, read.aliases.clone()),
            std::mem::replace(&mut self.structs, read.structs.clone()),
            std::mem::replace(&mut self.functions, read.functions.clone()),
        );
        let sub_name = resolved
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("graph")
            .to_owned();
        let sub_base = resolved.parent().unwrap_or(base).to_path_buf();
        self.out.documents.push(resolved.clone());
        self.stack.push(resolved.clone());
        let placed = Placement {
            prefix: format!("{}.", id.0),
            inputs,
            params: node_params,
            awake,
        };
        let level = self
            .level(
                &read.document,
                &read.text,
                &sub_name,
                &sub_base,
                &resolved,
                Some(&placed),
            )
            .map_err(|e| {
                // `path:line:col: why` where the fault is placed in it,
                // else `path: why`.
                let gap = if e.span.is_placed() { "" } else { " " };
                error(sub_path.span, format!("{}:{gap}{e}", sub_path.value))
            });
        self.stack.pop();
        (self.aliases, self.structs, self.functions) = outer;
        let level = level?;
        Ok(level.outputs)
    }

    /// Lowers one document whose source is `text`. `placement` is `None`
    /// for the root.
    fn level(
        &mut self,
        document: &Document,
        text: &str,
        name: &str,
        base: &Path,
        path: &Path,
        placement: Option<&Placement>,
    ) -> Result<Level, DocumentError> {
        let prefix = placement.map(|p| p.prefix.as_str()).unwrap_or("");
        for item in &document.uses {
            self.out
                .uses
                .push((base.join(&item.path.value), item.path.value.ends_with('/')));
        }
        let Inputs {
            externals,
            assets,
            texts,
            sequences,
            checks,
            files,
            outlines,
            feeds,
        } = self.inputs(document, base, placement)?;
        let params = self.params(document)?;
        // The root's params, read before any sub-graph: a sub-graph's
        // `times` may name one through its placement.
        if placement.is_none() {
            self.out.graph.params = params.clone();
        }
        if let Some(placement) = placement {
            check_placement(placement, &externals, &params)?;
        }
        let place = Place {
            prefix,
            placement,
            params: &params,
            known: self.known(),
            nodes: document
                .nodes
                .iter()
                .map(|node| node.name.value.clone())
                .collect(),
        };
        for decl in &document.requires {
            let requirement = place.requirement(decl)?;
            self.out.graph.requirements.push(Requirement {
                at: Some(written_at(path, decl.name.span)),
                ..requirement
            });
        }
        let mut aliases: BTreeMap<String, Vec<(String, Source)>> = BTreeMap::new();
        let first_node = self.out.graph.nodes.len();
        let mut fonts: Vec<FontNode> = Vec::new();
        let mut file_nodes: Vec<FilesNode> = Vec::new();
        let mut field_nodes: Vec<FieldsNode> = Vec::new();
        // Each file input's fallback while the file is absent, renamed
        // with the nodes it may name.
        let mut fallback_refs: Vec<SourceRef> = document
            .inputs
            .iter()
            .filter_map(|input| input.otherwise.clone())
            .collect();
        let (nodes, declared_outputs, repeated) =
            self.expand_times(document, placement, &params, &mut fallback_refs)?;
        // A repeated node's name stays readable as its last time's.
        self.out
            .graph
            .repeated
            .extend(repeated.into_iter().map(|(name, last)| {
                (
                    format!("{prefix}{name}"),
                    NodeId(format!("{prefix}{last}").into()),
                )
            }));
        for decl in &nodes {
            self.placed += 1;
            if self.placed > MAX_PLACED {
                return Err(error(
                    decl.name.span,
                    format!("the scene places more than {MAX_PLACED} nodes, every level together"),
                ));
            }
            let id = NodeId(format!("{prefix}{}", decl.name.value).into());
            let mut inputs = BTreeMap::new();
            let mut node_params = BTreeMap::new();
            let mut domain = None;
            let mut is_async = false;
            // What the node is awake by: its placement's, then its own.
            let mut awake: Vec<Awake> = placement.map(|p| p.awake.clone()).unwrap_or_default();
            let mut own_awake: Vec<Awake> = Vec::new();
            let mut bind = |port: &entelechy_language::Spanned<String>, binding: Binding| {
                if inputs.insert(port.value.clone(), binding).is_some() {
                    return Err(error(
                        port.span,
                        format!("port `{}` is bound twice", port.value),
                    ));
                }
                Ok(())
            };
            for member in &decl.members {
                match member {
                    NodeMember::Set(name, expr) if name.value == "domain" => {
                        if domain.replace(domain_spec(expr)?).is_some() {
                            return Err(error(name.span, "`domain` is set twice"));
                        }
                    }
                    NodeMember::Set(name, expr) if name.value == "async" => {
                        let ExprKind::Bool(value) = &expr.kind else {
                            return Err(error(expr.span, "`async` is `true` or `false`"));
                        };
                        is_async = *value;
                    }
                    NodeMember::Set(name, expr) if name.value == "awake" => {
                        own_awake.push(place.awake(expr)?);
                    }
                    NodeMember::Set(name, expr) => {
                        if node_params
                            .insert(name.value.clone(), place.value(expr)?)
                            .is_some()
                        {
                            return Err(error(
                                name.span,
                                format!("parameter `{}` is set twice", name.value),
                            ));
                        }
                    }
                    NodeMember::Bind(port, reference) => bind(port, place.now(reference)?)?,
                    NodeMember::Previous {
                        port,
                        source,
                        initial,
                        migration,
                    } => bind(port, place.previous(source, initial, migration.as_ref())?)?,
                    NodeMember::Carry(port, _) => {
                        return Err(error(port.span, "`carry` needs `times = N;` on the node"));
                    }
                }
            }
            awake.extend(own_awake.iter().cloned());
            match &decl.kind {
                NodeKind::Reference(kernel) => {
                    let kernel_id = kernel
                        .value
                        .parse::<KernelId>()
                        .map_err(|e| error(kernel.span, e.to_string()))?;
                    let domain = place.domain(domain, &node_params)?;
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                        awake: awake.clone(),
                    });
                }
                NodeKind::Inline(kernel) => {
                    let kernel_id = match &kernel.id {
                        Some(written) => written
                            .value
                            .parse::<KernelId>()
                            .map_err(|e| error(written.span, e.to_string()))?,
                        None => anonymous_id(name, &decl.name.value),
                    };
                    for input in &kernel.inputs {
                        match &input.binding {
                            Some(BindingDecl::Now(reference)) => {
                                bind(&input.name, place.now(reference)?)?;
                            }
                            Some(BindingDecl::Previous {
                                source,
                                initial,
                                migration,
                            }) => {
                                bind(
                                    &input.name,
                                    place.previous(source, initial, migration.as_ref())?,
                                )?;
                            }
                            None => {
                                return Err(error(
                                    input.name.span,
                                    format!(
                                        "inline kernel input `{}` must be wired: `input {}: Image <- source;`",
                                        input.name.value, input.name.value
                                    ),
                                ));
                            }
                        }
                    }
                    if let Some(expr) = &kernel.domain {
                        domain = Some(domain_spec(expr)?);
                    }
                    // Param defaults that name a graph param become node
                    // values; the compiled Kernel keeps a literal default,
                    // the graph param's as declared, so its text is the
                    // same wherever the graph is placed.
                    let mut compiled = (**kernel).clone();
                    let mut edits = Vec::new();
                    for param in &mut compiled.params {
                        if let ExprKind::Name(written) = &param.default.kind {
                            let value = place.value(&param.default)?;
                            let literal = place
                                .declared(written)
                                .unwrap_or_else(|| place.default_of(&value));
                            edits.push((param.default_range.clone(), literal.to_string()));
                            param.default = literal_expr(literal, param.default.span);
                            node_params.insert(param.name.value.clone(), value);
                        }
                    }
                    let text = publishable_text(text, kernel, &kernel_id, edits);
                    self.out.inline.push(InlineKernel {
                        node: id.clone(),
                        id: kernel_id.clone(),
                        kernel: compiled,
                        text,
                        document: path.to_path_buf(),
                    });
                    let domain = place.domain(domain, &node_params)?;
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                        awake: awake.clone(),
                    });
                }
                NodeKind::Fill(ty) => {
                    if domain.is_none() {
                        return Err(error(
                            ty.span,
                            "a filled Field needs a `domain`: `domain = 65;`, `domain = vec2(96, 64);`",
                        ));
                    }
                    let kernel_id = anonymous_id(name, &decl.name.value);
                    // The fill is the node's `value` parameter, defaulting
                    // to the element's zero.
                    let spelled = resolved(&ty.value, &self.aliases);
                    let zero = zero_of(&spelled).ok_or_else(|| {
                        error(
                            ty.span,
                            format!(
                                "a filled Field holds Float, Int, Bool, Vec2, Vec3 or Vec4 (and `Image` holds a pixel); `{}` holds something else, so a Kernel fills it",
                                ty.value
                            ),
                        )
                    })?;
                    // A written number takes the Field element's width.
                    let value = match node_params.get("value") {
                        Some(given) => {
                            let given = place.default_of(given);
                            given.shaped_like(&zero).ok_or_else(|| {
                                error(
                                    ty.span,
                                    format!(
                                        "`value = {}` does not fill a `{}`: give one of its elements",
                                        literal_text(&given),
                                        ty.value
                                    ),
                                )
                            })?
                        }
                        None => zero,
                    };
                    let text = format!(
                        "kernel {kernel_id} {{\n    requires {{ field.write; }}\n    param value: {} = {};\n    output result: {};\n    compute {{ return value; }}\n}}\n",
                        value
                            .ty()
                            .map_or_else(|| ty.value.clone(), |t| t.to_string()),
                        literal_text(&value),
                        ty.value
                    );
                    let mut kernel = entelechy_language::parse(&text).map_err(|e| {
                        error(ty.span, format!("a filled Field of `{}`: {e}", ty.value))
                    })?;
                    // It inherits the document's declarations, aliases
                    // included.
                    kernel.structs = self.structs.clone();
                    kernel.types = self.aliases.clone();
                    kernel.functions = self.functions.clone();
                    self.out.inline.push(InlineKernel {
                        node: id.clone(),
                        id: kernel_id.clone(),
                        kernel,
                        text,
                        document: path.to_path_buf(),
                    });
                    let domain = place.domain(domain, &node_params)?;
                    self.out.graph.nodes.push(Node {
                        at: Some(written_at(path, decl.name.span)),
                        id,
                        kernel: kernel_id,
                        inputs,
                        params: node_params,
                        domain,
                        is_async,
                        awake: awake.clone(),
                    });
                }
                NodeKind::Font(font_path) => {
                    if domain.is_some()
                        || is_async
                        || !own_awake.is_empty()
                        || !node_params.is_empty()
                    {
                        return Err(error(
                            font_path.span,
                            "a font node takes no `domain`, `async`, `awake` or parameters: it is a row a code of `codes`",
                        ));
                    }
                    if let Some(other) = inputs.keys().find(|port| port.as_str() != "codes") {
                        return Err(error(
                            font_path.span,
                            format!("a font node's one input is `codes`, not `{other}`"),
                        ));
                    }
                    let codes = match inputs.remove("codes") {
                        Some(Binding::Now(source)) => source,
                        Some(Binding::Previous { .. }) => {
                            return Err(error(
                                font_path.span,
                                "a font node reads `codes` as last evaluated already: write `codes <- node.port;`",
                            ));
                        }
                        None => {
                            return Err(error(
                                font_path.span,
                                "a font node needs `codes <- node.port;`, the characters it gives the glyphs of",
                            ));
                        }
                    };
                    let outputs: Vec<(String, String)> = FONT_OUTPUTS
                        .iter()
                        .map(|port| ((*port).to_owned(), format!("{}.{port}", id.0)))
                        .collect();
                    for (_, external) in &outputs {
                        self.out.graph.externals.push(external.clone());
                    }
                    aliases.insert(
                        id.0.to_string(),
                        outputs
                            .iter()
                            .map(|(port, external)| {
                                (port.clone(), Source::External(external.clone()))
                            })
                            .collect(),
                    );
                    fonts.push(FontNode {
                        node: id.0.to_string(),
                        path: base.join(&font_path.value),
                        codes,
                        outputs,
                    });
                }
                NodeKind::Files(dir_path) => {
                    if domain.is_some()
                        || is_async
                        || !own_awake.is_empty()
                        || !node_params.is_empty()
                    {
                        return Err(error(
                            dir_path.span,
                            "a files node takes no `domain`, `async`, `awake` or parameters: its outputs are as long as what it lists and reads",
                        ));
                    }
                    if let Some(other) = inputs.keys().find(|port| port.as_str() != "paths") {
                        return Err(error(
                            dir_path.span,
                            format!("a files node's one input is `paths`, not `{other}`"),
                        ));
                    }
                    let paths = match inputs.remove("paths") {
                        Some(Binding::Now(source)) => source,
                        Some(Binding::Previous { .. }) => {
                            return Err(error(
                                dir_path.span,
                                "a files node reads `paths` as last evaluated already: write `paths <- node.port;`",
                            ));
                        }
                        None => {
                            return Err(error(
                                dir_path.span,
                                "a files node needs `paths <- node.port;`, the files it reads (each path's bytes ended by a 0)",
                            ));
                        }
                    };
                    let outputs: Vec<(String, String)> = FILES_OUTPUTS
                        .iter()
                        .map(|port| ((*port).to_owned(), format!("{}.{port}", id.0)))
                        .collect();
                    for (_, external) in &outputs {
                        self.out.graph.externals.push(external.clone());
                    }
                    aliases.insert(
                        id.0.to_string(),
                        outputs
                            .iter()
                            .map(|(port, external)| {
                                (port.clone(), Source::External(external.clone()))
                            })
                            .collect(),
                    );
                    file_nodes.push(FilesNode {
                        node: id.0.to_string(),
                        dir: base.join(&dir_path.value),
                        paths,
                        outputs,
                    });
                }
                NodeKind::Fields { ty, dir } => {
                    if domain.is_some()
                        || is_async
                        || !own_awake.is_empty()
                        || !node_params.is_empty()
                    {
                        return Err(error(
                            dir.span,
                            "a fields node takes no `domain`, `async`, `awake` or parameters: its result is as long as the files it holds",
                        ));
                    }
                    let resolved_ty = resolved(&ty.value, &self.aliases);
                    if !resolved_ty.starts_with("Rows<") {
                        return Err(error(
                            ty.span,
                            format!(
                                "a fields node's files follow one another in its result: a `Rows` of their element, not `{resolved_ty}`"
                            ),
                        ));
                    }
                    if let Some(other) = inputs.keys().find(|port| port.as_str() != "paths") {
                        return Err(error(
                            dir.span,
                            format!("a fields node's one input is `paths`, not `{other}`"),
                        ));
                    }
                    let paths = match inputs.remove("paths") {
                        Some(Binding::Now(source)) => source,
                        Some(Binding::Previous { .. }) => {
                            return Err(error(
                                dir.span,
                                "a fields node reads `paths` as last evaluated already: write `paths <- node.port;`",
                            ));
                        }
                        None => {
                            return Err(error(
                                dir.span,
                                "a fields node needs `paths <- node.port;`, the Field files it reads (each path's bytes ended by a 0)",
                            ));
                        }
                    };
                    let outputs: Vec<(String, String)> = FIELDS_OUTPUTS
                        .iter()
                        .map(|port| ((*port).to_owned(), format!("{}.{port}", id.0)))
                        .collect();
                    for (_, external) in &outputs {
                        self.out.graph.externals.push(external.clone());
                    }
                    aliases.insert(
                        id.0.to_string(),
                        outputs
                            .iter()
                            .map(|(port, external)| {
                                (port.clone(), Source::External(external.clone()))
                            })
                            .collect(),
                    );
                    field_nodes.push(FieldsNode {
                        node: id.0.to_string(),
                        dir: base.join(&dir.value),
                        ty: resolved_ty,
                        paths,
                        outputs,
                    });
                }
                NodeKind::SubGraph(sub_path) => {
                    if domain.is_some() {
                        return Err(error(
                            sub_path.span,
                            "a sub-graph takes no `domain`; its nodes set their own",
                        ));
                    }
                    if is_async {
                        return Err(error(
                            sub_path.span,
                            "a sub-graph takes no `async`; its nodes say which of them run off the frame",
                        ));
                    }
                    let outputs =
                        self.place(sub_path, base, &id, inputs, node_params, awake.clone())?;
                    self.out
                        .graph
                        .placed
                        .push((id.0.to_string(), outputs.clone()));
                    aliases.insert(id.0.to_string(), outputs);
                }
            }
        }
        // Sources that named a placed sub-graph now point at its outputs.
        let resolve = |source: &mut Source| -> Result<(), DocumentError> {
            if let Source::Node { node, port } = source
                && let Some(outputs) = aliases.get(node.0.as_str())
            {
                *source = outputs
                    .iter()
                    .find(|(name, _)| name == port)
                    .map(|(_, source)| source.clone())
                    .ok_or_else(|| {
                        // Where the document wrote it, and what it has.
                        let span = document
                            .source_refs()
                            .into_iter()
                            .find(|r| {
                                r.node
                                    .as_ref()
                                    .is_some_and(|n| format!("{prefix}{n}") == node.0)
                                    && &r.name == port
                            })
                            .map_or(Span::default(), |r| r.span);
                        let names: Vec<&str> =
                            outputs.iter().map(|(name, _)| name.as_str()).collect();
                        let near = entelechy_language::nearest(port, names.iter().copied());
                        error(
                            span,
                            format!(
                                "sub-graph `{node}` has no output `{port}`{}; its outputs are {}",
                                entelechy_language::suggestion(near),
                                names
                                    .iter()
                                    .map(|name| format!("`{name}`"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        )
                    })?;
            }
            Ok(())
        };
        for font in &mut fonts {
            resolve(&mut font.codes)?;
        }
        self.out.files.fonts.extend(fonts);
        for files in &mut file_nodes {
            resolve(&mut files.paths)?;
        }
        self.out.files.file_nodes.extend(file_nodes);
        for fields in &mut field_nodes {
            resolve(&mut fields.paths)?;
        }
        self.out.files.field_nodes.extend(field_nodes);
        for node in &mut self.out.graph.nodes[first_node..] {
            for awake in &mut node.awake {
                resolve(&mut awake.source)?;
            }
            for binding in node.inputs.values_mut() {
                match binding {
                    Binding::Now(source) => resolve(source)?,
                    Binding::Previous {
                        source, initial, ..
                    } => {
                        resolve(source)?;
                        resolve(initial)?;
                    }
                }
            }
        }
        // A document with nodes, or with nothing at all, is a scene; one
        // of declarations alone (Kernels, functions, structs, types, uses,
        // a manifest) is a library and shows nothing.
        let a_scene = !document.nodes.is_empty()
            || (document.kernels.is_empty()
                && document.uses.is_empty()
                && document.functions.is_empty()
                && document.structs.is_empty()
                && document.types.is_empty()
                && document.package.is_none());
        if placement.is_none() && declared_outputs.is_empty() && a_scene {
            return Err(error(
                Span::default(),
                "the scene declares no output; it shows one of its nodes' outputs with `output name <- node.port;`",
            ));
        }
        let mut outputs: Vec<(String, Source)> = Vec::new();
        for (output, reference) in &declared_outputs {
            let mut source = match place.now(reference)? {
                Binding::Now(source) => source,
                Binding::Previous { .. } => {
                    return Err(error(reference.span, "an output must be a current value"));
                }
            };
            resolve(&mut source)?;
            if outputs.iter().any(|(name, _)| name == &output.value) {
                return Err(error(
                    output.span,
                    format!("output `{}` is declared twice", output.value),
                ));
            }
            outputs.push((output.value.clone(), source));
        }
        let mut fallbacks = Vec::new();
        let otherwise = document
            .inputs
            .iter()
            .filter_map(|input| input.otherwise.as_ref().map(|_| &input.name.value));
        for (input, reference) in otherwise.zip(&fallback_refs) {
            let mut source = match place.now(reference)? {
                Binding::Now(source) => source,
                Binding::Previous { .. } => {
                    return Err(error(reference.span, "a fallback must be a current value"));
                }
            };
            resolve(&mut source)?;
            if source == Source::External(input.clone()) {
                return Err(error(
                    reference.span,
                    format!("input `{input}` cannot fall back on itself"),
                ));
            }
            fallbacks.push((input.clone(), source));
        }
        let mut writes = Vec::new();
        for write in &document.writes {
            if placement.is_some() {
                return Err(error(
                    write.path.span,
                    "only the scene itself writes files; a placed graph's outputs are wired by the placing graph",
                ));
            }
            if writes
                .iter()
                .any(|w: &Write| w.path == base.join(&write.path.value))
            {
                return Err(error(
                    write.path.span,
                    format!("`{}` is written by two outputs", write.path.value),
                ));
            }
            writes.push(Write {
                output: write.output.clone(),
                path: base.join(&write.path.value),
                bytes: write.bytes,
                files: write.files,
            });
        }
        // A document's devices are its own when it is the scene; placed,
        // its outputs are wired by the placing graph, as its first output is
        // the window's only when it is the scene.
        let mut presented: Vec<Present> = Vec::new();
        for present in document.presents.iter().filter(|_| placement.is_none()) {
            if presented.iter().any(|p| p.device == present.device.value) {
                return Err(error(
                    present.device.span,
                    format!(
                        "two outputs go `to {}`; a device takes one",
                        present.device.value
                    ),
                ));
            }
            presented.push(Present {
                output: present.output.clone(),
                device: present.device.value.clone(),
                at: present.device.span,
            });
        }
        Ok(Level {
            externals,
            assets,
            texts,
            sequences,
            checks,
            files,
            outlines,
            feeds,
            params,
            outputs,
            fallbacks,
            writes,
            presented,
        })
    }
}

/// What a literal of a document may name: its structs, each with its
/// fields in order, and its enums, each with its variants' numbers.
#[derive(Clone, Debug, Default)]
struct Known {
    structs: Vec<(String, Vec<String>)>,
    enums: Vec<(String, Vec<(String, i32)>)>,
}

impl Known {
    /// `expr` with the variants it names written as their `Int`s.
    fn with_variants(&self, expr: &Expr) -> Result<Expr, DocumentError> {
        let variants_of = |name: &str| {
            self.enums
                .iter()
                .find(|(declared, _)| declared == name)
                .map(|(_, variants)| variants.clone())
        };
        entelechy_compiler::with_variants(expr, &variants_of).map_err(|e| compiled(expr.span, e))
    }
}

/// A literal as a scene writes it: any value a Kernel can hold, a struct
/// literal of one of `known`, or a variant of one of its enums. A number
/// is kept exact (a written float as a `Double`); the Kernel's param gives
/// its width.
fn literal(expr: &Expr, known: &Known) -> Result<Value, DocumentError> {
    let expr = known.with_variants(expr)?;
    let fields_of = |name: &str| {
        known
            .structs
            .iter()
            .find(|(struct_name, _)| struct_name == name)
            .map(|(_, fields)| fields.clone())
    };
    entelechy_compiler::literal_exact(&expr, &fields_of).map_err(|e| compiled(expr.span, e))
}

/// A value's own literal as an expression, for rewriting param defaults.
fn literal_expr(value: Value, span: Span) -> Expr {
    let mut expr = entelechy_language::parse_literal(&value.to_string())
        .expect("a value's literal reads back");
    expr.span = span;
    expr
}

/// Parses a `.esn` document that references published Kernels only.
/// Inline Kernels and sub-graphs take [`load`].
pub fn parse(text: &str) -> Result<Graph, DocumentError> {
    let document = entelechy_language::parse_document(text).map_err(DocumentError::from)?;
    let mut loader = |path: &std::path::Path| -> Result<(PathBuf, String), String> {
        Err(format!(
            "sub-graph `{}` cannot be loaded here",
            path.display()
        ))
    };
    let mut none: Box<Declare<'_>> =
        Box::new(|_, _, _| Ok(entelechy_language::Declarations::default()));
    let lowered = lower(
        &document,
        text,
        "graph",
        std::path::Path::new("."),
        &mut loader,
        &mut none,
        &BTreeMap::new(),
    )?;
    if let Some(inline) = lowered.inline.first() {
        return Err(DocumentError {
            span: inline
                .kernel
                .id
                .as_ref()
                .map(|id| id.span)
                .unwrap_or_default(),
            message: format!(
                "node `{}` carries an inline kernel; load this document through a session",
                inline.node
            ),
        });
    }
    Ok(lowered.graph)
}

/// Lowers a parsed document that places no sub-graphs into a Graph named
/// `name`, reporting inline Kernels for the loader to publish. Anonymous
/// inline Kernels are named by [`anonymous_id`].
pub fn from_document(
    document: &entelechy_language::Document,
    name: &str,
) -> Result<Lowered, DocumentError> {
    let mut loader = |path: &std::path::Path| -> Result<(PathBuf, String), String> {
        Err(format!(
            "sub-graph `{}` cannot be loaded here; use entelechy_scene::load with a loader",
            path.display()
        ))
    };
    let mut none: Box<Declare<'_>> =
        Box::new(|_, _, _| Ok(entelechy_language::Declarations::default()));
    lower(
        document,
        "",
        name,
        std::path::Path::new("."),
        &mut loader,
        &mut none,
        &BTreeMap::new(),
    )
}

/// Lowers a parsed document into a Graph, loading its sub-graphs
/// through `loader` (paths resolved against `base`) and flattening them
/// in place: nodes prefixed with the placing node's name, inputs and
/// params taken from the parent.
pub fn load(
    document: &entelechy_language::Document,
    text: &str,
    name: &str,
    base: &std::path::Path,
    loader: &mut Loader<'_>,
) -> Result<Lowered, DocumentError> {
    let mut none: Box<Declare<'_>> =
        Box::new(|_, _, _| Ok(entelechy_language::Declarations::default()));
    lower(
        document,
        text,
        name,
        base,
        loader,
        &mut none,
        &BTreeMap::new(),
    )
}

/// Loads like [`load`], also resolving every document's `use` items
/// through `declare` for all its Kernels.
pub fn load_with(
    document: &entelechy_language::Document,
    text: &str,
    name: &str,
    base: &std::path::Path,
    loader: &mut Loader<'_>,
    declare: &mut Declare<'_>,
) -> Result<Lowered, DocumentError> {
    lower(
        document,
        text,
        name,
        base,
        loader,
        declare,
        &BTreeMap::new(),
    )
}

/// As [`load_with`], with parameter overrides a node's `times` may read.
pub fn load_with_params(
    document: &entelechy_language::Document,
    text: &str,
    name: &str,
    base: &std::path::Path,
    loader: &mut Loader<'_>,
    declare: &mut Declare<'_>,
    overrides: &BTreeMap<String, Value>,
) -> Result<Lowered, DocumentError> {
    lower(document, text, name, base, loader, declare, overrides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelechy_abi::Domain;

    const DOCUMENT: &str = "
// a comment
input photo: Image;
param amount: Float = 0.5;

node gray: image.grayscale@1 {
    source <- photo;
    amount = amount;
}
node blocks: image.pixelate@1 {
    source <- gray.result;
    size = 8;
    tint = vec4(1.0, 0.5, 0.25, 1.0);
    domain = vec2(64, 32);
}
output result <- blocks.result;
";

    #[test]
    fn a_written_number_keeps_its_digits_until_a_width_is_named() {
        let graph = parse(
            "struct Site { at: DVec2; id: Long; }
param origin: Double = 500000.123456789;
param site: Site = Site { at: dvec2(0.1, -4000000.987654321), id: 5000000000 };
param rows: [Double; 2] = [0.1, 0.2];
param scale: Float = 0.1;
node a: k.a@1 { x = 0.001; }
output result <- a.result;
",
        )
        .unwrap();
        let defaults: Vec<&Value> = graph.params.iter().map(|p| &p.default).collect();
        assert_eq!(defaults[0], &Value::Double(500000.123456789));
        assert_eq!(
            defaults[1],
            &Value::Record(vec![
                Value::DVec2([0.1, -4000000.987654321]),
                Value::Long(5_000_000_000)
            ])
        );
        assert_eq!(
            defaults[2],
            &Value::Record(vec![Value::Double(0.1), Value::Double(0.2)])
        );
        assert_eq!(defaults[3], &Value::Float(0.1));
        // A node's value takes its width from the Kernel's param and
        // reads back as written.
        let value = &graph.nodes[0].params["x"];
        assert_eq!(value, &ParamValue::Literal(Value::Double(0.001)));
        assert!(print(&graph).contains("x = 0.001;"), "{}", print(&graph));
    }

    #[test]
    fn documents_round_trip() {
        let graph = parse(DOCUMENT).unwrap();
        assert_eq!(graph.externals, ["photo"]);
        assert_eq!(graph.params[0].default, Value::Float(0.5));
        assert_eq!(
            graph.nodes[0].inputs["source"],
            Binding::Now(Source::External("photo".into()))
        );
        assert_eq!(
            graph.nodes[0].params["amount"],
            ParamValue::Graph("amount".into())
        );
        assert_eq!(
            graph.nodes[1].params["size"],
            ParamValue::Literal(Value::Int(8))
        );
        assert_eq!(
            graph.nodes[1].domain,
            Some(DomainSpec::Literal(Domain::Grid(64, 32)))
        );
        let text = print(&graph);
        assert_eq!(parse(&text).unwrap(), graph);
    }

    #[test]
    fn inline_kernels_are_reported_with_their_wiring() {
        let source = "input photo: Image;
             param decay: Float = 0.9;
             node trail: kernel {
                 requires { field.read; field.write; }
                 input current: Image <- photo;
                 input previous: Image <~ trail.result from photo;
                 param fade: Float = decay;
                 output result: Image;
                 domain = vec2(4, 4);
                 compute { return sample(current) * fade + sample(previous); }
             }
             node named: kernel fx.glow@2 {
                 requires { field.read; field.write; }
                 input source: Image <- trail.result;
                 output result: Image;
                 compute { return sample(source); }
             }
             output result <- named.result;";
        let document = entelechy_language::parse_document(source).unwrap();
        let mut none =
            |path: &Path| -> Result<(PathBuf, String), String> { Err(path.display().to_string()) };
        let lowered = load(&document, source, "My-Scene", Path::new("."), &mut none).unwrap();
        // The text publishes as is: the node's id written in, its domain
        // removed.
        assert!(
            !lowered.inline[0].text.contains("domain"),
            "{}",
            lowered.inline[0].text
        );
        assert!(
            entelechy_language::parse(&lowered.inline[0].text).is_ok(),
            "{}",
            lowered.inline[0].text
        );
        assert!(
            lowered.inline[0]
                .text
                .starts_with("kernel my_scene.trail@1 {"),
            "{}",
            lowered.inline[0].text
        );
        assert_eq!(lowered.inline.len(), 2);
        assert_eq!(lowered.inline[0].id.to_string(), "my_scene.trail@1");
        assert_eq!(lowered.inline[1].id.to_string(), "fx.glow@2");
        let trail = &lowered.graph.nodes[0];
        assert_eq!(trail.kernel.to_string(), "my_scene.trail@1");
        assert!(matches!(trail.inputs["previous"], Binding::Previous { .. }));
        assert_eq!(trail.domain, Some(DomainSpec::Literal(Domain::Grid(4, 4))));
        assert_eq!(trail.params["fade"], ParamValue::Graph("decay".into()));
        let compiled = entelechy_compiler::compile_kernel(
            &lowered.inline[0].kernel,
            Some(&lowered.inline[0].id),
        )
        .unwrap();
        assert_eq!(compiled.manifest.params[0].default, Value::Float(0.9));
    }

    /// A param a placement gives that the graph lacks is refused naming
    /// the graph as written and the node placing it, with the nearest
    /// param, and never a path the graph was not found at.
    #[test]
    fn a_placement_lacking_a_param_names_the_node_and_the_nearest() {
        let placed = "input source: Image;
            param strength: Float = 1.0;
            node map: image.tonemap@1 { source <- source; exposure = strength; }
            output result <- map.result;";
        let root = entelechy_language::parse_document(
            "node scene: image.gradient@1 { domain = vec2(8, 8); }
             node look: graph \"fx.entpkg/graphs/look.esn\" { source <- scene.result; strenght = 0.5; }
             output result <- look.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            Ok((path.to_path_buf(), placed.to_owned()))
        };
        let error = load(&root, "", "main", Path::new("app"), &mut loader).unwrap_err();
        assert_eq!(
            error.message,
            "fx.entpkg/graphs/look.esn: sub-graph `look` has no param `strenght`; did you mean `strength`?; its params are `strength`"
        );
    }

    /// A sub-graph placed many times is read once: each placement is its
    /// own nodes, from the one document.
    #[test]
    fn a_font_node_in_a_placed_graph_is_three_externals_named_where_it_stands() {
        let text = "node codes: t.codes@1 { domain = 4; }
            node g: font \"fonts/x.ttf\" { codes <- codes.result; }
            node draw: t.draw@1 { glyphs <- g.glyphs; outlines <- g.outlines; domain = 1; }
            output result <- draw.result;";
        let root = entelechy_language::parse_document(
            "node ed: graph \"parts/editor.esn\" {}
             output result <- ed.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            Ok((path.to_path_buf(), text.to_owned()))
        };
        let lowered = load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
        assert_eq!(
            lowered.graph.externals,
            ["ed.g.glyphs", "ed.g.outlines", "ed.g.metrics"]
        );
        let font = &lowered.files.fonts[0];
        assert_eq!(font.node, "ed.g");
        assert_eq!(font.path, Path::new("root/parts/fonts/x.ttf"));
        assert_eq!(
            font.codes,
            Source::Node {
                node: NodeId("ed.codes".into()),
                port: "result".into()
            }
        );
        let draw = lowered
            .graph
            .nodes
            .iter()
            .find(|n| n.id.0 == "ed.draw")
            .unwrap();
        assert_eq!(
            draw.inputs["glyphs"],
            Binding::Now(Source::External("ed.g.glyphs".into()))
        );
    }

    #[test]
    fn a_sub_graph_placed_many_times_is_read_once() {
        let display = "input source: Image;
            node map: image.tonemap@1 { source <- source; }
            output result <- map.result;";
        let root = entelechy_language::parse_document(
            "node scene: image.gradient@1 { domain = vec2(8, 8); }
             node a: graph \"d.esn\" { source <- scene.result; }
             node b: graph \"d.esn\" { source <- a.result; }
             node c: graph \"d.esn\" { source <- b.result; }
             output result <- c.result;",
        )
        .unwrap();
        let mut reads = 0;
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            reads += 1;
            Ok((path.to_path_buf(), display.to_owned()))
        };
        let lowered = load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
        let ids: Vec<&str> = lowered
            .graph
            .nodes
            .iter()
            .map(|n| n.id.0.as_str())
            .collect();
        assert_eq!(ids, ["scene", "a.map", "b.map", "c.map"]);
        assert_eq!(reads, 1);
    }

    #[test]
    fn requirements_flatten_named_with_their_place_reading_what_the_parent_passed() {
        let grid = "param columns: Int = 64;
            param bits: Int = 12;
            require cells: columns * 2 <= 1 << bits;
            node n: t.k@1 { domain = columns; }
            output result <- n.result;";
        let root = entelechy_language::parse_document(
            "param wide: Int = 100;
             require positive: wide > 0;
             node a: graph \"grid.esn\" { columns = wide; }
             node b: graph \"grid.esn\" { bits = 7; }
             output result <- a.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            Ok((path.to_path_buf(), grid.to_owned()))
        };
        let lowered = load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
        let written: Vec<String> = lowered
            .graph
            .requirements
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            written,
            [
                "positive: wide > 0",
                "a.cells: wide * 2 <= 1 << 12",
                "b.cells: 64 * 2 <= 1 << 7"
            ]
        );
        assert!(
            lowered.graph.requirements[2]
                .at
                .as_deref()
                .unwrap()
                .ends_with(":3:21")
        );
        // What holds and what does not, with the root's param as given.
        let param = |wide: i32| move |name: &str| (name == "wide").then_some(Value::Int(wide));
        let none = |_: &str| None;
        assert!(
            lowered.graph.requirements[1]
                .check(&param(100), &none)
                .is_ok()
        );
        let failed = lowered.graph.requirements[1]
            .check(&param(3000), &none)
            .unwrap_err();
        assert!(
            failed
                .message
                .contains("here 6000 <= 4096, with wide = 3000"),
            "{failed}"
        );
        // Broken by the params as given, the document is refused as it loads.
        let broken = entelechy_language::parse_document(
            "node b: graph \"grid.esn\" { bits = 3; } output result <- b.result;",
        )
        .unwrap();
        let refused = load(&broken, "", "main", Path::new("root"), &mut loader).unwrap_err();
        assert!(refused.message.contains("here 128 <= 8"), "{refused}");
        // A root graph's requirement prints and reads back.
        let graph = parse(
            "param wide: Int = 4;\nrequire cells: wide * wide <= 1 << 5;\nnode n: t.k@1 { domain = wide; }\noutput result <- n.result;\n",
        )
        .unwrap();
        assert!(print(&graph).contains("require cells: wide * wide <= 1 << 5;"));
        assert_eq!(parse(&print(&graph)).unwrap(), graph);
        // A name not a param, a port, or no comparison is refused.
        for (text, says) in [
            (
                "param a: Int = 1; require r: a <= b; node n: t.k@1 { domain = 1; } output o <- n.result;",
                "`b`, which is not a param",
            ),
            (
                "param a: Int = 1; require r: count(x) <= a; node n: t.k@1 { domain = 1; } output o <- n.result;",
                "a node's ports",
            ),
            (
                "param a: Int = 1; require r: a + 1; node n: t.k@1 { domain = 1; } output o <- n.result;",
                "compares with",
            ),
        ] {
            let document = entelechy_language::parse_document(text).unwrap();
            let error = load(&document, text, "main", Path::new("root"), &mut loader).unwrap_err();
            assert!(error.message.contains(says), "{error}");
        }
    }

    #[test]
    fn sub_graphs_flatten_with_prefixed_nodes_wired_inputs_and_passed_params() {
        let display = "use \"lib/\";
            input source: Image;
            param exposure: Float = 1.0;
            node map: image.tonemap@1 { source <- source; exposure = exposure; }
            node fade: image.fade@1 { previous <~ fade.result from map.result; }
            output result <- fade.result;";
        let root = entelechy_language::parse_document(
            "param strength: Float = 2.0;
             node scene: image.gradient@1 { domain = vec2(8, 8); }
             node display: graph \"shared/display.esn\" { source <- scene.result; exposure = strength; }
             node again: graph \"shared/display.esn\" { source <- display.result; }
             output result <- again.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            assert_eq!(path, Path::new("root/shared/display.esn"));
            Ok((path.to_path_buf(), display.to_owned()))
        };
        let lowered = load(&root, "", "main", Path::new("root"), &mut loader).unwrap();
        let graph = &lowered.graph;
        let ids: Vec<&str> = graph.nodes.iter().map(|n| n.id.0.as_str()).collect();
        assert_eq!(
            ids,
            [
                "scene",
                "display.map",
                "display.fade",
                "again.map",
                "again.fade"
            ]
        );
        assert_eq!(
            graph.nodes[1].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("scene".into()),
                port: "result".into()
            })
        );
        assert_eq!(
            graph.nodes[1].params["exposure"],
            ParamValue::Graph("strength".into())
        );
        assert_eq!(
            graph.nodes[3].params["exposure"],
            ParamValue::Literal(Value::Float(1.0))
        );
        assert_eq!(
            graph.nodes[3].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("display.fade".into()),
                port: "result".into()
            })
        );
        assert!(
            matches!(&graph.nodes[2].inputs["previous"], Binding::Previous { source: Source::Node { node, .. }, .. } if node.0 == "display.fade")
        );
        assert_eq!(
            graph.outputs[0].1,
            Source::Node {
                node: NodeId("again.fade".into()),
                port: "result".into()
            }
        );
        assert_eq!(
            lowered.uses,
            vec![(PathBuf::from("root/shared/lib/"), true); 2]
        );
        assert_eq!(
            lowered.documents,
            vec![PathBuf::from("root/shared/display.esn"); 2]
        );
        assert!(graph.order().is_ok());
    }

    #[test]
    fn a_bare_node_name_and_a_scene_without_output_say_what_to_write() {
        let error =
            parse("node a: t.k@1 { }\nnode b: t.k@1 { x <- a; }\noutput r <- b.r;").unwrap_err();
        assert!(error.message.contains("`a.result`"), "{error}");
        assert_eq!(error.span.line, 2);
        for text in ["node a: t.k@1 { }", ""] {
            let error = parse(text).unwrap_err();
            assert!(
                error.message.contains("output name <- node.port;"),
                "{error}"
            );
        }
        // A document of declarations alone is a library, not a scene.
        for library in [
            "kernel t.k@1 { output result: Image; compute { return vec4(0.0); } }",
            "fn twice(x: Float) -> Float { return x * 2.0; }",
            "struct P { x: Float; }",
            "type Cells = Grid<Int>;",
        ] {
            assert!(parse(library).is_ok(), "{library}");
        }
    }

    #[test]
    fn errors_name_the_position() {
        let error = parse("node a: Bad {\n}\noutput r <- a.r;").unwrap_err();
        assert_eq!(error.span.line, 1);
        let error = parse("node a: t.k@1 { x <- y; x <- z; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("bound twice"), "{error}");
        let error = parse("node a: t.k@1 { domain = true; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("`domain` is a count"), "{error}");
        let error = parse("node a: kernel { requires { field.write; } input s: Image; output result: Image; compute { return sample(s); } }\noutput r <- a.result;").unwrap_err();
        assert!(error.message.contains("must be wired"), "{error}");
        let error = parse("node a: t.k@1 { y = 1.0 + 2.0; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("literal"), "{error}");
        let error = parse("node a: t.k@1 { y = nope; }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("not a param"), "{error}");
        let error = parse("node a: graph \"x.esn\" { }\noutput r <- a.r;").unwrap_err();
        assert!(error.message.contains("cannot load"), "{error}");
    }
}

#[cfg(test)]
mod times_tests {
    use super::*;

    const REPEATED: &str = "
param level: Int = 2;
node seed: test.zero@1 { domain = 4; }
node step: test.inc@1 { times = level; carry previous <- result; previous <- seed.result; }
node show: test.show@1 { source <- step.result; }
output result <- show.result;
";

    fn ids(graph: &Graph) -> Vec<String> {
        graph.nodes.iter().map(|n| n.id.0.to_string()).collect()
    }

    #[test]
    fn times_expands_a_node_and_carries_between_times() {
        let graph = parse(REPEATED).unwrap();
        assert_eq!(ids(&graph), ["seed", "step/1", "step/2", "show"]);
        assert_eq!(graph.times_params, ["level"]);
        let first = &graph.nodes[1];
        assert_eq!(
            first.inputs["previous"],
            Binding::Now(Source::Node {
                node: NodeId("seed".into()),
                port: "result".into()
            })
        );
        let second = &graph.nodes[2];
        assert_eq!(
            second.inputs["previous"],
            Binding::Now(Source::Node {
                node: NodeId("step/1".into()),
                port: "result".into()
            })
        );
        assert_eq!(
            graph.nodes[3].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("step/2".into()),
                port: "result".into()
            }),
            "the name means the last time"
        );
        // The caller's value sizes the graph; one time keeps the plain name.
        let document = entelechy_language::parse_document(REPEATED).unwrap();
        let mut none: Box<Declare<'_>> =
            Box::new(|_, _, _| Ok(entelechy_language::Declarations::default()));
        let mut loader: Box<Loader<'_>> = Box::new(|_| Err("no files".into()));
        let overrides = BTreeMap::from([("level".to_owned(), Value::Int(4))]);
        let lowered = load_with_params(
            &document,
            REPEATED,
            "g",
            Path::new("."),
            &mut loader,
            &mut none,
            &overrides,
        )
        .unwrap();
        assert_eq!(
            ids(&lowered.graph),
            ["seed", "step/1", "step/2", "step/3", "step/4", "show"]
        );
        let overrides = BTreeMap::from([("level".to_owned(), Value::Int(1))]);
        let lowered = load_with_params(
            &document,
            REPEATED,
            "g",
            Path::new("."),
            &mut loader,
            &mut none,
            &overrides,
        )
        .unwrap();
        assert_eq!(ids(&lowered.graph), ["seed", "step", "show"]);
    }

    /// A root param passed down through placed graphs to a `times` sizes
    /// the graph as one the root reads itself: it is recorded, so changing
    /// it lowers the document again, and the copies follow its value.
    #[test]
    fn a_root_param_sizes_a_times_placed_graphs_down() {
        let root = "param level: Int = 2;
node seed: test.zero@1 { domain = 4; }
node outer: graph \"outer.esn\" { source <- seed.result; level = level; }
output result <- outer.result;
";
        let document = entelechy_language::parse_document(root).unwrap();
        let mut none: Box<Declare<'_>> =
            Box::new(|_, _, _| Ok(entelechy_language::Declarations::default()));
        let mut loader: Box<Loader<'_>> = Box::new(|path: &Path| {
            let text = if path.ends_with("outer.esn") {
                "input source: Rows<Int>;
param level: Int = 5;
node inner: graph \"inner.esn\" { source <- source; level = level; }
output result <- inner.result;
"
            } else {
                "input source: Rows<Int>;
param level: Int = 5;
node step: test.inc@1 { times = level; carry previous <- result; previous <- source; }
output result <- step.result;
"
            };
            Ok((path.to_path_buf(), text.to_owned()))
        });
        for (level, copies) in [(None, 2), (Some(3), 3)] {
            let overrides: BTreeMap<String, Value> = level
                .map(|l| ("level".to_owned(), Value::Int(l)))
                .into_iter()
                .collect();
            let lowered = load_with_params(
                &document,
                root,
                "g",
                Path::new("."),
                &mut loader,
                &mut none,
                &overrides,
            )
            .unwrap();
            assert_eq!(lowered.graph.times_params, ["level"]);
            let steps = ids(&lowered.graph)
                .iter()
                .filter(|id| id.contains("step"))
                .count();
            assert_eq!(steps, copies, "{:?}", ids(&lowered.graph));
        }
    }

    /// `times = 0` places no node: a carried output is what its port is
    /// first bound to, through chains too; an uncarried output, or a
    /// carry started by `<~`, has nothing to pass.
    #[test]
    fn times_zero_passes_carried_outputs_through() {
        let graph =
            parse(&REPEATED.replace("param level: Int = 2;", "param level: Int = 0;")).unwrap();
        assert_eq!(ids(&graph), ["seed", "show"]);
        assert_eq!(
            graph.nodes[1].inputs["source"],
            Binding::Now(Source::Node {
                node: NodeId("seed".into()),
                port: "result".into()
            })
        );
        let chained = parse(
            "node seed: test.zero@1 { domain = 4; }
node a: test.inc@1 { times = 0; carry previous <- result; previous <- seed.result; }
node b: test.inc@1 { times = 0; carry previous <- result; previous <- a.result; }
node c: test.inc@1 { times = 2; carry previous <- result; previous <- b.result; }
output result <- b.result;
output last <- c.result;",
        )
        .unwrap();
        assert_eq!(ids(&chained), ["seed", "c/1", "c/2"]);
        assert_eq!(
            chained.nodes[1].inputs["previous"],
            Binding::Now(Source::Node {
                node: NodeId("seed".into()),
                port: "result".into()
            })
        );
        let uncarried = parse(
            "node seed: test.zero@1 { domain = 4; }
node a: test.two@1 { times = 0; carry previous <- result; previous <- seed.result; }
output result <- a.other;",
        )
        .unwrap_err();
        assert!(
            uncarried
                .message
                .contains("`a.other` has no value at `times = 0`"),
            "{uncarried}"
        );
        let stateful = parse(
            "node seed: test.zero@1 { domain = 4; }
node a: test.inc@1 { times = 0; carry previous <- result; previous <~ a.result from seed.result; }
output result <- a.result;",
        )
        .unwrap_err();
        assert!(stateful.message.contains("bound with `<~`"), "{stateful}");
    }

    /// A file output, and a file input's node fallback: the fallback
    /// follows the node's times to the last, orders before the input's
    /// readers and names a node, and only the root scene writes.
    #[test]
    fn writes_and_fallbacks_lower_into_the_graph() {
        let text = "input saved: Rows<Int> = \"saved.field\" else seed.result;
node seed: test.zero@1 { times = 2; carry previous <- result; previous <- first.result; }
node first: test.zero@1 { domain = 4; }
node show: test.show@1 { source <- saved; }
output result <- show.result;
output kept <- show.result to \"saves/kept.field\";
output raw <- first.result to bytes \"raw.bin\";";
        let lowered =
            from_document(&entelechy_language::parse_document(text).unwrap(), "graph").unwrap();
        let graph = lowered.graph;
        assert_eq!(
            graph.fallback("saved"),
            Some(&Source::Node {
                node: NodeId("seed/2".into()),
                port: "result".into()
            })
        );
        let order: Vec<&str> = graph
            .order()
            .unwrap()
            .into_iter()
            .map(|at| graph.nodes[at].id.0.as_str())
            .collect();
        let at = |name: &str| order.iter().position(|id| *id == name).unwrap();
        assert!(at("seed/2") < at("show"), "{order:?}");
        assert_eq!(
            lowered.files.writes,
            [
                Write {
                    output: "kept".into(),
                    path: PathBuf::from("./saves/kept.field"),
                    bytes: false,
                    files: false,
                },
                Write {
                    output: "raw".into(),
                    path: PathBuf::from("./raw.bin"),
                    bytes: true,
                    files: false,
                },
            ]
        );
        let itself = parse(
            "input saved: Rows<Int> = \"saved.field\" else saved;\nnode show: test.show@1 { source <- saved; }\noutput result <- show.result;",
        )
        .unwrap_err();
        assert!(itself.message.contains("fall back on itself"), "{itself}");
        let twice = parse(
            "node a: test.zero@1 { domain = 1; }\noutput x <- a.result to \"f.field\";\noutput y <- a.result to \"f.field\";",
        )
        .unwrap_err();
        assert!(twice.message.contains("written by two outputs"), "{twice}");
        let placed = entelechy_language::parse_document(
            "node inner: graph \"inner.esn\" { }\noutput result <- inner.result;",
        )
        .unwrap();
        let mut loader = |path: &Path| -> Result<(PathBuf, String), String> {
            Ok((
                path.to_path_buf(),
                "node a: test.zero@1 { domain = 1; }\noutput result <- a.result to \"f.field\";"
                    .into(),
            ))
        };
        let refused = load(&placed, "", "main", Path::new("."), &mut loader).unwrap_err();
        assert!(
            refused.message.contains("only the scene itself writes"),
            "{refused}"
        );
        // `to` and `else` are words of the document, not of any Kernel.
        let error =
            parse("node a: test.zero@1 { domain = 1; }\noutput x <- a.result to 3;").unwrap_err();
        assert!(error.message.contains("goes `to` a quoted path"), "{error}");
        // A bare name is a device, the root's only, one output a device.
        let lowered = from_document(
            &entelechy_language::parse_document(
                "node a: test.zero@1 { domain = 1; }\noutput tone <- a.result to sound;\noutput copied <- a.result to clipboard;",
            )
            .unwrap(),
            "graph",
        )
        .unwrap();
        let devices: Vec<(&str, &str)> = lowered
            .files
            .presented
            .iter()
            .map(|p| (p.output.as_str(), p.device.as_str()))
            .collect();
        assert_eq!(devices, [("tone", "sound"), ("copied", "clipboard")]);
        assert!(lowered.files.writes.is_empty());
        let twice = parse(
            "node a: test.zero@1 { domain = 1; }\noutput x <- a.result to sound;\noutput y <- a.result to sound;",
        );
        assert!(
            twice
                .unwrap_err()
                .message
                .contains("two outputs go `to sound`"),
            "a device takes one output"
        );
        let mut placing = |path: &Path| -> Result<(PathBuf, String), String> {
            Ok((
                path.to_path_buf(),
                "node a: test.zero@1 { domain = 1; }\noutput result <- a.result to sound;".into(),
            ))
        };
        let placing_lowered = load(&placed, "", "main", Path::new("."), &mut placing).unwrap();
        assert!(
            placing_lowered.files.presented.is_empty(),
            "a placed graph's devices are its own only when it is the scene"
        );
    }

    #[test]
    fn times_and_carry_are_checked() {
        let negative = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = -1; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(negative.message.contains("at least 0"), "{negative}");
        let unbound = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = 2; carry previous <- result; }\noutput result <- r.result;").unwrap_err();
        assert!(
            unbound
                .message
                .contains("needs a binding for the first time"),
            "{unbound}"
        );
        let orphan = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { carry previous <- result; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(orphan.message.contains("needs `times"), "{orphan}");
        let unknown = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = n; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(unknown.message.contains("not a param"), "{unknown}");
        // Counts past the scene's bound, alone or together, are refused
        // before any copy.
        let huge = parse("node s: t.z@1 { domain = 1; }\nnode r: t.i@1 { times = 2000000000; previous <- s.result; }\noutput result <- r.result;").unwrap_err();
        assert!(
            huge.message.contains("places more than 65536 nodes"),
            "{huge}"
        );
        let together = parse("node s: t.z@1 { domain = 1; }\nnode a: t.i@1 { times = 40000; previous <- s.result; }\nnode b: t.i@1 { times = 40000; previous <- s.result; }\noutput result <- b.result;").unwrap_err();
        assert!(together.message.contains("places more than"), "{together}");
    }

    #[test]
    fn placements_nested_in_placements_are_counted_together() {
        // Each document places the one before it twice: 2^18 nodes from
        // nineteen small files.
        let mut files: BTreeMap<PathBuf, String> = BTreeMap::new();
        files.insert(
            PathBuf::from("g0.esn"),
            "input x: Rows<Int>;\nnode n: t.i@1 { previous <- x; }\noutput result <- n.result;\n"
                .into(),
        );
        for k in 1..=18 {
            files.insert(
                PathBuf::from(format!("g{k}.esn")),
                format!(
                    "input x: Rows<Int>;\nnode a: graph \"g{j}.esn\" {{ x <- x; }}\nnode b: graph \"g{j}.esn\" {{ x <- a.result; }}\noutput result <- b.result;\n",
                    j = k - 1
                ),
            );
        }
        let text = "node s: t.z@1 { domain = 1; }\nnode top: graph \"g18.esn\" { x <- s.result; }\noutput result <- top.result;\n";
        let document = entelechy_language::parse_document(text).unwrap();
        let mut loader = |path: &Path| {
            files
                .get(path)
                .map(|text| (path.to_path_buf(), text.clone()))
                .ok_or_else(|| format!("{} is missing", path.display()))
        };
        let refused = load(&document, text, "scene", Path::new(""), &mut loader).unwrap_err();
        assert!(refused.message.contains("places more than"), "{refused}");
    }

    /// The Graph computes structure, never values: a node's values and
    /// `times` are written literals or params. Arithmetic belongs in a
    /// Kernel.
    #[test]
    fn a_scene_writes_values_and_never_computes_them() {
        let graph = "param n: Int = 2;\nnode s: t.z@1 { domain = 1; }\n";
        for member in [
            "gain = 0.5 * 2.0;",
            "gain = n + 1;",
            "gain = -n;",
            "gain = vec2(n, 1);",
            "times = n + 1;",
            "times = 2 * 2;",
        ] {
            let text = format!(
                "{graph}node r: t.i@1 {{ {member} previous <- s.result; }}\noutput result <- r.result;"
            );
            let error = parse(&text).unwrap_err();
            assert!(
                error.message.contains("literal"),
                "`{member}` was accepted or refused for another reason: {error}"
            );
        }
    }
}

/// A graph written as a scene document. One that places no graph lowers
/// back to the same graph; a placed graph's nodes are named by their path
/// (`title.advances`), which shows the plan and does not read back. A
/// file input is written as a caller-supplied image.
pub fn print(graph: &Graph) -> String {
    Printed(graph).to_string()
}

/// A graph as [`print()`] writes it.
struct Printed<'a>(&'a Graph);

impl std::fmt::Display for Printed<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "// {}", self.0.name)?;
        for external in &self.0.externals {
            match self.0.texts.iter().find(|(name, _)| name == external) {
                Some((_, text)) => writeln!(
                    f,
                    "input {external}: Rows<Int> = text {};",
                    entelechy_language::quoted(text)
                )?,
                None => writeln!(f, "input {external}: Image;")?,
            }
        }
        for param in &self.0.params {
            writeln!(
                f,
                "param {}: {} = {};",
                param.name,
                param.ty,
                literal_text(&param.default)
            )?;
        }
        for requirement in &self.0.requirements {
            writeln!(f, "require {requirement};")?;
        }
        for node in &self.0.nodes {
            writeln!(f)?;
            writeln!(f, "node {}: {} {{", node.id, node.kernel)?;
            for (port, binding) in &node.inputs {
                writeln!(f, "    {port} {binding};")?;
            }
            for (name, value) in &node.params {
                match value {
                    ParamValue::Literal(value) => {
                        writeln!(f, "    {name} = {};", literal_text(value))?
                    }
                    ParamValue::Graph(param) => writeln!(f, "    {name} = {param};")?,
                }
            }
            if let Some(domain) = &node.domain {
                writeln!(f, "    domain = {domain};")?;
            }
            if node.is_async {
                writeln!(f, "    async = true;")?;
            }
            for awake in &node.awake {
                writeln!(f, "    awake = value({}, {});", awake.source, awake.row)?;
            }
            writeln!(f, "}}")?;
        }
        writeln!(f)?;
        for (name, source) in &self.0.outputs {
            writeln!(f, "output {name} <- {source};")?;
        }
        Ok(())
    }
}

/// Renders a value in the literal grammar shared with `.ent`.
pub fn literal_text(value: &Value) -> String {
    // A scene number reads back as written, without `d`.
    match value {
        Value::Double(v) => format!("{v:?}"),
        Value::Long(v) => v.to_string(),
        Value::Record(members) => format!(
            "[{}]",
            members
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod enum_tests {
    use super::*;

    /// A scene writes an enum's variant where it writes a number: a node's
    /// param and a param of its own, read as the variant's `Int`.
    #[test]
    fn a_scene_names_an_enums_variants() {
        let graph = parse(
            "type Mode = enum { Idle, Walk = 5, Run };
param opening: Mode = Mode.Run;
node a: t.move@1 { domain = 1; mode = Mode.Walk; }
node b: t.move@1 { domain = 1; mode = opening; }
output result <- a.result;
",
        )
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(graph.params[0].default, Value::Int(6));
        assert_eq!(graph.params[0].ty, "Int");
        assert_eq!(
            graph.nodes[0].params["mode"],
            ParamValue::Literal(Value::Int(5))
        );
        assert_eq!(
            graph.nodes[1].params["mode"],
            ParamValue::Graph("opening".into())
        );
        let error = parse(
            "type Mode = enum { Idle, Walk };\nnode a: t.move@1 { domain = 1; mode = Mode.Wlak; }\noutput result <- a.result;\n",
        )
        .unwrap_err();
        assert!(error.message.contains("did you mean `Walk`"), "{error}");
        // Said once, where the variant is written.
        assert!(!error.message.contains("2:"), "{error}");
    }
}
