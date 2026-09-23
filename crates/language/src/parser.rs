//! Recursive-descent parser producing the AST in the crate root.

use crate::lexer::{Lexeme, Token, tokenize};
use crate::{
    BinaryOp, BindingDecl, Block, CarryDecl, Document, Expr, ExprKind, FnDecl, FramesDecl, Kernel,
    MatchArm, MergeDecl, NodeDecl, NodeKind, NodeMember, PackageDecl, ParamDecl, ParseError,
    PortDecl, PrefixDecl, ScatterDecl, SourceRef, Span, Spanned, StructDecl, TypeAlias, Use,
};

/// Parses a document: `use`, `kernel`, `input`, `output`, and `node` items.
pub fn parse_document(source: &str) -> Result<Document, ParseError> {
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        position: 0,
        temporaries: 0,
    };
    let mut document = Document {
        package: None,
        uses: Vec::new(),
        structs: Vec::new(),
        types: Vec::new(),
        functions: Vec::new(),
        kernels: Vec::new(),
        inputs: Vec::new(),
        params: Vec::new(),
        outputs: Vec::new(),
        nodes: Vec::new(),
    };
    loop {
        match parser.peek().clone() {
            Token::Eof => break,
            Token::Ident(word) if word == "use" => {
                parser.next();
                let (token, span) = parser.next();
                let Token::Str(path) = token else {
                    return Err(ParseError {
                        span,
                        message: format!("`use` needs a quoted path, found {}", token.describe()),
                    });
                };
                parser.expect_punct(";")?;
                document.uses.push(Use {
                    path: Spanned { value: path, span },
                });
            }
            Token::Ident(word) if word == "package" => {
                let span = parser.next().1;
                if document.package.is_some() {
                    return Err(ParseError {
                        span,
                        message: "a document declares one package".into(),
                    });
                }
                document.package = Some(parser.package()?);
            }
            Token::Ident(word) if word == "kernel" => {
                document.kernels.push(parser.kernel(false)?);
            }
            Token::Ident(word) if word == "struct" => {
                parser.next();
                document.structs.push(parser.struct_decl()?);
            }
            Token::Ident(word) if word == "type" => {
                parser.next();
                document.types.push(parser.type_alias()?);
            }
            Token::Ident(word) if word == "fn" => {
                parser.next();
                document.functions.push(parser.function()?);
            }
            Token::Ident(word) if word == "input" => {
                parser.next();
                document.inputs.push(parser.port(false)?);
            }
            Token::Ident(word) if word == "param" => {
                parser.next();
                document.params.push(parser.param()?);
            }
            Token::Ident(word) if word == "output" => {
                parser.next();
                let name = parser.ident("an output name")?;
                parser.expect_punct("<-")?;
                let source = parser.source_ref()?;
                parser.expect_punct(";")?;
                document.outputs.push((name, source));
            }
            Token::Ident(word) if word == "node" => {
                parser.next();
                document.nodes.push(parser.node()?);
            }
            _ => {
                return Err(parser.unexpected(
                    "`use`, `struct`, `type`, `fn`, `kernel`, `input`, `param`, `output`, or `node`",
                ));
            }
        }
    }
    // Document structs and aliases come first in every Kernel, so a
    // Rows<Particle> declared once means the same layout in all of them.
    if !document.structs.is_empty() || !document.types.is_empty() || !document.functions.is_empty()
    {
        let shared = document.structs.clone();
        let shared_types = document.types.clone();
        let shared_functions = document.functions.clone();
        let prepend = |kernel: &mut Kernel| {
            let mut structs = shared.clone();
            structs.append(&mut kernel.structs);
            kernel.structs = structs;
            let mut types = shared_types.clone();
            types.append(&mut kernel.types);
            kernel.types = types;
            let mut functions = shared_functions.clone();
            functions.append(&mut kernel.functions);
            kernel.functions = functions;
        };
        for kernel in &mut document.kernels {
            prepend(kernel);
        }
        for node in &mut document.nodes {
            if let NodeKind::Inline(kernel) = &mut node.kind {
                prepend(kernel);
            }
        }
    }
    Ok(document)
}

/// Parses one literal expression, as written in a document value position.
pub fn parse_literal(source: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        position: 0,
        temporaries: 0,
    };
    let expr = parser.expr()?;
    if parser.peek() != &Token::Eof {
        return Err(parser.unexpected("end of the value"));
    }
    Ok(expr)
}

/// Parses exactly one Kernel from source text.
pub fn parse(source: &str) -> Result<Kernel, ParseError> {
    let document = parse_document(source)?;
    let mut kernels = document.kernels;
    if !document.nodes.is_empty()
        || !document.uses.is_empty()
        || !document.inputs.is_empty()
        || !document.params.is_empty()
        || !document.outputs.is_empty()
        || !document.structs.is_empty()
        || !document.types.is_empty()
    {
        return Err(ParseError {
            span: Span { line: 1, column: 1 },
            message: "expected a single kernel, found graph items".into(),
        });
    }
    match kernels.len() {
        1 => Ok(kernels.remove(0)),
        count => Err(ParseError {
            span: Span { line: 1, column: 1 },
            message: format!("expected a single kernel, found {count}"),
        }),
    }
}

struct Parser {
    tokens: Vec<Lexeme>,
    position: usize,
    /// How many values a `let { ... }` has had to name for itself.
    temporaries: usize,
}

impl Parser {
    /// How many values one `match` range may hold: enough for a byte's
    /// worth of opcodes, few enough that the arms stay a table.
    const RANGE_LIMIT: i64 = 256;

    fn peek(&self) -> &Token {
        &self.tokens[self.position].token
    }

    fn span(&self) -> Span {
        self.tokens[self.position].span
    }

    /// Byte offset where the current token starts.
    fn start(&self) -> usize {
        self.tokens[self.position].start
    }

    /// Byte offset where the previous token ended.
    fn previous_end(&self) -> usize {
        self.tokens[self.position.saturating_sub(1)].end
    }

    fn next(&mut self) -> (Token, Span) {
        let lexeme = &self.tokens[self.position];
        let token = (lexeme.token.clone(), lexeme.span);
        if self.position + 1 < self.tokens.len() {
            self.position += 1;
        }
        token
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            span: self.span(),
            message: message.into(),
        }
    }

    fn unexpected(&self, expected: &str) -> ParseError {
        self.error(format!(
            "expected {expected}, found {}",
            self.peek().describe()
        ))
    }

    fn is_punct(&self, punct: &str) -> bool {
        matches!(self.peek(), Token::Punct(p) if *p == punct)
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Token::Ident(name) if name == keyword)
    }

    fn eat_punct(&mut self, punct: &str) -> bool {
        if self.is_punct(punct) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, punct: &str) -> Result<Span, ParseError> {
        if self.is_punct(punct) {
            Ok(self.next().1)
        } else {
            Err(self.unexpected(&format!("`{punct}`")))
        }
    }

    fn ident(&mut self, what: &str) -> Result<Spanned<String>, ParseError> {
        match self.peek() {
            Token::Ident(name) if !RESERVED.contains(&name.as_str()) => {
                let (Token::Ident(value), span) = self.next() else {
                    unreachable!()
                };
                Ok(Spanned { value, span })
            }
            _ => Err(self.unexpected(what)),
        }
    }

    /// A name that may be a keyword, so long as the keyword does not
    /// start an expression; those five would make a call ambiguous.
    fn name_not_starting_an_expression(
        &mut self,
        what: &str,
    ) -> Result<Spanned<String>, ParseError> {
        match self.peek() {
            Token::Ident(name) if STARTS_AN_EXPRESSION.contains(&name.as_str()) => {
                let name = name.clone();
                Err(ParseError {
                    span: self.span(),
                    message: format!(
                        "`{name}` begins an expression, so it cannot be {what}; any other word can, keywords included"
                    ),
                })
            }
            _ => self.raw_ident(what),
        }
    }

    fn keyword(&mut self, keyword: &str) -> Result<Span, ParseError> {
        if self.is_keyword(keyword) {
            Ok(self.next().1)
        } else {
            Err(self.unexpected(&format!("`{keyword}`")))
        }
    }

    /// Any identifier token, reserved words included; dotted names such as
    /// `input.read` are made of these.
    fn raw_ident(&mut self, what: &str) -> Result<Spanned<String>, ParseError> {
        match self.peek() {
            Token::Ident(_) => {
                let (Token::Ident(value), span) = self.next() else {
                    unreachable!()
                };
                Ok(Spanned { value, span })
            }
            _ => Err(self.unexpected(what)),
        }
    }

    fn dotted_name(&mut self, what: &str) -> Result<Spanned<String>, ParseError> {
        let first = self.raw_ident(what)?;
        let mut value = first.value;
        while self.eat_punct(".") {
            value.push('.');
            value.push_str(&self.raw_ident("a name segment")?.value);
        }
        Ok(Spanned {
            value,
            span: first.span,
        })
    }

    /// Parses `kernel [id@version] { ... }`. Inline Kernels (inside a node)
    /// may omit the id, wire their inputs, and set `domain`.
    fn kernel(&mut self, inline: bool) -> Result<Kernel, ParseError> {
        let start = self.start();
        self.keyword("kernel")?;
        let id = if self.is_punct("{") {
            if !inline {
                return Err(self.error(
                    "a kernel outside a node needs an id, for example `kernel image.blur@1`",
                ));
            }
            None
        } else {
            let mut id = self.dotted_name("a kernel name")?;
            if self.eat_punct("@") {
                match self.next() {
                    (Token::Int(version), _) => id.value = format!("{}@{version}", id.value),
                    (token, span) => {
                        return Err(ParseError {
                            span,
                            message: format!(
                                "expected a version number, found {}",
                                token.describe()
                            ),
                        });
                    }
                }
            } else {
                return Err(self.error("a kernel id needs a version, for example `image.blur@1`"));
            }
            Some(id)
        };
        let mut domain = None;
        let mut domain_range = None;
        self.expect_punct("{")?;
        let mut requires = None;
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let mut outputs = Vec::new();
        let mut memories = Vec::new();
        let mut structs = Vec::new();
        let mut types = Vec::new();
        let mut functions = Vec::new();
        let mut compute = None;
        let mut merge = None;
        let mut carry = None;
        let mut scatter = None;
        let mut prefix = None;
        while !self.is_punct("}") {
            let span = self.span();
            let Token::Ident(keyword) = self.peek().clone() else {
                return Err(self.unexpected("a kernel item"));
            };
            match keyword.as_str() {
                "requires" => {
                    if requires.is_some() {
                        return Err(self.error("`requires` is declared twice"));
                    }
                    self.next();
                    requires = Some(self.requires()?);
                }
                "input" => {
                    self.next();
                    inputs.push(self.port(inline)?);
                }
                "output" => {
                    self.next();
                    outputs.push(self.port(false)?);
                }
                "memory" => {
                    self.next();
                    memories.push(self.port(false)?);
                }
                "domain" if inline => {
                    let statement_start = self.start();
                    self.next();
                    self.expect_punct("=")?;
                    let value = self.expr()?;
                    self.expect_punct(";")?;
                    domain_range = Some(statement_start..self.previous_end());
                    if domain.replace(value).is_some() {
                        return Err(self.error("`domain` is set twice"));
                    }
                }
                "param" => {
                    self.next();
                    params.push(self.param()?);
                }
                "fn" => {
                    self.next();
                    functions.push(self.function()?);
                }
                "struct" => {
                    self.next();
                    structs.push(self.struct_decl()?);
                }
                "type" => {
                    self.next();
                    types.push(self.type_alias()?);
                }
                "compute" => {
                    if compute.is_some() {
                        return Err(self.error(
                            if carry.is_some() || scatter.is_some() || prefix.is_some() {
                                "a kernel has one of `compute`, `scan`, `scatter`, or `prefix`"
                            } else {
                                "`compute` is declared twice"
                            },
                        ));
                    }
                    self.next();
                    compute = Some(self.block()?);
                }
                "scatter" => {
                    if compute.is_some() {
                        return Err(self.error(
                            "a kernel has one of `compute`, `scan`, `scatter`, or `prefix`",
                        ));
                    }
                    self.next();
                    self.keyword("into")?;
                    let into = self.ident("an input name")?;
                    self.keyword("with")?;
                    let combine = self.ident("`sum`, `min`, `max`, or `last`")?;
                    compute = Some(self.block()?);
                    scatter = Some(ScatterDecl {
                        into,
                        combine,
                        span,
                    });
                }
                "prefix" => {
                    if compute.is_some() {
                        return Err(self.error(
                            "a kernel has one of `compute`, `scan`, `scatter`, or `prefix`",
                        ));
                    }
                    self.next();
                    self.keyword("with")?;
                    let combine = self.ident("`sum`, `min`, or `max`")?;
                    compute = Some(self.block()?);
                    prefix = Some(PrefixDecl { combine, span });
                }
                "scan" => {
                    if compute.is_some() {
                        return Err(self.error(
                            "a kernel has one of `compute`, `scan`, `scatter`, or `prefix`",
                        ));
                    }
                    self.next();
                    self.keyword("with")?;
                    let name = self.ident("a carried value name")?;
                    self.expect_punct("=")?;
                    let init = self.expr()?;
                    compute = Some(self.block()?);
                    carry = Some(CarryDecl { name, init, span });
                }
                "merge" => {
                    if merge.is_some() {
                        return Err(self.error("`merge` is declared twice"));
                    }
                    self.next();
                    self.expect_punct("(")?;
                    let a = self.ident("a value name")?;
                    self.expect_punct(",")?;
                    let b = self.ident("a value name")?;
                    self.expect_punct(")")?;
                    let body = self.block()?;
                    merge = Some(MergeDecl { a, b, body, span });
                }
                _ => {
                    return Err(ParseError {
                        span,
                        message: format!(
                            "expected `requires`, `input`, `param`, `output`, `memory`, `struct`, `type`, `fn`, `compute`, `scan`, `scatter`, `prefix`, or `merge`, found `{keyword}`"
                        ),
                    });
                }
            }
        }
        let compute = compute.ok_or_else(|| {
            self.error("kernel has no `compute`, `scan`, `scatter`, or `prefix` block")
        })?;
        self.expect_punct("}")?;
        let end = self.previous_end();
        Ok(Kernel {
            range: start..end,
            id,
            domain,
            domain_range,
            requires: requires.unwrap_or_default(),
            inputs,
            params,
            outputs,
            memories,
            structs,
            types,
            functions,
            compute,
            merge,
            carry,
            scatter,
            prefix,
        })
    }

    /// `type Name = Shape<Element>;` after the `type` keyword.
    fn type_alias(&mut self) -> Result<TypeAlias, ParseError> {
        let name = self.ident("a type name")?;
        if !name.value.starts_with(|c: char| c.is_ascii_uppercase()) {
            return Err(ParseError {
                span: name.span,
                message: format!(
                    "type name `{}` must start with an uppercase letter",
                    name.value
                ),
            });
        }
        self.expect_punct("=")?;
        let target = self.port_type()?;
        self.expect_punct(";")?;
        Ok(TypeAlias { name, target })
    }

    /// A port type as written: `Name` or `Name<Element>`; or a value
    /// type, which an array type `[T; N]` always is.
    fn port_type(&mut self) -> Result<Spanned<String>, ParseError> {
        if self.is_punct("[") {
            return self.type_name();
        }
        let mut ty = self.ident("a type")?;
        if self.is_punct("<") {
            self.next();
            let element = self.type_name()?;
            // How the Field keeps the element's scalars, when narrower
            // than a word: `Grid<Vec4 as unorm8>`.
            let kept = if self.is_keyword("as") {
                self.next();
                let encoding = self.ident("an encoding after `as`")?;
                format!(" as {}", encoding.value)
            } else {
                String::new()
            };
            self.expect_punct(">")?;
            ty.value = format!("{}<{}{kept}>", ty.value, element.value);
        }
        Ok(ty)
    }

    /// A value type as written: `Name`, or an array `[Type; N]`, spelled
    /// canonically (`[Int; 16]`) for the compiler to read back.
    fn type_name(&mut self) -> Result<Spanned<String>, ParseError> {
        if !self.is_punct("[") {
            return self.ident("a type");
        }
        let span = self.next().1;
        let element = self.type_name()?;
        self.expect_punct(";")?;
        let count = match self.next() {
            (Token::Int(count), _) if count > 0 => count as u32,
            (token, span) => {
                return Err(ParseError {
                    span,
                    message: format!(
                        "an array type is `[Type; count]` with a positive count, found {}",
                        token.describe()
                    ),
                });
            }
        };
        self.expect_punct("]")?;
        Ok(Spanned {
            value: format!("[{}; {count}]", element.value),
            span,
        })
    }

    fn struct_decl(&mut self) -> Result<StructDecl, ParseError> {
        let name = self.ident("a struct name")?;
        if !name.value.starts_with(|c: char| c.is_ascii_uppercase()) {
            return Err(ParseError {
                span: name.span,
                message: format!(
                    "struct names start with an uppercase letter; `{}` does not",
                    name.value
                ),
            });
        }
        self.expect_punct("{")?;
        let mut fields = Vec::new();
        while !self.is_punct("}") {
            let field = self.ident("a field name")?;
            self.expect_punct(":")?;
            let ty = self.type_name()?;
            self.expect_punct(";")?;
            fields.push((field, ty));
        }
        self.expect_punct("}")?;
        Ok(StructDecl { name, fields })
    }

    /// `Name { ..base, field: expr, ... }` after the name was consumed.
    fn struct_literal(&mut self, name: Spanned<String>) -> Result<Expr, ParseError> {
        let span = name.span;
        self.expect_punct("{")?;
        let mut base = None;
        if self.is_punct("..") {
            self.next();
            base = Some(Box::new(self.expr()?));
            if !self.eat_punct(",") && !self.is_punct("}") {
                return Err(self.unexpected("`,` or `}` after `..base`"));
            }
        }
        let mut fields = Vec::new();
        while !self.is_punct("}") {
            let field = self.ident("a field name")?;
            self.expect_punct(":")?;
            let value = self.expr()?;
            fields.push((field, value));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct("}")?;
        Ok(Expr {
            kind: ExprKind::Struct { name, base, fields },
            span,
        })
    }

    /// `name: kernel [id] { ... }` or `name: id@version { members }` after
    /// the `node` keyword.
    fn node(&mut self) -> Result<NodeDecl, ParseError> {
        // Any word names a node, a keyword too: the name follows `node`
        // and precedes `:`, so `scan` or `input` is not mistaken there.
        let name = self.raw_ident("a node name")?;
        self.expect_punct(":")?;
        if self.is_keyword("kernel") {
            let kernel = self.kernel(true)?;
            return Ok(NodeDecl {
                name,
                kind: NodeKind::Inline(Box::new(kernel)),
                members: Vec::new(),
            });
        }
        // A Field of a type, filled: the type name is a capital word, and
        // a kernel id is not.
        if let Token::Ident(word) = self.peek()
            && word.starts_with(|c: char| c.is_ascii_uppercase())
            && matches!(
                self.tokens.get(self.position + 1).map(|l| &l.token),
                Some(Token::Punct("{") | Token::Punct("<"))
            )
        {
            let ty = self.port_type()?;
            let members = self.node_members()?;
            return Ok(NodeDecl {
                name,
                kind: NodeKind::Fill(ty),
                members,
            });
        }
        if self.is_keyword("graph") {
            self.next();
            let (token, span) = self.next();
            let Token::Str(path) = token else {
                return Err(ParseError {
                    span,
                    message: format!("`graph` needs a quoted path, found {}", token.describe()),
                });
            };
            let members = self.node_members()?;
            return Ok(NodeDecl {
                name,
                kind: NodeKind::SubGraph(Spanned { value: path, span }),
                members,
            });
        }
        let mut kernel = self.dotted_name("a kernel id")?;
        self.expect_punct("@")?;
        match self.next() {
            (Token::Int(version), _) => kernel.value = format!("{}@{version}", kernel.value),
            (token, span) => {
                return Err(ParseError {
                    span,
                    message: format!("expected a version number, found {}", token.describe()),
                });
            }
        }
        let members = self.node_members()?;
        Ok(NodeDecl {
            name,
            kind: NodeKind::Reference(kernel),
            members,
        })
    }

    fn node_members(&mut self) -> Result<Vec<NodeMember>, ParseError> {
        self.expect_punct("{")?;
        let mut members = Vec::new();
        while !self.is_punct("}") {
            let port = self.ident("a parameter or port name")?;
            if port.value == "carry"
                && !(self.is_punct("=") || self.is_punct("<-") || self.is_punct("<~"))
            {
                let carried = self.ident("a port to carry")?;
                self.expect_punct("<-")?;
                let output = self.ident("an output to carry from")?;
                self.expect_punct(";")?;
                members.push(NodeMember::Carry(carried, output));
                continue;
            }
            if self.eat_punct("=") {
                let value = self.expr()?;
                self.expect_punct(";")?;
                members.push(NodeMember::Set(port, value));
            } else if self.eat_punct("<-") {
                let source = self.source_ref()?;
                self.expect_punct(";")?;
                members.push(NodeMember::Bind(port, source));
            } else if self.eat_punct("<~") {
                let source = self.source_ref()?;
                self.keyword("from")?;
                let initial = self.source_ref()?;
                let migration = self.migration()?;
                self.expect_punct(";")?;
                members.push(NodeMember::Previous {
                    port,
                    source,
                    initial,
                    migration,
                });
            } else {
                return Err(self.unexpected("`=`, `<-`, or `<~`"));
            }
        }
        self.expect_punct("}")?;
        Ok(members)
    }

    /// `name@version { language = N; ir = N; }` after `package`: the
    /// versions are the least the package needs, each named once.
    fn package(&mut self) -> Result<PackageDecl, ParseError> {
        let mut id = self.dotted_name("a package name")?;
        self.expect_punct("@")?;
        match self.next() {
            (Token::Int(version), _) => id.value = format!("{}@{version}", id.value),
            (token, span) => {
                return Err(ParseError {
                    span,
                    message: format!("expected a version number, found {}", token.describe()),
                });
            }
        }
        self.expect_punct("{")?;
        let (mut language, mut ir) = (None, None);
        while !self.is_punct("}") {
            let key = self.raw_ident("`language` or `ir`")?;
            self.expect_punct("=")?;
            let (token, span) = self.next();
            let Token::Int(version) = token else {
                return Err(ParseError {
                    span,
                    message: format!(
                        "`{}` needs a version number, found {}",
                        key.value,
                        token.describe()
                    ),
                });
            };
            self.expect_punct(";")?;
            let slot = match key.value.as_str() {
                "language" => &mut language,
                "ir" => &mut ir,
                other => {
                    return Err(ParseError {
                        span: key.span,
                        message: format!("a package needs `language` and `ir`, not `{other}`"),
                    });
                }
            };
            if slot.is_some() {
                return Err(ParseError {
                    span: key.span,
                    message: format!("`{}` is given twice", key.value),
                });
            }
            *slot = Some(version as u32);
        }
        self.expect_punct("}")?;
        let (Some(language), Some(ir)) = (language, ir) else {
            return Err(self.error("a package names the `language` and `ir` versions it needs"));
        };
        Ok(PackageDecl { id, language, ir })
    }

    /// An optional `migrate kernel.id@version` after a `<~` binding.
    fn migration(&mut self) -> Result<Option<Spanned<String>>, ParseError> {
        if !self.is_keyword("migrate") {
            return Ok(None);
        }
        self.next();
        let mut kernel = self.dotted_name("a migration kernel id")?;
        self.expect_punct("@")?;
        match self.next() {
            (Token::Int(version), _) => kernel.value = format!("{}@{version}", kernel.value),
            (token, span) => {
                return Err(ParseError {
                    span,
                    message: format!("expected a version number, found {}", token.describe()),
                });
            }
        }
        Ok(Some(kernel))
    }

    fn source_ref(&mut self) -> Result<SourceRef, ParseError> {
        // A source follows `<-`, `<~` or `from`, so a node named by a
        // keyword is read here as it was declared.
        let first = self.raw_ident("a source")?;
        if self.eat_punct(".") {
            let port = self.raw_ident("an output port")?;
            Ok(SourceRef {
                node: Some(first.value),
                name: port.value,
                span: first.span,
            })
        } else {
            Ok(SourceRef {
                node: None,
                name: first.value,
                span: first.span,
            })
        }
    }

    fn requires(&mut self) -> Result<Vec<Spanned<String>>, ParseError> {
        self.expect_punct("{")?;
        let mut names = Vec::new();
        while !self.is_punct("}") {
            names.push(self.dotted_name("a capability name")?);
            self.expect_punct(";")?;
        }
        self.expect_punct("}")?;
        Ok(names)
    }

    /// `name: Type;`, or with `wired`, optionally `name: Type <- source;` or
    /// `name: Type <~ source from initial;`.
    fn port(&mut self, wired: bool) -> Result<PortDecl, ParseError> {
        let name = self.ident("a port name")?;
        self.expect_punct(":")?;
        let ty = self.port_type()?;
        let binding_start = self.start();
        let binding = if wired && self.eat_punct("<-") {
            Some(BindingDecl::Now(self.source_ref()?))
        } else if wired && self.eat_punct("<~") {
            let source = self.source_ref()?;
            self.keyword("from")?;
            let initial = self.source_ref()?;
            let migration = self.migration()?;
            Some(BindingDecl::Previous {
                source,
                initial,
                migration,
            })
        } else {
            None
        };
        let binding_range = binding
            .is_some()
            .then(|| binding_start..self.previous_end());
        let mut file = None;
        let mut text = None;
        let mut frames = None;
        let mut bytes = None;
        if !wired && self.eat_punct("=") {
            let as_text = self.is_keyword("text");
            let as_frames = self.is_keyword("frames");
            let as_bytes = self.is_keyword("bytes");
            if as_text || as_frames || as_bytes {
                self.next();
            }
            let (token, span) = self.next();
            let Token::Str(value) = token else {
                return Err(ParseError {
                    span,
                    message: format!(
                        "an input's value is a quoted path, `text` and a quoted string, `frames`, a quoted directory and `at` a rate, or `bytes` and a quoted path, found {}",
                        token.describe()
                    ),
                });
            };
            if as_frames {
                self.keyword("at")?;
                let rate = match self.next() {
                    (Token::Int(rate), _) if rate > 0 => rate as u32,
                    (token, span) => {
                        return Err(ParseError {
                            span,
                            message: format!(
                                "`frames` are read `at` a whole number of frames a second, found {}",
                                token.describe()
                            ),
                        });
                    }
                };
                frames = Some(FramesDecl {
                    dir: Spanned { value, span },
                    rate,
                });
            } else if as_text {
                text = Some(Spanned { value, span });
            } else if as_bytes {
                bytes = Some(Spanned { value, span });
            } else {
                file = Some(Spanned { value, span });
            }
        }
        self.expect_punct(";")?;
        Ok(PortDecl {
            name,
            ty,
            binding,
            file,
            text,
            frames,
            bytes,
            binding_range,
        })
    }

    fn param(&mut self) -> Result<ParamDecl, ParseError> {
        let name = self.ident("a parameter name")?;
        self.expect_punct(":")?;
        let ty = self.type_name()?;
        self.expect_punct("=")?;
        let default_start = self.start();
        let default = self.expr()?;
        let default_range = default_start..self.previous_end();
        self.expect_punct(";")?;
        Ok(ParamDecl {
            name,
            ty,
            default,
            default_range,
        })
    }

    fn function(&mut self) -> Result<FnDecl, ParseError> {
        // Any word names a function, a keyword too, as long as it is not
        // one that starts an expression: a call is a word and a `(`, so
        // `scatter(n, seed)` reads as a call wherever it stands.
        let name = self.name_not_starting_an_expression("a function name")?;
        // Type parameters: names that stand for whatever type a call
        // gives them, one instance of the function lowered for each.
        let mut generics = Vec::new();
        if self.eat_punct("<") {
            while !self.is_punct(">") {
                generics.push(self.ident("a type parameter")?);
                if !self.eat_punct(",") {
                    break;
                }
            }
            self.expect_punct(">")?;
            if generics.is_empty() {
                return Err(self.error("a `<>` names at least one type parameter"));
            }
        }
        self.expect_punct("(")?;
        let mut args = Vec::new();
        while !self.is_punct(")") {
            let arg = self.ident("an argument name")?;
            self.expect_punct(":")?;
            // A value type, or a port type for a Field parameter.
            let ty = self.port_type()?;
            args.push((arg, ty));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct(")")?;
        self.expect_punct("->")?;
        let ret = self.type_name()?;
        let body = self.block()?;
        Ok(FnDecl {
            name,
            generics,
            args,
            ret,
            body,
        })
    }

    /// `{ let ...; return expr; }` or `{ let ...; expr }`.
    /// A `match` arm's body: a block, or a bare expression.
    fn arm_body(&mut self) -> Result<Block, ParseError> {
        if self.is_punct("{") {
            self.block()
        } else {
            Ok(Block {
                lets: Vec::new(),
                result: self.expr()?,
            })
        }
    }

    fn block(&mut self) -> Result<Block, ParseError> {
        // A block that is a record of outputs, `{ color: c, depth: d }`,
        // gives that record: a statement never begins with `name:`.
        if self.at_record() {
            let result = self.record_literal()?;
            return Ok(Block {
                lets: Vec::new(),
                result,
            });
        }
        self.expect_punct("{")?;
        self.block_body()
    }

    /// Whether a `{` here opens a record of outputs: `{ name:`.
    fn at_record(&self) -> bool {
        let token = |offset: usize| self.tokens.get(self.position + offset).map(|l| &l.token);
        matches!(token(0), Some(Token::Punct("{")))
            && matches!(token(1), Some(Token::Ident(name)) if !RESERVED.contains(&name.as_str()))
            && matches!(token(2), Some(Token::Punct(":")))
    }

    /// `{ name: expr, ... }`, a Kernel's outputs by name.
    fn record_literal(&mut self) -> Result<Expr, ParseError> {
        let span = self.expect_punct("{")?;
        let mut fields = Vec::new();
        while !self.is_punct("}") {
            let field = self.ident("an output name")?;
            self.expect_punct(":")?;
            let value = self.expr()?;
            fields.push((field, value));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct("}")?;
        Ok(Expr {
            kind: ExprKind::Record(fields),
            span,
        })
    }

    /// A block's statements, to its closing brace. An `if` with no
    /// `else` among them guards what follows: what it gives is the
    /// block's value, and the statements after it are the other arm.
    /// One `match` arm value, or a range of them: `3`, `-1`, `0..8`
    /// (up to but not including the last), `0..=7` (including it). A
    /// range is the values it holds, written out, so the arms stay a
    /// table and a value listed twice is still caught.
    fn arm_values(&mut self, values: &mut Vec<Spanned<i32>>) -> Result<(), ParseError> {
        let (first, span) = self.arm_value()?;
        let inclusive = self.is_punct("..=");
        if !inclusive && !self.is_punct("..") {
            values.push(Spanned { value: first, span });
            return Ok(());
        }
        self.next();
        let (end, end_span) = self.arm_value()?;
        let last = if inclusive { end } else { end - 1 };
        let count = i64::from(last) - i64::from(first) + 1;
        if count < 1 {
            return Err(ParseError {
                span,
                message: format!(
                    "`{first}..{}{end}` holds no values; a range runs upwards",
                    if inclusive { "=" } else { "" }
                ),
            });
        }
        if count > Self::RANGE_LIMIT {
            return Err(ParseError {
                span: end_span,
                message: format!(
                    "a `match` range holds at most {} values, this one holds {count}; \
                     an `if` on the bounds is the form for a wider one",
                    Self::RANGE_LIMIT
                ),
            });
        }
        for value in first..=last {
            values.push(Spanned { value, span });
        }
        Ok(())
    }

    /// An integer as a `match` arm writes it, `7` or `-1`.
    fn arm_value(&mut self) -> Result<(i32, Span), ParseError> {
        let negative = self.eat_punct("-");
        match self.next() {
            (Token::Int(value), span) => Ok((
                if negative {
                    value.wrapping_neg()
                } else {
                    value
                },
                span,
            )),
            (token, span) => Err(ParseError {
                span,
                message: format!(
                    "a `match` arm takes integer values, a range of them, or `_`, found {}",
                    token.describe()
                ),
            }),
        }
    }

    /// `let { x, y } = value;`, and `let { x: left, y: top } = value;`
    /// where the binding is to have another name: the value read once,
    /// then one binding a field, which is what one writes by hand.
    fn destructure(&mut self, lets: &mut Vec<(Spanned<String>, Expr)>) -> Result<(), ParseError> {
        let span = self.expect_punct("{")?;
        let mut fields: Vec<(Spanned<String>, Spanned<String>)> = Vec::new();
        while !self.is_punct("}") {
            let field = self.ident("a field name")?;
            let binding = if self.eat_punct(":") {
                self.ident("a binding name")?
            } else {
                field.clone()
            };
            fields.push((field, binding));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct("}")?;
        if fields.is_empty() {
            return Err(ParseError {
                span,
                message: "a `let { }` names at least one field".into(),
            });
        }
        self.expect_punct("=")?;
        let value = self.expr()?;
        self.expect_punct(";")?;
        // A name is already read once; anything else is bound first, so
        // the fields come from one value.
        let source = match &value.kind {
            ExprKind::Name(name) => Spanned {
                value: name.clone(),
                span: value.span,
            },
            _ => {
                let name = Spanned {
                    value: format!("${}", self.temporaries),
                    span: value.span,
                };
                self.temporaries += 1;
                lets.push((name.clone(), value));
                name
            }
        };
        for (field, binding) in fields {
            let read = Expr {
                kind: ExprKind::Member(
                    Box::new(Expr {
                        kind: ExprKind::Name(source.value.clone()),
                        span: source.span,
                    }),
                    field.value,
                ),
                span: field.span,
            };
            lets.push((binding, read));
        }
        Ok(())
    }

    fn block_body(&mut self) -> Result<Block, ParseError> {
        let mut lets = Vec::new();
        loop {
            if self.is_keyword("if") {
                let saved = self.position;
                let span = self.span();
                self.next();
                let condition = self.expr()?;
                let then = self.block()?;
                if self.is_keyword("else") {
                    // An `if` with both arms is an expression; read it
                    // again as one.
                    self.position = saved;
                } else {
                    let otherwise = self.block_body()?;
                    return Ok(Block {
                        lets,
                        result: Expr {
                            kind: ExprKind::If {
                                condition: Box::new(condition),
                                then: Box::new(then),
                                otherwise: Box::new(otherwise),
                            },
                            span,
                        },
                    });
                }
            }
            if self.is_keyword("let") {
                self.next();
                if self.is_punct("{") {
                    self.destructure(&mut lets)?;
                } else {
                    let name = self.ident("a binding name")?;
                    self.expect_punct("=")?;
                    let value = self.expr()?;
                    self.expect_punct(";")?;
                    lets.push((name, value));
                }
            } else if self.is_keyword("return") {
                self.next();
                let result = self.expr()?;
                self.expect_punct(";")?;
                self.expect_punct("}")?;
                return Ok(Block { lets, result });
            } else if self.is_keyword("break") {
                // `break expr;` ends its block, as `return` does; the
                // loop around it stops and gives that value.
                let span = self.span();
                self.next();
                let value = self.expr()?;
                self.expect_punct(";")?;
                self.expect_punct("}")?;
                return Ok(Block {
                    lets,
                    result: Expr {
                        kind: ExprKind::Break(Box::new(value)),
                        span,
                    },
                });
            } else if self.is_punct("}") {
                return Err(self.error("a block must end with `return expr;` or a value"));
            } else {
                let result = self.expr()?;
                // An expression followed by `;` is a statement for its effect
                // (a `put`), bound to `_`; without one it is the block's value.
                if self.eat_punct(";") {
                    let span = result.span;
                    lets.push((
                        Spanned {
                            value: "_".to_owned(),
                            span,
                        },
                        result,
                    ));
                    continue;
                }
                self.expect_punct("}")?;
                return Ok(Block { lets, result });
            }
        }
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.or()
    }

    fn or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.and()?;
        while self.is_punct("||") {
            self.next();
            let right = self.and()?;
            left = binary(BinaryOp::Or, left, right);
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.comparison()?;
        while self.is_punct("&&") {
            self.next();
            let right = self.comparison()?;
            left = binary(BinaryOp::And, left, right);
        }
        Ok(left)
    }

    fn comparison(&mut self) -> Result<Expr, ParseError> {
        let left = self.bit_or()?;
        let op = match self.peek() {
            Token::Punct("<") => BinaryOp::Less,
            Token::Punct("<=") => BinaryOp::LessEqual,
            Token::Punct(">") => BinaryOp::Greater,
            Token::Punct(">=") => BinaryOp::GreaterEqual,
            Token::Punct("==") => BinaryOp::Equal,
            Token::Punct("!=") => BinaryOp::NotEqual,
            _ => return Ok(left),
        };
        self.next();
        let right = self.bit_or()?;
        let result = binary(op, left, right);
        if matches!(
            self.peek(),
            Token::Punct("<" | "<=" | ">" | ">=" | "==" | "!=")
        ) {
            return Err(self.error("comparisons do not chain; parenthesize or use `&&`"));
        }
        Ok(result)
    }

    /// The integer bit operators bind between comparison and arithmetic,
    /// loosest first: `|`, then `^`, then `&`, then the shifts.
    fn bit_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.bit_xor()?;
        while self.is_punct("|") {
            self.next();
            let right = self.bit_xor()?;
            left = binary(BinaryOp::BitOr, left, right);
        }
        Ok(left)
    }

    fn bit_xor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.bit_and()?;
        while self.is_punct("^") {
            self.next();
            let right = self.bit_and()?;
            left = binary(BinaryOp::BitXor, left, right);
        }
        Ok(left)
    }

    fn bit_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.shift()?;
        while self.is_punct("&") {
            self.next();
            let right = self.shift()?;
            left = binary(BinaryOp::BitAnd, left, right);
        }
        Ok(left)
    }

    fn shift(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.additive()?;
        loop {
            let op = if self.is_punct("<<") {
                BinaryOp::Shl
            } else if self.is_punct(">>>") {
                BinaryOp::Ushr
            } else if self.is_punct(">>") {
                BinaryOp::Shr
            } else {
                return Ok(left);
            };
            self.next();
            let right = self.additive()?;
            left = binary(op, left, right);
        }
    }

    fn additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.multiplicative()?;
        loop {
            let op = if self.is_punct("+") {
                BinaryOp::Add
            } else if self.is_punct("-") {
                BinaryOp::Sub
            } else {
                return Ok(left);
            };
            self.next();
            let right = self.multiplicative()?;
            left = binary(op, left, right);
        }
    }

    fn multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.unary()?;
        loop {
            let op = if self.is_punct("*") {
                BinaryOp::Mul
            } else if self.is_punct("/") {
                BinaryOp::Div
            } else if self.is_punct("%") {
                BinaryOp::Rem
            } else {
                return Ok(left);
            };
            self.next();
            let right = self.unary()?;
            left = binary(op, left, right);
        }
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.is_punct("-") {
            let span = self.next().1;
            let inner = self.unary()?;
            return Ok(Expr {
                kind: ExprKind::Neg(Box::new(inner)),
                span,
            });
        }
        if self.is_punct("!") {
            let span = self.next().1;
            let inner = self.unary()?;
            return Ok(Expr {
                kind: ExprKind::Not(Box::new(inner)),
                span,
            });
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.primary()?;
        loop {
            if self.eat_punct(".") {
                let member = self.ident("a component name")?;
                expr = Expr {
                    span: expr.span,
                    kind: ExprKind::Member(Box::new(expr), member.value),
                };
            } else if self.eat_punct("[") {
                let index = self.expr()?;
                self.expect_punct("]")?;
                expr = Expr {
                    span: expr.span,
                    kind: ExprKind::Subscript(Box::new(expr), Box::new(index)),
                };
            } else {
                return Ok(expr);
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.span();
        match self.peek().clone() {
            Token::Float(value) => {
                self.next();
                Ok(Expr {
                    kind: ExprKind::Float(value),
                    span,
                })
            }
            Token::Int(value) => {
                self.next();
                Ok(Expr {
                    kind: ExprKind::Int(value),
                    span,
                })
            }
            Token::Ident(name) if name == "true" || name == "false" => {
                self.next();
                Ok(Expr {
                    kind: ExprKind::Bool(name == "true"),
                    span,
                })
            }
            Token::Ident(name) if name == "if" => {
                self.next();
                let condition = self.expr()?;
                let then = self.block()?;
                self.keyword("else")?;
                let otherwise = if self.is_keyword("if") {
                    let nested = self.primary()?;
                    Block {
                        lets: Vec::new(),
                        result: nested,
                    }
                } else {
                    self.block()?
                };
                Ok(Expr {
                    kind: ExprKind::If {
                        condition: Box::new(condition),
                        then: Box::new(then),
                        otherwise: Box::new(otherwise),
                    },
                    span,
                })
            }
            Token::Ident(name) if name == "match" => {
                self.next();
                let scrutinee = self.expr()?;
                self.expect_punct("{")?;
                let mut arms = Vec::new();
                let otherwise = loop {
                    if self.is_keyword("_") {
                        self.next();
                        self.expect_punct("=>")?;
                        let body = self.arm_body()?;
                        self.eat_punct(",");
                        self.expect_punct("}")?;
                        break body;
                    }
                    let mut values = Vec::new();
                    loop {
                        self.arm_values(&mut values)?;
                        if !self.eat_punct("|") {
                            break;
                        }
                    }
                    self.expect_punct("=>")?;
                    let body = self.arm_body()?;
                    arms.push(MatchArm { values, body });
                    if !self.eat_punct(",") && !self.is_punct("}") {
                        return Err(self.unexpected("`,` between `match` arms"));
                    }
                    if self.is_punct("}") {
                        return Err(self.error("a `match` ends with a `_ => ...` arm"));
                    }
                };
                Ok(Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(scrutinee),
                        arms,
                        otherwise: Box::new(otherwise),
                    },
                    span,
                })
            }
            Token::Ident(name) if name == "loop" => {
                self.next();
                let count = self.expr()?;
                if let ExprKind::Int(literal) = &count.kind
                    && *literal <= 0
                {
                    return Err(ParseError {
                        span: count.span,
                        message: format!("`loop` needs a positive integer bound, found {literal}"),
                    });
                }
                self.keyword("with")?;
                let name = self.ident("a carried value name")?;
                self.expect_punct("=")?;
                let init = self.expr()?;
                let until = if self.is_keyword("until") {
                    self.next();
                    Some(Box::new(self.expr()?))
                } else {
                    None
                };
                let body = self.block()?;
                Ok(Expr {
                    kind: ExprKind::Loop {
                        count: Box::new(count),
                        name,
                        init: Box::new(init),
                        until,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            // A keyword that does not start an expression names a
            // function the document declared: `scatter(n, seed)`.
            Token::Ident(name)
                if RESERVED.contains(&name.as_str())
                    && !STARTS_AN_EXPRESSION.contains(&name.as_str())
                    && matches!(
                        self.tokens.get(self.position + 1).map(|l| &l.token),
                        Some(Token::Punct("("))
                    ) =>
            {
                let (Token::Ident(name), span) = self.next() else {
                    unreachable!()
                };
                self.expect_punct("(")?;
                let mut args = Vec::new();
                while !self.is_punct(")") {
                    args.push(self.expr()?);
                    if !self.eat_punct(",") {
                        break;
                    }
                }
                self.expect_punct(")")?;
                Ok(Expr {
                    kind: ExprKind::Call(name, args),
                    span,
                })
            }
            Token::Ident(name) if RESERVED.contains(&name.as_str()) => {
                Err(self.unexpected("an expression"))
            }
            Token::Ident(name)
                if name.starts_with(|c: char| c.is_ascii_uppercase())
                    && matches!(
                        self.tokens.get(self.position + 1).map(|l| &l.token),
                        Some(Token::Punct("{"))
                    ) =>
            {
                let (_, span) = self.next();
                self.struct_literal(Spanned { value: name, span })
            }
            Token::Ident(name) => {
                self.next();
                if self.eat_punct("(") {
                    let mut args = Vec::new();
                    while !self.is_punct(")") {
                        args.push(self.expr()?);
                        if !self.eat_punct(",") {
                            break;
                        }
                    }
                    self.expect_punct(")")?;
                    Ok(Expr {
                        kind: ExprKind::Call(name, args),
                        span,
                    })
                } else {
                    Ok(Expr {
                        kind: ExprKind::Name(name),
                        span,
                    })
                }
            }
            Token::Punct("{") if self.at_record() => self.record_literal(),
            Token::Punct("(") => {
                self.next();
                let inner = self.expr()?;
                self.expect_punct(")")?;
                Ok(inner)
            }
            Token::Punct("[") => {
                self.next();
                if self.is_punct("]") {
                    return Err(self.error("an array literal has at least one element"));
                }
                if self.is_keyword("each")
                    && matches!(
                        self.tokens.get(self.position + 1).map(|l| &l.token),
                        Some(Token::Int(_))
                    )
                {
                    self.next();
                    let count = match self.next() {
                        (Token::Int(count), count_span) if count > 0 => Spanned {
                            value: count as u32,
                            span: count_span,
                        },
                        (token, count_span) => {
                            return Err(ParseError {
                                span: count_span,
                                message: format!(
                                    "`[each count {{ ... }}]` needs a positive integer count, found {}",
                                    token.describe()
                                ),
                            });
                        }
                    };
                    let body = self.block()?;
                    self.expect_punct("]")?;
                    return Ok(Expr {
                        kind: ExprKind::Each(count, Box::new(body)),
                        span,
                    });
                }
                let first = self.expr()?;
                if self.eat_punct(";") {
                    let count = match self.next() {
                        (Token::Int(count), count_span) if count > 0 => Spanned {
                            value: count as u32,
                            span: count_span,
                        },
                        (token, count_span) => {
                            return Err(ParseError {
                                span: count_span,
                                message: format!(
                                    "`[value; count]` needs a positive integer count, found {}",
                                    token.describe()
                                ),
                            });
                        }
                    };
                    self.expect_punct("]")?;
                    return Ok(Expr {
                        kind: ExprKind::Repeat(Box::new(first), count),
                        span,
                    });
                }
                let mut elements = vec![first];
                while self.eat_punct(",") {
                    if self.is_punct("]") {
                        break;
                    }
                    elements.push(self.expr()?);
                }
                self.expect_punct("]")?;
                Ok(Expr {
                    kind: ExprKind::Array(elements),
                    span,
                })
            }
            _ => Err(self.unexpected("an expression")),
        }
    }
}

/// Words that cannot name values.
/// The keywords an expression may begin with: no other word may name a
/// function, since a call would not be told from one of these.
const STARTS_AN_EXPRESSION: [&str; 5] = ["let", "return", "if", "match", "loop"];

const RESERVED: [&str; 25] = [
    "let", "return", "if", "else", "match", "loop", "with", "until", "break", "kernel", "struct",
    "type", "node", "use", "from", "input", "output", "memory", "graph", "merge", "scan",
    "scatter", "prefix", "migrate", "package",
];

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr {
        span: left.span,
        kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAYSCALE: &str = r#"
kernel image.grayscale@1 {
    requires { field.read; field.write; }
    input source: Image;
    param amount: Float = 1.0;
    output result: Image;
    // luminance helper
    fn luma(c: Vec4) -> Float {
        return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
    }
    compute {
        let c = sample(source);
        let g = luma(c);
        return mix(c, vec4(g, g, g, c.a), amount);
    }
}
"#;

    #[test]
    fn parses_the_grayscale_kernel() {
        let kernel = parse(GRAYSCALE).unwrap();
        assert_eq!(kernel.id.as_ref().unwrap().value, "image.grayscale@1");
        assert_eq!(
            kernel
                .requires
                .iter()
                .map(|r| r.value.as_str())
                .collect::<Vec<_>>(),
            ["field.read", "field.write"]
        );
        assert_eq!(kernel.inputs[0].name.value, "source");
        assert_eq!(kernel.params[0].ty.value, "Float");
        assert_eq!(kernel.outputs[0].ty.value, "Image");
        assert_eq!(kernel.functions[0].name.value, "luma");
        assert_eq!(kernel.functions[0].args[0].1.value, "Vec4");
        assert_eq!(kernel.compute.lets.len(), 2);
        let ExprKind::Call(name, args) = &kernel.compute.result.kind else {
            panic!("expected a call");
        };
        assert_eq!(name, "mix");
        assert_eq!(args.len(), 3);
    }

    #[test]
    fn bit_operators_bind_between_comparison_and_arithmetic() {
        let kernel = parse("kernel t.k@1 { compute { return a & b == c << 1 + d | e; } }").unwrap();
        let ExprKind::Binary(BinaryOp::Equal, left, right) = &kernel.compute.result.kind else {
            panic!("expected `==` at the top");
        };
        assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::BitAnd, ..)));
        let ExprKind::Binary(BinaryOp::BitOr, shifted, _) = &right.kind else {
            panic!("expected `|` below `==`");
        };
        let ExprKind::Binary(BinaryOp::Shl, _, count) = &shifted.kind else {
            panic!("expected `<<` below `|`");
        };
        assert!(matches!(count.kind, ExprKind::Binary(BinaryOp::Add, ..)));
    }

    #[test]
    fn precedence_and_unary_operators() {
        let kernel = parse(
            "kernel t.k@1 { compute { return -a.x * 2.0 + b / (c - 1) < 3.0 && !d || e == f; } }",
        )
        .unwrap();
        let ExprKind::Binary(BinaryOp::Or, left, right) = &kernel.compute.result.kind else {
            panic!("expected `||` at the top");
        };
        assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Equal, ..)));
        let ExprKind::Binary(BinaryOp::And, compare, not) = &left.kind else {
            panic!("expected `&&` below `||`");
        };
        assert!(matches!(not.kind, ExprKind::Not(_)));
        let ExprKind::Binary(BinaryOp::Less, sum, _) = &compare.kind else {
            panic!("expected `<` below `&&`");
        };
        let ExprKind::Binary(BinaryOp::Add, product, quotient) = &sum.kind else {
            panic!("expected `+` below `<`");
        };
        assert!(matches!(product.kind, ExprKind::Binary(BinaryOp::Mul, ..)));
        assert!(matches!(quotient.kind, ExprKind::Binary(BinaryOp::Div, ..)));
        let ExprKind::Binary(_, negated, _) = &product.kind else {
            unreachable!()
        };
        assert!(matches!(negated.kind, ExprKind::Neg(_)));
    }

    #[test]
    fn match_arms_take_values_blocks_and_a_last_underscore() {
        let kernel = parse(
            "kernel t.k@1 { compute {
                let k = match op & 15 { 0 => 1, 1 | -2 => { let x = 2; x }, _ => 0, };
                return k;
            } }",
        )
        .unwrap();
        let ExprKind::Match {
            arms, otherwise, ..
        } = &kernel.compute.lets[0].1.kind
        else {
            panic!("expected a match");
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(
            arms[1].values.iter().map(|v| v.value).collect::<Vec<_>>(),
            [1, -2]
        );
        assert_eq!(arms[1].body.lets.len(), 1);
        assert!(otherwise.lets.is_empty());
        let error =
            parse("kernel t.k@1 { compute { return match 1 { 0 => 1, 2 => 3 }; } }").unwrap_err();
        assert!(error.message.contains("`_ => ...`"), "{}", error.message);
        let error =
            parse("kernel t.k@1 { compute { return match 1 { x => 1, _ => 3 }; } }").unwrap_err();
        assert!(
            error.message.contains("integer values"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_match_arm_takes_a_range_of_values() {
        let kernel = parse(
            "kernel t.k@1 { compute { return match element() { 0..4 => 1, 4..=6 | 9 => 2, -2..0 => 3, _ => 0 }; } }",
        )
        .unwrap();
        let ExprKind::Match { arms, .. } = &kernel.compute.result.kind else {
            panic!("expected a match");
        };
        let values = |arm: &MatchArm| arm.values.iter().map(|v| v.value).collect::<Vec<_>>();
        assert_eq!(values(&arms[0]), [0, 1, 2, 3], "`..` stops before its end");
        assert_eq!(values(&arms[1]), [4, 5, 6, 9], "`..=` takes its end");
        assert_eq!(values(&arms[2]), [-2, -1], "a range may be negative");

        let error = parse("kernel t.k@1 { compute { return match 1 { 4..2 => 1, _ => 0 }; } }")
            .unwrap_err();
        assert!(error.message.contains("holds no values"), "{error}");
        let error = parse("kernel t.k@1 { compute { return match 1 { 0..1000 => 1, _ => 0 }; } }")
            .unwrap_err();
        assert!(error.message.contains("at most 256 values"), "{error}");
        // The arms stay a table, so a value written twice is still one
        // the compiler refuses; here the parser has already spread it.
        let kernel =
            parse("kernel t.k@1 { compute { return match 1 { 0..4 => 1, 3 => 2, _ => 0 }; } }")
                .unwrap();
        let ExprKind::Match { arms, .. } = &kernel.compute.result.kind else {
            panic!("expected a match");
        };
        assert_eq!(values(&arms[1]), [3]);
    }

    #[test]
    fn a_let_may_name_the_fields_it_takes() {
        let kernel = parse(
            "kernel t.k@1 { compute { let { x, y: top } = at(rows, 0); let { r, g, b } = colour; return vec4(r, g, b, x + top); } }",
        )
        .unwrap();
        let lets = &kernel.compute.lets;
        // The read is bound once, under a name no one can write, and
        // each field comes off it; a name is read again as it stands.
        assert_eq!(lets[0].0.value, "$0");
        assert!(matches!(lets[0].1.kind, ExprKind::Call(ref name, _) if name == "at"));
        let member = |index: usize| match &lets[index].1.kind {
            ExprKind::Member(value, field) => {
                let ExprKind::Name(name) = &value.kind else {
                    panic!("expected a name");
                };
                (name.clone(), field.clone())
            }
            other => panic!("expected a member, found {other:?}"),
        };
        assert_eq!(lets[1].0.value, "x");
        assert_eq!(member(1), ("$0".to_owned(), "x".to_owned()));
        assert_eq!(lets[2].0.value, "top");
        assert_eq!(member(2), ("$0".to_owned(), "y".to_owned()));
        assert_eq!(member(3), ("colour".to_owned(), "r".to_owned()));
        assert_eq!(lets.len(), 6, "three fields and one value, then three");

        let error = parse("kernel t.k@1 { compute { let { } = c; return 1.0; } }").unwrap_err();
        assert!(
            error.message.contains("names at least one field"),
            "{error}"
        );
    }

    #[test]
    fn a_break_ends_its_block_and_a_fn_may_take_type_parameters() {
        let kernel = parse(
            "kernel t.k@1 { fn twice<T, U>(a: T, b: U) -> T { return a; } compute { return loop 8 with a = 0 { if a > 2 { break a; } a + 1 }; } }",
        )
        .unwrap();
        let generics = &kernel.functions[0].generics;
        assert_eq!(generics.len(), 2);
        assert_eq!(generics[0].value, "T");
        assert_eq!(generics[1].value, "U");
        let ExprKind::Loop { body, .. } = &kernel.compute.result.kind else {
            panic!("expected a loop");
        };
        // The guard makes the `break` the block's result on one side and
        // the rest of the block on the other.
        let ExprKind::If { then, .. } = &body.result.kind else {
            panic!("expected a guard");
        };
        assert!(matches!(then.result.kind, ExprKind::Break(_)));

        let error = parse("kernel t.k@1 { compute { let break = 1; return break; } }").unwrap_err();
        assert!(error.message.contains("a binding name"), "{error}");
        let error =
            parse("kernel t.k@1 { fn f<>(a: Int) -> Int { return a; } compute { return f(1); } }")
                .unwrap_err();
        assert!(
            error.message.contains("at least one type parameter"),
            "{}",
            error.message
        );
    }

    #[test]
    fn if_and_loop_are_expressions() {
        let kernel = parse(
            "kernel t.k@1 { compute {
                let m = loop 64 with s = vec2(0.0, 1.0) until s.y < 0.001 {
                    let d = sdf(s.x);
                    vec2(s.x + d, d)
                };
                let c = if m.y < 0.001 { 1.0 } else if m.x > 9.0 { 0.5 } else { let z = 0.0; z };
                return vec4(c, c, c, 1.0);
            } }",
        )
        .unwrap();
        let ExprKind::Loop {
            count,
            name,
            until,
            body,
            ..
        } = &kernel.compute.lets[0].1.kind
        else {
            panic!("expected a loop");
        };
        assert!(matches!(count.kind, ExprKind::Int(64)));
        assert_eq!(name.value, "s");
        assert!(until.is_some());
        assert_eq!(body.lets.len(), 1);
        let ExprKind::If { otherwise, .. } = &kernel.compute.lets[1].1.kind else {
            panic!("expected an if");
        };
        assert!(matches!(otherwise.result.kind, ExprKind::If { .. }));

        let error = parse("kernel t.k@1 { compute { return if a { 1.0 }; } }").unwrap_err();
        assert!(error.message.contains("`else`"), "{error}");
        let error = parse("kernel t.k@1 { compute { return a < b < c; } }").unwrap_err();
        assert!(error.message.contains("do not chain"), "{error}");
        let error =
            parse("kernel t.k@1 { compute { return loop 0 with x = 1.0 { x }; } }").unwrap_err();
        assert!(error.message.contains("positive integer bound"), "{error}");
        // An `if` with no `else` among a block's statements guards what
        // follows it: what it gives is the block's value.
        let kernel = parse(
            "kernel t.k@1 { fn classify(x: Int) -> Int { if x < 0 { return -1; } let doubled = x * 2; if doubled > 9 { return 9; } return doubled; } compute { return classify(element()); } }",
        )
        .unwrap();
        let ExprKind::If { otherwise, .. } = &kernel.functions[0].body.result.kind else {
            panic!("a guard is an `if` over the rest of the block");
        };
        assert_eq!(
            otherwise.lets.len(),
            1,
            "the rest of the block is its other arm"
        );
        assert!(matches!(otherwise.result.kind, ExprKind::If { .. }));
        // An `if` with both arms is still an expression.
        let kernel = parse(
            "kernel t.k@1 { compute { let a = if element() > 0 { 1 } else { 2 }; return a; } }",
        )
        .unwrap();
        assert!(matches!(kernel.compute.lets[0].1.kind, ExprKind::If { .. }));
        // A keyword names a function, as long as it does not begin an
        // expression: a call is a word and a `(`.
        let kernel = parse(
            "kernel t.k@1 { fn scatter(a: Float) -> Float { return a * 2.0; } fn merge(a: Float) -> Float { return a + 1.0; } compute { return scatter(merge(1.0)); } }",
        )
        .unwrap();
        assert_eq!(kernel.functions[0].name.value, "scatter");
        assert_eq!(kernel.functions[1].name.value, "merge");
        let error = parse(
            "kernel t.k@1 { fn loop(a: Float) -> Float { return a; } compute { return 0.0; } }",
        )
        .unwrap_err();
        assert!(error.message.contains("begins an expression"), "{error}");
        // A count that is not a literal is an expression, read once.
        let kernel =
            parse("kernel t.k@1 { param n: Int = 4; compute { return loop n * 2 with x = 1.0 { x + 1.0 }; } }")
                .unwrap();
        let ExprKind::Loop { count, .. } = &kernel.compute.result.kind else {
            panic!("expected a loop");
        };
        assert!(matches!(count.kind, ExprKind::Binary(..)));
    }

    #[test]
    fn errors_carry_positions() {
        let error = parse("kernel image.blur { }").unwrap_err();
        assert_eq!(
            error.span,
            Span {
                line: 1,
                column: 19
            }
        );
        assert!(error.message.contains("version"), "{error}");

        let error = parse("kernel t.k@1 {\n  input source Image;\n}").unwrap_err();
        assert_eq!(
            error.span,
            Span {
                line: 2,
                column: 16
            }
        );
        assert_eq!(error.message, "expected `:`, found `Image`");

        let error = parse("kernel t.k@1 { input a: Image; }").unwrap_err();
        assert!(error.message.contains("no `compute`"), "{error}");

        let error = parse("kernel t.k@1 { compute { let x = 1 $ 2; return x; } }").unwrap_err();
        assert_eq!(error.message, "unexpected character `$`");
    }
}

#[cfg(test)]
mod document_tests {
    use super::*;

    #[test]
    fn a_package_manifest_names_the_package_and_the_versions_it_needs() {
        let document =
            parse_document("package nes_fx@2 { language = 1; ir = 1; }\nuse \"kernels/\";")
                .unwrap();
        let package = document.package.unwrap();
        assert_eq!(package.id.value, "nes_fx@2");
        assert_eq!((package.language, package.ir), (1, 1));
        assert_eq!(document.uses.len(), 1);
        let error = parse_document("package nes_fx@1 { language = 1; }").unwrap_err();
        assert!(error.message.contains("`ir`"), "{error}");
        let error =
            parse_document("package nes_fx@1 { language = 1; ir = 1; colour = 2; }").unwrap_err();
        assert!(error.message.contains("not `colour`"), "{error}");
        let error = parse_document("package nes_fx { language = 1; ir = 1; }").unwrap_err();
        assert!(error.message.contains("`@`"), "{error}");
    }

    #[test]
    fn a_node_may_take_a_keyword_for_its_name() {
        let document = parse_document(
            "node scan: kernel { requires { field.write; } output result: Rows<Int>; domain = 4; compute { return 0; } }
             node input: kernel { requires { field.read; field.write; } input previous: Rows<Int> <- scan.result; output result: Rows<Int>; compute { return sample(previous); } }
             output result <- input.result;",
        )
        .unwrap();
        let names: Vec<&str> = document
            .nodes
            .iter()
            .map(|n| n.name.value.as_str())
            .collect();
        assert_eq!(names, vec!["scan", "input"]);
        let error = parse_document("node scan kernel { }").unwrap_err();
        assert!(error.message.contains("`:`"), "{error}");
    }

    #[test]
    fn document_structs_are_shared_by_every_kernel() {
        let document = parse_document(
            "struct P { x: Float; }
             kernel a.k@1 { requires { field.write; } output r: Rows<P>; compute { return P { x: 1.0 }; } }
             node b: kernel { requires { field.read; field.write; } input p: Rows<P> <- a.r; output r: Rows<P>; struct Q { y: Int; } compute { return sample(p); } }
             output r <- b.r;",
        )
        .unwrap();
        assert_eq!(document.structs.len(), 1);
        assert_eq!(document.kernels[0].structs[0].name.value, "P");
        let NodeKind::Inline(kernel) = &document.nodes[0].kind else {
            panic!("inline");
        };
        assert_eq!(
            kernel
                .structs
                .iter()
                .map(|s| s.name.value.as_str())
                .collect::<Vec<_>>(),
            ["P", "Q"]
        );
    }

    #[test]
    fn document_functions_are_shared_and_may_take_fields() {
        let document = parse_document(
            "fn twice(x: Float) -> Float { return x * 2.0; }
             fn peek(f: Grid<Float>, c: Vec2) -> Float { return at(f, c); }
             kernel a.k@1 { requires { field.write; } output r: Rows<Float>; fn own(x: Float) -> Float { return x; } compute { return twice(1.0); } }",
        )
        .unwrap();
        assert_eq!(document.functions.len(), 2);
        assert_eq!(document.functions[1].args[0].1.value, "Grid<Float>");
        assert_eq!(
            document.kernels[0]
                .functions
                .iter()
                .map(|f| f.name.value.as_str())
                .collect::<Vec<_>>(),
            ["twice", "peek", "own"]
        );
    }
}
