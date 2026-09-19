//! Recursive-descent parser producing the AST in the crate root.

use crate::lexer::{Token, tokenize};
use crate::{
    BinaryOp, Block, Expr, ExprKind, FnDecl, Kernel, ParamDecl, ParseError, PortDecl, Span,
    Spanned, StructDecl,
};

/// Parses one Kernel from source text.
pub fn parse(source: &str) -> Result<Kernel, ParseError> {
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        position: 0,
    };
    let kernel = parser.kernel()?;
    parser.expect_punct("}")?;
    if parser.peek() != &Token::Eof {
        return Err(parser.error("expected end of file after the kernel"));
    }
    Ok(kernel)
}

struct Parser {
    tokens: Vec<(Token, Span)>,
    position: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.position].0
    }

    fn span(&self) -> Span {
        self.tokens[self.position].1
    }

    fn next(&mut self) -> (Token, Span) {
        let token = self.tokens[self.position].clone();
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

    fn keyword(&mut self, keyword: &str) -> Result<Span, ParseError> {
        if self.is_keyword(keyword) {
            Ok(self.next().1)
        } else {
            Err(self.unexpected(&format!("`{keyword}`")))
        }
    }

    fn dotted_name(&mut self, what: &str) -> Result<Spanned<String>, ParseError> {
        let first = self.ident(what)?;
        let mut value = first.value;
        while self.eat_punct(".") {
            value.push('.');
            value.push_str(&self.ident("a name segment")?.value);
        }
        Ok(Spanned {
            value,
            span: first.span,
        })
    }

    fn kernel(&mut self) -> Result<Kernel, ParseError> {
        self.keyword("kernel")?;
        let mut id = self.dotted_name("a kernel name")?;
        if self.eat_punct("@") {
            match self.next() {
                (Token::Int(version), _) => id.value = format!("{}@{version}", id.value),
                (token, span) => {
                    return Err(ParseError {
                        span,
                        message: format!("expected a version number, found {}", token.describe()),
                    });
                }
            }
        } else {
            return Err(self.error("a kernel id needs a version, for example `image.blur@1`"));
        }
        self.expect_punct("{")?;
        let mut requires = None;
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let mut outputs = Vec::new();
        let mut structs = Vec::new();
        let mut functions = Vec::new();
        let mut compute = None;
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
                    inputs.push(self.port()?);
                }
                "output" => {
                    self.next();
                    outputs.push(self.port()?);
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
                "compute" => {
                    if compute.is_some() {
                        return Err(self.error("`compute` is declared twice"));
                    }
                    self.next();
                    compute = Some(self.block()?);
                }
                _ => {
                    return Err(ParseError {
                        span,
                        message: format!(
                            "expected `requires`, `input`, `param`, `output`, `struct`, `fn`, or `compute`, found `{keyword}`"
                        ),
                    });
                }
            }
        }
        let compute = compute.ok_or_else(|| self.error("kernel has no `compute` block"))?;
        Ok(Kernel {
            id,
            requires: requires.unwrap_or_default(),
            inputs,
            params,
            outputs,
            structs,
            functions,
            compute,
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
            let ty = self.ident("a type")?;
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
        if self.is_punct(".") {
            self.next();
            self.expect_punct(".")?;
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

    fn port(&mut self) -> Result<PortDecl, ParseError> {
        let name = self.ident("a port name")?;
        self.expect_punct(":")?;
        let ty = self.ident("a type")?;
        self.expect_punct(";")?;
        Ok(PortDecl { name, ty })
    }

    fn param(&mut self) -> Result<ParamDecl, ParseError> {
        let name = self.ident("a parameter name")?;
        self.expect_punct(":")?;
        let ty = self.ident("a type")?;
        self.expect_punct("=")?;
        let default = self.expr()?;
        self.expect_punct(";")?;
        Ok(ParamDecl { name, ty, default })
    }

    fn function(&mut self) -> Result<FnDecl, ParseError> {
        let name = self.ident("a function name")?;
        self.expect_punct("(")?;
        let mut args = Vec::new();
        while !self.is_punct(")") {
            let arg = self.ident("an argument name")?;
            self.expect_punct(":")?;
            let ty = self.ident("a type")?;
            args.push((arg, ty));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct(")")?;
        self.expect_punct("->")?;
        let ret = self.ident("a return type")?;
        let body = self.block()?;
        Ok(FnDecl {
            name,
            args,
            ret,
            body,
        })
    }

    /// `{ let ...; return expr; }` or `{ let ...; expr }`.
    fn block(&mut self) -> Result<Block, ParseError> {
        self.expect_punct("{")?;
        let mut lets = Vec::new();
        loop {
            if self.is_keyword("let") {
                self.next();
                let name = self.ident("a binding name")?;
                self.expect_punct("=")?;
                let value = self.expr()?;
                self.expect_punct(";")?;
                lets.push((name, value));
            } else if self.is_keyword("return") {
                self.next();
                let result = self.expr()?;
                self.expect_punct(";")?;
                self.expect_punct("}")?;
                return Ok(Block { lets, result });
            } else if self.is_punct("}") {
                return Err(self.error("a block must end with `return expr;` or a value"));
            } else {
                let result = self.expr()?;
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
        let left = self.additive()?;
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
        let right = self.additive()?;
        let result = binary(op, left, right);
        if matches!(
            self.peek(),
            Token::Punct("<" | "<=" | ">" | ">=" | "==" | "!=")
        ) {
            return Err(self.error("comparisons do not chain; parenthesize or use `&&`"));
        }
        Ok(result)
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
        while self.eat_punct(".") {
            let member = self.ident("a component name")?;
            expr = Expr {
                span: expr.span,
                kind: ExprKind::Member(Box::new(expr), member.value),
            };
        }
        Ok(expr)
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
            Token::Ident(name) if name == "loop" => {
                self.next();
                let count = match self.next() {
                    (Token::Int(count), span) if count > 0 => Spanned {
                        value: count as u32,
                        span,
                    },
                    (token, span) => {
                        return Err(ParseError {
                            span,
                            message: format!(
                                "`loop` needs a positive integer bound, found {}",
                                token.describe()
                            ),
                        });
                    }
                };
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
                        count,
                        name,
                        init: Box::new(init),
                        until,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            Token::Ident(name) if RESERVED.contains(&name.as_str()) => {
                Err(self.unexpected("an expression"))
            }
            Token::Ident(name)
                if name.starts_with(|c: char| c.is_ascii_uppercase())
                    && matches!(
                        self.tokens.get(self.position + 1),
                        Some((Token::Punct("{"), _))
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
            Token::Punct("(") => {
                self.next();
                let inner = self.expr()?;
                self.expect_punct(")")?;
                Ok(inner)
            }
            _ => Err(self.unexpected("an expression")),
        }
    }
}

/// Words that cannot name values.
const RESERVED: [&str; 9] = [
    "let", "return", "if", "else", "loop", "with", "until", "kernel", "struct",
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
    requires { image.read; image.write; }
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
        assert_eq!(kernel.id.value, "image.grayscale@1");
        assert_eq!(
            kernel
                .requires
                .iter()
                .map(|r| r.value.as_str())
                .collect::<Vec<_>>(),
            ["image.read", "image.write"]
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
        assert_eq!(count.value, 64);
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
