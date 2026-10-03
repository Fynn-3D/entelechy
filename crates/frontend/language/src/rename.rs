//! Names renamed: how a document's functions keep calling the functions
//! they see, and naming the structs and types they see, when another
//! document in the same reach declares the same name.

use crate::{Block, Expr, ExprKind, FnDecl, MatchArm, Spanned};

/// What a rename maps: a function's name when it is called, or a struct's
/// or type's name wherever a type is named.
#[derive(Clone, Copy)]
struct Renames<'a> {
    call: &'a dyn Fn(&str) -> Option<String>,
    ty: &'a dyn Fn(&str) -> Option<String>,
}

fn keep(_: &str) -> Option<String> {
    None
}

impl FnDecl {
    /// This function named `name`, each call in its body to a name
    /// `rename` maps called by the new name instead. The body is shared
    /// when nothing in it changes.
    pub fn renamed(&self, name: String, rename: &dyn Fn(&str) -> Option<String>) -> FnDecl {
        let renames = Renames {
            call: rename,
            ty: &keep,
        };
        let body = if block_names(&self.body, renames) {
            std::sync::Arc::new(block(&self.body, renames))
        } else {
            self.body.clone()
        };
        FnDecl {
            name: Spanned {
                value: name,
                span: self.name.span,
            },
            body,
            ..self.clone()
        }
    }

    /// This function with each struct or type name `rename` maps, in its
    /// signature and its body (a struct built, an enum's variant), named
    /// by the new name instead. The body is shared when nothing in it
    /// changes.
    pub fn retyped(&self, rename: &dyn Fn(&str) -> Option<String>) -> FnDecl {
        let renames = Renames {
            call: &keep,
            ty: rename,
        };
        let body = if block_names(&self.body, renames) {
            std::sync::Arc::new(block(&self.body, renames))
        } else {
            self.body.clone()
        };
        let retype = |ty: &Spanned<String>| Spanned {
            value: renamed_type(&ty.value, rename),
            span: ty.span,
        };
        FnDecl {
            args: self
                .args
                .iter()
                .map(|(name, ty)| (name.clone(), retype(ty)))
                .collect(),
            ret: retype(&self.ret),
            body,
            ..self.clone()
        }
    }
}

/// A type as written, `Rows<Ray>` or `[Ray; 4]`, with each name in it
/// that `rename` maps written by the new name instead.
pub fn renamed_type(ty: &str, rename: &dyn Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(ty.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if !word.is_empty() {
            match rename(word) {
                Some(new) => out.push_str(&new),
                None => out.push_str(word),
            }
            word.clear();
        }
    };
    for c in ty.chars() {
        if c.is_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// Whether a block names anything `renames` maps: a call, a struct built,
/// or an enum named.
fn block_names(block: &Block, renames: Renames<'_>) -> bool {
    let mut found = false;
    visit_block(block, &mut |name, called| {
        let rename = if called { renames.call } else { renames.ty };
        found |= rename(name).is_some();
    });
    found
}

fn visit_block(block: &Block, seen: &mut dyn FnMut(&str, bool)) {
    for (_, value) in &block.lets {
        visit(value, seen);
    }
    visit(&block.result, seen);
}

fn visit(expr: &Expr, seen: &mut dyn FnMut(&str, bool)) {
    match &expr.kind {
        ExprKind::Float(_)
        | ExprKind::Int(_)
        | ExprKind::Double(_)
        | ExprKind::Long(_)
        | ExprKind::Bool(_) => {}
        ExprKind::Name(name) => seen(name, false),
        ExprKind::Call(name, args) => {
            seen(name, true);
            args.iter().for_each(|arg| visit(arg, seen));
        }
        ExprKind::Member(inner, _)
        | ExprKind::Repeat(inner, _)
        | ExprKind::Neg(inner)
        | ExprKind::Not(inner)
        | ExprKind::Break(inner) => visit(inner, seen),
        ExprKind::Array(items) => items.iter().for_each(|item| visit(item, seen)),
        ExprKind::Each(_, body) => visit_block(body, seen),
        ExprKind::Subscript(array, index) | ExprKind::Binary(_, array, index) => {
            visit(array, seen);
            visit(index, seen);
        }
        ExprKind::Struct { name, base, fields } => {
            seen(&name.value, false);
            if let Some(base) = base {
                visit(base, seen);
            }
            fields.iter().for_each(|(_, value)| visit(value, seen));
        }
        ExprKind::Record(fields) => fields.iter().for_each(|(_, value)| visit(value, seen)),
        ExprKind::If {
            condition,
            then,
            otherwise,
        } => {
            visit(condition, seen);
            visit_block(then, seen);
            visit_block(otherwise, seen);
        }
        ExprKind::Match {
            scrutinee,
            arms,
            otherwise,
        } => {
            visit(scrutinee, seen);
            arms.iter().for_each(|arm| visit_block(&arm.body, seen));
            if let Some(otherwise) = otherwise {
                visit_block(otherwise, seen);
            }
        }
        ExprKind::Loop {
            count,
            init,
            until,
            body,
            ..
        } => {
            visit(count, seen);
            visit(init, seen);
            if let Some(until) = until {
                visit(until, seen);
            }
            visit_block(body, seen);
        }
        ExprKind::Pick {
            count,
            condition,
            key,
            ..
        } => {
            visit(count, seen);
            visit(condition, seen);
            if let Some(key) = key {
                visit(key, seen);
            }
        }
    }
}

fn block(block_: &Block, rename: Renames<'_>) -> Block {
    Block {
        lets: block_
            .lets
            .iter()
            .map(|(name, value)| (name.clone(), expr(value, rename)))
            .collect(),
        result: expr(&block_.result, rename),
    }
}

fn boxed(inner: &Expr, rename: Renames<'_>) -> Box<Expr> {
    Box::new(expr(inner, rename))
}

fn expr(e: &Expr, rename: Renames<'_>) -> Expr {
    let kind = match &e.kind {
        ExprKind::Call(name, args) => ExprKind::Call(
            (rename.call)(name).unwrap_or_else(|| name.clone()),
            args.iter().map(|arg| expr(arg, rename)).collect(),
        ),
        ExprKind::Member(inner, field) => ExprKind::Member(boxed(inner, rename), field.clone()),
        ExprKind::Repeat(inner, n) => ExprKind::Repeat(boxed(inner, rename), n.clone()),
        ExprKind::Neg(inner) => ExprKind::Neg(boxed(inner, rename)),
        ExprKind::Not(inner) => ExprKind::Not(boxed(inner, rename)),
        ExprKind::Break(inner) => ExprKind::Break(boxed(inner, rename)),
        ExprKind::Array(items) => ExprKind::Array(items.iter().map(|i| expr(i, rename)).collect()),
        ExprKind::Each(n, body) => ExprKind::Each(n.clone(), Box::new(block(body, rename))),
        ExprKind::Subscript(array, index) => {
            ExprKind::Subscript(boxed(array, rename), boxed(index, rename))
        }
        ExprKind::Binary(op, a, b) => ExprKind::Binary(*op, boxed(a, rename), boxed(b, rename)),
        ExprKind::Struct { name, base, fields } => ExprKind::Struct {
            name: Spanned {
                value: (rename.ty)(&name.value).unwrap_or_else(|| name.value.clone()),
                span: name.span,
            },
            base: base.as_ref().map(|base| boxed(base, rename)),
            fields: fields
                .iter()
                .map(|(field, value)| (field.clone(), expr(value, rename)))
                .collect(),
        },
        ExprKind::Record(fields) => ExprKind::Record(
            fields
                .iter()
                .map(|(field, value)| (field.clone(), expr(value, rename)))
                .collect(),
        ),
        ExprKind::If {
            condition,
            then,
            otherwise,
        } => ExprKind::If {
            condition: boxed(condition, rename),
            then: Box::new(block(then, rename)),
            otherwise: Box::new(block(otherwise, rename)),
        },
        ExprKind::Match {
            scrutinee,
            arms,
            otherwise,
        } => ExprKind::Match {
            scrutinee: boxed(scrutinee, rename),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    body: block(&arm.body, rename),
                    ..arm.clone()
                })
                .collect(),
            otherwise: otherwise
                .as_ref()
                .map(|otherwise| Box::new(block(otherwise, rename))),
        },
        ExprKind::Loop {
            count,
            name,
            init,
            until,
            body,
        } => ExprKind::Loop {
            count: boxed(count, rename),
            name: name.clone(),
            init: boxed(init, rename),
            until: until.as_ref().map(|until| boxed(until, rename)),
            body: Box::new(block(body, rename)),
        },
        ExprKind::Pick {
            name,
            count,
            condition,
            key,
        } => ExprKind::Pick {
            name: name.clone(),
            count: boxed(count, rename),
            condition: boxed(condition, rename),
            key: key.as_ref().map(|key| boxed(key, rename)),
        },
        ExprKind::Float(_)
        | ExprKind::Int(_)
        | ExprKind::Double(_)
        | ExprKind::Long(_)
        | ExprKind::Bool(_) => e.kind.clone(),
        // A name a rename maps is an enum's, as in `Mat.Air`.
        ExprKind::Name(name) => ExprKind::Name((rename.ty)(name).unwrap_or_else(|| name.clone())),
    };
    Expr { kind, span: e.span }
}
