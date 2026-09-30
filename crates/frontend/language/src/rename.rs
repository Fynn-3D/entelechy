//! A function's calls renamed: how a document's functions keep calling
//! the functions they see when another document in the same reach
//! declares the same name.

use crate::{Block, Expr, ExprKind, FnDecl, MatchArm, Spanned};

impl FnDecl {
    /// This function named `name`, each call in its body to a name
    /// `rename` maps called by the new name instead. The body is shared
    /// when nothing in it changes.
    pub fn renamed(&self, name: String, rename: &dyn Fn(&str) -> Option<String>) -> FnDecl {
        let body = if block_calls(&self.body, rename) {
            std::sync::Arc::new(block(&self.body, rename))
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
}

/// Whether a block calls any name `rename` maps.
fn block_calls(block: &Block, rename: &dyn Fn(&str) -> Option<String>) -> bool {
    let mut found = false;
    visit_block(block, &mut |name| found |= rename(name).is_some());
    found
}

fn visit_block(block: &Block, seen: &mut dyn FnMut(&str)) {
    for (_, value) in &block.lets {
        visit(value, seen);
    }
    visit(&block.result, seen);
}

fn visit(expr: &Expr, seen: &mut dyn FnMut(&str)) {
    match &expr.kind {
        ExprKind::Float(_)
        | ExprKind::Int(_)
        | ExprKind::Double(_)
        | ExprKind::Long(_)
        | ExprKind::Bool(_)
        | ExprKind::Name(_) => {}
        ExprKind::Call(name, args) => {
            seen(name);
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
        ExprKind::Struct { base, fields, .. } => {
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
    }
}

fn block(block_: &Block, rename: &dyn Fn(&str) -> Option<String>) -> Block {
    Block {
        lets: block_
            .lets
            .iter()
            .map(|(name, value)| (name.clone(), expr(value, rename)))
            .collect(),
        result: expr(&block_.result, rename),
    }
}

fn boxed(inner: &Expr, rename: &dyn Fn(&str) -> Option<String>) -> Box<Expr> {
    Box::new(expr(inner, rename))
}

fn expr(e: &Expr, rename: &dyn Fn(&str) -> Option<String>) -> Expr {
    let kind = match &e.kind {
        ExprKind::Call(name, args) => ExprKind::Call(
            rename(name).unwrap_or_else(|| name.clone()),
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
            name: name.clone(),
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
        ExprKind::Float(_)
        | ExprKind::Int(_)
        | ExprKind::Double(_)
        | ExprKind::Long(_)
        | ExprKind::Bool(_)
        | ExprKind::Name(_) => e.kind.clone(),
    };
    Expr { kind, span: e.span }
}
