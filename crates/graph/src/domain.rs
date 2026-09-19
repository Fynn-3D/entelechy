//! A node's `domain` as written: a literal extent, or an expression over
//! the node's own inputs and parameters, resolved when the node binds.
//!
//! The expression language is deliberately small: numbers, `vec2` and
//! `vec3`, `+ - * /`, negation, `count(port)` and `size(port)` over the
//! node's input ports, `max`, `min`, `floor`, `ceil`, and names that are
//! the node's parameters or the graph's. It says how big an output is in
//! terms of what comes in, so a Kernel that changes topology (four
//! triangles per triangle, a map at sixteen pixels per tile) can size its
//! output without the graph repeating the arithmetic by hand.

use std::fmt;

use orla_abi::{Domain, Value};
use orla_language::{BinaryOp, Expr, ExprKind, Span};

/// How a node's output extent is given.
#[derive(Clone, Debug, PartialEq)]
pub enum DomainSpec {
    /// An extent written as a literal.
    Literal(Domain),
    /// An extent computed from the node's inputs and parameters.
    Expr(DomainExpr),
}

impl DomainSpec {
    /// The literal extent, when there is nothing to resolve.
    pub fn literal(&self) -> Option<Domain> {
        match self {
            Self::Literal(domain) => Some(*domain),
            Self::Expr(_) => None,
        }
    }
}

impl fmt::Display for DomainSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(domain) => write!(f, "{domain}"),
            Self::Expr(expr) => write!(f, "{expr}"),
        }
    }
}

/// An arithmetic operator of a domain expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

impl DomainOp {
    fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
        }
    }
}

/// A domain expression.
#[derive(Clone, Debug, PartialEq)]
pub enum DomainExpr {
    /// An integer literal.
    Int(i32),
    /// A float literal.
    Float(f32),
    /// A parameter of the node or of the graph.
    Name(String),
    /// `count(port)`: the number of elements the port is bound to.
    Count(String),
    /// `size(port)`: the extent the port is bound to, as `Int`, `Vec2`, or
    /// `Vec3`.
    Size(String),
    /// `vec2(...)` or `vec3(...)`.
    Vector(Vec<DomainExpr>),
    /// `max`, `min`, `floor`, `ceil` by name.
    Call(String, Vec<DomainExpr>),
    /// Negation.
    Neg(Box<DomainExpr>),
    /// Arithmetic.
    Binary(DomainOp, Box<DomainExpr>, Box<DomainExpr>),
}

/// Why a domain expression could not be accepted or resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct DomainError {
    /// Where in the document, when known.
    pub span: Span,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl DomainExpr {
    /// Accepts the subset of the expression language a domain may use.
    pub fn from_expr(expr: &Expr) -> Result<Self, DomainError> {
        let fail = |message: String| DomainError {
            span: expr.span,
            message,
        };
        Ok(match &expr.kind {
            ExprKind::Int(value) => Self::Int(*value),
            ExprKind::Float(value) => Self::Float(*value),
            ExprKind::Name(name) => Self::Name(name.clone()),
            ExprKind::Neg(inner) => Self::Neg(Box::new(Self::from_expr(inner)?)),
            ExprKind::Binary(op, a, b) => {
                let op = match op {
                    BinaryOp::Add => DomainOp::Add,
                    BinaryOp::Sub => DomainOp::Sub,
                    BinaryOp::Mul => DomainOp::Mul,
                    BinaryOp::Div => DomainOp::Div,
                    _ => {
                        return Err(fail("`domain` uses only `+`, `-`, `*`, and `/`".to_owned()));
                    }
                };
                Self::Binary(
                    op,
                    Box::new(Self::from_expr(a)?),
                    Box::new(Self::from_expr(b)?),
                )
            }
            ExprKind::Call(name, args) => match name.as_str() {
                "vec2" | "vec3" => {
                    let arity = if name == "vec2" { 2 } else { 3 };
                    if args.len() != arity {
                        return Err(fail(format!("`{name}` takes {arity} components")));
                    }
                    Self::Vector(
                        args.iter()
                            .map(Self::from_expr)
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
                "count" | "size" => {
                    let [ExprKind::Name(port)] =
                        args.iter().map(|arg| &arg.kind).collect::<Vec<_>>()[..]
                    else {
                        return Err(fail(format!(
                            "`{name}` takes one of this node's input ports by name"
                        )));
                    };
                    if name == "count" {
                        Self::Count(port.clone())
                    } else {
                        Self::Size(port.clone())
                    }
                }
                "max" | "min" | "floor" | "ceil" => {
                    let arity = if name == "floor" || name == "ceil" {
                        1
                    } else {
                        2
                    };
                    if args.len() != arity {
                        return Err(fail(format!("`{name}` takes {arity} argument(s)")));
                    }
                    Self::Call(
                        name.clone(),
                        args.iter()
                            .map(Self::from_expr)
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
                other => {
                    return Err(fail(format!(
                        "`{other}` cannot size a domain; use numbers, `vec2`/`vec3`, `count(port)`, `size(port)`, `max`, `min`, `floor`, `ceil`, and parameters"
                    )));
                }
            },
            _ => {
                return Err(fail(
                    "`domain` is a count, a `vec2`/`vec3`, or arithmetic over `count(port)`, `size(port)`, and parameters".to_owned(),
                ));
            }
        })
    }

    /// Whether the expression refers to nothing: a literal extent.
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Int(_) | Self::Float(_) => true,
            Self::Name(_) | Self::Count(_) | Self::Size(_) => false,
            Self::Vector(parts) | Self::Call(_, parts) => parts.iter().all(Self::is_constant),
            Self::Neg(inner) => inner.is_constant(),
            Self::Binary(_, a, b) => a.is_constant() && b.is_constant(),
        }
    }

    /// Resolves the expression to an extent. `port` gives the extent an
    /// input port is bound to and `param` the value of a parameter, each
    /// `None` when unknown.
    pub fn resolve(
        &self,
        port: &dyn Fn(&str) -> Option<Domain>,
        param: &dyn Fn(&str) -> Option<Value>,
    ) -> Result<Domain, DomainError> {
        let value = self.value(port, param)?;
        let fail = |message: String| DomainError {
            span: Span::default(),
            message,
        };
        let side = |v: f32| {
            (v > 0.0 && v.fract() == 0.0 && v <= u32::MAX as f32)
                .then_some(v as u32)
                .ok_or_else(|| {
                    fail(format!(
                        "`domain` resolves to {self}, whose sides must be positive whole numbers, found {v}"
                    ))
                })
        };
        Ok(match value {
            Value::Int(count) => Domain::Rows(side(count as f32)?),
            Value::Float(count) => Domain::Rows(side(count)?),
            Value::Vec2([width, height]) => Domain::Grid(side(width)?, side(height)?),
            Value::Vec3([width, height, depth]) => {
                Domain::Volume(side(width)?, side(height)?, side(depth)?)
            }
            other => {
                return Err(fail(format!(
                    "`domain` must resolve to a count, a `vec2`, or a `vec3`, found {other}"
                )));
            }
        })
    }

    fn value(
        &self,
        port: &dyn Fn(&str) -> Option<Domain>,
        param: &dyn Fn(&str) -> Option<Value>,
    ) -> Result<Value, DomainError> {
        let fail = |message: String| DomainError {
            span: Span::default(),
            message,
        };
        let extent = |domain: Domain| match domain {
            Domain::Rows(n) => Value::Int(n as i32),
            Domain::Grid(w, h) => Value::Vec2([w as f32, h as f32]),
            Domain::Volume(w, h, d) => Value::Vec3([w as f32, h as f32, d as f32]),
        };
        Ok(match self {
            Self::Int(v) => Value::Int(*v),
            Self::Float(v) => Value::Float(*v),
            Self::Name(name) => param(name).ok_or_else(|| {
                fail(format!(
                    "`domain` names `{name}`, which is neither a parameter of the node nor of the graph"
                ))
            })?,
            Self::Count(name) => Value::Int(
                port(name)
                    .ok_or_else(|| fail(format!("`count({name})`: `{name}` is not a bound input")))?
                    .count() as i32,
            ),
            Self::Size(name) => extent(
                port(name)
                    .ok_or_else(|| fail(format!("`size({name})`: `{name}` is not a bound input")))?,
            ),
            Self::Vector(parts) => {
                let mut components = [0.0; 3];
                for (slot, part) in components.iter_mut().zip(parts) {
                    *slot = match part.value(port, param)? {
                        Value::Int(v) => v as f32,
                        Value::Float(v) => v,
                        other => {
                            return Err(fail(format!(
                                "a vector component must be a number, found {other}"
                            )));
                        }
                    };
                }
                if parts.len() == 2 {
                    Value::Vec2([components[0], components[1]])
                } else {
                    Value::Vec3(components)
                }
            }
            Self::Call(name, args) => {
                let values = args
                    .iter()
                    .map(|arg| arg.value(port, param))
                    .collect::<Result<Vec<_>, _>>()?;
                match (name.as_str(), &values[..]) {
                    ("floor", [v]) => map(*v, f32::floor),
                    ("ceil", [v]) => map(*v, f32::ceil),
                    ("max", [a, b]) => zip(*a, *b, f32::max, i32::max)?,
                    ("min", [a, b]) => zip(*a, *b, f32::min, i32::min)?,
                    _ => unreachable!("accepted calls have their arity"),
                }
            }
            Self::Neg(inner) => map(inner.value(port, param)?, |v| -v),
            Self::Binary(op, a, b) => {
                let (a, b) = (a.value(port, param)?, b.value(port, param)?);
                match op {
                    DomainOp::Add => zip(a, b, |x, y| x + y, |x, y| x.wrapping_add(y))?,
                    DomainOp::Sub => zip(a, b, |x, y| x - y, |x, y| x.wrapping_sub(y))?,
                    DomainOp::Mul => zip(a, b, |x, y| x * y, |x, y| x.wrapping_mul(y))?,
                    DomainOp::Div => {
                        if let Value::Int(0) = b {
                            return Err(fail("`domain` divides by zero".to_owned()));
                        }
                        zip(a, b, |x, y| x / y, |x, y| x.checked_div(y).unwrap_or(0))?
                    }
                }
            }
        })
    }
}

/// Applies a float function to every component; ints stay ints when the
/// result is whole.
fn map(value: Value, f: impl Fn(f32) -> f32) -> Value {
    match value {
        Value::Int(v) => {
            let r = f(v as f32);
            if r.fract() == 0.0 {
                Value::Int(r as i32)
            } else {
                Value::Float(r)
            }
        }
        Value::Float(v) => Value::Float(f(v)),
        Value::Vec2([x, y]) => Value::Vec2([f(x), f(y)]),
        Value::Vec3([x, y, z]) => Value::Vec3([f(x), f(y), f(z)]),
        other => other,
    }
}

/// Combines two values componentwise: ints with ints stay ints, a scalar
/// spreads over a vector, vectors combine with vectors of the same size.
fn zip(
    a: Value,
    b: Value,
    f: impl Fn(f32, f32) -> f32,
    i: impl Fn(i32, i32) -> i32,
) -> Result<Value, DomainError> {
    let fail = || DomainError {
        span: Span::default(),
        message: format!("`domain` cannot combine {a} with {b}"),
    };
    let scalar = |v: Value| match v {
        Value::Int(v) => Some(v as f32),
        Value::Float(v) => Some(v),
        _ => None,
    };
    Ok(match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(i(x, y)),
        (Value::Vec2(x), Value::Vec2(y)) => Value::Vec2([f(x[0], y[0]), f(x[1], y[1])]),
        (Value::Vec3(x), Value::Vec3(y)) => {
            Value::Vec3([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])])
        }
        (Value::Vec2(x), s) if scalar(s).is_some() => {
            let s = scalar(s).unwrap_or(0.0);
            Value::Vec2([f(x[0], s), f(x[1], s)])
        }
        (s, Value::Vec2(y)) if scalar(s).is_some() => {
            let s = scalar(s).unwrap_or(0.0);
            Value::Vec2([f(s, y[0]), f(s, y[1])])
        }
        (Value::Vec3(x), s) if scalar(s).is_some() => {
            let s = scalar(s).unwrap_or(0.0);
            Value::Vec3([f(x[0], s), f(x[1], s), f(x[2], s)])
        }
        (s, Value::Vec3(y)) if scalar(s).is_some() => {
            let s = scalar(s).unwrap_or(0.0);
            Value::Vec3([f(s, y[0]), f(s, y[1]), f(s, y[2])])
        }
        (x, y) => match (scalar(x), scalar(y)) {
            (Some(x), Some(y)) => Value::Float(f(x, y)),
            _ => return Err(fail()),
        },
    })
}

impl fmt::Display for DomainExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 1e9 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::Name(name) => f.write_str(name),
            Self::Count(port) => write!(f, "count({port})"),
            Self::Size(port) => write!(f, "size({port})"),
            Self::Vector(parts) => {
                write!(f, "vec{}(", parts.len())?;
                for (index, part) in parts.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{part}")?;
                }
                f.write_str(")")
            }
            Self::Call(name, args) => {
                write!(f, "{name}(")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
            Self::Neg(inner) => write!(f, "-{}", Parenthesized(inner)),
            Self::Binary(op, a, b) => write!(
                f,
                "{} {} {}",
                Parenthesized(a),
                op.symbol(),
                Parenthesized(b)
            ),
        }
    }
}

/// A sub-expression, parenthesized when it is itself arithmetic so the
/// printed form reads back as the same tree.
struct Parenthesized<'a>(&'a DomainExpr);

impl fmt::Display for Parenthesized<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            DomainExpr::Binary(..) => write!(f, "({})", self.0),
            other => write!(f, "{other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> DomainExpr {
        DomainExpr::from_expr(&orla_language::parse_literal(text).unwrap()).unwrap()
    }

    #[test]
    fn expressions_resolve_over_ports_and_params_and_print_back() {
        let port = |name: &str| match name {
            "map" => Some(Domain::Grid(20, 15)),
            "faces" => Some(Domain::Rows(2304)),
            "cloud" => Some(Domain::Volume(4, 4, 2)),
            _ => None,
        };
        let param = |name: &str| match name {
            "tile" => Some(Value::Float(16.0)),
            "levels" => Some(Value::Int(2)),
            _ => None,
        };
        let cases = [
            ("size(map) * tile", Domain::Grid(320, 240)),
            ("4 * count(faces)", Domain::Rows(9216)),
            ("count(faces) + 3 * count(faces)", Domain::Rows(9216)),
            ("size(map) / vec2(2, 3)", Domain::Grid(10, 5)),
            ("size(cloud) * levels", Domain::Volume(8, 8, 4)),
            ("max(count(faces) / 10000, 1)", Domain::Rows(1)),
            (
                "vec2(ceil(count(faces) / 100.0), 3)",
                Domain::Rows(24).grid_or(3),
            ),
            ("256", Domain::Rows(256)),
            ("vec3(2, 2, 2)", Domain::Volume(2, 2, 2)),
        ];
        for (text, expected) in cases {
            let expr = parse(text);
            assert_eq!(expr.resolve(&port, &param).unwrap(), expected, "{text}");
            let printed = expr.to_string();
            assert_eq!(parse(&printed), expr, "{text} printed as {printed}");
        }
        assert!(parse("256").is_constant());
        assert!(!parse("size(map)").is_constant());
        assert_eq!(
            parse("(size(map) + 1) * tile").to_string(),
            "(size(map) + 1) * tile"
        );

        let error = parse("size(nowhere)").resolve(&port, &param).unwrap_err();
        assert!(error.to_string().contains("not a bound input"), "{error}");
        assert_eq!(
            parse("count(faces) / 7").resolve(&port, &param).unwrap(),
            Domain::Rows(329),
            "ints divide like ints"
        );
        let error = parse("count(faces) / 7.0")
            .resolve(&port, &param)
            .unwrap_err();
        assert!(error.to_string().contains("positive whole"), "{error}");
        let error = parse("size(map) * 1.5").resolve(&port, &param).unwrap_err();
        assert!(error.to_string().contains("positive whole"), "{error}");
        let error = parse("scale").resolve(&port, &param).unwrap_err();
        assert!(error.to_string().contains("neither a parameter"), "{error}");
        let error =
            DomainExpr::from_expr(&orla_language::parse_literal("sin(2.0)").unwrap()).unwrap_err();
        assert!(error.to_string().contains("cannot size"), "{error}");
    }

    trait GridOr {
        fn grid_or(self, height: u32) -> Domain;
    }

    impl GridOr for Domain {
        fn grid_or(self, height: u32) -> Domain {
            Domain::Grid(self.width(), height)
        }
    }
}
