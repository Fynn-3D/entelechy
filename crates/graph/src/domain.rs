//! A node's `domain` as written: a literal extent, or an expression over
//! the node's own inputs and parameters, resolved when the node binds.
//!
//! The expression language is deliberately small: numbers, `vec2` and
//! `vec3`, `+ - * /`, negation, `count(port)` and `size(port)` over the
//! node's input ports, `value(port)` and `value(port, i)` for a number a
//! Kernel put in one, `window()` and `scale()` from the context, `max`,
//! `min`, `floor`, `ceil`, and names that are the node's parameters or the
//! graph's. It says how big an output is in terms of what comes in, so a
//! Kernel that changes topology (four triangles per triangle, a map at
//! sixteen pixels per tile) can size its output without the graph
//! repeating the arithmetic by hand, and a picture can fill the window.
//!
//! `value(port)` is the one term that reads data rather than shape: the
//! Field bound to that port is materialized and one of its numbers sizes
//! the node. It is how a scene runs over as much as a Kernel decided
//! there is this frame — how many chunks are in front of the eye, how
//! many particles are alive — rather than over a maximum it always pays
//! for. The Field is brought to the Host to read, which on a device is a
//! wait, so the port is held to a row of scalars and to
//! [`VALUE_MAX_ELEMENTS`] of them: it is for counts, not for data.

use std::fmt;

use orla_abi::{Domain, Value};

/// The most elements a Field may hold and still be read by `value(port)`:
/// enough for a row of counts, few enough that reading one is never a
/// download of a scene's data.
pub const VALUE_MAX_ELEMENTS: usize = 1024;

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
    /// `value(port)` or `value(port, i)`: element `i` of the Field the
    /// port is bound to, which is a number a Kernel wrote.
    Value(String, u32),
    /// `window()`, the window's size in pixels, or `scale()`, its scale
    /// factor: context, by name.
    Context(String),
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
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl DomainExpr {
    /// Every `value(port, i)` the expression reads, so a caller can bring
    /// those Fields to the Host before resolving: the resolver itself
    /// only looks things up.
    pub fn values_read(&self) -> Vec<(String, u32)> {
        let mut out = Vec::new();
        self.collect_values(&mut out);
        out
    }

    fn collect_values(&self, out: &mut Vec<(String, u32)>) {
        match self {
            Self::Value(port, index) => {
                if !out.iter().any(|(p, i)| p == port && *i == *index) {
                    out.push((port.clone(), *index));
                }
            }
            Self::Int(_)
            | Self::Float(_)
            | Self::Name(_)
            | Self::Count(_)
            | Self::Size(_)
            | Self::Context(_) => {}
            Self::Vector(parts) | Self::Call(_, parts) => {
                for part in parts {
                    part.collect_values(out);
                }
            }
            Self::Neg(inner) => inner.collect_values(out),
            Self::Binary(_, a, b) => {
                a.collect_values(out);
                b.collect_values(out);
            }
        }
    }

    /// The same expression on a node that absorbed this one's: every
    /// port, and every name that is one of this Kernel's parameters,
    /// under the `prefix/name` fusion gives them. A name that is not a
    /// parameter of the Kernel is the graph's, which fusion does not
    /// rename.
    pub fn renamed(&self, prefix: &str, params: &[String]) -> Self {
        let rename = |name: &String| format!("{prefix}/{name}");
        let inner = |parts: &[Self]| -> Vec<Self> {
            parts
                .iter()
                .map(|part| part.renamed(prefix, params))
                .collect()
        };
        match self {
            Self::Int(v) => Self::Int(*v),
            Self::Float(v) => Self::Float(*v),
            Self::Name(name) => {
                if params.iter().any(|param| param == name) {
                    Self::Name(rename(name))
                } else {
                    Self::Name(name.clone())
                }
            }
            Self::Count(port) => Self::Count(rename(port)),
            Self::Size(port) => Self::Size(rename(port)),
            Self::Value(port, index) => Self::Value(rename(port), *index),
            Self::Context(name) => Self::Context(name.clone()),
            Self::Vector(parts) => Self::Vector(inner(parts)),
            Self::Call(name, parts) => Self::Call(name.clone(), inner(parts)),
            Self::Neg(expr) => Self::Neg(Box::new(expr.renamed(prefix, params))),
            Self::Binary(op, a, b) => Self::Binary(
                *op,
                Box::new(a.renamed(prefix, params)),
                Box::new(b.renamed(prefix, params)),
            ),
        }
    }

    /// Whether the expression refers to nothing: a literal extent.
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Int(_) | Self::Float(_) => true,
            Self::Name(_) | Self::Count(_) | Self::Size(_) | Self::Value(..) | Self::Context(_) => {
                false
            }
            Self::Vector(parts) | Self::Call(_, parts) => parts.iter().all(Self::is_constant),
            Self::Neg(inner) => inner.is_constant(),
            Self::Binary(_, a, b) => a.is_constant() && b.is_constant(),
        }
    }

    /// Resolves the expression to an extent. `port` gives the extent an
    /// input port is bound to, `param` the value of a parameter, and
    /// `context` the window's size (`window`) or scale (`scale`), each
    /// `None` when unknown.
    pub fn resolve(
        &self,
        port: &dyn Fn(&str) -> Option<Domain>,
        param: &dyn Fn(&str) -> Option<Value>,
        context: &dyn Fn(&str) -> Option<Value>,
        element: &dyn Fn(&str, u32) -> Option<Value>,
    ) -> Result<Domain, DomainError> {
        let value = self.value(port, param, context, element)?;
        let fail = |message: String| DomainError { message };
        let side = |v: f32| {
            (v > 0.0 && v.fract() == 0.0 && v <= u32::MAX as f32)
                .then_some(v as u32)
                .ok_or_else(|| {
                    // Two cases are worth naming the remedy for: a count
                    // a Kernel works out reaches zero the frame it culls
                    // everything, and a side is fractional as soon as
                    // somebody drags a window's corner to an odd width or
                    // a Kernel writes a count as a `Float`.
                    let hint = if v == 0.0 {
                        "; a count that can reach zero wants `max(..., 1)`"
                    } else if v.fract() != 0.0 {
                        "; a side that comes out fractional wants `floor(...)`"
                    } else {
                        ""
                    };
                    fail(format!(
                        "`domain` resolves to {self}, whose sides must be positive whole numbers, found {v}{hint}"
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
        context: &dyn Fn(&str) -> Option<Value>,
        element: &dyn Fn(&str, u32) -> Option<Value>,
    ) -> Result<Value, DomainError> {
        let fail = |message: String| DomainError { message };
        let extent = |domain: Domain| match domain {
            Domain::Rows(n) => Value::Int(n as i32),
            Domain::Grid(w, h) => Value::Vec2([w as f32, h as f32]),
            Domain::Volume(w, h, d) => Value::Vec3([w as f32, h as f32, d as f32]),
        };
        Ok(match self {
            Self::Int(v) => Value::Int(*v),
            Self::Float(v) => Value::Float(*v),
            // A value written in a scene is exact; an extent is counted in
            // floats and ints, which is what the arithmetic below takes.
            Self::Name(name) => extent_number(param(name).ok_or_else(|| {
                fail(format!(
                    "`domain` names `{name}`, which is neither a parameter of the node nor of the graph"
                ))
            })?),
            Self::Count(name) => Value::Int(
                port(name)
                    .ok_or_else(|| fail(format!("`count({name})`: `{name}` is not a bound input")))?
                    .count() as i32,
            ),
            Self::Size(name) => extent(
                port(name)
                    .ok_or_else(|| fail(format!("`size({name})`: `{name}` is not a bound input")))?,
            ),
            Self::Value(name, index) => element(name, *index).ok_or_else(|| {
                fail(format!(
                    "`value({name}, {index})`: `{name}` is not a bound input, or its Field has no element {index}"
                ))
            })?,
            Self::Context(name) => context(name).ok_or_else(|| {
                fail(format!(
                    "`{name}()`: the evaluation has no window; the Player opens one, `run` takes `--window`, and a replay says `window w h`"
                ))
            })?,
            Self::Vector(parts) => {
                let mut components = [0.0; 3];
                for (slot, part) in components.iter_mut().zip(parts) {
                    *slot = match part.value(port, param, context, element)? {
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
                    .map(|arg| arg.value(port, param, context, element))
                    .collect::<Result<Vec<_>, _>>()?;
                match (name.as_str(), &values[..]) {
                    ("floor", [v]) => map(v.clone(), f32::floor),
                    ("ceil", [v]) => map(v.clone(), f32::ceil),
                    ("max", [a, b]) => zip(a.clone(), b.clone(), f32::max, i32::max)?,
                    ("min", [a, b]) => zip(a.clone(), b.clone(), f32::min, i32::min)?,
                    _ => unreachable!("accepted calls have their arity"),
                }
            }
            Self::Neg(inner) => map(inner.value(port, param, context, element)?, |v| -v),
            Self::Binary(op, a, b) => {
                let (a, b) = (
                    a.value(port, param, context, element)?,
                    b.value(port, param, context, element)?,
                );
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

/// A number as an extent counts it: a double as a float, a long as an
/// int where it fits and a float where it does not.
fn extent_number(value: Value) -> Value {
    match value {
        Value::Double(v) => Value::Float(v as f32),
        Value::Long(v) => i32::try_from(v).map_or(Value::Float(v as f32), Value::Int),
        Value::DVec2([x, y]) => Value::Vec2([x as f32, y as f32]),
        Value::DVec3([x, y, z]) => Value::Vec3([x as f32, y as f32, z as f32]),
        other => other,
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
        message: format!("`domain` cannot combine {a} with {b}"),
    };
    let scalar = |v: &Value| match v {
        Value::Int(v) => Some(*v as f32),
        Value::Float(v) => Some(*v),
        _ => None,
    };
    Ok(match (&a, &b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(i(*x, *y)),
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
            Self::Value(port, 0) => write!(f, "value({port})"),
            Self::Value(port, index) => write!(f, "value({port}, {index})"),
            Self::Size(port) => write!(f, "size({port})"),
            Self::Context(name) => write!(f, "{name}()"),
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
