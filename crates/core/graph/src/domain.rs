//! A node's `domain`: a literal extent, or an expression over the node's
//! inputs, parameters and context, resolved when the node binds.
//!
//! Terms: numbers, `vec2`, `vec3`, `+ - * / <<`, negation, `count(port)`,
//! `size(port)`, `value(port)`, `value(port, i)`, `window()`, `scale()`,
//! `max`, `min`, `floor`, `ceil`, and parameter names.
//!
//! `value(port)` reads data: one number of the Field bound to the port,
//! brought to the Host. The port holds a row of at most
//! [`VALUE_MAX_ELEMENTS`] scalars: counts, not data.
//!
//! Values come in and only an extent comes out: the Graph computes
//! structure, never values.

use std::fmt;

use orla_abi::{Domain, Value};

/// The most elements a Field read by `value(port)` may hold.
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
    /// `<<`, of whole numbers.
    Shl,
}

impl DomainOp {
    fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Shl => "<<",
        }
    }
}

/// A domain expression.
#[derive(Clone, Debug, PartialEq)]
pub enum DomainExpr {
    /// An integer literal.
    Int(i64),
    /// A float literal, exact as written.
    Float(f64),
    /// A parameter of the node or of the graph.
    Name(String),
    /// `count(port)`: the number of elements the port is bound to.
    Count(String),
    /// `size(port)`: the extent the port is bound to, as `Int`, `Vec2`, or
    /// `Vec3`.
    Size(String),
    /// `value(port)` or `value(port, i)`: element `i` of the Field the
    /// port is bound to.
    Value(String, u32),
    /// `window()` (size in pixels) or `scale()` (scale factor).
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
    /// Every `value(port, i)` the expression reads; the caller brings
    /// those Fields to the Host before resolving.
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

    /// The expression on a fused node: ports and this Kernel's parameter
    /// names under `prefix/name`; graph parameter names stay.
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

    /// Replaces every name `given` answers for; other names stay. A
    /// placed graph's parameters take the values its parent passed.
    pub fn with_names(&self, given: &dyn Fn(&str) -> Option<DomainExpr>) -> Self {
        let inner = |parts: &[Self]| -> Vec<Self> {
            parts.iter().map(|part| part.with_names(given)).collect()
        };
        match self {
            Self::Name(name) => given(name).unwrap_or_else(|| Self::Name(name.clone())),
            Self::Int(_) | Self::Float(_) | Self::Count(_) | Self::Size(_) | Self::Value(..) => {
                self.clone()
            }
            Self::Context(_) => self.clone(),
            Self::Vector(parts) => Self::Vector(inner(parts)),
            Self::Call(name, parts) => Self::Call(name.clone(), inner(parts)),
            Self::Neg(expr) => Self::Neg(Box::new(expr.with_names(given))),
            Self::Binary(op, a, b) => Self::Binary(
                *op,
                Box::new(a.with_names(given)),
                Box::new(b.with_names(given)),
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

    /// Resolves the expression to an extent. `port` gives an input's
    /// extent, `param` a parameter, `context` `window` or `scale`, and
    /// `element` a number a Kernel wrote; each is `None` when unknown.
    /// The arithmetic is over [`Size`], so only structure comes out.
    pub fn resolve(
        &self,
        port: &dyn Fn(&str) -> Option<Domain>,
        param: &dyn Fn(&str) -> Option<Value>,
        context: &dyn Fn(&str) -> Option<Value>,
        element: &dyn Fn(&str, u32) -> Option<Value>,
    ) -> Result<Domain, DomainError> {
        let size = self.size(&Inputs {
            port,
            param,
            context,
            element,
        })?;
        let side = |n: Number| {
            let whole = match n {
                Number::Whole(v) => Some(v),
                Number::Measure(v) if v.fract() == 0.0 && v.abs() < 1e18 => Some(v as i64),
                Number::Measure(_) => None,
            };
            whole
                .filter(|v| (1..=i64::from(u32::MAX)).contains(v))
                .map(|v| v as u32)
                .ok_or_else(|| {
                    // A zero count and a fractional side each get a
                    // remedy in the message.
                    let hint = match n {
                        _ if n.exact() == 0.0 => {
                            "; a count that can reach zero wants `max(..., 1)`"
                        }
                        Number::Measure(v) if v.fract() != 0.0 => {
                            "; a side that comes out fractional wants `floor(...)`"
                        }
                        _ => "",
                    };
                    DomainError {
                        message: format!(
                            "`domain` resolves to {self}, whose sides must be positive whole numbers, found {n}{hint}"
                        ),
                    }
                })
        };
        Ok(match size {
            Size::One(n) => Domain::Rows(side(n)?),
            Size::Two([w, h]) => Domain::Grid(side(w)?, side(h)?),
            Size::Three([w, h, d]) => Domain::Volume(side(w)?, side(h)?, side(d)?),
        })
    }

    fn size(&self, inputs: &Inputs<'_>) -> Result<Size, DomainError> {
        let fail = |message: String| DomainError { message };
        Ok(match self {
            Self::Int(v) => Size::One(Number::Whole(*v)),
            Self::Float(v) => Size::One(Number::Measure(*v)),
            Self::Name(name) => {
                let value = (inputs.param)(name).ok_or_else(|| {
                    fail(format!(
                        "`domain` names `{name}`, which is neither a parameter of the node nor of the graph"
                    ))
                })?;
                Size::of(&value).ok_or_else(|| {
                    fail(format!(
                        "`domain` names `{name}`, which is {value}: a count or a size is a number, or a vector of two or three"
                    ))
                })?
            }
            Self::Count(name) => Size::One(Number::Whole(
                (inputs.port)(name)
                    .ok_or_else(|| fail(format!("`count({name})`: `{name}` is not a bound input")))?
                    .count() as i64,
            )),
            Self::Size(name) => {
                let whole = |v: u32| Number::Whole(i64::from(v));
                match (inputs.port)(name)
                    .ok_or_else(|| fail(format!("`size({name})`: `{name}` is not a bound input")))?
                {
                    Domain::Rows(n) => Size::One(whole(n)),
                    Domain::Grid(w, h) => Size::Two([whole(w), whole(h)]),
                    Domain::Volume(w, h, d) => Size::Three([whole(w), whole(h), whole(d)]),
                }
            }
            Self::Value(name, index) => {
                let value = (inputs.element)(name, *index).ok_or_else(|| {
                    fail(format!(
                        "`value({name}, {index})`: `{name}` is not a bound input, or its Field has no element {index}"
                    ))
                })?;
                match Size::of(&value) {
                    Some(Size::One(n)) => Size::One(n),
                    _ => {
                        return Err(fail(format!(
                            "`value({name}, {index})` is {value}, where a count is one number"
                        )));
                    }
                }
            }
            Self::Context(name) => {
                let value = (inputs.context)(name).ok_or_else(|| {
                    fail(format!(
                        "`{name}()`: the evaluation has no window; the Player opens one, `run` takes `--window`, and a replay says `window w h`"
                    ))
                })?;
                Size::of(&value).ok_or_else(|| fail(format!("`{name}()` is {value}")))?
            }
            Self::Vector(parts) => {
                let mut components = Vec::with_capacity(parts.len());
                for part in parts {
                    match part.size(inputs)? {
                        Size::One(n) => components.push(n),
                        other => {
                            return Err(fail(format!(
                                "a vector component must be a number, found {other}"
                            )));
                        }
                    }
                }
                match components[..] {
                    [x, y] => Size::Two([x, y]),
                    [x, y, z] => Size::Three([x, y, z]),
                    _ => unreachable!("accepted vectors have two or three components"),
                }
            }
            Self::Call(name, args) => {
                let sizes = args
                    .iter()
                    .map(|arg| arg.size(inputs))
                    .collect::<Result<Vec<_>, _>>()?;
                match (name.as_str(), &sizes[..]) {
                    ("floor", [v]) => v.map(|n| n.rounded(f64::floor))?,
                    ("ceil", [v]) => v.map(|n| n.rounded(f64::ceil))?,
                    ("max", [a, b]) => a.zip(*b, |x, y| Ok(if y.above(x) { y } else { x }))?,
                    ("min", [a, b]) => a.zip(*b, |x, y| Ok(if y.above(x) { x } else { y }))?,
                    _ => unreachable!("accepted calls have their arity"),
                }
            }
            Self::Neg(inner) => inner.size(inputs)?.map(|n| match n {
                Number::Whole(v) => v.checked_neg().map(Number::Whole).ok_or_else(overflow),
                Number::Measure(v) => Ok(Number::Measure(-v)),
            })?,
            Self::Binary(op, a, b) => {
                let (a, b) = (a.size(inputs)?, b.size(inputs)?);
                a.zip(b, |x, y| x.apply(*op, y))?
            }
        })
    }
}

/// What a domain expression reads, by where it comes from.
struct Inputs<'a> {
    port: &'a dyn Fn(&str) -> Option<Domain>,
    param: &'a dyn Fn(&str) -> Option<Value>,
    context: &'a dyn Fn(&str) -> Option<Value>,
    element: &'a dyn Fn(&str, u32) -> Option<Value>,
}

/// A number the Graph counts with: an exact whole number, or a measure
/// (a scale, a float parameter) that must come out whole to size anything.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Number {
    Whole(i64),
    Measure(f64),
}

impl Number {
    fn exact(self) -> f64 {
        match self {
            Self::Whole(v) => v as f64,
            Self::Measure(v) => v,
        }
    }

    /// Whether `self` is greater than `other`.
    fn above(self, other: Self) -> bool {
        match (self, other) {
            (Self::Whole(a), Self::Whole(b)) => a > b,
            (a, b) => a.exact() > b.exact(),
        }
    }

    /// `floor` or `ceil` to a whole number; fails when too large to count.
    fn rounded(self, f: fn(f64) -> f64) -> Result<Self, DomainError> {
        match self {
            Self::Whole(v) => Ok(Self::Whole(v)),
            Self::Measure(v) => {
                let r = f(v);
                (r.is_finite() && r.abs() < 1e18)
                    .then_some(Self::Whole(r as i64))
                    .ok_or_else(|| DomainError {
                        message: format!("`domain` rounds {v}, which is no count"),
                    })
            }
        }
    }

    /// `self op other`: whole numbers stay exact and fail on overflow; a
    /// division with a remainder, or a measure on either side, gives a
    /// measure.
    fn apply(self, op: DomainOp, other: Self) -> Result<Self, DomainError> {
        if other.exact() == 0.0 && op == DomainOp::Div {
            return Err(DomainError {
                message: "`domain` divides by zero".to_owned(),
            });
        }
        if op == DomainOp::Shl {
            return match (self, other) {
                (Self::Whole(a), Self::Whole(b)) if (0..=62).contains(&b) => {
                    a.checked_mul(1 << b).map(Self::Whole).ok_or_else(overflow)
                }
                _ => Err(DomainError {
                    message: format!("`{self} << {other}`: `<<` shifts a whole number by 0 to 62"),
                }),
            };
        }
        Ok(match (self, other) {
            (Self::Whole(a), Self::Whole(b))
                if op == DomainOp::Div && a.checked_rem(b).is_some_and(|r| r != 0) =>
            {
                Self::Measure(a as f64 / b as f64)
            }
            (Self::Whole(a), Self::Whole(b)) => Self::Whole(
                match op {
                    DomainOp::Add => a.checked_add(b),
                    DomainOp::Sub => a.checked_sub(b),
                    DomainOp::Mul => a.checked_mul(b),
                    DomainOp::Div => a.checked_div(b),
                    DomainOp::Shl => unreachable!("shifted above"),
                }
                .ok_or_else(overflow)?,
            ),
            (a, b) => {
                let (a, b) = (a.exact(), b.exact());
                Self::Measure(match op {
                    DomainOp::Add => a + b,
                    DomainOp::Sub => a - b,
                    DomainOp::Mul => a * b,
                    DomainOp::Div => a / b,
                    DomainOp::Shl => unreachable!("shifted above"),
                })
            }
        })
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whole(v) => write!(f, "{v}"),
            Self::Measure(v) => write!(f, "{v}"),
        }
    }
}

fn overflow() -> DomainError {
    DomainError {
        message: "`domain` overflows: a count past 2^63 sizes nothing".to_owned(),
    }
}

/// What a domain expression computes: a count, or the sides of a grid or
/// a volume.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Size {
    One(Number),
    Two([Number; 2]),
    Three([Number; 3]),
}

impl Size {
    /// A value as a size: a number, or a vector of two or three numbers.
    fn of(value: &Value) -> Option<Self> {
        let whole = |v: i64| Number::Whole(v);
        let measure = |v: f64| Number::Measure(v);
        Some(match value {
            Value::Int(v) => Self::One(whole(i64::from(*v))),
            Value::Long(v) => Self::One(whole(*v)),
            Value::Float(v) => Self::One(measure(f64::from(*v))),
            Value::Double(v) => Self::One(measure(*v)),
            Value::IVec2(v) => Self::Two(v.map(|x| whole(i64::from(x)))),
            Value::IVec3(v) => Self::Three(v.map(|x| whole(i64::from(x)))),
            Value::LVec2(v) => Self::Two(v.map(whole)),
            Value::LVec3(v) => Self::Three(v.map(whole)),
            Value::Vec2(v) => Self::Two(v.map(|x| measure(f64::from(x)))),
            Value::Vec3(v) => Self::Three(v.map(|x| measure(f64::from(x)))),
            Value::DVec2(v) => Self::Two(v.map(measure)),
            Value::DVec3(v) => Self::Three(v.map(measure)),
            _ => return None,
        })
    }

    fn parts(&self) -> &[Number] {
        match self {
            Self::One(n) => std::slice::from_ref(n),
            Self::Two(v) => v,
            Self::Three(v) => v,
        }
    }

    fn from_parts(parts: &[Number]) -> Self {
        match *parts {
            [n] => Self::One(n),
            [x, y] => Self::Two([x, y]),
            [x, y, z] => Self::Three([x, y, z]),
            _ => unreachable!("a size has one to three parts"),
        }
    }

    fn map(&self, f: impl Fn(Number) -> Result<Number, DomainError>) -> Result<Self, DomainError> {
        let parts = self
            .parts()
            .iter()
            .map(|n| f(*n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_parts(&parts))
    }

    /// Combines two sizes part by part: a number spreads over a vector,
    /// and vectors combine with vectors of their own length.
    fn zip(
        &self,
        other: Self,
        f: impl Fn(Number, Number) -> Result<Number, DomainError>,
    ) -> Result<Self, DomainError> {
        let (a, b) = (self.parts(), other.parts());
        let n = a.len().max(b.len());
        if a.len() != b.len() && a.len() != 1 && b.len() != 1 {
            return Err(DomainError {
                message: format!("`domain` cannot combine {self} with {other}"),
            });
        }
        let at = |parts: &[Number], k: usize| parts[if parts.len() == 1 { 0 } else { k }];
        let parts = (0..n)
            .map(|k| f(at(a, k), at(b, k)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_parts(&parts))
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::One(n) => write!(f, "{n}"),
            Self::Two([x, y]) => write!(f, "vec2({x}, {y})"),
            Self::Three([x, y, z]) => write!(f, "vec3({x}, {y}, {z})"),
        }
    }
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

/// A sub-expression, parenthesized when arithmetic, so it reads back as
/// the same tree.
struct Parenthesized<'a>(&'a DomainExpr);

impl fmt::Display for Parenthesized<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            DomainExpr::Binary(..) => write!(f, "({})", self.0),
            other => write!(f, "{other}"),
        }
    }
}

/// How a requirement compares its two sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// `<`
    Less,
    /// `<=`
    AtMost,
    /// `>`
    Greater,
    /// `>=`
    AtLeast,
    /// `==`
    Equal,
    /// `!=`
    Unequal,
}

impl Compare {
    /// How it is written.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Less => "<",
            Self::AtMost => "<=",
            Self::Greater => ">",
            Self::AtLeast => ">=",
            Self::Equal => "==",
            Self::Unequal => "!=",
        }
    }

    fn holds(self, a: Number, b: Number) -> bool {
        let order = match (a, b) {
            (Number::Whole(a), Number::Whole(b)) => a.partial_cmp(&b),
            (a, b) => a.exact().partial_cmp(&b.exact()),
        };
        let Some(order) = order else {
            return self == Self::Unequal;
        };
        match self {
            Self::Less => order.is_lt(),
            Self::AtMost => order.is_le(),
            Self::Greater => order.is_gt(),
            Self::AtLeast => order.is_ge(),
            Self::Equal => order.is_eq(),
            Self::Unequal => order.is_ne(),
        }
    }
}

/// What a graph's parameters must hold for it to run, `require name: a
/// <= b;`: two counts over its parameters and the window, compared, and
/// checked before each evaluation.
#[derive(Clone, Debug)]
pub struct Requirement {
    /// Its name; a placed graph's is prefixed as its nodes are.
    pub name: String,
    /// The left side.
    pub left: DomainExpr,
    /// The comparison.
    pub compare: Compare,
    /// The right side.
    pub right: DomainExpr,
    /// Where it was written, `file:line:column`, for diagnostics.
    pub at: Option<String>,
}

/// Equality ignores `at`, as a node's does.
impl PartialEq for Requirement {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.left == other.left
            && self.compare == other.compare
            && self.right == other.right
    }
}

impl Requirement {
    /// Every name `given` answers for replaced, as [`DomainExpr::with_names`].
    pub fn with_names(&self, given: &dyn Fn(&str) -> Option<DomainExpr>) -> Self {
        Self {
            name: self.name.clone(),
            left: self.left.with_names(given),
            compare: self.compare,
            right: self.right.with_names(given),
            at: self.at.clone(),
        }
    }

    /// Whether it holds under `param` and `context`; the error says what
    /// each side came to and the parameters it read.
    pub fn check(
        &self,
        param: &dyn Fn(&str) -> Option<Value>,
        context: &dyn Fn(&str) -> Option<Value>,
    ) -> Result<(), DomainError> {
        let inputs = Inputs {
            port: &|_| None,
            param,
            context,
            element: &|_, _| None,
        };
        let side = |expr: &DomainExpr| -> Result<Number, DomainError> {
            match expr.size(&inputs)? {
                Size::One(n) => Ok(n),
                other => Err(DomainError {
                    message: format!("`{}` compares numbers; `{expr}` is {other}", self.name),
                }),
            }
        };
        let (a, b) = (side(&self.left)?, side(&self.right)?);
        if self.compare.holds(a, b) {
            return Ok(());
        }
        let mut names = Vec::new();
        self.left.collect_names(&mut names);
        self.right.collect_names(&mut names);
        let read: Vec<String> = names
            .iter()
            .filter_map(|name| param(name).map(|value| format!("{name} = {value}")))
            .collect();
        let with = if read.is_empty() {
            String::new()
        } else {
            format!(", with {}", read.join(", "))
        };
        let at = self
            .at
            .as_ref()
            .map(|at| format!(" ({at})"))
            .unwrap_or_default();
        Err(DomainError {
            message: format!(
                "`{}`{at} does not hold: {} {} {}, here {a} {} {b}{with}",
                self.name,
                self.left,
                self.compare.symbol(),
                self.right,
                self.compare.symbol()
            ),
        })
    }
}

impl DomainExpr {
    fn collect_names(&self, out: &mut Vec<String>) {
        match self {
            Self::Name(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Self::Vector(parts) | Self::Call(_, parts) => {
                for part in parts {
                    part.collect_names(out);
                }
            }
            Self::Neg(inner) => inner.collect_names(out),
            Self::Binary(_, a, b) => {
                a.collect_names(out);
                b.collect_names(out);
            }
            _ => {}
        }
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} {} {}",
            self.name,
            self.left,
            self.compare.symbol(),
            self.right
        )
    }
}
