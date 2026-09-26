//! A node's `domain` as written, accepted into the Graph's own
//! [`DomainExpr`]: the subset of the expression language a domain may
//! use, checked where it was written.

use orla_graph::{DomainExpr, DomainOp};
use orla_language::{BinaryOp, Expr, ExprKind};

use crate::DocumentError;

/// Accepts the subset of the expression language a domain may use.
pub fn domain_expr(expr: &Expr) -> Result<DomainExpr, DocumentError> {
    let fail = |message: String| DocumentError {
        span: expr.span,
        message,
    };
    Ok(match &expr.kind {
        ExprKind::Int(value) => DomainExpr::Int(*value),
        ExprKind::Float(value) => DomainExpr::Float(*value),
        ExprKind::Name(name) => DomainExpr::Name(name.clone()),
        ExprKind::Neg(inner) => DomainExpr::Neg(Box::new(domain_expr(inner)?)),
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
            DomainExpr::Binary(op, Box::new(domain_expr(a)?), Box::new(domain_expr(b)?))
        }
        ExprKind::Call(name, args) => match name.as_str() {
            "vec2" | "vec3" => {
                let arity = if name == "vec2" { 2 } else { 3 };
                if args.len() != arity {
                    return Err(fail(format!("`{name}` takes {arity} components")));
                }
                DomainExpr::Vector(
                    args.iter()
                        .map(domain_expr)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            "value" => {
                let kinds: Vec<&ExprKind> = args.iter().map(|arg| &arg.kind).collect();
                let (port, index) = match kinds[..] {
                    [ExprKind::Name(port)] => (port, 0),
                    [ExprKind::Name(port), ExprKind::Int(index)] if index >= &0 => {
                        (port, *index as u32)
                    }
                    _ => {
                        return Err(fail(
                            "`value` takes one of this node's input ports by name, and an element index that is a whole number: `value(live)`, `value(counts, 2)`"
                                .to_owned(),
                        ));
                    }
                };
                DomainExpr::Value(port.clone(), index)
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
                    DomainExpr::Count(port.clone())
                } else {
                    DomainExpr::Size(port.clone())
                }
            }
            "window" | "scale" => {
                if !args.is_empty() {
                    return Err(fail(format!("`{name}` takes no arguments")));
                }
                DomainExpr::Context(name.clone())
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
                DomainExpr::Call(
                    name.clone(),
                    args.iter()
                        .map(domain_expr)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            other => {
                return Err(fail(format!(
                    "`{other}` cannot size a domain; use numbers, `vec2`/`vec3`, `count(port)`, `size(port)`, `value(port)`, `window()`, `scale()`, `max`, `min`, `floor`, `ceil`, and parameters"
                )));
            }
        },
        _ => {
            return Err(fail(
                "`domain` is a count, a `vec2`/`vec3`, or arithmetic over `count(port)`, `size(port)`, `value(port)`, and parameters".to_owned(),
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use orla_abi::{Domain, Value};

    fn parse(text: &str) -> DomainExpr {
        domain_expr(&orla_language::parse_literal(text).unwrap()).unwrap()
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
        let context = |name: &str| match name {
            "window" => Some(Value::Vec2([640.0, 480.0])),
            "scale" => Some(Value::Float(2.0)),
            _ => None,
        };
        let element = |name: &str, index: u32| match (name, index) {
            ("live", 0) => Some(Value::Int(300)),
            ("counts", 2) => Some(Value::Int(7)),
            _ => None,
        };
        let cases = [
            ("window()", Domain::Grid(640, 480)),
            ("window() / scale()", Domain::Grid(320, 240)),
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
            ("value(live)", Domain::Rows(300)),
            ("value(counts, 2) * 16", Domain::Rows(112)),
            ("max(value(live) - 300, 1)", Domain::Rows(1)),
        ];
        for (text, expected) in cases {
            let expr = parse(text);
            assert_eq!(
                expr.resolve(&port, &param, &context, &element).unwrap(),
                expected,
                "{text}"
            );
            let printed = expr.to_string();
            assert_eq!(parse(&printed), expr, "{text} printed as {printed}");
        }
        // A window is whatever size a hand drags it to, so a side that
        // divides is a fraction half the time; the message says what to
        // write instead.
        let odd = |name: &str| match name {
            "window" => Some(Value::Vec2([953.0, 541.0])),
            _ => None,
        };
        let fraction = parse("window() / 2")
            .resolve(&port, &param, &odd, &element)
            .unwrap_err()
            .message;
        assert!(fraction.contains("found 476.5"), "{fraction}");
        assert!(fraction.contains("`floor(...)`"), "{fraction}");
        assert_eq!(
            parse("max(floor(window() / 2), 1)")
                .resolve(&port, &param, &odd, &element)
                .unwrap(),
            Domain::Grid(476, 270),
        );
        // And the smallest window there is leaves a side of nothing,
        // which the other half of that spelling covers.
        let tiny = |name: &str| match name {
            "window" => Some(Value::Vec2([1.0, 1.0])),
            _ => None,
        };
        let zero = parse("floor(window() / 2)")
            .resolve(&port, &param, &tiny, &element)
            .unwrap_err()
            .message;
        assert!(zero.contains("`max(..., 1)`"), "{zero}");
        assert_eq!(
            parse("max(floor(window() / 2), 1)")
                .resolve(&port, &param, &tiny, &element)
                .unwrap(),
            Domain::Grid(1, 1),
        );

        assert!(parse("256").is_constant());
        assert!(!parse("size(map)").is_constant());
        assert!(!parse("value(live)").is_constant());
        assert_eq!(
            parse("value(counts, 2) * 16").values_read(),
            [("counts".to_owned(), 2)]
        );
        assert_eq!(parse("count(faces)").values_read(), []);
        // Fused onto its consumer, the expression follows its ports and
        // its own params under the names fusion gives them; `tile` is
        // not a param of this Kernel, so it is the graph's and stays.
        assert_eq!(
            parse("value(live) * count(faces) * levels * tile")
                .renamed("mesh", &["levels".to_owned()])
                .to_string(),
            "((value(mesh/live) * count(mesh/faces)) * mesh/levels) * tile"
        );
        assert!(!parse("window()").is_constant());
        assert_eq!(
            parse("(size(map) + 1) * tile").to_string(),
            "(size(map) + 1) * tile"
        );

        let error = parse("size(nowhere)")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("not a bound input"), "{error}");
        // A division that leaves a remainder is a fraction, as a
        // window's is: `floor` and `ceil` decide it, bare it sizes nothing.
        assert_eq!(
            parse("floor(count(faces) / 7)")
                .resolve(&port, &param, &context, &element)
                .unwrap(),
            Domain::Rows(329),
        );
        assert_eq!(
            parse("ceil(count(faces) / 7)")
                .resolve(&port, &param, &context, &element)
                .unwrap(),
            Domain::Rows(330),
        );
        let error = parse("count(faces) / 7")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("floor"), "{error}");
        let error = parse("count(faces) / 7.0")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("positive whole"), "{error}");
        let error = parse("size(map) * 1.5")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("positive whole"), "{error}");
        let error = parse("value(nothing)")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("no element 0"), "{error}");
        let error = parse("scale")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("neither a parameter"), "{error}");
        // Counts are exact and whole numbers never wrap: a count past
        // what anything can hold fails where it is worked out, and the
        // only thing that comes out is an extent.
        let error = parse("count(faces) * 4611686018427387904")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("overflows"), "{error}");
        let error = parse("count(faces) * 3000000")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("positive whole"), "{error}");
        assert_eq!(
            parse("floor(size(map) / 3)")
                .resolve(&port, &param, &context, &element)
                .unwrap(),
            Domain::Grid(6, 5),
            "a side divides like the window's"
        );
        let error = parse("window() / 0.0")
            .resolve(&port, &param, &context, &element)
            .unwrap_err();
        assert!(error.to_string().contains("divides by zero"), "{error}");
        // What is not a number or a size sizes nothing.
        let shaped = |name: &str| match name {
            "on" => Some(Value::Bool(true)),
            "tint" => Some(Value::Vec4([1.0; 4])),
            "row" => Some(Value::IVec2([4, 2])),
            _ => None,
        };
        for name in ["on", "tint"] {
            let error = parse(name)
                .resolve(&port, &shaped, &context, &element)
                .unwrap_err();
            assert!(error.to_string().contains("a count or a size"), "{error}");
        }
        assert_eq!(
            parse("row * 2")
                .resolve(&port, &shaped, &context, &element)
                .unwrap(),
            Domain::Grid(8, 4)
        );
        let vector = |_: &str, _: u32| Some(Value::Vec2([1.0, 2.0]));
        let error = parse("value(live)")
            .resolve(&port, &param, &context, &vector)
            .unwrap_err();
        assert!(error.to_string().contains("one number"), "{error}");
        let error = domain_expr(&orla_language::parse_literal("sin(2.0)").unwrap()).unwrap_err();
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
