//! A Kernel's declarations, checked into its manifest: structs, aliases,
//! ports and layouts, parameters and defaults, the execution mode; and
//! the names reserved from functions.

use std::str::FromStr;

use orla_ir::{
    Capability, Combine, Field, KernelId, Manifest, Mode, Param, Port, PortType, Shape, StructDef,
    Type, Value, schema_of,
};
use orla_language::{Expr, ExprKind, Kernel, Span, Spanned};

use crate::CompileError;

use super::{Structs, describe, resolve_type, semantic};

/// The Kernel's declared types: its enums, then its structs, whose fields
/// may hold an enum.
pub(super) fn structs(kernel: &Kernel) -> Result<Structs, CompileError> {
    let structs = Structs::default();
    for decl in &kernel.types {
        let Some(variants) = decl.variants() else {
            continue;
        };
        let name = &decl.name.value;
        if name == "Image" || Type::from_str(name).is_ok() {
            return Err(semantic(
                decl.name.span,
                format!("`{name}` is already a type name"),
            ));
        }
        if structs.variants(name).is_some() {
            return Err(semantic(
                decl.name.span,
                format!("type `{name}` is defined twice"),
            ));
        }
        structs.enums.borrow_mut().push((
            name.clone(),
            variants
                .into_iter()
                .map(|(variant, number)| (variant.to_owned(), number))
                .collect(),
        ));
    }
    struct_table_in(structs, &kernel.structs)
}

/// The struct table the declarations make, in order.
pub(super) fn struct_table(decls: &[orla_language::StructDecl]) -> Result<Structs, CompileError> {
    struct_table_in(Structs::default(), decls)
}

/// `structs` with the declared structs appended, in order.
fn struct_table_in(
    structs: Structs,
    decls: &[orla_language::StructDecl],
) -> Result<Structs, CompileError> {
    for decl in decls {
        if structs.variants(&decl.name.value).is_some() {
            return Err(semantic(
                decl.name.span,
                format!("`{}` is already an enum", decl.name.value),
            ));
        }
        if structs.find(&decl.name.value).is_some() {
            return Err(semantic(
                decl.name.span,
                format!("struct `{}` is defined twice", decl.name.value),
            ));
        }
        if Type::from_str(&decl.name.value).is_ok() {
            return Err(semantic(
                decl.name.span,
                format!("`{}` is a builtin type name", decl.name.value),
            ));
        }
        let mut fields: Vec<Field> = Vec::new();
        for (name, ty_name) in &decl.fields {
            if fields.iter().any(|field| field.name == name.value) {
                return Err(semantic(
                    name.span,
                    format!("field `{}` is declared twice", name.value),
                ));
            }
            if ty_name.value == decl.name.value {
                return Err(semantic(ty_name.span, "a struct cannot contain itself"));
            }
            let ty = resolve_type(ty_name, &structs)?;
            fields.push(Field {
                name: name.value.clone(),
                ty,
            });
        }
        let def = StructDef::record(decl.name.value.clone(), fields);
        super::fits(&structs.borrow(), &def, decl.name.span)?;
        structs.borrow_mut().push(def);
    }
    Ok(structs)
}

/// Evaluates an untyped literal: integers stay `Int`; a struct literal
/// is refused, with no struct declared.
pub(crate) fn literal(expr: &Expr) -> Result<Value, CompileError> {
    literal_in(expr, &|_| None)
}

/// Evaluates an untyped literal of any Kernel value: numbers (`Int`, or
/// `Long` when too big; suffixes `d` and `l` say `Double` and `Long`),
/// booleans, vector and matrix constructors and their `d`/`l` kin,
/// `[a, b]`, `[v; N]`, and `Name { field: .. }` with fields in the order
/// `fields_of` gives. An array or struct is a record of its members;
/// [`orla_ir::value_as`] shapes it into a type.
pub(crate) fn literal_in(
    expr: &Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<Value, CompileError> {
    literal_with(expr, fields_of, false)
}

/// Evaluates a literal a type will shape, keeping written floats exact
/// (as `Double`) until the type sets their width: `0.1` for a `Double`
/// is the double nearest a tenth.
pub(crate) fn literal_exact(
    expr: &Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Result<Value, CompileError> {
    literal_with(expr, fields_of, true)
}

/// An integer as written: an `Int` where it fits, else a `Long`.
pub(crate) fn integer(value: i64) -> Value {
    i32::try_from(value).map_or(Value::Long(value), Value::Int)
}

fn literal_with(
    expr: &Expr,
    fields_of: &dyn Fn(&str) -> Option<Vec<String>>,
    exact: bool,
) -> Result<Value, CompileError> {
    let fail = |message: String| semantic(expr.span, message);
    let literal_in = |expr: &Expr, fields_of: &dyn Fn(&str) -> Option<Vec<String>>| {
        literal_with(expr, fields_of, exact)
    };
    match &expr.kind {
        ExprKind::Float(value) if exact => Ok(Value::Double(*value)),
        ExprKind::Float(value) => Ok(Value::Float(*value as f32)),
        ExprKind::Int(value) => Ok(integer(*value)),
        ExprKind::Double(value) => Ok(Value::Double(*value)),
        ExprKind::Long(value) => Ok(Value::Long(*value)),
        ExprKind::Bool(value) => Ok(Value::Bool(*value)),
        ExprKind::Neg(inner) => {
            let value = literal_in(inner, fields_of)?;
            negated(&value).ok_or_else(|| match value {
                Value::Int(_) => fail("integer literal overflows when negated".into()),
                other => fail(format!("cannot negate `{other}`")),
            })
        }
        ExprKind::Array(elements) => Ok(Value::Record(
            elements
                .iter()
                .map(|element| literal_in(element, fields_of))
                .collect::<Result<_, _>>()?,
        )),
        ExprKind::Repeat(element, count) => {
            let value = literal_in(element, fields_of)?;
            Ok(Value::Record(vec![value; count.value as usize]))
        }
        ExprKind::Struct { name, base, fields } => {
            if base.is_some() {
                return Err(fail(
                    "a literal struct lists its fields; `..base` belongs in a Kernel".into(),
                ));
            }
            let order = fields_of(&name.value).ok_or_else(|| {
                semantic(
                    name.span,
                    format!(
                        "`{}` is not a struct declared where this literal is read",
                        name.value
                    ),
                )
            })?;
            let mut given: Vec<Option<Value>> = vec![None; order.len()];
            for (field, value) in fields {
                let Some(at) = order.iter().position(|f| *f == field.value) else {
                    return Err(semantic(
                        field.span,
                        format!("`{}` has no field `{}`", name.value, field.value),
                    ));
                };
                if given[at].is_some() {
                    return Err(semantic(
                        field.span,
                        format!("field `{}` is set twice", field.value),
                    ));
                }
                given[at] = Some(literal_in(value, fields_of)?);
            }
            let members = given
                .into_iter()
                .zip(&order)
                .map(|(value, field)| {
                    value.ok_or_else(|| {
                        semantic(
                            name.span,
                            format!("`{}` needs field `{field}`", name.value),
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(Value::Record(members))
        }
        ExprKind::Call(name, args) => {
            // A constructor of doubles takes its numbers exact.
            let exact = exact || name.starts_with("dvec") || name.starts_with("dmat");
            let values = args
                .iter()
                .map(|arg| literal_with(arg, fields_of, exact))
                .collect::<Result<Vec<_>, _>>()?;
            constructed(name, &values).ok_or_else(|| {
                fail(format!(
                    "`{name}(..)` is not a literal of numbers; a literal is a number, `true`/`false`, a vector, a matrix, or an array or struct of them"
                ))
            })
        }
        _ => Err(fail(
            "expected a literal: a number, `true`/`false`, a vector, a matrix, or an array or struct of them"
                .into(),
        )),
    }
}

/// A numeric literal negated component by component.
fn negated(value: &Value) -> Option<Value> {
    fn floats<T: std::ops::Neg<Output = T> + Copy, const N: usize>(v: &[T; N]) -> [T; N] {
        v.map(|x| -x)
    }
    fn ints<T: num::CheckedNeg + Copy + Default, const N: usize>(v: &[T; N]) -> Option<[T; N]> {
        let mut out = [T::default(); N];
        for (slot, x) in out.iter_mut().zip(v) {
            *slot = x.checked_neg()?;
        }
        Some(out)
    }
    Some(match value {
        Value::Float(v) => Value::Float(-v),
        Value::Int(v) => Value::Int(v.checked_neg()?),
        Value::Double(v) => Value::Double(-v),
        Value::Long(v) => Value::Long(v.checked_neg()?),
        Value::Bool(_) => return None,
        Value::Vec2(v) => Value::Vec2(floats(v)),
        Value::Vec3(v) => Value::Vec3(floats(v)),
        Value::Vec4(v) => Value::Vec4(floats(v)),
        Value::IVec2(v) => Value::IVec2(ints(v)?),
        Value::IVec3(v) => Value::IVec3(ints(v)?),
        Value::IVec4(v) => Value::IVec4(ints(v)?),
        Value::Mat2(v) => Value::Mat2(floats(v)),
        Value::Mat3(v) => Value::Mat3(floats(v)),
        Value::Mat4(v) => Value::Mat4(floats(v)),
        Value::DVec2(v) => Value::DVec2(floats(v)),
        Value::DVec3(v) => Value::DVec3(floats(v)),
        Value::DVec4(v) => Value::DVec4(floats(v)),
        Value::LVec2(v) => Value::LVec2(ints(v)?),
        Value::LVec3(v) => Value::LVec3(ints(v)?),
        Value::DMat2(v) => Value::DMat2(floats(v)),
        Value::DMat3(v) => Value::DMat3(floats(v)),
        Value::DMat4(v) => Value::DMat4(floats(v)),
        Value::Record(members) => {
            Value::Record(members.iter().map(negated).collect::<Option<Vec<_>>>()?)
        }
    })
}

/// Integer negation checked for overflow.
mod num {
    pub(super) trait CheckedNeg: Sized {
        fn checked_neg(self) -> Option<Self>;
    }
    impl CheckedNeg for i32 {
        fn checked_neg(self) -> Option<Self> {
            i32::checked_neg(self)
        }
    }
    impl CheckedNeg for i64 {
        fn checked_neg(self) -> Option<Self> {
            i64::checked_neg(self)
        }
    }
}

/// A constructor of literals: `vec3(1, 0, 0)` with integers widened,
/// `ivec2(3, 4)`, `mat3(c0, c1, c2)` from columns, `mat3(1.0)` down the
/// diagonal, and their double and long kin.
fn constructed(name: &str, args: &[Value]) -> Option<Value> {
    use orla_ir::{Number, Scalar};
    let (scalar, n, matrix) = match name {
        "vec2" => (Scalar::Float, 2, false),
        "vec3" => (Scalar::Float, 3, false),
        "vec4" => (Scalar::Float, 4, false),
        "ivec2" => (Scalar::Int, 2, false),
        "ivec3" => (Scalar::Int, 3, false),
        "ivec4" => (Scalar::Int, 4, false),
        "dvec2" => (Scalar::Double, 2, false),
        "dvec3" => (Scalar::Double, 3, false),
        "dvec4" => (Scalar::Double, 4, false),
        "lvec2" => (Scalar::Long, 2, false),
        "lvec3" => (Scalar::Long, 3, false),
        "mat2" => (Scalar::Float, 2, true),
        "mat3" => (Scalar::Float, 3, true),
        "mat4" => (Scalar::Float, 4, true),
        "dmat2" => (Scalar::Double, 2, true),
        "dmat3" => (Scalar::Double, 3, true),
        "dmat4" => (Scalar::Double, 4, true),
        _ => return None,
    };
    let kind = match scalar {
        Scalar::Float => Type::Float,
        Scalar::Int => Type::Int,
        Scalar::Double => Type::Double,
        _ => Type::Long,
    };
    // An integer constructor takes integers only.
    let number = |value: &Value| -> Option<Number> {
        let [number] = value.numbers()[..] else {
            return None;
        };
        let integral = matches!(number, Number::Int(_) | Number::Long(_));
        if matches!(scalar, Scalar::Int | Scalar::Long) && !integral {
            return None;
        }
        number.as_kind(scalar)
    };
    let numbers: Vec<Number> = if !matrix {
        if args.len() != n {
            return None;
        }
        args.iter().map(number).collect::<Option<_>>()?
    } else if let [diagonal] = args {
        let d = number(diagonal)?;
        let zero = Number::Int(0).as_kind(scalar)?;
        (0..n * n)
            .map(|k| if k % (n + 1) == 0 { d } else { zero })
            .collect()
    } else {
        if args.len() != n {
            return None;
        }
        let column = Type::vector(kind, n as u8)?;
        let mut entries = Vec::with_capacity(n * n);
        for part in args {
            if part.ty() != Some(column) {
                return None;
            }
            entries.extend(part.numbers());
        }
        entries
    };
    let ty = if matrix {
        column_matrix(kind, n)?
    } else {
        Type::vector(kind, n as u8)?
    };
    let mut words = Vec::new();
    for number in numbers {
        number.push_words(&mut words);
    }
    Some(orla_ir::template(ty, &[]).rebuilt(&mut words.into_iter()))
}

/// The matrix of order `n` of `kind`.
fn column_matrix(kind: Type, n: usize) -> Option<Type> {
    Type::vector(kind, n as u8)?.matrix_of()
}

/// The builtins a function cannot be named after: the Field, memory,
/// context and construction ones, whose forms the language owns. A
/// function named after a math builtin stands for it in its document.
pub(super) const RESERVED_BUILTINS: [&str; 46] = [
    "put",
    "emit",
    "sample",
    "at",
    "fetch",
    "at_linear",
    "count",
    "size",
    "isize",
    "element",
    "coord",
    "icoord",
    "domain",
    "idomain",
    "time",
    "rate",
    "window",
    "scale",
    "pointer",
    "pointer_down",
    "button",
    "wheel",
    "stick",
    "key",
    "index",
    "vec2",
    "vec3",
    "vec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "mat2",
    "mat3",
    "mat4",
    "dvec2",
    "dvec3",
    "dvec4",
    "lvec2",
    "lvec3",
    "dmat2",
    "dmat3",
    "dmat4",
    "float",
    "int",
    "double",
    "long",
];

/// The letters that name components, two ways.
pub(super) fn component_index(letter: char) -> Option<(u8, bool)> {
    match letter {
        'x' => Some((0, false)),
        'y' => Some((1, false)),
        'z' => Some((2, false)),
        'w' => Some((3, false)),
        'r' => Some((0, true)),
        'g' => Some((1, true)),
        'b' => Some((2, true)),
        'a' => Some((3, true)),
        _ => None,
    }
}

/// Whether a type is numbers all the way down: floats and ints, no Bool.
pub(super) fn numeric(ty: &Type, structs: &[StructDef]) -> bool {
    match ty {
        Type::Struct(index) => structs[*index as usize]
            .fields
            .iter()
            .all(|field| numeric(&field.ty, structs)),
        other => other.scalar() != Some(Type::Bool),
    }
}

/// Whether a type holds a float anywhere, making a `sum` depend on its
/// fold order.
pub(super) fn holds_float(ty: &Type, structs: &[StructDef]) -> bool {
    match ty {
        Type::Struct(index) => structs[*index as usize]
            .fields
            .iter()
            .any(|field| holds_float(&field.ty, structs)),
        other => other.is_float_kind(),
    }
}

/// The port type a written name stands for. `Image` and aliases spell a
/// shape and an element; checks compare the structure.
pub(super) fn port_type(
    text: &Spanned<String>,
    structs: &Structs,
    aliases: &[(String, PortType)],
) -> Result<PortType, CompileError> {
    let name = text.value.as_str();
    if name == "Image" {
        return Ok(PortType::image());
    }
    if let Some((_, ty)) = aliases.iter().find(|(alias, _)| alias == name) {
        return Ok(ty.clone());
    }
    let Some((shape, element)) = name
        .split_once('<')
        .and_then(|(shape, rest)| rest.strip_suffix('>').map(|element| (shape, element)))
    else {
        return Err(semantic(
            text.span,
            format!(
                "`{name}` is not a port type; use `Image`, `Rows<Element>`, `Grid<Element>`, `Volume<Element>`, or a declared `type`"
            ),
        ));
    };
    let shape = match shape {
        "Rows" => Shape::Rows,
        "Grid" => Shape::Grid,
        "Volume" => Shape::Volume,
        "Field" => {
            return Err(semantic(
                text.span,
                "`Rows<...>` names the concept; spell the shape: `Rows<Element>`, `Grid<Element>`, or `Volume<Element>`",
            ));
        }
        other => {
            return Err(semantic(
                text.span,
                format!("`{other}` is not a shape; use `Rows`, `Grid`, or `Volume`"),
            ));
        }
    };
    // `Element as encoding`: the element a Kernel sees, and how the Field
    // stores its scalars.
    let (element, kept) = match element.split_once(" as ") {
        Some((element, kept)) => (element, Some(kept)),
        None => (element, None),
    };
    let element = resolve_type(
        &Spanned {
            value: element.to_owned(),
            span: text.span,
        },
        structs,
    )?;
    let schema = schema_of(element, &structs.borrow()).ok_or_else(|| {
        semantic(
            text.span,
            format!(
                "`{}` cannot be an element type",
                describe(element, &structs.borrow())
            ),
        )
    })?;
    let schema = match kept {
        None => schema,
        Some(kept) => {
            let Some(encoding) = orla_ir::Encoding::parse(kept) else {
                let names: Vec<&str> = orla_ir::Encoding::NARROW.iter().map(|e| e.name()).collect();
                return Err(semantic(
                    text.span,
                    format!(
                        "`{kept}` is not an encoding; a Field keeps scalars as {}",
                        names.join(", ")
                    ),
                ));
            };
            schema.with_encoding(encoding).map_err(|why| {
                semantic(
                    text.span,
                    format!(
                        "`{}` cannot be kept as `{kept}`: {why}",
                        describe(element, &structs.borrow())
                    ),
                )
            })?
        }
    };
    Ok(PortType {
        element,
        shape,
        schema,
    })
}

/// Resolves the Kernel's `type` aliases in order; an alias names a shape
/// form directly, never another alias.
pub(super) fn aliases(
    kernel: &Kernel,
    structs: &Structs,
) -> Result<Vec<(String, PortType)>, CompileError> {
    let mut aliases: Vec<(String, PortType)> = Vec::new();
    for decl in &kernel.types {
        let name = &decl.name.value;
        // An enum is a value's type, among the structs.
        if decl.variants().is_some() {
            continue;
        }
        if structs.variants(name).is_some() {
            return Err(semantic(
                decl.name.span,
                format!("type `{name}` is defined twice"),
            ));
        }
        if name == "Image"
            || Type::from_str(name).is_ok()
            || structs.borrow().iter().any(|s| &s.name == name)
        {
            return Err(semantic(
                decl.name.span,
                format!("`{name}` is already a type name"),
            ));
        }
        if aliases.iter().any(|(existing, _)| existing == name) {
            return Err(semantic(
                decl.name.span,
                format!("type `{name}` is defined twice"),
            ));
        }
        if aliases
            .iter()
            .any(|(existing, _)| *existing == decl.target.value)
        {
            return Err(semantic(
                decl.target.span,
                "an alias names a shape directly, not another alias",
            ));
        }
        let ty = port_type(&decl.target, structs, &[])?;
        aliases.push((name.clone(), ty));
    }
    Ok(aliases)
}

pub(super) fn manifest(
    kernel: &Kernel,
    given: Option<&KernelId>,
    structs: &Structs,
    aliases: &[(String, PortType)],
) -> Result<Manifest, CompileError> {
    let id = match (&kernel.id, given) {
        (Some(written), _) => KernelId::from_str(&written.value)
            .map_err(|error| semantic(written.span, error.to_string()))?,
        (None, Some(given)) => given.clone(),
        (None, None) => {
            return Err(semantic(
                Span::default(),
                "an anonymous kernel can only live inside a node",
            ));
        }
    };
    let requires: orla_ir::Capabilities = kernel
        .requires
        .iter()
        .map(|name| {
            Capability::from_str(&name.value).map_err(|error| {
                let hint = match name.value.as_str() {
                    "field.read" => "; an image is a Field, so this is `field.read`",
                    "field.write" => "; an image is a Field, so this is `field.write`",
                    _ => "",
                };
                semantic(name.span, format!("{error}{hint}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let ports = |decls: &[orla_language::PortDecl]| {
        decls
            .iter()
            .map(|decl| {
                Ok(Port {
                    name: decl.name.value.clone(),
                    ty: port_type(&decl.ty, structs, aliases)?,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()
    };
    let mut inputs = ports(&kernel.inputs)?;
    // A reduction's output may be written as its element type:
    // `output result: Loss;` is `Rows<Loss>`.
    let mut outputs = kernel
        .outputs
        .iter()
        .map(|decl| {
            let ty = match port_type(&decl.ty, structs, aliases) {
                Ok(ty) => ty,
                Err(error) if kernel.merge.is_some() && !decl.ty.value.contains('<') => {
                    let rows = Spanned {
                        value: format!("Rows<{}>", decl.ty.value),
                        span: decl.ty.span,
                    };
                    port_type(&rows, structs, aliases).map_err(|_| error)?
                }
                Err(error) => return Err(error),
            };
            Ok(Port {
                name: decl.name.value.clone(),
                ty,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    // A memory is bound like an input and read like an output: it joins
    // both lists, last.
    let memory_ports = ports(&kernel.memories)?;
    let memories: Vec<String> = memory_ports.iter().map(|p| p.name.clone()).collect();
    if let Some(first) = kernel.memories.first() {
        if kernel.carry.is_none() {
            return Err(semantic(
                first.name.span,
                "a memory is written step by step, which needs ordered work: use `scan`",
            ));
        }
        if inputs.is_empty() {
            return Err(semantic(
                first.name.span,
                "a scan runs over its first input, which cannot be a memory; declare an input",
            ));
        }
    }
    inputs.extend(memory_ports.iter().cloned());
    outputs.extend(memory_ports);
    if let (Some(merge), Some(_)) = (&kernel.merge, &kernel.carry) {
        return Err(semantic(
            merge.span,
            "a kernel is a reduction (`merge`) or a scan (`scan`), not both",
        ));
    }
    if let (Some(merge), Some(_)) = (&kernel.merge, &kernel.scatter) {
        return Err(semantic(
            merge.span,
            "a kernel is a reduction (`merge`) or a scatter (`scatter`), not both",
        ));
    }
    let mode = if let Some(scatter) = &kernel.scatter {
        let Some(base) = inputs
            .iter()
            .position(|input| input.name == scatter.into.value)
        else {
            return Err(semantic(
                scatter.into.span,
                format!(
                    "`{}` is not an input; a scatter starts from an input port",
                    scatter.into.value
                ),
            ));
        };
        let Some(combine) = Combine::parse(&scatter.combine.value) else {
            return Err(semantic(
                scatter.combine.span,
                format!(
                    "`{}` is not a combine rule; scatter with `sum`, `min`, `max`, or `last`",
                    scatter.combine.value
                ),
            ));
        };
        match outputs.first() {
            Some(port) if outputs.len() == 1 && port.ty == inputs[base].ty => {}
            _ => {
                return Err(semantic(
                    scatter.span,
                    format!(
                        "a scatter produces one output shaped and laid out like `{}`, the input it scatters into",
                        scatter.into.value
                    ),
                ));
            }
        }
        if combine != Combine::Last && !numeric(&inputs[base].element(), &structs.borrow()) {
            return Err(semantic(
                scatter.combine.span,
                format!(
                    "`{}` combines numbers, and the element holds a Bool; scatter with `last`",
                    combine.name()
                ),
            ));
        }
        Mode::Scatter {
            base: base as u32,
            combine,
        }
    } else if let Some(prefix) = &kernel.prefix {
        let Some(combine) = Combine::parse(&prefix.combine.value) else {
            return Err(semantic(
                prefix.combine.span,
                format!(
                    "`{}` is not a combine rule; a prefix folds with `sum`, `min`, or `max`",
                    prefix.combine.value
                ),
            ));
        };
        if combine == Combine::Last {
            return Err(semantic(
                prefix.combine.span,
                "a prefix with `last` is every element's own value; write a `compute` block",
            ));
        }
        match (inputs.first(), outputs.first()) {
            (Some(first), Some(port))
                if outputs.len() == 1
                    && first.shape() == Shape::Rows
                    && port.shape() == Shape::Rows => {}
            _ => {
                return Err(semantic(
                    prefix.span,
                    "a prefix runs over a `Rows<T>` input, its first, and produces one `Rows` output of the same count",
                ));
            }
        }
        let element = outputs[0].element();
        let structs = structs.borrow();
        if !numeric(&element, &structs) {
            return Err(semantic(
                prefix.combine.span,
                format!(
                    "`{}` combines numbers, and the element holds a Bool",
                    combine.name()
                ),
            ));
        }
        // A prefix's fold is order-free: every backend groups it its own
        // way and matches the reference's words.
        if combine == Combine::Sum && holds_float(&element, &structs) {
            return Err(semantic(
                prefix.combine.span,
                "a prefix folds in whatever grouping a backend finds cheapest, and a float `sum` does not give one answer that way; walk it with `scan` instead",
            ));
        }
        drop(structs);
        Mode::Prefix { combine }
    } else if let Some(carry) = &kernel.carry {
        match (inputs.first(), outputs.first()) {
            (Some(first), Some(port))
                if first.shape() == Shape::Rows && port.shape() == Shape::Rows => {}
            _ => {
                return Err(semantic(
                    carry.span,
                    "a scan runs over a `Rows<T>` input, its first, and produces a `Rows` output of the same count",
                ));
            }
        }
        Mode::Scan
    } else if let Some(merge) = &kernel.merge {
        if inputs.is_empty() {
            return Err(semantic(
                merge.span,
                "a reduction maps over its first input; declare one",
            ));
        }
        match outputs.first() {
            Some(port) if port.shape() == Shape::Rows => {}
            _ => {
                return Err(semantic(
                    merge.span,
                    "a reduction produces one element: its output must be `Rows<T>` (the node needs no `domain`)",
                ));
            }
        }
        Mode::Reduce
    } else {
        Mode::Map
    };
    if let Some(decl) = kernel.outputs.first()
        && !requires.contains(Capability::FieldWrite)
    {
        return Err(semantic(
            decl.name.span,
            "writing an output needs `field.write`; add it to `requires`",
        ));
    }
    // Only a map or a scan writes a narrow output, an element at a time;
    // the other modes fold in place.
    if !matches!(mode, Mode::Map | Mode::Scan)
        && let Some((decl, _)) = kernel
            .outputs
            .iter()
            .zip(&outputs)
            .find(|(_, port)| !port.ty.schema.is_wide())
    {
        return Err(semantic(
            decl.ty.span,
            format!(
                "a {} folds into its output in place, in whole words; keep `{}` wide, and narrow it in a map after",
                mode.name(),
                decl.name.value
            ),
        ));
    }
    // Only a map has several outputs, each its own Field over one domain;
    // other modes give several results as one struct.
    if let Some(second) = kernel.outputs.get(1) {
        if mode != Mode::Map {
            let (what, form) = match mode {
                Mode::Reduce => ("a reduction folds one element", "its output's element"),
                Mode::Scan => ("a scan carries one value", "the carried value"),
                Mode::Prefix { .. } => ("a prefix folds one element", "its output's element"),
                _ => ("a scatter writes into one target", "the target's element"),
            };
            return Err(semantic(
                second.name.span,
                format!(
                    "{what}, so it has one output; several results belong in one struct as {form}, or in a map, which may write several outputs"
                ),
            ));
        }
        let shape = outputs[0].shape();
        for (decl, port) in kernel.outputs.iter().zip(&outputs).skip(1) {
            if port.shape() != shape {
                return Err(semantic(
                    decl.ty.span,
                    format!(
                        "a map writes all its outputs over one domain, so they share a shape; `{}` is a `{}` and `{}` a `{}`",
                        outputs[0].name,
                        shape.name(),
                        port.name,
                        port.shape().name()
                    ),
                ));
            }
        }
    }
    let params = kernel
        .params
        .iter()
        .map(|decl| {
            // Any Kernel value type crosses as its words.
            let ty = resolve_type(&decl.ty, structs)?;
            let default = constant(
                &with_variants(&decl.default, &|name| structs.variants(name))?,
                ty,
                &structs.borrow(),
            )?;
            Ok::<_, CompileError>(Param {
                name: decl.name.value.clone(),
                ty,
                default,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = std::collections::BTreeSet::new();
    let names = kernel
        .inputs
        .iter()
        .map(|d| &d.name)
        .chain(kernel.params.iter().map(|d| &d.name))
        .chain(kernel.outputs.iter().map(|d| &d.name))
        .chain(kernel.memories.iter().map(|d| &d.name));
    for name in names {
        if !seen.insert(&name.value) {
            return Err(semantic(
                name.span,
                format!("`{}` is declared twice", name.value),
            ));
        }
    }
    let budget = match &kernel.budget {
        None => orla_ir::DEFAULT_BUDGET,
        Some(declared) if (1..=orla_ir::MAX_BUDGET).contains(&declared.value) => declared.value,
        Some(declared) => {
            return Err(semantic(
                declared.span,
                format!(
                    "a Kernel's budget is 1 to {} loop iterations an element, found {}",
                    orla_ir::MAX_BUDGET,
                    declared.value
                ),
            ));
        }
    };
    Ok(Manifest {
        id,
        mode,
        requires,
        inputs,
        params,
        outputs,
        memories,
        budget,
    })
}

/// A literal read exact and shaped into the type `ty` names, over the
/// structs `decls` declare; a double keeps every digit written.
pub(crate) fn literal_of_type(
    expr: &Expr,
    ty: &Spanned<String>,
    decls: &[orla_language::StructDecl],
) -> Result<Value, CompileError> {
    let table = struct_table(decls)?;
    let ty = resolve_type(ty, &table)?;
    let structs = table.into_inner();
    constant(expr, ty, &structs)
}

/// Evaluates a parameter default at compile time as a value of `ty`,
/// shaped into its layout; an integer may stand for a float.
pub(super) fn constant(
    expr: &Expr,
    ty: Type,
    structs: &[StructDef],
) -> Result<Value, CompileError> {
    let fields_of = |name: &str| {
        structs
            .iter()
            .find(|def| def.name == name && !def.is_array())
            .map(|def| def.fields.iter().map(|field| field.name.clone()).collect())
    };
    let value = literal_exact(expr, &fields_of)?;
    orla_ir::value_as(&value, ty, structs).ok_or_else(|| {
        let value = literal_in(expr, &fields_of).unwrap_or(value);
        semantic(expr.span, {
            let name = describe(ty, structs);
            let article = if name.starts_with(['A', 'E', 'I', 'O', 'U']) {
                "an"
            } else {
                "a"
            };
            format!("{article} {name} default must be a literal of that type; `{value}` is not")
        })
    })
}

/// `expr` with each enum variant it names (`Phase.Walk`) written as its
/// `Int`, as a literal takes it; `variants_of` gives an enum's variants by
/// its name.
pub(crate) fn with_variants(
    expr: &Expr,
    variants_of: &dyn Fn(&str) -> Option<crate::Variants>,
) -> Result<Expr, CompileError> {
    let each = |exprs: &[Expr]| -> Result<Vec<Expr>, CompileError> {
        exprs
            .iter()
            .map(|expr| with_variants(expr, variants_of))
            .collect()
    };
    let kind = match &expr.kind {
        ExprKind::Member(base, variant) => match &base.kind {
            ExprKind::Name(name) => match variants_of(name) {
                Some(variants) => ExprKind::Int(i64::from(super::variant_in(
                    &variants, name, variant, expr.span,
                )?)),
                None => expr.kind.clone(),
            },
            _ => expr.kind.clone(),
        },
        ExprKind::Neg(inner) => ExprKind::Neg(Box::new(with_variants(inner, variants_of)?)),
        ExprKind::Array(elements) => ExprKind::Array(each(elements)?),
        ExprKind::Repeat(element, count) => ExprKind::Repeat(
            Box::new(with_variants(element, variants_of)?),
            count.clone(),
        ),
        ExprKind::Call(name, args) => ExprKind::Call(name.clone(), each(args)?),
        ExprKind::Struct { name, base, fields } => ExprKind::Struct {
            name: name.clone(),
            base: match base {
                Some(base) => Some(Box::new(with_variants(base, variants_of)?)),
                None => None,
            },
            fields: fields
                .iter()
                .map(|(field, value)| Ok((field.clone(), with_variants(value, variants_of)?)))
                .collect::<Result<_, CompileError>>()?,
        },
        other => other.clone(),
    };
    Ok(Expr {
        kind,
        span: expr.span,
    })
}
