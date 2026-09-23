//! What a Kernel declares, checked and turned into its manifest: the
//! structs, the type aliases, the ports and their layouts, the
//! parameters and their literal defaults, the execution mode; and the
//! names a function may not take.

use std::cell::RefCell;
use std::str::FromStr;

use orla_ir::{
    Capability, Combine, Field, KernelId, Manifest, Mode, Param, Port, PortType, Shape, StructDef,
    Type, Value, schema_of,
};
use orla_language::{Expr, ExprKind, Kernel, Span, Spanned};

use crate::CompileError;

use super::{Structs, describe, parse_type, resolve_type, semantic};

pub(super) fn structs(kernel: &Kernel) -> Result<Structs, CompileError> {
    let structs: Structs = RefCell::new(Vec::new());
    for decl in &kernel.structs {
        if structs
            .borrow()
            .iter()
            .any(|def| def.name == decl.name.value)
        {
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
        structs
            .borrow_mut()
            .push(StructDef::record(decl.name.value.clone(), fields));
    }
    Ok(structs)
}

/// Evaluates a literal without a declared type: integers stay `Int`.
pub(crate) fn literal(expr: &Expr) -> Result<Value, CompileError> {
    match &expr.kind {
        ExprKind::Float(value) => Ok(Value::Float(*value)),
        ExprKind::Int(value) => Ok(Value::Int(*value)),
        ExprKind::Bool(value) => Ok(Value::Bool(*value)),
        ExprKind::Neg(inner) => match literal(inner)? {
            Value::Float(v) => Ok(Value::Float(-v)),
            Value::Int(v) => v
                .checked_neg()
                .map(Value::Int)
                .ok_or_else(|| semantic(expr.span, "integer literal overflows when negated")),
            Value::Bool(_) => Err(semantic(expr.span, "cannot negate a Bool")),
            Value::Vec2([x, y]) => Ok(Value::Vec2([-x, -y])),
            Value::Vec3([x, y, z]) => Ok(Value::Vec3([-x, -y, -z])),
            Value::Vec4([x, y, z, w]) => Ok(Value::Vec4([-x, -y, -z, -w])),
        },
        ExprKind::Call(name, _) => {
            let ty = match name.as_str() {
                "vec2" => Type::Vec2,
                "vec3" => Type::Vec3,
                "vec4" => Type::Vec4,
                _ => {
                    return Err(semantic(
                        expr.span,
                        format!(
                            "`{name}` is not a literal; only numbers, booleans, and vec2/vec3/vec4 are"
                        ),
                    ));
                }
            };
            constant(expr, ty)
        }
        _ => Err(semantic(
            expr.span,
            "expected a literal: a number, `true`/`false`, or vec2/vec3/vec4",
        )),
    }
}

/// The builtins a function cannot be named after: the Field, memory,
/// context and construction ones, whose forms the language owns. The
/// math builtins are ordinary calls, and a function of one's name
/// stands for it in its document.
pub(super) const RESERVED_BUILTINS: [&str; 34] = [
    "put",
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
    "mat2",
    "mat3",
    "mat4",
    "float",
    "int",
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
        Type::Float
        | Type::Int
        | Type::Vec2
        | Type::Vec3
        | Type::Vec4
        | Type::IVec2
        | Type::IVec3
        | Type::Mat2
        | Type::Mat3
        | Type::Mat4 => true,
        Type::Bool => false,
        Type::Struct(index) => structs[*index as usize]
            .fields
            .iter()
            .all(|field| numeric(&field.ty, structs)),
    }
}

/// Whether a type holds a float anywhere: what decides that a `sum`
/// depends on the order it is folded in.
pub(super) fn holds_float(ty: &Type, structs: &[StructDef]) -> bool {
    match ty {
        Type::Float
        | Type::Vec2
        | Type::Vec3
        | Type::Vec4
        | Type::Mat2
        | Type::Mat3
        | Type::Mat4 => true,
        Type::Int | Type::IVec2 | Type::IVec3 | Type::Bool => false,
        Type::Struct(index) => structs[*index as usize]
            .fields
            .iter()
            .any(|field| holds_float(&field.ty, structs)),
    }
}

/// The port type a written name stands for. `Image` and the declared
/// aliases are spellings of a shape and an element; the structure is what
/// gets checked.
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
    // A reduction yields one element, so its output may be written as the
    // element's type: `output result: Loss;` is `Rows<Loss>`.
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
    // both lists, last, and its name says so.
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
        // What lets every backend group the fold its own way and still
        // answer with the reference's words.
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
    // Several outputs are what a map writes per element, each to a Field
    // of its own over the one domain. The other modes fold, carry or
    // place one element; several results there are one struct.
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
            if kernel
                .structs
                .iter()
                .any(|def| def.name.value == decl.ty.value)
                || decl.ty.value.starts_with('[')
            {
                return Err(semantic(
                    decl.ty.span,
                    "parameters cross the ABI and cannot be structs or arrays; use plain values",
                ));
            }
            let ty = parse_type(&decl.ty)?;
            let default = constant(&decl.default, ty)?;
            Ok::<_, CompileError>(Param {
                name: decl.name.value.clone(),
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
    Ok(Manifest {
        id,
        mode,
        requires,
        inputs,
        params,
        outputs,
        memories,
    })
}

/// Evaluates a parameter default at compile time.
pub(super) fn constant(expr: &Expr, ty: Type) -> Result<Value, CompileError> {
    let value = match (&expr.kind, ty) {
        (ExprKind::Float(value), Type::Float) => Value::Float(*value),
        (ExprKind::Int(value), Type::Float) => Value::Float(*value as f32),
        (ExprKind::Int(value), Type::Int) => Value::Int(*value),
        (ExprKind::Bool(value), Type::Bool) => Value::Bool(*value),
        (ExprKind::Neg(inner), _) => match constant(inner, ty)? {
            Value::Float(v) => Value::Float(-v),
            Value::Int(v) => Value::Int(
                v.checked_neg()
                    .ok_or_else(|| semantic(expr.span, "integer literal overflows when negated"))?,
            ),
            Value::Bool(_) => return Err(semantic(expr.span, "cannot negate a Bool")),
            Value::Vec2([x, y]) => Value::Vec2([-x, -y]),
            Value::Vec3([x, y, z]) => Value::Vec3([-x, -y, -z]),
            Value::Vec4([x, y, z, w]) => Value::Vec4([-x, -y, -z, -w]),
        },
        (ExprKind::Call(name, args), Type::Vec2 | Type::Vec3 | Type::Vec4)
            if constructor(ty) == Some(name.as_str()) =>
        {
            let count = ty.components().unwrap_or(0) as usize;
            if args.len() != count {
                return Err(semantic(
                    expr.span,
                    format!("`{name}` takes {count} components, found {}", args.len()),
                ));
            }
            let mut parts = [0.0; 4];
            for (slot, arg) in parts.iter_mut().zip(args) {
                let Value::Float(component) = constant(arg, Type::Float)? else {
                    unreachable!()
                };
                *slot = component;
            }
            match ty {
                Type::Vec2 => Value::Vec2([parts[0], parts[1]]),
                Type::Vec3 => Value::Vec3([parts[0], parts[1], parts[2]]),
                _ => Value::Vec4(parts),
            }
        }
        _ => {
            return Err(semantic(
                expr.span,
                format!("a {ty} default must be a literal of that type"),
            ));
        }
    };
    Ok(value)
}

/// The builtin that constructs a vector type.
pub(super) fn constructor(ty: Type) -> Option<&'static str> {
    match ty {
        Type::Vec2 => Some("vec2"),
        Type::Vec3 => Some("vec3"),
        Type::Vec4 => Some("vec4"),
        _ => None,
    }
}
