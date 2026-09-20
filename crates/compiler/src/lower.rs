//! Semantic analysis and lowering from AST to IR.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::str::FromStr;

use orla_ir::{
    BinaryOp, Block as IrBlock, Capability, Combine, Field, Function, Instruction, Intrinsic,
    KernelId, MAX_ARRAY_COUNT, MAX_LOOP_COUNT, Manifest, Mode, Module, Op, Param, Port, PortType,
    Shape, ShapeIndex, StructDef, Type, UnaryOp, Value, ValueId, schema_of,
};
use orla_language::{Block, Expr, ExprKind, FnDecl, Kernel, Span, Spanned};

use crate::CompileError;

fn semantic(span: Span, message: impl Into<String>) -> CompileError {
    CompileError::Semantic {
        span,
        message: message.into(),
    }
}

fn parse_type(name: &Spanned<String>) -> Result<Type, CompileError> {
    Type::from_str(&name.value).map_err(|error| semantic(name.span, error.to_string()))
}

/// The struct table while a Kernel lowers: the declared structs first,
/// then every array type met, `[T; N]` interned once by element and
/// count so two spellings of it are one type.
type Structs = RefCell<Vec<StructDef>>;

/// Resolves a type name against the plain types, the struct table and
/// the array spelling `[Type; N]`, whose type is interned.
fn resolve_type(name: &Spanned<String>, structs: &Structs) -> Result<Type, CompileError> {
    if let Some(inner) = name.value.strip_prefix('[') {
        let Some((element, count)) = inner
            .strip_suffix(']')
            .and_then(|inner| inner.rsplit_once(';'))
        else {
            return Err(semantic(
                name.span,
                format!(
                    "`{}` is not a type; an array is `[Type; count]`",
                    name.value
                ),
            ));
        };
        let element = resolve_type(
            &Spanned {
                value: element.trim().to_owned(),
                span: name.span,
            },
            structs,
        )?;
        let count: u32 = count.trim().parse().map_err(|_| {
            semantic(
                name.span,
                format!(
                    "`{}` is not a type; an array is `[Type; count]`",
                    name.value
                ),
            )
        })?;
        return array_type(structs, element, count, name.span);
    }
    let found = structs
        .borrow()
        .iter()
        .position(|def| def.name == name.value);
    if let Some(index) = found {
        return Ok(Type::Struct(index as u32));
    }
    parse_type(name)
}

/// The array type `[element; count]`, interned in the struct table.
fn array_type(
    structs: &Structs,
    element: Type,
    count: u32,
    span: Span,
) -> Result<Type, CompileError> {
    if count == 0 || count > MAX_ARRAY_COUNT {
        return Err(semantic(
            span,
            format!("an array has 1 to {MAX_ARRAY_COUNT} elements, not {count}"),
        ));
    }
    let existing = structs
        .borrow()
        .iter()
        .position(|def| def.element == Some(element) && def.fields.len() == count as usize);
    if let Some(index) = existing {
        return Ok(Type::Struct(index as u32));
    }
    let name = describe(element, &structs.borrow());
    let mut table = structs.borrow_mut();
    table.push(StructDef::array(element, &name, count));
    Ok(Type::Struct((table.len() - 1) as u32))
}

/// What `[index]` takes from a type, and how many there are: an array's
/// element, a matrix's column, a vector's component.
fn subscript_element(ty: Type, structs: &[StructDef]) -> Option<(Type, usize)> {
    match ty {
        Type::Struct(index) => {
            let def = &structs[index as usize];
            def.element.map(|element| (element, def.fields.len()))
        }
        _ => ty
            .column()
            .map(|column| (column, ty.order().unwrap_or(0) as usize))
            .or_else(|| ty.components().map(|n| (Type::Float, n as usize))),
    }
}

/// Human-readable type for diagnostics, naming structs.
fn describe(ty: Type, structs: &[StructDef]) -> String {
    match ty {
        Type::Struct(index) => structs
            .get(index as usize)
            .map(|def| def.name.clone())
            .unwrap_or_else(|| ty.to_string()),
        other => other.to_string(),
    }
}

fn structs(kernel: &Kernel) -> Result<Structs, CompileError> {
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

pub(crate) fn lower(kernel: &Kernel, id: Option<&KernelId>) -> Result<Module, CompileError> {
    let structs = structs(kernel)?;
    let aliases = aliases(kernel, &structs)?;
    let manifest = manifest(kernel, id, &structs, &aliases)?;
    // Functions: one with a Field parameter is a template, instantiated
    // per port it is called with; the others lower in order, so a
    // function calls only what is defined above it.
    let shared = Shared {
        manifest: &manifest,
        structs: &structs,
        aliases: &aliases,
        functions: RefCell::new(Vec::new()),
        templates: Vec::new(),
        instances: RefCell::new(BTreeMap::new()),
        instantiating: RefCell::new(Vec::new()),
    };
    let mut shared = shared;
    for declaration in &kernel.functions {
        let taken = shared
            .functions
            .borrow()
            .iter()
            .any(|(name, _)| *name == declaration.name.value)
            || shared
                .templates
                .iter()
                .any(|t| t.name.value == declaration.name.value);
        if taken {
            return Err(semantic(
                declaration.name.span,
                format!("function `{}` is defined twice", declaration.name.value),
            ));
        }
        let mut has_field = false;
        for (_, ty) in &declaration.args {
            if resolve_type(ty, &structs).is_err() {
                port_type(ty, &structs, &aliases)
                    .map_err(|_| semantic(ty.span, format!("`{}` is not a type", ty.value)))?;
                has_field = true;
            }
        }
        if has_field {
            shared.templates.push(declaration);
            continue;
        }
        let function = lower_function(&shared, declaration, Vec::new(), &declaration.name.value)?;
        shared
            .functions
            .borrow_mut()
            .push((declaration.name.value.clone(), function));
    }
    let expected = match manifest.outputs.first() {
        Some(port) => port.element(),
        None => {
            return Err(semantic(
                kernel.id.as_ref().map(|id| id.span).unwrap_or_default(),
                "kernel declares no output; add `output result: Image;`",
            ));
        }
    };
    let entry = {
        let mut builder = Builder::new(&shared, true, "compute", Vec::new());
        if let Some(carry) = &kernel.carry {
            builder.locals.push((
                carry.name.value.clone(),
                Local::Carried {
                    depth: 0,
                    ty: expected,
                },
            ));
        }
        let result = builder.block(&kernel.compute)?;
        let actual = builder.ty(result);
        if let Mode::Scatter { base, .. } = manifest.mode {
            // The body yields a pair: an address into the target and a
            // value of its element type, as a struct of two fields.
            let target = &manifest.inputs[base as usize];
            let address = target.ty.index_type();
            let table = structs.borrow();
            let fields = match actual {
                Type::Struct(index) if !table[index as usize].is_array() => {
                    Some(&table[index as usize].fields)
                }
                _ => None,
            };
            let well_formed = fields.is_some_and(|fields| {
                fields.len() == 2 && fields[0].ty == address && fields[1].ty == expected
            });
            if !well_formed {
                return Err(semantic(
                    kernel.compute.result.span,
                    format!(
                        "a scatter into `{}` gives a struct of two fields per element, an address ({}) and a value ({}), but the body gives {}",
                        target.name,
                        describe(address, &table),
                        describe(expected, &table),
                        describe(actual, &table)
                    ),
                ));
            }
        } else if actual != expected {
            return Err(semantic(
                kernel.compute.result.span,
                format!(
                    "output `{}` needs a {} per element, but the body gives {}",
                    manifest.outputs[0].name,
                    describe(expected, &structs.borrow()),
                    describe(actual, &structs.borrow())
                ),
            ));
        }
        builder.finish(vec![result])
    };
    let merge = match &kernel.merge {
        None => None,
        Some(decl) => {
            let mut builder = Builder::new(&shared, false, "merge", Vec::new());
            if decl.a.value == decl.b.value {
                return Err(semantic(
                    decl.b.span,
                    format!("`merge` names both values `{}`", decl.a.value),
                ));
            }
            builder.args.push((decl.a.value.clone(), expected));
            builder.args.push((decl.b.value.clone(), expected));
            let result = builder.block(&decl.body)?;
            let actual = builder.ty(result);
            if actual != expected {
                return Err(semantic(
                    decl.body.result.span,
                    format!(
                        "`merge` must give a {} like `compute`, but the body gives {}",
                        describe(expected, &structs.borrow()),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            Some(builder.finish(vec![result]))
        }
    };
    let init = match &kernel.carry {
        None => None,
        Some(carry) => {
            let mut builder = Builder::new(&shared, false, "init", Vec::new());
            let result = builder.value(&carry.init)?;
            let actual = builder.ty(result);
            if actual != expected {
                return Err(semantic(
                    carry.init.span,
                    format!(
                        "`scan` carries a {} like its output, but the initial value is {}",
                        describe(expected, &structs.borrow()),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            Some(builder.finish(vec![result]))
        }
    };
    let Shared { functions, .. } = shared;
    let functions: Vec<Function> = functions.into_inner().into_iter().map(|(_, f)| f).collect();
    Ok(Module {
        manifest,
        structs: structs.into_inner(),
        functions,
        entry,
        merge,
        init,
    })
}

/// Whether a type is numbers all the way down: floats and ints, no Bool.
fn numeric(ty: &Type, structs: &[StructDef]) -> bool {
    match ty {
        Type::Float
        | Type::Int
        | Type::Vec2
        | Type::Vec3
        | Type::Vec4
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

/// The port type a written name stands for. `Image` and the declared
/// aliases are spellings of a shape and an element; the structure is what
/// gets checked.
fn port_type(
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
fn aliases(kernel: &Kernel, structs: &Structs) -> Result<Vec<(String, PortType)>, CompileError> {
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

fn manifest(
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
    let mut outputs = ports(&kernel.outputs)?;
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
    if kernel.outputs.len() != 1 {
        return Err(semantic(
            kernel.id.as_ref().map(|id| id.span).unwrap_or_default(),
            format!(
                "a kernel declares exactly one output in this language version, found {}",
                kernel.outputs.len()
            ),
        ));
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
fn constant(expr: &Expr, ty: Type) -> Result<Value, CompileError> {
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
fn constructor(ty: Type) -> Option<&'static str> {
    match ty {
        Type::Vec2 => Some("vec2"),
        Type::Vec3 => Some("vec3"),
        Type::Vec4 => Some("vec4"),
        _ => None,
    }
}

/// What every block of a Kernel resolves names against: its manifest,
/// structs, aliases, the functions lowered so far (a call's index), the
/// templates with Field parameters, and their instances by binding.
struct Shared<'a> {
    manifest: &'a Manifest,
    structs: &'a Structs,
    aliases: &'a [(String, PortType)],
    functions: RefCell<Vec<(String, Function)>>,
    templates: Vec<&'a FnDecl>,
    /// Instantiated templates by (name, bound input indices) to function index.
    instances: RefCell<BTreeMap<(String, Vec<u32>), u32>>,
    /// Templates being instantiated, to refuse a template calling itself.
    instantiating: RefCell<Vec<String>>,
}

/// Lowers a function body. `fields` binds its Field parameters to input
/// ports, for an instance of a template; `name` names the result.
fn lower_function(
    shared: &Shared<'_>,
    declaration: &FnDecl,
    fields: Vec<(String, u32)>,
    name: &str,
) -> Result<Function, CompileError> {
    let structs = shared.structs;
    let mut builder = Builder::new(shared, false, name, fields);
    for (name, ty) in &declaration.args {
        if builder.fields.iter().any(|(field, _)| field == &name.value) {
            continue;
        }
        let ty = resolve_type(ty, structs)?;
        if builder
            .args
            .iter()
            .any(|(existing, _)| existing == &name.value)
        {
            return Err(semantic(
                name.span,
                format!("argument `{}` is declared twice", name.value),
            ));
        }
        builder.args.push((name.value.clone(), ty));
    }
    let ret = resolve_type(&declaration.ret, structs)?;
    let result = builder.block(&declaration.body)?;
    let actual = builder.ty(result);
    if actual != ret {
        return Err(semantic(
            declaration.body.result.span,
            format!(
                "function `{}` returns {} but the body gives {}",
                declaration.name.value,
                describe(ret, &structs.borrow()),
                describe(actual, &structs.borrow())
            ),
        ));
    }
    Ok(builder.finish(vec![result]))
}

/// What a name resolves to inside a block.
#[derive(Clone)]
enum Local {
    Value(ValueId),
    /// The carried value of the loop that introduced the name.
    Carried {
        depth: usize,
        ty: Type,
    },
}

struct Builder<'a> {
    manifest: &'a Manifest,
    shared: &'a Shared<'a>,
    is_entry: bool,
    name: String,
    args: Vec<(String, Type)>,
    /// Field parameters bound to input ports, for a template's instance.
    fields: Vec<(String, u32)>,
    /// Visible bindings, innermost last.
    locals: Vec<(String, Local)>,
    /// Index into `locals` where each open block scope starts.
    scopes: Vec<usize>,
    /// Instruction buffers, one per open block; the last is current.
    buffers: Vec<Vec<Instruction>>,
    /// Type of every emitted value by id.
    types: Vec<Type>,
    /// Carried types of enclosing loops, innermost last.
    loops: Vec<Type>,
    /// Every `let`, in source order, with its value: what a trace shows.
    bindings: Vec<(ValueId, String)>,
}

impl<'a> Builder<'a> {
    fn new(shared: &'a Shared<'a>, is_entry: bool, name: &str, fields: Vec<(String, u32)>) -> Self {
        Self {
            manifest: shared.manifest,
            shared,
            is_entry,
            name: name.to_owned(),
            args: Vec::new(),
            fields,
            locals: Vec::new(),
            scopes: vec![0],
            buffers: vec![Vec::new()],
            types: Vec::new(),
            loops: Vec::new(),
            bindings: Vec::new(),
        }
    }

    fn finish(mut self, results: Vec<ValueId>) -> Function {
        let body = self.buffers.pop().unwrap_or_default();
        Function {
            name: self.name,
            args: self.args.into_iter().map(|(_, ty)| ty).collect(),
            body,
            results,
            bindings: self.bindings,
        }
    }

    /// Reserves the next value id without emitting anything yet.
    fn reserve(&mut self, ty: Type) -> ValueId {
        self.types.push(ty);
        ValueId((self.types.len() - 1) as u32)
    }

    fn place(&mut self, id: ValueId, ty: Type, op: Op) -> ValueId {
        self.types[id.0 as usize] = ty;
        self.buffers
            .last_mut()
            .expect("a block is open")
            .push(Instruction { id, ty, op });
        id
    }

    fn emit(&mut self, ty: Type, op: Op) -> ValueId {
        let id = self.reserve(ty);
        self.place(id, ty, op)
    }

    fn ty(&self, id: ValueId) -> Type {
        self.types[id.0 as usize]
    }

    fn describe(&self, ty: Type) -> String {
        describe(ty, &self.structs())
    }

    /// The struct table as it stands; a borrow tied to the Kernel, not
    /// to the builder, so a value read through it may be emitted at once.
    fn structs(&self) -> std::cell::Ref<'a, Vec<StructDef>> {
        self.shared.structs.borrow()
    }

    /// The array type `[element; count]`.
    fn array_type(&self, element: Type, count: u32, span: Span) -> Result<Type, CompileError> {
        array_type(self.shared.structs, element, count, span)
    }

    /// Lowers a block into a fresh buffer and returns it with its result.
    fn nested(&mut self, block: &Block) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let result = self.block(block);
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: result?,
        })
    }

    /// Lowers a single expression as a block (for `else if` and `until`).
    fn nested_expr(&mut self, expr: &Expr) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let result = self.value(expr);
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: result?,
        })
    }

    fn block(&mut self, block: &Block) -> Result<ValueId, CompileError> {
        self.scopes.push(self.locals.len());
        let result = (|| {
            for (name, value) in &block.lets {
                // A statement for its effect: `put(...)`, or `let _ = ...`.
                if name.value == "_" {
                    self.value(value)?;
                    continue;
                }
                let scope_start = *self.scopes.last().expect("a scope is open");
                if self.locals[scope_start..]
                    .iter()
                    .any(|(existing, _)| existing == &name.value)
                {
                    return Err(semantic(
                        name.span,
                        format!(
                            "`{}` is already bound in this block; bindings are immutable",
                            name.value
                        ),
                    ));
                }
                let id = self.value(value)?;
                if name.value != "_" {
                    self.bindings.push((id, name.value.clone()));
                }
                self.locals.push((name.value.clone(), Local::Value(id)));
            }
            self.value(&block.result)
        })();
        let start = self.scopes.pop().expect("a scope is open");
        self.locals.truncate(start);
        result
    }

    /// Lowers an expression that must produce a plain value (not an image).
    fn value(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        match &expr.kind {
            ExprKind::Float(value) => Ok(self.emit(Type::Float, Op::Const(Value::Float(*value)))),
            ExprKind::Int(value) => Ok(self.emit(Type::Int, Op::Const(Value::Int(*value)))),
            ExprKind::Bool(value) => Ok(self.emit(Type::Bool, Op::Const(Value::Bool(*value)))),
            ExprKind::Name(name) => self.name(name, expr.span),
            ExprKind::Neg(inner) => self.unary(UnaryOp::Neg, "-", inner, expr.span),
            ExprKind::Not(inner) => self.unary(UnaryOp::Not, "!", inner, expr.span),
            ExprKind::Binary(op, left, right) => {
                let op = match op {
                    orla_language::BinaryOp::Add => BinaryOp::Add,
                    orla_language::BinaryOp::Sub => BinaryOp::Sub,
                    orla_language::BinaryOp::Mul => BinaryOp::Mul,
                    orla_language::BinaryOp::Div => BinaryOp::Div,
                    orla_language::BinaryOp::Rem => BinaryOp::Rem,
                    orla_language::BinaryOp::Less => BinaryOp::Less,
                    orla_language::BinaryOp::LessEqual => BinaryOp::LessEqual,
                    orla_language::BinaryOp::Greater => BinaryOp::Greater,
                    orla_language::BinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
                    orla_language::BinaryOp::Equal => BinaryOp::Equal,
                    orla_language::BinaryOp::NotEqual => BinaryOp::NotEqual,
                    orla_language::BinaryOp::And => BinaryOp::And,
                    orla_language::BinaryOp::Or => BinaryOp::Or,
                    orla_language::BinaryOp::BitAnd => BinaryOp::BitAnd,
                    orla_language::BinaryOp::BitOr => BinaryOp::BitOr,
                    orla_language::BinaryOp::BitXor => BinaryOp::BitXor,
                    orla_language::BinaryOp::Shl => BinaryOp::Shl,
                    orla_language::BinaryOp::Shr => BinaryOp::Shr,
                    orla_language::BinaryOp::Ushr => BinaryOp::Ushr,
                };
                let left = self.value(left)?;
                let right = self.value(right)?;
                let (a, b) = (self.ty(left), self.ty(right));
                let ty = op.result_type(a, b).ok_or_else(|| {
                    semantic(
                        expr.span,
                        format!("cannot apply `{}` to {a} and {b}", op.symbol()),
                    )
                })?;
                Ok(self.emit(ty, Op::Binary(op, left, right)))
            }
            ExprKind::Member(inner, member) => {
                let inner = self.value(inner)?;
                let ty = self.ty(inner);
                if let Type::Struct(index) = ty {
                    let table = self.structs();
                    let def = &table[index as usize];
                    if def.is_array() {
                        return Err(semantic(
                            expr.span,
                            format!("`{}` is an array; take an element with `[index]`", def.name),
                        ));
                    }
                    let field = def
                        .fields
                        .iter()
                        .position(|field| &field.name == member)
                        .ok_or_else(|| {
                            semantic(expr.span, format!("`{}` has no field `{member}`", def.name))
                        })?;
                    let field_ty = def.fields[field].ty;
                    drop(table);
                    return Ok(self.emit(field_ty, Op::Extract(inner, field as u32)));
                }
                if ty.order().is_some() {
                    return Err(semantic(
                        expr.span,
                        format!("{ty} is a matrix; take a column with `[index]`"),
                    ));
                }
                let count = ty.components().ok_or_else(|| {
                    semantic(
                        expr.span,
                        format!(
                            "`.{member}` needs a vector or struct, found {}",
                            self.describe(ty)
                        ),
                    )
                })?;
                let component = match member.as_str() {
                    "x" | "r" => 0,
                    "y" | "g" => 1,
                    "z" | "b" => 2,
                    "w" | "a" => 3,
                    _ => {
                        return Err(semantic(
                            expr.span,
                            format!("`.{member}` is not a component; use x, y, z, w or r, g, b, a"),
                        ));
                    }
                };
                if component >= count {
                    return Err(semantic(
                        expr.span,
                        format!("{ty} has no component `.{member}`"),
                    ));
                }
                Ok(self.emit(Type::Float, Op::Extract(inner, u32::from(component))))
            }
            ExprKind::Array(elements) => {
                let mut ids = Vec::with_capacity(elements.len());
                for element in elements {
                    ids.push(self.value(element)?);
                }
                let first = self.ty(ids[0]);
                for (id, element) in ids.iter().zip(elements) {
                    if self.ty(*id) != first {
                        return Err(semantic(
                            element.span,
                            format!(
                                "an array's elements are all one type: this one is {}, the first {}",
                                self.describe(self.ty(*id)),
                                self.describe(first)
                            ),
                        ));
                    }
                }
                let ty = self.array_type(first, ids.len() as u32, expr.span)?;
                Ok(self.emit(ty, Op::Construct(ids)))
            }
            ExprKind::Repeat(element, count) => {
                let id = self.value(element)?;
                let ty = self.array_type(self.ty(id), count.value, count.span)?;
                Ok(self.emit(ty, Op::Construct(vec![id; count.value as usize])))
            }
            ExprKind::Subscript(array, index) => {
                let array_id = self.value(array)?;
                let array_ty = self.ty(array_id);
                let Some((element, count)) = subscript_element(array_ty, &self.structs()) else {
                    return Err(semantic(
                        array.span,
                        format!(
                            "`[index]` takes an element of an array, a column of a matrix or a component of a vector, found {}",
                            self.describe(array_ty)
                        ),
                    ));
                };
                // A literal index is a field: checked here, free to run.
                let literal = match &index.kind {
                    ExprKind::Int(value) => Some(*value),
                    ExprKind::Neg(inner) => match &inner.kind {
                        ExprKind::Int(value) => Some(value.wrapping_neg()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(literal) = literal {
                    if literal < 0 || literal as usize >= count {
                        return Err(semantic(
                            index.span,
                            format!(
                                "index {literal} is outside {}, whose indices are 0 to {}",
                                self.describe(array_ty),
                                count - 1
                            ),
                        ));
                    }
                    return Ok(self.emit(element, Op::Extract(array_id, literal as u32)));
                }
                let index_id = self.value(index)?;
                if self.ty(index_id) != Type::Int {
                    return Err(semantic(
                        index.span,
                        format!(
                            "an array takes an Int index, found {}",
                            self.describe(self.ty(index_id))
                        ),
                    ));
                }
                Ok(self.emit(element, Op::Subscript(array_id, index_id)))
            }
            ExprKind::Call(name, args) => self.call(name, args, expr.span),
            ExprKind::Struct { name, base, fields } => {
                let index = self
                    .structs()
                    .iter()
                    .position(|def| def.name == name.value)
                    .ok_or_else(|| {
                        semantic(name.span, format!("`{}` is not a struct", name.value))
                    })?;
                let ty = Type::Struct(index as u32);
                let base_id = match base {
                    Some(base) => {
                        let id = self.value(base)?;
                        if self.ty(id) != ty {
                            return Err(semantic(
                                base.span,
                                format!(
                                    "`..base` must be a `{}`, found {}",
                                    name.value,
                                    self.describe(self.ty(id))
                                ),
                            ));
                        }
                        Some(id)
                    }
                    None => None,
                };
                let field_count = self.structs()[index].fields.len();
                let mut given: Vec<Option<ValueId>> = vec![None; field_count];
                for (field, value) in fields {
                    let position = self.structs()[index]
                        .fields
                        .iter()
                        .position(|f| f.name == field.value)
                        .ok_or_else(|| {
                            semantic(
                                field.span,
                                format!("`{}` has no field `{}`", name.value, field.value),
                            )
                        })?;
                    if given[position].is_some() {
                        return Err(semantic(
                            field.span,
                            format!("field `{}` is set twice", field.value),
                        ));
                    }
                    let id = self.value(value)?;
                    let expected = self.structs()[index].fields[position].ty;
                    if self.ty(id) != expected {
                        return Err(semantic(
                            value.span,
                            format!(
                                "`{}.{}` expects {}, found {}",
                                name.value,
                                field.value,
                                self.describe(expected),
                                self.describe(self.ty(id))
                            ),
                        ));
                    }
                    given[position] = Some(id);
                }
                let mut parts = Vec::with_capacity(field_count);
                for (position, slot) in given.into_iter().enumerate() {
                    let id = match (slot, base_id) {
                        (Some(id), _) => id,
                        (None, Some(base)) => {
                            let field_ty = self.structs()[index].fields[position].ty;
                            self.emit(field_ty, Op::Extract(base, position as u32))
                        }
                        (None, None) => {
                            return Err(semantic(
                                name.span,
                                format!(
                                    "`{}` needs field `{}`; set it or start from `..base`",
                                    name.value,
                                    self.structs()[index].fields[position].name
                                ),
                            ));
                        }
                    };
                    parts.push(id);
                }
                Ok(self.emit(ty, Op::Construct(parts)))
            }
            ExprKind::If {
                condition,
                then,
                otherwise,
            } => {
                let condition_id = self.value(condition)?;
                if self.ty(condition_id) != Type::Bool {
                    return Err(semantic(
                        condition.span,
                        format!(
                            "`if` needs a Bool condition, found {}",
                            self.ty(condition_id)
                        ),
                    ));
                }
                let id = self.reserve(Type::Bool);
                let then_block = self.nested(then)?;
                let otherwise_block = self.nested(otherwise)?;
                let (a, b) = (self.ty(then_block.result), self.ty(otherwise_block.result));
                if a != b {
                    return Err(semantic(
                        otherwise.result.span,
                        format!(
                            "`if` branches give {} and {}; they must agree",
                            self.describe(a),
                            self.describe(b)
                        ),
                    ));
                }
                Ok(self.place(
                    id,
                    a,
                    Op::If {
                        condition: condition_id,
                        then: then_block,
                        otherwise: otherwise_block,
                    },
                ))
            }
            ExprKind::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let scrutinee_id = self.value(scrutinee)?;
                if self.ty(scrutinee_id) != Type::Int {
                    return Err(semantic(
                        scrutinee.span,
                        format!(
                            "`match` takes an Int, found {}",
                            self.describe(self.ty(scrutinee_id))
                        ),
                    ));
                }
                let id = self.reserve(Type::Int);
                let mut seen: std::collections::BTreeMap<i32, Span> =
                    std::collections::BTreeMap::new();
                let mut lowered = Vec::with_capacity(arms.len());
                let mut ty: Option<(Type, Span)> = None;
                for arm in arms {
                    for value in &arm.values {
                        if seen.insert(value.value, value.span).is_some() {
                            return Err(semantic(
                                value.span,
                                format!("`match` already has an arm for {}", value.value),
                            ));
                        }
                    }
                    let block = self.nested(&arm.body)?;
                    let arm_ty = self.ty(block.result);
                    if let Some((first, _)) = ty
                        && first != arm_ty
                    {
                        return Err(semantic(
                            arm.body.result.span,
                            format!(
                                "`match` arms give {} and {}; they must agree",
                                self.describe(first),
                                self.describe(arm_ty)
                            ),
                        ));
                    }
                    ty = Some((arm_ty, arm.body.result.span));
                    lowered.push((arm.values.iter().map(|v| v.value).collect(), block));
                }
                let otherwise_block = self.nested(otherwise)?;
                let otherwise_ty = self.ty(otherwise_block.result);
                if let Some((first, _)) = ty
                    && first != otherwise_ty
                {
                    return Err(semantic(
                        otherwise.result.span,
                        format!(
                            "`match` arms give {} and {}; they must agree",
                            self.describe(first),
                            self.describe(otherwise_ty)
                        ),
                    ));
                }
                Ok(self.place(
                    id,
                    otherwise_ty,
                    Op::Match {
                        scrutinee: scrutinee_id,
                        arms: lowered,
                        otherwise: otherwise_block,
                    },
                ))
            }
            ExprKind::Loop {
                count,
                name,
                init,
                until,
                body,
            } => {
                if count.value > MAX_LOOP_COUNT {
                    return Err(semantic(
                        count.span,
                        format!("`loop` bound {} exceeds {MAX_LOOP_COUNT}", count.value),
                    ));
                }
                let init_id = self.value(init)?;
                let carried = self.ty(init_id);
                // The values the enclosing loop (or the scan) carries are
                // read here, before this loop begins, and stand under their
                // names inside it: they do not change while it runs, and
                // `Carried` inside would mean this loop's own value.
                let outer: Vec<(String, Type)> = {
                    let mut seen = std::collections::BTreeSet::new();
                    self.locals
                        .iter()
                        .rev()
                        .filter_map(|(local_name, local)| match local {
                            Local::Carried { depth, ty } if *depth == self.loops.len() => seen
                                .insert(local_name.clone())
                                .then(|| (local_name.clone(), *ty)),
                            Local::Value(_) => {
                                seen.insert(local_name.clone());
                                None
                            }
                            _ => None,
                        })
                        .collect()
                };
                let shadows = outer.len();
                for (outer_name, ty) in outer {
                    let value = self.emit(ty, Op::Carried);
                    self.locals.push((outer_name, Local::Value(value)));
                }
                let id = self.reserve(carried);
                self.loops.push(carried);
                self.locals.push((
                    name.value.clone(),
                    Local::Carried {
                        depth: self.loops.len(),
                        ty: carried,
                    },
                ));
                let lowered = (|| {
                    let until_block = match until {
                        Some(until) => {
                            let block = self.nested_expr(until)?;
                            if self.ty(block.result) != Type::Bool {
                                return Err(semantic(
                                    until.span,
                                    format!(
                                        "`until` needs a Bool, found {}",
                                        self.ty(block.result)
                                    ),
                                ));
                            }
                            Some(block)
                        }
                        None => None,
                    };
                    let body_block = self.nested(body)?;
                    let next = self.ty(body_block.result);
                    if next != carried {
                        return Err(semantic(
                            body.result.span,
                            format!(
                                "loop carries {} but the body gives {}",
                                self.describe(carried),
                                self.describe(next)
                            ),
                        ));
                    }
                    Ok((until_block, body_block))
                })();
                self.locals.pop();
                self.locals.truncate(self.locals.len() - shadows);
                self.loops.pop();
                let (until_block, body_block) = lowered?;
                Ok(self.place(
                    id,
                    carried,
                    Op::Loop {
                        count: count.value,
                        init: init_id,
                        until: until_block,
                        body: body_block,
                    },
                ))
            }
        }
    }

    /// A float constant.
    fn constant(&mut self, value: f32) -> ValueId {
        self.emit(Type::Float, Op::Const(orla_ir::Value::Float(value)))
    }

    /// The builtins that are spellings of other operations: lowered to
    /// those operations, so every backend has them by construction.
    fn composed(&mut self, name: &str, args: &[Expr], span: Span) -> Result<ValueId, CompileError> {
        let ids = args
            .iter()
            .map(|arg| self.value(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let types: Vec<Type> = ids.iter().map(|id| self.ty(*id)).collect();
        let float_like = |t: Type| matches!(t, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4);
        let vector = |t: Type| t.components().is_some();
        let list = types
            .iter()
            .map(|ty| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let refused = semantic(span, format!("`{name}` cannot be applied to ({list})"));
        match (name, types.as_slice()) {
            ("saturate", [a]) if float_like(*a) => {
                let zero = self.constant(0.0);
                let one = self.constant(1.0);
                Ok(self.emit(*a, Op::Intrinsic(Intrinsic::Clamp, vec![ids[0], zero, one])))
            }
            ("distance", [a, b]) if a == b && vector(*a) => {
                let difference = self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], ids[1]));
                Ok(self.emit(
                    Type::Float,
                    Op::Intrinsic(Intrinsic::Length, vec![difference]),
                ))
            }
            ("radians", [a]) if float_like(*a) => {
                let scale = self.constant(std::f32::consts::PI / 180.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("degrees", [a]) if float_like(*a) => {
                let scale = self.constant(180.0 / std::f32::consts::PI);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Mul, ids[0], scale)))
            }
            ("inversesqrt", [a]) if float_like(*a) => {
                let root = self.emit(*a, Op::Intrinsic(Intrinsic::Sqrt, vec![ids[0]]));
                let one = self.constant(1.0);
                Ok(self.emit(*a, Op::Binary(BinaryOp::Div, one, root)))
            }
            ("mod", [a, b]) if float_like(*a) && (a == b || *b == Type::Float) => {
                // x - y * floor(x / y): the remainder that takes y's sign.
                // At an exact multiple it is a knife-edge, 0 or nearly y,
                // as floor is at an integer; backends may fall either way.
                let quotient = self.emit(*a, Op::Binary(BinaryOp::Div, ids[0], ids[1]));
                let floored = self.emit(*a, Op::Intrinsic(Intrinsic::Floor, vec![quotient]));
                let product = self.emit(*a, Op::Binary(BinaryOp::Mul, floored, ids[1]));
                Ok(self.emit(*a, Op::Binary(BinaryOp::Sub, ids[0], product)))
            }
            ("isnan", [Type::Float]) => {
                Ok(self.emit(Type::Bool, Op::Binary(BinaryOp::NotEqual, ids[0], ids[0])))
            }
            ("isinf", [Type::Float]) => {
                let magnitude = self.emit(Type::Float, Op::Intrinsic(Intrinsic::Abs, vec![ids[0]]));
                let largest = self.constant(f32::MAX);
                Ok(self.emit(
                    Type::Bool,
                    Op::Binary(BinaryOp::Greater, magnitude, largest),
                ))
            }
            ("refract", [i, n, Type::Float]) if i == n && vector(*i) => {
                // k = 1 - eta^2 (1 - dot(n, i)^2); below zero, the zero
                // vector; else eta i - (eta dot(n, i) + sqrt(k)) n.
                let (incident, normal, eta) = (ids[0], ids[1], ids[2]);
                let d = self.emit(
                    Type::Float,
                    Op::Intrinsic(Intrinsic::Dot, vec![normal, incident]),
                );
                let one = self.constant(1.0);
                let dd = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, d, d));
                let rest = self.emit(Type::Float, Op::Binary(BinaryOp::Sub, one, dd));
                let eta2 = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta, eta));
                let scaled = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta2, rest));
                let k = self.emit(Type::Float, Op::Binary(BinaryOp::Sub, one, scaled));
                let zero = self.constant(0.0);
                let total = self.emit(Type::Bool, Op::Binary(BinaryOp::Less, k, zero));
                let id = self.reserve(*i);
                self.buffers.push(Vec::new());
                let parts = vec![zero; i.components().unwrap_or(0) as usize];
                let nothing = self.emit(*i, Op::Construct(parts));
                let then = IrBlock {
                    body: self.buffers.pop().expect("the nested buffer is open"),
                    result: nothing,
                };
                self.buffers.push(Vec::new());
                let bent = self.emit(*i, Op::Binary(BinaryOp::Mul, incident, eta));
                let along = self.emit(Type::Float, Op::Binary(BinaryOp::Mul, eta, d));
                let root = self.emit(Type::Float, Op::Intrinsic(Intrinsic::Sqrt, vec![k]));
                let sum = self.emit(Type::Float, Op::Binary(BinaryOp::Add, along, root));
                let back = self.emit(*i, Op::Binary(BinaryOp::Mul, normal, sum));
                let out = self.emit(*i, Op::Binary(BinaryOp::Sub, bent, back));
                let otherwise = IrBlock {
                    body: self.buffers.pop().expect("the nested buffer is open"),
                    result: out,
                };
                Ok(self.place(
                    id,
                    *i,
                    Op::If {
                        condition: total,
                        then,
                        otherwise,
                    },
                ))
            }
            _ => Err(refused),
        }
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        symbol: &str,
        inner: &Expr,
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let inner = self.value(inner)?;
        let ty = self.ty(inner);
        let result = op
            .result_type(ty)
            .ok_or_else(|| semantic(span, format!("cannot apply `{symbol}` to {ty}")))?;
        Ok(self.emit(result, Op::Unary(op, inner)))
    }

    fn name(&mut self, name: &str, span: Span) -> Result<ValueId, CompileError> {
        if let Some((_, local)) = self.locals.iter().rev().find(|(local, _)| local == name) {
            return match local.clone() {
                Local::Value(id) => Ok(id),
                Local::Carried { depth, ty } => {
                    if depth != self.loops.len() {
                        return Err(semantic(
                            span,
                            format!(
                                "`{name}` is carried by an outer loop; pass it through this loop's carried value"
                            ),
                        ));
                    }
                    Ok(self.emit(ty, Op::Carried))
                }
            };
        }
        if let Some(index) = self.args.iter().position(|(arg, _)| arg == name) {
            let ty = self.args[index].1;
            return Ok(self.emit(ty, Op::Arg(index as u32)));
        }
        if let Some(index) = self.manifest.params.iter().position(|p| p.name == name) {
            let ty = self.manifest.params[index].default.ty();
            return Ok(self.emit(ty, Op::Param(index as u32)));
        }
        if self.manifest.inputs.iter().any(|input| input.name == name) {
            return Err(semantic(
                span,
                format!(
                    "`{name}` is an input port; read it with sample({name}), fetch({name}, coord), size({name}), at({name}, i) or count({name})"
                ),
            ));
        }
        if self
            .manifest
            .outputs
            .iter()
            .any(|output| output.name == name)
        {
            return Err(semantic(
                span,
                format!("`{name}` is the output; produce it with `return`"),
            ));
        }
        Err(semantic(span, format!("`{name}` is not defined")))
    }

    /// Resolves an input port by name.
    /// Calls a template: its Field parameters bind to the input ports the
    /// arguments name, checked by shape and layout; the instance for that
    /// binding is lowered once and called by value with the other
    /// arguments.
    fn instantiate(
        &mut self,
        template: &FnDecl,
        args: &[Expr],
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let name = template.name.value.as_str();
        self.arity(name, args, template.args.len(), span)?;
        let mut fields = Vec::new();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for ((param, ty), arg) in template.args.iter().zip(args) {
            match resolve_type(ty, self.shared.structs) {
                Ok(expected) => {
                    let id = self.value(arg)?;
                    if self.ty(id) != expected {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` expects {} here, found {}",
                                self.describe(expected),
                                self.describe(self.ty(id))
                            ),
                        ));
                    }
                    values.push(id);
                }
                Err(_) => {
                    let declared = port_type(ty, self.shared.structs, self.shared.aliases)?;
                    let (index, port) = self.input_index(arg, name)?;
                    if port.ty.shape != declared.shape || port.ty.schema != declared.schema {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` expects {} for `{}`, but `{}` is {}",
                                ty.value, param.value, port.name, port.ty
                            ),
                        ));
                    }
                    fields.push((param.value.clone(), index));
                    indices.push(index);
                }
            }
        }
        let key = (name.to_owned(), indices.clone());
        let existing = self.shared.instances.borrow().get(&key).copied();
        let index = match existing {
            Some(index) => index,
            None => {
                if self.shared.instantiating.borrow().iter().any(|n| n == name) {
                    return Err(semantic(span, format!("function `{name}` calls itself")));
                }
                self.shared.instantiating.borrow_mut().push(name.to_owned());
                let bound = indices
                    .iter()
                    .map(|i| self.manifest.inputs[*i as usize].name.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                let instance =
                    lower_function(self.shared, template, fields, &format!("{name}<{bound}>"));
                self.shared.instantiating.borrow_mut().pop();
                let instance = instance?;
                let index = {
                    let mut functions = self.shared.functions.borrow_mut();
                    functions.push((format!("{name}<{bound}>"), instance));
                    (functions.len() - 1) as u32
                };
                self.shared.instances.borrow_mut().insert(key, index);
                index
            }
        };
        let ret = {
            let functions = self.shared.functions.borrow();
            result_type(&functions[index as usize].1)
        };
        Ok(self.emit(ret, Op::Call(index, values)))
    }

    fn input_index(&self, expr: &Expr, what: &str) -> Result<(u32, &Port), CompileError> {
        let ExprKind::Name(name) = &expr.kind else {
            return Err(semantic(
                expr.span,
                format!("`{what}` expects an input port name"),
            ));
        };
        // A Field parameter stands for the port it was bound to.
        if let Some((_, index)) = self.fields.iter().find(|(field, _)| field == name) {
            return Ok((*index, &self.manifest.inputs[*index as usize]));
        }
        let index = self
            .manifest
            .inputs
            .iter()
            .position(|input| &input.name == name)
            .ok_or_else(|| semantic(expr.span, format!("`{name}` is not an input")))?;
        Ok((index as u32, &self.manifest.inputs[index]))
    }

    /// The shape the entry point maps over: the output's, or the first
    /// input's for a reduction.
    fn output_shape(&self) -> Shape {
        match self.manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } => self
                .manifest
                .inputs
                .first()
                .map_or(Shape::Rows, Port::shape),
            Mode::Map => self
                .manifest
                .outputs
                .first()
                .map_or(Shape::Grid, Port::shape),
        }
    }

    fn entry_only(&self, name: &str, span: Span) -> Result<(), CompileError> {
        if self.is_entry {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!(
                    "`{name}` is only available inside `compute`; pass its value as an argument"
                ),
            ))
        }
    }

    fn require(&self, name: &str, capability: Capability, span: Span) -> Result<(), CompileError> {
        if self.manifest.requires.contains(capability) {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!("`{name}` needs `{capability}`; add it to `requires`"),
            ))
        }
    }

    fn arity(
        &self,
        name: &str,
        args: &[Expr],
        count: usize,
        span: Span,
    ) -> Result<(), CompileError> {
        if args.len() == count {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!("`{name}` takes {count} argument(s), found {}", args.len()),
            ))
        }
    }

    fn call(&mut self, name: &str, args: &[Expr], span: Span) -> Result<ValueId, CompileError> {
        match name {
            "put" => {
                // A write belongs to a step: the entry, or a function the
                // step calls with the memory as a Field parameter. The
                // carried value's start is before every step.
                if self.name == "init" {
                    return Err(semantic(
                        span,
                        "`put` writes step by step; the initial carried value is before the first step",
                    ));
                }
                self.require(name, Capability::FieldWrite, span)?;
                self.arity(name, args, 3, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let (port_name, port_ty) = (port.name.clone(), port.ty.clone());
                if !self.manifest.memories.contains(&port_name) {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`put` writes a memory, and `{port_name}` is not one; declare it with `memory {port_name}: {port_ty};`"
                        ),
                    ));
                }
                if self.manifest.mode != Mode::Scan {
                    return Err(semantic(
                        span,
                        "`put` writes step by step, which only `scan` does",
                    ));
                }
                let address = self.value(&args[1])?;
                let expected = port_ty.index_type();
                if self.ty(address) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`put` addresses `{port_name}`, a {}, with {}; it takes {}",
                            port_ty.shape,
                            self.describe(self.ty(address)),
                            self.describe(expected)
                        ),
                    ));
                }
                let value = self.value(&args[2])?;
                let element = port_ty.element;
                if self.ty(value) != element {
                    return Err(semantic(
                        args[2].span,
                        format!(
                            "`put` writes {} into `{port_name}`, whose element is {}",
                            self.describe(self.ty(value)),
                            self.describe(element)
                        ),
                    ));
                }
                Ok(self.emit(element, Op::Put(input, address, value)))
            }
            "sample" => {
                self.entry_only(name, span)?;
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                if port.shape() != self.output_shape() {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`sample` reads `{}` at this element's own address, but it is {} and the output is {}; use `at`",
                            port.name,
                            port.shape(),
                            self.output_shape()
                        ),
                    ));
                }
                let element = port.element();
                Ok(self.emit(element, Op::Sample(input)))
            }
            "at" | "fetch" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 2, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let element = port.element();
                let expected = port.ty.index_type();
                let (port_name, shape) = (port.name.clone(), port.shape());
                let index = self.value(&args[1])?;
                if self.ty(index) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`{name}` addresses `{port_name}`, a {shape}, with {expected}, found {}",
                            self.describe(self.ty(index))
                        ),
                    ));
                }
                Ok(self.emit(element, Op::At(input, index)))
            }
            "at_linear" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 2, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let element = port.element();
                let expected = orla_ir::linear_index_type(port.shape());
                let (port_name, shape) = (port.name.clone(), port.shape());
                let index = self.value(&args[1])?;
                if self.ty(index) != expected {
                    return Err(semantic(
                        args[1].span,
                        format!(
                            "`{name}` addresses `{port_name}`, a {shape}, with {expected}, found {}",
                            self.describe(self.ty(index))
                        ),
                    ));
                }
                let layout = self.manifest.inputs[input as usize].ty.schema.clone();
                if !layout
                    .scalars()
                    .iter()
                    .all(|s| *s == orla_ir::Scalar::Float)
                {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`at_linear` interpolates floats, but `{port_name}` lays out as {layout}"
                        ),
                    ));
                }
                Ok(self.emit(element, Op::AtLinear(input, index)))
            }
            "count" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, _) = self.input_index(&args[0], name)?;
                Ok(self.emit(Type::Int, Op::Count(input)))
            }
            "size" => {
                self.require(name, Capability::FieldRead, span)?;
                self.arity(name, args, 1, span)?;
                let (input, port) = self.input_index(&args[0], name)?;
                let ty = port.ty.index_type();
                Ok(self.emit(ty, Op::Size(input)))
            }
            "element" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Int, Op::Element))
            }
            "coord" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                let shape = self.output_shape();
                if shape == Shape::Rows {
                    return Err(semantic(
                        span,
                        "`coord` is for grids and volumes; over rows use `element()`",
                    ));
                }
                Ok(self.emit(shape.index_type(), Op::Coord))
            }
            "domain" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                let ty = self.output_shape().index_type();
                Ok(self.emit(ty, Op::Domain))
            }
            "time" => {
                self.require(name, Capability::TimeRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Time))
            }
            "rate" => {
                self.require(name, Capability::TimeRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Rate))
            }
            "window" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Window))
            }
            "scale" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Float, Op::Scale))
            }
            "pointer" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Pointer))
            }
            "pointer_down" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Bool, Op::PointerDown))
            }
            "button" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 1, span)?;
                let code = self.value(&args[0])?;
                if self.ty(code) != Type::Int {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`button` needs an Int number, found {}",
                            self.describe(self.ty(code))
                        ),
                    ));
                }
                Ok(self.emit(Type::Bool, Op::Button(code)))
            }
            "wheel" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Wheel))
            }
            "key" => {
                self.require(name, Capability::InputRead, span)?;
                self.arity(name, args, 1, span)?;
                let code = self.value(&args[0])?;
                if self.ty(code) != Type::Int {
                    return Err(semantic(
                        args[0].span,
                        format!(
                            "`key` needs an Int code, found {}",
                            self.describe(self.ty(code))
                        ),
                    ));
                }
                Ok(self.emit(Type::Bool, Op::Key(code)))
            }
            "index" => {
                self.arity(name, args, 0, span)?;
                let scanning = self.is_entry && self.manifest.mode == Mode::Scan;
                if self.loops.is_empty() && !scanning {
                    return Err(semantic(
                        span,
                        "`index()` is only available inside a loop or a `scan`",
                    ));
                }
                Ok(self.emit(Type::Int, Op::Index))
            }
            "vec2" | "vec3" | "vec4" => {
                let ty = match name {
                    "vec2" => Type::Vec2,
                    "vec3" => Type::Vec3,
                    _ => Type::Vec4,
                };
                self.arity(name, args, ty.components().unwrap_or(0) as usize, span)?;
                let mut parts = Vec::with_capacity(args.len());
                for arg in args {
                    let part = self.value(arg)?;
                    let part = match self.ty(part) {
                        Type::Float => part,
                        // An Int component widens; a cell address needs no
                        // `float()` at every use.
                        Type::Int => {
                            self.emit(Type::Float, Op::Intrinsic(Intrinsic::ToFloat, vec![part]))
                        }
                        other => {
                            return Err(semantic(
                                arg.span,
                                format!("`{name}` components must be Float or Int, found {other}"),
                            ));
                        }
                    };
                    parts.push(part);
                }
                Ok(self.emit(ty, Op::Construct(parts)))
            }
            "mat2" | "mat3" | "mat4" => {
                let ty = match name {
                    "mat2" => Type::Mat2,
                    "mat3" => Type::Mat3,
                    _ => Type::Mat4,
                };
                let order = ty.order().unwrap_or(0) as usize;
                let column = ty.column().unwrap_or(Type::Float);
                if args.len() == 1 {
                    // One Float: that value down the diagonal.
                    let diagonal = self.value(&args[0])?;
                    if self.ty(diagonal) != Type::Float {
                        return Err(semantic(
                            args[0].span,
                            format!(
                                "`{name}` takes {order} {column} columns, or one Float for the diagonal; found {}",
                                self.describe(self.ty(diagonal))
                            ),
                        ));
                    }
                    let zero = self.constant(0.0);
                    let columns = (0..order)
                        .map(|c| {
                            let parts = (0..order)
                                .map(|r| if r == c { diagonal } else { zero })
                                .collect();
                            self.emit(column, Op::Construct(parts))
                        })
                        .collect();
                    return Ok(self.emit(ty, Op::Construct(columns)));
                }
                self.arity(name, args, order, span)?;
                let mut columns = Vec::with_capacity(order);
                for arg in args {
                    let part = self.value(arg)?;
                    if self.ty(part) != column {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` takes {column} columns, found {}",
                                self.describe(self.ty(part))
                            ),
                        ));
                    }
                    columns.push(part);
                }
                Ok(self.emit(ty, Op::Construct(columns)))
            }
            "saturate" | "distance" | "radians" | "degrees" | "inversesqrt" | "mod" | "isnan"
            | "isinf" | "refract" => self.composed(name, args, span),
            _ => {
                if let Some(intrinsic) = Intrinsic::from_name(name) {
                    let ids = args
                        .iter()
                        .map(|arg| self.value(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let types = ids.iter().map(|id| self.ty(*id)).collect::<Vec<_>>();
                    let ty = intrinsic.result_type(&types).ok_or_else(|| {
                        let list = types
                            .iter()
                            .map(|ty| ty.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        semantic(span, format!("`{name}` cannot be applied to ({list})"))
                    })?;
                    return Ok(self.emit(ty, Op::Intrinsic(intrinsic, ids)));
                }
                let found = {
                    let functions = self.shared.functions.borrow();
                    functions.iter().position(|(f, _)| f == name).map(|index| {
                        (
                            index,
                            functions[index].1.args.clone(),
                            result_type(&functions[index].1),
                        )
                    })
                };
                if let Some((index, expected, ret)) = found {
                    self.arity(name, args, expected.len(), span)?;
                    let mut ids = Vec::with_capacity(args.len());
                    for (arg, expected) in args.iter().zip(expected) {
                        let id = self.value(arg)?;
                        if self.ty(id) != expected {
                            return Err(semantic(
                                arg.span,
                                format!(
                                    "`{name}` expects {} here, found {}",
                                    self.describe(expected),
                                    self.describe(self.ty(id))
                                ),
                            ));
                        }
                        ids.push(id);
                    }
                    return Ok(self.emit(ret, Op::Call(index as u32, ids)));
                }
                if let Some(template) = self.shared.templates.iter().find(|t| t.name.value == name)
                {
                    return self.instantiate(template, args, span);
                }
                Err(semantic(
                    span,
                    format!("`{name}` is not a builtin, intrinsic, or function defined above"),
                ))
            }
        }
    }
}

/// Type of a lowered helper function's result.
fn result_type(function: &Function) -> Type {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, function.results[0]).unwrap_or(Type::Float)
}
