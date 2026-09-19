//! Semantic analysis and lowering from AST to IR.

use std::str::FromStr;

use orla_ir::{
    BinaryOp, Block as IrBlock, Capability, Field, Function, Instruction, Intrinsic, KernelId,
    MAX_LOOP_COUNT, Manifest, Mode, Module, Op, Param, Port, PortType, Shape, ShapeIndex,
    StructDef, Type, UnaryOp, Value, ValueId, schema_of,
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

/// Resolves a type name against the plain types and the struct table.
fn resolve_type(name: &Spanned<String>, structs: &[StructDef]) -> Result<Type, CompileError> {
    if let Some(index) = structs.iter().position(|def| def.name == name.value) {
        return Ok(Type::Struct(index as u32));
    }
    parse_type(name)
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

fn structs(kernel: &Kernel) -> Result<Vec<StructDef>, CompileError> {
    let mut structs: Vec<StructDef> = Vec::new();
    for decl in &kernel.structs {
        if structs.iter().any(|def| def.name == decl.name.value) {
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
        structs.push(StructDef {
            name: decl.name.value.clone(),
            fields,
        });
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
    let mut functions: Vec<(String, Function)> = Vec::new();
    for declaration in &kernel.functions {
        if functions
            .iter()
            .any(|(name, _)| *name == declaration.name.value)
        {
            return Err(semantic(
                declaration.name.span,
                format!("function `{}` is defined twice", declaration.name.value),
            ));
        }
        let function = lower_function(&manifest, &structs, &functions, declaration)?;
        functions.push((declaration.name.value.clone(), function));
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
        let mut builder = Builder::new(&manifest, &structs, &functions, true, "compute");
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
        if actual != expected {
            return Err(semantic(
                kernel.compute.result.span,
                format!(
                    "output `{}` needs a {} per element, but the body gives {}",
                    manifest.outputs[0].name,
                    describe(expected, &structs),
                    describe(actual, &structs)
                ),
            ));
        }
        builder.finish(vec![result])
    };
    let merge = match &kernel.merge {
        None => None,
        Some(decl) => {
            let mut builder = Builder::new(&manifest, &structs, &functions, false, "merge");
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
                        describe(expected, &structs),
                        describe(actual, &structs)
                    ),
                ));
            }
            Some(builder.finish(vec![result]))
        }
    };
    let init = match &kernel.carry {
        None => None,
        Some(carry) => {
            let mut builder = Builder::new(&manifest, &structs, &functions, false, "init");
            let result = builder.value(&carry.init)?;
            let actual = builder.ty(result);
            if actual != expected {
                return Err(semantic(
                    carry.init.span,
                    format!(
                        "`scan` carries a {} like its output, but the initial value is {}",
                        describe(expected, &structs),
                        describe(actual, &structs)
                    ),
                ));
            }
            Some(builder.finish(vec![result]))
        }
    };
    Ok(Module {
        manifest,
        structs,
        functions: functions.into_iter().map(|(_, f)| f).collect(),
        entry,
        merge,
        init,
    })
}

/// The port type a written name stands for. `Image` and the declared
/// aliases are spellings of a shape and an element; the structure is what
/// gets checked.
fn port_type(
    text: &Spanned<String>,
    structs: &[StructDef],
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
    let schema = schema_of(element, structs).ok_or_else(|| {
        semantic(
            text.span,
            format!("`{}` cannot be an element type", describe(element, structs)),
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
fn aliases(
    kernel: &Kernel,
    structs: &[StructDef],
) -> Result<Vec<(String, PortType)>, CompileError> {
    let mut aliases: Vec<(String, PortType)> = Vec::new();
    for decl in &kernel.types {
        let name = &decl.name.value;
        if name == "Image"
            || Type::from_str(name).is_ok()
            || structs.iter().any(|s| &s.name == name)
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
    structs: &[StructDef],
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
    let inputs = ports(&kernel.inputs)?;
    let outputs = ports(&kernel.outputs)?;
    if let (Some(merge), Some(_)) = (&kernel.merge, &kernel.carry) {
        return Err(semantic(
            merge.span,
            "a kernel is a reduction (`merge`) or a scan (`scan`), not both",
        ));
    }
    let mode = if let Some(carry) = &kernel.carry {
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
    if outputs.len() != 1 {
        return Err(semantic(
            kernel.id.as_ref().map(|id| id.span).unwrap_or_default(),
            format!(
                "a kernel declares exactly one output in this language version, found {}",
                outputs.len()
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
            {
                return Err(semantic(
                    decl.ty.span,
                    "parameters cross the ABI and cannot be structs; use plain values",
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
        .chain(kernel.outputs.iter().map(|d| &d.name));
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

fn lower_function(
    manifest: &Manifest,
    structs: &[StructDef],
    functions: &[(String, Function)],
    declaration: &FnDecl,
) -> Result<Function, CompileError> {
    let mut builder = Builder::new(manifest, structs, functions, false, &declaration.name.value);
    for (name, ty) in &declaration.args {
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
                describe(ret, structs),
                describe(actual, structs)
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
    structs: &'a [StructDef],
    functions: &'a [(String, Function)],
    is_entry: bool,
    name: String,
    args: Vec<(String, Type)>,
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
}

impl<'a> Builder<'a> {
    fn new(
        manifest: &'a Manifest,
        structs: &'a [StructDef],
        functions: &'a [(String, Function)],
        is_entry: bool,
        name: &str,
    ) -> Self {
        Self {
            manifest,
            structs,
            functions,
            is_entry,
            name: name.to_owned(),
            args: Vec::new(),
            locals: Vec::new(),
            scopes: vec![0],
            buffers: vec![Vec::new()],
            types: Vec::new(),
            loops: Vec::new(),
        }
    }

    fn finish(mut self, results: Vec<ValueId>) -> Function {
        let body = self.buffers.pop().unwrap_or_default();
        Function {
            name: self.name,
            args: self.args.into_iter().map(|(_, ty)| ty).collect(),
            body,
            results,
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
        describe(ty, self.structs)
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
                    orla_language::BinaryOp::Less => BinaryOp::Less,
                    orla_language::BinaryOp::LessEqual => BinaryOp::LessEqual,
                    orla_language::BinaryOp::Greater => BinaryOp::Greater,
                    orla_language::BinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
                    orla_language::BinaryOp::Equal => BinaryOp::Equal,
                    orla_language::BinaryOp::NotEqual => BinaryOp::NotEqual,
                    orla_language::BinaryOp::And => BinaryOp::And,
                    orla_language::BinaryOp::Or => BinaryOp::Or,
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
                    let def = &self.structs[index as usize];
                    let field = def
                        .fields
                        .iter()
                        .position(|field| &field.name == member)
                        .ok_or_else(|| {
                            semantic(expr.span, format!("`{}` has no field `{member}`", def.name))
                        })?;
                    let field_ty = def.fields[field].ty;
                    return Ok(self.emit(field_ty, Op::Extract(inner, field as u8)));
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
                Ok(self.emit(Type::Float, Op::Extract(inner, component)))
            }
            ExprKind::Call(name, args) => self.call(name, args, expr.span),
            ExprKind::Struct { name, base, fields } => {
                let index = self
                    .structs
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
                let field_count = self.structs[index].fields.len();
                let mut given: Vec<Option<ValueId>> = vec![None; field_count];
                for (field, value) in fields {
                    let position = self.structs[index]
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
                    let expected = self.structs[index].fields[position].ty;
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
                            let field_ty = self.structs[index].fields[position].ty;
                            self.emit(field_ty, Op::Extract(base, position as u8))
                        }
                        (None, None) => {
                            return Err(semantic(
                                name.span,
                                format!(
                                    "`{}` needs field `{}`; set it or start from `..base`",
                                    name.value, self.structs[index].fields[position].name
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
    fn input_index(&self, expr: &Expr, what: &str) -> Result<(u32, &Port), CompileError> {
        let ExprKind::Name(name) = &expr.kind else {
            return Err(semantic(
                expr.span,
                format!("`{what}` expects an input port name"),
            ));
        };
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
            Mode::Reduce | Mode::Scan => self
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
                    if self.ty(part) != Type::Float {
                        return Err(semantic(
                            arg.span,
                            format!("`{name}` components must be Float, found {}", self.ty(part)),
                        ));
                    }
                    parts.push(part);
                }
                Ok(self.emit(ty, Op::Construct(parts)))
            }
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
                if let Some(index) = self.functions.iter().position(|(f, _)| f == name) {
                    let function = &self.functions[index].1;
                    self.arity(name, args, function.args.len(), span)?;
                    let expected = function.args.clone();
                    let ret = result_type(function);
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
