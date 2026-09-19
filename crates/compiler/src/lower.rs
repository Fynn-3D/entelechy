//! Semantic analysis and lowering from AST to IR.

use std::str::FromStr;

use orla_ir::{
    BinaryOp, Capability, Function, Instruction, Intrinsic, KernelId, Manifest, Module, Op, Param,
    Port, Type, UnaryOp, Value, ValueId,
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

pub(crate) fn lower(kernel: &Kernel) -> Result<Module, CompileError> {
    let manifest = manifest(kernel)?;
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
        let function = lower_function(&manifest, &functions, declaration)?;
        functions.push((declaration.name.value.clone(), function));
    }
    let entry = {
        let mut builder = Builder::new(&manifest, &functions, true, "compute");
        let result = builder.block(&kernel.compute)?;
        let expected = match manifest.outputs.first() {
            Some(port) if port.ty == Type::Image => Type::Vec4,
            Some(port) => port.ty,
            None => {
                return Err(semantic(
                    kernel.id.span,
                    "kernel declares no output; add `output result: Image;`",
                ));
            }
        };
        let actual = builder.ty(result);
        if actual != expected {
            return Err(semantic(
                kernel.compute.result.span,
                format!(
                    "output `{}` needs a {expected} per pixel, but `return` gives {actual}",
                    manifest.outputs[0].name
                ),
            ));
        }
        builder.finish(vec![result])
    };
    Ok(Module {
        manifest,
        functions: functions.into_iter().map(|(_, f)| f).collect(),
        entry,
    })
}

fn manifest(kernel: &Kernel) -> Result<Manifest, CompileError> {
    let id = KernelId::from_str(&kernel.id.value)
        .map_err(|error| semantic(kernel.id.span, error.to_string()))?;
    let requires = kernel
        .requires
        .iter()
        .map(|name| {
            Capability::from_str(&name.value)
                .map_err(|error| semantic(name.span, error.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let ports = |decls: &[orla_language::PortDecl], kind: &str| {
        decls
            .iter()
            .map(|decl| {
                let ty = parse_type(&decl.ty)?;
                if ty != Type::Image {
                    return Err(semantic(
                        decl.ty.span,
                        format!(
                            "{kind} `{}` must be an Image in this language version",
                            decl.name.value
                        ),
                    ));
                }
                Ok(Port {
                    name: decl.name.value.clone(),
                    ty,
                })
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let inputs = ports(&kernel.inputs, "input")?;
    let outputs = ports(&kernel.outputs, "output")?;
    if outputs.len() != 1 {
        return Err(semantic(
            kernel.id.span,
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
        (ExprKind::Neg(inner), _) => match constant(inner, ty)? {
            Value::Float(v) => Value::Float(-v),
            Value::Int(v) => Value::Int(
                v.checked_neg()
                    .ok_or_else(|| semantic(expr.span, "integer literal overflows when negated"))?,
            ),
            Value::Vec2([x, y]) => Value::Vec2([-x, -y]),
            Value::Vec4([x, y, z, w]) => Value::Vec4([-x, -y, -z, -w]),
        },
        (ExprKind::Call(name, args), Type::Vec2 | Type::Vec4)
            if (name == "vec2" && ty == Type::Vec2) || (name == "vec4" && ty == Type::Vec4) =>
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
            if ty == Type::Vec2 {
                Value::Vec2([parts[0], parts[1]])
            } else {
                Value::Vec4(parts)
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

fn lower_function(
    manifest: &Manifest,
    functions: &[(String, Function)],
    declaration: &FnDecl,
) -> Result<Function, CompileError> {
    let mut builder = Builder::new(manifest, functions, false, &declaration.name.value);
    for (name, ty) in &declaration.args {
        let ty = parse_type(ty)?;
        if ty == Type::Image {
            return Err(semantic(name.span, "functions cannot take Image arguments"));
        }
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
    let ret = parse_type(&declaration.ret)?;
    let result = builder.block(&declaration.body)?;
    let actual = builder.ty(result);
    if actual != ret {
        return Err(semantic(
            declaration.body.result.span,
            format!(
                "function `{}` returns {ret} but the body gives {actual}",
                declaration.name.value
            ),
        ));
    }
    Ok(builder.finish(vec![result]))
}

struct Builder<'a> {
    manifest: &'a Manifest,
    functions: &'a [(String, Function)],
    is_entry: bool,
    name: String,
    args: Vec<(String, Type)>,
    locals: Vec<(String, ValueId)>,
    body: Vec<Instruction>,
}

impl<'a> Builder<'a> {
    fn new(
        manifest: &'a Manifest,
        functions: &'a [(String, Function)],
        is_entry: bool,
        name: &str,
    ) -> Self {
        Self {
            manifest,
            functions,
            is_entry,
            name: name.to_owned(),
            args: Vec::new(),
            locals: Vec::new(),
            body: Vec::new(),
        }
    }

    fn finish(self, results: Vec<ValueId>) -> Function {
        Function {
            name: self.name,
            args: self.args.into_iter().map(|(_, ty)| ty).collect(),
            body: self.body,
            results,
        }
    }

    fn emit(&mut self, ty: Type, op: Op) -> ValueId {
        self.body.push(Instruction { ty, op });
        ValueId((self.body.len() - 1) as u32)
    }

    fn ty(&self, id: ValueId) -> Type {
        self.body[id.0 as usize].ty
    }

    fn block(&mut self, block: &Block) -> Result<ValueId, CompileError> {
        for (name, value) in &block.lets {
            if self
                .locals
                .iter()
                .any(|(existing, _)| existing == &name.value)
            {
                return Err(semantic(
                    name.span,
                    format!("`{}` is already bound; bindings are immutable", name.value),
                ));
            }
            let id = self.value(value)?;
            self.locals.push((name.value.clone(), id));
        }
        self.value(&block.result)
    }

    /// Lowers an expression that must produce a plain value (not an image).
    fn value(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        match &expr.kind {
            ExprKind::Float(value) => Ok(self.emit(Type::Float, Op::Const(Value::Float(*value)))),
            ExprKind::Int(value) => Ok(self.emit(Type::Int, Op::Const(Value::Int(*value)))),
            ExprKind::Name(name) => self.name(name, expr.span),
            ExprKind::Neg(inner) => {
                let inner = self.value(inner)?;
                let ty = self.ty(inner);
                Ok(self.emit(ty, Op::Unary(UnaryOp::Neg, inner)))
            }
            ExprKind::Binary(op, left, right) => {
                let op = match op {
                    orla_language::BinaryOp::Add => BinaryOp::Add,
                    orla_language::BinaryOp::Sub => BinaryOp::Sub,
                    orla_language::BinaryOp::Mul => BinaryOp::Mul,
                    orla_language::BinaryOp::Div => BinaryOp::Div,
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
                let count = ty.components().ok_or_else(|| {
                    semantic(expr.span, format!("`.{member}` needs a vector, found {ty}"))
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
        }
    }

    fn name(&mut self, name: &str, span: Span) -> Result<ValueId, CompileError> {
        if let Some((_, id)) = self.locals.iter().rev().find(|(local, _)| local == name) {
            return Ok(*id);
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
                    "`{name}` is an image; read it with sample({name}), fetch({name}, coord) or size({name})"
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

    fn input_index(&self, expr: &Expr, what: &str) -> Result<u32, CompileError> {
        let ExprKind::Name(name) = &expr.kind else {
            return Err(semantic(
                expr.span,
                format!("`{what}` expects an input image name"),
            ));
        };
        self.manifest
            .inputs
            .iter()
            .position(|input| &input.name == name)
            .map(|index| index as u32)
            .ok_or_else(|| semantic(expr.span, format!("`{name}` is not an input image")))
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
                self.arity(name, args, 1, span)?;
                let input = self.input_index(&args[0], name)?;
                Ok(self.emit(Type::Vec4, Op::ImageSample(input)))
            }
            "fetch" => {
                self.arity(name, args, 2, span)?;
                let input = self.input_index(&args[0], name)?;
                let coord = self.value(&args[1])?;
                if self.ty(coord) != Type::Vec2 {
                    return Err(semantic(
                        args[1].span,
                        format!("`fetch` needs a Vec2 coordinate, found {}", self.ty(coord)),
                    ));
                }
                Ok(self.emit(Type::Vec4, Op::ImageFetch(input, coord)))
            }
            "size" => {
                self.arity(name, args, 1, span)?;
                let input = self.input_index(&args[0], name)?;
                Ok(self.emit(Type::Vec2, Op::ImageSize(input)))
            }
            "coord" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Coord))
            }
            "domain" => {
                self.entry_only(name, span)?;
                self.arity(name, args, 0, span)?;
                Ok(self.emit(Type::Vec2, Op::Domain))
            }
            "vec2" | "vec4" => {
                let ty = if name == "vec2" {
                    Type::Vec2
                } else {
                    Type::Vec4
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
                    let ret = function.body[function.results[0].0 as usize].ty;
                    let mut ids = Vec::with_capacity(args.len());
                    for (arg, expected) in args.iter().zip(expected) {
                        let id = self.value(arg)?;
                        if self.ty(id) != expected {
                            return Err(semantic(
                                arg.span,
                                format!("`{name}` expects {expected} here, found {}", self.ty(id)),
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
