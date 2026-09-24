//! Semantic analysis and lowering from AST to IR.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use orla_ir::{
    Block as IrBlock, Capability, Field, Function, Instruction, Intrinsic, KernelId,
    MAX_ARRAY_COUNT, Manifest, Mode, Module, Op, Port, PortType, Shape, StructDef, Type, ValueId,
};
use orla_language::{Block, Expr, ExprKind, FnDecl, Kernel, Span, Spanned};

use crate::CompileError;

mod builtins;
mod composite;
mod declare;
mod expr;
mod matrix;
mod written;

use declare::{RESERVED_BUILTINS, aliases, manifest, port_type, structs};
pub(crate) use declare::{literal, literal_exact, literal_in, literal_of_type};

fn semantic(span: Span, message: impl Into<String>) -> CompileError {
    CompileError::Semantic {
        span,
        message: message.into(),
    }
}

fn parse_type(name: &Spanned<String>) -> Result<Type, CompileError> {
    Type::from_str(&name.value).map_err(|error| {
        let known: Vec<&str> = Type::NAMED.iter().map(|(_, name)| *name).collect();
        semantic(
            name.span,
            format!("{error}{}", orla_language::did_you_mean(&name.value, known)),
        )
    })
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
            .or_else(|| ty.lanes().map(|(n, lane)| (lane, n as usize))),
    }
}

/// Human-readable type for diagnostics, naming structs.
fn describe(ty: Type, structs: &[StructDef]) -> String {
    orla_ir::type_name(ty, structs)
}

/// Registers the record of a map's outputs when there are several: a
/// struct named after them, `{ color, depth }`, whose fields are their
/// elements in declaration order. Its layout is theirs one after another,
/// which is the words every backend writes an element as.
fn outputs_record(manifest: &Manifest, structs: &Structs) -> Option<u32> {
    if manifest.outputs.len() < 2 || manifest.mode != Mode::Map {
        return None;
    }
    let names: Vec<&str> = manifest
        .outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    let fields = manifest
        .outputs
        .iter()
        .map(|port| Field {
            name: port.name.clone(),
            ty: port.element(),
        })
        .collect();
    let mut table = structs.borrow_mut();
    table.push(StructDef::record(
        format!("{{ {} }}", names.join(", ")),
        fields,
    ));
    Some((table.len() - 1) as u32)
}

pub(crate) fn lower(kernel: &Kernel, id: Option<&KernelId>) -> Result<Module, CompileError> {
    let structs = structs(kernel)?;
    let aliases = aliases(kernel, &structs)?;
    let manifest = manifest(kernel, id, &structs, &aliases)?;
    let record = outputs_record(&manifest, &structs);
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
        declared: kernel
            .functions
            .iter()
            .map(|f| f.name.value.clone())
            .collect(),
        record,
    };
    let mut shared = shared;
    for declaration in &kernel.functions {
        if RESERVED_BUILTINS.contains(&declaration.name.value.as_str()) {
            return Err(semantic(
                declaration.name.span,
                format!(
                    "`{}` is a builtin of the language; a function cannot take its name (a math builtin's, like `step`, it can, and stands for it here)",
                    declaration.name.value
                ),
            ));
        }
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
            if declaration.generics.iter().any(|g| g.value == ty.value) {
                continue;
            }
            if resolve_type(ty, &structs).is_err() {
                port_type(ty, &structs, &aliases)
                    .map_err(|_| semantic(ty.span, format!("`{}` is not a type", ty.value)))?;
                has_field = true;
            }
        }
        if has_field || !declaration.generics.is_empty() {
            shared.templates.push(declaration);
            continue;
        }
        let function = lower_function(
            &shared,
            declaration,
            Vec::new(),
            Vec::new(),
            &declaration.name.value,
        )?;
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
                    pair: None,
                },
            ));
        }
        let mut result = builder.block(&kernel.compute)?;
        if shared.record.is_none() && !matches!(manifest.mode, Mode::Scatter { .. }) {
            result = builder.adapt(result, expected);
        }
        let mut split: Option<Vec<ValueId>> = None;
        let actual = builder.ty(result);
        if let Mode::Scatter { .. } = manifest.mode {
            result = builder.scatter_pair(result, kernel.compute.result.span)?;
        } else if let Some(record) = shared.record {
            // Several outputs: the body gives their record, and the entry
            // gives each output its field, in declaration order.
            if actual != Type::Struct(record) {
                let named: Vec<String> = manifest
                    .outputs
                    .iter()
                    .map(|port| format!("{}: ..", port.name))
                    .collect();
                return Err(semantic(
                    kernel.compute.result.span,
                    format!(
                        "a Kernel with several outputs gives them by name, `{{ {} }}`, but the body gives {}",
                        named.join(", "),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            let results = manifest
                .outputs
                .iter()
                .enumerate()
                .map(|(position, port)| {
                    builder.emit(port.element(), Op::Extract(result, position as u32))
                })
                .collect();
            split = Some(results);
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
        builder.finish(split.unwrap_or_else(|| vec![result]))
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
            let result = builder.adapt(result, expected);
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
            let result = builder.adapt(result, expected);
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

/// What every block of a Kernel resolves names against: its manifest,
/// structs, aliases, the functions lowered so far (a call's index), the
/// templates with Field parameters, and their instances by binding.
struct Shared<'a> {
    manifest: &'a Manifest,
    structs: &'a Structs,
    aliases: &'a [(String, PortType)],
    functions: RefCell<Vec<(String, Function)>>,
    templates: Vec<&'a FnDecl>,
    /// Instantiated templates by what they were bound with, to function
    /// index.
    instances: RefCell<BTreeMap<Instance, u32>>,
    /// Templates being instantiated, to refuse a template calling itself.
    instantiating: RefCell<Vec<String>>,
    /// Every function name the Kernel declares, in order, so a call to
    /// one that is not lowered yet can say why.
    declared: Vec<String>,
    /// The record of a map's outputs, when it declares several: a struct
    /// whose fields are the outputs' elements in declaration order, which
    /// `{ name: .., ... }` builds and `compute` gives.
    record: Option<u32>,
}

/// Lowers a function body. `fields` binds its Field parameters to input
/// ports, for an instance of a template; `name` names the result.
fn lower_function(
    shared: &Shared<'_>,
    declaration: &FnDecl,
    fields: Vec<(String, u32)>,
    generics: Vec<(String, Type)>,
    name: &str,
) -> Result<Function, CompileError> {
    let structs = shared.structs;
    let bound = |ty: &Spanned<String>| -> Option<Type> {
        generics
            .iter()
            .find(|(parameter, _)| parameter == &ty.value)
            .map(|(_, bound)| *bound)
    };
    let mut builder = Builder::new(shared, false, name, fields);
    for (name, ty) in &declaration.args {
        if builder.fields.iter().any(|(field, _)| field == &name.value) {
            continue;
        }
        let ty = match bound(ty) {
            Some(ty) => ty,
            None => resolve_type(ty, structs)?,
        };
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
    let ret = match bound(&declaration.ret) {
        Some(ty) => ty,
        None => {
            if declaration
                .generics
                .iter()
                .any(|parameter| parameter.value == declaration.ret.value)
            {
                return Err(semantic(
                    declaration.ret.span,
                    format!(
                        "`{}` returns `{}`, which no argument binds; a type parameter is bound by an argument's type",
                        declaration.name.value, declaration.ret.value
                    ),
                ));
            }
            resolve_type(&declaration.ret, structs)?
        }
    };
    let result = builder.block(&declaration.body)?;
    let result = builder.adapt(result, ret);
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

/// What a template was instantiated with: its name, the input ports
/// bound to its Field parameters, and the types bound to its type
/// parameters, each written as the IR spells it.
type Instance = (String, Vec<u32>, Vec<String>);

/// What a name resolves to inside a block.
#[derive(Clone)]
enum Local {
    Value(ValueId),
    /// The carried value of the loop that introduced the name.
    Carried {
        depth: usize,
        ty: Type,
        /// When the loop's body may `break`, what it carries is that
        /// value and whether it stopped; the name is the first field
        /// of this struct.
        pair: Option<Type>,
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
    /// The names bound in each open block scope. A binding may not
    /// repeat a name of its own block, and asking that by walking the
    /// block's bindings is a walk once a binding: a Kernel of a few
    /// thousand `let`s spent most of its lowering on it.
    bound: Vec<BTreeSet<String>>,
    /// Instruction buffers, one per open block; the last is current.
    buffers: Vec<Vec<Instruction>>,
    /// Type of every emitted value by id.
    types: Vec<Type>,
    /// Carried types of enclosing loops, innermost last.
    loops: Vec<Type>,
    /// The `[each ...]` builders being lowered: each element's position,
    /// and how many loops were open when the builder began, so `index()`
    /// inside it is this position unless a loop opened since.
    each: Vec<(u32, usize)>,
    /// Every `let`, in source order, with its value: what a trace shows.
    bindings: Vec<(ValueId, String)>,
    /// The numbers written without a width that nothing has named yet,
    /// exact as written (a float as a `Double`, an integer as a `Long`):
    /// each becomes a `Double` or a `Long` where it meets one.
    literals: std::collections::HashMap<u32, orla_ir::Number>,
    /// The vectors, matrices and arrays built of written numbers alone
    /// that nothing has named yet: each takes the kind it meets as whole.
    aggregates: BTreeSet<u32>,
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
            bound: vec![BTreeSet::new()],
            buffers: vec![Vec::new()],
            types: Vec::new(),
            loops: Vec::new(),
            each: Vec::new(),
            bindings: Vec::new(),
            literals: std::collections::HashMap::new(),
            aggregates: BTreeSet::new(),
        }
    }

    /// A scalar converted to `kind`, when it is another kind: what a
    /// constructor makes of a component.
    fn converted(&mut self, id: ValueId, kind: Type) -> ValueId {
        let from = self.ty(id);
        if from == kind {
            return id;
        }
        let intrinsic = match kind {
            Type::Float => Intrinsic::ToFloat,
            Type::Int => Intrinsic::ToInt,
            Type::Double => Intrinsic::ToDouble,
            _ => Intrinsic::ToLong,
        };
        self.emit(kind, Op::Intrinsic(intrinsic, vec![id]))
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
        if let Op::Construct(parts) = &op
            && !matches!(ty, Type::Struct(index) if !self.structs()[index as usize].is_array())
            && parts.iter().all(|part| self.is_written(*part))
        {
            self.aggregates.insert(id.0);
        }
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

    /// A scatter's pair, as the body returns it or `emit` gives it: a
    /// struct of an address into the target and a value of its element.
    /// An integer address is the float one converted, exact as every
    /// coordinate a Field can have is.
    fn scatter_pair(&mut self, pair: ValueId, span: Span) -> Result<ValueId, CompileError> {
        let Mode::Scatter { base, .. } = self.manifest.mode else {
            return Ok(pair);
        };
        let expected = self.manifest.outputs[0].element();
        let actual = self.ty(pair);
        let mut pair = pair;
        // The body yields a pair: an address into the target and a
        // value of its element type, as a struct of two fields.
        let target = &self.manifest.inputs[base as usize];
        let address = target.ty.index_type();
        let table = self.shared.structs.borrow();
        let fields = match actual {
            Type::Struct(index) if !table[index as usize].is_array() => {
                Some(&table[index as usize].fields)
            }
            _ => None,
        };
        let well_formed = fields.is_some_and(|fields| {
            fields.len() == 2 && fields[0].ty == address && fields[1].ty == expected
        });
        // An integer address (`IVec2` into a grid) is the float one
        // converted: exact, as every coordinate a Field can have is.
        let integer_address = fields.is_some_and(|fields| {
            fields.len() == 2
                && fields[0].ty.other_kind() == Some(address)
                && fields[1].ty == expected
        });
        if integer_address {
            let integer = fields.map(|fields| fields[0].ty).unwrap_or(address);
            let name = format!(
                "({}, {})",
                describe(address, &table),
                describe(expected, &table)
            );
            let existing = table.iter().position(|def| def.name == name);
            drop(table);
            let normal = match existing {
                Some(index) => Type::Struct(index as u32),
                None => {
                    let mut table = self.shared.structs.borrow_mut();
                    table.push(StructDef::record(
                        name,
                        vec![
                            Field {
                                name: "address".into(),
                                ty: address,
                            },
                            Field {
                                name: "value".into(),
                                ty: expected,
                            },
                        ],
                    ));
                    Type::Struct((table.len() - 1) as u32)
                }
            };
            let at = self.emit(integer, Op::Extract(pair, 0));
            let at = self.emit(address, Op::Intrinsic(Intrinsic::ToFloat, vec![at]));
            let value = self.emit(expected, Op::Extract(pair, 1));
            pair = self.emit(normal, Op::Construct(vec![at, value]));
        } else if !well_formed {
            return Err(semantic(
                span,
                format!(
                    "a scatter into `{}` gives a struct of two fields per element, an address ({}) and a value ({}), but the body gives {}",
                    target.name,
                    describe(address, &table),
                    describe(expected, &table),
                    describe(actual, &table)
                ),
            ));
        }
        Ok(pair)
    }

    fn describe(&self, ty: Type) -> String {
        describe(ty, &self.structs())
    }

    /// Reads the carried value of the loop being lowered: the value
    /// itself, or the first field of what a loop with a `break` carries.
    pub(super) fn read_carried(&mut self, ty: Type, pair: Option<Type>) -> ValueId {
        match pair {
            None => self.emit(ty, Op::Carried),
            Some(pair) => {
                let carried = self.emit(pair, Op::Carried);
                self.emit(ty, Op::Extract(carried, 0))
            }
        }
    }

    /// The struct type a loop with a `break` carries: the value and
    /// whether the loop has stopped, made once per element type.
    pub(super) fn pair_type(&self, value: Type) -> Type {
        let name = format!("({}, stopped)", self.describe(value));
        if let Some(index) = self.structs().iter().position(|def| def.name == name) {
            return Type::Struct(index as u32);
        }
        let mut table = self.shared.structs.borrow_mut();
        table.push(StructDef::record(
            name,
            vec![
                Field {
                    name: "value".into(),
                    ty: value,
                },
                Field {
                    name: "stopped".into(),
                    ty: Type::Bool,
                },
            ],
        ));
        Type::Struct((table.len() - 1) as u32)
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
        self.scoped(block, |builder, result| builder.value(result))
    }

    /// Lowers a block's bindings in their own scope and then its result,
    /// which the caller lowers: a loop body's result may `break`.
    pub(super) fn scoped<F>(&mut self, block: &Block, result: F) -> Result<ValueId, CompileError>
    where
        F: FnOnce(&mut Self, &Expr) -> Result<ValueId, CompileError>,
    {
        self.scopes.push(self.locals.len());
        self.bound.push(BTreeSet::new());
        let lowered = (|| {
            for (name, value) in &block.lets {
                // A statement for its effect: `put(...)`, or `let _ = ...`.
                if name.value == "_" {
                    self.value(value)?;
                    continue;
                }
                if self
                    .bound
                    .last()
                    .expect("a scope is open")
                    .contains(&name.value)
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
                // A name fixes a number's width: every use sees one value.
                self.literals.remove(&id.0);
                self.aggregates.remove(&id.0);
                if name.value != "_" {
                    self.bindings.push((id, name.value.clone()));
                }
                self.bound
                    .last_mut()
                    .expect("a scope is open")
                    .insert(name.value.clone());
                self.locals.push((name.value.clone(), Local::Value(id)));
            }
            result(self, &block.result)
        })();
        let start = self.scopes.pop().expect("a scope is open");
        self.bound.pop();
        self.locals.truncate(start);
        lowered
    }

    fn name(&mut self, name: &str, span: Span) -> Result<ValueId, CompileError> {
        if let Some((_, local)) = self.locals.iter().rev().find(|(local, _)| local == name) {
            return match local.clone() {
                Local::Value(id) => Ok(id),
                Local::Carried { depth, ty, pair } => {
                    if depth != self.loops.len() {
                        return Err(semantic(
                            span,
                            format!(
                                "`{name}` is carried by an outer loop; pass it through this loop's carried value"
                            ),
                        ));
                    }
                    Ok(self.read_carried(ty, pair))
                }
            };
        }
        if let Some(index) = self.args.iter().position(|(arg, _)| arg == name) {
            let ty = self.args[index].1;
            return Ok(self.emit(ty, Op::Arg(index as u32)));
        }
        if let Some(index) = self.manifest.params.iter().position(|p| p.name == name) {
            let ty = self.manifest.params[index].ty;
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
        // Everything a name here could have been, so a near miss is
        // named rather than left to be found.
        let known: Vec<&str> = self
            .locals
            .iter()
            .map(|(local, _)| local.as_str())
            .chain(self.args.iter().map(|(arg, _)| arg.as_str()))
            .chain(self.manifest.params.iter().map(|p| p.name.as_str()))
            .chain(self.manifest.inputs.iter().map(|port| port.name.as_str()))
            .collect();
        Err(semantic(
            span,
            format!(
                "`{name}` is not defined{}",
                orla_language::did_you_mean(name, known)
            ),
        ))
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
        let mut generics: Vec<(String, Type)> = Vec::new();
        for ((param, ty), arg) in template.args.iter().zip(args) {
            // A type parameter takes the type of the argument written
            // here, and every later argument of that parameter must
            // agree with it.
            if template.generics.iter().any(|g| g.value == ty.value) {
                let id = self.value(arg)?;
                let actual = self.ty(id);
                match generics
                    .iter()
                    .find(|(parameter, _)| parameter == &ty.value)
                {
                    Some((_, bound)) if *bound != actual => {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` takes `{}` as {} here, but as {} for an earlier argument",
                                ty.value,
                                self.describe(actual),
                                self.describe(*bound)
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => generics.push((ty.value.clone(), actual)),
                }
                values.push(id);
                continue;
            }
            match resolve_type(ty, self.shared.structs) {
                Ok(expected) => {
                    let id = self.value(arg)?;
                    let id = self.adapt(id, expected);
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
                                ty.value,
                                param.value,
                                port.name,
                                orla_ir::port_type_name(&port.ty, &self.structs())
                            ),
                        ));
                    }
                    fields.push((param.value.clone(), index));
                    indices.push(index);
                }
            }
        }
        let key = (
            name.to_owned(),
            indices.clone(),
            generics
                .iter()
                .map(|(_, ty)| format!("{ty:?}"))
                .collect::<Vec<_>>(),
        );
        let existing = self.shared.instances.borrow().get(&key).copied();
        let index = match existing {
            Some(index) => index,
            None => {
                if self.shared.instantiating.borrow().iter().any(|n| n == name) {
                    return Err(semantic(
                        span,
                        format!(
                            "`{name}` calls itself, and a Kernel has no stack to call into; a `loop` with the work as its carried value is the form"
                        ),
                    ));
                }
                self.shared.instantiating.borrow_mut().push(name.to_owned());
                let bound = indices
                    .iter()
                    .map(|i| self.manifest.inputs[*i as usize].name.clone())
                    .chain(generics.iter().map(|(_, ty)| self.describe(*ty)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let instance = lower_function(
                    self.shared,
                    template,
                    fields,
                    generics.clone(),
                    &format!("{name}<{bound}>"),
                )
                .map_err(|error| match (error, generics.is_empty()) {
                    // A body that does not hold for the types this call
                    // gives it is reported where it is written, and says
                    // which instance was being lowered.
                    (CompileError::Semantic { span, message }, false) => {
                        let bound = generics
                            .iter()
                            .map(|(parameter, ty)| {
                                format!("`{parameter}` as {}", self.describe(*ty))
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        CompileError::Semantic {
                            span,
                            message: format!("{message} (in `{name}` with {bound})"),
                        }
                    }
                    (error, _) => error,
                });
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
            .ok_or_else(|| {
                let known: Vec<&str> = self
                    .manifest
                    .inputs
                    .iter()
                    .map(|port| port.name.as_str())
                    .collect();
                semantic(
                    expr.span,
                    format!(
                        "`{name}` is not an input{}",
                        orla_language::did_you_mean(name, known)
                    ),
                )
            })?;
        Ok((index as u32, &self.manifest.inputs[index]))
    }

    /// The shape the entry point maps over: the output's, or the first
    /// input's for a reduction.
    fn output_shape(&self) -> Shape {
        match self.manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } | Mode::Prefix { .. } => self
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

    /// An integer vector address converted to the float one a Field
    /// takes: exact for every coordinate a Field can have, so `at` and
    /// `put` need no integer path of their own. Any other value passes.
    fn float_address(&mut self, address: ValueId, expected: Type) -> ValueId {
        if self.ty(address).other_kind() == Some(expected) {
            self.emit(expected, Op::Intrinsic(Intrinsic::ToFloat, vec![address]))
        } else {
            address
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
