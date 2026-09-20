//! Structural and capability validation of a [`Module`].
//!
//! Validation proves the properties backends rely on: pre-order value ids,
//! block scoping, type consistency, bounded calls and loops, and that every
//! Host operation is covered by a declared capability. A [`Validated`]
//! module is the only thing a backend will execute.

use std::fmt;
use std::ops::Deref;

use crate::{
    Block, Capability, Combine, Function, Instruction, MAX_LOOP_COUNT, Manifest, Mode, Module, Op,
    Port, PortType, Shape, ShapeIndex, StructDef, Type, UnaryOp, ValueId, schema_of,
};

/// A module that passed [`validate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Validated(Module);

impl Validated {
    /// Gives the module back.
    pub fn into_module(self) -> Module {
        self.0
    }
}

impl fmt::Display for Validated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for Validated {
    type Target = Module;

    fn deref(&self) -> &Module {
        &self.0
    }
}

/// Why a module is not well-formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    /// Function name where the problem lies, or `manifest`.
    pub location: String,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.location, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Checks a module and wraps it as [`Validated`].
pub fn validate(module: Module) -> Result<Validated, ValidationError> {
    validate_structs(&module.structs)?;
    validate_manifest(&module.manifest, &module.structs)?;
    for (index, function) in module.functions.iter().enumerate() {
        let types = validate_function(&module, function, index, false)?;
        if function.results.len() != 1 {
            return Err(error(
                &function.name,
                format!(
                    "helper functions return exactly one value, found {}",
                    function.results.len()
                ),
            ));
        }
        let _ = types;
    }
    let types = validate_function(&module, &module.entry, module.functions.len(), true)?;
    let entry = &module.entry;
    if !entry.args.is_empty() {
        return Err(error(&entry.name, "the entry point takes no arguments"));
    }
    let outputs = &module.manifest.outputs;
    if module.manifest.mode != Mode::Scan && module.init.is_some() {
        return Err(error("manifest", "only a scan Kernel has an `init`"));
    }
    match (module.manifest.mode, &module.merge) {
        (Mode::Map, None) => {}
        (Mode::Map | Mode::Scan | Mode::Scatter { .. }, Some(_)) => {
            return Err(error("manifest", "only a reduce Kernel has a `merge`"));
        }
        (Mode::Scatter { base, combine }, None) => {
            let manifest = &module.manifest;
            let target = manifest.inputs.get(base as usize).ok_or_else(|| {
                error(
                    "manifest",
                    format!("a scatter into input {base} needs that input"),
                )
            })?;
            if manifest.inputs.is_empty() {
                return Err(error(
                    "manifest",
                    "a scatter maps over its first input, so it needs one",
                ));
            }
            let output = outputs
                .first()
                .ok_or_else(|| error("manifest", "a scatter produces one output"))?;
            if outputs.len() != 1 || output.ty != target.ty {
                return Err(error(
                    "manifest",
                    format!(
                        "a scatter produces exactly one output, shaped and laid out like the input it scatters into (`{}`)",
                        target.name
                    ),
                ));
            }
            let element = output.element();
            if combine != Combine::Last && !numeric(&module.structs, element) {
                return Err(error(
                    "manifest",
                    format!(
                        "`{}` combines numbers; `{element}` holds a Bool, so scatter with `last`",
                        combine.name()
                    ),
                ));
            }
            let address = target.ty.index_type();
            let pair = entry.results.first().map(|r| types[r.0 as usize]);
            let fields = match pair {
                Some(Type::Struct(index)) => Some(&module.structs[index as usize].fields),
                _ => None,
            };
            let well_formed = fields.is_some_and(|fields| {
                fields.len() == 2 && fields[0].ty == address && fields[1].ty == element
            });
            if entry.results.len() != 1 || !well_formed {
                return Err(error(
                    &entry.name,
                    format!(
                        "a scatter's entry returns a struct of two fields, an address ({address}) into `{}` and a value ({element})",
                        target.name
                    ),
                ));
            }
        }
        (Mode::Reduce, None) => {
            return Err(error("manifest", "a reduce Kernel needs a `merge`"));
        }
        (Mode::Scan, None) => {
            let init = module
                .init
                .as_ref()
                .ok_or_else(|| error("manifest", "a scan Kernel needs an `init`"))?;
            let first = module.manifest.inputs.first().ok_or_else(|| {
                error(
                    "manifest",
                    "a scan runs over its first input, so it needs one",
                )
            })?;
            let output = outputs
                .first()
                .ok_or_else(|| error("manifest", "a scan produces one `Rows` output"))?;
            let plain = outputs.len() - module.manifest.memories.len();
            if plain != 1 || output.shape() != Shape::Rows || first.shape() != Shape::Rows {
                return Err(error(
                    "manifest",
                    "a scan runs over a `Rows` input and produces a `Rows` output of the same count",
                ));
            }
            let element = output.element();
            if !init.args.is_empty() {
                return Err(error(&init.name, "`init` takes no arguments"));
            }
            let init_types = validate_function(&module, init, module.functions.len() + 1, false)?;
            // The steps write; the carried value's start is before them,
            // so `init` may not call a function that writes.
            let writers = writers(&module.functions);
            if let Some(callee) = first_writing_call(&init.body, &writers) {
                return Err(error(
                    &init.name,
                    format!(
                        "`init` calls `{}`, which writes a memory; the initial carried value is before the first step",
                        module.functions[callee].name
                    ),
                ));
            }
            let actual = init.results.first().map(|r| init_types[r.0 as usize]);
            if init.results.len() != 1 || actual != Some(element) {
                return Err(error(
                    &init.name,
                    format!("`init` gives one value of the output element type {element}"),
                ));
            }
        }
        (Mode::Reduce, Some(merge)) => {
            if module.manifest.inputs.is_empty() {
                return Err(error(
                    "manifest",
                    "a reduction maps over its first input, so it needs one",
                ));
            }
            let output = outputs
                .first()
                .ok_or_else(|| error("manifest", "a reduction produces one output element"))?;
            if outputs.len() != 1 || output.shape() != Shape::Rows {
                return Err(error(
                    "manifest",
                    "a reduction produces exactly one output, a `Rows`",
                ));
            }
            let element = output.element();
            if merge.args != [element, element] {
                return Err(error(
                    &merge.name,
                    format!("`merge` takes two values of the output element type {element}"),
                ));
            }
            let merge_types = validate_function(&module, merge, module.functions.len() + 1, false)?;
            let actual = merge.results.first().map(|r| merge_types[r.0 as usize]);
            if merge.results.len() != 1 || actual != Some(element) {
                return Err(error(
                    &merge.name,
                    format!("`merge` returns one value of the output element type {element}"),
                ));
            }
        }
    }
    let plain_outputs = outputs.len() - module.manifest.memories.len();
    if entry.results.len() != plain_outputs {
        return Err(error(
            &entry.name,
            format!(
                "entry returns {} values but the manifest declares {} outputs besides its memories",
                entry.results.len(),
                plain_outputs
            ),
        ));
    }
    let scattering = matches!(module.manifest.mode, Mode::Scatter { .. });
    for (result, output) in entry.results.iter().zip(outputs) {
        if scattering {
            break;
        }
        let actual = types[result.0 as usize];
        let expected = output.element();
        if actual != expected {
            return Err(error(
                &entry.name,
                format!(
                    "output `{}` expects {expected} per element but the entry returns {actual}",
                    output.name
                ),
            ));
        }
    }
    let requires = &module.manifest.requires;
    if !outputs.is_empty() && !requires.contains(Capability::FieldWrite) {
        return Err(error(
            "manifest",
            "writing an output requires `field.write`".to_owned(),
        ));
    }
    Ok(Validated(module))
}

/// Whether a type is numbers all the way down: floats and ints, no Bool.
fn numeric(structs: &[StructDef], ty: Type) -> bool {
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
        Type::Struct(index) => structs[index as usize]
            .fields
            .iter()
            .all(|field| numeric(structs, field.ty)),
    }
}

fn error(location: &str, message: impl Into<String>) -> ValidationError {
    ValidationError {
        location: location.to_owned(),
        message: message.into(),
    }
}

fn validate_structs(structs: &[StructDef]) -> Result<(), ValidationError> {
    for (index, def) in structs.iter().enumerate() {
        if let Some(element) = def.element {
            let count = def.fields.len();
            if count == 0 || count > crate::MAX_ARRAY_COUNT as usize {
                return Err(error(
                    "structs",
                    format!(
                        "array `{}` has {count} elements; an array has 1 to {}",
                        def.name,
                        crate::MAX_ARRAY_COUNT
                    ),
                ));
            }
            if let Some(field) = def.fields.iter().find(|field| field.ty != element) {
                return Err(error(
                    "structs",
                    format!(
                        "array `{}` of {element} has an element `{}` of {}",
                        def.name, field.name, field.ty
                    ),
                ));
            }
        }
        let mut names = std::collections::BTreeSet::new();
        for field in &def.fields {
            if !names.insert(&field.name) {
                return Err(error(
                    "structs",
                    format!(
                        "struct `{}` declares field `{}` twice",
                        def.name, field.name
                    ),
                ));
            }
            match field.ty {
                Type::Struct(inner) if inner as usize >= index => {
                    return Err(error(
                        "structs",
                        format!(
                            "struct `{}` field `{}` refers to struct {inner}, which is not defined earlier",
                            def.name, field.name
                        ),
                    ));
                }
                _ => {}
            }
        }
        if def.element.is_none() && def.fields.len() > u8::MAX as usize {
            return Err(error(
                "structs",
                format!("struct `{}` has more than {} fields", def.name, u8::MAX),
            ));
        }
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest, structs: &[StructDef]) -> Result<(), ValidationError> {
    // A memory is one port in both lists, last in each, typed alike.
    for (k, name) in manifest.memories.iter().enumerate() {
        let position = |ports: &[Port]| ports.len() + k - manifest.memories.len().min(ports.len());
        let as_input = manifest.inputs.get(position(&manifest.inputs));
        let as_output = manifest.outputs.get(position(&manifest.outputs));
        match (as_input, as_output) {
            (Some(input), Some(output)) if input.name == *name && output.name == *name => {
                if input.ty != output.ty {
                    return Err(error(
                        "manifest",
                        format!("memory `{name}` is typed differently as an input and an output"),
                    ));
                }
            }
            _ => {
                return Err(error(
                    "manifest",
                    format!(
                        "memory `{name}` must be the last of the inputs and of the outputs, in order"
                    ),
                ));
            }
        }
    }
    if !manifest.memories.is_empty() && manifest.mode != Mode::Scan {
        return Err(error(
            "manifest",
            "a memory is written step by step, which needs ordered work: `scan`",
        ));
    }
    if !manifest.memories.is_empty() && manifest.inputs.len() == manifest.memories.len() {
        return Err(error(
            "manifest",
            "a scan runs over its first input, which cannot be a memory; declare an input",
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    let ports = manifest
        .inputs
        .iter()
        .map(|port| (&port.name, "input"))
        .chain(manifest.params.iter().map(|param| (&param.name, "param")))
        .chain(
            manifest
                .outputs
                .iter()
                .filter(|port| !manifest.memories.contains(&port.name))
                .map(|port| (&port.name, "output")),
        );
    for (name, kind) in ports {
        if !names.insert(name.clone()) {
            return Err(error(
                "manifest",
                format!("{kind} `{name}` is declared twice"),
            ));
        }
    }
    for port in manifest.inputs.iter().chain(&manifest.outputs) {
        let PortType {
            element, schema, ..
        } = &port.ty;
        let expected = schema_of(*element, structs).ok_or_else(|| {
            error(
                "manifest",
                format!(
                    "port `{}` has element type {element}, which has no layout",
                    port.name
                ),
            )
        })?;
        if *schema != expected {
            return Err(error(
                "manifest",
                format!(
                    "port `{}` declares schema {schema} but its element lays out as {expected}",
                    port.name
                ),
            ));
        }
    }
    for param in &manifest.params {
        if !param.default.ty().is_value() {
            return Err(error(
                "manifest",
                format!("param `{}` must be a plain value", param.name),
            ));
        }
    }
    Ok(())
}

/// Per-function checking state.
struct Checker<'a> {
    module: &'a Module,
    function: &'a Function,
    index: usize,
    is_entry: bool,
    /// Type of every slot once defined.
    types: Vec<Type>,
    /// Whether a slot is defined and in scope at the current point.
    visible: Vec<bool>,
    /// Next expected pre-order id.
    next: u32,
    /// Carried type of each enclosing loop, innermost last.
    loops: Vec<Type>,
}

impl<'a> Checker<'a> {
    fn fail(&self, message: impl Into<String>) -> ValidationError {
        error(&self.function.name, message)
    }

    fn operand(&self, id: ValueId, position: ValueId) -> Result<Type, ValidationError> {
        if !self.visible.get(id.0 as usize).copied().unwrap_or(false) {
            return Err(self.fail(format!(
                "{position} refers to {id}, which is not defined before it in an enclosing block"
            )));
        }
        Ok(self.types[id.0 as usize])
    }

    fn require(&self, capability: Capability, position: ValueId) -> Result<(), ValidationError> {
        if self.module.manifest.requires.contains(capability) {
            Ok(())
        } else {
            Err(self.fail(format!(
                "{position} needs `{capability}`, which is not declared"
            )))
        }
    }

    fn input(&self, input: u32, position: ValueId) -> Result<&'a Port, ValidationError> {
        self.module
            .manifest
            .inputs
            .get(input as usize)
            .ok_or_else(|| self.fail(format!("{position} reads unknown input {input}")))
    }

    /// The shape the entry point maps over: the output's, or the first
    /// input's for a reduction or a scan.
    fn output_shape(&self) -> Shape {
        let manifest = &self.module.manifest;
        match manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } => {
                manifest.inputs.first().map_or(Shape::Rows, Port::shape)
            }
            Mode::Map => manifest.outputs.first().map_or(Shape::Grid, Port::shape),
        }
    }

    /// The type a scan carries, when this is a scan's entry point.
    fn scan_carry(&self) -> Option<Type> {
        (self.is_entry && self.module.manifest.mode == Mode::Scan)
            .then(|| self.module.manifest.outputs.first().map(Port::element))
            .flatten()
    }

    fn entry_only(&self, position: ValueId) -> Result<(), ValidationError> {
        if self.is_entry {
            Ok(())
        } else {
            Err(self.fail(format!(
                "{position} uses a per-pixel operation outside the entry point"
            )))
        }
    }

    /// Checks a block and returns the type of its result. Values defined
    /// inside go out of scope afterwards.
    fn block(&mut self, block: &Block) -> Result<Type, ValidationError> {
        let first = self.next;
        self.instructions(&block.body)?;
        let ty = self.operand(block.result, ValueId(self.next))?;
        for slot in first..self.next {
            self.visible[slot as usize] = false;
        }
        Ok(ty)
    }

    fn instructions(&mut self, instructions: &[Instruction]) -> Result<(), ValidationError> {
        for instruction in instructions {
            self.instruction(instruction)?;
        }
        Ok(())
    }

    fn instruction(&mut self, instruction: &Instruction) -> Result<(), ValidationError> {
        let Instruction { id, ty, op } = instruction;
        let position = *id;
        if id.0 != self.next {
            return Err(self.fail(format!(
                "{position} is out of order; expected %{}",
                self.next
            )));
        }
        self.next += 1;
        let manifest = &self.module.manifest;
        let actual = match op {
            Op::Const(value) => value.ty(),
            Op::Param(param) => manifest
                .params
                .get(*param as usize)
                .map(|param| param.default.ty())
                .ok_or_else(|| self.fail(format!("{position} reads unknown param {param}")))?,
            Op::Arg(arg) => *self
                .function
                .args
                .get(*arg as usize)
                .ok_or_else(|| self.fail(format!("{position} reads unknown argument {arg}")))?,
            Op::Unary(op, a) => {
                let a = self.operand(*a, position)?;
                op.result_type(a).ok_or_else(|| {
                    let symbol = match op {
                        UnaryOp::Neg => "-",
                        UnaryOp::Not => "!",
                    };
                    self.fail(format!("{position} cannot apply `{symbol}` to {a}"))
                })?
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (self.operand(*a, position)?, self.operand(*b, position)?);
                op.result_type(a, b).ok_or_else(|| {
                    self.fail(format!(
                        "{position} cannot apply `{}` to {a} and {b}",
                        op.symbol()
                    ))
                })?
            }
            Op::Construct(parts) => {
                if let Type::Struct(index) = ty {
                    let def = self.module.structs.get(*index as usize).ok_or_else(|| {
                        self.fail(format!("{position} constructs unknown struct {index}"))
                    })?;
                    if parts.len() != def.fields.len() {
                        return Err(self.fail(format!(
                            "{position} constructs `{}` from {} values, expected {}",
                            def.name,
                            parts.len(),
                            def.fields.len()
                        )));
                    }
                    for (part, field) in parts.iter().zip(&def.fields) {
                        let actual = self.operand(*part, position)?;
                        if actual != field.ty {
                            return Err(self.fail(format!(
                                "{position} gives `{}.{}` a {actual}, expected {}",
                                def.name, field.name, field.ty
                            )));
                        }
                    }
                    *ty
                } else if let (Some(order), Some(column)) = (ty.order(), ty.column()) {
                    if parts.len() != order as usize {
                        return Err(self.fail(format!(
                            "{position} constructs {ty} from {} columns",
                            parts.len()
                        )));
                    }
                    for part in parts {
                        let actual = self.operand(*part, position)?;
                        if actual != column {
                            return Err(self.fail(format!(
                                "{position} constructs {ty} from a {actual} column, expected {column}"
                            )));
                        }
                    }
                    *ty
                } else {
                    let (expected, lane) = ty.lanes().ok_or_else(|| {
                        self.fail(format!("{position} constructs non-vector {ty}"))
                    })?;
                    if parts.len() != expected as usize {
                        return Err(self.fail(format!(
                            "{position} constructs {ty} from {} components",
                            parts.len()
                        )));
                    }
                    for part in parts {
                        if self.operand(*part, position)? != lane {
                            return Err(self.fail(format!(
                                "{position} constructs {ty} from a component that is not {lane}"
                            )));
                        }
                    }
                    *ty
                }
            }
            Op::Extract(source, component) => {
                let source = self.operand(*source, position)?;
                if let Type::Struct(index) = source {
                    let def = &self.module.structs[index as usize];
                    let field = def.fields.get(*component as usize).ok_or_else(|| {
                        self.fail(format!(
                            "{position} extracts field {component} of `{}`",
                            def.name
                        ))
                    })?;
                    field.ty
                } else if let (Some(order), Some(column)) = (source.order(), source.column()) {
                    if *component >= u32::from(order) {
                        return Err(self.fail(format!(
                            "{position} extracts column {component} of {source}"
                        )));
                    }
                    column
                } else {
                    let (count, lane) = source.lanes().ok_or_else(|| {
                        self.fail(format!("{position} extracts from non-vector {source}"))
                    })?;
                    if *component >= u32::from(count) {
                        return Err(self.fail(format!(
                            "{position} extracts component {component} of {source}"
                        )));
                    }
                    lane
                }
            }
            Op::Subscript(array, index) => {
                let (array, index) = (
                    self.operand(*array, position)?,
                    self.operand(*index, position)?,
                );
                let element = match array {
                    Type::Struct(struct_index) => {
                        let def = &self.module.structs[struct_index as usize];
                        def.element.ok_or_else(|| {
                            self.fail(format!(
                                "{position} subscripts `{}`, a struct, not an array",
                                def.name
                            ))
                        })?
                    }
                    _ => array.column().or_else(|| array.lanes().map(|(_, lane)| lane)).ok_or_else(|| {
                        self.fail(format!(
                            "{position} subscripts {array}, which is not an array, a matrix or a vector"
                        ))
                    })?,
                };
                if index != Type::Int {
                    return Err(
                        self.fail(format!("{position} subscripts with {index}, not an Int"))
                    );
                }
                element
            }
            Op::Call(callee, args) => {
                if *callee as usize >= self.index {
                    return Err(self.fail(format!(
                        "{position} calls function {callee}, which is not defined earlier"
                    )));
                }
                let callee = &self.module.functions[*callee as usize];
                if callee.args.len() != args.len() {
                    return Err(self.fail(format!(
                        "{position} calls `{}` with {} arguments, expected {}",
                        callee.name,
                        args.len(),
                        callee.args.len()
                    )));
                }
                for (arg, expected) in args.iter().zip(&callee.args) {
                    let actual = self.operand(*arg, position)?;
                    if actual != *expected {
                        return Err(self.fail(format!(
                            "{position} passes {actual} to `{}` where {expected} is expected",
                            callee.name
                        )));
                    }
                }
                result_type(callee)
            }
            Op::Intrinsic(intrinsic, args) => {
                let types = args
                    .iter()
                    .map(|arg| self.operand(*arg, position))
                    .collect::<Result<Vec<_>, _>>()?;
                intrinsic.result_type(&types).ok_or_else(|| {
                    let list = types
                        .iter()
                        .map(|ty| ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.fail(format!(
                        "{position} calls `{}` with ({list})",
                        intrinsic.name()
                    ))
                })?
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                if self.operand(*condition, position)? != Type::Bool {
                    return Err(self.fail(format!("{position} branches on a non-Bool")));
                }
                let then_ty = self.block(then)?;
                let otherwise_ty = self.block(otherwise)?;
                if then_ty != otherwise_ty {
                    return Err(self.fail(format!(
                        "{position} branches yield {then_ty} and {otherwise_ty}"
                    )));
                }
                then_ty
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                if self.operand(*scrutinee, position)? != Type::Int {
                    return Err(self.fail(format!("{position} matches on a non-Int")));
                }
                let mut seen = std::collections::BTreeSet::new();
                let mut ty = None;
                for (values, block) in arms {
                    for value in values {
                        if !seen.insert(*value) {
                            return Err(
                                self.fail(format!("{position} matches {value} in two arms"))
                            );
                        }
                    }
                    let arm_ty = self.block(block)?;
                    if ty.is_some_and(|ty| ty != arm_ty) {
                        return Err(self.fail(format!("{position} arms yield different types")));
                    }
                    ty = Some(arm_ty);
                }
                let otherwise_ty = self.block(otherwise)?;
                if ty.is_some_and(|ty| ty != otherwise_ty) {
                    return Err(self.fail(format!("{position} arms yield different types")));
                }
                otherwise_ty
            }
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                if *count == 0 || *count > MAX_LOOP_COUNT {
                    return Err(self.fail(format!(
                        "{position} loops {count} times; the bound must be 1..={MAX_LOOP_COUNT}"
                    )));
                }
                let carried = self.operand(*init, position)?;
                self.loops.push(carried);
                if let Some(until) = until
                    && self.block(until)? != Type::Bool
                {
                    self.loops.pop();
                    return Err(self.fail(format!(
                        "{position} has an `until` that does not yield Bool"
                    )));
                }
                let next = self.block(body)?;
                self.loops.pop();
                if next != carried {
                    return Err(self.fail(format!(
                        "{position} carries {carried} but its body yields {next}"
                    )));
                }
                carried
            }
            Op::Carried => match self.loops.last() {
                Some(carried) => *carried,
                None => self.scan_carry().ok_or_else(|| {
                    self.fail(format!("{position} reads a carried value outside a loop"))
                })?,
            },
            Op::Index => {
                if self.loops.is_empty() && self.scan_carry().is_none() {
                    return Err(
                        self.fail(format!("{position} reads the loop index outside a loop"))
                    );
                }
                Type::Int
            }
            Op::Coord => {
                self.entry_only(position)?;
                match self.output_shape() {
                    Shape::Rows => {
                        return Err(
                            self.fail(format!("{position} reads coord() over rows; use element()"))
                        );
                    }
                    shape => shape.index_type(),
                }
            }
            Op::Domain => {
                self.entry_only(position)?;
                self.output_shape().index_type()
            }
            Op::Pointer => {
                self.require(Capability::InputRead, position)?;
                Type::Vec2
            }
            Op::PointerDown => {
                self.require(Capability::InputRead, position)?;
                Type::Bool
            }
            Op::Key(code) => {
                self.require(Capability::InputRead, position)?;
                if self.operand(*code, position)? != Type::Int {
                    return Err(self.fail(format!("{position} tests a key with a non-Int code")));
                }
                Type::Bool
            }
            Op::Button(code) => {
                self.require(Capability::InputRead, position)?;
                if self.operand(*code, position)? != Type::Int {
                    return Err(self.fail(format!(
                        "{position} tests a pointer button with a non-Int number"
                    )));
                }
                Type::Bool
            }
            Op::Wheel => {
                self.require(Capability::InputRead, position)?;
                Type::Vec2
            }
            Op::Window => {
                self.require(Capability::InputRead, position)?;
                Type::Vec2
            }
            Op::Scale => {
                self.require(Capability::InputRead, position)?;
                Type::Float
            }
            Op::Put(input, index, value) => {
                // The entry and the functions it calls write; `init`
                // (validated after the functions) does not.
                if !self.is_entry && self.index >= self.module.functions.len() {
                    return Err(self.fail(format!("{position} writes a memory outside the steps")));
                }
                self.require(Capability::FieldWrite, position)?;
                let manifest = &self.module.manifest;
                if manifest.mode != Mode::Scan {
                    return Err(self.fail(format!(
                        "{position} writes a memory, which only ordered work (`scan`) may do"
                    )));
                }
                let port = self.input(*input, position)?;
                if !manifest.memories.contains(&port.name) {
                    return Err(self.fail(format!(
                        "{position} writes input `{}`, which is not a memory",
                        port.name
                    )));
                }
                let address = self.operand(*index, position)?;
                if address != port.ty.index_type() {
                    return Err(self.fail(format!(
                        "{position} addresses memory `{}`, a {}, with {address} instead of {}",
                        port.name,
                        port.shape(),
                        port.ty.index_type()
                    )));
                }
                let written = self.operand(*value, position)?;
                if written != port.element() {
                    return Err(self.fail(format!(
                        "{position} writes {written} into memory `{}`, whose element is {}",
                        port.name,
                        port.element()
                    )));
                }
                port.element()
            }
            Op::Time => {
                self.require(Capability::TimeRead, position)?;
                Type::Float
            }
            Op::Rate => {
                self.require(Capability::TimeRead, position)?;
                Type::Float
            }
            Op::Element => {
                self.entry_only(position)?;
                Type::Int
            }
            Op::Sample(input) => {
                self.entry_only(position)?;
                self.require(Capability::FieldRead, position)?;
                let port = self.input(*input, position)?;
                if port.shape() != self.output_shape() {
                    return Err(self.fail(format!(
                        "{position} samples input `{}`, a {}, over a {} output; use at()",
                        port.name,
                        port.shape(),
                        self.output_shape()
                    )));
                }
                port.element()
            }
            Op::At(input, index) => {
                self.require(Capability::FieldRead, position)?;
                let port = self.input(*input, position)?;
                let actual = self.operand(*index, position)?;
                if actual != port.ty.index_type() {
                    return Err(self.fail(format!(
                        "{position} addresses input `{}`, a {}, with {actual} instead of {}",
                        port.name,
                        port.shape(),
                        port.ty.index_type()
                    )));
                }
                port.element()
            }
            Op::AtLinear(input, index) => {
                self.require(Capability::FieldRead, position)?;
                let port = self.input(*input, position)?;
                let actual = self.operand(*index, position)?;
                let expected = linear_index_type(port.shape());
                if actual != expected {
                    return Err(self.fail(format!(
                        "{position} interpolates input `{}`, a {}, at {actual} instead of {expected}",
                        port.name,
                        port.shape(),
                    )));
                }
                if !port
                    .ty
                    .schema
                    .scalars()
                    .iter()
                    .all(|s| *s == crate::Scalar::Float)
                {
                    return Err(self.fail(format!(
                        "{position} interpolates input `{}`, whose layout {} is not all floats",
                        port.name, port.ty.schema
                    )));
                }
                port.element()
            }
            Op::Count(input) => {
                self.require(Capability::FieldRead, position)?;
                self.input(*input, position)?;
                Type::Int
            }
            Op::Size(input) => {
                self.require(Capability::FieldRead, position)?;
                self.input(*input, position)?.ty.index_type()
            }
        };
        if actual != *ty {
            return Err(self.fail(format!("{position} is declared {ty} but computes {actual}")));
        }
        if let Type::Struct(index) = ty
            && *index as usize >= self.module.structs.len()
        {
            return Err(self.fail(format!("{position} has unknown struct type {ty}")));
        }
        let slot = id.0 as usize;
        if slot >= self.types.len() {
            return Err(self.fail(format!("{position} exceeds the function's slot count")));
        }
        self.types[slot] = *ty;
        self.visible[slot] = true;
        Ok(())
    }
}

/// Type of a helper function's single result.
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
                Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .find_map(|(_, block)| find(&block.body, id))
                    .or_else(|| find(&otherwise.body, id)),
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

/// The address type of a linear interpolation: a fractional position, so
/// `Float` over rows where `at` takes an `Int`.
pub fn linear_index_type(shape: Shape) -> Type {
    match shape {
        Shape::Rows => Type::Float,
        Shape::Grid => Type::Vec2,
        Shape::Volume => Type::Vec3,
    }
}

/// Checks one function and returns the type of every slot.
/// Which functions write a memory, themselves or through a call: a
/// fixed point over the call graph (recursion is refused elsewhere).
fn writers(functions: &[Function]) -> Vec<bool> {
    let mut writers = vec![false; functions.len()];
    loop {
        let mut changed = false;
        for (index, function) in functions.iter().enumerate() {
            if !writers[index] && block_writes(&function.body, &writers) {
                writers[index] = true;
                changed = true;
            }
        }
        if !changed {
            return writers;
        }
    }
}

/// Whether a block writes a memory: a `put`, or a call to a writer.
fn block_writes(instructions: &[Instruction], writers: &[bool]) -> bool {
    instructions
        .iter()
        .any(|instruction| match &instruction.op {
            Op::Put(..) => true,
            Op::Call(callee, _) => writers[*callee as usize],
            Op::If {
                then, otherwise, ..
            } => block_writes(&then.body, writers) || block_writes(&otherwise.body, writers),
            Op::Match {
                arms, otherwise, ..
            } => {
                arms.iter()
                    .any(|(_, block)| block_writes(&block.body, writers))
                    || block_writes(&otherwise.body, writers)
            }
            Op::Loop { until, body, .. } => {
                until
                    .as_ref()
                    .is_some_and(|block| block_writes(&block.body, writers))
                    || block_writes(&body.body, writers)
            }
            _ => false,
        })
}

/// The first call in a block to a function that writes a memory.
fn first_writing_call(instructions: &[Instruction], writers: &[bool]) -> Option<usize> {
    instructions
        .iter()
        .find_map(|instruction| match &instruction.op {
            Op::Call(callee, _) if writers[*callee as usize] => Some(*callee as usize),
            Op::If {
                then, otherwise, ..
            } => first_writing_call(&then.body, writers)
                .or_else(|| first_writing_call(&otherwise.body, writers)),
            Op::Match {
                arms, otherwise, ..
            } => arms
                .iter()
                .find_map(|(_, block)| first_writing_call(&block.body, writers))
                .or_else(|| first_writing_call(&otherwise.body, writers)),
            Op::Loop { until, body, .. } => until
                .as_ref()
                .and_then(|block| first_writing_call(&block.body, writers))
                .or_else(|| first_writing_call(&body.body, writers)),
            _ => None,
        })
}

fn validate_function(
    module: &Module,
    function: &Function,
    index: usize,
    is_entry: bool,
) -> Result<Vec<Type>, ValidationError> {
    let slots = function.slot_count();
    let mut checker = Checker {
        module,
        function,
        index,
        is_entry,
        types: vec![Type::Float; slots],
        visible: vec![false; slots],
        next: 0,
        loops: Vec::new(),
    };
    checker.instructions(&function.body)?;
    for result in &function.results {
        checker.operand(*result, ValueId(checker.next))?;
    }
    Ok(checker.types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelId, Scalar};

    fn manifest(requires: &[Capability]) -> Manifest {
        Manifest {
            id: "test.kernel@1".parse::<KernelId>().unwrap(),
            mode: Mode::Map,
            memories: Vec::new(),
            requires: requires.iter().copied().collect(),
            inputs: vec![Port {
                name: "source".into(),
                ty: PortType::image(),
            }],
            params: vec![],
            outputs: vec![Port {
                name: "result".into(),
                ty: PortType::image(),
            }],
        }
    }

    fn instruction(id: u32, ty: Type, op: Op) -> Instruction {
        Instruction {
            id: ValueId(id),
            ty,
            op,
        }
    }

    fn sample_module(requires: &[Capability]) -> Module {
        Module {
            manifest: manifest(requires),
            structs: vec![],
            functions: vec![],
            entry: Function {
                name: "compute".into(),
                args: vec![],
                body: vec![instruction(0, Type::Vec4, Op::Sample(0))],
                results: vec![ValueId(0)],
                bindings: vec![],
            },
            merge: None,
            init: None,
        }
    }

    #[test]
    fn host_access_needs_declared_capabilities() {
        let module = sample_module(&[Capability::FieldWrite]);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("needs `field.read`"), "{error}");
        let module = sample_module(&[Capability::FieldRead]);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("`field.write`"), "{error}");
        assert!(
            validate(sample_module(&[
                Capability::FieldRead,
                Capability::FieldWrite
            ]))
            .is_ok()
        );
    }

    #[test]
    fn ids_are_pre_order_and_block_values_go_out_of_scope() {
        let mut module = sample_module(&[Capability::FieldRead, Capability::FieldWrite]);
        // %1: if true { %2 = const 1.0; %2 } else { %3 = const 2.0; %3 }
        // %4: uses %2 after the block closed: rejected.
        module.entry.body = vec![
            instruction(0, Type::Bool, Op::Const(crate::Value::Bool(true))),
            instruction(
                1,
                Type::Float,
                Op::If {
                    condition: ValueId(0),
                    then: Block {
                        body: vec![instruction(
                            2,
                            Type::Float,
                            Op::Const(crate::Value::Float(1.0)),
                        )],
                        result: ValueId(2),
                    },
                    otherwise: Block {
                        body: vec![instruction(
                            3,
                            Type::Float,
                            Op::Const(crate::Value::Float(2.0)),
                        )],
                        result: ValueId(3),
                    },
                },
            ),
            instruction(4, Type::Vec4, Op::Construct(vec![ValueId(2); 4])),
        ];
        module.entry.results = vec![ValueId(4)];
        let error = validate(module.clone()).unwrap_err();
        assert!(error.message.contains("%4 refers to %2"), "{error}");

        module.entry.body[2] = instruction(4, Type::Vec4, Op::Construct(vec![ValueId(1); 4]));
        assert!(validate(module.clone()).is_ok());

        module.entry.body[2].id = ValueId(7);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("out of order"), "{error}");
    }

    #[test]
    fn loops_are_bounded_and_carry_one_type() {
        let mut module = sample_module(&[Capability::FieldRead, Capability::FieldWrite]);
        let body = |ty: Type, op: Op| Block {
            body: vec![instruction(2, ty, op)],
            result: ValueId(2),
        };
        module.entry.body = vec![
            instruction(0, Type::Float, Op::Const(crate::Value::Float(0.0))),
            instruction(
                1,
                Type::Float,
                Op::Loop {
                    count: 0,
                    init: ValueId(0),
                    until: None,
                    body: body(Type::Float, Op::Carried),
                },
            ),
            instruction(3, Type::Vec4, Op::Construct(vec![ValueId(1); 4])),
        ];
        module.entry.results = vec![ValueId(3)];
        let error = validate(module.clone()).unwrap_err();
        assert!(error.message.contains("bound must be"), "{error}");

        module.entry.body[1] = instruction(
            1,
            Type::Float,
            Op::Loop {
                count: 4,
                init: ValueId(0),
                until: None,
                body: body(Type::Int, Op::Index),
            },
        );
        let error = validate(module.clone()).unwrap_err();
        assert!(
            error
                .message
                .contains("carries Float but its body yields Int"),
            "{error}"
        );

        module.entry.body[1] = instruction(
            1,
            Type::Float,
            Op::Loop {
                count: 4,
                init: ValueId(0),
                until: None,
                body: body(Type::Float, Op::Carried),
            },
        );
        assert!(validate(module.clone()).is_ok());

        module.entry.body[0] = instruction(0, Type::Float, Op::Carried);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("outside a loop"), "{error}");
    }

    #[test]
    fn ports_carry_their_layout_and_shape_and_gate_their_ops() {
        let mut module = sample_module(&[Capability::FieldRead, Capability::FieldWrite]);
        module.structs = vec![StructDef {
            name: "P".into(),
            fields: vec![
                crate::Field {
                    name: "p".into(),
                    ty: Type::Vec2,
                },
                crate::Field {
                    name: "n".into(),
                    ty: Type::Int,
                },
            ],
            element: None,
        }];
        let schema = schema_of(Type::Struct(0), &module.structs).unwrap();
        assert_eq!(schema.to_string(), "f,f,i");
        let rows = PortType {
            element: Type::Struct(0),
            shape: Shape::Rows,
            schema: schema.clone(),
        };
        module.manifest.inputs[0].ty = rows.clone();
        module.manifest.outputs[0].ty = rows;
        module.entry.body = vec![instruction(0, Type::Struct(0), Op::Sample(0))];
        assert!(validate(module.clone()).is_ok());

        // Rows have no coordinate.
        module.entry.body = vec![
            instruction(0, Type::Vec2, Op::Coord),
            instruction(1, Type::Struct(0), Op::Sample(0)),
        ];
        module.entry.results = vec![ValueId(1)];
        let error = validate(module.clone()).unwrap_err();
        assert!(error.message.contains("use element()"), "{error}");

        // Sampling needs matching shapes; addressing needs the index type.
        module.manifest.outputs[0].ty = PortType {
            element: Type::Struct(0),
            shape: Shape::Grid,
            schema: schema.clone(),
        };
        module.entry.body = vec![instruction(0, Type::Struct(0), Op::Sample(0))];
        module.entry.results = vec![ValueId(0)];
        let error = validate(module.clone()).unwrap_err();
        assert!(error.message.contains("over a Grid output"), "{error}");
        module.entry.body = vec![
            instruction(0, Type::Vec2, Op::Coord),
            instruction(1, Type::Struct(0), Op::At(0, ValueId(0))),
        ];
        module.entry.results = vec![ValueId(1)];
        let error = validate(module.clone()).unwrap_err();
        assert!(
            error.message.contains("with Vec2 instead of Int"),
            "{error}"
        );

        module.entry.body = vec![
            instruction(0, Type::Int, Op::Element),
            instruction(1, Type::Struct(0), Op::At(0, ValueId(0))),
        ];
        assert!(validate(module.clone()).is_ok());

        module.manifest.inputs[0].ty = PortType {
            element: Type::Struct(0),
            shape: Shape::Rows,
            schema: crate::Schema::new(vec![Scalar::Float]),
        };
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("lays out as f,f,i"), "{error}");
    }
}
