//! Structural and capability validation of a [`Module`].
//!
//! Validation proves the properties backends rely on: pre-order value ids,
//! block scoping, type consistency, bounded calls and loops, and that every
//! Host operation is covered by a declared capability. A [`Validated`]
//! module is the only thing a backend will execute.

use std::fmt;
use std::ops::Deref;

use crate::{
    Block, Capability, Function, Instruction, MAX_LOOP_COUNT, Manifest, Module, Op, Type, UnaryOp,
    ValueId,
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
    validate_manifest(&module.manifest)?;
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
    if entry.results.len() != outputs.len() {
        return Err(error(
            &entry.name,
            format!(
                "entry returns {} values but the manifest declares {} outputs",
                entry.results.len(),
                outputs.len()
            ),
        ));
    }
    for (result, output) in entry.results.iter().zip(outputs) {
        let actual = types[result.0 as usize];
        let expected = match output.ty {
            Type::Image => Type::Vec4,
            other => other,
        };
        if actual != expected {
            return Err(error(
                &entry.name,
                format!(
                    "output `{}` expects {expected} per pixel but the entry returns {actual}",
                    output.name
                ),
            ));
        }
    }
    if !outputs.is_empty() && !module.manifest.requires.contains(Capability::ImageWrite) {
        return Err(error(
            "manifest",
            "writing outputs requires `image.write`".to_owned(),
        ));
    }
    Ok(Validated(module))
}

fn error(location: &str, message: impl Into<String>) -> ValidationError {
    ValidationError {
        location: location.to_owned(),
        message: message.into(),
    }
}

fn validate_manifest(manifest: &Manifest) -> Result<(), ValidationError> {
    let mut names = std::collections::BTreeSet::new();
    let ports = manifest
        .inputs
        .iter()
        .map(|port| (&port.name, "input"))
        .chain(manifest.params.iter().map(|param| (&param.name, "param")))
        .chain(manifest.outputs.iter().map(|port| (&port.name, "output")));
    for (name, kind) in ports {
        if !names.insert(name.clone()) {
            return Err(error(
                "manifest",
                format!("{kind} `{name}` is declared twice"),
            ));
        }
    }
    for input in &manifest.inputs {
        if input.ty != Type::Image {
            return Err(error(
                "manifest",
                format!(
                    "input `{}` must be an Image, found {}",
                    input.name, input.ty
                ),
            ));
        }
    }
    for output in &manifest.outputs {
        if output.ty != Type::Image {
            return Err(error(
                "manifest",
                format!(
                    "output `{}` must be an Image, found {}",
                    output.name, output.ty
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

impl Checker<'_> {
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

    fn input(&self, input: u32, position: ValueId) -> Result<(), ValidationError> {
        if (input as usize) < self.module.manifest.inputs.len() {
            Ok(())
        } else {
            Err(self.fail(format!("{position} reads unknown input {input}")))
        }
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
                let expected = ty
                    .components()
                    .ok_or_else(|| self.fail(format!("{position} constructs non-vector {ty}")))?;
                if parts.len() != expected as usize {
                    return Err(self.fail(format!(
                        "{position} constructs {ty} from {} components",
                        parts.len()
                    )));
                }
                for part in parts {
                    if self.operand(*part, position)? != Type::Float {
                        return Err(self.fail(format!(
                            "{position} constructs {ty} from a non-float component"
                        )));
                    }
                }
                *ty
            }
            Op::Extract(vector, component) => {
                let vector = self.operand(*vector, position)?;
                let count = vector.components().ok_or_else(|| {
                    self.fail(format!("{position} extracts from non-vector {vector}"))
                })?;
                if *component >= count {
                    return Err(self.fail(format!(
                        "{position} extracts component {component} of {vector}"
                    )));
                }
                Type::Float
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
            Op::Carried => *self.loops.last().ok_or_else(|| {
                self.fail(format!("{position} reads a carried value outside a loop"))
            })?,
            Op::Index => {
                if self.loops.is_empty() {
                    return Err(
                        self.fail(format!("{position} reads the loop index outside a loop"))
                    );
                }
                Type::Int
            }
            Op::Coord | Op::Domain => {
                self.entry_only(position)?;
                Type::Vec2
            }
            Op::Time => {
                self.require(Capability::TimeRead, position)?;
                Type::Float
            }
            Op::ImageSize(image) => {
                self.input(*image, position)?;
                self.require(Capability::ImageRead, position)?;
                Type::Vec2
            }
            Op::ImageSample(image) => {
                self.entry_only(position)?;
                self.input(*image, position)?;
                self.require(Capability::ImageRead, position)?;
                Type::Vec4
            }
            Op::ImageFetch(image, coord) => {
                self.input(*image, position)?;
                self.require(Capability::ImageRead, position)?;
                if self.operand(*coord, position)? != Type::Vec2 {
                    return Err(self.fail(format!("{position} fetches with a non-Vec2 coordinate")));
                }
                Type::Vec4
            }
        };
        if actual != *ty {
            return Err(self.fail(format!("{position} is declared {ty} but computes {actual}")));
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

/// Checks one function and returns the type of every slot.
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
    use crate::{KernelId, Port};

    fn manifest(requires: &[Capability]) -> Manifest {
        Manifest {
            id: "test.kernel@1".parse::<KernelId>().unwrap(),
            requires: requires.iter().copied().collect(),
            inputs: vec![Port {
                name: "source".into(),
                ty: Type::Image,
            }],
            params: vec![],
            outputs: vec![Port {
                name: "result".into(),
                ty: Type::Image,
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
            functions: vec![],
            entry: Function {
                name: "compute".into(),
                args: vec![],
                body: vec![instruction(0, Type::Vec4, Op::ImageSample(0))],
                results: vec![ValueId(0)],
            },
        }
    }

    #[test]
    fn host_access_needs_declared_capabilities() {
        let module = sample_module(&[Capability::ImageWrite]);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("needs `image.read`"), "{error}");
        let module = sample_module(&[Capability::ImageRead]);
        let error = validate(module).unwrap_err();
        assert!(error.message.contains("`image.write`"), "{error}");
        assert!(
            validate(sample_module(&[
                Capability::ImageRead,
                Capability::ImageWrite
            ]))
            .is_ok()
        );
    }

    #[test]
    fn ids_are_pre_order_and_block_values_go_out_of_scope() {
        let mut module = sample_module(&[Capability::ImageRead, Capability::ImageWrite]);
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
        let mut module = sample_module(&[Capability::ImageRead, Capability::ImageWrite]);
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
}
