//! Structural and capability validation of a [`Module`].
//!
//! Validation proves the properties backends rely on: SSA ordering, type
//! consistency, bounded calls, and that every Host operation is covered by a
//! declared capability. A [`Validated`] module is the only thing a backend
//! will execute.

use std::fmt;
use std::ops::Deref;

use crate::{Capability, Function, Instruction, Manifest, Module, Op, Type, ValueId};

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
        validate_function(&module, function, index, false)?;
        if function.results.len() != 1 {
            return Err(error(
                &function.name,
                format!(
                    "helper functions return exactly one value, found {}",
                    function.results.len()
                ),
            ));
        }
    }
    validate_function(&module, &module.entry, module.functions.len(), true)?;
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
        let actual = entry.body[result.0 as usize].ty;
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

fn validate_function(
    module: &Module,
    function: &Function,
    index: usize,
    is_entry: bool,
) -> Result<(), ValidationError> {
    let name = &function.name;
    let manifest = &module.manifest;
    let ty_of = |id: ValueId, position: usize| -> Result<Type, ValidationError> {
        if id.0 as usize >= position {
            return Err(error(
                name,
                format!("%{position} refers to {id}, which is not defined before it"),
            ));
        }
        Ok(function.body[id.0 as usize].ty)
    };
    let require = |capability: Capability, position: usize| -> Result<(), ValidationError> {
        if manifest.requires.contains(capability) {
            Ok(())
        } else {
            Err(error(
                name,
                format!("%{position} needs `{capability}`, which is not declared"),
            ))
        }
    };
    let input = |input: u32, position: usize| -> Result<(), ValidationError> {
        if (input as usize) < manifest.inputs.len() {
            Ok(())
        } else {
            Err(error(
                name,
                format!("%{position} reads unknown input {input}"),
            ))
        }
    };
    let entry_only = |position: usize| -> Result<(), ValidationError> {
        if is_entry {
            Ok(())
        } else {
            Err(error(
                name,
                format!("%{position} uses a per-pixel operation outside the entry point"),
            ))
        }
    };
    for (position, Instruction { ty, op }) in function.body.iter().enumerate() {
        let actual = match op {
            Op::Const(value) => value.ty(),
            Op::Param(param) => manifest
                .params
                .get(*param as usize)
                .map(|param| param.default.ty())
                .ok_or_else(|| error(name, format!("%{position} reads unknown param {param}")))?,
            Op::Arg(arg) => *function
                .args
                .get(*arg as usize)
                .ok_or_else(|| error(name, format!("%{position} reads unknown argument {arg}")))?,
            Op::Unary(_, a) => {
                let a = ty_of(*a, position)?;
                if a == Type::Image {
                    return Err(error(name, format!("%{position} negates an image")));
                }
                a
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (ty_of(*a, position)?, ty_of(*b, position)?);
                op.result_type(a, b).ok_or_else(|| {
                    error(
                        name,
                        format!("%{position} cannot apply `{}` to {a} and {b}", op.symbol()),
                    )
                })?
            }
            Op::Construct(parts) => {
                let expected = ty.components().ok_or_else(|| {
                    error(name, format!("%{position} constructs non-vector {ty}"))
                })?;
                if parts.len() != expected as usize {
                    return Err(error(
                        name,
                        format!(
                            "%{position} constructs {ty} from {} components",
                            parts.len()
                        ),
                    ));
                }
                for part in parts {
                    if ty_of(*part, position)? != Type::Float {
                        return Err(error(
                            name,
                            format!("%{position} constructs {ty} from a non-float component"),
                        ));
                    }
                }
                *ty
            }
            Op::Extract(vector, component) => {
                let vector = ty_of(*vector, position)?;
                let count = vector.components().ok_or_else(|| {
                    error(
                        name,
                        format!("%{position} extracts from non-vector {vector}"),
                    )
                })?;
                if *component >= count {
                    return Err(error(
                        name,
                        format!("%{position} extracts component {component} of {vector}"),
                    ));
                }
                Type::Float
            }
            Op::Call(callee, args) => {
                if *callee as usize >= index {
                    return Err(error(
                        name,
                        format!(
                            "%{position} calls function {callee}, which is not defined earlier"
                        ),
                    ));
                }
                let callee = &module.functions[*callee as usize];
                if callee.args.len() != args.len() {
                    return Err(error(
                        name,
                        format!(
                            "%{position} calls `{}` with {} arguments, expected {}",
                            callee.name,
                            args.len(),
                            callee.args.len()
                        ),
                    ));
                }
                for (arg, expected) in args.iter().zip(&callee.args) {
                    let actual = ty_of(*arg, position)?;
                    if actual != *expected {
                        return Err(error(
                            name,
                            format!(
                                "%{position} passes {actual} to `{}` where {expected} is expected",
                                callee.name
                            ),
                        ));
                    }
                }
                callee.body[callee.results[0].0 as usize].ty
            }
            Op::Intrinsic(intrinsic, args) => {
                let types = args
                    .iter()
                    .map(|arg| ty_of(*arg, position))
                    .collect::<Result<Vec<_>, _>>()?;
                intrinsic.result_type(&types).ok_or_else(|| {
                    let list = types
                        .iter()
                        .map(|ty| ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    error(
                        name,
                        format!("%{position} calls `{}` with ({list})", intrinsic.name()),
                    )
                })?
            }
            Op::Coord | Op::Domain => {
                entry_only(position)?;
                Type::Vec2
            }
            Op::Time => {
                require(Capability::TimeRead, position)?;
                Type::Float
            }
            Op::ImageSize(image) => {
                input(*image, position)?;
                require(Capability::ImageRead, position)?;
                Type::Vec2
            }
            Op::ImageSample(image) => {
                entry_only(position)?;
                input(*image, position)?;
                require(Capability::ImageRead, position)?;
                Type::Vec4
            }
            Op::ImageFetch(image, coord) => {
                input(*image, position)?;
                require(Capability::ImageRead, position)?;
                if ty_of(*coord, position)? != Type::Vec2 {
                    return Err(error(
                        name,
                        format!("%{position} fetches with a non-Vec2 coordinate"),
                    ));
                }
                Type::Vec4
            }
        };
        if actual != *ty {
            return Err(error(
                name,
                format!("%{position} is declared {ty} but computes {actual}"),
            ));
        }
    }
    for result in &function.results {
        if result.0 as usize >= function.body.len() {
            return Err(error(name, format!("result {result} is not defined")));
        }
    }
    Ok(())
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

    fn sample_module(requires: &[Capability]) -> Module {
        Module {
            manifest: manifest(requires),
            functions: vec![],
            entry: Function {
                name: "compute".into(),
                args: vec![],
                body: vec![Instruction {
                    ty: Type::Vec4,
                    op: Op::ImageSample(0),
                }],
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
    fn ssa_order_and_types_are_enforced() {
        let mut module = sample_module(&[Capability::ImageRead, Capability::ImageWrite]);
        module.entry.body.insert(
            0,
            Instruction {
                ty: Type::Float,
                op: Op::Extract(ValueId(1), 0),
            },
        );
        module.entry.results = vec![ValueId(1)];
        let error = validate(module.clone()).unwrap_err();
        assert!(error.message.contains("not defined before"), "{error}");

        module.entry.body[0] = Instruction {
            ty: Type::Int,
            op: Op::Const(crate::Value::Float(1.0)),
        };
        let error = validate(module).unwrap_err();
        assert!(
            error.message.contains("declared Int but computes Float"),
            "{error}"
        );
    }
}
