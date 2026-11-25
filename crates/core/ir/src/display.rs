//! Human-readable rendering of a module for inspection and diagnostics.

use std::fmt;

use crate::{
    Block, Function, Instruction, Manifest, Module, Op, Step, StructDef, Type, UnaryOp, ValueId,
};

/// A type's name with a Kernel's structs known: the source's name, or
/// `[T; N]` for an array.
pub fn type_name(ty: Type, structs: &[StructDef]) -> String {
    match ty {
        Type::Struct(index) => structs
            .get(index as usize)
            .map(|def| def.name.clone())
            .unwrap_or_else(|| ty.to_string()),
        other => other.to_string(),
    }
}

/// A port type with a Kernel's struct names in it (`Rows<Note>`).
pub fn port_type_name(ty: &entelechy_abi::PortType, structs: &[StructDef]) -> String {
    if ty.is_image() {
        return "Image".to_owned();
    }
    let kept = match ty.schema.uniform_encoding() {
        _ if ty.schema.is_wide() => String::new(),
        Some(encoding) => format!(" as {encoding}"),
        None => format!(" as {}", ty.schema),
    };
    format!("{}<{}{kept}>", ty.shape, type_name(ty.element, structs))
}

impl Manifest {
    /// The manifest printed with a module's struct names.
    pub fn named<'a>(&'a self, structs: &'a [StructDef]) -> NamedManifest<'a> {
        NamedManifest {
            manifest: self,
            structs,
        }
    }
}

/// A manifest and the structs whose names it prints. See
/// [`Manifest::named`].
pub struct NamedManifest<'a> {
    manifest: &'a Manifest,
    structs: &'a [StructDef],
}

impl fmt::Display for NamedManifest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.manifest.write(f, self.structs)
    }
}

impl fmt::Display for Manifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f, &[])
    }
}

impl Manifest {
    fn write(&self, f: &mut fmt::Formatter<'_>, structs: &[StructDef]) -> fmt::Result {
        writeln!(f, "kernel {}", self.id)?;
        if self.mode != crate::Mode::Map {
            writeln!(f, "  mode {}", self.mode)?;
        }
        writeln!(f, "  requires {}", self.requires)?;
        if self.budget != crate::DEFAULT_BUDGET {
            writeln!(f, "  budget {}", self.budget)?;
        }
        for input in &self.inputs {
            if !self.memories.contains(&input.name) {
                writeln!(
                    f,
                    "  input {}: {}",
                    input.name,
                    port_type_name(&input.ty, structs)
                )?;
            }
        }
        for param in &self.params {
            writeln!(
                f,
                "  param {}: {} = {}",
                param.name,
                type_name(param.ty, structs),
                param.default
            )?;
        }
        for output in &self.outputs {
            if !self.memories.contains(&output.name) {
                writeln!(
                    f,
                    "  output {}: {}",
                    output.name,
                    port_type_name(&output.ty, structs)
                )?;
            }
        }
        for memory in self
            .inputs
            .iter()
            .filter(|p| self.memories.contains(&p.name))
        {
            writeln!(
                f,
                "  memory {}: {}",
                memory.name,
                port_type_name(&memory.ty, structs)
            )?;
        }
        Ok(())
    }
}

fn list(ids: &[ValueId]) -> String {
    ids.iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn write_block(f: &mut fmt::Formatter<'_>, block: &Block, depth: usize) -> fmt::Result {
    let indent = "  ".repeat(depth);
    for instruction in &block.body {
        write_instruction(f, instruction, depth)?;
    }
    writeln!(f, "{indent}yield {}", block.result)
}

fn write_instruction(
    f: &mut fmt::Formatter<'_>,
    Instruction { id, ty, op }: &Instruction,
    depth: usize,
) -> fmt::Result {
    let indent = "  ".repeat(depth);
    match op {
        Op::If {
            condition,
            then,
            otherwise,
        } => {
            writeln!(f, "{indent}{id}: {ty} = if {condition} {{")?;
            write_block(f, then, depth + 1)?;
            writeln!(f, "{indent}}} else {{")?;
            write_block(f, otherwise, depth + 1)?;
            writeln!(f, "{indent}}}")
        }
        Op::Match {
            scrutinee,
            arms,
            otherwise,
        } => {
            writeln!(f, "{indent}{id}: {ty} = match {scrutinee} {{")?;
            for (values, block) in arms {
                let values = values
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" | ");
                writeln!(f, "{indent}  {values} => {{")?;
                write_block(f, block, depth + 2)?;
                writeln!(f, "{indent}  }}")?;
            }
            writeln!(f, "{indent}  _ => {{")?;
            write_block(f, otherwise, depth + 2)?;
            writeln!(f, "{indent}  }}")?;
            writeln!(f, "{indent}}}")
        }
        Op::Loop {
            count,
            bound,
            init,
            until,
            body,
        } => {
            match bound {
                Some(bound) => writeln!(
                    f,
                    "{indent}{id}: {ty} = loop {bound} times (at most {count}) carrying {init} {{"
                )?,
                None => writeln!(f, "{indent}{id}: {ty} = loop {count} carrying {init} {{")?,
            }
            if let Some(until) = until {
                writeln!(f, "{indent}  until {{")?;
                write_block(f, until, depth + 2)?;
                writeln!(f, "{indent}  }}")?;
            }
            write_block(f, body, depth + 1)?;
            writeln!(f, "{indent}}}")
        }
        Op::Each { count, body } => {
            writeln!(f, "{indent}{id}: {ty} = each {count} {{")?;
            write_block(f, body, depth + 1)?;
            writeln!(f, "{indent}}}")
        }
        Op::Lanes { body } => {
            writeln!(f, "{indent}{id}: {ty} = lanes {{")?;
            write_block(f, body, depth + 1)?;
            writeln!(f, "{indent}}}")
        }
        other => writeln!(f, "{indent}{id}: {ty} = {other}"),
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args = self
            .args
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "fn {}({args}) {{", self.name)?;
        for instruction in &self.body {
            write_instruction(f, instruction, 1)?;
        }
        writeln!(f, "  return {}", list(&self.results))?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(value) => write!(f, "const {value}"),
            Self::Param(index) => write!(f, "param {index}"),
            Self::Arg(index) => write!(f, "arg {index}"),
            Self::Unary(UnaryOp::Neg, a) => write!(f, "neg {a}"),
            Self::Unary(UnaryOp::Not, a) => write!(f, "not {a}"),
            Self::Binary(op, a, b) => write!(f, "{a} {} {b}", op.symbol()),
            Self::Construct(parts) => write!(f, "construct {}", list(parts)),
            Self::Extract(vector, component) => write!(f, "extract {vector}[{component}]"),
            Self::Subscript(array, index) => write!(f, "subscript {array}[{index}]"),
            Self::Call(function, args) => write!(f, "call fn{function}({})", list(args)),
            Self::Intrinsic(intrinsic, args) => {
                write!(f, "{}({})", intrinsic.name(), list(args))
            }
            Self::If { condition, .. } => write!(f, "if {condition} {{ ... }}"),
            Self::Match { scrutinee, .. } => write!(f, "match {scrutinee} {{ ... }}"),
            Self::Loop { count, init, .. } => write!(f, "loop {count} carrying {init} {{ ... }}"),
            Self::Each { count, .. } => write!(f, "each {count} {{ ... }}"),
            Self::Lanes { .. } => f.write_str("lanes { ... }"),
            Self::Carried => f.write_str("carried"),
            Self::Index => f.write_str("index"),
            Self::Coord => f.write_str("coord"),
            Self::Domain => f.write_str("domain"),
            Self::Time => f.write_str("context time"),
            Self::Clock => f.write_str("context clock"),
            Self::Rate => f.write_str("context rate"),
            Self::Queued => f.write_str("context queued"),
            Self::Window => f.write_str("context window"),
            Self::Scale => f.write_str("context scale"),
            Self::Pointer => f.write_str("context pointer"),
            Self::PointerDown => f.write_str("context pointer.down"),
            Self::Key(code) => write!(f, "context key {code}"),
            Self::Typed(k) => write!(f, "context typed {k}"),
            Self::Button(code) => write!(f, "context button {code}"),
            Self::Wheel => f.write_str("context wheel"),
            Self::Pads => f.write_str("context pads"),
            Self::Stick(which) => write!(f, "context stick {which}"),
            Self::Element => f.write_str("element"),
            Self::Lane => f.write_str("lane"),
            Self::Exchange(value, lane) => write!(f, "exchange {value} {lane}"),
            Self::Meet(Some(given)) => write!(f, "meet {given}"),
            Self::Meet(None) => f.write_str("meet"),
            Self::Sample(input) => write!(f, "host sample input{input}"),
            Self::At(input, index) => write!(f, "host at input{input} {index}"),
            Self::Part { input, at, path } => {
                match at {
                    Some(at) => write!(f, "host part input{input} {at}")?,
                    None => write!(f, "host part input{input}")?,
                }
                for step in path {
                    match step {
                        Step::Field(k) => write!(f, ".{k}")?,
                        Step::Index(index) => write!(f, "[{index}]")?,
                    }
                }
                Ok(())
            }
            Self::AtLinear(input, index) => write!(f, "host at.linear input{input} {index}"),
            Self::AtFiltered {
                input,
                at,
                repeat,
                lod,
                grad,
            } => {
                let kind = if *repeat { "at.repeat" } else { "at.linear" };
                match (lod, grad) {
                    (Some(lod), _) => write!(f, "host {kind} input{input} {at} lod {lod}"),
                    (_, Some((dx, dy))) => {
                        write!(f, "host {kind} input{input} {at} grad {dx} {dy}")
                    }
                    (None, None) => write!(f, "host {kind} input{input} {at}"),
                }
            }
            Self::Count(input) => write!(f, "host count input{input}"),
            Self::Size(input) => write!(f, "host size input{input}"),
            Self::Put(input, index, value) => write!(f, "host put input{input} {index} {value}"),
            Self::Emit(pair) => write!(f, "host emit {pair}"),
            Self::Spend(counted) => write!(f, "work spend {counted}"),
            Self::Look => write!(f, "work look"),
        }
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.manifest.named(&self.structs))?;
        for (index, def) in self.structs.iter().enumerate() {
            let fields = def
                .fields
                .iter()
                .map(|field| format!("{}: {}", field.name, field.ty))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(f, "struct S{index} {} {{ {fields} }}", def.name)?;
        }
        for function in &self.functions {
            writeln!(f)?;
            write!(f, "{function}")?;
        }
        writeln!(f)?;
        write!(f, "{}", self.entry)?;
        if let Some(merge) = &self.merge {
            writeln!(f)?;
            write!(f, "{merge}")?;
        }
        if let Some(init) = &self.init {
            writeln!(f)?;
            write!(f, "{init}")?;
        }
        Ok(())
    }
}
