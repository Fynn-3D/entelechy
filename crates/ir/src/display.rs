//! Human-readable rendering of a module for inspection and diagnostics.

use std::fmt;

use crate::{Block, Function, Instruction, Manifest, Module, Op, UnaryOp, ValueId};

impl fmt::Display for Manifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "kernel {}", self.id)?;
        writeln!(f, "  requires {}", self.requires)?;
        for input in &self.inputs {
            writeln!(f, "  input {}: {}", input.name, input.ty)?;
        }
        for param in &self.params {
            writeln!(
                f,
                "  param {}: {} = {}",
                param.name,
                param.default.ty(),
                param.default
            )?;
        }
        for output in &self.outputs {
            writeln!(f, "  output {}: {}", output.name, output.ty)?;
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
        Op::Loop {
            count,
            init,
            until,
            body,
        } => {
            writeln!(f, "{indent}{id}: {ty} = loop {count} carrying {init} {{")?;
            if let Some(until) = until {
                writeln!(f, "{indent}  until {{")?;
                write_block(f, until, depth + 2)?;
                writeln!(f, "{indent}  }}")?;
            }
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
            Self::Call(function, args) => write!(f, "call fn{function}({})", list(args)),
            Self::Intrinsic(intrinsic, args) => {
                write!(f, "{}({})", intrinsic.name(), list(args))
            }
            Self::If { condition, .. } => write!(f, "if {condition} {{ ... }}"),
            Self::Loop { count, init, .. } => write!(f, "loop {count} carrying {init} {{ ... }}"),
            Self::Carried => f.write_str("carried"),
            Self::Index => f.write_str("index"),
            Self::Coord => f.write_str("coord"),
            Self::Domain => f.write_str("domain"),
            Self::Time => f.write_str("context time"),
            Self::Pointer => f.write_str("context pointer"),
            Self::PointerDown => f.write_str("context pointer.down"),
            Self::Key(code) => write!(f, "context key {code}"),
            Self::Element => f.write_str("element"),
            Self::Sample(input) => write!(f, "host sample input{input}"),
            Self::At(input, index) => write!(f, "host at input{input} {index}"),
            Self::Count(input) => write!(f, "host count input{input}"),
            Self::Size(input) => write!(f, "host size input{input}"),
        }
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.manifest)?;
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
        write!(f, "{}", self.entry)
    }
}
