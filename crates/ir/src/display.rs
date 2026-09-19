//! Human-readable rendering of a module for inspection and diagnostics.

use std::fmt;

use crate::{Function, Instruction, Manifest, Module, Op, UnaryOp};

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
        for (index, Instruction { ty, op }) in self.body.iter().enumerate() {
            writeln!(f, "  %{index}: {ty} = {op}")?;
        }
        let results = self
            .results
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "  return {results}")?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |ids: &[crate::ValueId]| {
            ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            Self::Const(value) => write!(f, "const {value}"),
            Self::Param(index) => write!(f, "param {index}"),
            Self::Arg(index) => write!(f, "arg {index}"),
            Self::Unary(UnaryOp::Neg, a) => write!(f, "neg {a}"),
            Self::Binary(op, a, b) => write!(f, "{a} {} {b}", op.symbol()),
            Self::Construct(parts) => write!(f, "construct {}", list(parts)),
            Self::Extract(vector, component) => write!(f, "extract {vector}[{component}]"),
            Self::Call(function, args) => write!(f, "call fn{function}({})", list(args)),
            Self::Intrinsic(intrinsic, args) => {
                write!(f, "{}({})", intrinsic.name(), list(args))
            }
            Self::Coord => f.write_str("coord"),
            Self::Domain => f.write_str("domain"),
            Self::ImageSize(input) => write!(f, "host image.size input{input}"),
            Self::ImageSample(input) => write!(f, "host image.sample input{input}"),
            Self::ImageFetch(input, coord) => write!(f, "host image.fetch input{input} {coord}"),
        }
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.manifest)?;
        for function in &self.functions {
            writeln!(f)?;
            write!(f, "{function}")?;
        }
        writeln!(f)?;
        write!(f, "{}", self.entry)
    }
}
