//! The IR interpreter.
//!
//! A module is compiled once per execution into word-addressed code and
//! that code runs once per element. Every value lives in a flat stack of
//! 32-bit words at an offset fixed at compile time, in the layout the ABI
//! gives elements: a scalar is one word, a vector one per component, a
//! struct its fields in order. Reading an argument, a struct field, a
//! vector component, or a loop's carried value is therefore an alias
//! resolved at compile time and costs nothing at run time; a Field read
//! is a copy of words out of a slice the scope lent once; and no
//! execution allocates per element.

mod code;
mod compile;
mod run;

pub(crate) use compile::Program;
pub(crate) use run::{CALL_RECORDS, Machine};
