//! The IR interpreter.
//!
//! A module compiles once, at prepare, into word-addressed code that runs
//! once per element. Every value lives in a flat stack of 32-bit words at
//! a compile-time offset, in the ABI's element layout: a scalar one word,
//! a vector one per component, a struct its fields in order. Arguments,
//! fields, components and carried values are compile-time aliases; a
//! Field read copies words from a slice the scope lent once; execution
//! allocates nothing per element.

mod code;
mod compile;
mod run;

pub(crate) use compile::{Compiled, Program};
pub(crate) use run::{CALL_RECORDS, Machine};
