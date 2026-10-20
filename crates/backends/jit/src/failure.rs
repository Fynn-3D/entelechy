//! The failure codes a run stores in its context, shared by native
//! code, WebAssembly and the helpers, and their reading as errors.

use entelechy_rules::{ArithmeticError, ExecutionError};

use crate::Shared;

/// A division or a remainder by zero.
pub(crate) const DIVISION_BY_ZERO: u32 = 1;
/// An integer result that does not fit.
pub(crate) const OVERFLOW: u32 = 2;
/// More loop iterations than the Kernel's budget.
pub(crate) const EXHAUSTED: u32 = 3;
/// The run went past its total.
pub(crate) const OVERRUN: u32 = 4;
/// An element that emitted past the room held for it.
pub(crate) const HELD_OVER: u32 = 5;
/// A lane of a group stopped at an exchange for the group to meet there:
/// no failure, and never said beyond the group ([`crate::Meeting`]).
pub(crate) const MET: u32 = 6;

/// The failure `code` names, at the element `x`, `y`.
pub(crate) fn failure(code: u32, x: u32, y: u32, shared: &Shared) -> ExecutionError {
    match code {
        DIVISION_BY_ZERO => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::DivisionByZero,
        },
        EXHAUSTED => ExecutionError::Exhausted {
            x,
            y,
            budget: shared.budget,
        },
        OVERRUN => shared.run.overrun(),
        HELD_OVER => shared.held.over(),
        _ => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::Overflow,
        },
    }
}
