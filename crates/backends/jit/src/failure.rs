//! What a run leaves in its context when an element fails, and the
//! failure each code names. The native code and the WebAssembly the JIT
//! writes store the same codes, and the helpers they call store one too,
//! so one list names them and one reading turns them into errors.

use orla_rules::{ArithmeticError, ExecutionError};

use crate::Shared;

/// A division or a remainder by zero.
pub(crate) const DIVISION_BY_ZERO: u32 = 1;
/// An integer result that does not fit.
pub(crate) const OVERFLOW: u32 = 2;
/// More loop iterations than the Kernel's budget.
pub(crate) const EXHAUSTED: u32 = 3;
/// The run past its total.
pub(crate) const OVERRUN: u32 = 4;
/// An element that emitted past the room held for it.
pub(crate) const HELD_OVER: u32 = 5;

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
